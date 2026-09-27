"""Strategy Audit Engine.

Runs a proper backtest for every rule in config/rules.yaml on the extended
8-year panel where possible, or on the 3-year production feature panel when a
rule uses features the extended panel does not carry. Writes
data/screen/strategy_audit.json sorted by CAGR desc.

Non-negotiables (see CLAUDE.md #1, #2, #7, #12, #13):
  * DROP contaminated rows FIRST, before any derived feature is computed.
  * Never recompute a shipped feature the panel already carries. Only aux
    scaffolding not present on either panel (lo7_prior / new_lo7 for 8yr) is
    computed from adj_close.
  * Entry = next open after signal close. Stop = FILL_PRICE - ATR_MULT * atr14
    (close-based - only a close below stop triggers, exit at next open).
  * Fresh event = first True after 20 sessions of False, per isin.
  * 10 concurrent slots. Sort by rs_rank desc. Skip if all slots full.
  * Cost 0.50 percent round trip.
  * MAX DD = daily equity peak-to-trough, mark-to-market every session.
  * Sort output by CAGR desc.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
PANEL_8Y = "data/backtest_panels/bottom_panel.parquet"
PANEL_3Y = ROOT / "data" / "screen" / "features.parquet"
OUT_JSON = ROOT / "data" / "screen" / "strategy_audit.json"
COST_RT = 0.50           # 0.50% round trip, mid-cap band (CLAUDE.md #13)
QUIET = 20               # sessions of False before first True counts as fresh
SLOTS = 10
TRADING_DAYS = 252

# Exit families and their execution parameters (CLAUDE.md #1)
EXIT_PARAMS = {
    "mean_reversion":     {"atr_mult": 4.0, "cap": 20, "exit_rule": "hi7"},
    "trend_continuation": {"atr_mult": 3.0, "cap": 60, "exit_rule": "ema21"},
    "momentum_leader":    {"atr_mult": 4.0, "cap": 60, "exit_rule": "ema21"},
    "watchlist":          {"atr_mult": 0.0, "cap": 0,  "exit_rule": "none"},
}

# Static family + coverage map. Coverage chosen by whether the extended 8-year
# panel carries every feature the rule expression touches.
RULE_TABLE = {
    # ---- MEAN-REVERSION (7d-high exit, cap 20, 4 ATR stop) ----------------
    "sw_rsi2_snapback":         ("mean_reversion",     "8yr"),
    "sw_double_seven":          ("mean_reversion",     "8yr"),
    "sw_leader_dip_concentrated": ("mean_reversion",   "8yr"),
    "sw_monthly_r1_retest":     ("mean_reversion",     "3yr"),
    "leader_dip_playbook":      ("mean_reversion",     "8yr"),
    "rsi2_reversion":           ("mean_reversion",     "8yr"),
    "oversold_in_uptrend":      ("mean_reversion",     "8yr"),
    "double_seven":             ("mean_reversion",     "8yr"),
    "leader_dip":               ("mean_reversion",     "8yr"),
    "spring_reclaim":           ("mean_reversion",     "3yr"),
    "vl_wyckoff_spring":        ("mean_reversion",     "8yr"),
    # ---- TREND-CONTINUATION (close<21EMA trail, cap 60, 3 ATR stop) -------
    "momentum_leaders":         ("trend_continuation", "3yr"),
    "trend_stack":              ("trend_continuation", "3yr"),
    "trend_stack_swing":        ("trend_continuation", "3yr"),
    "momentum_swing":           ("trend_continuation", "3yr"),
    "near_52w_high":            ("trend_continuation", "8yr"),
    "donchian_breakout":        ("trend_continuation", "3yr"),
    "stage2_base_breakout":     ("trend_continuation", "3yr"),
    "high_tight_breakout":      ("trend_continuation", "3yr"),
    "minervini_trend_template": ("trend_continuation", "3yr"),
    "ema21_pullback":           ("trend_continuation", "3yr"),
    "vl_volume_spike_breakout": ("trend_continuation", "3yr"),
    "vl_long_base_breakout":    ("trend_continuation", "3yr"),
    "vl_pocket_pivot":          ("trend_continuation", "3yr"),
    "pocket_pivot":             ("trend_continuation", "3yr"),
    "held_breakout":            ("trend_continuation", "3yr"),
    "three_weeks_tight":        ("trend_continuation", "3yr"),
    "stage2_portfolio":         ("trend_continuation", "3yr"),
    "smooth_momentum":          ("trend_continuation", "3yr"),
    "r1_breakout_retest":       ("trend_continuation", "3yr"),
    "base_breakout_r1":         ("trend_continuation", "3yr"),
    "monthly_r1_breakout":      ("trend_continuation", "3yr"),
    "monthly_r1_retest":        ("trend_continuation", "3yr"),
    "vcp_squeeze":              ("trend_continuation", "3yr"),
    "episodic_pivot":           ("trend_continuation", "3yr"),
    "volume_shocker":           ("trend_continuation", "3yr"),
    # ---- WATCHLIST (skip trading) -----------------------------------------
    "close_above_monthly_tc":   ("watchlist",          "3yr"),
    "weekly_tc_reclaim":        ("watchlist",          "3yr"),
    "inside_bar_coil":          ("watchlist",          "3yr"),
    "narrow_cpr_breakout":      ("watchlist",          "3yr"),
    "accumulation_thrust":      ("watchlist",          "3yr"),
}


# --------------------------------------------------------------- panel loading

def _rank_group(x: pd.Series) -> pd.Series:
    return x.rank(pct=True) * 100


def load_panel(coverage: str):
    """Load the panel for a coverage window with the guarantee that
    contaminated rows are dropped BEFORE any derived column is computed."""
    if coverage == "8yr":
        raw = pd.read_parquet(PANEL_8Y, engine="pyarrow")
    else:
        raw = pd.read_parquet(PANEL_3Y, engine="pyarrow")
    # Step 1: drop contaminated rows FIRST (CLAUDE.md #13)
    before = len(raw)
    raw = raw[~raw["contaminated"].astype(bool)].copy()
    dropped = before - len(raw)
    raw = raw.sort_values(["isin", "date"]).reset_index(drop=True)
    # Ensure a datetime dtype date column and stable dtypes
    raw["date"] = pd.to_datetime(raw["date"])
    print(f"  [{coverage}] loaded {before:,} rows, dropped {dropped:,} contaminated -> {len(raw):,}",
          flush=True)
    # Step 2: derive aux scaffolding the panel doesn't carry, from adj_close.
    # These are auxiliary rolling stats on adj_close (already contamination
    # aware because we dropped first). Never override a shipped column.
    g = raw.groupby("isin", sort=False)["adj_close"]
    if "lo7_prior" not in raw.columns:
        raw["lo7_prior"] = g.transform(lambda s: s.shift(1).rolling(7).min())
    if "new_lo7" not in raw.columns:
        raw["new_lo7"] = (raw["adj_close"] < raw["lo7_prior"]).astype(int)
    if "new_hi7" not in raw.columns and "hi7_prior" in raw.columns:
        raw["new_hi7"] = (raw["adj_close"] > raw["hi7_prior"]).astype(int)
    # rs_rank / rs_score reconstruction (features.parquet does not carry them;
    # the extended panel does). Uses the SAME construction as
    # screener/backtest.py::add_panel_columns so the two panels agree.
    if "rs_rank" not in raw.columns:
        for col in ("ret_60d", "ret_120d", "ret_250d"):
            raw[col] = pd.to_numeric(raw.get(col), errors="coerce")
        by_date = raw.groupby("date")
        score = (0.4 * by_date["ret_60d"].rank(pct=True)
                 + 0.2 * by_date["ret_120d"].rank(pct=True)
                 + 0.4 * by_date["ret_250d"].rank(pct=True))
        raw["rs_score"] = score
        raw["rs_rank"] = (raw.groupby("date")["rs_score"].rank(pct=True) * 99).round(0)
    return raw


# --------------------------------------------------------- fresh-event helper

def fresh_events(mask: np.ndarray, isin_idx: np.ndarray, N: int = 20) -> np.ndarray:
    """First True after N sessions of False, per isin. mask, isin_idx are
    aligned to the sorted panel. Returns bool array of same length."""
    out = np.zeros_like(mask, dtype=bool)
    # count of consecutive Falses immediately preceding each row, per isin.
    # We rely on the sort by (isin, date). Walk once.
    run_false = 0
    prev = -1
    for i in range(len(mask)):
        ii = isin_idx[i]
        if ii != prev:
            run_false = N   # treat start-of-group as "quiet enough"
            prev = ii
        if mask[i]:
            if run_false >= N:
                out[i] = True
            run_false = 0
        else:
            run_false += 1
    return out


# --------------------------------------------------- rule expression -> mask

def eval_rule(feat: pd.DataFrame, expr: str) -> np.ndarray:
    """Evaluate a rule expression to a boolean numpy array. Cast because
    pd.DataFrame.eval returns object dtype for boolean expressions (CLAUDE.md
    #11). Any NaN is treated as False."""
    # YAML folded scalars can leave residual whitespace / trailing newlines
    # that trip the eval parser on multi-line parens. Normalize.
    norm = " ".join(expr.split())
    try:
        v = feat.eval(norm, engine="python")
    except Exception as ex:
        print(f"    eval failed: {ex}", flush=True)
        return np.zeros(len(feat), dtype=bool)
    if isinstance(v, pd.Series):
        return v.fillna(False).astype(bool).to_numpy()
    return np.asarray(v, dtype=bool)


# ----------------------------------------------- per-signal exit simulation

def simulate_signals(events_ix: np.ndarray, isin_arr: np.ndarray, isin_idx: np.ndarray,
                     O: np.ndarray, C: np.ndarray, ATR_PCT: np.ndarray,
                     HI7: np.ndarray, EMA21: np.ndarray, LAST_ROW: np.ndarray,
                     atr_mult: float, cap: int, exit_rule: str,
                     row_to_sess: np.ndarray, rs_rank: np.ndarray) -> pd.DataFrame:
    """Given a set of event row indices, produce per-trade records:
    entry_row, exit_row, fill_price, exit_price, net_pct (after cost),
    stopped, capped, isin_idx, rs_rank_at_signal, entry_sess, exit_sess."""
    rows = []
    for t in events_ix:
        # Need a next bar to enter, and finite atr_pct.
        if t >= LAST_ROW[t]:
            continue
        atr_frac = ATR_PCT[t] / 100.0
        if not np.isfinite(atr_frac) or atr_frac <= 0:
            continue
        entry_row = t + 1
        fill = O[entry_row]
        if not np.isfinite(fill) or fill <= 0:
            continue
        stop_level = fill * (1.0 - atr_mult * atr_frac)
        end = min(int(LAST_ROW[t]), t + cap)   # cap in sessions past signal
        exit_row = None
        stopped = False
        capped = False
        k = entry_row
        # Bar loop: check close-based stop first, then exit trigger.
        while k <= end:
            c = C[k]
            # CLOSE-BASED stop (CLAUDE.md #7): only a close < stop triggers.
            if np.isfinite(c) and atr_mult > 0 and c < stop_level:
                # Exit at next open; if no next bar, exit at this close.
                if k + 1 <= LAST_ROW[t]:
                    exit_row = k + 1
                else:
                    exit_row = k
                stopped = True
                break
            # Family-specific exit trigger, checked on the close.
            if exit_rule == "hi7":
                hi7 = HI7[k]
                if np.isfinite(c) and np.isfinite(hi7) and c > hi7 and k > entry_row - 1:
                    # First close ABOVE the previous 7 closes -> next open.
                    if k + 1 <= LAST_ROW[t]:
                        exit_row = k + 1
                    else:
                        exit_row = k
                    break
            elif exit_rule == "ema21":
                e21 = EMA21[k]
                if np.isfinite(c) and np.isfinite(e21) and c < e21:
                    if k + 1 <= LAST_ROW[t]:
                        exit_row = k + 1
                    else:
                        exit_row = k
                    break
            k += 1
        if exit_row is None:
            # time cap - exit at close of last bar in window
            exit_row = min(k, end)
            capped = True
        # Exit price: for stop and trigger, we exit AT OPEN of exit_row (which
        # is signal-triggered on k, exit_row = k+1). For cap we exit at CLOSE
        # of the last in-window bar. Model both.
        if capped:
            xp = C[exit_row]
        else:
            xp = O[exit_row]
        if not np.isfinite(xp):
            xp = C[exit_row]
        if not np.isfinite(xp) or xp <= 0:
            continue
        gross = (xp / fill - 1.0) * 100.0
        net = gross - COST_RT
        rows.append((
            int(t), int(entry_row), int(exit_row),
            float(fill), float(xp), float(gross), float(net),
            int(stopped), int(capped),
            int(isin_idx[t]), float(rs_rank[t]) if np.isfinite(rs_rank[t]) else -1.0,
            int(row_to_sess[entry_row]), int(row_to_sess[exit_row]),
        ))
    if not rows:
        return pd.DataFrame(columns=["sig_row", "entry_row", "exit_row",
            "fill", "exit_price", "gross", "net", "stopped", "capped",
            "isin_idx", "rs_at_signal", "entry_sess", "exit_sess"])
    return pd.DataFrame(rows, columns=["sig_row", "entry_row", "exit_row",
        "fill", "exit_price", "gross", "net", "stopped", "capped",
        "isin_idx", "rs_at_signal", "entry_sess", "exit_sess"])


# ----------------------------------------- portfolio + daily mark-to-market

def portfolio_daily_mtm(trades: pd.DataFrame, close_by_isin_sess: np.ndarray,
                        session_dates: pd.DatetimeIndex, slots: int = SLOTS) -> dict:
    """Ten-slot portfolio. At each session:
      1) Book realized P&L for positions exiting today (exit_sess == t).
      2) Consider new signals whose entry_sess == t, sorted by rs desc; skip
         a signal if slots full or the isin is already held.
      3) Mark equity = cash + sum(committed * close[isin, t] / fill) for
         each open position. If close is NaN, keep last mark.
    Returns equity Series (session index) + summary stats.
    """
    if len(trades) == 0:
        eq = pd.Series([100.0], index=session_dates[:1])
        return {
            "n_offered": 0, "n_taken": 0, "equity": eq,
            "total": 0.0, "cagr": 0.0, "maxdd": 0.0, "taken_mask": np.array([], dtype=bool),
        }
    # Group offers by entry session
    by_entry: dict[int, list] = {}
    d = trades.sort_values(["entry_sess", "rs_at_signal"], ascending=[True, False]).reset_index(drop=True)
    taken_mask = np.zeros(len(d), dtype=bool)
    for r in d.itertuples():
        by_entry.setdefault(int(r.entry_sess), []).append(r)
    first = int(d.entry_sess.min())
    last = int(d.exit_sess.max())
    n_sess = last - first + 1
    curve = np.empty(n_sess, dtype=float)
    cash = 100.0
    open_pos = []      # dicts: exit_sess, fill, committed, isin_idx, net, last_mark
    held = set()
    taken = 0
    for i, t in enumerate(range(first, last + 1)):
        # 1) Book exits from prior session
        still = []
        for p in open_pos:
            if p["exit_sess"] <= t:
                # settle: cash += committed * (1 + net/100)
                cash += p["committed"] * (1.0 + p["net"] / 100.0)
                held.discard(p["isin_idx"])
            else:
                still.append(p)
        open_pos = still
        # 2) New entries
        equity_pre = cash + sum(_mv(p, close_by_isin_sess, t) for p in open_pos)
        for r in by_entry.get(t, []):
            if len(open_pos) >= slots:
                break
            if r.isin_idx in held:
                continue
            want = equity_pre / slots
            if want > cash:
                # not enough cash - skip (no partial fills)
                continue
            open_pos.append({
                "exit_sess": int(r.exit_sess),
                "fill": float(r.fill),
                "committed": float(want),
                "isin_idx": int(r.isin_idx),
                "net": float(r.net),
                "last_mark": float(want),   # falls back if close missing
            })
            held.add(int(r.isin_idx))
            cash -= want
            taken += 1
            taken_mask[int(r.Index)] = True
        # 3) Mark equity - MtM open positions
        mv = 0.0
        for p in open_pos:
            m = _mv(p, close_by_isin_sess, t)
            mv += m
        curve[i] = cash + mv
    # Settle any still-open on the terminal bar using their scheduled net
    for p in open_pos:
        cash += p["committed"] * (1.0 + p["net"] / 100.0)
    # Overwrite final point with settled cash (should already match roughly)
    curve[-1] = cash
    eq = pd.Series(curve, index=session_dates[first:last + 1])
    years = len(eq) / TRADING_DAYS
    dd = float((eq / eq.cummax() - 1.0).min() * 100.0)
    final = float(eq.iloc[-1])
    cagr = (final / 100.0) ** (1.0 / max(years, 1e-9)) * 100.0 - 100.0
    # Re-map taken_mask back to original trade rows (before sort)
    return {
        "n_offered": int(len(d)),
        "n_taken": int(taken),
        "equity": eq,
        "total": final - 100.0,
        "cagr": float(cagr),
        "maxdd": float(dd),
        "taken_mask": taken_mask,   # aligned to sorted `d`
        "taken_sig_rows": d.loc[taken_mask, "sig_row"].to_numpy() if len(d) else np.array([], dtype=int),
    }


def _mv(p: dict, close_lookup: np.ndarray, t: int) -> float:
    """Current market value of an open position: committed * close/fill.
    Falls back to last known mark if today's close is NaN."""
    c = close_lookup[p["isin_idx"], t] if 0 <= t < close_lookup.shape[1] else np.nan
    if not np.isfinite(c) or c <= 0:
        return p["last_mark"]
    mv = p["committed"] * (c / p["fill"])
    p["last_mark"] = mv
    return mv


# ---------------------------------------------------- per-year breakdown

def by_year_breakdown(eq: pd.Series, trades: pd.DataFrame, taken_rows: np.ndarray) -> list[dict]:
    if eq.empty:
        return []
    out = []
    ann = eq.resample("YE").last()
    first_year = eq.index.year.min()
    taken_set = set(int(x) for x in taken_rows) if len(taken_rows) else set()
    for yr, val in ann.items():
        y = int(yr.year)
        if y == first_year:
            base = 100.0
        else:
            prev = ann[ann.index.year == y - 1]
            base = float(prev.iloc[-1]) if len(prev) else 100.0
        ret_pct = float((val / base - 1.0) * 100.0)
        y_eq = eq[eq.index.year == y]
        y_dd = float((y_eq / y_eq.cummax() - 1.0).min() * 100.0) if len(y_eq) else 0.0
        yr_tr = trades[trades.entry_year == y]
        n_sig = int(len(yr_tr))
        taken_yr = yr_tr[yr_tr["sig_row"].isin(taken_set)] if n_sig else yr_tr
        n_taken = int(len(taken_yr))
        # win_pct is per-signal (over offered) to match the top-level metric
        wr = float((yr_tr.net > 0).mean() * 100.0) if n_sig else 0.0
        out.append({
            "year": y,
            "n_signals": n_sig,
            "n_taken": n_taken,
            "win_pct": round(wr, 2),
            "ret_pct": round(ret_pct, 2),
            "max_dd_pct": round(y_dd, 2),
        })
    return out


# ------------------------------------------------------ per-rule audit runner

def build_panel_lookups(feat: pd.DataFrame):
    """From a sorted feature panel, build the numeric arrays and lookup maps
    needed by the simulator."""
    isins = feat["isin"].to_numpy()
    _, isin_idx = np.unique(isins, return_inverse=True)
    n_isin = int(isin_idx.max()) + 1
    dates = pd.to_datetime(feat["date"]).to_numpy()
    session_dates = pd.DatetimeIndex(sorted(pd.unique(feat["date"])))
    sess_to_i = {d: i for i, d in enumerate(session_dates)}
    row_to_sess = np.array([sess_to_i[d] for d in pd.to_datetime(feat["date"])],
                           dtype=np.int64)
    n_sess = len(session_dates)
    # per-isin last-row index for stop-guard
    order = np.arange(len(feat))
    grp = pd.Series(order).groupby(isin_idx).transform("max").to_numpy()
    LAST_ROW = grp
    # numeric arrays
    O = feat["adj_open"].to_numpy(float)
    H = feat["adj_high"].to_numpy(float)
    L = feat["adj_low"].to_numpy(float)
    C = feat["adj_close"].to_numpy(float)
    ATR_PCT = feat["atr_pct"].to_numpy(float)
    HI7 = feat["hi7_prior"].to_numpy(float) if "hi7_prior" in feat else np.full(len(feat), np.nan)
    EMA21 = feat["ema_21"].to_numpy(float) if "ema_21" in feat else np.full(len(feat), np.nan)
    RS = feat["rs_rank"].to_numpy(float) if "rs_rank" in feat else np.zeros(len(feat))
    # Close-by-(isin,sess) lookup for MtM
    close_lookup = np.full((n_isin, n_sess), np.nan, dtype=float)
    close_lookup[isin_idx, row_to_sess] = C
    return dict(isin_idx=isin_idx, isins=isins, session_dates=session_dates,
                row_to_sess=row_to_sess, LAST_ROW=LAST_ROW,
                O=O, H=H, L=L, C=C, ATR_PCT=ATR_PCT, HI7=HI7, EMA21=EMA21, RS=RS,
                close_lookup=close_lookup, n_isin=n_isin, n_sess=n_sess)


def run_rule(rule: dict, feat: pd.DataFrame, lu: dict, family: str, coverage: str) -> dict:
    name = rule["name"]
    title = rule.get("title", name)
    params = EXIT_PARAMS[family]
    # ------------- evaluate rule expression (state mask) -------------
    expr = rule["expr"]
    state = eval_rule(feat, expr)
    n_state_true = int(state.sum())
    # ------------- convert to fresh events -------------
    events = fresh_events(state, lu["isin_idx"], N=QUIET)
    n_fresh = int(events.sum())
    # ------------- simulate per-signal outcomes ---------------------
    ev_ix = np.flatnonzero(events)
    trades = simulate_signals(
        ev_ix, lu["isins"], lu["isin_idx"],
        lu["O"], lu["C"], lu["ATR_PCT"], lu["HI7"], lu["EMA21"],
        lu["LAST_ROW"], params["atr_mult"], params["cap"], params["exit_rule"],
        lu["row_to_sess"], lu["RS"],
    )
    trades["entry_year"] = pd.to_datetime(lu["session_dates"][trades.entry_sess]).year if len(trades) else pd.Series([], dtype=int)
    # ------------- portfolio simulation + daily MtM DD ---------------
    port = portfolio_daily_mtm(trades, lu["close_lookup"], lu["session_dates"], slots=SLOTS)
    eq = port["equity"]
    # ------------- summary metrics -----------------------------------
    # Per-trade quality stats are over the OFFERED fresh signals: the slot
    # filter is a portfolio-level cap that biases toward high-RS names, so
    # per-signal win/mean is a cleaner statement of the signal edge.
    # CAGR / DD in the portfolio block below reflect the account actually
    # traded through the 10-slot constraint.
    taken_rows_arr = port.get("taken_sig_rows", np.array([], dtype=int))
    if len(trades):
        wr = float((trades.net > 0).mean() * 100.0)
        mean_ret = float(trades.net.mean())
        med_ret = float(trades.net.median())
        stopped = float(trades.stopped.mean() * 100.0)
        capped = float(trades.capped.mean() * 100.0)
    else:
        wr = mean_ret = med_ret = stopped = capped = 0.0
    # by-year
    yrs = by_year_breakdown(eq, trades, taken_rows_arr)
    if yrs:
        ranked = sorted(yrs, key=lambda x: x["ret_pct"])
        worst_year = ranked[0]
        best_year = ranked[-1]
    else:
        worst_year = {"year": None, "ret_pct": 0.0, "max_dd_pct": 0.0}
        best_year = {"year": None, "ret_pct": 0.0}
    # coverage window
    ds = pd.to_datetime(feat["date"])
    return {
        "name": name,
        "title": title,
        "family": family,
        "coverage": coverage,
        "start_date": str(ds.min().date()),
        "end_date": str(ds.max().date()),
        "n_signals_raw": n_state_true,
        "n_signals_fresh": n_fresh,
        "n_signals_taken": int(port["n_taken"]),
        "win_pct": round(wr, 2),
        "mean_ret_pct": round(mean_ret, 3),
        "median_ret_pct": round(med_ret, 3),
        "stopped_pct": round(stopped, 2),
        "capped_pct": round(capped, 2),
        "equity_multiplier": round(eq.iloc[-1] / 100.0, 3) if len(eq) else 1.0,
        "cagr_pct": round(port["cagr"], 2),
        "max_dd_pct": round(port["maxdd"], 2),
        "worst_year": {"year": worst_year["year"],
                       "ret_pct": worst_year["ret_pct"],
                       "max_dd_pct": worst_year["max_dd_pct"]},
        "best_year": {"year": best_year["year"], "ret_pct": best_year["ret_pct"]},
        "by_year": yrs,
        "exit_family": params["exit_rule"],
        "stop_atr_mult": params["atr_mult"],
        "cost_rt": COST_RT,
        "notes": None,
    }


# ------------------------------------------------------------------------ main

def main():
    t0 = time.time()
    rules_path = ROOT / "config" / "rules.yaml"
    rules = yaml.safe_load(rules_path.read_text())
    rules_by_name = {r["name"]: r for r in rules}
    # Group rules by coverage window so each panel loads only once.
    by_cov: dict[str, list[str]] = {"8yr": [], "3yr": []}
    for name in RULE_TABLE:
        if name not in rules_by_name:
            print(f"  WARN: rule '{name}' missing from rules.yaml", flush=True)
            continue
        _, cov = RULE_TABLE[name]
        by_cov[cov].append(name)
    print(f"Rules: 8yr={len(by_cov['8yr'])}, 3yr={len(by_cov['3yr'])}", flush=True)

    results = []
    failures = []
    for cov, names in by_cov.items():
        if not names:
            continue
        print(f"\n---- loading {cov} panel ----", flush=True)
        feat = load_panel(cov)
        print(f"  building lookups...", flush=True)
        lu = build_panel_lookups(feat)
        for name in names:
            rule = rules_by_name[name]
            family, coverage = RULE_TABLE[name]
            print(f"  [{coverage}] {name} ({family})", flush=True)
            try:
                if family == "watchlist":
                    # Still surface a signal count for context.
                    state = eval_rule(feat, rule["expr"])
                    n_state = int(state.sum())
                    events = fresh_events(state, lu["isin_idx"], N=QUIET)
                    n_fresh = int(events.sum())
                    ds = pd.to_datetime(feat["date"])
                    res = {
                        "name": name,
                        "title": rule.get("title", name),
                        "family": "watchlist",
                        "coverage": coverage,
                        "start_date": str(ds.min().date()),
                        "end_date": str(ds.max().date()),
                        "n_signals_raw": n_state,
                        "n_signals_fresh": n_fresh,
                        "n_signals_taken": 0,
                        "win_pct": None, "mean_ret_pct": None, "median_ret_pct": None,
                        "stopped_pct": None, "capped_pct": None,
                        "equity_multiplier": None, "cagr_pct": None, "max_dd_pct": None,
                        "worst_year": None, "best_year": None, "by_year": [],
                        "exit_family": "none", "stop_atr_mult": 0.0, "cost_rt": COST_RT,
                        "notes": "Watchlist: not traded, signal count only.",
                    }
                else:
                    res = run_rule(rule, feat, lu, family, coverage)
            except Exception as ex:
                import traceback
                traceback.print_exc()
                failures.append((name, str(ex)))
                continue
            results.append(res)
    # Sort by CAGR desc; watchlist rules sink to bottom (None sort)
    def sort_key(r):
        c = r["cagr_pct"]
        return c if c is not None else -1e9
    results.sort(key=sort_key, reverse=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {OUT_JSON} ({OUT_JSON.stat().st_size:,} bytes) in {time.time()-t0:.0f}s", flush=True)
    # Summary table to stdout
    print()
    print(f"{'rank':>4} {'name':32s} {'fam':4s} {'cov':4s} {'nfresh':>7} {'ntaken':>7} {'win%':>6} "
          f"{'meanR%':>7} {'CAGR%':>7} {'MaxDD%':>7} {'stop%':>6} {'cap%':>6}")
    for i, r in enumerate(results, 1):
        fam = {"mean_reversion": "MR", "trend_continuation": "TC",
               "momentum_leader": "ML", "watchlist": "WL"}.get(r["family"], "?")
        def fmt(x, w, digits=1, dash="-"):
            if x is None:
                return f"{dash:>{w}}"
            return f"{x:>{w}.{digits}f}"
        print(f"{i:>4d} {r['name']:32s} {fam:4s} {r['coverage']:4s} "
              f"{r['n_signals_fresh']:>7d} {r['n_signals_taken']:>7d} "
              f"{fmt(r['win_pct'], 6, 1)} {fmt(r['mean_ret_pct'], 7, 2)} "
              f"{fmt(r['cagr_pct'], 7, 2)} {fmt(r['max_dd_pct'], 7, 2)} "
              f"{fmt(r['stopped_pct'], 6, 1)} {fmt(r['capped_pct'], 6, 1)}")
    if failures:
        print("\nFailures:")
        for n, msg in failures:
            print(f"  {n}: {msg}")


if __name__ == "__main__":
    main()
