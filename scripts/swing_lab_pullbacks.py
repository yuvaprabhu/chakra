"""Swing lab, pullback / continuation family.

Popular pullback setups, each with a HARD stop, backtested on 2-8 week holds,
then groomed with signal-bar filters chosen in-sample (H1 quintile edges) and
judged out-of-sample (H2), then stressed as a 10-slot account.

Setups (config/rules.yaml): ema21_pullback, oversold_in_uptrend,
monthly_r1_retest, r1_breakout_retest, spring_reclaim, weekly_tc_reclaim,
close_above_monthly_tc, rsi2_reversion, double_seven.  Plus two defined here:
sma50_pullback (50-DMA pullback in an uptrend) and leader_3day (three lower
closes in a relative-strength leader still above the 21 EMA).

Trade engine: scripts/stop_reward_study.py's run(), unchanged in its
conventions (entry next open, stop checked first, gap fills at the open,
close-based exits fill at the next open, one open position per name, 0.30%
cost, time cap), extended with a structural stop, the 7-day-high reversal exit
('rev7'), a cap argument and MAE / MFE.

Output: data/screen/swing_lab_pullbacks.json
Run:    python scripts/swing_lab_pullbacks.py            (matrix, grooming, stress, overlap)
        python scripts/swing_lab_pullbacks.py --stock-only  (grooming with market features excluded)
        python scripts/swing_lab_pullbacks.py --quarterly   (quarterly consistency, refined verdicts)
"""
import json, sys, time, itertools
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.rules import load_rules
from screener.backtest import add_panel_columns, rule_hits
from screener.robustness import stress
from screener import quality
from screener.store import Store

COST, SLOTS = 0.30, 10
START, SPLIT = pd.Timestamp("2024-06-01"), pd.Timestamp("2025-03-19")
CAPS = [60, 20]
MIN_N = 100                       # per half
MIN_KEPT_PER_MONTH = 20
T_MIN = 2.0
SETUPS = ["ema21_pullback", "oversold_in_uptrend", "monthly_r1_retest", "r1_breakout_retest",
          "spring_reclaim", "weekly_tc_reclaim", "close_above_monthly_tc", "rsi2_reversion",
          "double_seven", "sma50_pullback", "leader_3day"]
EXITS = [("trail", "ema21"), ("trail", "ema50"), ("trail", "low10"), ("trail", "chand3"),
         ("target", 2.0), ("target", 3.0), ("none",), ("rev7",)]
ATR_STOPS = [("atr", 2.5), ("atr", 3.0), ("atr", 4.0)]
STRUCT_STOP = {n: ("struct", "low5") for n in SETUPS}
STRUCT_STOP["sma50_pullback"] = ("struct", "sma50atr")
STRUCT_STOP["spring_reclaim"] = ("struct", "spring")

# ------------------------------------------------------------------ panel (verbatim from stop_reward_study.py)
t0 = time.time()
feat = pd.read_parquet("data/screen/features.parquet")
lat = pd.read_parquet("data/screen/latest_features.parquet")
feat = feat.merge(lat[["isin", "cap_band"]], on="isin", how="left")
st = Store("data")
b = feat[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
feat["contaminated"] = quality.contamination_mask(b, quality.detect_price_jumps(b, st.read_adjustments())).to_numpy()
feat = add_panel_columns(feat).sort_values(["isin", "date"]).reset_index(drop=True)
g = feat.groupby("isin")
feat["low10_prior"] = g["adj_low"].transform(lambda s: s.shift(1).rolling(10).min())
feat["low5_prior"] = g["adj_low"].transform(lambda s: s.shift(1).rolling(5).min())
feat["ret1"] = g["adj_close"].pct_change()
feat["liquid"] = ((feat["turnover_median_20d"] >= 1e7) & (feat["bars_available"] >= 250)
                  & (~feat["contaminated"]))
# ---- additions for this lab
feat["low5_incl"] = g["adj_low"].transform(lambda s: s.rolling(5).min())           # the pullback low, today included
feat["dist_sma_200"] = (feat["adj_close"] / feat["sma_200"] - 1) * 100
feat["max_dist_sma50_20"] = g["dist_sma_50"].transform(lambda s: s.shift(1).rolling(20).max())
feat["down3"] = ((feat["ret1"] < 0) & (g["ret1"].shift(1) < 0) & (g["ret1"].shift(2) < 0))
feat["sma50_minus_atr"] = feat["sma_50"] - feat["atr_14"]

CAL = pd.DatetimeIndex(sorted(feat["date"].unique()))
CALPOS = {d: i for i, d in enumerate(CAL)}
END = CAL[-1]
mkt = feat.loc[feat["liquid"]].groupby("date")["ret1"].mean().reindex(CAL).fillna(0)
BENCH = (1 + mkt).cumprod()

O = feat["adj_open"].to_numpy(float); H = feat["adj_high"].to_numpy(float)
L = feat["adj_low"].to_numpy(float);  C = feat["adj_close"].to_numpy(float)
ATRP = feat["atr_pct"].to_numpy(float) / 100.0
E21 = feat["ema_21"].to_numpy(float); E50 = feat["ema_50"].to_numpy(float)
LO10 = feat["low10_prior"].to_numpy(float); LO5 = feat["low5_prior"].to_numpy(float)
HI7 = feat["hi7_prior"].to_numpy(float)
RS = feat["rs_rank"].to_numpy(float)
ISIN = feat["isin"].to_numpy(); DATES = feat["date"].to_numpy()
LAST = pd.Series(np.arange(len(feat))).groupby(pd.factorize(feat["isin"])[0]).transform("max").to_numpy()
IN_WIN = ((feat["date"] >= START) & (feat["date"] <= END)).to_numpy()
STRUCT = {"low5": feat["low5_incl"].to_numpy(float), "sma50atr": feat["sma50_minus_atr"].to_numpy(float),
          "spring": feat["spring_low"].to_numpy(float)}
SYM = feat["symbol"].to_numpy()
print(f"panel {len(feat):,} rows | {CAL[0].date()} to {END.date()} | signals from {START.date()}, "
      f"halves split {SPLIT.date()} | {time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------------------------ market state, gate
GATE = pd.read_parquet("data/screen/gate_r3.parquet").set_index("date")["R3"]
MS_COLS = ["nifty_above_20dma", "nifty_above_50dma", "nifty_above_200dma", "vix", "vix_pctile_1y",
           "breadth_above_50", "breadth_thrust_10", "dispersion", "nifty_range_20d"]
MS = pd.read_parquet("data/screen/market_state.parquet").set_index("date")[MS_COLS]
R3 = feat["date"].map(GATE).fillna(0).to_numpy() > 0

# ------------------------------------------------------------------ signals
rules = {r.name: r for r in load_rules("config/rules.yaml")}
SIG = {}
for n in SETUPS:
    if n in rules:
        SIG[n] = rule_hits(feat, rules[n]).to_numpy() & IN_WIN
base_ok = (feat["liquid"] & (feat["close"].fillna(0) >= 20)).to_numpy()
SIG["sma50_pullback"] = (base_ok & IN_WIN & (feat["sma_50"] > feat["sma_200"]).to_numpy()
                         & (feat["dist_sma_50"].abs() <= 3).to_numpy() & (feat["max_dist_sma50_20"] >= 8).to_numpy())
SIG["leader_3day"] = (base_ok & IN_WIN & (feat["rs_rank"] >= 80).to_numpy() & (feat["adj_close"] > feat["sma_50"]).to_numpy()
                      & feat["down3"].fillna(False).to_numpy() & (feat["adj_close"] > feat["ema_21"]).to_numpy())
# the bar to beat: leader_dip in its groomed final form
SIG["leader_dip_final"] = (rule_hits(feat, rules["leader_dip"]).to_numpy() & IN_WIN
                           & (feat["dist_ema_21"] >= 2.6).to_numpy() & R3)
for n, s in SIG.items():
    print(f"  {n:24s} signals {s.sum():>8,d}  days {feat.loc[s, 'date'].nunique():>4d}", flush=True)

TITLE = {n: (rules[n].title if n in rules else n) for n in SETUPS}
TITLE.update({"sma50_pullback": "50-DMA Pullback", "leader_3day": "Leader 3-Day Dip", "leader_dip_final": "Leader Dip (final)"})
EMIT_SKIP = {"adj_close", "sma_200", "m_cpr_tc", "m_cpr_bc", "m_r1", "lo7_prior", "hi7_prior", "spring_low", "w_cpr_tc"}
STOCK_BASE = ["rs_rank", "dist_ema_21", "dist_sma_50", "dist_sma_200", "rsi_14", "atr_pct", "vol_ratio",
              "ret_20d", "ret_60d", "pct_from_52w_high", "turnover_median_20d"]
EXTRA = {"close_above_monthly_tc": ["since_m_cpr_tc_cross", "dist_m_cpr_tc", "m_cpr_width_rank"],
         "sma50_pullback": ["max_dist_sma50_20", "sma_50_slope", "rsi_2"],
         "leader_3day": ["rsi_2", "ret_5d", "close_pos"]}
CANDS = {}
for n in SETUPS:
    emits = [c for c in (rules[n].emit if n in rules else []) if c not in EMIT_SKIP and c in feat.columns]
    CANDS[n] = list(dict.fromkeys(STOCK_BASE + emits + EXTRA.get(n, [])))
ALL_STOCK_COLS = sorted(set(itertools.chain.from_iterable(CANDS.values())) | {"cap_band"})
MARKET_NUM = ["vix", "vix_pctile_1y", "breadth_above_50", "breadth_thrust_10", "dispersion", "nifty_range_20d"]
MARKET_BIN = ["R3", "nifty_above_20dma", "nifty_above_50dma", "nifty_above_200dma"]


# ------------------------------------------------------------------ engine
def run(sig, stop, reward, cap):
    """One trade per signal, with a hard initial stop.

    stop:   ('pct', w) flat w% | ('atr', m) m x ATR(14)% at the signal bar
            ('struct', 'low5'|'sma50atr'|'spring') a level known at the signal close
    reward: ('target', rr) | ('trail', 'ema21'|'ema50'|'low10'|'chand3') exit next open after a close below
            ('rev7',) exit next open after the first close above the prior 7 closes
            ('none',) stop and time cap only
    """
    rows = []; busy = {}
    for t in np.flatnonzero(sig):
        if t + 1 > LAST[t] or busy.get(ISIN[t], -1) >= t or not np.isfinite(ATRP[t]):
            continue
        e = O[t + 1]
        if not np.isfinite(e) or e <= 0:
            continue
        if stop[0] == "pct":
            risk = e * stop[1] / 100.0
        elif stop[0] == "atr":
            risk = stop[1] * ATRP[t] * e
        else:
            lvl = STRUCT[stop[1]][t]
            if not np.isfinite(lvl):
                continue
            risk = e - lvl
        rpct = risk / e * 100.0
        if not (0.2 < rpct < 20):
            continue
        sl = e - risk
        tgt = e + reward[1] * risk if reward[0] == "target" else np.inf
        atr_abs = ATRP[t] * e
        end = min(LAST[t], t + cap)
        k = t + 1; peak = e; lo = np.inf; hi = -np.inf
        out = None; why = "time"
        while k <= end:
            lo = min(lo, L[k]); hi = max(hi, H[k])
            if L[k] <= sl:
                out = min(O[k], sl); why = "stop"; break
            if reward[0] == "target" and H[k] >= tgt:
                out = max(O[k], tgt); why = "target"; break
            peak = max(peak, H[k])
            if reward[0] == "trail":
                lvl = (E21[k] if reward[1] == "ema21" else E50[k] if reward[1] == "ema50"
                       else LO10[k] if reward[1] == "low10" else peak - 3.0 * atr_abs)
                if np.isfinite(lvl) and C[k] < lvl and k + 1 <= LAST[t]:
                    out = O[k + 1]; why = "trail"; k += 1; break
            elif reward[0] == "rev7":
                if np.isfinite(HI7[k]) and C[k] > HI7[k] and k + 1 <= LAST[t]:
                    out = O[k + 1]; why = "rev7"; k += 1; break
            k += 1
        if out is None:
            k = min(k, end); out = C[k]
        if not np.isfinite(out):
            continue
        busy[ISIN[t]] = k
        ei, xi = CALPOS[pd.Timestamp(DATES[t + 1])], CALPOS[pd.Timestamp(DATES[k])]
        rows.append((pd.Timestamp(DATES[t]), ISIN[t], t, ei, xi, e, out, (out / e - 1) * 100 - COST, k - t,
                     rpct, RS[t], why, (min(lo, out) / e - 1) * 100, (max(hi, out) / e - 1) * 100))
    return pd.DataFrame(rows, columns=["date", "isin", "row", "entry_i", "exit_i", "entry", "exit", "net", "bars",
                                       "risk_pct", "rs", "why", "mae", "mfe"])


def attach(tr):
    """Signal-bar stock features and same-day market state, for grooming."""
    if not len(tr):
        return tr
    f = feat.iloc[tr["row"].to_numpy()][ALL_STOCK_COLS].reset_index(drop=True)
    tr = pd.concat([tr.reset_index(drop=True), f], axis=1)
    tr["R3"] = tr["date"].map(GATE).fillna(0).astype(int)
    ms = MS.reindex(tr["date"]).reset_index(drop=True)
    tr = pd.concat([tr, ms], axis=1)
    tr["half"] = np.where(tr["date"] < SPLIT, "H1", "H2")
    tr["win"] = tr["net"] > 0
    return tr


def stats(t):
    if len(t) < MIN_N:
        return None
    n = t["net"]; w = n[n > 0]; l = n[n <= 0]
    if not len(w) or not len(l):
        return None
    why = t["why"].value_counts(normalize=True) * 100
    return {"n": int(len(t)), "win": float((n > 0).mean() * 100), "net": float(n.mean()), "median": float(n.median()),
            "avg_win": float(w.mean()), "avg_loss": float(l.mean()), "payoff": float(w.mean() / -l.mean()),
            "pf": float(w.sum() / -l.sum()), "stopped": float(why.get("stop", 0)), "bars": float(t["bars"].mean()),
            "exp_r": float((n / t["risk_pct"]).mean()), "risk": float(t["risk_pct"].mean()),
            "mae_med": float(t["mae"].median()), "mfe_med": float(t["mfe"].median()),
            "days": int(t["date"].nunique()), "per_month": float(len(t) / max((t["date"].max() - t["date"].min()).days / 30.44, 1))}


def label(stop, reward, cap):
    s = f"{stop[1]:g} ATR" if stop[0] == "atr" else f"{stop[1]:g}%" if stop[0] == "pct" else \
        {"low5": "5d low", "sma50atr": "50DMA-1ATR", "spring": "spring low"}[stop[1]]
    r = (f"target 1:{reward[1]:g}" if reward[0] == "target" else
         {"ema21": "close<21EMA", "ema50": "close<50EMA", "low10": "close<10d low", "chand3": "chandelier 3ATR"}[reward[1]]
         if reward[0] == "trail" else "7d-high reversal" if reward[0] == "rev7" else "stop only")
    return s, f"{r} cap{cap}"


def cell(job):
    setup, stop, reward, cap = job
    tr = run(SIG[setup], stop, reward, cap)
    if not len(tr):
        return None
    a, s1, s2 = stats(tr), stats(tr[tr["date"] < SPLIT]), stats(tr[tr["date"] >= SPLIT])
    if not (a and s1 and s2):
        return None
    sl, rl = label(stop, reward, cap)
    return {"setup": setup, "stop": sl, "exit": rl, "stop_spec": list(stop), "exit_spec": list(reward), "cap": cap,
            "all": a, "h1": s1, "h2": s2, "both": bool(s1["net"] > 0 and s2["net"] > 0),
            "worse_net": float(min(s1["net"], s2["net"])), "worse_win": float(min(s1["win"], s2["win"]))}


# ------------------------------------------------------------------ grooming
def cluster_t(y, x, grp):
    y = np.asarray(y, float); x = np.asarray(x, float)
    if len(y) < 10 or x.std() == 0:
        return np.nan
    X = np.column_stack([np.ones(len(x)), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    S = pd.DataFrame({"g": np.asarray(grp), "e0": e, "e1": e * x}).groupby("g")[["e0", "e1"]].sum().to_numpy()
    V = XtX_inv @ (S.T @ S) @ XtX_inv
    return float(beta[1] / np.sqrt(V[1, 1])) if V[1, 1] > 0 else np.nan


def r(x, k=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), k)


def eval_filter(sub, mask):
    res = {}
    for h in ("H1", "H2"):
        s = sub[sub["half"] == h]; m = mask.loc[s.index].fillna(False).to_numpy(bool)
        kept, rem = s[m], s[~m]
        months = max((s["date"].max() - s["date"].min()).days / 30.44, 1) if len(s) else 1
        n = kept["net"]; w = n[n > 0]; l = n[n <= 0]
        res[h] = {"n_all": int(len(s)), "n_kept": int(len(kept)), "kept_pct": r(len(kept) / max(len(s), 1)),
                  "kept_per_month": r(len(kept) / months, 1),
                  "wr_kept": r(kept["win"].mean()) if len(kept) else None,
                  "mean_kept": r(kept["net"].mean()) if len(kept) else None,
                  "pf_kept": r(w.sum() / -l.sum(), 2) if len(l) and l.sum() < 0 else None,
                  "stopped_kept": r((kept["why"] == "stop").mean()) if len(kept) else None,
                  "wr_removed": r(rem["win"].mean()) if len(rem) else None,
                  "mean_removed": r(rem["net"].mean()) if len(rem) else None,
                  "t_cluster": r(cluster_t(s["net"], m.astype(float), s["date"]), 2) if 0 < m.sum() < len(s) else None}
    return res


def passes(res, base):
    ts = []
    for h in ("H1", "H2"):
        d = res[h]
        if d["wr_kept"] is None or d["mean_kept"] is None or d["t_cluster"] is None:
            return False
        if not (d["wr_kept"] > base[h]["win"] and d["mean_kept"] > base[h]["net"]):
            return False
        if d["kept_per_month"] < MIN_KEPT_PER_MONTH:
            return False
        ts.append(d["t_cluster"])
    return bool(max(ts) >= T_MIN and min(ts) > 0)


def score(res):
    return (min(res["H1"]["wr_kept"], res["H2"]["wr_kept"]), min(res["H1"]["mean_kept"], res["H2"]["mean_kept"]),
            min(res["H1"]["kept_pct"], res["H2"]["kept_pct"]))


def mask_of(sub, f):
    if f["op"] == ">=":
        return sub[f["feature"]] >= f["thr"]
    if f["op"] == "<=":
        return sub[f["feature"]] <= f["thr"]
    if f["op"] == "in":
        return sub[f["feature"]].isin(f["thr"])
    return sub[f["feature"]] == f["thr"]


def groom(sub, cands):
    base = {h: {"win": float(sub.loc[sub["half"] == h, "win"].mean()), "net": float(sub.loc[sub["half"] == h, "net"].mean())}
            for h in ("H1", "H2")}
    h1 = sub[sub["half"] == "H1"]
    singles = []; n_tests = 0
    for f in cands + MARKET_NUM:
        x = h1[f].dropna()
        if len(x) < 100:
            continue
        edges = np.unique(np.nanpercentile(x, [20, 40, 60, 80]))
        for thr in edges:
            for op in (">=", "<="):
                spec = {"feature": f, "op": op, "thr": float(thr), "level": "market" if f in MARKET_NUM else "stock"}
                m = mask_of(sub, spec); n_tests += 1
                res = eval_filter(sub, m)
                if res["H1"]["n_kept"] < 30 or res["H2"]["n_kept"] < 30:
                    continue
                spec.update({"filter": f"{f} {op} {thr:.4g}", "passes": passes(res, base), "H1": res["H1"], "H2": res["H2"]})
                singles.append(spec)
    bins = [{"feature": f, "op": "==", "thr": v, "filter": f"{f} == {v}", "level": "market"} for f in MARKET_BIN for v in (1, 0)]
    bins += [{"feature": "cap_band", "op": "in", "thr": v, "filter": f"cap_band in {v}", "level": "stock"}
             for v in (["Mid"], ["Large", "Mega"], ["Mega"])]
    for spec in bins:
        m = mask_of(sub, spec); n_tests += 1
        res = eval_filter(sub, m)
        if res["H1"]["n_kept"] < 30 or res["H2"]["n_kept"] < 30:
            continue
        spec.update({"passes": passes(res, base), "H1": res["H1"], "H2": res["H2"]})
        singles.append(spec)
    passing = sorted([f for f in singles if f["passes"]], key=lambda f: score(f), reverse=True)
    best_per_feat = {}
    for f in passing:
        best_per_feat.setdefault(f["feature"], f)
    top = list(best_per_feat.values())[:8]
    combos = []
    for a, bb in itertools.combinations(top, 2):
        m = mask_of(sub, a) & mask_of(sub, bb); n_tests += 1
        res = eval_filter(sub, m)
        if res["H1"]["n_kept"] < 30 or res["H2"]["n_kept"] < 30:
            continue
        combos.append({"filter": f"{a['filter']} AND {bb['filter']}", "parts": [a, bb], "passes": passes(res, base),
                       "H1": res["H1"], "H2": res["H2"]})
    board = sorted([f for f in singles + combos if f["passes"]], key=lambda f: score(f), reverse=True)
    return {"base": base, "n_tests": n_tests, "n_single_passing": len(passing), "n_combo_passing": sum(f["passes"] for f in combos),
            "leaderboard": board[:25], "all_singles": [{k: v for k, v in f.items()} for f in singles]}


def groom_job(job):
    setup, stop, reward, cap = job
    tr = attach(run(SIG[setup], stop, reward, cap))
    out = groom(tr, CANDS[setup])
    out["cell"] = {"setup": setup, "stop_spec": list(stop), "exit_spec": list(reward), "cap": cap}
    return out


# ------------------------------------------------------------------ portfolio
_CACHE = {}


def make_trades_from(spec):
    """spec: setup, stop_spec, exit_spec, cap, filters [{feature, op, thr}], gate."""
    def mk(p):
        stop = ("atr", p["atr_mult"]) if spec["stop_spec"][0] == "atr" else tuple(spec["stop_spec"])
        reward = ("target", p["rr"]) if spec["exit_spec"][0] == "target" else tuple(spec["exit_spec"])
        key = (spec["setup"], stop, reward, int(p["cap"]))
        if key not in _CACHE:
            _CACHE[key] = attach(run(SIG[spec["setup"]], stop, reward, int(p["cap"])))
        tr = _CACHE[key]
        m = pd.Series(True, index=tr.index)
        for i, f in enumerate(spec["filters"]):
            thr = p.get(f"thr{i}", f["thr"])
            m &= mask_of(tr, {**f, "thr": thr}).fillna(False)
        if spec.get("gate"):
            m &= tr["R3"] == 1
        return tr[m][["date", "isin", "entry_i", "exit_i", "net", "rs", "why", "bars"]].reset_index(drop=True)
    return mk


def stress_job(spec):
    p = {"cap": int(spec["cap"])}
    if spec["stop_spec"][0] == "atr":
        p["atr_mult"] = float(spec["stop_spec"][1])
    if spec["exit_spec"][0] == "target":
        p["rr"] = float(spec["exit_spec"][1])
    for i, f in enumerate(spec["filters"]):
        if f["op"] in (">=", "<=") and isinstance(f["thr"], float) and f["thr"] != 0:
            p[f"thr{i}"] = float(f["thr"])
    mk = make_trades_from(spec)
    res = stress(mk, p, sessions=CAL, split=SPLIT, slots=SLOTS, order_col="rs", n_random=100)
    tr = mk(p)
    for h, sub in (("all", tr), ("H1", tr[tr["date"] < SPLIT]), ("H2", tr[tr["date"] >= SPLIT])):
        n = sub["net"]
        res[f"trades_{h}"] = {"n": int(len(sub)), "win": r(float((n > 0).mean() * 100), 1) if len(sub) else None,
                              "net": r(float(n.mean()), 2) if len(sub) else None,
                              "per_month": r(len(sub) / max((sub["date"].max() - sub["date"].min()).days / 30.44, 1), 1) if len(sub) else None}
    res["label"] = spec["label"]; res["setup"] = spec["setup"]; res["kind"] = spec.get("kind")
    res["spec"] = {k: v for k, v in spec.items() if k != "label"}
    res["keys"] = sorted(set(zip(tr["isin"], tr["date"].astype(str))))
    return res


# ------------------------------------------------------------------ second pass: quarterly consistency
from screener.portfolio import simulate

QUARTERS = [str(p) for p in pd.period_range("2024Q2", "2026Q3", freq="Q")]
TEST_QUARTERS = QUARTERS[1:]                 # 2024Q3 .. 2026Q3: nine quarters; 2024Q2 holds June only
Q_MIN_POS, Q_WORST = 6, -8.0


def quarter_table(tr):
    """Per calendar quarter of the SIGNAL date: trades, win %, mean net; and the
    10-slot account's return over that quarter (equity curve of one rs-ordered
    simulation, quarter-end to quarter-end; profit lands on the exit session)."""
    t = tr.sort_values(["entry_i", "rs"], ascending=[True, False])
    eq = simulate(t, slots=SLOTS, sessions=CAL)["equity"] if len(t) >= 1 else pd.Series(dtype=float)
    qe = eq.groupby(eq.index.to_period("Q").astype(str)).last() if len(eq) else pd.Series(dtype=float)
    rows = []; prev = 100.0
    for q in QUARTERS:
        sub = tr[tr["date"].dt.to_period("Q").astype(str) == q]
        n = sub["net"]
        pr = None
        if q in qe.index:
            pr = float((qe[q] / prev - 1) * 100); prev = float(qe[q])
        rows.append({"q": q, "n": int(len(sub)), "win": r(float((n > 0).mean() * 100), 1) if len(sub) else None,
                     "net": r(float(n.mean()), 2) if len(sub) else None, "port_ret": r(pr, 2),
                     "stopped": r(float((sub["why"] == "stop").mean() * 100), 0) if len(sub) else None})
    test = [x for x in rows if x["q"] in TEST_QUARTERS and x["port_ret"] is not None]
    pos = sum(x["port_ret"] > 0 for x in test)
    worst = min((x["port_ret"] for x in test), default=None)
    lost = [x["q"] for x in test if x["port_ret"] <= 0]
    return {"table": rows, "quarters_tested": len(test), "quarters_positive": pos, "share_positive": r(pos / max(len(test), 1)),
            "worst_quarter": r(worst, 2), "worst_q": min(test, key=lambda x: x["port_ret"])["q"] if test else None,
            "lost_quarters": lost, "quarter_pass": bool(len(test) and pos >= Q_MIN_POS and worst is not None and worst > Q_WORST)}


def market_context():
    """What the market did each quarter: equal-weight liquid universe return,
    mean VIX, mean % above 50-DMA, share of sessions with the R3 gate on."""
    ms = pd.read_parquet("data/screen/market_state.parquet").set_index("date")
    out = []
    for q in QUARTERS:
        p = pd.Period(q, freq="Q"); lo, hi = p.start_time, p.end_time
        bh = BENCH.loc[lo:hi]; m_ = ms.loc[lo:hi]; g_ = GATE.loc[lo:hi]
        prev = BENCH.loc[:lo - pd.Timedelta(days=1)]
        base = float(prev.iloc[-1]) if len(prev) else float(bh.iloc[0])
        out.append({"q": q, "universe_ret": r((float(bh.iloc[-1]) / base - 1) * 100, 1) if len(bh) else None,
                    "universe_maxdd": r(float((bh / bh.cummax() - 1).min() * 100), 1) if len(bh) else None,
                    "vix_mean": r(float(m_["vix"].mean()), 1), "breadth50_mean": r(float(m_["breadth_above_50"].mean()), 0),
                    "r3_on_share": r(float(g_.mean()), 2), "nifty_above_200_share": r(float(m_["nifty_above_200dma"].mean()), 2)})
    return out


def quarterly_pass():
    J = json.load(open("data/screen/swing_lab_pullbacks.json"))
    specs = [p["spec"] | {"label": p["label"]} for p in J["portfolio"]]
    specs += [p["spec"] | {"label": p["label"]} for p in J.get("supplement", {}).get("portfolio_stock_only", [])]
    ctx = market_context()
    print("\nMARKET BY QUARTER")
    print(f"{'q':8s}{'univ ret':>9s}{'univ DD':>8s}{'VIX':>6s}{'b50':>5s}{'R3 on':>6s}{'N>200':>6s}")
    for c in ctx:
        print(f"{c['q']:8s}{c['universe_ret']:>+9.1f}{c['universe_maxdd']:>8.1f}{c['vix_mean']:>6.1f}{c['breadth50_mean']:>5.0f}{c['r3_on_share']:>6.2f}{c['nifty_above_200_share']:>6.2f}")
    with Pool(4) as pool:
        Q = pool.map(quarter_job, specs, chunksize=1)
    print("\nQUARTERLY, 10-slot account return per quarter (signal-date quarter for n/win/net)")
    print(f"{'strategy':70s}" + "".join(f"{q[2:]:>8s}" for q in QUARTERS) + f" | {'pos':>4s}{'worst':>7s} q-pass")
    for x in Q:
        print(f"{x['label'][:70]:70s}" + "".join(f"{(t['port_ret'] if t['port_ret'] is not None else float('nan')):>+8.1f}" for t in x["table"])
              + f" | {x['quarters_positive']:>2d}/{x['quarters_tested']:<2d}{x['worst_quarter'] if x['worst_quarter'] is not None else float('nan'):>+7.1f} {'YES' if x['quarter_pass'] else 'no'}")
    # refined verdicts: per setup, the best variant by (verdict rank, worst quarter)
    RANK = {"TRADE": 2, "MAYBE": 1, "DROP": 0}
    verd = {}
    for n in SETUPS + ["leader_dip_final"]:
        vs = [x for x in Q if x["setup"] == n]
        if not vs:
            continue
        best = max(vs, key=lambda x: (RANK[x["verdict"]], x["worst_quarter"] if x["worst_quarter"] is not None else -99, x["win_all"]))
        verd[n] = {"verdict": best["verdict"], "variant": best["label"], "win_all": best["win_all"], "both_halves_positive": best["halves_positive"],
                   "quarters_positive": f"{best['quarters_positive']}/{best['quarters_tested']}", "worst_quarter": best["worst_quarter"],
                   "worst_q": best["worst_q"], "lost_quarters": best["lost_quarters"], "maxdd": best["maxdd"], "cagr": best["cagr"],
                   "n": best["n_all"], "per_month": best["per_month"]}
    print("\nREFINED VERDICTS (TRADE = win>=65% overall, both halves > 0, >= 6/9 quarters > 0, no quarter < -8% on the account)")
    for n, v in verd.items():
        print(f"  {n:24s} {v['verdict']:6s} win {v['win_all']:.0f}% quarters {v['quarters_positive']} worst {v['worst_quarter']:+.1f} ({v['worst_q']}) "
              f"lost {v['lost_quarters']}  <- {v['variant'][:70]}")
    J["quarterly"] = {"note": "port_ret = 10-slot rs-ordered account, quarter-end to quarter-end equity (marked at cost, profit on exit); "
                              "n/win/net by signal-date quarter; test quarters 2024Q3..2026Q3",
                      "rule": {"min_quarters_positive": Q_MIN_POS, "worst_quarter_floor": Q_WORST, "min_win": 65},
                      "market_context": ctx, "strategies": Q, "verdicts": verd}
    Path("data/screen/swing_lab_pullbacks.json").write_text(json.dumps(J, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else (o.item() if hasattr(o, "item") else str(o))))
    print(f"\nquarterly pass written | {time.time()-t0:.0f}s")


def quarter_job(spec):
    p = {"cap": int(spec["cap"])}
    if spec["stop_spec"][0] == "atr":
        p["atr_mult"] = float(spec["stop_spec"][1])
    if spec["exit_spec"][0] == "target":
        p["rr"] = float(spec["exit_spec"][1])
    tr = make_trades_from(spec)(p)
    qt = quarter_table(tr)
    n = tr["net"]; h1, h2 = tr[tr["date"] < SPLIT]["net"], tr[tr["date"] >= SPLIT]["net"]
    full = simulate(tr.sort_values(["entry_i", "rs"], ascending=[True, False]), slots=SLOTS, sessions=CAL) if len(tr) >= 10 else {"cagr": np.nan, "maxdd": np.nan}
    win = float((n > 0).mean() * 100) if len(tr) else 0.0
    halves = bool(len(h1) and len(h2) and h1.mean() > 0 and h2.mean() > 0)
    halves_pass = halves and win >= 65
    verdict = "TRADE" if (halves_pass and qt["quarter_pass"]) else "MAYBE" if halves_pass else "DROP"
    return {"label": spec["label"], "setup": spec["setup"], "kind": spec.get("kind"), "spec": {k: v for k, v in spec.items() if k != "label"},
            "n_all": int(len(tr)), "win_all": r(win, 1), "net_all": r(float(n.mean()), 2) if len(tr) else None,
            "per_month": r(len(tr) / max((tr["date"].max() - tr["date"].min()).days / 30.44, 1), 1) if len(tr) else None,
            "h1_net": r(float(h1.mean()), 2) if len(h1) else None, "h2_net": r(float(h2.mean()), 2) if len(h2) else None,
            "halves_positive": halves, "halves_pass": halves_pass, "cagr": r(full["cagr"], 1), "maxdd": r(full["maxdd"], 1),
            "verdict": verdict, **qt}


# ------------------------------------------------------------------ second pass (a): stock-only grooming
def months_active(tr, mask):
    mo_all = tr["date"].dt.to_period("M").nunique()
    mo_kept = tr.loc[mask, "date"].dt.to_period("M").value_counts()
    return float((mo_kept >= 5).sum() / max(mo_all, 1)), int(mo_all)


def stock_only_job(args):
    """Groom one cell with market-level features excluded (R3 gate allowed), and
    measure the regime concentration of the first pass's market-filter pick."""
    setup, cellspec, main_pick = args
    stop, reward, cap = tuple(cellspec["stop_spec"]), tuple(cellspec["exit_spec"]), cellspec["cap"]
    tr = attach(run(SIG[setup], stop, reward, cap))
    gr = groom(tr, CANDS[setup])
    out = {"setup": setup, "cell": {"stop_spec": list(stop), "exit_spec": list(reward), "cap": cap}, "base": gr["base"],
           "n_single_passing": gr["n_single_passing"], "n_combo_passing": gr["n_combo_passing"], "leaderboard": gr["leaderboard"][:10]}
    if main_pick:
        parts = main_pick["parts"] if "parts" in main_pick else [main_pick]
        mk = pd.Series(True, index=tr.index)
        for q in parts:
            mk &= mask_of(tr, q).fillna(False)
        share, n_mo = months_active(tr, mk)
        out["market_pick"] = {"filter": main_pick["filter"], "months_active_share": share, "n_months": n_mo}
    if gr["leaderboard"]:
        f = gr["leaderboard"][0]; parts = f["parts"] if "parts" in f else [f]
        mk = pd.Series(True, index=tr.index)
        for q in parts:
            mk &= mask_of(tr, q).fillna(False)
        share, n_mo = months_active(tr, mk)
        out["stock_pick"] = {"filter": f["filter"], "months_active_share": share, "n_months": n_mo,
                             "parts": [{"feature": q["feature"], "op": q["op"], "thr": q["thr"]} for q in parts]}
        kept = tr[mk]
        for h, sub in (("all", kept), ("H1", kept[kept["date"] < SPLIT]), ("H2", kept[kept["date"] >= SPLIT])):
            n = sub["net"]; w = n[n > 0]; l = n[n <= 0]
            out["stock_pick"][h] = {"n": int(len(sub)), "win": float((n > 0).mean() * 100), "net": float(n.mean()),
                                    "pf": float(w.sum() / -l.sum()) if len(l) else None, "bars": float(sub["bars"].mean()),
                                    "stopped": float((sub["why"] == "stop").mean() * 100), "mae_med": float(sub["mae"].median()),
                                    "worst": float(n.min())}
    return out


def stock_only_pass():
    global MARKET_NUM, MARKET_BIN
    J = json.load(open("data/screen/swing_lab_pullbacks.json"))
    MARKET_NUM, MARKET_BIN = [], ["R3"]
    jobs = []
    for n in SETUPS:
        for c in J["best_cells"][n]:
            main = [g for g in J["grooming"] if g["cell"]["setup"] == n and g["cell"]["stop_spec"] == c["stop_spec"]
                    and g["cell"]["exit_spec"] == c["exit_spec"] and g["cell"]["cap"] == c["cap"]]
            jobs.append((n, c, main[0]["leaderboard"][0] if main and main[0]["leaderboard"] else None))
    with Pool(4) as pool:
        SUP = pool.map(stock_only_job, jobs, chunksize=1)
    for s in SUP:
        c = s["cell"]; sl, rl = label(tuple(c["stop_spec"]), tuple(c["exit_spec"]), c["cap"]); b = s["base"]
        print(f"\nS. {TITLE[s['setup']]} | {sl} | {rl}  base H1 {b['H1']['win']*100:.0f}% {b['H1']['net']:+.2f} | H2 {b['H2']['win']*100:.0f}% {b['H2']['net']:+.2f}"
              f"  stock singles pass={s['n_single_passing']} combos={s['n_combo_passing']}")
        if "market_pick" in s:
            print(f"   market pick {s['market_pick']['filter'][:50]:50s} months-active {s['market_pick']['months_active_share']*100:.0f}% of {s['market_pick']['n_months']}")
        for f in s["leaderboard"][:6]:
            print(f"   {f['filter'][:58]:58s} H1 {f['H1']['n_kept']:>5d} {f['H1']['wr_kept']*100:>3.0f}% {f['H1']['mean_kept']:>+5.2f} t{f['H1']['t_cluster']:>+5.1f} {f['H1']['kept_per_month']:>4.0f}/mo"
                  f" | H2 {f['H2']['n_kept']:>5d} {f['H2']['wr_kept']*100:>3.0f}% {f['H2']['mean_kept']:>+5.2f} t{f['H2']['t_cluster']:>+5.1f} {f['H2']['kept_per_month']:>4.0f}/mo")
    specs = []
    for n in SETUPS:
        cands = [s for s in SUP if s["setup"] == n and "stock_pick" in s]
        if not cands:
            continue
        best = max(cands, key=lambda s: (min(s["stock_pick"]["H1"]["win"], s["stock_pick"]["H2"]["win"]),
                                         min(s["stock_pick"]["H1"]["net"], s["stock_pick"]["H2"]["net"])))
        c = best["cell"]
        for gate in (False, True):
            specs.append({"label": f"{n} | stock-groomed{' + R3 gate' if gate else ''}: {best['stock_pick']['filter']}", "setup": n,
                          "stop_spec": c["stop_spec"], "exit_spec": c["exit_spec"], "cap": c["cap"],
                          "filters": best["stock_pick"]["parts"], "gate": gate, "kind": "stock_groomed" + ("_gate" if gate else "")})
    with Pool(4) as pool:
        PORT = pool.map(stress_job, specs, chunksize=1)
    print(f"\n{'strategy':80s}{'n':>6s}{'win':>5s}{'net':>6s}{'/mo':>5s} | {'CAGR':>6s}{'maxDD':>6s} | {'H1':>6s}{'H2':>6s} | {'rnd p5':>7s}{'nbr min':>8s} robust")
    for p in PORT:
        ta = p["trades_all"]
        print(f"{p['label'][:80]:80s}{ta['n']:>6d}{ta['win'] or 0:>5.0f}{ta['net'] or 0:>+6.2f}{ta['per_month'] or 0:>5.0f} | "
              f"{p['full']['cagr']:>+6.1f}{p['full']['maxdd']:>6.1f} | {p['h1']['cagr']:>+6.1f}{p['h2']['cagr']:>+6.1f} | "
              f"{p['random_order']['p5']:>+7.1f}{p['neighbour_min']:>+8.1f} {'YES' if p['robust'] else 'no'}")
        p["keys_n"] = len(p["keys"]); del p["keys"]
    J["supplement"] = {"note": "market-level features excluded from grooming (R3 gate allowed); months_active_share = share of calendar months in which the filter still lets >= 5 trades through",
                       "grooming_stock_only": SUP, "portfolio_stock_only": PORT}
    Path("data/screen/swing_lab_pullbacks.json").write_text(json.dumps(J, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else (o.item() if hasattr(o, "item") else str(o))))
    print(f"\nstock-only pass written | {time.time()-t0:.0f}s")


# ------------------------------------------------------------------ main
# Run order: (1) plain -> matrix, grooming, stress, overlap; (2) --stock-only; (3) --quarterly.
if __name__ == "__main__" and "--stock-only" in sys.argv:
    stock_only_pass(); sys.exit(0)
if __name__ == "__main__" and "--quarterly" in sys.argv:
    quarterly_pass(); sys.exit(0)

if __name__ == "__main__":
    # sanity: five trades by hand
    tr = run(SIG["ema21_pullback"], ("atr", 3.0), ("trail", "ema21"), 60)
    print("\nSANITY: ema21_pullback, 3 ATR stop, close<21EMA trail, cap 60 - first 5 trades of 2025")
    for x in tr[tr["date"] >= "2025-01-01"].head(5).itertuples():
        t = x.row
        print(f"  {str(SYM[t]):12s} sig {x.date.date()} entry {x.entry:.2f} (open of {pd.Timestamp(DATES[t+1]).date()}) "
              f"stop {x.entry*(1-x.risk_pct/100):.2f} ({x.risk_pct:.1f}%) exit {x.exit:.2f} on {pd.Timestamp(DATES[t+x.bars]).date()} "
              f"why={x.why} bars={x.bars} net={x.net:+.2f} mae={x.mae:.1f} mfe={x.mfe:.1f}")
        if x.why == "trail":
            kk = t + x.bars - 1
            print(f"      trigger bar {pd.Timestamp(DATES[kk]).date()}: close {C[kk]:.2f} < ema21 {E21[kk]:.2f}; next open {O[kk+1]:.2f}")
        elif x.why == "stop":
            kk = t + x.bars
            print(f"      stop bar {pd.Timestamp(DATES[kk]).date()}: open {O[kk]:.2f} low {L[kk]:.2f} <= stop; fill {x.exit:.2f}")

    # A. the matrix
    jobs = [(n, s, e, c) for n in SETUPS for s in ATR_STOPS + [STRUCT_STOP[n]] for e in EXITS for c in CAPS]
    jobs += [("leader_dip_final", ("atr", 3.0), ("rev7",), 20)]
    print(f"\n{len(jobs)} cells ...", flush=True)
    with Pool(4) as pool:
        MATRIX = [x for x in pool.map(cell, jobs, chunksize=4) if x]
    print(f"{len(MATRIX)} cells with >= {MIN_N} trades per half | {time.time()-t0:.0f}s", flush=True)

    def show(rows, title):
        print(f"\n{title}")
        print(f"{'setup':22s}{'stop':>11s} {'exit':24s}{'n':>6s}{'win':>5s}{'net':>6s}{'aW':>6s}{'aL':>6s}{'pay':>5s}{'PF':>5s}{'stp%':>5s}{'bars':>5s}"
              f" | {'H1 n':>5s}{'win':>4s}{'net':>6s} | {'H2 n':>5s}{'win':>4s}{'net':>6s}")
        for x in rows:
            a, h1, h2 = x["all"], x["h1"], x["h2"]
            print(f"{TITLE[x['setup']][:22]:22s}{x['stop']:>11s} {x['exit']:24s}{a['n']:>6,d}{a['win']:>5.0f}{a['net']:>+6.2f}{a['avg_win']:>6.2f}"
                  f"{a['avg_loss']:>6.2f}{a['payoff']:>5.2f}{a['pf']:>5.2f}{a['stopped']:>5.0f}{a['bars']:>5.0f}"
                  f" | {h1['n']:>5d}{h1['win']:>4.0f}{h1['net']:>+6.2f} | {h2['n']:>5d}{h2['win']:>4.0f}{h2['net']:>+6.2f}"
                  f"  {'BOTH' if x['both'] else ''}")

    BEST = {}
    for n in SETUPS:
        rows = sorted([x for x in MATRIX if x["setup"] == n], key=lambda x: (-x["worse_net"], -x["worse_win"]))
        show(rows[:6], f"A. {TITLE[n]} - top 6 cells by worse-half mean net (of {len(rows)})")
        picks = rows[:3]
        by_win = sorted(rows, key=lambda x: (-x["worse_win"], -x["worse_net"]))
        if by_win and by_win[0] not in picks:
            picks = picks[:2] + [by_win[0]]
        BEST[n] = picks
    show([x for x in MATRIX if x["setup"] == "leader_dip_final"], "BAR TO BEAT: leader_dip final form")

    # B. grooming
    gjobs = [(n, tuple(x["stop_spec"]), tuple(x["exit_spec"]), x["cap"]) for n in SETUPS for x in BEST[n]]
    print(f"\ngrooming {len(gjobs)} cells ...", flush=True)
    with Pool(4) as pool:
        GROOM = pool.map(groom_job, gjobs, chunksize=1)
    print(f"grooming done | {time.time()-t0:.0f}s", flush=True)
    for gr in GROOM:
        c = gr["cell"]; b = gr["base"]
        sl, rl = label(tuple(c["stop_spec"]), tuple(c["exit_spec"]), c["cap"])
        print(f"\nB. {TITLE[c['setup']]} | {sl} | {rl}  base H1 win {b['H1']['win']*100:.0f}% net {b['H1']['net']:+.2f} | "
              f"H2 win {b['H2']['win']*100:.0f}% net {b['H2']['net']:+.2f}  tests={gr['n_tests']} singles pass={gr['n_single_passing']} combos pass={gr['n_combo_passing']}")
        for f in gr["leaderboard"][:8]:
            print(f"   {f['filter'][:60]:60s} H1 {f['H1']['n_kept']:>5d} {f['H1']['wr_kept']*100:>3.0f}% {f['H1']['mean_kept']:>+5.2f} t{f['H1']['t_cluster']:>+5.1f} "
                  f"{f['H1']['kept_per_month']:>4.0f}/mo | H2 {f['H2']['n_kept']:>5d} {f['H2']['wr_kept']*100:>3.0f}% {f['H2']['mean_kept']:>+5.2f} t{f['H2']['t_cluster']:>+5.1f} {f['H2']['kept_per_month']:>4.0f}/mo")

    # C. portfolio stress: baseline / groomed / groomed + gate, per setup
    specs = []
    for n in SETUPS:
        # baseline = the best cell; groomed = best leaderboard entry among that setup's groomed cells
        cands = [gr for gr in GROOM if gr["cell"]["setup"] == n and gr["leaderboard"]]
        best_cell = BEST[n][0]
        specs.append({"label": f"{n} | baseline", "setup": n, "stop_spec": best_cell["stop_spec"], "exit_spec": best_cell["exit_spec"],
                      "cap": best_cell["cap"], "filters": [], "gate": False, "kind": "baseline"})
        if cands:
            gr = max(cands, key=lambda gr: score(gr["leaderboard"][0]))
            f = gr["leaderboard"][0]
            parts = f["parts"] if "parts" in f else [f]
            filt = [{"feature": p["feature"], "op": p["op"], "thr": p["thr"]} for p in parts]
            c = gr["cell"]
            specs.append({"label": f"{n} | groomed: {f['filter']}", "setup": n, "stop_spec": c["stop_spec"], "exit_spec": c["exit_spec"],
                          "cap": c["cap"], "filters": filt, "gate": False, "kind": "groomed"})
            has_gate = any(p["feature"] == "R3" for p in parts)
            specs.append({"label": f"{n} | groomed + R3 gate", "setup": n, "stop_spec": c["stop_spec"], "exit_spec": c["exit_spec"],
                          "cap": c["cap"], "filters": filt, "gate": True, "kind": "groomed_gate", "gate_redundant": has_gate})
        else:
            specs.append({"label": f"{n} | baseline + R3 gate", "setup": n, "stop_spec": best_cell["stop_spec"], "exit_spec": best_cell["exit_spec"],
                          "cap": best_cell["cap"], "filters": [], "gate": True, "kind": "baseline_gate"})
    specs.append({"label": "leader_dip_final", "setup": "leader_dip_final", "stop_spec": ["atr", 3.0], "exit_spec": ["rev7"],
                  "cap": 20, "filters": [], "gate": False, "kind": "benchmark"})
    print(f"\nstressing {len(specs)} portfolios ...", flush=True)
    with Pool(4) as pool:
        PORT = pool.map(stress_job, specs, chunksize=1)
    print(f"stress done | {time.time()-t0:.0f}s", flush=True)
    print(f"\nC. PORTFOLIO, 10 slots, rs order, 100 random orderings, +/-20% neighbours")
    print(f"{'strategy':64s}{'n':>6s}{'win':>5s}{'net':>6s}{'/mo':>5s} | {'CAGR':>6s}{'maxDD':>6s} | {'H1':>6s}{'H2':>6s} | {'rnd p5':>7s}{'nbr min':>8s} robust")
    for p in PORT:
        ta = p["trades_all"]
        print(f"{p['label'][:64]:64s}{ta['n']:>6d}{ta['win'] or 0:>5.0f}{ta['net'] or 0:>+6.2f}{ta['per_month'] or 0:>5.0f} | "
              f"{p['full']['cagr']:>+6.1f}{p['full']['maxdd']:>6.1f} | {p['h1']['cagr']:>+6.1f}{p['h2']['cagr']:>+6.1f} | "
              f"{p['random_order']['p5']:>+7.1f}{p['neighbour_min']:>+8.1f} {'YES' if p['robust'] else 'no'}")

    # D. overlap with leader_dip final
    ld = [p for p in PORT if p["setup"] == "leader_dip_final"][0]
    LD_KEYS = set(map(tuple, ld["keys"]))
    ld_by_isin = {}
    for isin, d in LD_KEYS:
        ld_by_isin.setdefault(isin, []).append(CALPOS[pd.Timestamp(d)])
    OVERLAP = []
    for p in PORT:
        if p["setup"] == "leader_dip_final":
            continue
        keys = list(map(tuple, p["keys"]))
        exact = sum(k in LD_KEYS for k in keys)
        near = 0
        for isin, d in keys:
            pos = CALPOS[pd.Timestamp(d)]
            near += any(abs(pos - q) <= 5 for q in ld_by_isin.get(isin, []))
        OVERLAP.append({"label": p["label"], "kind": p["kind"], "n": len(keys), "exact_in_leader_dip": exact,
                        "within_5_sessions": near, "exact_pct": r(exact / max(len(keys), 1) * 100, 1),
                        "near_pct": r(near / max(len(keys), 1) * 100, 1)})
    print(f"\nD. OVERLAP WITH leader_dip FINAL ({len(LD_KEYS)} trades): share of each strategy's trades that leader_dip also took")
    for o in OVERLAP:
        print(f"  {o['label'][:64]:64s} n={o['n']:>5d}  same isin+date {o['exact_pct']:>5.1f}%  within 5 sessions {o['near_pct']:>5.1f}%")

    # E. verdict flags (mechanical; the narrative is in the report)
    VERDICT = {}
    for n in SETUPS:
        g_ = [p for p in PORT if p["setup"] == n and p["kind"] == "groomed"]
        b_ = [p for p in PORT if p["setup"] == n and p["kind"] == "baseline"][0]
        best = g_[0] if g_ else b_
        w = min(best["trades_H1"]["win"] or 0, best["trades_H2"]["win"] or 0)
        both = (best["trades_H1"]["net"] or -1) > 0 and (best["trades_H2"]["net"] or -1) > 0
        flag = "TRADE" if (w >= 65 and both and best["robust"]) else "MAYBE" if (w >= 60 and both) else "DROP"
        VERDICT[n] = {"flag": flag, "best_label": best["label"], "min_half_win": w, "both_halves_positive": bool(both),
                      "robust": bool(best["robust"]), "maxdd": best["full"]["maxdd"], "cagr": best["full"]["cagr"]}
        print(f"  {n:24s} {flag:6s} {best['label'][:60]:60s} min-half win {w:.0f}%  CAGR {best['full']['cagr']:+.1f}  maxDD {best['full']['maxdd']:.1f}  robust={best['robust']}")

    for p in PORT:
        del p["keys"]
    Path("data/screen/swing_lab_pullbacks.json").write_text(json.dumps({
        "window": {"start": str(START.date()), "split": str(SPLIT.date()), "end": str(END.date()), "cost": COST, "caps": CAPS,
                   "slots": SLOTS, "min_n_per_half": MIN_N, "min_kept_per_month": MIN_KEPT_PER_MONTH, "t_min": T_MIN},
        "signals": {n: int(s.sum()) for n, s in SIG.items()}, "candidates": CANDS,
        "matrix": MATRIX, "best_cells": {n: BEST[n] for n in SETUPS}, "grooming": GROOM, "portfolio": PORT,
        "overlap": OVERLAP, "verdict_flags": VERDICT}, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else (o.item() if hasattr(o, "item") else str(o))))
    print(f"\nwrote data/screen/swing_lab_pullbacks.json in {time.time()-t0:.0f}s")
