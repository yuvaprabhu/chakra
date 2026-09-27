"""Swing lab, breakout family: backtest, then groom for mostly-winning trades.

Twelve breakout-style rules from config/rules.yaml, each traded as a 2-8 week
swing with a hard stop on every trade:

  1. MATRIX   every (exit x stop) cell per setup, full sample and per half.
  2. GROOMING for the best 2-3 cells per setup, a filter search on stock-level
              features at the signal bar plus market state on the signal date.
              Thresholds are H1 quintile edges (chosen in-sample), and a filter
              passes only if it lifts win rate AND mean net in BOTH halves,
              keeps >= 20 trades a month, and its kept-vs-removed difference has
              a day-clustered t >= 2 in one half with the same sign in the other.
  3. PORTFOLIO screener.robustness.stress on baseline, groomed, groomed + R3 gate.
  4. VERDICT  TRADE / MAYBE / DROP per setup.

Conventions (inherited from scripts/stop_reward_study.py): entry next open;
stop checked first on every bar; a gap through the stop fills at the open; a
close-based trail exits at the next open; one open position per name; 0.30%
per round trip; 60-session time cap closed at that bar's close.

Run: python scripts/swing_lab_breakouts.py   (about 3-4 minutes on 4 cores)
Out: data/screen/swing_lab_breakouts.json
"""
import json, sys, time, itertools
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.rules import load_rules
from screener.backtest import add_panel_columns, rule_hits
from screener.robustness import stress
from screener.portfolio import simulate
from screener import quality
from screener.store import Store

COST, CAP, SLOTS = 0.30, 60, 10
START, SPLIT = pd.Timestamp("2024-06-01"), pd.Timestamp("2025-03-19")
MIN_N = 100                      # trades per half for a matrix cell to count
MIN_KEPT_PER_MONTH = 20          # full-sample average, after a filter
MIN_KEPT_HALF = 50               # and at least this many kept trades in each half
T_MIN = 2.0
N_CELLS_TO_GROOM = 3
SETUPS = ["donchian_breakout", "high_tight_breakout", "near_52w_high", "stage2_base_breakout",
          "three_weeks_tight", "inside_bar_coil", "vcp_squeeze", "pocket_pivot", "episodic_pivot",
          "monthly_r1_breakout", "base_breakout_r1", "held_breakout"]
# One structural stop per setup: the level the setup itself says must hold.
STRUCT_STOP = {"donchian_breakout": "hi_20_prior",      # the channel top just broken
               "high_tight_breakout": "hi_250_prior",   # the 52-week box top
               "near_52w_high": "sma_50",               # no breakout level; the trend average
               "stage2_base_breakout": "hi_60_prior",   # the quarter base top
               "three_weeks_tight": "low_15",           # low of the three tight weeks
               "inside_bar_coil": "adj_low",            # the inside bar's low
               "vcp_squeeze": "low_10",                 # low of the final contraction
               "pocket_pivot": "sma_50",                # under the 50-day (tighter than base low)
               "episodic_pivot": "adj_low",             # the gap day's low
               "monthly_r1_breakout": "m_r1",           # back under R1
               "base_breakout_r1": "low_60",            # the base's low
               "held_breakout": "brk_level"}            # the level that has held
EXITS = [("trail", "ema21"), ("trail", "ema50"), ("trail", "low10"), ("trail", "chand3"),
         ("target", 2.0), ("target", 3.0), ("none",)]
ATR_STOPS = [("atr", 2.5), ("atr", 3.0), ("atr", 4.0)]
STOCK_FEATS = ["rs_rank", "dist_ema_21", "dist_sma_50", "dist_sma_200", "rsi_14", "atr_pct", "vol_ratio",
               "ret_20d", "ret_60d", "pct_from_52w_high", "turnover_median_20d"]
# Emit columns that are price levels or raw counts, not features to threshold on.
NOT_A_FEATURE = {"adj_close", "volume", "high_52w", "m_r1", "m_r2", "brk_level", "hi_20_prior", "hi_60_prior",
                 "hi_250_prior", "down_vol_max_10", "contract_a", "contract_b", "contract_c"}
MARKET_NUM = ["vix", "vix_pctile_1y", "breadth_above_50", "breadth_thrust_10", "dispersion", "nifty_range_20d"]
MARKET_BIN = ["nifty_above_20dma", "nifty_above_50dma", "nifty_above_200dma", "R3"]

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
feat["low_10"] = g["adj_low"].transform(lambda s: s.rolling(10).min())
feat["low_15"] = g["adj_low"].transform(lambda s: s.rolling(15).min())
feat["low_60"] = g["adj_low"].transform(lambda s: s.rolling(60).min())
feat["dist_sma_200"] = (feat["adj_close"] / feat["sma_200"] - 1) * 100
feat["ret1"] = g["adj_close"].pct_change()
feat["liquid"] = ((feat["turnover_median_20d"] >= 1e7) & (feat["bars_available"] >= 250)
                  & (~feat["contaminated"]))

CAL = pd.DatetimeIndex(sorted(feat["date"].unique()))
CALPOS = {d: i for i, d in enumerate(CAL)}
END = CAL[-1]
mkt = feat.loc[feat["liquid"]].groupby("date")["ret1"].mean().reindex(CAL).fillna(0)
BENCH = (1 + mkt).cumprod()

O = feat["adj_open"].to_numpy(float); H = feat["adj_high"].to_numpy(float)
L = feat["adj_low"].to_numpy(float);  C = feat["adj_close"].to_numpy(float)
ATRP = feat["atr_pct"].to_numpy(float) / 100.0
E21 = feat["ema_21"].to_numpy(float); E50 = feat["ema_50"].to_numpy(float)
LO10 = feat["low10_prior"].to_numpy(float)
RS = feat["rs_rank"].to_numpy(float)
ISIN = feat["isin"].to_numpy(); DATES = feat["date"].to_numpy()
SYMBOL = lat.drop_duplicates("isin").set_index("isin")["symbol"].to_dict()
LAST = pd.Series(np.arange(len(feat))).groupby(pd.factorize(feat["isin"])[0]).transform("max").to_numpy()
IN_WIN = ((feat["date"] >= START) & (feat["date"] <= END)).to_numpy()
print(f"panel {len(feat):,} rows | {CAL[0].date()} to {END.date()} | signals from {START.date()}, "
      f"halves split {SPLIT.date()} | {time.time()-t0:.0f}s", flush=True)

rules = {r.name: r for r in load_rules("config/rules.yaml")}
TITLE = {n: rules[n].title for n in SETUPS}
SIG = {}
for n in SETUPS:
    SIG[n] = rule_hits(feat, rules[n]).to_numpy() & IN_WIN
    print(f"  {TITLE[n]:28s} signals {SIG[n].sum():>7,d}  first {feat.loc[SIG[n], 'date'].min().date()}", flush=True)

# Market state and the R3 gate, mapped onto every panel row by date.
MS = pd.read_parquet("data/screen/market_state.parquet"); MS["date"] = pd.to_datetime(MS["date"]); MS = MS.set_index("date")
GATE = pd.read_parquet("data/screen/gate_r3.parquet"); GATE["date"] = pd.to_datetime(GATE["date"]); GATE = GATE.set_index("date")["R3"]
EXTRA = sorted({c for n in SETUPS for c in rules[n].emit if c not in NOT_A_FEATURE} | set(STOCK_FEATS))
KEEP = sorted(set(EXTRA) | set(STRUCT_STOP.values()) | {"cap_band", "adj_low", "sma_50"})
FE = feat[KEEP].copy()
FE["R3"] = feat["date"].map(GATE).fillna(0).astype(int).to_numpy()
for c in MARKET_NUM + MARKET_BIN[:-1]:
    FE[c] = feat["date"].map(MS[c]).to_numpy()
LVL = {c: FE[c].to_numpy(float) for c in set(STRUCT_STOP.values())}
COL = {c: FE[c].to_numpy() for c in FE.columns}
del feat, g, b, lat        # the children only need the arrays above


def run(sig, stop, reward, mgmt=None):
    """One trade per signal, with a hard initial stop. See module docstring.

    stop:   ('atr', m) m x ATR(14)% at the signal bar | ('lvl', col) the level in column col at the signal bar
    reward: ('target', rr) | ('trail', 'ema21'|'ema50'|'low10'|'chand3') | ('none',)
    """
    rows = []; busy = {}
    for t in np.flatnonzero(sig):
        if t + 1 > LAST[t] or busy.get(ISIN[t], -1) >= t or not np.isfinite(ATRP[t]):
            continue
        e = O[t + 1]
        if not np.isfinite(e) or e <= 0:
            continue
        if stop[0] == "atr":
            risk = stop[1] * ATRP[t] * e
        else:
            lvl = LVL[stop[1]][t]
            if not np.isfinite(lvl):
                continue
            risk = e - lvl
        rpct = risk / e * 100.0
        if not (0.2 < rpct < 20):
            continue
        sl = e - risk; sl0 = sl
        tgt = e + reward[1] * risk if reward[0] == "target" else np.inf
        atr_abs = ATRP[t] * e
        end = min(LAST[t], t + CAP)
        k = t + 1; peak = e
        out = None; why = "time"
        while k <= end:
            if L[k] <= sl:
                out = min(O[k], sl); why = "stop" if sl <= sl0 else "tstop"; break
            if reward[0] == "target" and H[k] >= tgt:
                out = max(O[k], tgt); why = "target"; break
            peak = max(peak, H[k])
            if reward[0] == "trail":
                lvl = (E21[k] if reward[1] == "ema21" else E50[k] if reward[1] == "ema50"
                       else LO10[k] if reward[1] == "low10" else peak - 3.0 * atr_abs)
                if np.isfinite(lvl) and C[k] < lvl and k + 1 <= LAST[t]:
                    out = O[k + 1]; why = "trail"; k += 1; break
            k += 1
        if out is None:
            k = min(k, end); out = C[k]
        if not np.isfinite(out):
            continue
        busy[ISIN[t]] = k
        gross = out / e - 1
        ei, xi = CALPOS[pd.Timestamp(DATES[t + 1])], CALPOS[pd.Timestamp(DATES[k])]
        mae = np.nanmin(L[t + 1:k + 1]) / e * 100 - 100
        mfe = np.nanmax(H[t + 1:k + 1]) / e * 100 - 100
        rows.append((pd.Timestamp(DATES[t]), ISIN[t], t, ei, xi, e, out, gross * 100 - COST, k - t,
                     rpct, RS[t], why, mae, mfe))
    return pd.DataFrame(rows, columns=["date", "isin", "row", "entry_i", "exit_i", "entry", "exit", "net", "bars",
                                       "risk_pct", "rs", "why", "mae", "mfe"])


def stats(t, min_n=MIN_N):
    if len(t) < min_n:
        return None
    n = t["net"]; w = n[n > 0]; l = n[n <= 0]
    if not len(w) or not len(l):
        return None
    why = t["why"].value_counts(normalize=True) * 100
    months = max((t["date"].max() - t["date"].min()).days / 30.44, 1)
    return {"n": int(len(t)), "win": float((n > 0).mean() * 100), "net": float(n.mean()),
            "avg_win": float(w.mean()), "avg_loss": float(l.mean()),
            "payoff": float(w.mean() / -l.mean()), "pf": float(w.sum() / -l.sum()),
            "exp_r": float((n / t["risk_pct"]).mean()),
            "stopped": float(why.get("stop", 0)), "target": float(why.get("target", 0)),
            "trail": float(why.get("trail", 0)), "time": float(why.get("time", 0)),
            "bars": float(t["bars"].mean()), "risk": float(t["risk_pct"].mean()),
            "mae": float(t["mae"].mean()), "mfe": float(t["mfe"].mean()),
            "per_month": float(len(t) / months), "days": int(t["date"].nunique())}


def label(stop, reward):
    s = f"{stop[1]:g} ATR" if stop[0] == "atr" else f"under {stop[1]}"
    r = (f"1:{reward[1]:g}" if reward[0] == "target" else
         {"ema21": "close<21EMA", "ema50": "close<50EMA", "low10": "close<10d low",
          "chand3": "chandelier 3ATR"}[reward[1]] if reward[0] == "trail" else "stop only")
    return s, r


def cell(job):
    setup, stop, reward = job
    tr = run(SIG[setup], stop, reward)
    if not len(tr):
        return None
    h1, h2 = tr[tr["date"] < SPLIT], tr[tr["date"] >= SPLIT]
    a, s1, s2 = stats(tr), stats(h1), stats(h2)
    sl, rl = label(stop, reward)
    return {"setup": setup, "stop": sl, "exit": rl, "stop_spec": list(stop), "exit_spec": list(reward),
            "all": a, "h1": s1, "h2": s2, "counts": bool(a and s1 and s2),
            "both": bool(s1 and s2 and s1["net"] > 0 and s2["net"] > 0),
            "min_win": float(min(s1["win"], s2["win"])) if (s1 and s2) else None,
            "min_net": float(min(s1["net"], s2["net"])) if (s1 and s2) else None}


# ------------------------------------------------------------ sanity check
print("\nSANITY: five held_breakout trades, 2.5 ATR stop, close<21EMA trail", flush=True)
ex = run(SIG["held_breakout"], ("atr", 2.5), ("trail", "ema21"))
print(f"  ({len(ex)} trades)")
for r in ex.iloc[np.linspace(0, len(ex) - 1, 5).astype(int)].itertuples():
    t = r.row
    print(f"  {str(SYMBOL.get(ISIN[t], ISIN[t])):12s} signal {pd.Timestamp(DATES[t]).date()} close {C[t]:.2f} atr% {ATRP[t]*100:.2f} | "
          f"entry {pd.Timestamp(DATES[t+1]).date()} @ {r.entry:.2f} stop {r.entry*(1-r.risk_pct/100):.2f} | "
          f"exit {pd.Timestamp(CAL[r.exit_i]).date()} @ {r.exit:.2f} ({r.why}, {r.bars} bars) net {r.net:+.2f}% "
          f"mae {r.mae:+.1f} mfe {r.mfe:+.1f} [{ISIN[t]}]")

# ------------------------------------------------------------ 1. the matrix
jobs = [(n, s, r) for n in SETUPS for s in ATR_STOPS + [("lvl", STRUCT_STOP[n])] for r in EXITS]
print(f"\n{len(jobs)} cells ...", flush=True)
with Pool(4) as pool:
    MATRIX = [x for x in pool.map(cell, jobs, chunksize=4) if x]
print(f"matrix done | {time.time()-t0:.0f}s", flush=True)


def show_cells(rows, title):
    print(f"\n{title}")
    print(f"{'setup':22s}{'stop':>16s} {'exit':16s}{'n':>6s}{'win':>5s}{'net':>6s}{'aW':>6s}{'aL':>6s}{'pay':>5s}"
          f"{'PF':>5s}{'stp%':>5s}{'bars':>5s} | {'H1 n':>5s}{'win':>4s}{'net':>6s} | {'H2 n':>5s}{'win':>4s}{'net':>6s}  both")
    print("-" * 140)
    for x in rows:
        a, h1, h2 = x["all"], x["h1"], x["h2"]
        f1 = f"{h1['n']:>5d}{h1['win']:>4.0f}{h1['net']:>+6.2f}" if h1 else f"{'-':>15s}"
        f2 = f"{h2['n']:>5d}{h2['win']:>4.0f}{h2['net']:>+6.2f}" if h2 else f"{'-':>15s}"
        if a:
            print(f"{x['setup'][:22]:22s}{x['stop']:>16s} {x['exit']:16s}{a['n']:>6d}{a['win']:>5.0f}{a['net']:>+6.2f}{a['avg_win']:>6.2f}"
                  f"{a['avg_loss']:>6.2f}{a['payoff']:>5.2f}{a['pf']:>5.2f}{a['stopped']:>5.0f}{a['bars']:>5.0f} | {f1} | {f2}  {'YES' if x['both'] else ''}")


for n in SETUPS:
    show_cells(sorted([x for x in MATRIX if x["setup"] == n], key=lambda x: (x["stop_spec"][0] != "atr", str(x["stop_spec"][1]), x["exit"])),
               f"MATRIX: {TITLE[n]}")


def rank_key(x):
    return (x["counts"], x["both"], x["min_win"] or -1, x["min_net"] or -99)


BEST = {}
for n in SETUPS:
    cs = sorted([x for x in MATRIX if x["setup"] == n and x["h1"] and x["h2"]], key=rank_key, reverse=True)
    BEST[n] = cs[:N_CELLS_TO_GROOM]

# ------------------------------------------------------------ 2. grooming
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


def attach(tr):
    """Stock features at the signal bar and market state on the signal date."""
    tr = tr.copy()
    for c in FE.columns:
        tr[c] = COL[c][tr["row"].to_numpy()]
    tr["half"] = np.where(tr["date"] < SPLIT, "H1", "H2")
    tr["win"] = tr["net"] > 0
    return tr


def r3(x, k=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


def eval_filter(sub, mask, base, months_full):
    res = {}
    m_all = mask.fillna(False).to_numpy(bool)
    res["kept_per_month"] = r3(m_all.sum() / months_full, 1)
    for h in ("H1", "H2"):
        s = sub[sub["half"] == h]; m = mask.loc[s.index].fillna(False).to_numpy(bool)
        kept, rem = s[m], s[~m]
        d = {"n_all": int(len(s)), "n_kept": int(len(kept)), "kept_pct": r3(len(kept) / max(len(s), 1)),
             "wr_kept": r3(kept["win"].mean() * 100, 1) if len(kept) else None,
             "mean_kept": r3(kept["net"].mean()) if len(kept) else None,
             "pf_kept": r3(kept.loc[kept["net"] > 0, "net"].sum() / max(-kept.loc[kept["net"] <= 0, "net"].sum(), 1e-9), 2) if len(kept) else None,
             "wr_removed": r3(rem["win"].mean() * 100, 1) if len(rem) else None,
             "mean_removed": r3(rem["net"].mean()) if len(rem) else None,
             "t_cluster": r3(cluster_t(s["net"], m.astype(float), s["date"]), 2) if 0 < m.sum() < len(s) else None,
             "n_days_kept": int(kept["date"].nunique())}
        res[h] = d
    return res


def filter_passes(res, base):
    ts = []
    for h in ("H1", "H2"):
        d = res[h]
        if d["wr_kept"] is None or d["mean_kept"] is None or d["t_cluster"] is None:
            return False
        if not (d["wr_kept"] > base[h]["win"] and d["mean_kept"] > base[h]["net"]):
            return False
        if d["n_kept"] < MIN_KEPT_HALF:
            return False
        ts.append(d["t_cluster"])
    if res["kept_per_month"] < MIN_KEPT_PER_MONTH:
        return False
    return bool(max(ts) >= T_MIN and min(ts) > 0)


def score(res):
    return (min(res["H1"]["mean_kept"], res["H2"]["mean_kept"]) > 0,
            min(res["H1"]["wr_kept"], res["H2"]["wr_kept"]), min(res["H1"]["mean_kept"], res["H2"]["mean_kept"]),
            min(res["H1"]["kept_pct"], res["H2"]["kept_pct"]))


def mask_of(sub, f):
    if f["op"] == ">=":
        return sub[f["feature"]] >= f["thr"]
    if f["op"] == "<=":
        return sub[f["feature"]] <= f["thr"]
    return sub[f["feature"]] == f["thr"]


def groom(setup, x):
    """Filter search on one matrix cell. Returns the leaderboard and the base rates."""
    tr = attach(run(SIG[setup], tuple(x["stop_spec"]), tuple(x["exit_spec"])))
    base = {h: stats(tr[tr["half"] == h], 1) for h in ("H1", "H2")}
    months_full = max((tr["date"].max() - tr["date"].min()).days / 30.44, 1)
    h1 = tr[tr["half"] == "H1"]
    feats = STOCK_FEATS + [c for c in rules[setup].emit if c not in NOT_A_FEATURE and c not in STOCK_FEATS] + MARKET_NUM
    singles, n_tests = [], 0
    for f in feats:
        v = h1[f].astype(float)
        if v.notna().sum() < 100:
            continue
        edges = np.unique(np.nanpercentile(v, [20, 40, 60, 80]))
        for p, thr in zip((20, 40, 60, 80), edges):
            for op in (">=", "<="):
                spec = {"feature": f, "op": op, "thr": float(thr), "pctile_H1": p, "filter": f"{f} {op} {thr:.4g}"}
                n_tests += 1
                res = eval_filter(tr, mask_of(tr, spec), base, months_full)
                spec.update({"passes": filter_passes(res, base), "res": res})
                singles.append(spec)
    for f in MARKET_BIN:
        for val in (0, 1):
            spec = {"feature": f, "op": "==", "thr": val, "filter": f"{f} == {val}"}
            n_tests += 1
            res = eval_filter(tr, mask_of(tr, spec), base, months_full)
            spec.update({"passes": filter_passes(res, base), "res": res}); singles.append(spec)
    for val in ("Mid", "Large", "Mega"):
        spec = {"feature": "cap_band", "op": "==", "thr": val, "filter": f"cap_band == {val}"}
        n_tests += 1
        res = eval_filter(tr, mask_of(tr, spec), base, months_full)
        spec.update({"passes": filter_passes(res, base), "res": res}); singles.append(spec)
    passing = sorted([s for s in singles if s["passes"]], key=lambda s: score(s["res"]), reverse=True)
    best_per_feat = {}
    for s in passing:
        best_per_feat.setdefault(s["feature"], s)
    top = list(best_per_feat.values())[:8]
    combos = []
    for a, b2 in itertools.combinations(top, 2):
        n_tests += 1
        res = eval_filter(tr, mask_of(tr, a) & mask_of(tr, b2), base, months_full)
        combos.append({"filter": f"{a['filter']} AND {b2['filter']}", "parts": [a, b2], "passes": filter_passes(res, base), "res": res})
    board = sorted([s for s in singles + combos if s["passes"]], key=lambda s: score(s["res"]), reverse=True)
    return {"setup": setup, "stop": x["stop"], "exit": x["exit"], "stop_spec": x["stop_spec"], "exit_spec": x["exit_spec"],
            "base": base, "n_tests": n_tests, "n_single_passing": len(passing), "n_combo_passing": sum(c["passes"] for c in combos),
            "leaderboard": board[:25], "n_trades": int(len(tr))}


def groom_job(job):
    return groom(*job)


gjobs = [(n, x) for n in SETUPS for x in BEST[n]]
print(f"\ngrooming {len(gjobs)} cells ...", flush=True)
with Pool(4) as pool:
    GROOM = pool.map(groom_job, gjobs, chunksize=1)
print(f"grooming done | {time.time()-t0:.0f}s", flush=True)


def parts_of(f):
    return f["parts"] if "parts" in f else [f]


def fmt_res(d):
    return (f"{d['H1']['n_kept']:>5d}{d['H1']['wr_kept']:>5.0f}{d['H1']['mean_kept']:>+6.2f}{d['H1']['t_cluster']:>5.1f} | "
            f"{d['H2']['n_kept']:>5d}{d['H2']['wr_kept']:>5.0f}{d['H2']['mean_kept']:>+6.2f}{d['H2']['t_cluster']:>5.1f} | {d['kept_per_month']:>5.1f}")


for gr in GROOM:
    b1, b2 = gr["base"]["H1"], gr["base"]["H2"]
    print(f"\nGROOM: {TITLE[gr['setup']]} | {gr['stop']} | {gr['exit']}  ({gr['n_tests']} filters tested, "
          f"{gr['n_single_passing']} singles + {gr['n_combo_passing']} combos pass)")
    print(f"  {'baseline':60s}{b1['n']:>5d}{b1['win']:>5.0f}{b1['net']:>+6.2f}{'':>5s} | {b2['n']:>5d}{b2['win']:>5.0f}{b2['net']:>+6.2f}{'':>5s} | "
          f"{(b1['n']+b2['n'])/max((gr['base']['H2'] and 1) and 1, 1):>5.0f}")
    for f in gr["leaderboard"][:10]:
        print(f"  {f['filter'][:60]:60s}{fmt_res(f['res'])}")

# ------------------------------------------------------------ 3. portfolio stress
def make_trades_factory(setup, stop_spec, exit_spec, parts):
    """params: atr_mult / rr when applicable, one f{i}_thr per numeric filter part, gate flag."""
    def make_trades(params):
        sig = SIG[setup].copy()
        if params.get("gate"):
            sig &= COL["R3"] == 1
        for i, p in enumerate(parts):
            v = COL[p["feature"]]
            thr = params.get(f"f{i}_thr", p["thr"])
            if p["op"] == ">=":
                m = v.astype(float) >= thr
            elif p["op"] == "<=":
                m = v.astype(float) <= thr
            else:
                m = v == p["thr"]
            sig &= np.nan_to_num(m, nan=False) if m.dtype != bool else m
        stop = ("atr", params["atr_mult"]) if stop_spec[0] == "atr" else tuple(stop_spec)
        rew = ("target", params["rr"]) if exit_spec[0] == "target" else tuple(exit_spec)
        return run(sig, stop, rew)
    return make_trades


def params_for(stop_spec, exit_spec, parts, gate):
    p = {}
    if stop_spec[0] == "atr":
        p["atr_mult"] = float(stop_spec[1])
    if exit_spec[0] == "target":
        p["rr"] = float(exit_spec[1])
    for i, part in enumerate(parts):
        if part["op"] in (">=", "<=") and part["thr"] != 0:
            p[f"f{i}_thr"] = float(part["thr"])
    p["gate"] = bool(gate)
    return p


def stress_row(setup, stop_spec, exit_spec, parts, gate, tag):
    mk = make_trades_factory(setup, stop_spec, exit_spec, parts)
    s = stress(mk, params_for(stop_spec, exit_spec, parts, gate), sessions=CAL, split=SPLIT, slots=SLOTS,
               order_col="rs", n_random=100)
    tr = mk(s["params"])
    st_ = stats(tr, 1); s1 = stats(tr[tr["date"] < SPLIT], 1); s2 = stats(tr[tr["date"] >= SPLIT], 1)
    # Fixed calendar quarters: a regime filter that only fires in a few windows shows up here.
    q = tr["date"].dt.to_period("Q").astype(str)
    quarterly = [{"q": k, "n": int(len(g_)), "win": float((g_["net"] > 0).mean() * 100), "net": float(g_["net"].mean()),
                  "days": int(g_["date"].nunique())} for k, g_ in tr.groupby(q)]
    big = [x for x in quarterly if x["n"] >= 10]
    q_pos = sum(x["net"] > 0 for x in big) / max(len(big), 1)
    # PASS 2: the 10-slot account's return in each calendar quarter (rs order, same as stress 'full').
    eq = simulate(tr.sort_values(["entry_i", "rs"], ascending=[True, False]), slots=SLOTS, sessions=CAL)["equity"]
    qe = eq.groupby(eq.index.to_period("Q")).last()
    prev = qe.shift(1); prev.iloc[0] = 100.0
    port_q = {str(k): float(v) for k, v in ((qe / prev - 1) * 100).items()}
    return {"setup": setup, "version": tag, "quarterly": quarterly, "q_pos_share": float(q_pos), "port_quarterly": port_q,
            "q_active": int(len(big)), "q_total": int(tr["date"].dt.to_period("Q").nunique()),
            "stop": label(tuple(stop_spec), tuple(exit_spec))[0],
            "exit": label(tuple(stop_spec), tuple(exit_spec))[1],
            "filter": " AND ".join(p["filter"] for p in parts) if parts else "", "gate": gate,
            "n": int(len(tr)), "win": st_["win"] if st_ else None, "net": st_["net"] if st_ else None,
            "win_h1": s1["win"] if s1 else None, "win_h2": s2["win"] if s2 else None,
            "net_h1": s1["net"] if s1 else None, "net_h2": s2["net"] if s2 else None,
            "stress": {k: v for k, v in s.items()}}


STRESS = []
CHOSEN = {}
for n in SETUPS:
    grs = [g_ for g_ in GROOM if g_["setup"] == n]
    if not grs:
        continue
    # the groomed version: the best leaderboard entry across the groomed cells
    cands = [(g_, f) for g_ in grs for f in g_["leaderboard"][:1]]
    if cands:
        g_, f = max(cands, key=lambda c: score(c[1]["res"]))
        parts = parts_of(f)
    else:
        g_, f, parts = grs[0], None, []
    CHOSEN[n] = {"cell": g_, "filter": f}
    base_cell = BEST[n][0]
    versions = [("baseline", base_cell, [], False), ("baseline+gate", base_cell, [], True)]
    if parts:
        versions += [("groomed", g_, parts, False), ("groomed+gate", g_, parts, True)]
    for tag, cellx, prt, gate in versions:
        STRESS.append(stress_row(n, cellx["stop_spec"], cellx["exit_spec"], prt, gate, tag))
    print(f"  stress {TITLE[n]} done | {time.time()-t0:.0f}s", flush=True)

print("\nPORTFOLIO STRESS (10 slots, rs order; random = 100 shuffles; neighbours = +/-20% on numeric params)")
print(f"{'setup':22s}{'version':14s}{'n':>6s}{'win':>5s}{'net':>6s} | {'CAGR':>6s}{'maxDD':>6s} | {'H1cagr':>7s}{'H1dd':>6s} | "
      f"{'H2cagr':>7s}{'H2dd':>6s} | {'rnd p5':>7s}{'nb min':>7s} robust  filter / exit")
print("-" * 160)
for s in STRESS:
    x = s["stress"]
    print(f"{s['setup'][:22]:22s}{s['version']:14s}{s['n']:>6d}{s['win'] or 0:>5.0f}{s['net'] or 0:>+6.2f} | {x['full']['cagr']:>+6.1f}{x['full']['maxdd']:>6.1f} | "
          f"{x['h1']['cagr']:>+7.1f}{x['h1']['maxdd']:>6.1f} | {x['h2']['cagr']:>+7.1f}{x['h2']['maxdd']:>6.1f} | "
          f"{x['random_order']['p5']:>+7.1f}{x['neighbour_min']:>+7.1f}  {'YES' if x['robust'] else 'no ':4s}  "
          f"{s['stop']} / {s['exit']} / {s['filter'][:50]}")

# ------------------------------------------------------------ 4. verdicts
VERDICT = {}
for n in SETUPS:
    rows = {s["version"]: s for s in STRESS if s["setup"] == n}
    gsel = rows.get("groomed") or rows.get("baseline")
    if not gsel:
        VERDICT[n] = {"verdict": "DROP", "why": "not enough trades for the matrix"}; continue
    x = gsel["stress"]
    mw = min(gsel["win_h1"] or 0, gsel["win_h2"] or 0)
    both = (gsel["net_h1"] or -1) > 0 and (gsel["net_h2"] or -1) > 0
    if mw >= 65 and both and x["robust"] and x["full"]["maxdd"] > -15:
        v = "TRADE"
    elif (mw >= 50 and both and x["robust"] and x["full"]["maxdd"] > -12 and gsel["q_pos_share"] >= 2 / 3):
        v = "MAYBE"
    else:
        v = "DROP"
    VERDICT[n] = {"verdict": v, "version": gsel["version"], "min_half_win": mw, "both_positive": bool(both),
                  "robust": x["robust"], "maxdd": x["full"]["maxdd"], "cagr": x["full"]["cagr"],
                  "n_taken_10slot": x["full"]["n_taken"], "q_pos_share": gsel["q_pos_share"], "q_active": gsel["q_active"],
                  "rule": {"entry": rules[n].expr, "filter": gsel["filter"], "stop": gsel["stop"], "exit": gsel["exit"], "cap": CAP}}
    print(f"  {v:6s} {TITLE[n]:28s} min-half win {mw:.0f}%  both {both}  robust {x['robust']}  maxDD {x['full']['maxdd']:.1f}  "
          f"CAGR {x['full']['cagr']:+.1f}  taken {x['full']['n_taken']}  quarters+ {gsel['q_pos_share']*100:.0f}% of {gsel['q_active']} | "
          f"{gsel['stop']} / {gsel['exit']} / {gsel['filter']}")
    print("         " + "  ".join(f"{qq['q'][2:]}:{qq['n']}/{qq['win']:.0f}%/{qq['net']:+.1f}" for qq in gsel["quarterly"]))


# ------------------------------------------------------------ PASS 2: quarter consistency
# Fixed calendar quarters of the signal date, 2024Q2 (June only) .. 2026Q3. For every setup's
# baseline / groomed / groomed+gate: trades, win %, mean net, the 10-slot account's return, and the
# market that quarter, so a lost quarter can be named. The first-pass keys above are untouched.
QUARTERS = [str(q) for q in pd.period_range(START, END, freq="Q")]
bq = BENCH.loc[START:END]
bqe = bq.groupby(bq.index.to_period("Q")).last(); bprev = bqe.shift(1); bprev.iloc[0] = bq.iloc[0]
MSQ = MS.loc[START:END]
MSQ = MSQ.assign(R3=GATE.reindex(MSQ.index).fillna(0))
mq = MSQ.groupby(MSQ.index.to_period("Q")).agg(breadth_above_50=("breadth_above_50", "mean"), vix=("vix", "mean"),
                                                 nifty_vs_200dma=("nifty_vs_200dma", "mean"), nifty_above_200=("nifty_above_200dma", "mean"),
                                                 r3_share=("R3", "mean"), nifty_range_20d=("nifty_range_20d", "mean"))
MARKET_Q = {}
for q in QUARTERS:
    pq = pd.Period(q)
    MARKET_Q[q] = {"bench_ret": float((bqe[pq] / bprev[pq] - 1) * 100) if pq in bqe.index else None,
                   **({k: float(v) for k, v in mq.loc[pq].items()} if pq in mq.index else {})}


def describe_q(q):
    m = MARKET_Q[q]
    if m.get("bench_ret") is None:
        return q
    return (f"universe {m['bench_ret']:+.0f}%, breadth>50dma {m['breadth_above_50']:.0f}%, VIX {m['vix']:.0f}, "
            f"Nifty {'above' if m['nifty_above_200'] >= 0.5 else 'below'} 200dma, R3 on {m['r3_share']*100:.0f}%")


QUARTERLY = {"quarters": QUARTERS, "market": MARKET_Q, "setups": {}, "verdict": {},
             "rule": {"TRADE": "win >= 65% overall, both halves positive (trade-level), >= 6 quarters with a positive 10-slot "
                               "return, no quarter worse than -8% on the account",
                      "MAYBE": "win >= 65% and both halves positive, but fails the quarter test", "DROP": "otherwise"}}
print("\nPASS 2: QUARTERLY (signal-date quarter; n / win% / mean net% / 10-slot account return that quarter)")
for q in QUARTERS:
    print(f"  {q}: {describe_q(q)}")
for n in SETUPS:
    rows = [s_ for s_ in STRESS if s_["setup"] == n]
    if not rows:
        QUARTERLY["verdict"][n] = {"verdict": "DROP", "why": "fewer than 100 trades in H1 in every cell; nothing to groom"}
        continue
    QUARTERLY["setups"][n] = {}
    best = None
    for s_ in rows:
        tq = {x["q"]: x for x in s_["quarterly"]}
        table = []
        for q in QUARTERS:
            t_ = tq.get(q); pr = s_["port_quarterly"].get(q)
            table.append({"q": q, "n": t_["n"] if t_ else 0, "win": t_["win"] if t_ else None, "net": t_["net"] if t_ else None,
                          "port_ret": pr})
        have = [r_ for r_ in table if r_["port_ret"] is not None]
        pos = sum(r_["port_ret"] > 0 for r_ in have)
        worst = min(have, key=lambda r_: r_["port_ret"]) if have else None
        lost = [r_ for r_ in have if r_["port_ret"] < 0]
        both = (s_["net_h1"] or -1) > 0 and (s_["net_h2"] or -1) > 0
        win_all = s_["win"] or 0
        q_ok = pos >= 6 and worst is not None and worst["port_ret"] >= -8
        v = "TRADE" if (win_all >= 65 and both and q_ok) else "MAYBE" if (win_all >= 65 and both) else "DROP"
        rec = {"table": table, "n_quarters": len(have), "pos_quarters": pos, "pos_share": pos / max(len(have), 1),
               "worst_q": worst["q"] if worst else None, "worst_port_ret": worst["port_ret"] if worst else None,
               "lost_quarters": [{"q": r_["q"], "port_ret": r_["port_ret"], "n": r_["n"], "win": r_["win"], "market": describe_q(r_["q"])} for r_ in lost],
               "win_all": win_all, "both_halves_positive": bool(both), "quarter_test": bool(q_ok), "verdict": v,
               "stop": s_["stop"], "exit": s_["exit"], "filter": s_["filter"], "gate": s_["gate"]}
        QUARTERLY["setups"][n][s_["version"]] = rec
        rank = ({"TRADE": 2, "MAYBE": 1, "DROP": 0}[v], q_ok, pos, worst["port_ret"] if worst else -99)
        if best is None or rank > best[0]:
            best = (rank, s_["version"], rec)
        print(f"\n  {TITLE[n]} | {s_['version']} | {s_['stop']} / {s_['exit']} / {s_['filter'] or '-'}{' / R3 gate' if s_['gate'] else ''}")
        print("   " + " ".join(f"{q[2:]:>14s}" for q in QUARTERS))
        print("   " + " ".join(f"{r_['n']:>5d}/{(r_['win'] if r_['win'] is not None else 0):>3.0f}%/{(r_['net'] if r_['net'] is not None else 0):>+5.1f}" for r_ in table))
        print("   " + " ".join(f"{'acct ' + format(r_['port_ret'], '+6.1f') if r_['port_ret'] is not None else '-':>14s}" for r_ in table)
              + f"   | +{pos}/{len(have)} quarters, worst {worst['q'] if worst else '-'} {worst['port_ret'] if worst else 0:+.1f}%  -> {v}")
    QUARTERLY["verdict"][n] = {"verdict": best[2]["verdict"], "version": best[1], "win_all": best[2]["win_all"],
                               "both_halves_positive": best[2]["both_halves_positive"], "pos_quarters": best[2]["pos_quarters"],
                               "n_quarters": best[2]["n_quarters"], "worst_q": best[2]["worst_q"], "worst_port_ret": best[2]["worst_port_ret"],
                               "lost_quarters": [x["q"] for x in best[2]["lost_quarters"]],
                               "rule": {"entry": rules[n].expr, "filter": best[2]["filter"], "gate": best[2]["gate"],
                                        "stop": best[2]["stop"], "exit": best[2]["exit"], "cap": CAP}}
print("\nPASS 2 VERDICTS")
for n, v in QUARTERLY["verdict"].items():
    print(f"  {v['verdict']:6s} {TITLE[n]:28s} {v.get('version', '-'):14s} win {v.get('win_all', 0):.0f}%  halves+ {v.get('both_halves_positive')}  "
          f"quarters+ {v.get('pos_quarters')}/{v.get('n_quarters')}  worst {v.get('worst_q')} {v.get('worst_port_ret') or 0:+.1f}%  lost {v.get('lost_quarters')}")


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if k != "parts" or True}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return o


Path("data/screen/swing_lab_breakouts.json").write_text(json.dumps(clean({
    "window": {"start": str(START.date()), "split": str(SPLIT.date()), "end": str(END.date()), "cost": COST, "cap": CAP,
               "slots": SLOTS, "min_n_per_half": MIN_N, "min_kept_per_month": MIN_KEPT_PER_MONTH, "t_min": T_MIN},
    "setups": {n: {"title": TITLE[n], "expr": rules[n].expr, "signals": int(SIG[n].sum()), "struct_stop": STRUCT_STOP[n]} for n in SETUPS},
    "matrix": MATRIX, "best_cells": {n: [x["stop"] + " / " + x["exit"] for x in BEST[n]] for n in SETUPS},
    "grooming": GROOM, "stress": STRESS, "verdict": VERDICT, "quarterly": QUARTERLY}), indent=1))
print(f"\nwrote data/screen/swing_lab_breakouts.json in {time.time()-t0:.0f}s")
