"""Swing lab, trend family: can a popular positional-momentum entry be groomed
into a mostly-winning 2-8 week swing trade with a hard stop?

The six trend/momentum screens in config/rules.yaml are STATE screens (true on
every day the state holds). Traded as-is they produce thousands of overlapping
entries, so each is converted into two EVENTS:

  fresh  - first day the rule is true after >= 10 sessions of being false
  high   - rule true AND today's close is the first new 20-day closing high in
           >= 5 sessions (a fresh high inside a confirmed trend)

plus two popular event entries defined here as masks:

  ema_cross - EMA9 crosses above EMA21 today, close > SMA50, rs_rank >= 60
  macd_turn - EMA21 - EMA50 turns positive today (<= 0 yesterday), close > SMA200

Trade engine, conventions and the stats discipline are those of
scripts/stop_reward_study.py and scripts/loss_forensics.py (entry next open,
stop checked first, gap through the stop fills at the open, close-based trail
exits at the next open, one open position per name, 0.30% cost, time cap).
Every filter threshold is chosen on H1 quintile edges and must hold in H2.

Output: data/screen/swing_lab_trend.json
Run:    python scripts/swing_lab_trend.py   (about 20-40 minutes on 4 cores)
"""
import json, sys, time, itertools
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.rules import load_rules
from screener.backtest import add_panel_columns, rule_hits
from screener.portfolio import simulate
from screener.robustness import stress
from screener import quality
from screener.store import Store

COST, CAP, SLOTS = 0.30, 60, 10
ALT_CAPS = [40, 120]
START, SPLIT = pd.Timestamp("2024-06-01"), pd.Timestamp("2025-03-19")
MIN_N = 100                      # per half, for a cell to count
MIN_PER_MONTH = 15               # kept trades per month, each half, for a filter to pass
T_MIN = 2.0
RULES = ["momentum_leaders", "smooth_momentum", "minervini_trend_template", "trend_stack",
         "stage2_portfolio", "accumulation_thrust"]
STOPS = [("atr", 2.5), ("atr", 3.0), ("atr", 4.0), ("struct", 10)]
EXITS = [("trail", "ema21"), ("trail", "ema50"), ("trail", "low10"), ("trail", "chand3"),
         ("target", 2.0), ("target", 3.0), ("none",),
         # supplementary: the only way a trend entry can approach a 65-70% win rate is a target
         # nearer than the stop; these are included so that claim is tested rather than assumed
         ("target", 1.0), ("target", 0.5)]
STOCK_FEATS = ["rs_rank", "dist_ema_21", "dist_sma_50", "dist_sma_200", "rsi_14", "atr_pct", "vol_ratio",
               "ret_20d", "ret_60d", "mom_12_1", "smooth_60", "pct_from_52w_high", "turnover_median_20d"]
MARKET_NUM = ["vix", "vix_pctile_1y", "breadth_above_50", "breadth_thrust_10", "dispersion", "nifty_range_20d"]
MARKET_BIN = ["R3", "nifty_above_20dma", "nifty_above_50dma", "nifty_above_200dma"]
OUT = Path("data/screen/swing_lab_trend.json")
# --quarterly: skip the matrix / grooming / stress (kept from the JSON on disk) and only add the
# quarter-by-quarter tables and the quarter-consistency verdicts to that JSON.
QONLY = "--quarterly" in sys.argv

t0 = time.time()
# ------------------------------------------------------------------ panel (as stop_reward_study.py)
feat = pd.read_parquet("data/screen/features.parquet")
lat = pd.read_parquet("data/screen/latest_features.parquet")
feat = feat.merge(lat[["isin", "cap_band"]], on="isin", how="left")
st = Store("data")
b = feat[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
feat["contaminated"] = quality.contamination_mask(b, quality.detect_price_jumps(b, st.read_adjustments())).to_numpy()
feat = add_panel_columns(feat).sort_values(["isin", "date"]).reset_index(drop=True)
g = feat.groupby("isin")
feat["low10_prior"] = g["adj_low"].transform(lambda s: s.shift(1).rolling(10).min())
feat["low10_incl"] = g["adj_low"].transform(lambda s: s.rolling(10).min())        # structural stop: prior 10 incl. today
feat["ret1"] = g["adj_close"].pct_change()
feat["liquid"] = ((feat["turnover_median_20d"] >= 1e7) & (feat["bars_available"] >= 250)
                  & (~feat["contaminated"]))
feat["dist_sma_200"] = (feat["adj_close"] / feat["sma_200"] - 1) * 100
feat["e9_prev"] = g["ema_9"].shift(1); feat["e21_prev"] = g["ema_21"].shift(1); feat["e50_prev"] = g["ema_50"].shift(1)
feat["hi20c_prior"] = g["adj_close"].transform(lambda s: s.shift(1).rolling(20).max())
feat["nh20"] = (feat["adj_close"] > feat["hi20c_prior"]).astype(int)
feat["nh20_prev5"] = feat.groupby("isin")["nh20"].transform(lambda s: s.shift(1).rolling(5).sum())

CAL = pd.DatetimeIndex(sorted(feat["date"].unique()))
CALPOS = {d: i for i, d in enumerate(CAL)}
END = CAL[-1]
mkt = feat.loc[feat["liquid"]].groupby("date")["ret1"].mean().reindex(CAL).fillna(0)
BENCH = (1 + mkt).cumprod()

# market state + gate, joined on the signal date
ms = pd.read_parquet("data/screen/market_state.parquet"); ms["date"] = pd.to_datetime(ms["date"])
gate = pd.read_parquet("data/screen/gate_r3.parquet"); gate["date"] = pd.to_datetime(gate["date"])
ms = ms.merge(gate[["date", "R3"]], on="date", how="outer").sort_values("date")
MS = ms.set_index("date")[MARKET_NUM + MARKET_BIN]
feat = feat.merge(MS, left_on="date", right_index=True, how="left")

O = feat["adj_open"].to_numpy(float); H = feat["adj_high"].to_numpy(float)
L = feat["adj_low"].to_numpy(float);  C = feat["adj_close"].to_numpy(float)
ATRP = feat["atr_pct"].to_numpy(float) / 100.0
E21 = feat["ema_21"].to_numpy(float); E50 = feat["ema_50"].to_numpy(float)
LO10 = feat["low10_prior"].to_numpy(float); LO10S = feat["low10_incl"].to_numpy(float)
RS = feat["rs_rank"].to_numpy(float)
ISIN = feat["isin"].to_numpy(); DATES = feat["date"].to_numpy()
LAST = pd.Series(np.arange(len(feat))).groupby(pd.factorize(feat["isin"])[0]).transform("max").to_numpy()
IN_WIN = ((feat["date"] >= START) & (feat["date"] <= END)).to_numpy()
LIQ = feat["liquid"].to_numpy() & (feat["close"].fillna(0) >= 20).to_numpy()
print(f"panel {len(feat):,} rows | {CAL[0].date()} to {END.date()} | signals from {START.date()}, "
      f"split {SPLIT.date()} | {time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------------------------ events
rules = {r.name: r for r in load_rules("config/rules.yaml")}
grp_id = pd.factorize(feat["isin"])[0]


def first_after_quiet(mask, quiet):
    """True where mask is true today and was false on each of the prior `quiet` sessions (per name)."""
    s = pd.Series(mask.astype(int))
    prior = s.groupby(grp_id).transform(lambda x: x.shift(1).rolling(quiet).sum())
    return mask & (prior.to_numpy() == 0)


SIG, SIG_INFO = {}, {}
NH_FRESH = (feat["nh20"].to_numpy() == 1) & (feat["nh20_prev5"].to_numpy() == 0)
for n in RULES:
    h = rule_hits(feat, rules[n]).to_numpy()
    SIG[f"{n}:fresh"] = first_after_quiet(h, 10) & IN_WIN & LIQ
    SIG[f"{n}:high"] = h & NH_FRESH & IN_WIN & LIQ
    SIG_INFO[n] = {"state_days": int((h & IN_WIN).sum())}
    print(f"  {n:26s} state-days {int((h & IN_WIN).sum()):>8,d}  fresh {SIG[f'{n}:fresh'].sum():>6,d}  "
          f"high {SIG[f'{n}:high'].sum():>6,d}", flush=True)
ema_x = ((feat["ema_9"] > feat["ema_21"]) & (feat["e9_prev"] <= feat["e21_prev"]) & (feat["adj_close"] > feat["sma_50"])
         & (feat["rs_rank"] >= 60)).fillna(False).to_numpy()
macd = (((feat["ema_21"] - feat["ema_50"]) > 0) & ((feat["e21_prev"] - feat["e50_prev"]) <= 0)
        & (feat["adj_close"] > feat["sma_200"])).fillna(False).to_numpy()
SIG["ema_cross:event"] = ema_x & IN_WIN & LIQ & (feat["bars_available"].to_numpy() >= 200)
SIG["macd_turn:event"] = macd & IN_WIN & LIQ & (feat["bars_available"].to_numpy() >= 200)
for k in ("ema_cross:event", "macd_turn:event"):
    print(f"  {k:26s} events {SIG[k].sum():>6,d}", flush=True)
SETUPS = list(SIG.keys())
TITLE = {f"{n}:{v}": f"{rules[n].title} [{v}]" for n in RULES for v in ("fresh", "high")}
TITLE.update({"ema_cross:event": "EMA9/21 cross >50DMA", "macd_turn:event": "EMA21-50 turn >200DMA"})
FEAT_COLS = STOCK_FEATS + MARKET_NUM + MARKET_BIN + ["cap_band"]
FEATS = feat[FEAT_COLS]


def run(sig, stop, reward, cap=CAP):
    """One trade per signal with a hard initial stop (engine of scripts/stop_reward_study.py).

    stop:   ('atr', m) m x ATR(14)% at the signal bar | ('struct', 10) lowest low of the last 10
            sessions, accepted only when 3-20% below the entry
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
            if not np.isfinite(LO10S[t]):
                continue
            risk = e - LO10S[t]
        rpct = risk / e * 100.0
        lo_r, hi_r = (3.0, 20.0) if stop[0] == "struct" else (0.2, 20.0)
        if not (lo_r < rpct < hi_r):
            continue
        sl = e - risk
        tgt = e + reward[1] * risk if reward[0] == "target" else np.inf
        atr_abs = ATRP[t] * e
        end = min(LAST[t], t + cap)
        k = t + 1; peak = e; lowest = e
        out = None; why = "time"
        while k <= end:
            lowest = min(lowest, L[k])
            if L[k] <= sl:
                out = min(O[k], sl); why = "stop"; break
            peak = max(peak, H[k])
            if reward[0] == "target" and H[k] >= tgt:
                out = max(O[k], tgt); why = "target"; break
            if reward[0] == "trail":
                lvl = (E21[k] if reward[1] == "ema21" else E50[k] if reward[1] == "ema50"
                       else LO10[k] if reward[1] == "low10" else peak - 3.0 * atr_abs)
                if np.isfinite(lvl) and C[k] < lvl and k + 1 <= LAST[t]:
                    out = O[k + 1]; why = "trail"; k += 1
                    lowest = min(lowest, L[k]); peak = max(peak, H[k]); break
            k += 1
        if out is None:
            k = min(k, end); out = C[k]
        if not np.isfinite(out):
            continue
        busy[ISIN[t]] = k
        ei, xi = CALPOS[pd.Timestamp(DATES[t + 1])], CALPOS[pd.Timestamp(DATES[k])]
        bench = (BENCH.iloc[xi] / BENCH.iloc[ei] - 1) * 100
        rows.append((pd.Timestamp(DATES[t]), ISIN[t], t, ei, xi, (out / e - 1) * 100 - COST, bench, k - t,
                     rpct, RS[t], why, e, out, (lowest / e - 1) * 100, (peak / e - 1) * 100))
    return pd.DataFrame(rows, columns=["date", "isin", "sig_i", "entry_i", "exit_i", "net", "bench", "bars",
                                       "risk_pct", "rs", "why", "entry", "exit", "mae", "mfe"])


def with_feats(tr):
    return pd.concat([tr.reset_index(drop=True), FEATS.iloc[tr["sig_i"].to_numpy()].reset_index(drop=True)], axis=1)


def stats(t, min_n=MIN_N):
    if len(t) < min_n:
        return None
    n = t["net"]; w = n[n > 0]; l = n[n <= 0]
    if not len(w) or not len(l):
        return None
    why = t["why"].value_counts(normalize=True) * 100
    return {"n": int(len(t)), "win": float((n > 0).mean() * 100), "net": float(n.mean()), "median": float(n.median()),
            "avg_win": float(w.mean()), "avg_loss": float(l.mean()),
            "payoff": float(w.mean() / -l.mean()), "pf": float(w.sum() / -l.sum()),
            "stopped": float(why.get("stop", 0)), "target": float(why.get("target", 0)),
            "trail": float(why.get("trail", 0)), "time": float(why.get("time", 0)),
            "bars": float(t["bars"].mean()), "risk": float(t["risk_pct"].mean()),
            "exp_r": float((n / t["risk_pct"]).mean()), "bench": float(t["bench"].mean()),
            "days": int(t["date"].nunique()), "mae": float(t["mae"].mean()), "mfe": float(t["mfe"].mean())}


def label(stop, reward):
    s = f"{stop[1]:g} ATR" if stop[0] == "atr" else "10d low"
    r = (f"1:{reward[1]:g}" if reward[0] == "target" else
         {"ema21": "close<21EMA", "ema50": "close<50EMA", "low10": "close<10d low",
          "chand3": "chandelier 3ATR"}[reward[1]] if reward[0] == "trail" else "stop only")
    return s, r


def cell(job):
    setup, stop, reward, cap = job
    tr = run(SIG[setup], stop, reward, cap)
    if not len(tr):
        return None
    h1, h2 = tr[tr["date"] < SPLIT], tr[tr["date"] >= SPLIT]
    a, s1, s2 = stats(tr), stats(h1), stats(h2)
    sl, rl = label(stop, reward)
    d = {"setup": setup, "stop": sl, "exit": rl, "stop_spec": list(stop), "exit_spec": list(reward), "cap": cap,
         "n_raw": int(len(tr)), "all": a, "h1": s1, "h2": s2, "counts": False}
    if a and s1 and s2:
        d["counts"] = True
        d["both"] = bool(s1["net"] > 0 and s2["net"] > 0)
        d["worse_win"] = float(min(s1["win"], s2["win"])); d["worse_net"] = float(min(s1["net"], s2["net"]))
    return d


# ------------------------------------------------------------------ sanity: five trades by hand
if not QONLY:
    print("\nSANITY: Momentum Leaders [fresh], 3 ATR stop, close<50EMA trail, cap 60 - five trades", flush=True)
    sn = with_feats(run(SIG["momentum_leaders:fresh"], ("atr", 3.0), ("trail", "ema50")))
    sym = feat.dropna(subset=["symbol"]).groupby("isin")["symbol"].last().astype(str)
    for r_ in sn.sample(5, random_state=3).sort_values("date").itertuples():
        print(f"  {str(sym.get(r_.isin, r_.isin)):12s} signal {r_.date.date()} entry {CAL[r_.entry_i].date()} @ {r_.entry:9.2f}  "
              f"exit {CAL[r_.exit_i].date()} @ {r_.exit:9.2f}  {r_.why:6s} bars {r_.bars:3d} net {r_.net:+6.2f}% "
              f"risk {r_.risk_pct:4.1f}% mae {r_.mae:+5.1f} mfe {r_.mfe:+5.1f} rs {r_.rs_rank:.0f}", flush=True)
    SANITY = sn.sample(5, random_state=3).sort_values("date")[["date", "isin", "entry_i", "exit_i", "entry", "exit", "why", "bars", "net", "risk_pct", "mae", "mfe"]]
    SANITY_ROWS = [{**{k: (str(v.date()) if isinstance(v, pd.Timestamp) else (float(v) if isinstance(v, (float, np.floating)) else v))
                       for k, v in r_.items()},
                    "symbol": str(sym.get(r_["isin"], r_["isin"])), "entry_date": str(CAL[r_["entry_i"]].date()),
                    "exit_date": str(CAL[r_["exit_i"]].date())} for r_ in SANITY.to_dict("records")]
    # hand-check invariants
    chk = sn
    assert (chk["exit_i"] > chk["entry_i"]).all() or (chk["exit_i"] >= chk["entry_i"]).all()
    assert (chk.loc[chk["why"] == "stop", "net"] <= 0.5).all(), "a stop-out with a profit?"
    assert (chk["bars"] <= CAP + 1).all()
    print(f"  invariants ok: {len(chk):,} trades, stop-outs never profitable, bars <= cap+1, "
          f"stop-out mean net {chk.loc[chk['why']=='stop','net'].mean():+.2f}% vs mean risk {chk['risk_pct'].mean():.2f}%", flush=True)

    # ------------------------------------------------------------------ the matrix
    jobs = [(s, st_, ex, CAP) for s in SETUPS for st_ in STOPS for ex in EXITS]
    print(f"\n{len(jobs)} cells at cap {CAP} ...", flush=True)
    with Pool(4) as pool:
        GRID = [x for x in pool.map(cell, jobs, chunksize=4) if x]
    print(f"{sum(x['counts'] for x in GRID)} of {len(GRID)} cells have >= {MIN_N} trades per half | {time.time()-t0:.0f}s", flush=True)


def show_cells(rows, title):
    print(f"\n{title}")
    print(f"{'setup':34s}{'stop':>8s} {'exit':16s}{'n':>6s}{'win%':>5s}{'net':>6s}{'aW':>6s}{'aL':>6s}{'pay':>5s}{'PF':>5s}"
          f"{'stp%':>5s}{'bars':>5s} | {'H1 n':>5s}{'win':>4s}{'net':>6s} | {'H2 n':>5s}{'win':>4s}{'net':>6s}  ok")
    print("-" * 140)
    for x in rows:
        a, h1, h2 = x["all"], x["h1"], x["h2"]
        print(f"{TITLE[x['setup']][:34]:34s}{x['stop']:>8s} {x['exit']:16s}{a['n']:>6,d}{a['win']:>5.0f}{a['net']:>+6.2f}"
              f"{a['avg_win']:>6.2f}{a['avg_loss']:>6.2f}{a['payoff']:>5.2f}{a['pf']:>5.2f}{a['stopped']:>5.0f}{a['bars']:>5.0f} | "
              f"{h1['n']:>5d}{h1['win']:>4.0f}{h1['net']:>+6.2f} | {h2['n']:>5d}{h2['win']:>4.0f}{h2['net']:>+6.2f}  {'YES' if x['both'] else ''}")


def pick_cells(setup, k=4):
    """Two cells by the worse half's win rate and two by its mean net (both-halves-positive first),
    so the tiny-target cells that win often and the wide-target cells that pay do not crowd each other out."""
    cs = [x for x in GRID if x["setup"] == setup and x["counts"]]
    both = [x for x in cs if x["both"]]
    by_win = sorted(both, key=lambda x: (-x["worse_win"], -x["worse_net"]))
    by_net = sorted(both, key=lambda x: (-x["worse_net"], -x["worse_win"]))
    rest = sorted([x for x in cs if not x["both"]], key=lambda x: (-x["worse_net"], -x["worse_win"]))
    out = []
    for x in by_win[:2] + by_net[:2] + rest:
        if x not in out:
            out.append(x)
        if len(out) >= k:
            break
    return out


if not QONLY:
    for s in SETUPS:
        show_cells(sorted([x for x in GRID if x["setup"] == s and x["counts"]], key=lambda x: (x["stop"], x["exit"])),
                   f"MATRIX  {TITLE[s]}  (raw signals {SIG[s].sum():,d})")
    PICKS = {s: pick_cells(s) for s in SETUPS}
    show_cells([x for s in SETUPS for x in PICKS[s]], "BASELINE WINNERS: best 3 cells per setup (both-halves-positive first, then worse-half win rate)")

    # cap sensitivity on the picked cells
    alt_jobs = [(x["setup"], tuple(x["stop_spec"]), tuple(x["exit_spec"]), c) for s in SETUPS for x in PICKS[s] for c in ALT_CAPS]
    with Pool(4) as pool:
        ALT = [x for x in pool.map(cell, alt_jobs, chunksize=2) if x]
    print(f"\nCAP SENSITIVITY on picked cells (cap 40 / 60 / 120): worse-half win% / net")
    for s in SETUPS:
        for x in PICKS[s]:
            line = f"  {TITLE[s][:34]:34s}{x['stop']:>8s} {x['exit']:16s}"
            for c in (40, 60, 120):
                y = x if c == 60 else next((z for z in ALT if z["setup"] == s and z["stop"] == x["stop"] and z["exit"] == x["exit"] and z["cap"] == c), None)
                line += f" | cap{c:<3d} " + (f"{y['worse_win']:4.0f}% {y['worse_net']:+5.2f}" if y and y["counts"] else "   -      -")
            print(line)

# ------------------------------------------------------------------ grooming
def cluster_t(y, x, g):
    y = np.asarray(y, float); x = np.asarray(x, float)
    if len(y) < 10 or x.std() == 0:
        return np.nan
    X = np.column_stack([np.ones(len(x)), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    S = pd.DataFrame({"g": np.asarray(g), "e0": e, "e1": e * x}).groupby("g")[["e0", "e1"]].sum().to_numpy()
    V = XtX_inv @ (S.T @ S) @ XtX_inv
    return float(beta[1] / np.sqrt(V[1, 1])) if V[1, 1] > 0 else np.nan


def r(x, k=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), k)


def half_base(tr):
    out = {}
    for h, s in (("H1", tr[tr["date"] < SPLIT]), ("H2", tr[tr["date"] >= SPLIT])):
        months = max((s["date"].max() - s["date"].min()).days / 30.44, 1) if len(s) else 1
        out[h] = {"n": int(len(s)), "win": r((s["net"] > 0).mean()), "net": r(s["net"].mean()), "months": r(months, 1),
                  "per_month": r(len(s) / months, 1)}
    return out


def eval_filter(tr, mask, base):
    res = {}
    for h in ("H1", "H2"):
        hm = (tr["date"] < SPLIT) if h == "H1" else (tr["date"] >= SPLIT)
        s = tr[hm]; m = mask[hm].fillna(False).to_numpy(bool)
        kept, rem = s[m], s[~m]
        d = {"n_all": int(len(s)), "n_kept": int(len(kept)), "kept_pct": r(len(kept) / max(len(s), 1)),
             "kept_per_month": r(len(kept) / base[h]["months"], 1),
             "wr_kept": r((kept["net"] > 0).mean()) if len(kept) else None,
             "mean_kept": r(kept["net"].mean()) if len(kept) else None,
             "median_kept": r(kept["net"].median()) if len(kept) else None,
             "pf_kept": r(kept.loc[kept["net"] > 0, "net"].sum() / max(-kept.loc[kept["net"] <= 0, "net"].sum(), 1e-9), 2) if len(kept) else None,
             "wr_removed": r((rem["net"] > 0).mean()) if len(rem) else None,
             "mean_removed": r(rem["net"].mean()) if len(rem) else None,
             "t_cluster": r(cluster_t(s["net"], m.astype(float), s["date"]), 2) if 0 < m.sum() < len(s) else None,
             "n_days_kept": int(kept["date"].nunique()),
             # regime concentration: share of the half's calendar months in which >= 5 trades were kept
             "months_active_share": r((kept["date"].dt.to_period("M").value_counts() >= 5).sum()
                                      / max(s["date"].dt.to_period("M").nunique(), 1))}
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
        if d["kept_per_month"] < MIN_PER_MONTH:
            return False
        ts.append(d["t_cluster"])
    return bool(max(ts) >= T_MIN and min(ts) > 0)


def score(res):
    return (min(res["H1"]["wr_kept"], res["H2"]["wr_kept"]), min(res["H1"]["mean_kept"], res["H2"]["mean_kept"]))


def candidates(tr):
    """Single filters: numeric features at H1 quintile edges in the direction the H1 day-clustered t
    points (in-sample choice), binaries both ways, cap_band in/out of each band."""
    h1 = tr[tr["date"] < SPLIT]
    out = []
    for f in STOCK_FEATS + MARKET_NUM:
        x1 = h1[f].dropna()
        if len(x1) < 100:
            continue
        edges = np.unique(np.nanpercentile(x1, [20, 40, 60, 80]))
        t_dir = cluster_t(h1.loc[x1.index, "net"], x1.rank(pct=True), h1.loc[x1.index, "date"])
        if not np.isfinite(t_dir):
            continue
        op = ">=" if t_dir > 0 else "<="
        for i, thr in enumerate(edges):
            out.append({"feature": f, "op": op, "thr": float(thr), "edge_pct": int([20, 40, 60, 80][i]) if len(edges) == 4 else None,
                        "label": f"{f} {op} {thr:.4g}", "level": "stock" if f in STOCK_FEATS else "market"})
    for f in MARKET_BIN:
        for v in (1, 0):
            out.append({"feature": f, "op": "==", "thr": float(v), "label": f"{f} == {v}", "level": "market"})
    for band in ("Mid", "Large", "Mega"):
        out.append({"feature": "cap_band", "op": "==", "thr": band, "label": f"cap_band == {band}", "level": "stock"})
        out.append({"feature": "cap_band", "op": "!=", "thr": band, "label": f"cap_band != {band}", "level": "stock"})
    return out


def mask_of(tr, f):
    x = tr[f["feature"]]
    if f["op"] == ">=":
        return x >= f["thr"]
    if f["op"] == "<=":
        return x <= f["thr"]
    if f["op"] == "==":
        return x == f["thr"]
    return (x != f["thr"]) & x.notna()


def groom(tr):
    base = half_base(tr)
    singles = []
    for f in candidates(tr):
        m = mask_of(tr, f)
        res = eval_filter(tr, m, base)
        if res["H1"]["n_kept"] < 30 or res["H2"]["n_kept"] < 30:
            continue
        singles.append({**f, "passes": filter_passes(res, base), "H1": res["H1"], "H2": res["H2"], "k": 1})
    passing = sorted([f for f in singles if f["passes"]], key=score, reverse=True)
    best_per = {}
    for f in passing:
        best_per.setdefault(f["feature"], f)
    top = list(best_per.values())[:8]
    combos = []
    for a, b_ in itertools.combinations(top, 2):
        m = mask_of(tr, a) & mask_of(tr, b_)
        res = eval_filter(tr, m, base)
        if res["H1"]["n_kept"] < 30 or res["H2"]["n_kept"] < 30:
            continue
        combos.append({"label": f"{a['label']} AND {b_['label']}", "parts": [a, b_], "k": 2,
                       "passes": filter_passes(res, base), "H1": res["H1"], "H2": res["H2"]})
    board = sorted([f for f in singles + combos if f["passes"]], key=score, reverse=True)
    return {"base": base, "n_single_tested": len(singles), "n_single_passing": len(passing),
            "n_combo_tested": len(combos), "n_combo_passing": sum(f["passes"] for f in combos),
            "leaderboard": board[:25], "top_singles": top,
            "all_singles": [{k: v for k, v in f.items() if k != "parts"} for f in singles]}


def mask_of_any(tr, f):
    if f.get("k", 1) == 2:
        return mask_of(tr, f["parts"][0]) & mask_of(tr, f["parts"][1])
    return mask_of(tr, f)


def trades_for(x, cap=CAP):
    return with_feats(run(SIG[x["setup"]], tuple(x["stop_spec"]), tuple(x["exit_spec"]), cap))


def groom_job(x):
    tr = trades_for(x)
    return x["setup"], x["stop"], x["exit"], groom(tr)


if not QONLY:
    print(f"\nGROOMING {sum(len(v) for v in PICKS.values())} cells ...", flush=True)
    with Pool(4) as pool:
        GR = pool.map(groom_job, [x for s in SETUPS for x in PICKS[s]], chunksize=1)
    GROOM = {}
    for setup, stop, exit_, g_ in GR:
        GROOM[(setup, stop, exit_)] = g_
    print(f"grooming done | {time.time()-t0:.0f}s", flush=True)


def show_board(setup, stop, exit_, g_, k=8):
    b = g_["base"]
    print(f"\nFILTERS  {TITLE[setup]} | {stop} | {exit_}   base H1 n={b['H1']['n']} win {b['H1']['win']*100:.0f}% net {b['H1']['net']:+.2f} | "
          f"H2 n={b['H2']['n']} win {b['H2']['win']*100:.0f}% net {b['H2']['net']:+.2f} | "
          f"tested {g_['n_single_tested']}+{g_['n_combo_tested']}, passing {g_['n_single_passing']}+{g_['n_combo_passing']}")
    if not g_["leaderboard"]:
        print("  no filter passes in both halves")
        return
    print(f"  {'filter':70s}{'H1 kept':>8s}{'win':>5s}{'net':>6s}{'t':>5s}{'/mo':>5s} | {'H2 kept':>8s}{'win':>5s}{'net':>6s}{'t':>5s}{'/mo':>5s}")
    for f in g_["leaderboard"][:k]:
        h1, h2 = f["H1"], f["H2"]
        print(f"  {f['label'][:70]:70s}{h1['n_kept']:>8d}{h1['wr_kept']*100:>5.0f}{h1['mean_kept']:>+6.2f}{h1['t_cluster']:>5.1f}{h1['kept_per_month']:>5.0f} | "
              f"{h2['n_kept']:>8d}{h2['wr_kept']*100:>5.0f}{h2['mean_kept']:>+6.2f}{h2['t_cluster']:>5.1f}{h2['kept_per_month']:>5.0f}")


if not QONLY:
    for s in SETUPS:
        for x in PICKS[s]:
            show_board(s, x["stop"], x["exit"], GROOM[(s, x["stop"], x["exit"])])

# ------------------------------------------------------------------ portfolio stress
CACHE = {}


def cached_trades(setup, stop, reward, cap):
    key = (setup, stop, reward, cap)
    if key not in CACHE:
        CACHE[key] = with_feats(run(SIG[setup], stop, reward, cap))
    return CACHE[key]


def make_maker(setup, stop, reward, filt, gate_on):
    """Build make_trades(params). Numeric params (atr multiple, rr, cap, filter thresholds) are what
    stress() nudges by +/-20%."""
    def make(params):
        st_ = ("atr", float(params["atr"])) if stop[0] == "atr" else ("struct", 10)
        rw = ("target", float(params["rr"])) if reward[0] == "target" else reward
        tr = cached_trades(setup, st_, rw, int(params["cap"]))
        m = pd.Series(True, index=tr.index)
        if filt is not None:
            parts = filt["parts"] if filt.get("k", 1) == 2 else [filt]
            for i, p in enumerate(parts):
                thr = params.get(f"thr{i}", p["thr"])
                m &= mask_of(tr, {**p, "thr": thr})
        if gate_on:
            m &= tr["R3"] == 1
        return tr[m.fillna(False)].reset_index(drop=True)
    return make


def params_for(stop, reward, filt, cap=CAP):
    p = {"cap": int(cap)}
    if stop[0] == "atr":
        p["atr"] = float(stop[1])
    if reward[0] == "target":
        p["rr"] = float(reward[1])
    if filt is not None:
        parts = filt["parts"] if filt.get("k", 1) == 2 else [filt]
        for i, q in enumerate(parts):
            if q["op"] in (">=", "<="):
                p[f"thr{i}"] = float(q["thr"])
    return p


def per_trade(tr):
    a = stats(tr, min_n=1)
    return {"n": int(len(tr)), "win": r(a["win"], 1) if a else None, "net": r(a["net"]) if a else None,
            "pf": r(a["pf"], 2) if a else None, "per_month": r(len(tr) / max((tr["date"].max() - tr["date"].min()).days / 30.44, 1), 1) if len(tr) else 0}


def best_groomed(setup, by="win"):
    """Best (cell, filter) across the setup's picked cells: by the worse half's kept win rate, or by
    its kept mean net (each cell contributes its own leaderboard leader under that ordering)."""
    best = None
    for x in PICKS[setup]:
        g_ = GROOM[(setup, x["stop"], x["exit"])]
        if not g_["leaderboard"]:
            continue
        key = (lambda f: (min(f["H1"]["wr_kept"], f["H2"]["wr_kept"]), min(f["H1"]["mean_kept"], f["H2"]["mean_kept"]))) if by == "win" \
            else (lambda f: (min(f["H1"]["mean_kept"], f["H2"]["mean_kept"]), min(f["H1"]["wr_kept"], f["H2"]["wr_kept"])))
        f = max(g_["leaderboard"], key=key)
        cand = (key(f), x, f)
        if best is None or cand[0] > best[0]:
            best = cand
    return best


if not QONLY:
    PORT = []
    print(f"\nPORTFOLIO STRESS: 10 slots, rs order, {100} random orders, neighbours +/-20% on numeric params")
    print(f"{'version':78s}{'n':>6s}{'win%':>5s}{'net':>6s} | {'CAGR':>6s}{'maxDD':>6s} | {'H1':>6s}{'DD':>6s} | {'H2':>6s}{'DD':>6s} | {'rnd p5':>7s}{'nb min':>7s}  robust")
    print("-" * 160)
    for s in SETUPS:
        fams = []
        for by in ("win", "net"):
            bg = best_groomed(s, by)
            x = bg[1] if bg else (PICKS[s][0] if PICKS[s] else None)
            if x is None:
                continue
            key = (x["stop"], x["exit"], bg[2]["label"] if bg else None)
            if any(k == key for k, _, _ in fams):
                continue
            fams.append((key, x, bg[2] if bg else None))
        for (_, x, filt) in fams:
          stop, reward = tuple(x["stop_spec"]), tuple(x["exit_spec"])
          versions = [("baseline", None, False), ("baseline+gate", None, True)]
          if filt is not None:
              versions += [("groomed", filt, False), ("groomed+gate", filt, True)]
          for name, f_, gate_on in versions:
              mk = make_maker(s, stop, reward, f_, gate_on)
              p = params_for(stop, reward, f_)
              try:
                  res = stress(mk, p, sessions=CAL, split=SPLIT, slots=SLOTS, order_col="rs", n_random=100)
              except Exception as ex:                      # too few trades, etc.
                  res = {"error": str(ex)}
              tr = mk(p)
              row = {"setup": s, "title": TITLE[s], "version": name, "stop": x["stop"], "exit": x["exit"], "cap": CAP,
                     "filter": f_["label"] if f_ else None, "gate": gate_on, "params": p, "per_trade": per_trade(tr),
                     "h1_trade": per_trade(tr[tr["date"] < SPLIT]), "h2_trade": per_trade(tr[tr["date"] >= SPLIT]), "stress": res}
              PORT.append(row)
              pt = row["per_trade"]
              if "error" in res:
                  print(f"{(TITLE[s] + ' | ' + name)[:78]:78s}{pt['n']:>6d}  error: {res['error'][:60]}")
                  continue
              fl, h1, h2 = res["full"], res["h1"], res["h2"]
              print(f"{(TITLE[s] + ' | ' + x['stop'] + ' | ' + x['exit'] + ' | ' + name)[:78]:78s}{pt['n']:>6d}{pt['win'] or 0:>5.0f}{pt['net'] or 0:>+6.2f} | "
                    f"{fl['cagr']:>+6.1f}{fl['maxdd']:>6.1f} | {h1['cagr']:>+6.1f}{h1['maxdd']:>6.1f} | {h2['cagr']:>+6.1f}{h2['maxdd']:>6.1f} | "
                    f"{res['random_order']['p5']:>+7.1f}{res['neighbour_min']:>+7.1f}  {'YES' if res['robust'] else 'no'}")
    print(f"portfolio done | {time.time()-t0:.0f}s", flush=True)

if QONLY:
    PREV = json.loads(OUT.read_text())
    PORT = PREV["portfolio"]
    print(f"--quarterly: {len(PORT)} portfolio versions loaded from {OUT}", flush=True)

# ------------------------------------------------------------------ verdicts
if not QONLY:
    VERD = []
    for s in SETUPS:
        rows = [p for p in PORT if p["setup"] == s]
        if not rows:
            VERD.append({"setup": s, "title": TITLE[s], "verdict": "DROP", "why": "no cell reached 100 trades per half"}); continue
        def wmin_of(p):
            return min(p["h1_trade"]["win"] or 0, p["h2_trade"]["win"] or 0)
        cands = rows
        robust_c = [p for p in cands if "error" not in p["stress"] and p["stress"]["robust"]]
        best = max(robust_c, key=wmin_of) if robust_c else max(cands, key=wmin_of)
        wmin = min(best["h1_trade"]["win"] or 0, best["h2_trade"]["win"] or 0)
        nmin = min(best["h1_trade"]["net"] or -9, best["h2_trade"]["net"] or -9)
        st_ = best["stress"]
        robust = ("error" not in st_) and st_["robust"]
        dd = st_["full"]["maxdd"] if "error" not in st_ else np.nan
        if wmin >= 65 and nmin > 0 and robust and dd > -15:
            v = "TRADE"
        elif wmin >= 50 and nmin > 0 and robust and dd > -20:
            v = "MAYBE"
        else:
            v = "DROP"
        VERD.append({"setup": s, "title": TITLE[s], "verdict": v, "version": best["version"], "stop": best["stop"], "exit": best["exit"],
                     "filter": best["filter"], "gate": best["gate"], "worse_half_win": r(wmin, 1), "worse_half_net": r(nmin),
                     "maxdd": r(dd, 1), "robust": bool(robust), "n": best["per_trade"]["n"],
                     "cagr": r(st_["full"]["cagr"], 1) if "error" not in st_ else None})
    print("\nVERDICTS")
    for v in VERD:
        print(f"  {v['verdict']:6s} {v['title']:36s} {v.get('version','')!s:14s} worse-half win {v.get('worse_half_win')} net {v.get('worse_half_net')} "
              f"maxDD {v.get('maxdd')} robust {v.get('robust')} | {v.get('stop')} | {v.get('exit')} | {v.get('filter')}{' | R3 gate' if v.get('gate') else ''}")


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


if not QONLY:
    OUT.write_text(json.dumps(clean({
        "window": {"start": str(START.date()), "split": str(SPLIT.date()), "end": str(END.date()), "cost": COST, "cap": CAP,
                   "alt_caps": ALT_CAPS, "slots": SLOTS, "min_n_per_half": MIN_N, "min_kept_per_month": MIN_PER_MONTH, "t_min": T_MIN},
        "setups": {s: {"title": TITLE[s], "signals": int(SIG[s].sum())} for s in SETUPS},
        "state_days": SIG_INFO, "sanity": SANITY_ROWS,
        "grid": GRID, "picks": {s: [{"stop": x["stop"], "exit": x["exit"]} for x in PICKS[s]] for s in SETUPS},
        "cap_sensitivity": ALT,
        "grooming": [{"setup": k[0], "stop": k[1], "exit": k[2], **{kk: vv for kk, vv in v.items() if kk != "all_singles"},
                      "all_singles": v["all_singles"]} for k, v in GROOM.items()],
        "portfolio": PORT, "verdicts": VERD}), indent=1))
    print(f"\nwrote {OUT} in {time.time()-t0:.0f}s")


# ------------------------------------------------------------------ quarterly consistency (second pass)
# Calendar quarter of the signal date: per-trade n / win / mean net, and the 10-slot account's return
# over that quarter (rs order, equity marked at cost while open, as screener.portfolio.simulate).
# The refined verdict asks for quarter-by-quarter consistency, not just the two halves.
Q_FIRST, Q_MIN_POS, Q_WORST = "2024Q3", 6, -8.0


def parse_filter(labelstr):
    if not labelstr:
        return None
    parts = []
    for part in labelstr.split(" AND "):
        f, op, thr = part.split(" ", 2)
        try:
            thr = float(thr)
        except ValueError:
            pass
        parts.append({"feature": f, "op": op, "thr": thr})
    return parts[0] if len(parts) == 1 else {"k": 2, "parts": parts}


def quarter_context():
    q = {}
    b = BENCH.loc[START:END]
    ends = b.groupby(b.index.to_period("Q")).last()
    starts = pd.concat([pd.Series([b.iloc[0]]), ends.iloc[:-1].reset_index(drop=True)]).to_numpy()
    m = MS.loc[START:END]
    for i, (per, v) in enumerate(ends.items()):
        mm = m[m.index.to_period("Q") == per]
        q[str(per)] = {"ew_universe_ret": r((v / starts[i] - 1) * 100, 1),
                       "breadth_above_50": r(mm["breadth_above_50"].mean(), 1), "vix": r(mm["vix"].mean(), 1),
                       "vix_pctile_1y": r(mm["vix_pctile_1y"].mean(), 1),
                       "r3_share": r(mm["R3"].mean(), 2), "nifty_above_200_share": r(mm["nifty_above_200dma"].mean(), 2),
                       "sessions": int(len(mm))}
    return q


def quarterly_table(tr):
    tr = tr.sort_values(["entry_i", "rs"], ascending=[True, False])
    rows = {}
    per = tr["date"].dt.to_period("Q")
    for qq, g_ in tr.groupby(per):
        rows[str(qq)] = {"n": int(len(g_)), "win": r((g_["net"] > 0).mean() * 100, 1), "net": r(g_["net"].mean(), 2)}
    if len(tr) >= 10:
        eq = simulate(tr, slots=SLOTS, sessions=CAL)["equity"]
        qe = eq.groupby(eq.index.to_period("Q")).last()
        prev = 100.0
        for qq, v in qe.items():
            rows.setdefault(str(qq), {"n": 0, "win": None, "net": None})["port_ret"] = r((v / prev - 1) * 100, 2)
            prev = v
    for k in rows:
        rows[k].setdefault("port_ret", None)
    tested = {k: v for k, v in rows.items() if k >= Q_FIRST and v["port_ret"] is not None}
    # a quarter in which the filter let nothing through (n = 0, account flat) is IDLE, not lost;
    # it still counts against the ">= 6 positive quarters" test, which is the conservative reading
    idle = sorted(k for k, v in tested.items() if v["n"] == 0 and v["port_ret"] == 0)
    pos = sum(v["port_ret"] > 0 for v in tested.values())
    worst = min((v["port_ret"] for v in tested.values()), default=None)
    worst_q = min(tested, key=lambda k: tested[k]["port_ret"]) if tested else None
    return {"quarters": dict(sorted(rows.items())), "n_quarters_tested": len(tested), "n_positive": int(pos),
            "n_idle": len(idle), "idle_quarters": idle,
            "share_positive": r(pos / max(len(tested), 1)), "worst_quarter": worst_q, "worst_quarter_ret": r(worst, 2) if worst is not None else None,
            "losing_quarters": sorted(k for k, v in tested.items() if v["port_ret"] < 0 or (v["port_ret"] == 0 and v["n"] > 0)),
            "quarter_test": bool(pos >= Q_MIN_POS and worst is not None and worst > Q_WORST)}


QCTX = quarter_context()
QUART = []
print(f"\nQUARTERLY: signal-date quarter | n win net | 10-slot account return  (test: >= {Q_MIN_POS} positive quarters from {Q_FIRST}, none below {Q_WORST:+.0f}%)")
for p in PORT:
    stop = ("atr", float(p["params"]["atr"])) if "atr" in p["params"] else ("struct", 10)
    ex = next(e for e in EXITS if label(stop, e)[1] == p["exit"])
    filt = parse_filter(p["filter"])
    tr = make_maker(p["setup"], stop, ex, filt, p["gate"])(p["params"])
    qt = quarterly_table(tr)
    QUART.append({"setup": p["setup"], "title": p["title"], "version": p["version"], "stop": p["stop"], "exit": p["exit"],
                  "filter": p["filter"], "gate": p["gate"], "n": int(len(tr)),
                  "win_all": r((tr["net"] > 0).mean() * 100, 1) if len(tr) else None,
                  "h1_net": p["h1_trade"]["net"], "h2_net": p["h2_trade"]["net"], "robust": bool(p["stress"].get("robust", False)),
                  **qt})
    qs = qt["quarters"]
    line = " ".join(f"{k[2:]}:{(v['port_ret'] if v['port_ret'] is not None else float('nan')):+5.1f}" for k, v in qs.items())
    print(f"  {(p['title'] + ' | ' + p['stop'] + ' | ' + p['exit'] + ' | ' + p['version'])[:70]:70s} pos {qt['n_positive']}/{qt['n_quarters_tested']} "
          f"worst {qt['worst_quarter']} {qt['worst_quarter_ret']:+.1f} | {line}")

VERD_Q = []
for s in SETUPS:
    rows = [q for q in QUART if q["setup"] == s]
    if not rows:
        continue
    def tier(q):
        halves = (q["h1_net"] or -9) > 0 and (q["h2_net"] or -9) > 0
        if (q["win_all"] or 0) >= 65 and halves and q["quarter_test"]:
            return 2
        if halves and q["robust"]:
            return 1
        return 0
    best = max(rows, key=lambda q: (tier(q), q["quarter_test"], q["n_positive"], q["win_all"] or 0))
    t = tier(best)
    v = {2: "TRADE", 1: "MAYBE", 0: "DROP"}[t]
    VERD_Q.append({"setup": s, "title": TITLE[s], "verdict": v, "version": best["version"], "stop": best["stop"], "exit": best["exit"],
                   "filter": best["filter"], "gate": best["gate"], "n": best["n"], "win_all": best["win_all"],
                   "h1_net": best["h1_net"], "h2_net": best["h2_net"], "robust": best["robust"],
                   "positive_quarters": f"{best['n_positive']}/{best['n_quarters_tested']}", "worst_quarter": best["worst_quarter"],
                   "worst_quarter_ret": best["worst_quarter_ret"], "losing_quarters": best["losing_quarters"],
                   "idle_quarters": best["idle_quarters"],
                   "losing_quarter_context": {k: QCTX.get(k) for k in best["losing_quarters"]},
                   "why": ("win rate below 65%" if (best["win_all"] or 0) < 65 else "") + ("; fails quarter test" if not best["quarter_test"] else "")
                          + ("; a half is negative" if not ((best["h1_net"] or -9) > 0 and (best["h2_net"] or -9) > 0) else "")
                          + ("; not robust" if not best["robust"] else "")})
print("\nQUARTER-CONSISTENCY VERDICTS")
for v in VERD_Q:
    print(f"  {v['verdict']:6s} {v['title']:30s} {v['version']:14s} win {v['win_all']} pos {v['positive_quarters']} worst {v['worst_quarter']} "
          f"{v['worst_quarter_ret']} lost {v['losing_quarters']} idle {v['idle_quarters']} | {v['stop']} | {v['exit']} | {v['filter']}{' | R3' if v['gate'] else ''} | {v['why'].strip('; ')}")

doc = PREV if QONLY else json.loads(OUT.read_text())
doc["quarterly"] = {"rule": {"first_quarter": Q_FIRST, "min_positive": Q_MIN_POS, "worst_allowed": Q_WORST},
                    "market_context": QCTX, "versions": QUART}
doc["verdicts_quarterly"] = VERD_Q
OUT.write_text(json.dumps(clean(doc), indent=1))
print(f"wrote quarterly section to {OUT} | {time.time()-t0:.0f}s")
