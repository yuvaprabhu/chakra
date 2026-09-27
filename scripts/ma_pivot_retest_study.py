"""Moving-average and yearly-pivot RETESTS as entries, 2018-2026.

The question: does buying a stock when it comes back to its 50-DMA / 50-EMA /
200-DMA, or to a yearly pivot level, and holds there, work - with a hard stop?
Setups (all long, all require turnover >= 1 Cr, >= 250 bars, price >= 10):
  sma50_retest   uptrend (50 > 200, 200 rising); was >= 8% above the 50-DMA in
                 the last 20 sessions; today's low touches the 50-DMA (within 1%)
                 and the close holds above it and above yesterday's close
  ema50_retest   same, on the 50-EMA
  sma200_retest  200-DMA rising, price above it for the prior 60 sessions, today
                 touches it (low within 2%) and closes above it and above
                 yesterday's close
  yr_p_retest    close above the YEARLY pivot P (from last calendar year's
                 H/L/C); a low within the last 3 sessions came within 2% of P and
                 today closes above P and above yesterday's close
  yr_s1_retest   same on yearly S1
  yr_r1_retest   broke above yearly R1 in the last 2-30 sessions, a low in the
                 last 3 sessions came within 2% of R1, today closes above R1 and
                 above yesterday's close (resistance turned support)
Twists: +leader (rs_rank >= 60), +gate (breadth gate on).
Exits: rev7 (next open after a close above the prior 7 closes), close<50EMA
trail (next open), 1:2 target, stop-only; caps 20 / 60.  Stops: 3 ATR, or the
retest bar's low (3-20% below entry).  Entry next open, stop first, gap through
stop fills at the open, 0.30% cost, one open position per name.
Windows: extended 2018-06-01..2026-09-22 (split 2022-06-30) and recent
2024-06-01.. (split 2025-03-19).  Panel = the bottom lab's cached 2017-2026
panel (today's universe: survivorship-biased, like every study here).
"""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from multiprocessing import Pool
from screener.portfolio import simulate

t0 = time.time()
PANEL = "/tmp/claude-0/-home-user/397386f6-c392-5ccc-b9f1-52240babee74/scratchpad/bl/bottom_panel.parquet"
COST = 0.30
feat = pd.read_parquet(PANEL).sort_values(["isin", "date"]).reset_index(drop=True)
print("panel", len(feat), feat.date.min().date(), feat.date.max().date(), f"{time.time()-t0:.0f}s", flush=True)
g = feat.groupby("isin", sort=False)
C, O, H, L = (feat[k].to_numpy(float) for k in ("adj_close", "adj_open", "adj_high", "adj_low"))
prevC = g["adj_close"].shift(1).to_numpy(float)

# ---- yearly pivots from the prior calendar year's high/low/close ----
feat["yr"] = feat.date.dt.year
ya = feat.groupby(["isin", "yr"]).agg(yh=("adj_high", "max"), yl=("adj_low", "min"), yc=("adj_close", "last")).reset_index()
ya["yr"] += 1                                                    # applies to the NEXT year
feat = feat.merge(ya, on=["isin", "yr"], how="left")
feat["y_p"] = (feat.yh + feat.yl + feat.yc) / 3
feat["y_r1"] = 2 * feat.y_p - feat.yl
feat["y_s1"] = 2 * feat.y_p - feat.yh
g = feat.groupby("isin", sort=False)

base = ((feat.turnover_median_20d >= 1e7) & (feat.bars_available >= 250) & (feat.adj_close >= 10) & (~feat.contaminated)).to_numpy()
up_day = (C > prevC)
s50, e50, s200 = feat.sma_50.to_numpy(float), feat.ema_50.to_numpy(float), feat.sma_200.to_numpy(float)
slope200 = feat.sma_200_slope.to_numpy(float)
ext50_prev = g["dist_sma_50"].transform(lambda s: s.shift(1).rolling(20).max()).to_numpy(float)
dist_e50 = (C / e50 - 1) * 100
ext_e50_prev = pd.Series(dist_e50).groupby(feat["isin"].to_numpy()).transform(lambda s: s.shift(1).rolling(20).max()).to_numpy(float)
above200_60 = g["adj_close"].transform(lambda s: s.rolling(60).min()).to_numpy(float)   # min close over last 60 incl today
above200_prev = pd.Series((C > s200).astype(float)).groupby(feat["isin"].to_numpy()).transform(lambda s: s.shift(1).rolling(60).min()).to_numpy(float)

SIG = {}
SIG["sma50_retest"] = base & (s50 > s200) & (slope200 > 0) & (ext50_prev >= 8) & (L <= s50 * 1.01) & (C > s50) & up_day
SIG["ema50_retest"] = base & (e50 > s200) & (slope200 > 0) & (ext_e50_prev >= 8) & (L <= e50 * 1.01) & (C > e50) & up_day
SIG["sma200_retest"] = base & (slope200 > 0) & (above200_prev == 1) & (L <= s200 * 1.02) & (C > s200) & up_day
def near_lvl(lvl, k=3, tol=0.02):
    lo3 = g["adj_low"].transform(lambda s: s.rolling(k).min()).to_numpy(float)
    return (lo3 <= lvl * (1 + tol)) & (lo3 >= lvl * (1 - tol))
yp, yr1, ys1 = feat.y_p.to_numpy(float), feat.y_r1.to_numpy(float), feat.y_s1.to_numpy(float)
SIG["yr_p_retest"] = base & (C > yp) & near_lvl(yp) & up_day
SIG["yr_s1_retest"] = base & (C > ys1) & near_lvl(ys1) & up_day
above_r1 = (C > yr1).astype(float)
cross_r1 = (above_r1 == 1) & (pd.Series(above_r1).groupby(feat["isin"].to_numpy()).shift(1).to_numpy() == 0)
# sessions since the last R1 cross, capped at 40
idx_arr = np.arange(len(feat)); last_cross = pd.Series(np.where(cross_r1, idx_arr, np.nan)).groupby(feat["isin"].to_numpy()).ffill().to_numpy()
since = idx_arr - last_cross
SIG["yr_r1_retest"] = base & (C > yr1) & (since >= 2) & (since <= 30) & near_lvl(yr1) & up_day
RS = feat.rs_rank.to_numpy(float); GATE = feat.mkt_gate_on.to_numpy(float)
for k in list(SIG):
    SIG[k + "+leader"] = SIG[k] & (RS >= 60)
    SIG[k + "+gate"] = SIG[k] & (GATE == 1)

ISIN = feat["isin"].to_numpy(); DATES = feat.date.to_numpy()
LAST = pd.Series(np.arange(len(feat))).groupby(pd.factorize(feat["isin"])[0]).transform("max").to_numpy()
ATRP = feat.atr_pct.to_numpy(float) / 100; HI7 = feat.hi7_prior.to_numpy(float); E50 = e50
CAL = pd.DatetimeIndex(sorted(feat.date.unique())); CALPOS = {d: i for i, d in enumerate(CAL)}
for k, v in SIG.items():
    n_ext = int((v & (DATES >= np.datetime64("2018-06-01"))).sum()); n_rec = int((v & (DATES >= np.datetime64("2024-06-01"))).sum())
    if "+" not in k: print(f"  {k:16s} signals ext {n_ext:6d}  recent {n_rec:5d}", flush=True)

def run(sig, stop, exit_, cap, start):
    rows = []; busy = {}
    for t in np.flatnonzero(sig & (DATES >= np.datetime64(start))):
        if t + 1 > LAST[t] or busy.get(ISIN[t], -1) >= t or not np.isfinite(ATRP[t]): continue
        e = O[t + 1]
        if not np.isfinite(e) or e <= 0: continue
        if stop == "atr3": sl = e * (1 - 3 * ATRP[t])
        else:
            sl = L[t]
            if not (0.03 <= 1 - sl / e <= 0.20): continue
        risk = e - sl; tgt = e + 2 * risk if exit_ == "t2" else np.inf
        end = min(LAST[t], t + cap); k = t + 1; out = None; why = "time"
        while k <= end:
            if L[k] <= sl: out = min(O[k], sl); why = "stop"; break
            if exit_ == "t2" and H[k] >= tgt: out = max(O[k], tgt); why = "target"; break
            if exit_ == "rev7" and np.isfinite(HI7[k]) and C[k] > HI7[k]:
                if k + 1 <= LAST[t]: out = O[k + 1]; k += 1
                else: out = C[k]
                why = "rev7"; break
            if exit_ == "ema50" and np.isfinite(E50[k]) and C[k] < E50[k] and k + 1 <= LAST[t]:
                out = O[k + 1]; k += 1; why = "trail"; break
            k += 1
        if out is None: k = min(k, end); out = C[k]
        if not np.isfinite(out): continue
        busy[ISIN[t]] = k
        rows.append((pd.Timestamp(DATES[t]), ISIN[t], CALPOS[pd.Timestamp(DATES[t + 1])], CALPOS[pd.Timestamp(DATES[k])],
                     (out / e - 1) * 100 - COST, k - t, why, RS[t]))
    return pd.DataFrame(rows, columns=["date", "isin", "entry_i", "exit_i", "net", "bars", "why", "rs"])

WINDOWS = {"ext": ("2018-06-01", "2022-06-30"), "recent": ("2024-06-01", "2025-03-19")}
def cell(job):
    name, stop, exit_, cap, win = job
    start, split = WINDOWS[win]
    t = run(SIG[name], stop, exit_, cap, start)
    def st(x):
        if len(x) < 60: return None
        w = x.net > 0
        return {"n": int(len(x)), "win": round(w.mean() * 100, 1), "net": round(x.net.mean(), 2),
                "pf": round(x.net[w].sum() / max(1e-9, -x.net[~w].sum()), 2), "stopped": round((x.why == "stop").mean() * 100), "bars": round(x.bars.mean(), 1)}
    h1, h2 = t[t.date < split], t[t.date >= split]
    return {"setup": name, "stop": stop, "exit": exit_, "cap": cap, "window": win, "all": st(t), "h1": st(h1), "h2": st(h2)}

jobs = [(n, s, x, c, w) for n in SIG for s in ("atr3", "barlow") for x in ("rev7", "ema50", "t2", "none") for c in (20, 60) for w in WINDOWS
        if not (x in ("ema50", "none") and c == 20) and not (x in ("rev7", "t2") and c == 60)]
print(f"{len(jobs)} cells", flush=True)
with Pool(4) as p: res = p.map(cell, jobs, chunksize=8)
ok = [r for r in res if r["h1"] and r["h2"]]
def both(r): return r["h1"]["net"] > 0 and r["h2"]["net"] > 0
print("\nBEST CELL PER SETUP (ranked by worse-half net; * = both halves positive)")
print(f"{'setup':24s}{'win':>7s}{'stop':>8s}{'exit':>6s}{'cap':>4s} | {'n':>6s}{'win%':>6s}{'net':>7s}{'PF':>6s}{'stp%':>5s} | {'H1 win/net':>13s} {'H2 win/net':>13s}")
best = {}
for w in WINDOWS:
    for n in SIG:
        c = [r for r in ok if r["setup"] == n and r["window"] == w]
        if not c: continue
        r = max(c, key=lambda r: min(r["h1"]["net"], r["h2"]["net"])); best[(n, w)] = r
        a = r["all"]
        print(f"{n:24s}{w:>7s}{r['stop']:>8s}{r['exit']:>6s}{r['cap']:>4d} | {a['n']:>6d}{a['win']:>6.1f}{a['net']:>+7.2f}{a['pf']:>6.2f}{a['stopped']:>5.0f} | "
              f"{r['h1']['win']:>5.1f}/{r['h1']['net']:>+6.2f} {r['h2']['win']:>5.1f}/{r['h2']['net']:>+6.2f} {'*' if both(r) else ''}", flush=True)

# ---- 10-slot account + quarterly for the best both-halves-positive cell per base setup, extended window ----
print("\n10-SLOT ACCOUNT, extended window, best both-halves-positive cell per setup (incl. twists)")
port = {}
for n in SIG:
    c = [r for r in ok if r["setup"] == n and r["window"] == "ext" and both(r)]
    if not c: continue
    r = max(c, key=lambda r: min(r["h1"]["net"], r["h2"]["net"]))
    t = run(SIG[n], r["stop"], r["exit"], r["cap"], "2018-06-01").sort_values(["entry_i", "rs"], ascending=[True, False]).reset_index(drop=True)
    res_ = simulate(t, slots=10, sessions=CAL); eq = res_["equity"]
    q = eq.resample("QE").last().pct_change().dropna() * 100
    qs = {str(k.to_period("Q")): round(float(v), 1) for k, v in q.items()}
    crash = {k: qs.get(k) for k in ("2018Q4", "2020Q1", "2020Q2", "2022Q1", "2025Q1", "2026Q1")}
    port[n] = {"cell": f"{r['stop']}/{r['exit']}/cap{r['cap']}", "n": int(len(t)), "taken": res_["n_taken"], "win": round((t.net > 0).mean() * 100, 1), "net": round(t.net.mean(), 2),
               "cagr": round(res_["cagr"], 1), "maxdd": round(res_["maxdd"], 1), "pos_q": f"{int((q > 0).sum())}/{len(q)}", "worst_q": round(float(q.min()), 1), "crash_q": crash}
    print(f"  {n:24s} {port[n]['cell']:18s} n={port[n]['n']:5d} win {port[n]['win']:5.1f}% net {port[n]['net']:+.2f} | CAGR {port[n]['cagr']:+6.1f} maxDD {port[n]['maxdd']:6.1f} | +q {port[n]['pos_q']:>6s} worst {port[n]['worst_q']:+.1f} | {crash}", flush=True)
json.dump({"cells": res, "best": {f"{k[0]}|{k[1]}": v for k, v in best.items()}, "portfolio": port}, open("data/screen/ma_pivot_retest.json", "w"), default=float)
print(f"\ndone {time.time()-t0:.0f}s")
