"""Harden the Leader Dip Playbook: which extra conditions raise accuracy, on 8 years.

The Playbook (rs_rank >= 80, above the 200-DMA, new 7-day closing low or
RSI(2) < 10, close >= 2.6% above the 21-EMA, breadth gate on, buy next open,
3 ATR stop, sell the open after the first close above the prior 7 closes,
20-bar cap) is rebuilt on the 2017-2026 panel, so every candidate filter is
judged on 2018-2026 (halves split 2022-06-30) including the 2020 crash, and
on 2024-2026 separately.  Candidates: market state (Nifty vs its EMAs, VIX,
breadth), stock state (extension, RSI, ATR, beta, cap, turnover, trigger leg,
volume on the dip day, close position), and execution (gap at the open).
A filter is worth keeping only if it raises BOTH win rate and mean net in BOTH
halves while keeping enough trades to matter.
"""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.portfolio import simulate

t0 = time.time()
PANEL = "/tmp/claude-0/-home-user/397386f6-c392-5ccc-b9f1-52240babee74/scratchpad/bl/bottom_panel.parquet"
COST = 0.30
feat = pd.read_parquet(PANEL).sort_values(["isin", "date"]).reset_index(drop=True)
g = feat.groupby("isin", sort=False)
c = feat["adj_close"]
# RSI(2) and the 7-day closing low, which the cached panel lacks
d = g["adj_close"].diff()
up = d.clip(lower=0); dn = (-d).clip(lower=0)
au = up.groupby(feat["isin"]).transform(lambda s: s.ewm(alpha=1/2, adjust=False).mean())
ad = dn.groupby(feat["isin"]).transform(lambda s: s.ewm(alpha=1/2, adjust=False).mean())
feat["rsi_2"] = 100 - 100 / (1 + au / ad.replace(0, np.nan))
feat["lo7_prior"] = g["adj_close"].transform(lambda s: s.shift(1).rolling(7).min())
feat["new_lo7"] = (c < feat["lo7_prior"]).astype(int)
feat["rsi_14"] = feat.get("rsi_14", np.nan)
# beta vs Nifty 50 over 250 sessions
ix = pd.read_parquet("data/raw/indices_daily.parquet")
nf = ix[ix.index_name == "Nifty 50"].set_index("date")["close"].sort_index()
nret = (nf.pct_change() * 100).rename("nret")
feat = feat.merge(nret, left_on="date", right_index=True, how="left")
feat["nifty"] = feat["date"].map(nf)
n_e20 = nf.ewm(span=20, adjust=False).mean(); n_e50 = nf.ewm(span=50, adjust=False).mean(); n_s200 = nf.rolling(200).mean()
feat["nifty_above_e20"] = (feat["date"].map(nf > n_e20)).astype(float)
feat["nifty_above_e50"] = (feat["date"].map(nf > n_e50)).astype(float)
feat["nifty_above_s200"] = (feat["date"].map(nf > n_s200)).astype(float)
feat["nifty_ret_5d"] = feat["date"].map(nf.pct_change(5) * 100)
r = feat["ret_1d"]; m = feat["nret"]
cov = (r * m).groupby(feat["isin"]).transform(lambda s: s.rolling(250, min_periods=200).mean()) - \
      r.groupby(feat["isin"]).transform(lambda s: s.rolling(250, min_periods=200).mean()) * m.groupby(feat["isin"]).transform(lambda s: s.rolling(250, min_periods=200).mean())
var = m.groupby(feat["isin"]).transform(lambda s: s.rolling(250, min_periods=200).var(ddof=0))
feat["beta"] = cov / var
feat["vol_ratio"] = feat.get("vol_ratio", np.nan)
feat["dow"] = feat["date"].dt.dayofweek
print("panel ready", f"{time.time()-t0:.0f}s", flush=True)

base = ((feat.turnover_median_20d >= 1e7) & (feat.bars_available >= 250) & (feat.adj_close >= 10) & (~feat.contaminated)
        & (feat.rs_rank >= 80) & (feat.adj_close > feat.sma_200) & ((feat.new_lo7 == 1) | (feat.rsi_2 < 10))
        & (feat.dist_ema_21 >= 2.6) & (feat.mkt_gate_on == 1)).to_numpy()
O, H, L, C = (feat[k].to_numpy(float) for k in ("adj_open", "adj_high", "adj_low", "adj_close"))
ISIN = feat["isin"].to_numpy(); DATES = feat["date"].to_numpy()
LAST = pd.Series(np.arange(len(feat))).groupby(pd.factorize(feat["isin"])[0]).transform("max").to_numpy()
ATRP = feat.atr_pct.to_numpy(float) / 100; HI7 = feat.hi7_prior.to_numpy(float)
CAL = pd.DatetimeIndex(sorted(feat.date.unique())); CALPOS = {d: i for i, d in enumerate(CAL)}
KEEP = ["rs_rank", "dist_ema_21", "dist_sma_50", "dist_sma_200", "atr_pct", "beta", "turnover_median_20d", "pct_from_52w_high",
        "ret_20d", "ret_60d", "close_pos", "new_lo7", "rsi_2", "mkt_vix", "mkt_vix_pctile_1y", "mkt_breadth_50", "mkt_nifty_range_20d",
        "nifty_above_e20", "nifty_above_e50", "nifty_above_s200", "nifty_ret_5d", "dow"]
F = {k: feat[k].to_numpy(float) for k in KEEP if k in feat}

rows = []; busy = {}
for t in np.flatnonzero(base & (DATES >= np.datetime64("2018-06-01"))):
    if t + 1 > LAST[t] or busy.get(ISIN[t], -1) >= t or not np.isfinite(ATRP[t]): continue
    e = O[t + 1]
    if not np.isfinite(e) or e <= 0: continue
    sl = e * (1 - 3 * ATRP[t]); end = min(LAST[t], t + 20); k = t + 1; out = None; why = "time"
    while k <= end:
        if L[k] <= sl: out = min(O[k], sl); why = "stop"; break
        if np.isfinite(HI7[k]) and C[k] > HI7[k]:
            if k + 1 <= LAST[t]: out = O[k + 1]; k += 1
            else: out = C[k]
            why = "rev7"; break
        k += 1
    if out is None: k = min(k, end); out = C[k]
    if not np.isfinite(out): continue
    busy[ISIN[t]] = k
    rows.append({"date": pd.Timestamp(DATES[t]), "isin": ISIN[t], "entry_i": CALPOS[pd.Timestamp(DATES[t + 1])], "exit_i": CALPOS[pd.Timestamp(DATES[k])],
                 "net": (out / e - 1) * 100 - COST, "bars": k - t, "why": why, "rs": F["rs_rank"][t], "gap_open": (e / C[t] - 1) * 100,
                 **{kk: F[kk][t] for kk in F}})
T = pd.DataFrame(rows); T["q"] = T.date.dt.to_period("Q").astype(str)
print(f"Playbook trades 2018-06..2026-09: {len(T)}  win {(T.net>0).mean()*100:.1f}%  mean {T.net.mean():+.2f}  stopped {(T.why=='stop').mean()*100:.0f}%", flush=True)

def stats(x):
    return dict(n=int(len(x)), win=round((x.net > 0).mean() * 100, 1), mean=round(x.net.mean(), 2)) if len(x) else dict(n=0, win=np.nan, mean=np.nan)
def report(label, mask, split="2022-06-30"):
    a, b = T[mask], T[~mask]
    h1, h2 = a[a.date < split], a[a.date >= split]
    r1, r2 = b[b.date < split], b[b.date >= split]
    ok = (len(h1) >= 80 and len(h2) >= 80 and stats(h1)["win"] > stats(r1)["win"] and stats(h2)["win"] > stats(r2)["win"]
          and stats(h1)["mean"] > stats(r1)["mean"] and stats(h2)["mean"] > stats(r2)["mean"])
    rec = T[T.date >= "2024-06-01"]; rk = rec[mask[T.date >= "2024-06-01"]]
    return dict(filter=label, kept_pct=round(mask.mean() * 100), per_month=round(len(a) / 99, 1), all=stats(a), H1=stats(h1), H2=stats(h2),
                removed_H1=stats(r1), removed_H2=stats(r2), recent=stats(rk), passes=bool(ok))

BASE = stats(T); print("BASE", BASE, "| H1", stats(T[T.date < "2022-06-30"]), "H2", stats(T[T.date >= "2022-06-30"]), "| 2024-26", stats(T[T.date >= "2024-06-01"]))
cands = {}
def q(col, lo=None, hi=None):
    v = T[col]
    return (v >= lo if lo is not None else True) & (v <= hi if hi is not None else True)
for col in ("dist_sma_50", "dist_sma_200", "dist_ema_21", "atr_pct", "beta", "ret_20d", "ret_60d", "pct_from_52w_high", "turnover_median_20d", "close_pos", "mkt_breadth_50", "mkt_vix", "mkt_vix_pctile_1y", "mkt_nifty_range_20d", "nifty_ret_5d", "rs_rank", "gap_open"):
    if col not in T or T[col].isna().all(): continue
    h1v = T.loc[T.date < "2022-06-30", col].dropna()
    for pct in (20, 40, 60, 80):
        thr = float(np.percentile(h1v, pct))
        cands[f"{col} >= {thr:.3g}"] = T[col] >= thr
        cands[f"{col} <= {thr:.3g}"] = T[col] <= thr
cands["Nifty above 20-EMA"] = T.nifty_above_e20 == 1; cands["Nifty below 20-EMA"] = T.nifty_above_e20 == 0
cands["Nifty above 50-EMA"] = T.nifty_above_e50 == 1; cands["Nifty below 50-EMA"] = T.nifty_above_e50 == 0
cands["Nifty above 200-SMA"] = T.nifty_above_s200 == 1; cands["Nifty below 200-SMA"] = T.nifty_above_s200 == 0
cands["trigger = 7-day low"] = T.new_lo7 == 1; cands["trigger = RSI(2)<10 only"] = (T.new_lo7 == 0)
cands["both triggers"] = (T.new_lo7 == 1) & (T.rsi_2 < 10)
cands["no gap-up at open (<= 1%)"] = T.gap_open <= 1.0; cands["no gap-up at open (<= 2%)"] = T.gap_open <= 2.0
cands["not Monday"] = T.dow != 0; cands["not Friday"] = T.dow != 4
cands["beta >= 1 (high beta)"] = T.beta >= 1.0; cands["beta < 1 (low beta)"] = T.beta < 1.0
cands["VIX < 15"] = T.mkt_vix < 15; cands["VIX >= 15"] = T.mkt_vix >= 15
res = [report(k, v) for k, v in cands.items()]
res.sort(key=lambda r: (r["passes"], min(r["H1"]["win"] or 0, r["H2"]["win"] or 0)), reverse=True)
print("\nSINGLE FILTERS THAT PASS BOTH HALVES (win & mean up in both, >= 80 trades each half)")
print(f"{'filter':38s}{'kept':>5s}{'/mo':>5s} | {'all n/win/mean':>18s} | {'H1 win/mean':>12s} {'H2 win/mean':>12s} | {'removed H1/H2 win':>18s} | {'2024-26 win/mean':>16s}")
for r in res:
    if not r["passes"]: continue
    print(f"{r['filter']:38s}{r['kept_pct']:>4d}%{r['per_month']:>5.1f} | {r['all']['n']:>5d}/{r['all']['win']:>5.1f}/{r['all']['mean']:>+5.2f} | "
          f"{r['H1']['win']:>5.1f}/{r['H1']['mean']:>+5.2f} {r['H2']['win']:>5.1f}/{r['H2']['mean']:>+5.2f} | {r['removed_H1']['win']:>7.1f}/{r['removed_H2']['win']:>7.1f} | {r['recent']['win']:>6.1f}/{r['recent']['mean']:>+5.2f}")
print("\nFILTERS THAT FAIL (for the record): Nifty EMA/SMA, VIX, beta, day of week")
for r in res:
    if r["filter"].startswith(("Nifty", "VIX", "beta", "not ", "trigger", "both")):
        print(f"  {r['filter']:34s} kept {r['kept_pct']:>3d}% | H1 {r['H1']['win']:>5.1f}/{r['H1']['mean']:>+5.2f}  H2 {r['H2']['win']:>5.1f}/{r['H2']['mean']:>+5.2f} | removed H1 {r['removed_H1']['win']:>5.1f} H2 {r['removed_H2']['win']:>5.1f} | {'PASS' if r['passes'] else 'fail'}")

# ---- stacks: portfolio + quarterly for base and the top combos ----
def port(mask, label):
    t = T[mask].sort_values(["entry_i", "rs"], ascending=[True, False]).reset_index(drop=True)
    r_ = simulate(t[["date", "isin", "entry_i", "exit_i", "net"]], slots=10, sessions=CAL); eq = r_["equity"]
    qq = eq.resample("QE").last().pct_change().dropna() * 100
    qq.index = qq.index.to_period("Q").astype(str)
    rec = t[t.date >= "2024-06-01"]
    print(f"  {label:44s} n={len(t):5d} win {(t.net>0).mean()*100:5.1f}% mean {t.net.mean():+.2f} | CAGR {r_['cagr']:+6.1f} maxDD {r_['maxdd']:6.1f} | +q {int((qq>0).sum())}/{len(qq)} worst {qq.min():+.1f} | 2020Q1 {qq.get('2020Q1', np.nan):+.1f} 2025Q1 {qq.get('2025Q1', np.nan):+.1f} 2026Q1 {qq.get('2026Q1', np.nan):+.1f} | 2024-26 win {(rec.net>0).mean()*100:.1f}%", flush=True)
    return dict(label=label, n=int(len(t)), win=round((t.net > 0).mean() * 100, 1), mean=round(t.net.mean(), 2), cagr=round(r_["cagr"], 1), maxdd=round(r_["maxdd"], 1),
                pos_q=f"{int((qq>0).sum())}/{len(qq)}", worst_q=round(float(qq.min()), 1), quarters={k: round(float(v), 1) for k, v in qq.items()})
print("\n10-SLOT ACCOUNT 2018-2026 (rs order)")
passing = [r for r in res if r["passes"] and r["kept_pct"] >= 30]
P = [port(np.ones(len(T), bool), "Playbook as shipped")]
for r in passing[:6]:
    P.append(port(cands[r["filter"]], "+ " + r["filter"]))
if len(passing) >= 2:
    for i in range(min(3, len(passing))):
        for j in range(i + 1, min(4, len(passing))):
            P.append(port(cands[passing[i]["filter"]] & cands[passing[j]["filter"]], f"+ {passing[i]['filter']} & {passing[j]['filter']}"))
json.dump({"base": BASE, "filters": res, "portfolio": P}, open("data/screen/harden_playbook.json", "w"), default=float)
print(f"\ndone {time.time()-t0:.0f}s")
