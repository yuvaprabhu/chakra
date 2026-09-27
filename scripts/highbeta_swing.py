"""High-beta swing and rotation over three years, against an absolute hurdle.

The bar is not "positive excess" any more, it is "beats a bond". A bond pays
about 7.5%, so a strategy carrying equity risk has to clear roughly 12% a year
to be worth doing at all. Everything here is therefore reported as an ABSOLUTE
portfolio return, next to what simply holding the same universe would have paid.

Beta is computed as of each date from the trailing 250 sessions against an
equal-weight index of the liquid universe, then shifted a day, so the universe
on any date is chosen from information available that morning.

Two shapes are tested, because they are genuinely different businesses:
  SWING     enter on a screen, hold days to weeks, ten slots recycling
  ROTATION  hold the top N momentum names, rebalance monthly, equal weight
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.rules import load_rules
from screener.backtest import add_panel_columns, rule_hits
from screener.portfolio import simulate
from screener import quality
from screener.store import Store

COST, SLOTS = 0.30, 10
END = pd.Timestamp("2026-09-19")
BETA_WIN, BETA_MIN = 250, 1.20
t0 = time.time()

feat = pd.read_parquet("data/screen/features.parquet")
lat = pd.read_parquet("data/screen/latest_features.parquet")
feat = feat.merge(lat[["isin", "cap_band"]], on="isin", how="left")
st = Store("data")
b = feat[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
feat["contaminated"] = quality.contamination_mask(b, quality.detect_price_jumps(b, st.read_adjustments())).to_numpy()
feat = add_panel_columns(feat).sort_values(["isin", "date"]).reset_index(drop=True)
g = feat.groupby("isin")
feat["ret1"] = g["adj_close"].pct_change()
feat["liquid"] = ((feat["turnover_median_20d"] >= 1e7) & (feat["bars_available"] >= 250)
                  & (~feat["contaminated"]))

CAL = pd.DatetimeIndex(sorted(feat["date"].unique()))
mkt = feat.loc[feat["liquid"]].groupby("date")["ret1"].mean().reindex(CAL).fillna(0)
BENCH = (1 + mkt).cumprod()
feat["mret"] = feat["date"].map(mkt)

# Rolling beta, as of each date: cov(stock, market) / var(market) on the past
# BETA_WIN sessions, then shifted so today's bar is never in today's beta.
def beta_of(s):
    x, y = s["mret"], s["ret1"]
    cov = y.rolling(BETA_WIN, min_periods=120).cov(x)
    var = x.rolling(BETA_WIN, min_periods=120).var()
    return (cov / var).shift(1)
feat["beta"] = feat.groupby("isin", group_keys=False).apply(beta_of)
START = END - pd.DateOffset(years=3)
MID = START + pd.DateOffset(months=18)
IN = ((feat["date"] >= START) & (feat["date"] <= END)).to_numpy()
HIGH = (feat["beta"] >= BETA_MIN).to_numpy() & feat["liquid"].to_numpy()
w = feat[IN]
print(f"panel {len(feat):,} rows | window {START.date()} to {END.date()} | "
      f"liquid names {w.loc[w['liquid'],'isin'].nunique():,} | high-beta (>={BETA_MIN}) names ever: "
      f"{w.loc[(w['beta']>=BETA_MIN)&w['liquid'],'isin'].nunique():,} | {time.time()-t0:.0f}s", flush=True)

O = feat["adj_open"].to_numpy(float); H = feat["adj_high"].to_numpy(float)
L = feat["adj_low"].to_numpy(float);  C = feat["adj_close"].to_numpy(float)
ATRP = feat["atr_pct"].to_numpy(float) / 100.0
E50 = feat["ema_50"].to_numpy(float); HI7 = feat["hi7_prior"].to_numpy(float)
RS = feat["rs_rank"].to_numpy(float)
ISIN = feat["isin"].to_numpy(); DATES = feat["date"].to_numpy()
LAST = pd.Series(np.arange(len(feat))).groupby(pd.factorize(feat["isin"])[0]).transform("max").to_numpy()
CALPOS = {d: i for i, d in enumerate(CAL)}


def swing(sig, spec):
    kind = spec[0]; rows = []; busy = {}
    for t in np.flatnonzero(sig):
        if t + 1 > LAST[t] or busy.get(ISIN[t], -1) >= t or not np.isfinite(ATRP[t]):
            continue
        e = O[t + 1]
        if not np.isfinite(e) or e <= 0:
            continue
        k = t + 1; out = None
        if kind == "hold":
            k = min(t + spec[1], LAST[t]); out = C[k]
        elif kind == "rev":
            cap = min(LAST[t], t + spec[1])
            while k <= cap:
                if np.isfinite(HI7[k]) and C[k] > HI7[k]:
                    out = O[k + 1] if k + 1 <= LAST[t] else C[k]
                    if k + 1 <= LAST[t]:
                        k += 1
                    break
                k += 1
            if out is None:
                k = min(t + spec[1], LAST[t]); out = C[k]
        else:                                     # trailing 50 EMA with an ATR stop
            risk = spec[1] * ATRP[t] * e
            if risk <= 0:
                continue
            stop = e - risk; end = min(LAST[t], t + spec[2])
            while k <= end:
                if L[k] <= stop:
                    out = min(O[k], stop); break
                if np.isfinite(E50[k]) and C[k] < E50[k] and k + 1 <= LAST[t]:
                    out = O[k + 1]; k += 1; break
                k += 1
            if out is None:
                k = min(k, end); out = C[k]
        if not np.isfinite(out):
            continue
        busy[ISIN[t]] = k
        rows.append((pd.Timestamp(DATES[t]), ISIN[t], CALPOS[pd.Timestamp(DATES[t + 1])],
                     CALPOS[pd.Timestamp(DATES[k])], (out / e - 1) * 100 - COST, k - t, RS[t]))
    return pd.DataFrame(rows, columns=["date", "isin", "entry_i", "exit_i", "net", "bars", "rs"])


def rotation(mask, top_n, months=1):
    """Hold the top N by relative strength, rebalanced every month end."""
    d = feat.loc[mask & IN, ["isin", "date", "adj_close", "rs_rank"]].dropna(subset=["rs_rank"])
    if not len(d):
        return None
    ends = d["date"].groupby(d["date"].dt.to_period("M")).max().sort_values()
    ends = ends.iloc[::months]
    px = feat.pivot_table(index="date", columns="isin", values="adj_close")
    eq = 100.0; curve = []
    for i in range(len(ends) - 1):
        a, bdt = ends.iloc[i], ends.iloc[i + 1]
        held = d[d["date"] == a].nlargest(top_n, "rs_rank")["isin"].tolist()
        if not held:
            curve.append((bdt, eq)); continue
        try:
            r = (px.loc[bdt, held] / px.loc[a, held] - 1).dropna()
        except KeyError:
            curve.append((bdt, eq)); continue
        if len(r):
            eq *= (1 + r.mean() - COST / 100)
        curve.append((bdt, eq))
    s = pd.Series(dict(curve)).sort_index()
    if len(s) < 3:
        return None
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return {"total": s.iloc[-1] - 100, "cagr": (s.iloc[-1] / 100) ** (1 / yrs) * 100 - 100,
            "maxdd": float((s / s.cummax() - 1).min() * 100), "curve": s}


def pstats(t, label):
    if t is None or not len(t):
        return None
    p = simulate(t.sort_values(["entry_i", "rs"], ascending=[True, False]), slots=SLOTS, sessions=CAL)
    n = t["net"]
    return {"label": label, "n": int(len(t)), "taken": p["n_taken"], "win": float((n > 0).mean() * 100),
            "net": float(n.mean()), "bars": float(t["bars"].mean()),
            "total": p["total"], "cagr": p["cagr"], "maxdd": p["maxdd"]}


rules = {r.name: r for r in load_rules("config/rules.yaml")}
ENTRIES = ["leader_dip", "rsi2_reversion", "momentum_leaders", "minervini_trend_template",
           "stage2_base_breakout", "smooth_momentum"]
EXITS = [("hold 20d", ("hold", 20)), ("hold 40d", ("hold", 40)), ("hold 60d", ("hold", 60)),
         ("7-day high", ("rev", 20)), ("50 EMA trail, 60d cap", ("trail", 2.5, 60)),
         ("50 EMA trail, 120d cap", ("trail", 2.5, 120))]
res = []
print(f"\nSWING on HIGH-BETA names only ({SLOTS} slots, {COST}% a trade)")
print(f"{'entry':26s}{'exit':23s}{'trades':>7s}{'took':>6s}{'win%':>6s}{'net/tr':>8s}{'bars':>6s}"
      f"{'3yr tot':>9s}{'CAGR':>7s}{'maxDD':>7s}")
print("-" * 105)
for name in ENTRIES:
    if name not in rules:
        continue
    hits = rule_hits(feat, rules[name]).to_numpy() & IN & HIGH
    if hits.sum() < 100:
        continue
    for lab, spec in EXITS:
        s = pstats(swing(hits, spec), f"{rules[name].title} | {lab}")
        if not s:
            continue
        res.append(s)
        print(f"{rules[name].title[:26]:26s}{lab:23s}{s['n']:>7,d}{s['taken']:>6d}{s['win']:>6.0f}"
              f"{s['net']:>+8.2f}{s['bars']:>6.0f}{s['total']:>+9.1f}{s['cagr']:>+7.1f}{s['maxdd']:>7.1f}",
              flush=True)

print(f"\nMONTHLY ROTATION, equal weight, rebalanced month end")
print(f"{'universe':26s}{'hold':>6s}{'3yr tot':>10s}{'CAGR':>8s}{'maxDD':>8s}")
print("-" * 60)
rot = []
for uni, mask in (("high beta only", HIGH), ("all liquid", feat["liquid"].to_numpy())):
    for n in (10, 15, 25):
        r = rotation(mask, n)
        if r:
            rot.append({"uni": uni, "n": n, **{k: v for k, v in r.items() if k != "curve"}})
            print(f"{uni:26s}{n:>6d}{r['total']:>+10.1f}{r['cagr']:>+8.1f}{r['maxdd']:>8.1f}", flush=True)

bh = BENCH.loc[START:END]
yrs = (bh.index[-1] - bh.index[0]).days / 365.25
print(f"\n{'BENCHMARK equal-weight liquid universe, buy and hold':52s}"
      f"{(bh.iloc[-1]/bh.iloc[0]-1)*100:>+10.1f}{((bh.iloc[-1]/bh.iloc[0])**(1/yrs)-1)*100:>+8.1f}"
      f"{float((bh/bh.cummax()-1).min()*100):>8.1f}")
Path("data/screen/highbeta.json").write_text(json.dumps({"swing": res, "rotation": rot}, default=float))
best = sorted([r for r in res if r["cagr"] > 12], key=lambda r: -r["cagr"])[:8]
print(f"\n{len(best)} swing combination(s) cleared 12% a year:")
for r in best:
    print(f"  {r['label'][:52]:52s} CAGR {r['cagr']:+.1f}%  maxDD {r['maxdd']:.1f}%  {r['taken']} trades")
print(f"\ndone in {time.time()-t0:.0f}s")
