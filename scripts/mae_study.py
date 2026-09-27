"""Why a 3% stop is too tight: maximum adverse excursion.

A stop is not a risk setting, it is a filter on which trades you are allowed to
win. Every trade that eventually works spends some time under water first; the
depth of that dip is its Maximum Adverse Excursion. If the typical winner dips
4% before it works, a 3% stop does not reduce your losses - it converts your
winners into losses and leaves the losers exactly as they were.

This measures, for real entries on real screens: how deep did the trades that
EVENTUALLY reached the target go against you first, and what fraction of those
winners would each stop width have thrown away.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.rules import load_rules
from screener.backtest import add_panel_columns, add_forward_returns, rule_hits, HORIZONS
from screener import quality
from screener.store import Store

HOLD = 22
st = Store("data")
feat = pd.read_parquet("data/screen/features.parquet")
lat = pd.read_parquet("data/screen/latest_features.parquet")
feat = feat.merge(lat[["isin", "cap_band"]], on="isin", how="left")
bars = feat[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
feat["contaminated"] = quality.contamination_mask(
    bars, quality.detect_price_jumps(bars, st.read_adjustments())).to_numpy()
feat = add_forward_returns(add_panel_columns(feat), HORIZONS)
END = feat["date"].max()
W = feat[feat["date"] >= END - pd.DateOffset(years=2)].copy()

print("1. HOW BIG IS A NORMAL DAY?  ATR(14) as a % of price, at the moment of entry")
a = W["atr_pct"].dropna()
for q in (10, 25, 50, 75, 90):
    print(f"   {q}th percentile: {np.percentile(a, q):.2f}%")
print(f"   share of the universe whose AVERAGE DAY is bigger than 3%: "
      f"{(a > 3).mean()*100:.1f}%\n")

# One pooled set of real entries across every screen, de-duplicated.
rules = load_rules("config/rules.yaml")
m = pd.Series(False, index=W.index)
for r in rules:
    m |= rule_hits(W, r)
d = W.sort_values(["isin", "date"]).reset_index(drop=True)
hit = m.reindex(W.index).fillna(False)
hit = pd.Series(hit.to_numpy(), index=W.index).reindex(W.sort_values(["isin","date"]).index).to_numpy()

o = d["adj_open"].to_numpy(float); h = d["adj_high"].to_numpy(float)
lo = d["adj_low"].to_numpy(float)
codes = pd.factorize(d["isin"])[0]
last = pd.Series(np.arange(len(d))).groupby(codes).transform("max").to_numpy()
idx = np.flatnonzero(hit); idx = idx[idx + 1 <= last[idx]]
entry = o[idx + 1]
win = idx[:, None] + 1 + np.arange(HOLD)[None, :]
valid = win <= last[idx][:, None]
safe = np.where(valid, win, 0)
H = np.where(valid, h[safe], -np.inf)
L = np.where(valid, lo[safe], np.inf)
mfe = (H.max(1) / entry - 1) * 100          # best it ever got
mae = (L.min(1) / entry - 1) * 100          # worst it ever got
ok = np.isfinite(mfe) & np.isfinite(mae) & (entry > 0)
mfe, mae = mfe[ok], mae[ok]
print(f"2. {len(mfe):,} real entries pooled across all {len(rules)} screens, held up to one month\n")

for tgt in (6, 10):
    w = mae[mfe >= tgt]
    print(f"   Of trades that DID reach +{tgt}% at some point ({len(w):,} of {len(mfe):,}, "
          f"{len(w)/len(mfe)*100:.1f}%),")
    print(f"   how far did they fall FIRST?   median {np.median(w):+.2f}%   "
          f"25th pct {np.percentile(w,25):+.2f}%   10th pct {np.percentile(w,10):+.2f}%")
    for sl in (3, 4, 5, 6, 8):
        killed = (w <= -sl).mean() * 100
        print(f"     a {sl}% stop would have thrown away {killed:5.1f}% of these winners")
    print()

print("3. THE TRADE-OFF - a wider stop also means a bigger loss when wrong")
print(f"   {'stop':>5s} {'winners kept':>13s} {'losers cut':>11s} {'expectancy on a 10% target':>28s}")
for sl in (3, 4, 5, 6, 8):
    reached = mfe >= 10
    survive = reached & (mae > -sl)
    stopped = ~survive
    exp = survive.mean() * 10 + stopped.mean() * -sl
    print(f"   {sl:>4d}% {survive.mean()*100:>12.1f}% {stopped.mean()*100:>10.1f}% "
          f"{exp:>+27.2f}%")
