"""Backtest of the "correction, then breakout above every EMA on volume" signal.

Signal at the close of day t (all on the adjusted basis):
  uptrend    : adj_close > sma_200 and sma_200 rising (20-session slope > 0)
  correction : close was below the 21 EMA on day t-1 (and, in the strict
               variant, on at least 3 of the last 5 sessions)
  reclaim    : close on day t is above EMA 9, EMA 21 and EMA 50 at once
  volume     : today's volume >= V x the 21-EMA of volume
  strength   : close in the top 40% of the day's range (close_pos >= 0.6)
Entry is next open. Returns are next-open to close k sessions later, quoted
raw and in excess of the equal-weight universe. In-sample before 2025-07-01,
out-of-sample after. Costs are not deducted from the excess figures; subtract
0.30% per trade for a round trip.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener import quality
from screener.store import Store
from screener.backtest import add_panel_columns

SPLIT = pd.Timestamp("2025-07-01"); COST = 0.30
cols = ["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume", "sma_200",
        "sma_200_slope", "ema_9", "ema_21", "ema_50", "vol_vs_ema21", "close_pos",
        "turnover_median_20d", "ret_60d", "ret_120d", "ret_250d"]
f = pd.read_parquet("data/screen/features.parquet", columns=[c for c in cols])
st = Store("data")
bars = f[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
f["contaminated"] = quality.contamination_mask(bars, quality.detect_price_jumps(bars, st.read_adjustments())).to_numpy()
f = add_panel_columns(f).sort_values(["isin", "date"]).reset_index(drop=True)
g = f.groupby("isin")
c = f["adj_close"]
below21 = (c < f["ema_21"]).astype(int)
f["below21_prev"] = g["adj_close"].shift(1) < g["ema_21"].shift(1)
f["below21_5"] = below21.groupby(f["isin"]).transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
f["above_all"] = (c > f["ema_9"]) & (c > f["ema_21"]) & (c > f["ema_50"])
f["uptrend"] = (c > f["sma_200"]) & (f["sma_200_slope"] > 0)
f["ok"] = (~f["contaminated"]) & (f["turnover_median_20d"] >= 1e7) & (f["bars_available"] >= 200)
# next-open entry, forward closes
f["entry"] = g["adj_open"].shift(-1)
for k in (5, 10, 20):
    f[f"fwd_{k}"] = (g["adj_close"].shift(-(k + 1)) / f["entry"] - 1) * 100
    f[f"exc_{k}"] = f[f"fwd_{k}"] - f.groupby("date")[f"fwd_{k}"].transform("mean")

def study(name, mask):
    d = f[mask & f["ok"] & f["entry"].notna()]
    out = []
    for lab, w in (("IS", d[d.date < SPLIT]), ("OOS", d[d.date >= SPLIT])):
        w20 = w["fwd_20"].dropna()
        out.append(f"{lab} n={len(w):5d} ({len(w)/max(w.date.nunique(),1):.1f}/day) "
                   f"raw20 {w20.mean():+.2f}% win {(w20 > COST).mean()*100:3.0f}% | exc 5d {w['exc_5'].mean():+.2f} "
                   f"10d {w['exc_10'].mean():+.2f} 20d {w['exc_20'].mean():+.2f} | lose>8% {(w20 < -8).mean()*100:.0f}%")
    print(f"{name:44s} {out[0]}\n{'':44s} {out[1]}")
    return d

base = f["uptrend"] & f["below21_prev"] & f["above_all"]
print("EMA RECLAIM ON VOLUME - entry next open, 0.30% cost not deducted\n")
study("A  reclaim, any volume", base)
study("B  A + vol >= 1.0x ema21", base & (f.vol_vs_ema21 >= 1.0))
study("C  A + vol >= 1.5x", base & (f.vol_vs_ema21 >= 1.5))
study("D  A + vol >= 2.0x", base & (f.vol_vs_ema21 >= 2.0))
study("E  C + strong close (close_pos>=0.6)", base & (f.vol_vs_ema21 >= 1.5) & (f.close_pos >= 0.6))
study("F  E + 3 of last 5 closes below 21EMA", base & (f.vol_vs_ema21 >= 1.5) & (f.close_pos >= 0.6) & (f.below21_5 >= 3))
study("G  F + 21EMA above 50EMA (stacked)", base & (f.vol_vs_ema21 >= 1.5) & (f.close_pos >= 0.6) & (f.below21_5 >= 3) & (f.ema_21 > f.ema_50))
study("H  F + rs_rank >= 70 (leaders only)", base & (f.vol_vs_ema21 >= 1.5) & (f.close_pos >= 0.6) & (f.below21_5 >= 3) & (f.rs_rank >= 70))
study("U  universe, every day (reference)", f["ok"])
# today's hits for variant F
lat = pd.read_parquet("data/screen/latest_features.parquet", columns=["isin", "symbol"]).set_index("isin")["symbol"]
last = f.date.max()
hits = f[(f.date == last) & base & (f.vol_vs_ema21 >= 1.5) & (f.close_pos >= 0.6) & (f.below21_5 >= 3) & f.ok]
names = ', '.join(sorted(lat.get(i, i) for i in hits['isin']))
print(f"\nVariant F fired on {last.date()} for: {names}")
sjs = f[(f["isin"] == "INE284S01014") & (f.date == last)].iloc[0]
print(f"SJS today: prev below 21EMA={sjs.below21_prev} closes below in last5={sjs.below21_5:.0f} above all EMAs={sjs.above_all} "
      f"vol/ema21={sjs.vol_vs_ema21:.2f} close_pos={sjs.close_pos:.2f} uptrend={sjs.uptrend}")
