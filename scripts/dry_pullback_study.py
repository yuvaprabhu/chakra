"""Signal discovered from IOLCP's last three months: the dry pullback after a power day.

IOLCP printed a +14% day on 5x volume (2026-07-07), then three pullbacks
(Jul 15-31, Aug 12-19, Sep 11-15) that all shared the same shape: volume
dried to 0.2-0.5x its 21-day EMA, price held above the 50 EMA, and the
first close back above the 9 EMA started the next leg.

Signal at close of day t (adjusted basis):
  uptrend   : adj_close > sma_200
  anchor    : a "power day" (ret_1d >= 8% on volume >= 3x its 21-EMA) within
              the last 60 sessions, and today's close still above that day's low
  pullback  : close was below the 9 EMA on each of the last 3 sessions (t-1..t-3)
  held      : lowest low of the pullback (last 10 sessions) above the 50 EMA
  dry       : mean volume of the last 5 sessions (t-1..t-5) <= 0.7x the 21-EMA of volume
  trigger   : today closes above the 9 EMA and above yesterday's high
Entry next open; returns to the close k sessions later, raw and in excess of
the equal-weight universe. IS before 2025-07-01, OOS after. Costs (0.30%) not
deducted from the excess figures.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener import quality
from screener.store import Store
from screener.backtest import add_panel_columns

SPLIT = pd.Timestamp("2025-07-01"); COST = 0.30
cols = ["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume", "sma_200",
        "ema_9", "ema_21", "ema_50", "vol_ema_21", "vol_vs_ema21", "close_pos", "ret_1d",
        "turnover_median_20d", "ret_60d", "ret_120d", "ret_250d"]
f = pd.read_parquet("data/screen/features.parquet", columns=cols)
st = Store("data")
bars = f[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
f["contaminated"] = quality.contamination_mask(bars, quality.detect_price_jumps(bars, st.read_adjustments())).to_numpy()
f = add_panel_columns(f).sort_values(["isin", "date"]).reset_index(drop=True)
g = f.groupby("isin")
c = f["adj_close"]
below9 = (c < f["ema_9"]).astype(int)
f["below9_3"] = below9.groupby(f["isin"]).transform(lambda s: s.shift(1).rolling(3).sum()) == 3
f["low10"] = g["adj_low"].transform(lambda s: s.rolling(10).min())
f["held50"] = f["low10"] > f["ema_50"]
f["vol5_prev"] = g["volume"].transform(lambda s: s.shift(1).rolling(5).mean())
f["dry"] = f["vol5_prev"] <= 0.7 * f["vol_ema_21"]
f["trigger"] = (c > f["ema_9"]) & (c > g["adj_high"].shift(1))
f["uptrend"] = c > f["sma_200"]
power = (f["ret_1d"] >= 8) & (f["vol_vs_ema21"] >= 3)
f["power_low"] = f["adj_low"].where(power)
f["power_low"] = g["power_low"].ffill()
pidx = pd.Series(np.where(power, np.arange(len(f)), np.nan)).groupby(f["isin"]).ffill()
f["since_power"] = np.arange(len(f)) - pidx.to_numpy()
f["anchor"] = (f["since_power"] >= 5) & (f["since_power"] <= 60) & (c > f["power_low"])
f["ok"] = (~f["contaminated"]) & (f["turnover_median_20d"] >= 1e7) & (f["bars_available"] >= 200)
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
    print(f"{name:46s} {out[0]}\n{'':46s} {out[1]}")

core = f["uptrend"] & f["below9_3"] & f["held50"] & f["trigger"]
print("DRY PULLBACK AFTER A POWER DAY - entry next open, 0.30% cost not deducted\n")
study("A  pullback below 9EMA, held 50EMA, reclaim", core)
study("B  A + dry volume (5d avg <= 0.7x)", core & f["dry"])
study("C  A + power day in last 60 sessions", core & f["anchor"])
study("D  A + dry + power day (the IOLCP shape)", core & f["dry"] & f["anchor"])
study("E  D + strong close (close_pos >= 0.6)", core & f["dry"] & f["anchor"] & (f.close_pos >= 0.6))
study("F  D + rs_rank >= 80", core & f["dry"] & f["anchor"] & (f.rs_rank >= 80))
study("G  B + rs_rank >= 80 (no power-day needed)", core & f["dry"] & (f.rs_rank >= 80))
study("U  universe, every day (reference)", f["ok"])
lat = pd.read_parquet("data/screen/latest_features.parquet", columns=["isin", "symbol"]).set_index("isin")["symbol"]
io = f[(f["isin"] == "INE485C01029") & (f.date >= "2026-06-18")]
print("\nIOLCP days matching variant D:", [str(x.date()) for x in io[core & f["dry"] & f["anchor"]].date])
print("IOLCP days matching variant B:", [str(x.date()) for x in io[core & f["dry"]].date])
last = f.date.max()
hits = f[(f.date == last) & core & f["dry"] & f["ok"]]
print(f"Variant B fired on {last.date()} for:", ", ".join(sorted(lat.get(i, i) for i in hits["isin"])))
