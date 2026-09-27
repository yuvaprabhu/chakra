"""Stage 1 base turning up: a beaten-down stock, flat for months, first thrust up.

Signal at close of day t:
  downtrend base : close below the 200-day average; 60-session range <= 20%
  thrust         : close crosses above the 50-day (yesterday below, today above)
                   on volume >= 1.5x its 21-day EMA
Variants add a Weinstein-style 150-day (30-week) cross, a relative-strength
turn (rs_rank up 10+ points in 20 sessions), and a tighter base.
Exits: 20-session hold (excess vs the equal-weight universe) and the 7-day-high
exit used by the swing screens, net of 0.30%. IS < 2025-07-01, OOS after.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener import quality
from screener.store import Store
from screener.backtest import add_panel_columns

SPLIT = pd.Timestamp("2025-07-01"); COST = 0.30; CAP = 20
cols = ["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "sma_50", "sma_150", "sma_200", "sma_150_slope",
        "base_range_60", "vol_vs_ema21", "hi7_prior", "turnover_median_20d", "ret_60d", "ret_120d", "ret_250d", "pct_from_52w_high"]
f = pd.read_parquet("data/screen/features.parquet", columns=cols)
st = Store("data")
bars = f[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close"]].assign(volume=0)
f["contaminated"] = quality.contamination_mask(bars, quality.detect_price_jumps(bars, st.read_adjustments())).to_numpy()
f = add_panel_columns(f).sort_values(["isin", "date"]).reset_index(drop=True)
g = f.groupby("isin"); c = f["adj_close"]
f["cross50"] = (c > f["sma_50"]) & (g["adj_close"].shift(1) <= g["sma_50"].shift(1))
f["cross150"] = (c > f["sma_150"]) & (g["adj_close"].shift(1) <= g["sma_150"].shift(1))
f["rs_turn"] = f["rs_rank"] - g["rs_rank"].shift(20) >= 10
f["ok"] = (~f["contaminated"]) & (f["turnover_median_20d"] >= 1e7) & (f["bars_available"] >= 250)
f["entry"] = g["adj_open"].shift(-1)
f["fwd_20"] = (g["adj_close"].shift(-21) / f["entry"] - 1) * 100
f["exc_20"] = f["fwd_20"] - f.groupby("date")["fwd_20"].transform("mean")
o = f["adj_open"].to_numpy(); cl = c.to_numpy(); hi7 = f["hi7_prior"].to_numpy(); isin = f["isin"].to_numpy(); dates = f["date"].to_numpy()
last = pd.Series(np.arange(len(f))).groupby(pd.factorize(f["isin"])[0]).transform("max").to_numpy()
def d7(idx):
    out = []; busy = {}
    for t in idx:
        if t + 1 > last[t] or busy.get(isin[t], -1) >= t: continue
        e = o[t + 1]; k = t + 1; x = None
        while k <= min(last[t], t + CAP):
            if cl[k] > hi7[k]: x = o[k + 1] if k + 1 <= last[t] else cl[k]; k += 1; break
            k += 1
        if x is None: k = min(k, last[t]); x = cl[k]
        busy[isin[t]] = k; out.append((dates[t], (x / e - 1) * 100 - COST))
    return pd.DataFrame(out, columns=["date", "net"])
def study(name, mask):
    d = f[mask & f["ok"] & f["entry"].notna()]; t = d7(d.index.to_numpy()); out = []
    for lab, w, tw in (("IS", d[d.date < SPLIT], t[t.date < SPLIT]), ("OOS", d[d.date >= SPLIT], t[t.date >= SPLIT])):
        w20 = w.fwd_20.dropna(); n = tw.net
        out.append(f"{lab} n={len(w):5d} hold20 raw {w20.mean():+.2f}% exc {w['exc_20'].mean():+.2f}% win {(w20>COST).mean()*100:3.0f}% lose>8% {(w20<-8).mean()*100:3.0f}% | "
                   f"7d-high exit net {n.mean():+.2f}% win {(n>0).mean()*100:3.0f}%")
    print(f"{name:46s}{out[0]}\n{'':46s}{out[1]}")
base = (c < f["sma_200"]) & (f["base_range_60"] <= 20)
print("BASE TURN BELOW THE 200-DAY (the HDFCLIFE shape) - entry next open\n")
study("A  flat base, crosses 50-day on 1.5x volume", base & f["cross50"] & (f.vol_vs_ema21 >= 1.5))
study("B  A + relative strength turning up", base & f["cross50"] & (f.vol_vs_ema21 >= 1.5) & f["rs_turn"])
study("C  A + base range <= 12%", base & f["cross50"] & (f.vol_vs_ema21 >= 1.5) & (f.base_range_60 <= 12))
study("D  flat base, crosses 150-day (Weinstein)", base & f["cross150"] & (f.vol_vs_ema21 >= 1.5))
study("E  D + 150-day slope flat or rising", base & f["cross150"] & (f.vol_vs_ema21 >= 1.5) & (f.sma_150_slope >= -1))
study("F  A + more than 25% below 52w high", base & f["cross50"] & (f.vol_vs_ema21 >= 1.5) & (f.pct_from_52w_high <= -25))
study("G  same thrust but ABOVE the 200-day (contrast)", (c > f["sma_200"]) & (f.base_range_60 <= 20) & f["cross50"] & (f.vol_vs_ema21 >= 1.5))
study("U  universe reference", f["ok"])
