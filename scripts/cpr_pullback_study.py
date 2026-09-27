"""Uptrend stock pulls back to the monthly CPR and holds it.

Monthly CPR: pivot P = (H+L+C)/3 of the previous month, TC/BC the top and
bottom central lines. "Mid" is P. Signal at the close of day t:
  uptrend  : adj_close > sma_200 and sma_200 rising (20-session slope > 0)
  from above: the close 5 sessions ago was above TC (price came down to the CPR)
  touch    : today's low <= TC (into the zone); "mid" variant: low <= P
  hold     : today's close >= BC (did not close below the zone);
             "hold mid" variant: close >= P
Entry next open; returns to the close k sessions later, raw and in excess of
the equal-weight universe; IS before 2025-07-01, OOS after; 0.30% cost not
deducted from the excess figures. Also shows the Double-Seven-style exit
(open after the first close above the previous 7 closes, 20-bar cap) for
the main variants, net of cost.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener import quality
from screener.store import Store
from screener.backtest import add_panel_columns

SPLIT = pd.Timestamp("2025-07-01"); COST = 0.30; HOLD = 20
cols = ["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume", "sma_200", "sma_200_slope",
        "m_p", "m_cpr_tc", "m_cpr_bc", "m_cpr_width", "m_cpr_width_rank", "close_pos", "vol_vs_ema21", "ema_50",
        "turnover_median_20d", "ret_60d", "ret_120d", "ret_250d", "hi7_prior"]
f = pd.read_parquet("data/screen/features.parquet", columns=cols)
st = Store("data")
bars = f[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
f["contaminated"] = quality.contamination_mask(bars, quality.detect_price_jumps(bars, st.read_adjustments())).to_numpy()
f = add_panel_columns(f).sort_values(["isin", "date"]).reset_index(drop=True)
g = f.groupby("isin")
c, lo = f["adj_close"], f["adj_low"]
f["uptrend"] = (c > f["sma_200"]) & (f["sma_200_slope"] > 0)
f["from_above"] = g["adj_close"].shift(5) > f["m_cpr_tc"]
f["touch_zone"] = lo <= f["m_cpr_tc"]
f["touch_mid"] = lo <= f["m_p"]
f["hold_bc"] = c >= f["m_cpr_bc"]
f["hold_mid"] = c >= f["m_p"]
f["month"] = f["date"].dt.to_period("M")
first = f["touch_zone"] & f["uptrend"]
f["first_touch"] = first & (first.groupby([f["isin"], f["month"]]).cumsum() == 1)
f["rev"] = (c > f["adj_open"]) & (f["close_pos"] >= 0.5)
f["ok"] = (~f["contaminated"]) & (f["turnover_median_20d"] >= 1e7) & (f["bars_available"] >= 200) & f["m_p"].notna()
f["entry"] = g["adj_open"].shift(-1)
for k in (5, 10, 20):
    f[f"fwd_{k}"] = (g["adj_close"].shift(-(k + 1)) / f["entry"] - 1) * 100
    f[f"exc_{k}"] = f[f"fwd_{k}"] - f.groupby("date")[f"fwd_{k}"].transform("mean")

o = f["adj_open"].to_numpy(); cl = c.to_numpy(); hi7 = f["hi7_prior"].to_numpy()
last = pd.Series(np.arange(len(f))).groupby(pd.factorize(f["isin"])[0]).transform("max").to_numpy()
def d7_exit(idx):
    """Net return with the Double Seven exit, one position per name."""
    out = []; busy = {}
    isin = f["isin"].to_numpy()
    for t in idx:
        if t + 1 > last[t] or busy.get(isin[t], -1) >= t: continue
        e = o[t + 1]; k = t + 1; x = None
        while k <= min(last[t], t + HOLD):
            if cl[k] > hi7[k]: x = o[k + 1] if k + 1 <= last[t] else cl[k]; k += 1; break
            k += 1
        if x is None: k = min(k, last[t]); x = cl[k]
        busy[isin[t]] = k; out.append((f["date"].iloc[t], (x / e - 1) * 100 - COST))
    return pd.DataFrame(out, columns=["date", "net"])

def study(name, mask, d7=False):
    d = f[mask & f["ok"] & f["entry"].notna()]
    out = []
    for lab, w in (("IS", d[d.date < SPLIT]), ("OOS", d[d.date >= SPLIT])):
        w20 = w["fwd_20"].dropna()
        out.append(f"{lab} n={len(w):5d} ({len(w)/max(w.date.nunique(),1):.1f}/day) raw20 {w20.mean():+.2f}% win {(w20 > COST).mean()*100:3.0f}% | "
                   f"exc 5d {w['exc_5'].mean():+.2f} 10d {w['exc_10'].mean():+.2f} 20d {w['exc_20'].mean():+.2f} | lose>8% {(w20 < -8).mean()*100:.0f}%")
    print(f"{name:40s} {out[0]}\n{'':40s} {out[1]}")
    if d7:
        t = d7_exit(d.index.to_numpy())
        for lab, w in (("IS", t[t.date < SPLIT]), ("OOS", t[t.date >= SPLIT])):
            wn = w.net; win = wn[wn > 0]; los = wn[wn <= 0]
            print(f"{'':40s} {lab} D7-exit: n={len(w):5d} net {wn.mean():+.2f}% win {(wn>0).mean()*100:3.0f}% avgW {win.mean():+.2f} avgL {los.mean():+.2f} PF {win.sum()/max(-los.sum(),1e-9):.2f} lose>8% {(wn<-8).mean()*100:.0f}%")

base = f["uptrend"] & f["from_above"] & f["touch_zone"] & f["hold_bc"]
print("UPTREND PULLBACK TO THE MONTHLY CPR - entry next open\n")
study("A  into CPR zone from above, holds BC", base, d7=True)
study("B  A + touched the mid (P)", base & f["touch_mid"])
study("C  A + closes back above the mid", base & f["touch_mid"] & f["hold_mid"], d7=True)
study("D  A + first touch this month", base & f["first_touch"])
study("E  C + up close (close > open, upper half)", base & f["touch_mid"] & f["hold_mid"] & f["rev"])
study("F  C + rs_rank >= 80", base & f["touch_mid"] & f["hold_mid"] & (f.rs_rank >= 80), d7=True)
study("G  C + narrow CPR (width rank <= 30%)", base & f["touch_mid"] & f["hold_mid"] & (f.m_cpr_width_rank <= 0.3))
study("H  C + wide CPR (width rank >= 70%)", base & f["touch_mid"] & f["hold_mid"] & (f.m_cpr_width_rank >= 0.7))
study("I  C + dry volume (< 0.8x 21-EMA)", base & f["touch_mid"] & f["hold_mid"] & (f.vol_vs_ema21 < 0.8))
study("J  C + above 50 EMA too", base & f["touch_mid"] & f["hold_mid"] & (c > f.ema_50))
study("U  universe, every day (reference)", f["ok"])
lat = pd.read_parquet("data/screen/latest_features.parquet", columns=["isin", "symbol"]).set_index("isin")["symbol"]
lastd = f.date.max()
hits = f[(f.date == lastd) & base & f["touch_mid"] & f["hold_mid"] & f["ok"]]
print(f"\nVariant C fired on {lastd.date()} for:", ", ".join(sorted(lat.get(i, i) for i in hits["isin"])))
