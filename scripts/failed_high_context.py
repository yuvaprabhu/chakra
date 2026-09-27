"""Does a dip after repeated failed attempts at the high (an M / double top) do worse?

For every leader dip signal (rs_rank >= 80, close > SMA200, RSI(2) < 10 or new
7-day closing low), count the separate "pushes" in the last 30 sessions whose
high came within 2% of the 60-day high without a close above it. Also flag a
lower high (latest push high below the earlier push high). Trades use the
standard rule: entry next open, exit at the open after the first close above
the previous 7 closes or after 20 bars, one position per name, 0.30% cost.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener import quality
from screener.store import Store
from screener.backtest import add_panel_columns

SPLIT = pd.Timestamp("2025-07-01"); COST = 0.30; CAP = 20
cols = ["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "sma_200", "rsi_2", "new_lo7", "hi7_prior",
        "turnover_median_20d", "ret_60d", "ret_120d", "ret_250d", "pct_from_52w_high"]
f = pd.read_parquet("data/screen/features.parquet", columns=cols)
st = Store("data")
bars = f[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close"]].assign(volume=0)
f["contaminated"] = quality.contamination_mask(bars, quality.detect_price_jumps(bars, st.read_adjustments())).to_numpy()
f = add_panel_columns(f).sort_values(["isin", "date"]).reset_index(drop=True)
g = f.groupby("isin")
hi60 = g["adj_high"].transform(lambda s: s.shift(1).rolling(60).max())
near = (f["adj_high"] >= 0.98 * hi60) & (f["adj_close"] <= hi60)          # touched the zone, did not close through
# a "push" starts when near turns on after being off
push_start = near & ~near.groupby(f["isin"]).shift(1).fillna(False).astype(bool)
f["pushes_30"] = push_start.astype(int).groupby(f["isin"]).transform(lambda s: s.shift(1).rolling(30).sum())
# lower high: high of the most recent push < high of the previous push (within last 30 sessions)
ph = f["adj_high"].where(push_start)
lastp = ph.groupby(f["isin"]).transform(lambda s: s.shift(1).ffill())
prevp = ph.groupby(f["isin"]).transform(lambda s: s.shift(1).where(push_start.shift(1).fillna(False)).ffill().shift(1).ffill())
f["lower_high"] = (f["pushes_30"] >= 2) & (lastp < prevp)
f["below_high"] = (1 - f["adj_close"] / hi60) * 100
ok = (~f["contaminated"]) & (f["turnover_median_20d"] >= 1e7) & (f["bars_available"] >= 200) & (f["adj_close"] > f["sma_200"]) & (f["rs_rank"] >= 80)
sig = ok & ((f["rsi_2"] < 10) | (f["new_lo7"] == 1))
o = f["adj_open"].to_numpy(); cl = f["adj_close"].to_numpy(); hi7 = f["hi7_prior"].to_numpy()
isin = f["isin"].to_numpy(); dates = f["date"].to_numpy()
last = pd.Series(np.arange(len(f))).groupby(pd.factorize(f["isin"])[0]).transform("max").to_numpy()
rows = []; busy = {}
for t in np.flatnonzero(sig.to_numpy()):
    if t + 1 > last[t] or busy.get(isin[t], -1) >= t: continue
    e = o[t + 1]; k = t + 1; x = None
    while k <= min(last[t], t + CAP):
        if cl[k] > hi7[k]: x = o[k + 1] if k + 1 <= last[t] else cl[k]; k += 1; break
        k += 1
    if x is None: k = min(k, last[t]); x = cl[k]
    busy[isin[t]] = k
    rows.append((dates[t], f["pushes_30"].iat[t], bool(f["lower_high"].iat[t]), f["below_high"].iat[t], (x / e - 1) * 100 - COST))
d = pd.DataFrame(rows, columns=["date", "pushes", "lower_high", "below_high", "net"])
d["ctx"] = np.select([d.pushes >= 2, d.pushes == 1], ["2+ failed pushes", "1 push"], "no push at high")
def line(w):
    n = w.net; win = n[n > 0]; los = n[n <= 0]
    return f"n={len(w):5d} net {n.mean():+.2f}% win {(n>0).mean()*100:3.0f}% PF {win.sum()/max(-los.sum(),1e-9):.2f} lose>8% {(n<-8).mean()*100:3.0f}%"
print("Leader dips (RSI2<10 or 7-day low), by how many times the stock failed at its 60-day high in the prior 30 sessions\n")
for ctx in ["no push at high", "1 push", "2+ failed pushes"]:
    for per, m in (("IS ", d.date < SPLIT), ("OOS", d.date >= SPLIT)):
        print(f"{ctx:18s}{per} {line(d[(d.ctx==ctx)&m])}")
print()
for per, m in (("IS ", d.date < SPLIT), ("OOS", d.date >= SPLIT)):
    print(f"2+ pushes AND lower high  {per} {line(d[d.lower_high&m])}")
    print(f"2+ pushes, no lower high  {per} {line(d[(d.pushes>=2)&~d.lower_high&m])}")
print("\nSame, split by how far below the high the signal close is:")
d["depth"] = pd.cut(d.below_high, [-1, 3, 6, 10, 100], labels=["0-3%", "3-6%", "6-10%", ">10%"])
for dep in ["0-3%", "3-6%", "6-10%", ">10%"]:
    for per, m in (("IS ", d.date < SPLIT), ("OOS", d.date >= SPLIT)):
        print(f"{dep:6s}{per} {line(d[(d.depth==dep)&m])}")
