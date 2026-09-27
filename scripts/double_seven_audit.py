"""Adversarial audit of the Double Seven result before believing it.

1. NON-OVERLAPPING. A stock making a 7-day low on Monday usually makes another
   on Tuesday. Counting both is counting one position twice. Re-run allowing
   ONE open position per stock, which is the only way it could be traded.
2. CONCENTRATION. Is the return carried by a few names or a few months?
3. COSTS. 0.3% per round trip, which is roughly brokerage + STT + slippage on a
   liquid mid-cap. A system that dies here was never alive.
4. FIVE RANDOM MIDCAPS, trade by trade, the same way the failures were shown.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.store import Store
from screener import quality
from screener.backtest import add_panel_columns

SPLIT = pd.Timestamp("2025-07-01"); COST = 0.30; HOLD = 20
st = Store("data")
feat = pd.read_parquet("data/screen/features.parquet")
lat = pd.read_parquet("data/screen/latest_features.parquet")
sym = lat.set_index("isin")["symbol"].to_dict()
bars = feat[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
feat["contaminated"] = quality.contamination_mask(
    bars, quality.detect_price_jumps(bars, st.read_adjustments())).to_numpy()
feat = add_panel_columns(feat).sort_values(["isin", "date"]).reset_index(drop=True)
g = feat.groupby("isin")["adj_close"]
feat["lo7"] = g.transform(lambda s: s.shift(1).rolling(7).min())
feat["hi7"] = g.transform(lambda s: s.shift(1).rolling(7).max())
sig = ((feat["adj_close"] < feat["lo7"]) & (feat["adj_close"] > feat["sma_200"])
       & ~feat["contaminated"] & (feat["turnover_median_20d"] >= 1e7)
       & (feat["bars_available"] >= 200)).to_numpy()
c = feat["adj_close"].to_numpy(); o = feat["adj_open"].to_numpy(); hi7 = feat["hi7"].to_numpy()
isin = feat["isin"].to_numpy(); dates = feat["date"].to_numpy()
codes = pd.factorize(feat["isin"])[0]
last = pd.Series(np.arange(len(feat))).groupby(codes).transform("max").to_numpy()

rows = []; busy_until = {}
for t in np.flatnonzero(sig):
    if t + 1 > last[t]: continue
    if busy_until.get(isin[t], -1) >= t: continue          # one position per name
    e = o[t + 1]; k = t + 1; out = None; why = "time"
    while k <= min(last[t], t + HOLD):
        if c[k] > hi7[k]:
            out = o[k + 1] if k + 1 <= last[t] else c[k]; why = "rev"; k += 1; break
        k += 1
    if out is None: k = min(k, last[t]); out = c[k]
    busy_until[isin[t]] = k
    rows.append((isin[t], dates[t], e, out, (out / e - 1) * 100, k - t, why))
d = pd.DataFrame(rows, columns=["isin", "date", "entry", "exit", "ret", "bars", "why"])
d["net"] = d["ret"] - COST
d["mo"] = pd.to_datetime(d["date"]).dt.to_period("M")

print("DOUBLE SEVEN, one open position per stock at a time\n")
for lab, w in (("IS ", d[d["date"] < SPLIT]), ("OOS", d[d["date"] >= SPLIT])):
    m = w.groupby("mo")["net"].mean()
    wins, loss = w.loc[w.net > 0, "net"].sum(), -w.loc[w.net <= 0, "net"].sum()
    print(f"  {lab}: {len(w):,} trades | gross {w['ret'].mean():+.2f}%  NET of {COST}% cost {w['net'].mean():+.2f}% "
          f"| win {(w.net>0).mean()*100:.0f}% | PF {wins/loss:.2f} | hold {w['bars'].mean():.1f} bars "
          f"| exit on reversal {(w.why=='rev').mean()*100:.0f}% | months up {(m>0).mean()*100:.0f}% "
          f"| worst month {m.min():+.2f}% | trades/month {len(w)/len(m):.0f}")

o_ = d[d["date"] >= SPLIT]
by = o_.groupby("isin")["net"].sum().sort_values(ascending=False)
gross_up = by[by > 0].sum()
print(f"\nCONCENTRATION (OOS): {o_['isin'].nunique()} names traded; "
      f"{(by>0).sum()} net winners; top 10 names = {by.head(10).sum()/gross_up*100:.0f}% of winners' gains")
print(f"  median trade {o_['net'].median():+.2f}%   p10 {o_['net'].quantile(.1):+.2f}%   p90 {o_['net'].quantile(.9):+.2f}%")
by_yr = d.groupby(pd.to_datetime(d["date"]).dt.year)["net"].agg(["size", "mean", lambda s: (s > 0).mean() * 100])
by_yr.columns = ["trades", "avg net %", "win %"]
print("\nBY YEAR:\n" + by_yr.round(2).to_string())

print("\nFIVE RANDOM NAMES FROM THE 50 LARGEST MIDCAP-150 MEMBERS, OOS, every trade:")
mem = st.read_membership(); live = mem[mem["to_date"].isna()]
m150 = set(live[live["index_name"] == "NIFTY MIDCAP 150"]["isin"])
pool = lat[lat["isin"].isin(m150)].nlargest(50, "mcap")
pick = pool.iloc[np.random.default_rng(11).choice(len(pool), 5, replace=False)]
tot = 0
for r in pick.itertuples():
    w = o_[o_["isin"] == r.isin]
    print(f"  {r.symbol:12s} {len(w)} trades, {(w.net>0).sum()} wins, net total {w['net'].sum():+.2f}%")
    for x in w.itertuples():
        print(f"      {str(pd.Timestamp(x.date).date())}  {x.entry:9.2f} -> {x.exit:9.2f}  {x.net:>+6.2f}%  {x.bars:>2d}d  {x.why}")
    tot += w["net"].sum()
print(f"  five names combined, net of costs: {tot:+.2f}%")
d.to_csv("data/screen/double_seven_trades.csv", index=False)
