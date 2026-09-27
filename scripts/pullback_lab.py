"""Pullback lab: which way of defining "the dip" works, and with what entry and exit.

Every cell is a full trade list on all ~830 names, one open position per
name, entry at the next open, 0.30% round-trip cost, split IS (< 2025-07-01)
vs OOS. Reported per cell: trades, net %/trade, win %, profit factor,
share of trades losing more than 8%.

DIP definitions (all require close > SMA200; "leaders" adds rs_rank >= 80):
  d7        new 7-day closing low
  rsi2      RSI(2) < 10
  ema9      close below EMA9 after 3+ closes above it
  ema21     low touches EMA21 (low <= EMA21 <= high) with the prior 5 closes above it
  ema50     low touches EMA50 the same way
  cpr       low into the monthly CPR from above, close holds the bottom line
  down3     three consecutive lower closes
  off20     close 5-12% below the 20-day high, above the 50-day
  bb        Bollinger %B < 0.1 (close at or below the lower band)
TRIGGERS:
  now       buy at the next open after the dip day
  rev       wait: buy the next open after the first close above the previous day's high
            (within 5 sessions of the dip)
EXITS:
  hi7       open after the first close above the previous 7 closes, 20-bar cap
  hold10    close 10 sessions after entry
"""
import sys; sys.path.insert(0, ".")
import itertools, numpy as np, pandas as pd
from screener import quality
from screener.store import Store
from screener.backtest import add_panel_columns

SPLIT = pd.Timestamp("2025-07-01"); COST = 0.30; CAP = 20
cols = ["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "sma_200", "ema_9", "ema_21", "ema_50",
        "rsi_2", "m_p", "m_cpr_tc", "m_cpr_bc", "hi_20_prior", "bb_pct_b", "hi7_prior", "new_lo7", "sma_50",
        "turnover_median_20d", "ret_60d", "ret_120d", "ret_250d"]
f = pd.read_parquet("data/screen/features.parquet", columns=cols)
st = Store("data")
bars = f[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close"]].assign(volume=0)
f["contaminated"] = quality.contamination_mask(bars, quality.detect_price_jumps(bars, st.read_adjustments())).to_numpy()
f = add_panel_columns(f).sort_values(["isin", "date"]).reset_index(drop=True)
g = f.groupby("isin")
c, lo, hi = f["adj_close"], f["adj_low"], f["adj_high"]
above = lambda col, n: (c > f[col]).astype(int).groupby(f["isin"]).transform(lambda s: s.shift(1).rolling(n).sum()) == n
ok = (~f["contaminated"]) & (f["turnover_median_20d"] >= 1e7) & (f["bars_available"] >= 200) & (c > f["sma_200"])
DIPS = {
    "d7":    f["new_lo7"] == 1,
    "rsi2":  f["rsi_2"] < 10,
    "ema9":  (c < f["ema_9"]) & above("ema_9", 3),
    "ema21": (lo <= f["ema_21"]) & (hi >= f["ema_21"]) & above("ema_21", 5),
    "ema50": (lo <= f["ema_50"]) & (hi >= f["ema_50"]) & above("ema_50", 5),
    "cpr":   (lo <= f["m_cpr_tc"]) & (c >= f["m_cpr_bc"]) & (g["adj_close"].shift(5) > f["m_cpr_tc"]),
    "down3": (c < g["adj_close"].shift(1)) & (g["adj_close"].shift(1) < g["adj_close"].shift(2)) & (g["adj_close"].shift(2) < g["adj_close"].shift(3)),
    "off20": ((c / f["hi_20_prior"] - 1) * 100).between(-12, -5) & (c > f["sma_50"]),
    "bb":    f["bb_pct_b"] < 0.1,
}
o = f["adj_open"].to_numpy(); cl = c.to_numpy(); hh = hi.to_numpy(); hi7 = f["hi7_prior"].to_numpy()
prev_hi = g["adj_high"].shift(1).to_numpy()
isin = f["isin"].to_numpy(); dates = f["date"].to_numpy()
last = pd.Series(np.arange(len(f))).groupby(pd.factorize(f["isin"])[0]).transform("max").to_numpy()

def trades(sig, trigger, exit_):
    out = []; busy = {}
    for t in np.flatnonzero(sig):
        if busy.get(isin[t], -1) >= t: continue
        s = t
        if trigger == "rev":                      # wait for a close above the prior day's high
            s = None
            for k in range(t + 1, min(t + 6, last[t] + 1)):
                if cl[k] > prev_hi[k]: s = k; break
            if s is None or busy.get(isin[t], -1) >= s: continue
        if s + 1 > last[t]: continue
        e = o[s + 1]; k = s + 1; x = None
        if exit_ == "hi7":
            while k <= min(last[t], s + CAP):
                if cl[k] > hi7[k]: x = o[k + 1] if k + 1 <= last[t] else cl[k]; k += 1; break
                k += 1
            if x is None: k = min(k, last[t]); x = cl[k]
        else:
            k = min(s + 10, last[t]); x = cl[k]
        busy[isin[t]] = k
        out.append((dates[s], (x / e - 1) * 100 - COST))
    return pd.DataFrame(out, columns=["date", "net"])

def stats(w):
    n = w.net; win = n[n > 0]; los = n[n <= 0]
    return dict(n=len(w), net=n.mean(), win=(n > 0).mean() * 100, pf=win.sum() / max(-los.sum(), 1e-9), big=(n < -8).mean() * 100)

rows = []
for uni, umask in (("uptrend", ok), ("leaders", ok & (f["rs_rank"] >= 80))):
    for dip, trig, ex in itertools.product(DIPS, ("now", "rev"), ("hi7", "hold10")):
        tr = trades((DIPS[dip] & umask).to_numpy(), trig, ex)
        a = stats(tr[tr.date < SPLIT]); b = stats(tr[tr.date >= SPLIT])
        rows.append(dict(universe=uni, dip=dip, trigger=trig, exit=ex, n_is=a["n"], net_is=a["net"], win_is=a["win"], pf_is=a["pf"],
                         n_oos=b["n"], net_oos=b["net"], win_oos=b["win"], pf_oos=b["pf"], big_oos=b["big"]))
        print(f"{uni:8s}{dip:6s}{trig:4s}{ex:7s} IS n={a['n']:5d} net {a['net']:+.2f} win {a['win']:3.0f} PF {a['pf']:.2f} | "
              f"OOS n={b['n']:5d} net {b['net']:+.2f} win {b['win']:3.0f} PF {b['pf']:.2f} big {b['big']:.0f}%", flush=True)
R = pd.DataFrame(rows)
R.to_csv("data/screen/pullback_lab.csv", index=False)
R["min_net"] = R[["net_is", "net_oos"]].min(axis=1)
print("\nTOP 15 BY THE WORSE OF THE TWO PERIODS (robust ranking)")
print(R.sort_values("min_net", ascending=False).head(15).round(2).to_string(index=False))
