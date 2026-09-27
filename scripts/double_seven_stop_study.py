"""Where to put the stop on the ranked Double Seven basket.

Takes the top-5-per-day selected trades (composite from
double_seven_selection.py) and re-walks each one bar by bar with a stop
rule. Intraday stops fill at the stop price, or at the open if the day gaps
through it. Close-based stops fill at the next open. Every other rule
(entry next open, exit at the open after a new 7-day closing high, 20-bar
time stop, 0.30% cost) is unchanged. IS before 2025-07-01, OOS after.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scripts.double_seven_selection import load_trades, build_signal_features, composite_score, top_k_per_day, SPLIT

COST = 0.30; K = 5
FEATS = {"rs_rank": 1, "dist_sma_10": -1, "sma_150_slope": 1, "dist_sma_50": 1}
d = build_signal_features(load_trades())
sel = d[top_k_per_day(d, composite_score(d, FEATS), K)].copy()

f = pd.read_parquet("data/screen/features.parquet",
                    columns=["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "ema_50", "ema_21", "atr_pct", "sma_50"])
f = f.sort_values(["isin", "date"]).reset_index(drop=True)
f["pos"] = f.groupby("isin").cumcount()
arr = {k: f[k].to_numpy() for k in ["adj_open", "adj_high", "adj_low", "adj_close", "ema_50", "ema_21", "sma_50", "atr_pct"]}
start = f.groupby("isin")["pos"].transform(lambda s: s.index[0]).to_numpy()   # row index of first bar per isin
row_of = dict(zip(zip(f["isin"], f["date"]), f.index))
sel["row"] = [row_of[(i, dt)] for i, dt in zip(sel["isin"], sel["date"])]

def walk(t, rule, p):
    """Return (net, why) for one selected trade under a stop rule."""
    r0 = int(t.row); e = r0 + 1; xr = r0 + int(t.bars)        # original exit row (open of that bar, or last close)
    entry = arr["adj_open"][e]
    lo_sig = arr["adj_low"][r0]; atr = arr["atr_pct"][r0]
    if rule == "none": stop = -np.inf
    elif rule == "pct": stop = entry * (1 - p / 100)
    elif rule == "atr": stop = entry * (1 - p * atr / 100)
    elif rule == "siglow": stop = lo_sig * (1 - p / 100)
    elif rule in ("ema50", "sma50", "ema21"): stop = None
    for r in range(e, xr):                                    # bars held before the original exit
        if stop is not None:
            if arr["adj_open"][r] <= stop: return (arr["adj_open"][r] / entry - 1) * 100 - COST, "gap"
            if arr["adj_low"][r] <= stop: return (stop / entry - 1) * 100 - COST, "stop"
        else:
            lvl = arr[{"ema50": "ema_50", "sma50": "sma_50", "ema21": "ema_21"}[rule]][r]
            if arr["adj_close"][r] < lvl * (1 - p / 100) and r + 1 <= xr:
                return (arr["adj_open"][r + 1] / entry - 1) * 100 - COST, "close"
    return t.net, t.why

def report(name, rule, p):
    out = []
    for per, m in (("IS", sel.date < SPLIT), ("OOS", sel.date >= SPLIT)):
        w = sel[m]
        res = [walk(t, rule, p) for t in w.itertuples()]
        net = pd.Series([r[0] for r in res], index=w.index); why = pd.Series([r[1] for r in res], index=w.index)
        stopped = why.isin(["stop", "gap", "close"])
        regret = (stopped & (w.net > 0)).sum() / max(stopped.sum(), 1) * 100     # stopped trades that would have won
        win = net[net > 0]; los = net[net <= 0]
        mo = net.groupby(w.date.dt.to_period("M")).mean()
        pf = win.sum() / max(-los.sum(), 1e-9)
        out.append(f"{per:3s} net {net.mean():+.2f}% win {(net>0).mean()*100:3.0f}% avgW {win.mean():+.2f} avgL {los.mean():+.2f} "
                   f"pay {win.mean()/-los.mean():.2f} PF {pf:.2f} | worst {net.min():+.1f} | >8% loss {(net<-8).mean()*100:3.0f}% | "
                   f"stopped {stopped.mean()*100:3.0f}% (of which {regret:3.0f}% were winners) | mo up {(mo>0).mean()*100:.0f}% worst mo {mo.min():+.1f}")
    print(f"{name:34s}{out[0]}\n{'':34s}{out[1]}")

print(f"Ranked Double Seven, top {K} per day, {len(sel)} trades (IS {int((sel.date<SPLIT).sum())}, OOS {int((sel.date>=SPLIT).sum())})\n")
report("no stop (baseline)", "none", 0)
for p in (3, 5, 8, 10, 12, 15): report(f"fixed {p}% below entry", "pct", p)
for p in (1.5, 2, 3): report(f"{p}x ATR(14) below entry", "atr", p)
for p in (0, 1, 2): report(f"{p}% below signal-day low", "siglow", p)
report("close below 21 EMA", "ema21", 0)
report("close below 50 EMA", "ema50", 0)
report("close 2% below 50 EMA", "ema50", 2)
print("\nsignal-day low as % below entry: median", f"{((sel.entry - arr['adj_low'][sel.row.astype(int)]) / sel.entry * 100).median():.1f}%",
      "| ATR% at signal: median", f"{np.median(arr['atr_pct'][sel.row.astype(int)]):.1f}%")
