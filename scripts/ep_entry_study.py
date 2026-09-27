"""Episodic Pivot: does buying the gap work, or is waiting for a retest better?

The objection this answers: by the time the screen fires the stock is already
up 10%, so where is the entry and where is the risk? A base breakout gives you
a structure - buy the pivot, stop under the base. A gap seems to give you
nothing but a chase.

It does give you one structure: the GAP DAY'S LOW. Everyone who bought the
news is above it, so losing it means the news is being sold. All three models
below use that as the stop, so they differ only in WHERE you get in - which is
the whole question.

  A  Buy the next open.                     (what the harness does today)
  B  Wait for a retest of the gap day's close, enter if it holds.
  C  Wait for it to hold above the gap low for 3 sessions, then buy a break
     of the gap day's high.

Same target rule for all three: 2R from the entry, where R is the distance to
the gap day's low. Same tie-breaking as the main backtest - a bar that touches
both the stop and the target is scored as a stop, and gaps fill at the open.

B and C do not always trigger. The fill rate is reported, because a model that
only trades the good cases by never filling on the bad ones is not better, it
is just smaller.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.rules import load_rules
from screener.backtest import add_panel_columns, rule_hits
from screener.store import Store
from screener import quality

st = Store("data")
feat = pd.read_parquet("data/screen/features.parquet")
lat = pd.read_parquet("data/screen/latest_features.parquet")
feat = feat.merge(lat[["isin", "symbol"]].rename(columns={"symbol": "sym"}), on="isin", how="left")
bars = feat[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
feat["contaminated"] = quality.contamination_mask(
    bars, quality.detect_price_jumps(bars, st.read_adjustments())).to_numpy()
feat = add_panel_columns(feat)
rules = {r.name: r for r in load_rules("config/rules.yaml")}

MAXWAIT, MAXHOLD = 10, 25

def walk(g, i, entry_px, stop, target, start, maxhold):
    """Score one trade bar by bar from ``start``. Stop wins ties; gaps fill at the open."""
    for k in range(start, min(start + maxhold, len(g))):
        o, h, l = g.adj_open[k], g.adj_high[k], g.adj_low[k]
        if l <= stop:
            return (min(stop, o) / entry_px - 1) * 100, "stop", k - start + 1
        if h >= target:
            return (max(target, o) / entry_px - 1) * 100, "target", k - start + 1
    k = min(start + maxhold, len(g)) - 1
    return (g.adj_close[k] / entry_px - 1) * 100, "time", k - start + 1

rows = []
mask = rule_hits(feat, rules["episodic_pivot"])
panel = {i: g.sort_values("date").reset_index(drop=True) for i, g in feat.groupby("isin")}

for isin, date in zip(feat.loc[mask, "isin"], feat.loc[mask, "date"]):
    g = panel[isin]
    idx = g.index[g["date"] == date]
    if not len(idx):
        continue
    t = int(idx[0])
    if t + 2 >= len(g):
        continue
    G = g.itertuples()
    gl, gh, gc = g.adj_low[t], g.adj_high[t], g.adj_close[t]
    gap = (g.adj_open[t] / g.adj_close[t - 1] - 1) * 100 if t else np.nan

    # --- A: buy the next open ---------------------------------------------
    e = g.adj_open[t + 1]
    if e > gl:
        r = (e - gl) / e * 100
        ret, out, held = walk(g, t, e, gl, e + 2 * (e - gl), t + 1, MAXHOLD)
        rows.append(dict(model="A next open", sym=g.sym[t], date=date, gap=gap,
                         risk=r, ret=ret, R=ret / r, outcome=out, bars=held, filled=1))

    # --- B: wait for a retest of the gap day's close ------------------------
    filled = False
    for k in range(t + 1, min(t + 1 + MAXWAIT, len(g))):
        if g.adj_low[k] <= gc:                       # came back to the gap close
            if g.adj_close[k] <= gl:                 # and broke the gap low: no trade
                break
            e = min(gc, g.adj_open[k])
            if e <= gl:
                break
            r = (e - gl) / e * 100
            ret, out, held = walk(g, t, e, gl, e + 2 * (e - gl), k + 1, MAXHOLD)
            rows.append(dict(model="B retest", sym=g.sym[t], date=date, gap=gap,
                             risk=r, ret=ret, R=ret / r, outcome=out, bars=held, filled=1))
            filled = True
            break
    if not filled:
        rows.append(dict(model="B retest", sym=g.sym[t], date=date, gap=gap,
                         risk=np.nan, ret=np.nan, R=np.nan, outcome="no fill", bars=0, filled=0))

    # --- C: hold the gap low for 3 sessions, then buy the gap high ---------
    filled = False
    held_ok = 0
    for k in range(t + 1, min(t + 1 + MAXWAIT, len(g))):
        if g.adj_low[k] < gl:
            break                                    # gave up the gap low
        held_ok += 1
        if held_ok >= 3 and g.adj_high[k] > gh:
            e = max(gh, g.adj_open[k])
            r = (e - gl) / e * 100
            ret, out, hb = walk(g, t, e, gl, e + 2 * (e - gl), k + 1, MAXHOLD)
            rows.append(dict(model="C break", sym=g.sym[t], date=date, gap=gap,
                             risk=r, ret=ret, R=ret / r, outcome=out, bars=hb, filled=1))
            filled = True
            break
    if not filled:
        rows.append(dict(model="C break", sym=g.sym[t], date=date, gap=gap,
                         risk=np.nan, ret=np.nan, R=np.nan, outcome="no fill", bars=0, filled=0))

t = pd.DataFrame(rows)
n_sig = t[t["model"] == "A next open"].shape[0]
print(f"Episodic Pivot: {n_sig} signals, stop = the gap day's low, target = 2R\n")
print(f"{'model':14s} {'fills':>7s} {'fill%':>6s} {'avg risk':>9s} {'stop%':>6s} {'tgt%':>6s} "
      f"{'avg R':>7s} {'exp R':>7s} {'PF':>6s}")
print("-" * 80)
for m in ("A next open", "B retest", "C break"):
    s = t[t["model"] == m]
    f = s[s["filled"] == 1]
    if not len(f):
        continue
    wins = f.loc[f["ret"] > 0, "ret"].sum()
    loss = -f.loc[f["ret"] <= 0, "ret"].sum()
    oc = f["outcome"].value_counts()
    print(f"{m:14s} {len(f):>7d} {len(f)/n_sig*100:>5.0f}% {f['risk'].mean():>8.2f}% "
          f"{oc.get('stop',0)/len(f)*100:>5.1f} {oc.get('target',0)/len(f)*100:>5.1f} "
          f"{f['R'].mean():>+7.3f} {f['R'].sum()/n_sig:>+7.3f} "
          f"{(wins/loss if loss else float('nan')):>6.2f}")
print("\n  exp R is per SIGNAL, not per fill - a model that skips trades must be")
print("  credited with the zeroes, or waiting always looks better than acting.\n")
print("Gap size vs what happened next (model A, buying the open):")
a = t[(t["model"] == "A next open") & t["ret"].notna()].copy()
a["band"] = pd.cut(a["gap"], [4, 6, 8, 12, 100], labels=["4-6%", "6-8%", "8-12%", ">12%"])
print(a.groupby("band", observed=True).agg(n=("R", "size"), avg_R=("R", "mean"),
      risk=("risk", "mean"), stopped=("outcome", lambda s: (s == "stop").mean() * 100)).round(2).to_string())
t.to_csv("data/screen/ep_entry_study.csv", index=False)
