"""The same screens, traded as a portfolio instead of as individual trades.

Every bracket test in this project says the screens lose money. Every
hold-to-horizon test says they pick stocks that outperform. Both are computed
from the same signals on the same data, so one of two things is true: either
the hold test is wrong, or the STOP is what destroys the edge.

This settles it. Each month, take the screen's names, hold them equal-weighted
for a month, rebalance. No stop, no target - the only risk control is holding
many names and sizing each one small. That is how a momentum fund is actually
run, and it is the obvious alternative to a stop.

Benchmark is the equal-weight universe over the same months, so what is
reported is outperformance, not the market.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.rules import load_rules
from screener.backtest import add_panel_columns, rule_hits
from screener import quality
from screener.store import Store

TOPN = 20
st = Store("data")
feat = pd.read_parquet("data/screen/features.parquet")
lat = pd.read_parquet("data/screen/latest_features.parquet")
feat = feat.merge(lat[["isin", "cap_band"]], on="isin", how="left")
bars = feat[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
feat["contaminated"] = quality.contamination_mask(
    bars, quality.detect_price_jumps(bars, st.read_adjustments())).to_numpy()
feat = add_panel_columns(feat)

# Forward 1-month return, and the same for the whole universe that month.
d = feat.sort_values(["isin", "date"])
d["fwd"] = (d.groupby("isin")["adj_close"].shift(-22) / d["adj_close"] - 1) * 100
d["bench"] = d.groupby("date")["fwd"].transform("mean")
d["mo"] = d["date"].dt.to_period("M")
# One rebalance date per month: the first session of the month.
first = d.groupby("mo")["date"].transform("min")
REB = d[(d["date"] == first) & d["fwd"].notna()]
print(f"{REB['mo'].nunique()} monthly rebalances, "
      f"{REB['mo'].min()} to {REB['mo'].max()}\n")

rules = load_rules("config/rules.yaml")
print(f"{'screen':26s} {'months':>7s} {'avg held':>9s} {'port %':>8s} {'bench %':>8s} "
      f"{'excess':>8s} {'mo beat':>8s} {'worst':>8s} {'CAGR-ish':>9s}")
print("-" * 100)
rows = []
for r in rules:
    m = rule_hits(REB, r)
    if m.sum() < 60:
        continue
    hits = REB[m].copy()
    port, bench, held = [], [], []
    for mo, g in hits.groupby("mo"):
        if r.sort_by and r.sort_by in g.columns:
            g = g.sort_values(r.sort_by, ascending=bool(r.ascending))
        g = g.head(TOPN)
        if len(g) < 3:
            continue
        port.append(g["fwd"].mean()); bench.append(g["bench"].iloc[0]); held.append(len(g))
    if len(port) < 18:
        continue
    p, b = np.array(port), np.array(bench)
    exc = p - b
    # Compounded, so a big loss costs what it really costs.
    growth = np.prod(1 + p / 100)
    cagr = (growth ** (12 / len(p)) - 1) * 100
    rows.append((r.name, len(p), np.mean(held), p.mean(), b.mean(), exc.mean(),
                 (exc > 0).mean() * 100, p.min(), cagr))
for n, k, h, pm, bm, e, beat, worst, cg in sorted(rows, key=lambda x: -x[5]):
    print(f"{n[:26]:26s} {k:>7d} {h:>9.1f} {pm:>+8.2f} {bm:>+8.2f} {e:>+8.2f} "
          f"{beat:>7.0f}% {worst:>+8.1f} {cg:>+8.1f}%")
