"""Lookahead audit: does any feature at date t use data from after t?

This is the failure mode neither other check can see. audit_screens.py proves a
hit satisfies its rule; validate_definitions.py proves the rule matches its
source. Both would pass cleanly on a feature that quietly peeks at tomorrow -
and such a feature makes every backtest number in the app fiction, while the
live screen still looks fine because there is no tomorrow yet.

The test is mechanical and leaves nowhere to hide: recompute the whole feature
stack from bars TRUNCATED at date T, and compare the row at T against the same
row computed from the full history. A causal indicator cannot notice the
difference. Anything that moves, peeks.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.store import Store
from screener.adjust import derive_raw_all
from screener.indicators import build_features

st = Store("data")
latest = pd.read_parquet("data/screen/latest_features.parquet")
rng = np.random.default_rng(7)
isins = list(rng.choice(sorted(latest["isin"].unique()), size=40, replace=False))

bars = st.read_bars(start=pd.Timestamp("2023-06-01"), isins=isins)
bars = derive_raw_all(bars, st.read_adjustments())
full = build_features(bars)

sessions = sorted(full["date"].unique())
cuts = [sessions[-1], sessions[-25], sessions[-90], sessions[-260]]
num = [c for c in full.columns
       if full[c].dtype.kind in "fi" and c not in ("volume", "trades")]

print(f"Causality audit: {len(isins)} symbols x {len(cuts)} cut dates x {len(num)} numeric features")
print("A feature is causal if truncating the future does not change its value today.\n")

bad = {}
for cut in cuts:
    trunc = build_features(bars[bars["date"] <= cut])
    a = full[full["date"] == cut].set_index("isin")[num].sort_index()
    b = trunc[trunc["date"] == cut].set_index("isin")[num].sort_index()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    diff = ~np.isclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-9, atol=1e-9,
                       equal_nan=True)
    per_col = pd.Series(diff.sum(0), index=num)
    hits = per_col[per_col > 0]
    tag = str(pd.Timestamp(cut).date())
    print(f"  cut at {tag}: {len(common)} symbols, "
          f"{'CLEAN' if hits.empty else str(len(hits)) + ' LEAKING FEATURES'}")
    for c, n in hits.items():
        bad.setdefault(c, []).append((tag, int(n)))

print()
if not bad:
    print("PASS - every feature is causal at every cut date.")
else:
    print("FAIL - these features change when the future is removed:")
    for c, occ in sorted(bad.items()):
        print(f"  {c:24s} {occ}")
sys.exit(1 if bad else 0)
