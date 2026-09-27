"""Run every screen across the full history and write the forward-return study."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
import pandas as pd
from screener.rules import load_rules
from screener.backtest import add_panel_columns, add_forward_returns, run, HORIZONS

ap = argparse.ArgumentParser()
ap.add_argument("--rules", default="config/rules.yaml")
ap.add_argument("--features", default="data/screen/features.parquet")
ap.add_argument("--latest", default="data/screen/latest_features.parquet")
ap.add_argument("--out", default="data/screen/backtest.json")
a = ap.parse_args()

t0 = time.time()
feat = pd.read_parquet(a.features)
print(f"loaded {len(feat):,} feature rows, {feat['isin'].nunique()} symbols, "
      f"{feat['date'].min().date()} to {feat['date'].max().date()}", flush=True)

# Cap band is a classification carried back from today. It is a slicing
# dimension, never an input to a rule, so it cannot change whether a screen
# fired - only which bucket the result is reported in.
latest = pd.read_parquet(a.latest)
feat = feat.merge(latest[["isin", "cap_band", "symbol"]].rename(columns={"symbol": "sym"}),
                  on="isin", how="left")

from screener import quality
from screener.store import Store as _Store
_st = _Store("data")
_bars = feat[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
_jumps = quality.detect_price_jumps(_bars, _st.read_adjustments())
feat["contaminated"] = quality.contamination_mask(_bars, _jumps).to_numpy()
print(f"quality gate: {int(feat['contaminated'].sum()):,} of {len(feat):,} rows excluded "
      f"({feat['contaminated'].mean()*100:.2f}%) - windows spanning an unadjusted action", flush=True)
feat = add_panel_columns(feat)
print(f"panel columns in {time.time()-t0:.1f}s", flush=True)
feat = add_forward_returns(feat, HORIZONS)
cov = {k: int(feat[f"fwd_{k}d"].notna().sum()) for k in HORIZONS}
print(f"forward returns: {cov}", flush=True)

rules = load_rules(a.rules)
t1 = time.time()
res = run(feat, rules, HORIZONS)
print(f"backtested {len(rules)} rules over {res['sessions']} sessions in {time.time()-t1:.1f}s\n", flush=True)

rows = sorted(res["rules"], key=lambda r: -(r["horizons"]["20"].get("mean") or -99))
print(f"{'screen':26s} {'hits':>7s} {'/day':>6s} {'20d exc':>8s} {'win%':>6s} {'median':>7s} {'5d exc':>7s}")
print("-" * 72)
for r in rows:
    h20, h5 = r["horizons"]["20"], r["horizons"]["5"]
    if "mean" not in h20:
        print(f"{r['name'][:26]:26s} {r['hits']:>7,d}   (too few observations)")
        continue
    print(f"{r['name'][:26]:26s} {r['hits']:>7,d} {r['per_session']:>6.1f} "
          f"{h20['mean']:>+8.2f} {h20['win']:>6.1f} {h20['median']:>+7.2f} {h5.get('mean', 0):>+7.2f}")

Path(a.out).write_text(json.dumps(res))
print(f"\nwritten to {a.out}")
