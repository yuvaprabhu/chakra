"""Run every screen as a real 1:R bracket trade and report what it did."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
import pandas as pd
from screener.rules import load_rules
from screener.backtest import (add_panel_columns, add_forward_returns, rule_hits,
                               bracket_returns, bracket_stats, HORIZONS)

ap = argparse.ArgumentParser()
ap.add_argument("--rules", default="config/rules.yaml")
ap.add_argument("--rr", type=float, default=2.0, help="reward:risk, 2 means 1:2")
ap.add_argument("--max-bars", type=int, default=20)
ap.add_argument("--out", default="data/screen/bracket.json")
a = ap.parse_args()

feat = pd.read_parquet("data/screen/features.parquet")
latest = pd.read_parquet("data/screen/latest_features.parquet")
feat = feat.merge(latest[["isin", "cap_band"]], on="isin", how="left")
from screener import quality
from screener.store import Store as _Store
_st = _Store("data")
_bars = feat[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
_jumps = quality.detect_price_jumps(_bars, _st.read_adjustments())
feat["contaminated"] = quality.contamination_mask(_bars, _jumps).to_numpy()
print(f"quality gate: {int(feat['contaminated'].sum()):,} of {len(feat):,} rows excluded "
      f"({feat['contaminated'].mean()*100:.2f}%) - windows spanning an unadjusted action", flush=True)
feat = add_forward_returns(add_panel_columns(feat), HORIZONS)
rules = load_rules(a.rules)

# Two risk definitions, because they answer different questions. The ATR stop
# adapts to each stock's own volatility - a 1% ATR name and a 6% ATR name
# cannot share a stop. The flat stop is what most people actually type in.
SETUPS = [
    ("atr1", dict(risk="atr", atr_mult=1.0), "1 x ATR(14)"),
    ("atr2", dict(risk="atr", atr_mult=2.0), "2 x ATR(14)"),
    ("pct3", dict(risk="pct", risk_pct=3.0), "flat 3%"),
    ("pct5", dict(risk="pct", risk_pct=5.0), "flat 5%"),
]

out = {"rr": a.rr, "max_bars": a.max_bars,
       "setups": {k: lab for k, _, lab in SETUPS},
       "start": str(feat["date"].min().date()), "end": str(feat["date"].max().date()),
       "rules": []}

t0 = time.time()
for r in rules:
    mask = rule_hits(feat, r)
    row = {"name": r.name, "title": r.title or r.name, "by_setup": {}}
    for key, kw, _lab in SETUPS:
        trades = bracket_returns(feat, mask, rr=a.rr, max_bars=a.max_bars, **kw)
        row["by_setup"][key] = bracket_stats(trades)
    out["rules"].append(row)
print(f"simulated {len(rules)} rules x {len(SETUPS)} stop settings in {time.time()-t0:.1f}s\n")

print(f"1:{a.rr:g} bracket, {a.max_bars}-session time stop, entry at next open, "
      f"stop wins ties, gaps fill at the open")
print(f"\n{'screen':26s} {'trades':>7s} {'stop%':>6s} {'tgt%':>6s} {'time%':>6s} "
      f"{'avg%':>7s} {'exp R':>7s} {'PF':>5s}")
print("-" * 76)
rows = sorted(out["rules"], key=lambda x: -(x["by_setup"]["atr1"].get("expectancy_r") or -9))
for x in rows:
    s = x["by_setup"]["atr1"]
    if not s.get("n"):
        print(f"{x['name'][:26]:26s}       0"); continue
    pf = s["profit_factor"]
    print(f"{x['name'][:26]:26s} {s['n']:>7,d} {s['stop_pct']:>6.1f} {s['target_pct']:>6.1f} "
          f"{s['time_pct']:>6.1f} {s['avg']:>+7.2f} {s['expectancy_r']:>+7.3f} "
          f"{(f'{pf:.2f}' if pf else '  -'):>5s}")

Path(a.out).write_text(json.dumps(out))
print(f"\nwritten to {a.out}")
