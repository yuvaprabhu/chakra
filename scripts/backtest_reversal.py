"""Every screen, exited on the first close above the previous seven closes.

The other two studies cut a trade at a fixed distance (the bracket) or a fixed
date (the hold). Both answer a question, and both destroyed the only edge this
universe actually had. This one lets the trade run to the move it was bought
for. One open position per name, entry at the next open, 0.30% round trip.
Split in-sample before 2025-07-01 and out-of-sample after, because a number
that only works on the half of history it was chosen from is not a number.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
import pandas as pd
from screener.rules import load_rules
from screener.backtest import (add_panel_columns, rule_hits, reversal_trades, reversal_stats,
                               REV_COST, REV_CAP)
from screener import quality
from screener.store import Store

SPLIT = pd.Timestamp("2025-07-01")
t0 = time.time()
feat = pd.read_parquet("data/screen/features.parquet")
latest = pd.read_parquet("data/screen/latest_features.parquet")
feat = feat.merge(latest[["isin", "cap_band", "symbol"]].rename(columns={"symbol": "sym"}),
                  on="isin", how="left")
st = Store("data")
bars = feat[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
feat["contaminated"] = quality.contamination_mask(
    bars, quality.detect_price_jumps(bars, st.read_adjustments())).to_numpy()
feat = add_panel_columns(feat).sort_values(["isin", "date"]).reset_index(drop=True)
print(f"{len(feat):,} rows, {feat['isin'].nunique()} names, {time.time()-t0:.0f}s", flush=True)

rules = load_rules("config/rules.yaml")
out = []
print(f"\n{'screen':28s}{'IS n':>7s}{'net':>7s}{'win':>6s}{'PF':>6s}  |{'OOS n':>7s}{'net':>7s}{'win':>6s}{'PF':>6s}{'>8%':>6s}")
print("-" * 92)
for r in rules:
    hits = rule_hits(feat, r)
    t = reversal_trades(feat, hits.to_numpy())
    if not len(t):
        continue
    s_is = reversal_stats(t[t["date"] < SPLIT])
    s_oos = reversal_stats(t[t["date"] >= SPLIT])
    if s_is.get("n", 0) < 30 or s_oos.get("n", 0) < 30:
        continue
    out.append({"name": r.name, "title": r.title, "is": s_is, "oos": s_oos,
                "all": reversal_stats(t)})
    print(f"{r.title[:28]:28s}{s_is['n']:>7,d}{s_is['net']:>+7.2f}{s_is['win']:>6.0f}{s_is['profit_factor']:>6.2f}  |"
          f"{s_oos['n']:>7,d}{s_oos['net']:>+7.2f}{s_oos['win']:>6.0f}{s_oos['profit_factor']:>6.2f}"
          f"{s_oos['big_loss']:>6.0f}", flush=True)

out.sort(key=lambda r: -min(r["is"]["net"], r["oos"]["net"]))
payload = {"cost": REV_COST, "max_bars": REV_CAP, "split": str(SPLIT.date()),
           "sessions": int(feat["date"].nunique()), "rules": out}
Path("data/screen/reversal.json").write_text(json.dumps(payload))
print(f"\nwritten to data/screen/reversal.json in {time.time()-t0:.0f}s")
print("\nranked by the WORSE of the two periods (the only ranking that survives a holdout):")
for r in out[:8]:
    print(f"  {r['title'][:30]:30s} IS {r['is']['net']:+.2f}%  OOS {r['oos']['net']:+.2f}%  "
          f"win {r['oos']['win']:.0f}%  PF {r['oos']['profit_factor']:.2f}")
