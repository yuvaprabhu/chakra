"""The winning list: every screen scored on data it was never designed against.

Every number I have reported until now came from the full history - the same
period the screens were written while looking at. That is not a forecast, it is
a description. This splits the history and reports what each screen did on the
second half only.
"""
import sys, json; sys.path.insert(0, ".")
import pandas as pd
from screener.rules import load_rules
from screener.backtest import (add_panel_columns, add_forward_returns, rule_hits,
                               bracket_returns, bracket_stats, HORIZONS)
from screener import quality
from screener.store import Store

SPLIT = pd.Timestamp("2025-07-01")
st = Store("data")
feat = pd.read_parquet("data/screen/features.parquet")
lat = pd.read_parquet("data/screen/latest_features.parquet")
feat = feat.merge(lat[["isin", "cap_band"]], on="isin", how="left")
bars = feat[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
feat["contaminated"] = quality.contamination_mask(
    bars, quality.detect_price_jumps(bars, st.read_adjustments())).to_numpy()
feat = add_forward_returns(add_panel_columns(feat), HORIZONS)
IS, OOS = feat[feat["date"] < SPLIT], feat[feat["date"] >= SPLIT]

def run(df, rule):
    m = rule_hits(df, rule)
    if m.sum() < 40:
        return None
    return bracket_stats(bracket_returns(df, m, rr=2.0, max_bars=20, risk="atr", atr_mult=1.0))

rows = []
for r in load_rules("config/rules.yaml"):
    a, b = run(IS, r), run(OOS, r)
    if not (a and b):
        continue
    rows.append(dict(name=r.name, title=r.title or r.name,
                     is_exp=a["expectancy_r"], is_n=a["n"],
                     oos_exp=b["expectancy_r"], oos_n=b["n"],
                     oos_pf=b["profit_factor"], oos_tgt=b["target_pct"],
                     oos_stop=b["stop_pct"], decay=b["expectancy_r"] - a["expectancy_r"]))
d = pd.DataFrame(rows).sort_values("oos_exp", ascending=False)
print(f"IS  {IS['date'].min().date()} to {IS['date'].max().date()}")
print(f"OOS {OOS['date'].min().date()} to {OOS['date'].max().date()}   (1:2 bracket, 1xATR stop)\n")
print(f"{'screen':26s} {'IS exp':>8s} {'OOS exp':>8s} {'decay':>8s} {'OOS n':>8s} "
      f"{'stop%':>6s} {'tgt%':>6s} {'PF':>5s}  holds?")
print("-" * 96)
for _, r in d.iterrows():
    ok = "YES" if (r.oos_exp > 0 and r.is_exp > 0) else ("BROKE" if r.is_exp > 0 else "no")
    print(f"{r['name'][:26]:26s} {r.is_exp:>+8.3f} {r.oos_exp:>+8.3f} {r.decay:>+8.3f} "
          f"{r.oos_n:>8,d} {r.oos_stop:>6.1f} {r.oos_tgt:>6.1f} {r.oos_pf:>5.2f}  {ok}")
d.to_json("data/screen/oos.json", orient="records")
surv = d[(d.is_exp > 0) & (d.oos_exp > 0)]
print(f"\n{len(d[d.is_exp>0])} screens were positive in-sample; "
      f"{len(surv)} of them stayed positive out-of-sample.")
