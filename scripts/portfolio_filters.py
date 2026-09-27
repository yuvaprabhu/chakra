"""Do confirmation filters improve the monthly PORTFOLIO, where the edge actually lives?

The bracket tests say confirmation constraints do not rescue a stop-and-target
trade. The portfolio test says the base setups already work when held a month
with no stop. The constructive question is whether the same confirmations,
used as a FILTER on which names the portfolio holds, raise its excess return.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.rules import Rule, load_rules
from screener.backtest import add_panel_columns, rule_hits
from screener import quality
from screener.store import Store

TOPN = 20; SPLIT = pd.Period("2025-07", "M")
st = Store("data")
feat = pd.read_parquet("data/screen/features.parquet")
lat = pd.read_parquet("data/screen/latest_features.parquet")
feat = feat.merge(lat[["isin", "cap_band"]], on="isin", how="left")
bars = feat[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
feat["contaminated"] = quality.contamination_mask(
    bars, quality.detect_price_jumps(bars, st.read_adjustments())).to_numpy()
feat = add_panel_columns(feat)
d = feat.sort_values(["isin", "date"])
d["fwd"] = (d.groupby("isin")["adj_close"].shift(-22) / d["adj_close"] - 1) * 100
d["bench"] = d.groupby("date")["fwd"].transform("mean")
d["mo"] = d["date"].dt.to_period("M")
first = d.groupby("mo")["date"].transform("min")
REB = d[(d["date"] == first) & d["fwd"].notna()]
BASE = {r.name: r for r in load_rules("config/rules.yaml")}

def port(expr, sort_by, asc, sub):
    m = rule_hits(sub, Rule(name="p", expr=expr, min_history=200))
    if m.sum() < 15: return None
    P, B, H = [], [], []
    for mo, g in sub[m].groupby("mo"):
        if sort_by in g.columns: g = g.sort_values(sort_by, ascending=asc)
        g = g.head(TOPN)
        if len(g) < 3: continue
        P.append(g["fwd"].mean()); B.append(g["bench"].iloc[0]); H.append(len(g))
    return (np.array(P), np.array(B), np.mean(H)) if len(P) >= 6 else None

FILTERS = [
    ("(none)", ""),
    ("strong close", "close_pos > 0.7"),
    ("accumulation", "ud_vol_20 >= 1.3"),
    ("pocket pivot day", "pocket_pivot == 1"),
    ("no gap chase", "gap_pct <= 2"),
    ("above monthly P", "above_m_p == 1"),
    ("contraction", "contract_seq == 1"),
    ("RS >= 80", "rs_rank >= 80"),
    ("not extended", "dist_sma_50 < 12"),
    ("shallow base", "base_depth > -10"),
    ("accum + strong close", "ud_vol_20 >= 1.3 and close_pos > 0.7"),
    ("accum + not extended", "ud_vol_20 >= 1.3 and dist_sma_50 < 12"),
]
for base in ("stage2_base_breakout", "donchian_breakout", "trend_stack", "minervini_trend_template"):
    r = BASE[base]
    print(f"=== {base}  (top {TOPN}, monthly rebalance, no stop; excess over equal-weight universe) ===")
    print(f"{'filter':24s} {'IS mo':>6s} {'IS exc':>7s} | {'OOS mo':>7s} {'OOS exc':>8s} {'beat%':>6s} {'held':>5s}")
    for name, clause in FILTERS:
        expr = r.expr if not clause else f"({r.expr}) and ({clause})"
        a = port(expr, r.sort_by, bool(r.ascending), REB[REB["mo"] < SPLIT])
        b = port(expr, r.sort_by, bool(r.ascending), REB[REB["mo"] >= SPLIT])
        if not (a and b): print(f"{name:24s}   too few names"); continue
        ia = (a[0] - a[1]).mean(); ob = b[0] - b[1]
        tag = "  <-" if (ia > 0 and ob.mean() > 0 and (ob > 0).mean() >= 0.7) else ""
        print(f"{name:24s} {len(a[0]):>6d} {ia:>+7.2f} | {len(b[0]):>7d} {ob.mean():>+8.2f} "
              f"{(ob>0).mean()*100:>5.0f}% {b[2]:>5.1f}{tag}")
    print()
