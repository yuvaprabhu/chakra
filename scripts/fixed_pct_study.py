"""Every screen, every entry, with a FIXED percentage bracket.

The spec this answers, stated in the trader's own terms rather than in ATR
multiples: stop 3%, target 6-10%, held two weeks to one month, over the last
two years. Percentages, not R-multiples, because that is what actually gets
typed into the order window.

One consequence is worth knowing before reading the table: a flat 3% stop is
INSIDE the daily noise for a lot of Indian mid-caps. Their ATR runs 2-3%, so a
single ordinary session can take the stop out without the idea being wrong.
The average ATR at entry is reported for exactly that reason.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.rules import load_rules
from screener.backtest import (add_panel_columns, add_forward_returns, rule_hits,
                               bracket_returns, HORIZONS)
from screener import quality
from screener.store import Store

STOP_PCT = 3.0
TARGETS = {"6%": 2.0, "9%": 3.0, "10%": 10.0 / 3.0}
HOLDS = {"2 weeks": 10, "1 month": 22}

st = Store("data")
feat = pd.read_parquet("data/screen/features.parquet")
lat = pd.read_parquet("data/screen/latest_features.parquet")
feat = feat.merge(lat[["isin", "cap_band"]], on="isin", how="left")
bars = feat[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
feat["contaminated"] = quality.contamination_mask(
    bars, quality.detect_price_jumps(bars, st.read_adjustments())).to_numpy()
feat = add_forward_returns(add_panel_columns(feat), HORIZONS)

END = feat["date"].max()
START = END - pd.DateOffset(years=2)
W = feat[feat["date"] >= START]
print(f"window: {W['date'].min().date()} to {W['date'].max().date()}  "
      f"({W['date'].nunique()} sessions)")
print(f"bracket: stop {STOP_PCT:g}% fixed, target 6/9/10%, entry at the next open\n")

def streak(s):
    b = c = 0
    for v in s:
        c = c + 1 if v <= 0 else 0
        b = max(b, c)
    return b

rules = load_rules("config/rules.yaml")
masks = {r.name: rule_hits(W, r) for r in rules}
atr = W["atr_pct"]

def study(rule, rr, hold):
    m = masks[rule.name]
    t = bracket_returns(W, m, rr=rr, max_bars=hold, risk="pct", risk_pct=STOP_PCT)
    if t is None or t.empty or len(t) < 150:
        return None
    oc = t["outcome"].value_counts()
    wins, losses = t.loc[t["ret"] > 0, "ret"], t.loc[t["ret"] <= 0, "ret"]
    tt = t.copy()
    tt["mo"] = pd.to_datetime(tt["date"]).dt.to_period("M")
    g = tt.groupby("mo")["ret"].agg(["mean", "size"])
    g = g[g["size"] >= 5]["mean"]
    return dict(
        n=len(t), stop=oc.get("stop", 0) / len(t) * 100, tgt=oc.get("target", 0) / len(t) * 100,
        time=oc.get("time", 0) / len(t) * 100, win=(t["ret"] > 0).mean() * 100,
        avg=t["ret"].mean(), pf=(wins.sum() / -losses.sum()) if len(losses) else np.nan,
        bars=t["bars_held"].mean(),
        mo_pos=(g > 0).mean() * 100 if len(g) >= 8 else np.nan,
        worst=g.min() if len(g) >= 8 else np.nan, red=streak(g) if len(g) >= 8 else np.nan,
        months=len(g), atr=float(atr[m].mean()))

for hold_name, hold in HOLDS.items():
    for tgt_name, rr in TARGETS.items():
        rows = []
        for r in rules:
            s = study(r, rr, hold)
            if s:
                rows.append({"screen": r.name, **s})
        if not rows:
            continue
        d = pd.DataFrame(rows).sort_values("avg", ascending=False)
        print(f"=== stop 3%  target {tgt_name}  held up to {hold_name} "
              f"({hold} sessions) ===")
        print(f"{'screen':24s} {'trades':>7s} {'stop%':>6s} {'tgt%':>6s} {'time%':>6s} "
              f"{'win%':>6s} {'avg%':>7s} {'PF':>5s} {'mo+':>5s} {'worst':>7s} {'red':>4s} {'ATR':>5s}")
        print("-" * 104)
        for _, r in d.head(8).iterrows():
            print(f"{r['screen'][:24]:24s} {r['n']:>7,d} {r['stop']:>6.1f} {r['tgt']:>6.1f} "
                  f"{r['time']:>6.1f} {r['win']:>6.1f} {r['avg']:>+7.2f} {r['pf']:>5.2f} "
                  f"{r['mo_pos']:>4.0f}% {r['worst']:>+7.2f} {r['red']:>4.0f} {r['atr']:>5.1f}")
        pos = d[(d["avg"] > 0) & (d["mo_pos"] >= 50)]
        print(f"  -> {len(d[d['avg']>0])} of {len(d)} screens profitable; "
              f"{len(pos)} of those also had a majority of months positive\n")
