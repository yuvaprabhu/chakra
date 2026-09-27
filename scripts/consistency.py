"""Which screen and which reward:risk is actually steady, not just profitable.

Expectancy says what the average trade returns. It says nothing about whether
you would survive trading it - a screen that makes its year in two months and
bleeds for ten has the same expectancy as one that grinds up every month, and
they are not the same thing to hold.

So this measures the shape of the return, monthly:
  * months positive  - how often a month ends up
  * worst month      - the hole you have to sit in
  * longest red run  - consecutive losing months, which is what makes people
                       abandon a system that was working
  * monthly Sharpe   - mean over standard deviation, annualised

And it does it across reward:risk ratios, because RR is not a free parameter.
A wider target wins less often and needs a bigger winner to pay for the losses;
a tighter one wins more often but takes less each time. Which is steadier is an
empirical question, not a preference.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.rules import load_rules
from screener.backtest import (add_panel_columns, add_forward_returns, rule_hits,
                               bracket_returns, HORIZONS)
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
rules = {r.name: r for r in load_rules("config/rules.yaml")}

def monthly(trades):
    """Average R per trade, bucketed by the month the trade was entered."""
    if trades is None or trades.empty:
        return None
    t = trades.copy()
    t["m"] = pd.to_datetime(t["date"]).dt.to_period("M")
    g = t.groupby("m")["r_multiple"].agg(["mean", "size"])
    return g[g["size"] >= 5]["mean"]

def streak(s):
    best = cur = 0
    for v in s:
        cur = cur + 1 if v <= 0 else 0
        best = max(best, cur)
    return best

CANDIDATES = ["rsi2_reversion", "ema21_pullback", "momentum_leaders",
              "minervini_trend_template", "held_breakout", "near_52w_high"]
RRS = [1.0, 1.5, 2.0, 3.0]

print(f"{'screen':24s} {'RR':>5s} {'trades':>8s} {'win%':>6s} {'exp R':>7s} "
      f"{'mo +':>7s} {'worst mo':>9s} {'red run':>8s} {'Sharpe':>7s}")
print("-" * 92)
best = []
for name in CANDIDATES:
    if name not in rules:
        continue
    mask = rule_hits(feat, rules[name])
    for rr in RRS:
        t = bracket_returns(feat, mask, rr=rr, max_bars=20, risk="atr", atr_mult=1.0)
        if t is None or t.empty or len(t) < 200:
            continue
        m = monthly(t)
        if m is None or len(m) < 12:
            continue
        exp = float(t["r_multiple"].mean())
        win = float((t["ret"] > 0).mean() * 100)
        pos = float((m > 0).mean() * 100)
        sharpe = float(m.mean() / m.std() * np.sqrt(12)) if m.std() > 0 else np.nan
        print(f"{name[:24]:24s} {rr:>5.1f} {len(t):>8,d} {win:>6.1f} {exp:>+7.3f} "
              f"{pos:>6.0f}% {m.min():>+9.3f} {streak(m):>8d} {sharpe:>+7.2f}")
        best.append((name, rr, exp, pos, streak(m), sharpe, len(t)))
    print()

print("Ranked by monthly consistency (months positive, then Sharpe):")
b = sorted([x for x in best if x[2] > 0], key=lambda x: (-x[3], -x[5]))
for name, rr, exp, pos, run, sh, n in b[:8]:
    print(f"  {name:24s} 1:{rr:g}  {pos:.0f}% of months up, "
          f"worst red run {run} months, Sharpe {sh:+.2f}, exp {exp:+.3f} R on {n:,}")
