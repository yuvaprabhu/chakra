"""Parameter sweep with an out-of-sample split.

Sweeping parameters does two things at once: it finds settings that work, and
it manufactures settings that only appear to. With 25 combinations, the best
one looks good by chance even when the screen has no edge at all, so a table of
"best parameters" is worth nothing on its own.

So every sweep here is scored twice. Parameters are chosen ONLY on the
in-sample window; the number that gets reported is what that choice then did on
data it never saw. Three things are printed alongside it:

  * how the best in-sample cell ranked out-of-sample,
  * how many of ALL cells were positive out-of-sample - a real effect is a
    plateau where most settings work, a fluke is a single spike,
  * the degradation from in-sample to out-of-sample.

A screen whose best cell collapses out-of-sample has been fitted, not found.
"""
import sys, itertools, json, time; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.rules import Rule
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
IS = feat[feat["date"] < SPLIT]
OOS = feat[feat["date"] >= SPLIT]
print(f"in-sample  {IS['date'].min().date()} to {IS['date'].max().date()}  {len(IS):,} rows")
print(f"out-sample {OOS['date'].min().date()} to {OOS['date'].max().date()}  {len(OOS):,} rows\n")

TREND = "adj_close > sma_50 and sma_50 > sma_200"

SWEEPS = {
  # Raju (2023): in India, PROXIMITY to the 52-week high beats return-based
  # momentum and gives more stable alpha. p is how close, q the RS floor.
  "52w_proximity": ("pct_from_52w_high > -{p} and " + TREND + " and rs_rank >= {q}",
                    {"p": [2, 5, 8, 12], "q": [50, 65, 80]}),
  # Does making a breakout prove itself for longer keep helping, or is 3 days
  # already the whole effect?
  "held_breakout": ("since_brk_60 >= {p} and since_brk_60 <= 15 and brk_cushion > 0 "
                    "and closes_above_brk >= {p} and brk_vol > {q} and adj_close > brk_level "
                    "and sma_50 > sma_200 and rs_rank >= 55",
                    {"p": [2, 3, 5, 8], "q": [0.8, 1.0, 1.5, 2.0]}),
  # How extended does a pullback need to have been, and how deep may it go?
  "ema21_pullback": ("ext_ema21_15 >= {p} and dist_ema_21 > -{q} and dist_ema_21 < 1.5 "
                     "and off_high_10 < -2 and " + TREND + " and sma_200_slope > 0",
                     {"p": [3, 5, 8, 12], "q": [2, 4, 6]}),
  # Connors' threshold, and whether the trend filter should be the 200 or the 50.
  "rsi2_reversion": ("rsi_2 < {p} and adj_close > sma_200 and dist_sma_50 > -{q} "
                     "and sma_200_slope > 0 and ret_120d > 0",
                     {"p": [2, 5, 8, 15], "q": [2, 5, 10]}),
  # How fresh must the shakeout be, and how far outside the noise the stop?
  "spring_reclaim": ("since_spring <= {p} and adj_close > lo_20_prior and atr_to_spring >= {q} "
                     "and adj_close > sma_200 and sma_200_slope > 0 and ud_vol_20 > 1.0",
                     {"p": [1, 3, 5], "q": [1.0, 1.5, 2.5]}),
}

def score(df, expr):
    r = Rule(name="sweep", expr=expr, min_history=200)
    try:
        m = rule_hits(df, r)
    except Exception:
        return None
    if m.sum() < 60:
        return None
    t = bracket_returns(df, m, rr=2.0, max_bars=20, risk="atr", atr_mult=1.0)
    s = bracket_stats(t)
    return None if s.get("n", 0) < 60 else s

results = {}
for name, (tpl, grid) in SWEEPS.items():
    keys = list(grid)
    rows = []
    t0 = time.time()
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        expr = tpl.format(**params)
        a, b = score(IS, expr), score(OOS, expr)
        if a and b:
            rows.append({**params, "is_exp": a["expectancy_r"], "is_n": a["n"],
                         "oos_exp": b["expectancy_r"], "oos_n": b["n"],
                         "oos_pf": b["profit_factor"], "oos_tgt": b["target_pct"]})
    if not rows:
        print(f"{name}: no cell produced enough trades\n"); continue
    d = pd.DataFrame(rows)
    best = d.loc[d["is_exp"].idxmax()]
    d_sorted = d.sort_values("oos_exp", ascending=False).reset_index(drop=True)
    rank = int(d_sorted.index[(d_sorted[keys] == best[keys]).all(axis=1)][0]) + 1
    pos = int((d["oos_exp"] > 0).sum())
    results[name] = dict(best={k: best[k] for k in keys},
                         is_exp=float(best["is_exp"]), oos_exp=float(best["oos_exp"]),
                         oos_n=int(best["oos_n"]), oos_pf=best["oos_pf"],
                         cells=len(d), rank=rank, pos=pos,
                         median_oos=float(d["oos_exp"].median()))
    p = results[name]
    print(f"{name}  ({len(d)} cells, {time.time()-t0:.0f}s)")
    print(f"  best in-sample: {p['best']}  ->  IS {p['is_exp']:+.3f} R")
    print(f"  that same cell out-of-sample: {p['oos_exp']:+.3f} R on {p['oos_n']:,} trades "
          f"(PF {p['oos_pf']}), ranked {p['rank']} of {p['cells']} OOS")
    print(f"  cells positive out-of-sample: {p['pos']}/{p['cells']}"
          f"   median cell OOS {p['median_oos']:+.3f} R")
    verdict = ("HOLDS - most settings work, the choice was not the edge"
               if p["pos"] >= 0.6 * p["cells"] and p["oos_exp"] > 0 else
               "FITTED - the best in-sample cell does not survive"
               if p["oos_exp"] <= 0 else
               "FRAGILE - it works, but only for some settings")
    print(f"  verdict: {verdict}\n")

json.dump(results, open("data/screen/sweep.json", "w"), indent=1, default=str)
print("written to data/screen/sweep.json")
