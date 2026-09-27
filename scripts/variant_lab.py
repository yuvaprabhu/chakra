"""Variant lab: take a base setup, add confirmation constraints, keep what survives.

This is data mining, and it is run as data mining: every variant is scored on
the in-sample window only, the survivors are then checked on the out-of-sample
window, and the total number of variants tried is printed at the top so the
reader can judge how many draws produced the winners. With N variants, the best
in-sample cell is expected to look good by chance; only out-of-sample survival
means anything.

Usage: python scripts/variant_lab.py variants.txt
Each line of the file:  base | name | extra_clause | stop | rationale
stop is one of fixed3, confirm_low, low3, level.
"""
import sys, re; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.rules import Rule, load_rules
from screener.backtest import (add_panel_columns, add_forward_returns, rule_hits,
                               bracket_returns, HORIZONS)
from screener import quality
from screener.store import Store

SPLIT = pd.Timestamp("2025-07-01")
HOLD = 22
RR = 2.0
MAX_RISK = 8.0          # a "structural" stop 15% away is not the same trade

# Which level a stop-under-the-level goes beneath, per base setup.
LEVEL_COL = {
    "ema21_pullback": "ema_21", "donchian_breakout": "hi_20_prior",
    "narrow_cpr_breakout": "m_cpr_bc", "monthly_r1_breakout": "m_r1",
    "high_tight_breakout": "hi_250_prior", "rsi2_reversion": "sma_200",
    "stage2_base_breakout": "hi_60_prior", "r1_breakout_retest": "m_r1",
    "held_breakout": "brk_level",
}

st = Store("data")
feat = pd.read_parquet("data/screen/features.parquet")
lat = pd.read_parquet("data/screen/latest_features.parquet")
feat = feat.merge(lat[["isin", "cap_band"]], on="isin", how="left")
bars = feat[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
feat["contaminated"] = quality.contamination_mask(
    bars, quality.detect_price_jumps(bars, st.read_adjustments())).to_numpy()
feat = add_forward_returns(add_panel_columns(feat), HORIZONS)
# Structural stop prices, a hair under the structure so a touch is not a fill.
feat["stop_confirm"] = feat["adj_low"] * 0.998
feat["stop_low3"] = feat.groupby("isin")["adj_low"].transform(
    lambda s: s.rolling(3, min_periods=1).min()) * 0.998
for base, col in LEVEL_COL.items():
    if col in feat.columns:
        feat[f"stop_level_{base}"] = feat[col] * 0.995

IS, OOS = feat[feat["date"] < SPLIT], feat[feat["date"] >= SPLIT]
BASE = {r.name: r for r in load_rules("config/rules.yaml")}

def streak(s):
    b = c = 0
    for v in s:
        c = c + 1 if v <= 0 else 0; b = max(b, c)
    return b

def run(df, expr, stop, base, min_n=int(__import__("os").environ.get("MIN_N", "40"))):
    m = rule_hits(df, Rule(name="v", expr=expr, min_history=200))
    if m.sum() < min_n:
        return None
    if stop == "fixed3":
        t = bracket_returns(df, m, rr=RR, max_bars=HOLD, risk="pct", risk_pct=3.0)
    else:
        col = {"confirm_low": "stop_confirm", "low3": "stop_low3",
               "level": f"stop_level_{base}"}.get(stop)
        if col not in df.columns:
            return None
        t = bracket_returns(df, m, rr=RR, max_bars=HOLD, risk="price", stop_price_col=col)
        t = t[t["risk_pct"] <= MAX_RISK] if t is not None and len(t) else t
    if t is None or t.empty or len(t) < min_n:
        return None
    tt = t.copy(); tt["mo"] = pd.to_datetime(tt["date"]).dt.to_period("M")
    g = tt.groupby("mo")["ret"].agg(["mean", "size"]); g = g[g["size"] >= 4]["mean"]
    w, l = t.loc[t["ret"] > 0, "ret"], t.loc[t["ret"] <= 0, "ret"]
    oc = t["outcome"].value_counts()
    return dict(n=len(t), avg=t["ret"].mean(), expR=t["r_multiple"].mean(),
                win=(t["ret"] > 0).mean() * 100, tgt=oc.get("target", 0) / len(t) * 100,
                stop=oc.get("stop", 0) / len(t) * 100, risk=t["risk_pct"].mean(),
                pf=(w.sum() / -l.sum()) if len(l) and l.sum() < 0 else np.nan,
                mo=(g > 0).mean() * 100 if len(g) >= 6 else np.nan,
                red=streak(g) if len(g) >= 6 else -1)

variants = []
for line in open(sys.argv[1]):
    line = line.strip()
    if not line or line.startswith("#") or "|" not in line:
        continue
    parts = [x.strip() for x in line.split("|")]
    if len(parts) < 4:
        continue
    base, name, clause, stop = parts[:4]
    if base not in BASE:
        continue
    variants.append((base, name, clause, stop.lower()))

print(f"{len(variants)} variants tried across {len(set(v[0] for v in variants))} base setups. "
      f"With this many draws, expect a few good in-sample numbers by chance.\n")
print(f"IS {IS['date'].min().date()}..{IS['date'].max().date()}   "
      f"OOS {OOS['date'].min().date()}..{OOS['date'].max().date()}   "
      f"target = {RR:g}x the stop, hold <= {HOLD} sessions\n")

results = []
# Baselines first, so every variant has something to be compared against.
for base in sorted(set(v[0] for v in variants)):
    for stop in ("fixed3",):
        a = run(IS, BASE[base].expr, stop, base); b = run(OOS, BASE[base].expr, stop, base)
        if a and b:
            results.append(dict(base=base, name="(baseline)", clause="", stop=stop, IS=a, OOS=b))
for base, name, clause, stop in variants:
    expr = f"({BASE[base].expr}) and ({clause})"
    try:
        a = run(IS, expr, stop, base); b = run(OOS, expr, stop, base)
    except Exception as e:
        print(f"  skipped {name}: {str(e)[:80]}"); continue
    if a and b:
        results.append(dict(base=base, name=name, clause=clause, stop=stop, IS=a, OOS=b))

df = pd.DataFrame([{**{k: r[k] for k in ("base", "name", "clause", "stop")},
                    "is_avg": r["IS"]["avg"], "is_n": r["IS"]["n"], "is_win": r["IS"]["win"],
                    "oos_avg": r["OOS"]["avg"], "oos_n": r["OOS"]["n"], "oos_win": r["OOS"]["win"],
                    "oos_pf": r["OOS"]["pf"], "oos_mo": r["OOS"]["mo"], "oos_red": r["OOS"]["red"],
                    "oos_risk": r["OOS"]["risk"], "oos_expR": r["OOS"]["expR"]}
                   for r in results])
df.to_csv("data/screen/variant_lab.csv", index=False)

for base, g in df.groupby("base"):
    g = g.sort_values("is_avg", ascending=False)
    print(f"=== {base} ===")
    print(f"{'variant':30s} {'stop':>11s} {'IS n':>6s} {'IS avg':>7s} | {'OOS n':>6s} "
          f"{'OOS avg':>8s} {'win%':>5s} {'PF':>5s} {'mo+':>4s} {'red':>3s} {'risk':>5s}")
    for _, r in g.iterrows():
        tag = "  <- survives" if (r.is_avg > 0 and r.oos_avg > 0 and r.oos_mo >= 50) else ""
        print(f"{r['name'][:30]:30s} {r['stop']:>11s} {r.is_n:>6,d} {r.is_avg:>+7.2f} | "
              f"{r.oos_n:>6,d} {r.oos_avg:>+8.2f} {r.oos_win:>5.0f} {r.oos_pf:>5.2f} "
              f"{r.oos_mo:>3.0f}% {r.oos_red:>3.0f} {r.oos_risk:>4.1f}%{tag}")
    print()

surv = df[(df.is_avg > 0) & (df.oos_avg > 0) & (df.oos_mo >= 50) & (df.name != "(baseline)")]
print("=" * 100)
print(f"SURVIVORS: positive in-sample, positive out-of-sample, majority of OOS months up "
      f"- {len(surv)} of {len(df) - df['name'].eq('(baseline)').sum()} variants")
for _, r in surv.sort_values("oos_avg", ascending=False).iterrows():
    print(f"  {r.base:22s} {r['name']:30s} stop={r.stop:11s} OOS {r.oos_avg:+.2f}%/trade, "
          f"win {r.oos_win:.0f}%, PF {r.oos_pf:.2f}, {r.oos_mo:.0f}% months up, n={r.oos_n}")
    print(f"      + {r.clause}")
