"""Phase 2: the candidate strategy, simulated properly and stressed.

Entry   Leader Dip (RS rank >= 80, above the 200-day, new 7-day low or RSI(2) < 10)
Filter  the dip is still holding: close at least MIN_DIST % above the 21 EMA
Exit    first close above the previous seven closes, sold next open; 20-bar cap
Stop    a HARD stop below entry, flat % or ATR multiple, filled intraday at the
        stop or at the open if the bar gaps through it
Every trade has a stop, per the user's rule. Stress: both halves, 100 random
slot orderings, and +/-20% on each numeric parameter.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.rules import load_rules
from screener.backtest import add_panel_columns, rule_hits
from screener.robustness import stress
from screener import quality
from screener.store import Store

COST = 0.30
t0 = time.time()
feat = pd.read_parquet("data/screen/features.parquet")
lat = pd.read_parquet("data/screen/latest_features.parquet")
feat = feat.merge(lat[["isin", "cap_band"]], on="isin", how="left")
st = Store("data")
b = feat[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
feat["contaminated"] = quality.contamination_mask(b, quality.detect_price_jumps(b, st.read_adjustments())).to_numpy()
feat = add_panel_columns(feat).sort_values(["isin", "date"]).reset_index(drop=True)
CAL = pd.DatetimeIndex(sorted(feat["date"].unique())); CALPOS = {d: i for i, d in enumerate(CAL)}
O = feat["adj_open"].to_numpy(float); H = feat["adj_high"].to_numpy(float)
L = feat["adj_low"].to_numpy(float);  C = feat["adj_close"].to_numpy(float)
HI7 = feat["hi7_prior"].to_numpy(float); ATRP = feat["atr_pct"].to_numpy(float) / 100
DE21 = feat["dist_ema_21"].to_numpy(float); RS = feat["rs_rank"].to_numpy(float)
DISP = None
ISIN = feat["isin"].to_numpy(); DATES = feat["date"].to_numpy()
LAST = pd.Series(np.arange(len(feat))).groupby(pd.factorize(feat["isin"])[0]).transform("max").to_numpy()
rules = {r.name: r for r in load_rules("config/rules.yaml")}
BASE = rule_hits(feat, rules["leader_dip"]).to_numpy() & (feat["date"] >= "2024-06-01").to_numpy()
SPLIT = pd.Timestamp("2025-03-19")
print(f"panel ready {time.time()-t0:.0f}s, base signals {int(BASE.sum()):,}", flush=True)


def make_trades(p):
    sig = BASE & (DE21 >= p["min_dist_ema21"])
    rows = []; busy = {}
    for t in np.flatnonzero(sig):
        if t + 1 > LAST[t] or busy.get(ISIN[t], -1) >= t or not np.isfinite(ATRP[t]):
            continue
        e = O[t + 1]
        if not np.isfinite(e) or e <= 0:
            continue
        stop_d = (p["atr_mult"] * ATRP[t]) if p.get("atr_mult") else p["stop_pct"] / 100
        stop = e * (1 - stop_d)
        cap = min(LAST[t], t + int(p["cap"])); k = t + 1; out = None; why = "time"
        while k <= cap:
            if L[k] <= stop:                       # hard stop first: the day's low is known before its close
                out = min(O[k], stop); why = "stop"; break
            if np.isfinite(HI7[k]) and C[k] > HI7[k]:
                if k + 1 <= LAST[t]:
                    out = O[k + 1]; k += 1
                else:
                    out = C[k]
                why = "reversal"; break
            k += 1
        if out is None:
            k = min(t + int(p["cap"]), LAST[t]); out = C[k]
        if not np.isfinite(out):
            continue
        busy[ISIN[t]] = k
        rows.append((pd.Timestamp(DATES[t]), ISIN[t], CALPOS[pd.Timestamp(DATES[t + 1])],
                     CALPOS[pd.Timestamp(DATES[k])], (out / e - 1) * 100 - COST, k - t, why, RS[t], stop_d * 100))
    return pd.DataFrame(rows, columns=["date", "isin", "entry_i", "exit_i", "net", "bars", "why", "rs", "stop_pct"])


def describe(t, lbl):
    for h, w in (("H1", t.date < SPLIT), ("H2", t.date >= SPLIT)):
        x = t[w]; n = x.net; wn = n[n > 0]; ls = n[n <= 0]
        print(f"  {lbl:30s}{h} n={len(x):5d} win {(n>0).mean()*100:5.1f}%  mean {n.mean():+.2f}  avgW {wn.mean():+.2f} avgL {ls.mean():+.2f}  "
              f"pay {wn.mean()/-ls.mean():.2f}  stopped {(x.why=='stop').mean()*100:3.0f}%  held {x.bars.mean():4.1f}")


CANDS = [("no stop, filtered", {"min_dist_ema21": 1.41, "stop_pct": 99.0, "cap": 20}),
         ("flat 15% stop", {"min_dist_ema21": 1.41, "stop_pct": 15.0, "cap": 20}),
         ("flat 12% stop", {"min_dist_ema21": 1.41, "stop_pct": 12.0, "cap": 20}),
         ("flat 10% stop", {"min_dist_ema21": 1.41, "stop_pct": 10.0, "cap": 20}),
         ("flat 8% stop", {"min_dist_ema21": 1.41, "stop_pct": 8.0, "cap": 20}),
         ("3.0x ATR stop", {"min_dist_ema21": 1.41, "atr_mult": 3.0, "stop_pct": 0, "cap": 20}),
         ("2.5x ATR stop", {"min_dist_ema21": 1.41, "atr_mult": 2.5, "stop_pct": 0, "cap": 20}),
         ("2.0x ATR stop", {"min_dist_ema21": 1.41, "atr_mult": 2.0, "stop_pct": 0, "cap": 20})]
res = []
print("\nPER TRADE (proper simulation, stop fills intraday, gap fills at the open)")
for lbl, p in CANDS:
    describe(make_trades(p), lbl)
print("\nPORTFOLIO STRESS: 10 slots, both halves, 100 orderings, +/-20% on each parameter")
print(f"{'candidate':20s}{'full CAGR':>10s}{'maxDD':>7s} | {'H1':>7s}{'H2':>7s} | {'rnd mean':>9s}{'p5':>7s}{'p95':>7s} | {'nbr min':>8s}{'nbr mean':>9s} | robust")
print("-" * 112)
for lbl, p in CANDS:
    r = stress(make_trades, p, sessions=CAL, split=SPLIT, slots=10, order_col="rs", n_random=100)
    res.append({"label": lbl, **{k: v for k, v in r.items() if k != "params"}, "params": p})
    print(f"{lbl:20s}{r['full']['cagr']:>+10.1f}{r['full']['maxdd']:>7.1f} | {r['h1']['cagr']:>+7.1f}{r['h2']['cagr']:>+7.1f} | "
          f"{r['random_order']['mean']:>+9.1f}{r['random_order']['p5']:>+7.1f}{r['random_order']['p95']:>+7.1f} | "
          f"{r['neighbour_min']:>+8.1f}{r['neighbour_mean']:>+9.1f} | {'YES' if r['robust'] else 'no'}", flush=True)
Path("data/screen/synthesis.json").write_text(json.dumps(res, default=float))
print(f"\ndone in {time.time()-t0:.0f}s")
