"""Independent audit: recompute each event from the raw bar history, not from
the feature columns the rule used, and count disagreements."""
import sys; sys.path.insert(0, ".")
import pandas as pd, numpy as np
from screener.store import Store

feat = pd.read_parquet("data/screen/features.parquet")
hits = pd.read_parquet("data/screen/hits.parquet")
latest = pd.read_parquet("data/screen/latest_features.parquet")
as_of = feat["date"].max()
sym = latest.set_index("isin")["symbol"].to_dict()

def hist(isin, n=40):
    g = feat[feat["isin"] == isin].sort_values("date")
    return g.tail(n)

def audit(rule, check, label):
    h = hits[hits["rule"] == rule]
    bad = []
    for isin in h["isin"]:
        g = feat[feat["isin"] == isin].sort_values("date")
        ok, why = check(g)
        if not ok:
            bad.append((sym.get(isin, isin), why))
    flag = "OK " if not bad else "BAD"
    print(f"{flag} {rule:24s} hits={len(h):4d}  false positives={len(bad):3d}   [{label}]")
    for s, w in bad[:6]:
        print(f"      {s:14s} {w}")
    return len(bad)

def pullback(g):
    # did price actually get 5% above the 21 EMA in the 15 sessions before today?
    d = (g["adj_close"] / g["ema_21"] - 1) * 100
    prior = d.iloc[-16:-1]
    m = prior.max()
    if m < 5:
        return False, f"max extension above 21EMA in prior 15d = {m:.2f}%"
    off = (g["adj_close"].iloc[-1] / g["adj_high"].iloc[-10:].max() - 1) * 100
    if off >= -2:
        return False, f"only {off:.2f}% off its own 10d high - no pullback"
    return True, ""

def new_high(g):
    prior = g["adj_close"].iloc[-21:-1].max()
    c = g["adj_close"].iloc[-1]
    if c <= prior:
        return False, f"close {c:.2f} <= prior 20d high {prior:.2f}"
    return True, ""

def new_high_60(g):
    prior = g["adj_close"].iloc[-61:-1].max()
    c = g["adj_close"].iloc[-1]
    return (c > prior, f"close {c:.2f} <= prior 60d high {prior:.2f}")

def fresh_cross(level, maxbars):
    def f(g):
        above = g["adj_close"] > g[level]
        cross = above & ~above.shift(1, fill_value=False)
        idx = np.flatnonzero(cross.values)
        if not len(idx):
            return False, f"never crossed {level}"
        since = len(g) - 1 - idx[-1]
        if since > maxbars:
            return False, f"crossed {level} {since} sessions ago (limit {maxbars})"
        return True, ""
    return f

def retest(g):
    above = g["adj_close"] > g["m_r1"]
    cross = above & ~above.shift(1, fill_value=False)
    idx = np.flatnonzero(cross.values)
    if not len(idx):
        return False, "never crossed m_r1"
    since = len(g) - 1 - idx[-1]
    if not (2 <= since <= 15):
        return False, f"crossed m_r1 {since} sessions ago"
    low3 = g["adj_low"].iloc[-3:].min()
    d = (low3 / g["m_r1"].iloc[-1] - 1) * 100
    if d >= 1.0:
        return False, f"low never came back to R1 ({d:+.2f}%)"
    if g["adj_close"].iloc[-1] <= g["m_r1"].iloc[-1]:
        return False, "closed back under R1"
    return True, ""

def pocket(g):
    a = g["adj_close"]; v = g["volume"].astype(float)
    if a.iloc[-1] <= a.iloc[-2]:
        return False, "not an up day"
    dv = v.where(a < a.shift(1)).iloc[-11:-1].max()
    if not (v.iloc[-1] > dv):
        return False, f"volume {v.iloc[-1]:,.0f} <= max down-day volume {dv:,.0f}"
    return True, ""

def gap(g):
    gp = (g["adj_open"].iloc[-1] / g["adj_close"].iloc[-2] - 1) * 100
    return (gp > 4, f"gap only {gp:+.2f}%")

total = 0
total += audit("ema21_pullback", pullback, "was extended, then came back")
total += audit("high_tight_breakout", new_high, "new 20-day high today")
total += audit("donchian_breakout", new_high, "new 20-day high today")
total += audit("stage2_base_breakout", new_high_60, "new 60-day high today")
total += audit("narrow_cpr_breakout", fresh_cross("m_cpr_tc", 5), "crossed monthly TC <=5d ago")
total += audit("weekly_tc_reclaim", fresh_cross("w_cpr_tc", 3), "crossed weekly TC <=3d ago")
total += audit("monthly_r1_breakout", fresh_cross("m_r1", 3), "crossed monthly R1 <=3d ago")
total += audit("monthly_r1_retest", retest, "broke R1, came back, held")
total += audit("pocket_pivot", pocket, "up day beating 10d down-volume")
total += audit("episodic_pivot", gap, "gap > 4%")
print(f"\nTOTAL FALSE POSITIVES: {total}")
