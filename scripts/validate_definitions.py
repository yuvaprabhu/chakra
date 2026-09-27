"""Independent definition audit.

This is deliberately NOT the same check as audit_screens.py. That one asks
"does each hit satisfy the rule expression" - which is circular, because a rule
whose expression is the wrong test passes it every time. Two real defects got
through exactly that way: a pullback screen with no pullback in it, and a
volatility-contraction screen that never tested a contraction.

This one asks a different question: **does the rule encode what the strategy's
canonical source actually says?** Each check below is written from the source
definition and computed from raw bars, independently of the feature frame the
rule used. Where the answer is no, it reports the fraction of today's hits that
would fail a faithful test.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd

feat = pd.read_parquet("data/screen/features.parquet")
hits = pd.read_parquet("data/screen/hits.parquet")
latest = pd.read_parquet("data/screen/latest_features.parquet")
sym = latest.set_index("isin")["symbol"].to_dict()
as_of = feat["date"].max()
panel = {i: g.sort_values("date") for i, g in feat.groupby("isin")}

def hi_of_highs(g, n):        # a channel is built on HIGHS, not closes
    return g["adj_high"].iloc[-(n+1):-1].max()
def hi_of_closes(g, n):
    return g["adj_close"].iloc[-(n+1):-1].max()

CHECKS = {}
def check(rule, what):
    def deco(fn): CHECKS.setdefault(rule, []).append((what, fn)); return fn
    return deco

# -- 1. Minervini's template is EIGHT points. Point 4 is 50 > 150 > 200. ------
@check("minervini_trend_template", "point 4: the 50 DMA is above the 150 and the 200")
def _(g, r): return r["sma_50"] > r["sma_150"] > r["sma_200"]

# -- 2. A 52-week-high breakout has to make a 52-week high. ------------------
@check("high_tight_breakout", "closes above the highest HIGH of the prior 250 sessions")
def _(g, r): return r["adj_close"] > hi_of_highs(g, 250)

# -- 3. A Donchian channel is the highest HIGH, not the highest close. -------
@check("donchian_breakout", "closes above the highest HIGH of the prior 20 sessions")
def _(g, r): return r["adj_close"] > hi_of_highs(g, 20)

# -- 4. Weinstein's average is the 30-WEEK (150 day), not the 200 day. -------
@check("stage2_base_breakout", "above a rising 30-week (150 DMA) average")
def _(g, r):
    s150 = g["adj_close"].rolling(150).mean()
    return bool(r["adj_close"] > s150.iloc[-1] and s150.iloc[-1] > s150.iloc[-21])
@check("stage2_base_breakout", "closes above the highest HIGH of the prior 60 sessions")
def _(g, r): return r["adj_close"] > hi_of_highs(g, 60)

# -- 5. Three weeks tight is three WEEKLY closes within ~1.5%. ---------------
@check("three_weeks_tight", "3 CLOSED Friday closes within 1.5% (partial week excluded)")
def _(g, r):
    w = g.set_index("date")["adj_close"].resample("W-FRI").last().dropna()
    last_day = g["date"].iloc[-1]
    if len(w) and w.index[-1] >= last_day:      # this week has not closed yet
        w = w.iloc[:-1]
    if len(w) < 3: return False
    last3 = w.iloc[-3:]
    return float((last3.max() - last3.min()) / last3.max() * 100) < 1.5

# -- 6. Academic momentum is 12-1: it EXCLUDES the most recent month. --------
@check("momentum_leaders", "12-1 momentum positive (skips the last month)")
def _(g, r):
    a = g["adj_close"]
    if len(a) < 252: return False
    return float(a.iloc[-23] / a.iloc[-253] - 1) * 100 > 15

# -- 7. A pullback should actually reach the average it is named for. --------
@check("ema21_pullback", "a session in the last 5 traded down to within 1% of the 21 EMA")
def _(g, r):
    e = g["adj_close"].ewm(span=21, adjust=False).mean()
    lo = g["adj_low"].iloc[-5:]
    return bool(((lo / e.iloc[-5:] - 1) * 100 <= 1.0).any())

# -- 8. A pocket pivot belongs inside a base, near its own short average. ----
@check("pocket_pivot", "within 5% of the 10-day average (inside the base, not extended)")
def _(g, r):
    s10 = g["adj_close"].rolling(10).mean().iloc[-1]
    return abs(float(r["adj_close"] / s10 - 1) * 100) < 5

# -- 9. A narrow CPR should be narrow against its OWN history. ---------------
# NOTE: this check uses the stock's FULL history; the rule ranks against the
# last twelve months. They disagree on names whose CPR has been narrowing for
# years - a documented difference of window, not a defect. Left divergent on
# purpose: making the check agree with the rule is how circular audits happen.
@check("narrow_cpr_breakout", "CPR width in the narrowest third of its full history")
def _(g, r):
    w = g["m_cpr_width"].dropna()
    if len(w) < 60: return False
    return float(r["m_cpr_width"]) <= float(w.quantile(0.33))

print(f"Independent definition audit, as of {as_of.date()}")
print("Each row is a canonical requirement the rule does NOT currently encode.\n")
print(f"{'screen':26s} {'hits':>5s} {'pass':>5s} {'%':>6s}  canonical requirement")
print("-" * 118)
summary = []
for rule, checks in CHECKS.items():
    h = hits[hits["rule"] == rule]
    for what, fn in checks:
        ok = fails = 0
        bad = []
        for isin in h["isin"]:
            g = panel.get(isin)
            if g is None or len(g) < 60: continue
            r = g.iloc[-1]
            try: good = bool(fn(g, r))
            except Exception: good = False
            if good: ok += 1
            else: fails += 1; bad.append(sym.get(isin, isin))
        n = ok + fails
        pctv = (ok / n * 100) if n else float("nan")
        print(f"{rule[:26]:26s} {n:>5d} {ok:>5d} {pctv:>5.1f}%  {what}")
        if bad: print(f"{'':39s}fails: {', '.join(bad[:8])}{' ...' if len(bad)>8 else ''}")
        summary.append((rule, what, n, ok))
print("\n" + "-"*118)
tot_n = sum(s[2] for s in summary); tot_ok = sum(s[3] for s in summary)
print(f"across every canonical check: {tot_ok} of {tot_n} hit-checks pass "
      f"({tot_ok/tot_n*100:.1f}%)" if tot_n else "no hits to check")
