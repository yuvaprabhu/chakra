"""Out-of-the-box hardening of the Playbook: drawdown, shakeouts, fewer-but-better.

Everything is tested on the 8-year Playbook stream (2018-06..2026-09, ~3,050
signals) so the 2018-19 bear, 2020 and 2025-26 all count.  Three families:

STOP STRUCTURE (the shakeout problem)
  intraday 3 ATR (shipped) | close-based 3 ATR (exit next open only if the
  CLOSE is under the level: wicks cannot take you out) | close-based 2.5 ATR |
  two consecutive closes under the 21-EMA | "not working by day 5": if the
  trade is under water at the close of session 5, exit next open (disaster
  stop 3 ATR still on) | structure stop = 1 ATR under the 21-EMA (whichever is
  wider of that and 3 ATR)
ENTRY TIMING (buy after the hunt, not before it)
  next open (shipped) | after the shakeout bar: first session within 5 where
  the low undercuts the prior low and the close is above the prior close, buy
  next open | after confirmation: first close above the prior day's high
  within 3 sessions, buy next open | limit -2% valid 3 sessions
PORTFOLIO RULES (drawdown is an account event, not a trade event)
  10 slots (shipped) | max 2 new entries a day | pause 10 sessions after 3
  consecutive stops | pause while the last 20 trades won < 55% | halve slots
  while the account is > 10% under its high | slots scaled by breadth
  (10 above 60%, 5 at 45-60, 0 below 45)
SELECTION (fewer, better)
  both triggers (7-day low AND RSI(2) < 10) | 21-EMA band 2.6-6.7% |
  top-3 RS per day | quiet dip day (volume <= 0.8x 20-day) if volume exists
"""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd
src = open("scripts/harden_playbook.py").read().split("rows = []; busy = {}")[0]
exec(src)                                                        # panel, base, arrays, F, CAL
E21 = feat["ema_21"].to_numpy(float)
HAVE_VOL = "volume" in feat.columns
if HAVE_VOL:
    VR = (feat["volume"] / feat.groupby("isin")["volume"].transform(lambda s: s.shift(1).rolling(20).mean())).to_numpy(float)
NLO7 = feat["new_lo7"].to_numpy(); RSI2 = feat["rsi_2"].to_numpy(float); DE21 = feat["dist_ema_21"].to_numpy(float)
B50 = feat["mkt_breadth_50"].to_numpy(float)

def trades(entry="open", stop="atr3", select=None):
    rows = []; busy = {}
    for t in np.flatnonzero(base & (DATES >= np.datetime64("2018-06-01"))):
        if t + 1 > LAST[t] or busy.get(ISIN[t], -1) >= t or not np.isfinite(ATRP[t]): continue
        if select == "both" and not (NLO7[t] == 1 and RSI2[t] < 10): continue
        if select == "band" and not (DE21[t] <= 6.7): continue
        if select == "quiet" and not (HAVE_VOL and VR[t] <= 0.8): continue
        # ---- entry ----
        ei = None
        if entry == "open": ei, e = t + 1, O[t + 1]
        elif entry == "limit2":
            lim = C[t] * 0.98
            for k in range(t + 1, min(LAST[t], t + 3) + 1):
                if O[k] <= lim: ei, e = k, O[k]; break
                if L[k] <= lim: ei, e = k, lim; break
        elif entry == "shakeout":
            for k in range(t + 1, min(LAST[t] - 1, t + 5) + 1):
                if L[k] < L[k - 1] and C[k] > C[k - 1]: ei, e = k + 1, O[k + 1]; break
        elif entry == "confirm":
            for k in range(t + 1, min(LAST[t] - 1, t + 3) + 1):
                if C[k] > H[k - 1]: ei, e = k + 1, O[k + 1]; break
        if ei is None or not np.isfinite(e) or e <= 0: continue
        atr = ATRP[t] * e
        sl = e - 3 * atr; close_based = stop in ("c3", "c25", "ema2", "day5", "struct")
        if stop == "c25": sl = e - 2.5 * atr
        if stop == "struct": sl = min(e - 3 * atr, E21[ei] - atr) if np.isfinite(E21[ei]) else sl
        end = min(LAST[t], ei + 19); k = ei; out = None; why = "time"; below = 0
        while k <= end:
            if k > ei:
                if not close_based and L[k] <= sl: out = min(O[k], sl); why = "stop"; break
                if stop in ("c3", "c25", "struct", "day5") and C[k] < sl and k + 1 <= LAST[t]: out = O[k + 1]; k += 1; why = "stop"; break
                if stop == "ema2":
                    below = below + 1 if (np.isfinite(E21[k]) and C[k] < E21[k]) else 0
                    if L[k] <= e - 3 * atr: out = min(O[k], e - 3 * atr); why = "stop"; break       # disaster stop stays intraday
                    if below >= 2 and k + 1 <= LAST[t]: out = O[k + 1]; k += 1; why = "ema"; break
                if stop == "day5" and k - ei >= 5 and C[k] < e and k + 1 <= LAST[t]: out = O[k + 1]; k += 1; why = "day5"; break
                if np.isfinite(HI7[k]) and C[k] > HI7[k]:
                    if k + 1 <= LAST[t]: out = O[k + 1]; k += 1
                    else: out = C[k]
                    why = "rev7"; break
            else:
                if not close_based and L[k] <= sl: out = sl; why = "stop"; break
            k += 1
        if out is None: k = min(k, end); out = C[k]
        if not np.isfinite(out): continue
        busy[ISIN[t]] = k
        rows.append((pd.Timestamp(DATES[t]), ISIN[t], CALPOS[pd.Timestamp(DATES[ei])], CALPOS[pd.Timestamp(DATES[k])], (out / e - 1) * 100 - COST, why, F["rs_rank"][t], B50[t]))
    return pd.DataFrame(rows, columns=["date", "isin", "entry_i", "exit_i", "net", "why", "rs", "b50"])

def account(t, slots=10, max_new=99, pause_after=None, wr_pause=None, dd_half=None, breadth_slots=False, top_per_day=None):
    """10-slot book, sessions, profit at exit; rules applied at entry time only."""
    t = t.sort_values(["entry_i", "rs"], ascending=[True, False]).reset_index(drop=True)
    by_day = {k: g for k, g in t.groupby("entry_i")}
    cash = 100.0; open_pos = []; eq = np.full(len(CAL), np.nan); hist = []; streak = 0; pause_until = -1; peak = 100.0
    first = int(t.entry_i.min()); last = int(t.exit_i.max())
    for i in range(first, last + 1):
        still = []
        for p in open_pos:
            if p["exit_i"] == i:
                cash += p["amt"] * (1 + p["net"] / 100); hist.append(p["net"] > 0)
                streak = streak + 1 if p["net"] <= 0 else 0
                if pause_after and streak >= pause_after: pause_until = i + 10; streak = 0
            else: still.append(p)
        open_pos = still
        equity = cash + sum(p["amt"] for p in open_pos); peak = max(peak, equity)
        n_slots = slots
        if dd_half and equity < peak * (1 - dd_half): n_slots = slots // 2
        if breadth_slots:
            b = by_day[i].b50.iloc[0] if i in by_day else np.nan
            n_slots = 10 if b > 60 else 5 if b > 45 else 0
        if i in by_day and i > pause_until and not (wr_pause and len(hist) >= 20 and np.mean(hist[-20:]) < wr_pause):
            cands = by_day[i]
            if top_per_day: cands = cands.head(top_per_day)
            new = 0
            for _, r in cands.iterrows():
                if len(open_pos) >= n_slots or new >= max_new: break
                if any(p["isin"] == r.isin for p in open_pos): continue
                amt = equity / slots
                if amt > cash: break
                cash -= amt; open_pos.append({"isin": r.isin, "amt": amt, "net": r.net, "exit_i": int(r.exit_i)}); new += 1
        eq[i] = cash + sum(p["amt"] for p in open_pos)
    s = pd.Series(eq, index=CAL).dropna(); s.iloc[-1] = cash + sum(p["amt"] * (1 + p["net"] / 100) for p in open_pos)
    yrs = len(s) / 252; dd = (s / s.cummax() - 1).min() * 100
    q = s.resample("QE").last().pct_change().dropna() * 100
    return dict(cagr=round(((s.iloc[-1] / 100) ** (1 / yrs) - 1) * 100, 1), maxdd=round(dd, 1), pos_q=f"{int((q>0).sum())}/{len(q)}", worst_q=round(float(q.min()), 1), n_taken=len(hist))

def line(label, t, **kw):
    a = account(t, **kw); w = t.net > 0
    print(f"  {label:46s} n={len(t):5d} win {w.mean()*100:5.1f}% mean {t.net.mean():+.2f} avgL {t.net[~w].mean():+.2f} stop {(t.why=='stop').mean()*100:3.0f}% | CAGR {a['cagr']:+6.1f} maxDD {a['maxdd']:6.1f} +q {a['pos_q']:>5s} worst {a['worst_q']:+.1f} taken {a['n_taken']}", flush=True)
    return dict(label=label, n=int(len(t)), win=round(w.mean() * 100, 1), mean=round(t.net.mean(), 2), avg_loss=round(t.net[~w].mean(), 2), stopped=round((t.why == "stop").mean() * 100), **a)

R = []
base_t = trades()
print("BASELINE (shipped Playbook, 8 years)"); R.append(line("next open, intraday 3 ATR, 10 slots", base_t))
print("\nSTOP STRUCTURE"); 
for s, lbl in (("c3", "close-based 3 ATR (wicks cannot stop you)"), ("c25", "close-based 2.5 ATR"), ("struct", "1 ATR under the 21-EMA (min 3 ATR), close-based"), ("ema2", "two closes under the 21-EMA + 3 ATR disaster"), ("day5", "not working by day 5 -> out (+3 ATR disaster)")):
    R.append(line(lbl, trades(stop=s)))
print("\nENTRY TIMING")
for e, lbl in (("shakeout", "buy AFTER the shakeout bar (undercut & close up)"), ("confirm", "buy after a close above the prior high"), ("limit2", "limit -2% valid 3 sessions")):
    R.append(line(lbl, trades(entry=e)))
print("\nPORTFOLIO RULES (on the shipped trade stream)")
R.append(line("max 2 new entries per day", base_t, max_new=2))
R.append(line("pause 10 sessions after 3 straight stops", base_t, pause_after=3))
R.append(line("pause while last-20 win rate < 55%", base_t, wr_pause=0.55))
R.append(line("halve slots while account > 10% under its high", base_t, dd_half=0.10))
R.append(line("slots by breadth: 10 / 5 / 0 (>60 / 45-60 / <45)", base_t, breadth_slots=True))
R.append(line("top 3 by RS per day only", base_t, top_per_day=3))
print("\nSELECTION")
R.append(line("both triggers (7-day low AND RSI2<10)", trades(select="both")))
R.append(line("21-EMA band 2.6-6.7%", trades(select="band")))
if HAVE_VOL: R.append(line("quiet dip day (volume <= 0.8x 20d)", trades(select="quiet")))
print("\nSTACKS")
R.append(line("band + close-based 3 ATR + max 2/day", trades(stop="c3", select="band"), max_new=2))
R.append(line("band + close 3 ATR + breadth slots", trades(stop="c3", select="band"), breadth_slots=True))
R.append(line("band + shakeout entry + close 3 ATR", trades(entry="shakeout", stop="c3", select="band")))
R.append(line("band + limit -2% + close 3 ATR + max 2/day", trades(entry="limit2", stop="c3", select="band"), max_new=2))
R.append(line("both triggers + close 3 ATR + breadth slots", trades(stop="c3", select="both"), breadth_slots=True))
R.append(line("band + close 3 ATR + breadth slots + max 2/day", trades(stop="c3", select="band"), breadth_slots=True, max_new=2))
json.dump(R, open("data/screen/harden_dd.json", "w"), default=float)
print("\ndone")
