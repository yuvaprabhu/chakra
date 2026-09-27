"""Every trade has a hard stop. Which stop, which reward rule, which size?

The constraint is fixed: an initial stop on every trade, no exceptions. What is
free is (a) how wide it sits, (b) how the winner is let go, (c) whether the stop
is moved once the trade is ahead, and (d) how much of the account each trade
risks. This script sweeps all four on three fixed entries and reports each half
of the window separately, because a structure that only works in one of them
is a curve fit, not a rule.

Conventions, all inherited from ``scripts/exit_study.py`` and ``backtest``:

* Entry at the next open. The stop is checked FIRST on every bar, so a bar that
  touches both stop and target is a stop-out. A gap through the stop fills at
  the open.
* A close-based trigger (EMA, prior low, chandelier) fills at the NEXT open. A
  stop level fills intraday at the level. A moved stop is still a stop.
* A trailing level on bar k uses data through bar k only. A stop raised because
  bar k reached +1R applies from bar k+1.
* One open position per name. 0.30% per round trip. 40-session time cap, closed
  at that bar's close.
* MAE (max adverse excursion) comes from ``data/screen/trades_tagged.parquet``,
  the intraday low of every trade relative to its entry, which is the direct
  measure of what a stop width does to eventual winners.
"""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.rules import load_rules
from screener.backtest import add_panel_columns, rule_hits
from screener.portfolio import simulate
from screener.sizing import simulate_risk
from screener import quality
from screener.store import Store

COST, CAP, SLOTS = 0.30, 40, 10
START, MID = pd.Timestamp("2023-09-19"), pd.Timestamp("2025-03-19")
ENTRIES = ["leader_dip", "minervini_trend_template", "momentum_leaders"]
BETA_WIN, BETA_MIN = 250, 1.20
STOPS = [("pct", 3.0), ("pct", 5.0), ("pct", 7.0),
         ("atr", 1.0), ("atr", 1.5), ("atr", 2.0), ("atr", 2.5), ("atr", 3.0)]
REWARDS = [("target", 2.0), ("target", 3.0), ("trail", "ema21"), ("trail", "ema50"),
           ("trail", "low10"), ("trail", "chand3")]
MGMTS = ["be", "half", "low5"]
RISKS = [0.5, 1.0, 1.5]
MIN_N = 150            # per half, before a structure is even considered

t0 = time.time()
feat = pd.read_parquet("data/screen/features.parquet")
lat = pd.read_parquet("data/screen/latest_features.parquet")
feat = feat.merge(lat[["isin", "cap_band"]], on="isin", how="left")
st = Store("data")
b = feat[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
feat["contaminated"] = quality.contamination_mask(b, quality.detect_price_jumps(b, st.read_adjustments())).to_numpy()
feat = add_panel_columns(feat).sort_values(["isin", "date"]).reset_index(drop=True)
g = feat.groupby("isin")
feat["low10_prior"] = g["adj_low"].transform(lambda s: s.shift(1).rolling(10).min())
feat["low5_prior"] = g["adj_low"].transform(lambda s: s.shift(1).rolling(5).min())
feat["ret1"] = g["adj_close"].pct_change()
feat["liquid"] = ((feat["turnover_median_20d"] >= 1e7) & (feat["bars_available"] >= 250)
                  & (~feat["contaminated"]))

CAL = pd.DatetimeIndex(sorted(feat["date"].unique()))
CALPOS = {d: i for i, d in enumerate(CAL)}
END = CAL[-1]
mkt = feat.loc[feat["liquid"]].groupby("date")["ret1"].mean().reindex(CAL).fillna(0)
BENCH = (1 + mkt).cumprod()
feat["mret"] = feat["date"].map(mkt)


def beta_of(s):
    """Trailing-250 beta against the equal-weight liquid universe, shifted a
    day so today's bar never sits inside today's beta (as scripts/highbeta_swing.py)."""
    x, y = s["mret"], s["ret1"]
    cov = y.rolling(BETA_WIN, min_periods=120).cov(x)
    var = x.rolling(BETA_WIN, min_periods=120).var()
    return (cov / var).shift(1)


feat["beta"] = feat.groupby("isin", group_keys=False).apply(beta_of)
HIGH = (feat["beta"] >= BETA_MIN).to_numpy() & feat["liquid"].to_numpy()

O = feat["adj_open"].to_numpy(float); H = feat["adj_high"].to_numpy(float)
L = feat["adj_low"].to_numpy(float);  C = feat["adj_close"].to_numpy(float)
ATRP = feat["atr_pct"].to_numpy(float) / 100.0
E21 = feat["ema_21"].to_numpy(float); E50 = feat["ema_50"].to_numpy(float)
LO10 = feat["low10_prior"].to_numpy(float); LO5 = feat["low5_prior"].to_numpy(float)
RS = feat["rs_rank"].to_numpy(float)
ISIN = feat["isin"].to_numpy(); DATES = feat["date"].to_numpy()
LAST = pd.Series(np.arange(len(feat))).groupby(pd.factorize(feat["isin"])[0]).transform("max").to_numpy()
IN_WIN = ((feat["date"] >= START) & (feat["date"] <= END)).to_numpy()
print(f"panel {len(feat):,} rows | {CAL[0].date()} to {END.date()} | signals from {START.date()}, "
      f"halves split {MID.date()} | {time.time()-t0:.0f}s", flush=True)


def run(sig, stop, reward, mgmt=None):
    """One trade per signal, with a hard initial stop.

    stop:   ('pct', w) a flat w% below entry | ('atr', m) m x ATR(14)% at the signal bar
    reward: ('target', rr) a fixed target at rr x the stop distance
            ('trail', 'ema21'|'ema50'|'low10'|'chand3') exit at the next open after a close below
            ('none',) nothing but the stop and the time cap
    mgmt:   None | 'be' stop to entry once +1R traded
            'half' half out at +1R, the rest under a 2-ATR stop from the highest high
            'low5' stop to the prior 5-day low every day once +1R traded
    """
    rows = []; busy = {}
    for t in np.flatnonzero(sig):
        if t + 1 > LAST[t] or busy.get(ISIN[t], -1) >= t or not np.isfinite(ATRP[t]):
            continue
        e = O[t + 1]
        if not np.isfinite(e) or e <= 0:
            continue
        risk = e * stop[1] / 100.0 if stop[0] == "pct" else stop[1] * ATRP[t] * e
        rpct = risk / e * 100.0
        if not (0.2 < rpct < 20):
            continue
        sl = e - risk; sl0 = sl
        tgt = e + reward[1] * risk if reward[0] == "target" else np.inf
        one_r = e + risk; atr_abs = ATRP[t] * e
        end = min(LAST[t], t + CAP)
        k = t + 1; peak = e; armed = False; half_px = None
        out = None; why = "time"
        while k <= end:
            # A raised stop from the prior 5-day low: the level is known at the
            # open of bar k (it excludes bar k), so it is live on bar k.
            if mgmt == "low5" and armed and np.isfinite(LO5[k]):
                sl = max(sl, LO5[k])
            if L[k] <= sl:
                out = min(O[k], sl); why = "stop" if sl <= sl0 else "tstop"; break
            if reward[0] == "target" and H[k] >= tgt:
                out = max(O[k], tgt); why = "target"; break
            if mgmt and not armed and H[k] >= one_r:
                armed = True
                if mgmt == "be":
                    sl = max(sl, e)                       # live from the next bar
                elif mgmt == "half":
                    half_px = max(O[k], one_r)            # first half out at +1R
            peak = max(peak, H[k])
            if mgmt == "half" and armed:
                sl = max(sl, peak - 2.0 * atr_abs)        # live from the next bar
            if reward[0] == "trail":
                lvl = (E21[k] if reward[1] == "ema21" else E50[k] if reward[1] == "ema50"
                       else LO10[k] if reward[1] == "low10" else peak - 3.0 * atr_abs)
                if np.isfinite(lvl) and C[k] < lvl and k + 1 <= LAST[t]:
                    out = O[k + 1]; why = "trail"; k += 1; break
            k += 1
        if out is None:
            k = min(k, end); out = C[k]
        if not np.isfinite(out):
            continue
        busy[ISIN[t]] = k
        gross = (out / e - 1) if half_px is None else 0.5 * (half_px / e - 1) + 0.5 * (out / e - 1)
        ei, xi = CALPOS[pd.Timestamp(DATES[t + 1])], CALPOS[pd.Timestamp(DATES[k])]
        bench = (BENCH.iloc[xi] / BENCH.iloc[ei] - 1) * 100
        rows.append((pd.Timestamp(DATES[t]), ISIN[t], ei, xi, gross * 100 - COST, bench, k - t,
                     rpct, RS[t], why, half_px is not None))
    return pd.DataFrame(rows, columns=["date", "isin", "entry_i", "exit_i", "net", "bench", "bars",
                                       "risk_pct", "rs", "why", "scaled"])


def stats(t):
    if len(t) < MIN_N:
        return None
    n = t["net"]; w = n[n > 0]; l = n[n <= 0]
    if not len(w) or not len(l):
        return None
    why = t["why"].value_counts(normalize=True) * 100
    return {"n": int(len(t)), "win": float((n > 0).mean() * 100), "avg_win": float(w.mean()),
            "avg_loss": float(l.mean()), "payoff": float(w.mean() / -l.mean()) if l.mean() < 0 else None,
            "net": float(n.mean()), "exp_r": float((n / t["risk_pct"]).mean()),
            "pf": float(w.sum() / -l.sum()) if l.sum() < 0 else None,
            "stop": float(why.get("stop", 0)), "tstop": float(why.get("tstop", 0)),
            "target": float(why.get("target", 0)), "trail": float(why.get("trail", 0)),
            "time": float(why.get("time", 0)), "bars": float(t["bars"].mean()),
            "risk": float(t["risk_pct"].mean()), "bench": float(t["bench"].mean()),
            "excess": float((n - t["bench"]).mean()), "days": int(t["date"].nunique())}


def label(stop, reward, mgmt):
    s = f"{stop[1]:g}%" if stop[0] == "pct" else f"{stop[1]:g} ATR"
    r = (f"1:{reward[1]:g}" if reward[0] == "target" else
         {"ema21": "close<21EMA", "ema50": "close<50EMA", "low10": "close<10d low",
          "chand3": "chandelier 3ATR"}.get(reward[1], reward[0]) if reward[0] == "trail" else "stop only")
    m = {None: "", "be": " +BE@1R", "half": " +half@1R/2ATR", "low5": " +5d-low@1R"}[mgmt]
    return s, r + m


def one(job):
    entry, uni, stop, reward, mgmt = job
    t = TRADES_SIG[(entry, uni)]
    tr = run(t, stop, reward, mgmt)
    if not len(tr):
        return None
    h1, h2 = tr[tr["date"] < MID], tr[tr["date"] >= MID]
    a, s1, s2 = stats(tr), stats(h1), stats(h2)
    if not (a and s1 and s2):
        return None
    sl, rl = label(stop, reward, mgmt)
    both = s1["net"] > 0 and s2["net"] > 0
    return {"entry": entry, "universe": uni, "stop": sl, "reward": rl, "stop_spec": list(stop),
            "reward_spec": list(reward), "mgmt": mgmt, "all": a, "h1": s1, "h2": s2, "both": bool(both),
            "worse_r": float(min(s1["exp_r"], s2["exp_r"])), "worse_net": float(min(s1["net"], s2["net"]))}


rules = {r.name: r for r in load_rules("config/rules.yaml")}
TITLE = {n: rules[n].title for n in ENTRIES}
TRADES_SIG = {}
for n in ENTRIES:
    h = rule_hits(feat, rules[n]).to_numpy() & IN_WIN
    TRADES_SIG[(n, "all")] = h
    TRADES_SIG[(n, "highbeta")] = h & HIGH
    print(f"  {TITLE[n]:20s} signals {h.sum():>7,d}  high-beta {(h & HIGH).sum():>7,d}  "
          f"first {feat.loc[h, 'date'].min().date()}", flush=True)

# ------------------------------------------------------------- D: MAE first
tagged = pd.read_parquet("data/screen/trades_tagged.parquet")
mae = []
for s, grp in tagged.groupby("strategy"):
    win = grp[grp["net"] > 0]; big = grp[grp["mfe"] >= 10]
    mae.append({"strategy": s, "n": int(len(grp)), "winners": int(len(win)),
                "winner_mae_median": float(win["mae"].median()),
                "winner_mae_p25": float(win["mae"].quantile(0.25)),
                **{f"win_beyond_{x}": float((win["mae"] <= -x).mean() * 100) for x in (3, 5, 7, 10)},
                "reached10": int(len(big)),
                **{f"r10_beyond_{x}": float((big["mae"] <= -x).mean() * 100) for x in (3, 5, 7, 10)},
                "atr_median": float(grp["atr_pct"].median())})
print("\nD. MAE OF EVENTUAL WINNERS (trades_tagged.parquet, intraday low vs entry)")
print(f"{'strategy':32s}{'n':>7s}{'wins':>6s}{'medMAE':>8s}{'p25':>7s} | winners lost by a stop at "
      f"{'3%':>6s}{'5%':>6s}{'7%':>6s}{'10%':>6s} | of +10% MFE trades: {'3%':>5s}{'5%':>6s}{'7%':>6s}")
for m in mae:
    print(f"{m['strategy'][:32]:32s}{m['n']:>7,d}{m['winners']:>6,d}{m['winner_mae_median']:>8.2f}{m['winner_mae_p25']:>7.2f} | "
          f"{'':26s}{m['win_beyond_3']:>6.1f}{m['win_beyond_5']:>6.1f}{m['win_beyond_7']:>6.1f}{m['win_beyond_10']:>6.1f} | "
          f"{'':20s}{m['r10_beyond_3']:>5.1f}{m['r10_beyond_5']:>6.1f}{m['r10_beyond_7']:>6.1f}")

# ------------------------------------------------------------- A and B: the grid
jobs = [(n, u, s, r, None) for n in ENTRIES for u in ("all", "highbeta") for s in STOPS for r in REWARDS]
jobs += [(n, "all", s, r, m) for n in ENTRIES for s in STOPS for r in REWARDS + [("none",)] for m in MGMTS
         if not (r[0] == "none" and m != "half")]
print(f"\n{len(jobs)} structures to run ...", flush=True)
with Pool(4) as pool:
    res = [x for x in pool.map(one, jobs, chunksize=8) if x]
print(f"{len(res)} produced enough trades in both halves | {time.time()-t0:.0f}s", flush=True)


def show(rows, title):
    print(f"\n{title}")
    print(f"{'entry':18s}{'stop':>8s} {'reward':26s}{'n':>6s}{'win%':>5s}{'aW':>6s}{'aL':>6s}{'pay':>5s}"
          f"{'expR':>6s}{'PF':>5s}{'stp%':>5s}{'bars':>5s} | {'H1 net':>7s}{'H1 R':>6s} | {'H2 net':>7s}{'H2 R':>6s}  ok")
    print("-" * 132)
    for x in rows:
        a, h1, h2 = x["all"], x["h1"], x["h2"]
        print(f"{TITLE[x['entry']][:18]:18s}{x['stop']:>8s} {x['reward']:26s}{a['n']:>6,d}{a['win']:>5.0f}{a['avg_win']:>6.2f}"
              f"{a['avg_loss']:>6.2f}{a['payoff']:>5.2f}{a['exp_r']:>6.2f}{a['pf']:>5.2f}{a['stop']+a['tstop']:>5.0f}{a['bars']:>5.0f} | "
              f"{h1['net']:>+7.2f}{h1['exp_r']:>+6.2f} | {h2['net']:>+7.2f}{h2['exp_r']:>+6.2f}  {'YES' if x['both'] else ''}")


A = sorted([x for x in res if x["mgmt"] is None and x["universe"] == "all"],
           key=lambda x: (x["entry"], x["stop_spec"][0] != "pct", x["stop_spec"][1], x["reward"]))
show(A, "A. STOP WIDTH x REWARD RULE (all liquid names)")
AH = sorted([x for x in res if x["mgmt"] is None and x["universe"] == "highbeta"],
            key=lambda x: (x["entry"], x["stop_spec"][0] != "pct", x["stop_spec"][1], x["reward"]))
show(AH, "A'. SAME GRID, HIGH-BETA NAMES ONLY (beta >= 1.2)")
B = sorted([x for x in res if x["mgmt"] is not None], key=lambda x: -x["worse_r"])
show(B[:40], "B. STOP MANAGEMENT, top 40 by the worse half's expectancy (all liquid names)")

ok = sorted([x for x in res if x["both"] and x["universe"] == "all"], key=lambda x: -x["worse_r"])
print(f"\n{len(ok)} of {len([x for x in res if x['universe']=='all'])} all-universe structures positive in BOTH halves.")
show(ok[:15], "RECOMMENDABLE, ranked by the worse half's expectancy in R")
least_bad = sorted([x for x in res if x["universe"] == "all"], key=lambda x: -x["worse_net"])[:5]

# ------------------------------------------------------------- C: the account
picks = []
seen = set()
for x in ok + least_bad:
    key = (x["entry"], x["stop"], x["reward"])
    if key in seen:
        continue
    seen.add(key); picks.append(x)
    if len(picks) == 4:
        break


N_RANDOM = 100
rng = np.random.default_rng(7)


def spread(tr, sizer):
    """The rule's own rs ordering is one arbitrary choice of who gets a contested
    slot; N_RANDOM shuffles show how much of the result is that choice."""
    tot = np.array([sizer(tr.iloc[rng.permutation(len(tr))].sort_values("entry_i", kind="stable"))["total"]
                    for _ in range(N_RANDOM)])
    return {"rnd_median": float(np.median(tot)), "rnd_p5": float(np.percentile(tot, 5)),
            "rnd_p95": float(np.percentile(tot, 95)), "rnd_pos": float((tot > 0).mean() * 100)}


def account(tr, label_):
    tr = tr.sort_values(["entry_i", "rs"], ascending=[True, False])
    out = {"label": label_, "n": int(len(tr))}
    eq = simulate(tr, slots=SLOTS, sessions=CAL)
    m = eq["equity"].groupby(eq["equity"].index.to_period("M")).last()
    m = pd.concat([pd.Series([100.0]), m.reset_index(drop=True)]).pct_change().dropna() * 100
    out["equal"] = {"taken": eq["n_taken"], "total": eq["total"], "cagr": eq["cagr"], "maxdd": eq["maxdd"],
                    "worst_month": float(m.min()) if len(m) else None,
                    # ten slots of 10% each, every one at its stop: the mean stop distance
                    "risk_when_loaded": float(tr["risk_pct"].mean()),
                    **spread(tr, lambda t: simulate(t, slots=SLOTS, sessions=CAL))}
    out["risk"] = {}
    for r in RISKS:
        s = simulate_risk(tr, risk_per_trade=r, max_positions=SLOTS, sessions=CAL)
        out["risk"][str(r)] = {k: s[k] for k in ("n_taken", "declined_cash", "total", "cagr", "maxdd", "worst_month",
                                                 "months_up", "max_open_risk", "risk_when_loaded", "avg_open_risk",
                                                 "sessions_loaded", "avg_open", "max_weight")}
        out["risk"][str(r)].update(spread(tr, lambda t, r=r: simulate_risk(t, risk_per_trade=r, max_positions=SLOTS, sessions=CAL)))
    return out


ACCT = []
print("\nC. THE ACCOUNT: equal money (10 slots) vs fixed-fractional risk, max 10 open, no leverage")
print("   'rs order' takes the highest rs_rank first on a contested day; 'random' is the median and 5th-95th of "
      f"{N_RANDOM} shuffles")
print(f"{'structure':62s}{'sizing':>10s}{'took':>6s}{'total':>8s}{'CAGR':>7s}{'maxDD':>7s}{'wMonth':>8s}"
      f"{'risk@full':>10s}{'maxWt':>7s} | {'random med':>10s}{'p5':>7s}{'p95':>7s}")
print("-" * 150)
for x in picks:
    for uni in ("all", "highbeta"):
        tr = run(TRADES_SIG[(x["entry"], uni)], tuple(x["stop_spec"]), tuple(x["reward_spec"]), x["mgmt"])
        name = f"{TITLE[x['entry']]}{' HB' if uni == 'highbeta' else ''} | {x['stop']} stop | {x['reward']}"
        for span, sub in (("full", tr), ("H1", tr[tr["date"] < MID]), ("H2", tr[tr["date"] >= MID])):
            a = account(sub, f"{name} [{span}]")
            a["span"] = span; a["entry"] = x["entry"]; a["universe"] = uni; a["stop"] = x["stop"]; a["reward"] = x["reward"]
            ACCT.append(a)
            e = a["equal"]
            print(f"{a['label'][:62]:62s}{'equal/10':>10s}{e['taken']:>6d}{e['total']:>+8.1f}{e['cagr']:>+7.1f}{e['maxdd']:>7.1f}"
                  f"{e['worst_month']:>+8.1f}{e['risk_when_loaded']:>9.1f}%{100/SLOTS:>7.0f} | "
                  f"{e['rnd_median']:>+10.1f}{e['rnd_p5']:>+7.1f}{e['rnd_p95']:>+7.1f}")
            for r in RISKS:
                s = a["risk"][str(r)]
                rl = s["risk_when_loaded"] if s["risk_when_loaded"] is not None else s["max_open_risk"]
                print(f"{'':62s}{f'risk {r:g}%':>10s}{s['n_taken']:>6d}{s['total']:>+8.1f}{s['cagr']:>+7.1f}{s['maxdd']:>7.1f}"
                      f"{s['worst_month']:>+8.1f}{rl:>9.1f}%{s['max_weight']:>7.0f} | "
                      f"{s['rnd_median']:>+10.1f}{s['rnd_p5']:>+7.1f}{s['rnd_p95']:>+7.1f}")

bh = BENCH.loc[START:END]
yrs = (bh.index[-1] - bh.index[0]).days / 365.25
bench = {"total": float((bh.iloc[-1] / bh.iloc[0] - 1) * 100),
         "cagr": float(((bh.iloc[-1] / bh.iloc[0]) ** (1 / yrs) - 1) * 100),
         "maxdd": float((bh / bh.cummax() - 1).min() * 100),
         "h1": float((bh.loc[:MID].iloc[-1] / bh.iloc[0] - 1) * 100),
         "h2": float((bh.iloc[-1] / bh.loc[MID:].iloc[0] - 1) * 100)}
print(f"\nequal-weight liquid universe, buy and hold: total {bench['total']:+.1f}%  CAGR {bench['cagr']:+.1f}%  "
      f"maxDD {bench['maxdd']:.1f}%  H1 {bench['h1']:+.1f}%  H2 {bench['h2']:+.1f}%")

Path("data/screen/stop_reward.json").write_text(json.dumps({
    "window": {"start": str(START.date()), "mid": str(MID.date()), "end": str(END.date()),
               "first_signal": str(min(feat.loc[m, "date"].min() for m in TRADES_SIG.values()).date()),
               "cost": COST, "cap": CAP, "slots": SLOTS, "min_n_per_half": MIN_N},
    "mae": mae, "grid": res, "recommendable": [x for x in ok], "picks": picks, "account": ACCT,
    "benchmark": bench}, default=float))
print(f"\ndone in {time.time()-t0:.0f}s")
