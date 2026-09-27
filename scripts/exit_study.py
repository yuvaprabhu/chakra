"""Five entries, every popular exit, two years, measured honestly.

What an independent audit of the earlier version changed:

* RAW return is not the answer. A screen that buys stocks in a rising market
  earns the market. Every number here is EXCESS over an equal-weight index of
  the same liquid universe, over each trade's own holding window.
* 5,328 trades are not 5,328 independent observations. They land on ~470
  signal days, so the t-statistic is clustered by day.
* A ten-slot portfolio cannot take every signal. On a busy day it must choose,
  and an arbitrary choice flatters or ruins the result, so the portfolio is run
  on the rule's own ranking AND on 200 random orderings to show the spread.
* Slots are released in SESSIONS, and profit lands on the EXIT.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.rules import load_rules
from screener.backtest import add_panel_columns, rule_hits
from screener.portfolio import simulate
from screener import quality
from screener.store import Store

COST, CAP = 0.30, 60
END = pd.Timestamp("2026-09-21"); START = END - pd.DateOffset(years=2)
MID = END - pd.DateOffset(years=1)
ENTRIES = ["leader_dip", "rsi2_reversion", "ema21_pullback", "double_seven", "oversold_in_uptrend"]
N_RANDOM, SLOTS = 200, 10

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

CAL = pd.DatetimeIndex(sorted(feat["date"].unique()))
CALPOS = {d: i for i, d in enumerate(CAL)}
O = feat["adj_open"].to_numpy(float); H = feat["adj_high"].to_numpy(float)
L = feat["adj_low"].to_numpy(float);  C = feat["adj_close"].to_numpy(float)
ATRP = feat["atr_pct"].to_numpy(float) / 100.0
E21 = feat["ema_21"].to_numpy(float); E50 = feat["ema_50"].to_numpy(float)
LO10 = feat["low10_prior"].to_numpy(float); HI7 = feat["hi7_prior"].to_numpy(float)
RS = feat["rs_rank"].to_numpy(float)
ISIN = feat["isin"].to_numpy(); DATES = feat["date"].to_numpy()
LAST = pd.Series(np.arange(len(feat))).groupby(pd.factorize(feat["isin"])[0]).transform("max").to_numpy()
IN_WIN = ((feat["date"] >= START) & (feat["date"] <= END)).to_numpy()

# Equal-weight index of the liquid universe: what the screen is competing with.
elig = feat[(feat["turnover_median_20d"] >= 1e7) & (feat["bars_available"] >= 250) & (~feat["contaminated"])]
day_ret = elig.assign(r=elig["adj_close"] / elig.groupby("isin")["adj_close"].shift(1) - 1) \
              .groupby("date")["r"].mean()
BENCH = (1 + day_ret.fillna(0)).cumprod().reindex(CAL).ffill().bfill()
print(f"panel {len(feat):,} rows | benchmark from {elig['isin'].nunique():,} liquid names | {time.time()-t0:.0f}s", flush=True)


def run(sig, spec):
    """spec: ('rev',) | ('bracket', rr, stop_mult) | ('trail', kind, stop_mult)"""
    kind = spec[0]
    rows = []; busy = {}
    for t in np.flatnonzero(sig):
        if t + 1 > LAST[t] or busy.get(ISIN[t], -1) >= t or not np.isfinite(ATRP[t]):
            continue
        e = O[t + 1]
        if not np.isfinite(e) or e <= 0:
            continue
        end = min(LAST[t], t + CAP); k = t + 1; out = None; why = "time"
        if kind == "rev":
            risk = np.nan
            cap = min(LAST[t], t + 20)
            while k <= cap:
                if np.isfinite(HI7[k]) and C[k] > HI7[k]:
                    out, why = (O[k + 1], "reversal") if k + 1 <= LAST[t] else (C[k], "reversal")
                    if k + 1 <= LAST[t]:
                        k += 1
                    break
                k += 1
            if out is None:
                k = min(t + 20, LAST[t]); out = C[k]
        else:
            sm = spec[2]; risk = sm * ATRP[t] * e
            if risk <= 0:
                continue
            stop = e - risk
            tgt = e + spec[1] * risk if kind == "bracket" else np.inf
            peak = e
            while k <= end:
                if L[k] <= stop:
                    out = min(O[k], stop); why = "stop"; break
                if kind == "bracket" and H[k] >= tgt:
                    out = max(O[k], tgt); why = "target"; break
                peak = max(peak, H[k])
                if kind == "trail":
                    lvl = (E21[k] if spec[1] == "ema21" else E50[k] if spec[1] == "ema50"
                           else LO10[k] if spec[1] == "low10" else peak - 3 * ATRP[t] * e)
                    if np.isfinite(lvl) and C[k] < lvl and k + 1 <= LAST[t]:
                        out = O[k + 1]; why = spec[1]; k += 1; break
                k += 1
            if out is None:
                k = min(k, end); out = C[k]
        if not np.isfinite(out):
            continue
        busy[ISIN[t]] = k
        ei, xi = CALPOS[pd.Timestamp(DATES[t + 1])], CALPOS[pd.Timestamp(DATES[k])]
        bench = (BENCH.iloc[xi] / BENCH.iloc[ei] - 1) * 100
        rows.append((pd.Timestamp(DATES[t]), ISIN[t], ei, xi,
                     (out / e - 1) * 100 - COST, bench, k - t,
                     (risk / e * 100) if np.isfinite(risk) else np.nan, RS[t], why))
    return pd.DataFrame(rows, columns=["date", "isin", "entry_i", "exit_i", "net", "bench",
                                       "bars", "risk_pct", "rs", "why"])


def clustered_t(x, day):
    """One observation per signal day, so 70 trades on one day count once."""
    m = pd.Series(np.asarray(x)).groupby(np.asarray(day)).mean()
    return float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m)))) if len(m) > 2 and m.std() > 0 else 0.0


def stats(t):
    if len(t) < 25:
        return None
    n = t["net"]; exc = t["net"] - t["bench"]; w = n[n > 0]; l = n[n <= 0]
    if not len(w) or not len(l) or l.sum() >= 0:
        return None
    return {"n": int(len(t)), "win": float((n > 0).mean() * 100), "avg_win": float(w.mean()),
            "avg_loss": float(l.mean()), "payoff": float(w.mean() / -l.mean()),
            "net": float(n.mean()), "bench": float(t["bench"].mean()), "excess": float(exc.mean()),
            "t_excess": clustered_t(exc, t["date"]), "pf": float(w.sum() / -l.sum()),
            "bars": float(t["bars"].mean()), "days": int(t["date"].nunique())}


def port(t, rng):
    """Ten slots: the rule's own ranking, then 200 arbitrary orderings."""
    if not len(t):
        return None
    byrs = t.sort_values(["entry_i", "rs"], ascending=[True, False])
    base = simulate(byrs, slots=SLOTS, sessions=CAL)
    tot = []
    for _ in range(N_RANDOM):
        sh = t.iloc[rng.permutation(len(t))].sort_values("entry_i", kind="stable")
        tot.append(simulate(sh, slots=SLOTS, sessions=CAL)["total"])
    tot = np.array(tot)
    return {"rs_total": base["total"], "rs_cagr": base["cagr"], "rs_maxdd": base["maxdd"],
            "taken": base["n_taken"], "offered": base["n_offered"],
            "rnd_mean": float(tot.mean()), "rnd_p5": float(np.percentile(tot, 5)),
            "rnd_p95": float(np.percentile(tot, 95))}


EXITS = [("7-day high (no stop)", ("rev",))] \
    + [(f"1:{r:g} bracket", ("bracket", float(r), 2.5)) for r in (2, 3, 4, 5)] \
    + [("close < 50 EMA", ("trail", "ema50", 2.5)), ("close < 21 EMA", ("trail", "ema21", 2.5)),
       ("close < 10-day low", ("trail", "low10", 2.5)), ("chandelier 3 ATR", ("trail", "chand3", 2.5))]
rules = {r.name: r for r in load_rules("config/rules.yaml")}
rng = np.random.default_rng(7)
out = []
print(f"\n{'entry':22s}{'exit':21s}{'n':>6s}{'win%':>6s}{'R:R':>6s}{'net':>7s}{'bench':>7s}"
      f"{'EXCESS':>8s}{'t':>6s}{'bars':>6s} | {'h1 exc':>7s}{'h2 exc':>7s}  both?")
print("-" * 118)
for name in ENTRIES:
    r = rules[name]
    hits = rule_hits(feat, r).to_numpy() & IN_WIN
    for label, spec in EXITS:
        t = run(hits, spec)
        if not len(t):
            continue
        t = t[(t["date"] >= START) & (t["date"] <= END)]
        a, h1, h2 = stats(t), stats(t[t["date"] < MID]), stats(t[t["date"] >= MID])
        if not (a and h1 and h2):
            continue
        both = h1["excess"] > 0 and h2["excess"] > 0
        out.append({"entry": r.title, "exit": label, "all": a, "h1": h1, "h2": h2,
                    "both": bool(both), "port": port(t, rng) if both else None})
        print(f"{r.title[:22]:22s}{label:21s}{a['n']:>6,d}{a['win']:>6.0f}{a['payoff']:>6.2f}"
              f"{a['net']:>+7.2f}{a['bench']:>+7.2f}{a['excess']:>+8.2f}{a['t_excess']:>6.2f}"
              f"{a['bars']:>6.0f} | {h1['excess']:>+7.2f}{h2['excess']:>+7.2f}  {'YES' if both else ''}",
              flush=True)
Path("data/screen/exit_study.json").write_text(json.dumps(out, default=float))
win = [x for x in out if x["both"]]
win.sort(key=lambda x: -min(x["h1"]["excess"], x["h2"]["excess"]))
print(f"\n{len(win)} of {len(out)} combinations beat their own universe in BOTH halves.\n")
if win:
    print(f"{'entry':22s}{'exit':21s}{'excess':>8s}{'t':>6s} | {'10 slots, rule order':>21s}{'CAGR':>7s}{'maxDD':>7s} | "
          f"{'200 random orders: mean':>25s}{'5th':>7s}{'95th':>7s}{'took':>6s}")
    for x in win:
        p = x["port"]
        print(f"{x['entry'][:22]:22s}{x['exit']:21s}{x['all']['excess']:>+8.2f}{x['all']['t_excess']:>6.2f} | "
              f"{p['rs_total']:>+20.1f}%{p['rs_cagr']:>+7.1f}{p['rs_maxdd']:>7.1f} | "
              f"{p['rnd_mean']:>+24.1f}%{p['rnd_p5']:>+7.1f}{p['rnd_p95']:>+7.1f}{p['taken']:>6d}")
print(f"\ndone in {time.time()-t0:.0f}s")
