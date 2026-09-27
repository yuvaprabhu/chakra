"""Can any screen here deliver a realised reward-to-risk of 2 or better?

A fixed target cannot: the market supplies the move or it does not, and asking
a two-week bounce for 28% just means never getting paid. The way real trend
traders get 1:3 is the opposite - a small initial stop and a TRAILING exit that
lets a winner run as long as it keeps running.

Exits tested, all with an initial stop and a hard time cap:
  ema21 / ema50   exit the next open after a close below that average
  low10           exit after a close below the lowest low of the prior 10 bars
  chand3          chandelier: exit after a close below (highest high since
                  entry) minus 3 x ATR at entry

Reported per screen and exit: realised payoff (average win / average loss),
expectancy in R, profit factor - split in-sample and out-of-sample. A cell only
counts if BOTH periods are positive AND the payoff is at least 2.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.rules import load_rules
from screener.backtest import add_panel_columns, rule_hits
from screener import quality
from screener.store import Store

SPLIT = pd.Timestamp("2025-07-01"); COST = 0.30; CAP = 60
t0 = time.time()
feat = pd.read_parquet("data/screen/features.parquet")
lat = pd.read_parquet("data/screen/latest_features.parquet")
feat = feat.merge(lat[["isin", "cap_band"]], on="isin", how="left")
st = Store("data")
bars = feat[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
feat["contaminated"] = quality.contamination_mask(
    bars, quality.detect_price_jumps(bars, st.read_adjustments())).to_numpy()
feat = add_panel_columns(feat).sort_values(["isin", "date"]).reset_index(drop=True)
g = feat.groupby("isin")
feat["low10_prior"] = g["adj_low"].transform(lambda s: s.shift(1).rolling(10).min())
print(f"{len(feat):,} rows in {time.time()-t0:.0f}s", flush=True)

O = feat["adj_open"].to_numpy(float); H = feat["adj_high"].to_numpy(float)
L = feat["adj_low"].to_numpy(float);  C = feat["adj_close"].to_numpy(float)
ATR = (feat["atr_pct"].to_numpy(float) / 100.0)
E21 = feat["ema_21"].to_numpy(float); E50 = feat["ema_50"].to_numpy(float)
LO10 = feat["low10_prior"].to_numpy(float)
ISIN = feat["isin"].to_numpy(); DATES = feat["date"].to_numpy()
LAST = pd.Series(np.arange(len(feat))).groupby(pd.factorize(feat["isin"])[0]).transform("max").to_numpy()


def trail(sig, exit_kind, stop_mult, cap=CAP):
    """One trade per signal: initial stop, trailing exit, time cap."""
    rows = []; busy = {}
    for t in np.flatnonzero(sig):
        if t + 1 > LAST[t] or busy.get(ISIN[t], -1) >= t:
            continue
        e = O[t + 1]
        if not np.isfinite(e) or e <= 0 or not np.isfinite(ATR[t]):
            continue
        risk = stop_mult * ATR[t] * e            # rupees at risk per share
        stop = e - risk
        if risk <= 0:
            continue
        peak = e; out = None; why = "time"; k = t + 1
        end = min(LAST[t], t + cap)
        while k <= end:
            if L[k] <= stop:                      # initial stop, intraday
                out = min(O[k], stop) if O[k] <= stop else stop
                why = "stop"; break
            peak = max(peak, H[k])
            lvl = (E21[k] if exit_kind == "ema21" else
                   E50[k] if exit_kind == "ema50" else
                   LO10[k] if exit_kind == "low10" else
                   peak - 3 * ATR[t] * e)
            if np.isfinite(lvl) and C[k] < lvl and k + 1 <= LAST[t]:
                out = O[k + 1]; why = exit_kind; k += 1; break
            k += 1
        if out is None:
            k = min(k, end); out = C[k]
        if not np.isfinite(out):
            continue
        busy[ISIN[t]] = k
        rows.append((DATES[t], (out / e - 1) * 100 - COST, (risk / e) * 100, k - t, why))
    return pd.DataFrame(rows, columns=["date", "net", "risk_pct", "bars", "why"])


def stats(t):
    if len(t) < 30:
        return None
    n = t["net"]; w = n[n > 0]; l = n[n <= 0]
    if not len(w) or not len(l) or l.sum() >= 0:
        return None
    return {"n": int(len(t)), "win": float((n > 0).mean() * 100),
            "avg_win": float(w.mean()), "avg_loss": float(l.mean()),
            "payoff": float(w.mean() / -l.mean()), "avg": float(n.mean()),
            "expR": float((n / t["risk_pct"]).mean()),
            "pf": float(w.sum() / -l.sum()), "bars": float(t["bars"].mean())}


rules = load_rules("config/rules.yaml")
found = []
print(f"\n{'screen':26s}{'exit':8s}{'stop':6s} | {'IS n':>6s}{'IS pay':>7s}{'IS expR':>8s}{'IS PF':>6s} | "
      f"{'OOS n':>6s}{'OOSpay':>7s}{'OOSexpR':>8s}{'OOS PF':>7s}{'bars':>6s}")
print("-" * 108)
for r in rules:
    hits = rule_hits(feat, r).to_numpy()
    if hits.sum() < 200:
        continue
    for kind in ("ema21", "ema50", "low10", "chand3"):
        for sm in (1.5, 2.5):
            t = trail(hits, kind, sm)
            if not len(t):
                continue
            t["date"] = pd.to_datetime(t["date"])
            a = stats(t[t.date < SPLIT]); b = stats(t[t.date >= SPLIT])
            if not a or not b:
                continue
            good = a["expR"] > 0 and b["expR"] > 0 and min(a["payoff"], b["payoff"]) >= 2.0
            if good:
                found.append({"rule": r.name, "title": r.title, "exit": kind, "stop": sm, "is": a, "oos": b})
                print(f"{r.title[:26]:26s}{kind:8s}{sm:<6g} | {a['n']:>6,d}{a['payoff']:>7.2f}{a['expR']:>8.3f}{a['pf']:>6.2f} | "
                      f"{b['n']:>6,d}{b['payoff']:>7.2f}{b['expR']:>8.3f}{b['pf']:>7.2f}{b['bars']:>6.1f}", flush=True)

Path("data/screen/rr_lab.json").write_text(json.dumps(found))
print(f"\n{len(found)} combination(s) cleared BOTH periods with a realised payoff of 2 or better, "
      f"in {time.time()-t0:.0f}s")
if found:
    found.sort(key=lambda x: -min(x["is"]["expR"], x["oos"]["expR"]))
    print("\nbest by the worse of the two periods:")
    for f in found[:6]:
        print(f"  {f['title'][:28]:28s} {f['exit']:7s} stop {f['stop']}x ATR | "
              f"payoff IS {f['is']['payoff']:.2f} OOS {f['oos']['payoff']:.2f} | "
              f"net/trade IS {f['is']['avg']:+.2f}% OOS {f['oos']['avg']:+.2f}% | "
              f"win {f['oos']['win']:.0f}% | held {f['oos']['bars']:.0f} bars")
