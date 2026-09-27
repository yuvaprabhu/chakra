"""The four surviving screens, every exit, over the last two years.

Same entries, different ways out, measured the same way so they can actually be
compared. Split into two twelve-month halves rather than one block, because a
number that only holds in one of them is a number about that year.

Per trade is not comparable across exits when one holds 10 sessions and another
holds 25, so the ranking uses return per SLOT per month: what one of ten
portfolio slots earns in a month if it keeps recycling into the next signal.
A ten-slot portfolio is then simulated for real, which is the only number that
answers "which is best".
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.rules import load_rules
from screener.backtest import add_panel_columns, rule_hits, reversal_trades
from screener import quality
from screener.store import Store

COST, CAP = 0.30, 60
END = pd.Timestamp("2026-09-21")
START = END - pd.DateOffset(years=2)
MID = END - pd.DateOffset(years=1)
WANT = ["ema21_pullback", "rsi2_reversion", "leader_dip", "double_seven"]

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
print(f"panel ready: {len(feat):,} rows, {feat['isin'].nunique()} names, {time.time()-t0:.0f}s", flush=True)

O = feat["adj_open"].to_numpy(float); H = feat["adj_high"].to_numpy(float)
L = feat["adj_low"].to_numpy(float);  C = feat["adj_close"].to_numpy(float)
ATR = feat["atr_pct"].to_numpy(float) / 100.0
E21 = feat["ema_21"].to_numpy(float); E50 = feat["ema_50"].to_numpy(float)
LO10 = feat["low10_prior"].to_numpy(float)
ISIN = feat["isin"].to_numpy(); DATES = feat["date"].to_numpy()
LAST = pd.Series(np.arange(len(feat))).groupby(pd.factorize(feat["isin"])[0]).transform("max").to_numpy()
IN_WIN = (feat["date"] >= START).to_numpy() & (feat["date"] <= END).to_numpy()


def trail(sig, kind, stop_mult):
    rows = []; busy = {}
    for t in np.flatnonzero(sig):
        if t + 1 > LAST[t] or busy.get(ISIN[t], -1) >= t or not np.isfinite(ATR[t]):
            continue
        e = O[t + 1]
        if not np.isfinite(e) or e <= 0:
            continue
        risk = stop_mult * ATR[t] * e
        if risk <= 0:
            continue
        stop = e - risk; peak = e; out = None; why = "time"; k = t + 1
        end = min(LAST[t], t + CAP)
        while k <= end:
            if L[k] <= stop:
                out = min(O[k], stop); why = "stop"; break
            peak = max(peak, H[k])
            lvl = (E21[k] if kind == "ema21" else E50[k] if kind == "ema50"
                   else LO10[k] if kind == "low10" else peak - 3 * ATR[t] * e)
            if np.isfinite(lvl) and C[k] < lvl and k + 1 <= LAST[t]:
                out = O[k + 1]; why = kind; k += 1; break
            k += 1
        if out is None:
            k = min(k, end); out = C[k]
        if not np.isfinite(out):
            continue
        busy[ISIN[t]] = k
        rows.append((DATES[t], ISIN[t], e, out, (out / e - 1) * 100 - COST, (risk / e) * 100, k - t))
    return pd.DataFrame(rows, columns=["date", "isin", "entry", "exit", "net", "risk_pct", "bars"])


def portfolio(t, slots=10):
    """Ten equal slots, a slot reused the day after its trade closes."""
    if not len(t):
        return None
    d = t.sort_values("date").copy()
    d["exit_i"] = d.index
    eq = 100.0; open_pos = []; curve = []
    d["end"] = pd.to_datetime(d["date"]) + pd.to_timedelta(d["bars"], unit="D")
    for day, grp in d.groupby(pd.to_datetime(d["date"])):
        open_pos = [p for p in open_pos if p[0] > day]
        free = slots - len(open_pos)
        for r in grp.itertuples():
            if free <= 0:
                break
            open_pos.append((r.end, r.net)); free -= 1
            eq *= (1 + r.net / 100 / slots)
        curve.append((day, eq))
    s = pd.Series(dict(curve)).sort_index()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return {"total": s.iloc[-1] - 100, "cagr": (s.iloc[-1] / 100) ** (1 / max(yrs, .1)) * 100 - 100,
            "maxdd": float((s / s.cummax() - 1).min() * 100)}


def stats(t):
    if len(t) < 25:
        return None
    n = t["net"]; w = n[n > 0]; l = n[n <= 0]
    if not len(w) or not len(l) or l.sum() >= 0:
        return None
    run = mx = 0
    for x in n.to_numpy():
        run = run + 1 if x <= 0 else 0
        mx = max(mx, run)
    mo = n.groupby(pd.to_datetime(t["date"]).dt.to_period("M")).mean()
    bars = float(t["bars"].mean())
    return {"n": int(len(t)), "win": float((n > 0).mean() * 100), "avg_win": float(w.mean()),
            "avg_loss": float(l.mean()), "payoff": float(w.mean() / -l.mean()),
            "net": float(n.mean()), "pf": float(w.sum() / -l.sum()), "bars": bars,
            "per_slot_month": float(n.mean() * 21.0 / max(bars, 1)),
            "streak": int(mx), "months_up": float((mo > 0).mean() * 100),
            "worst_month": float(mo.min())}


EXITS = [("7-day high", None, None), ("50 EMA trail", "ema50", 2.5), ("50 EMA trail tight", "ema50", 1.5),
         ("21 EMA trail", "ema21", 2.5), ("10-day low trail", "low10", 2.5), ("chandelier 3 ATR", "chand3", 2.5)]
rules = {r.name: r for r in load_rules("config/rules.yaml")}
out = []
for name in WANT:
    r = rules[name]
    hits = (rule_hits(feat, r).to_numpy() & IN_WIN)
    for label, kind, sm in EXITS:
        t = (reversal_trades(feat, hits) if kind is None else trail(hits, kind, sm))
        if not len(t):
            continue
        t["date"] = pd.to_datetime(t["date"])
        t = t[(t["date"] >= START) & (t["date"] <= END)]
        y1 = stats(t[t["date"] < MID]); y2 = stats(t[t["date"] >= MID]); al = stats(t)
        if not (y1 and y2 and al):
            continue
        out.append({"screen": r.title, "exit": label, "y1": y1, "y2": y2, "all": al,
                    "port": portfolio(t)})

both = [x for x in out if x["y1"]["net"] > 0 and x["y2"]["net"] > 0]
both.sort(key=lambda x: -x["all"]["per_slot_month"])
print(f"\nTWO YEARS, {START.date()} to {END.date()}. Ranked by return per slot per month.")
print(f"Only combinations that made money in BOTH twelve-month halves are listed "
      f"({len(both)} of {len(out)} tested).\n")
hdr = (f"{'entry':24s}{'exit':19s}{'trades':>7s}{'win%':>6s}{'avgW':>7s}{'avgL':>7s}{'R:R':>6s}"
       f"{'net/tr':>8s}{'bars':>6s}{'/slot/mo':>9s}{'PF':>6s}{'lose run':>9s}")
print(hdr); print("-" * len(hdr))
for x in both:
    a = x["all"]
    print(f"{x['screen'][:24]:24s}{x['exit']:19s}{a['n']:>7,d}{a['win']:>6.0f}{a['avg_win']:>+7.1f}"
          f"{a['avg_loss']:>+7.1f}{a['payoff']:>6.2f}{a['net']:>+8.2f}{a['bars']:>6.0f}"
          f"{a['per_slot_month']:>+9.2f}{a['pf']:>6.2f}{a['streak']:>9d}")

print(f"\n{'entry':24s}{'exit':19s}| {'yr1 net':>8s}{'yr1 win':>8s} | {'yr2 net':>8s}{'yr2 win':>8s} | "
      f"{'10-slot 2yr':>12s}{'CAGR':>7s}{'maxDD':>7s}")
for x in both:
    p = x["port"] or {}
    print(f"{x['screen'][:24]:24s}{x['exit']:19s}| {x['y1']['net']:>+8.2f}{x['y1']['win']:>8.0f} | "
          f"{x['y2']['net']:>+8.2f}{x['y2']['win']:>8.0f} | {p.get('total',0):>+11.1f}%{p.get('cagr',0):>+7.1f}{p.get('maxdd',0):>7.1f}")
Path("data/screen/four_signals.json").write_text(json.dumps(both, default=float))
print(f"\nwritten to data/screen/four_signals.json in {time.time()-t0:.0f}s")
