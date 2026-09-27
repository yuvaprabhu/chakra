"""Three things from the literature the screens have not used.

1. A MARKET REGIME GATE. Faber (2007): hold only when the index is above its
   10-month average. Minervini and O'Neil both refuse to buy breakouts in a
   market that is under distribution. Every screen here lost money in the same
   window - the one where the equal-weight universe fell - so the question is
   whether staying out of that window is worth more than any entry rule.

2. CONNORS' DOUBLE SEVEN. Buy a new 7-day closing low in a stock above its
   200-day average; sell on a new 7-day closing high. No stop, no target: the
   exit is the reversal itself. Published, simple, and the mechanism is the
   same short-term reversal RSI(2) exploits.

3. FROG IN THE PAN (Da, Gurun, Warachka 2014). Momentum works better when the
   path was smooth - many small up days - than when it was one jump. Rank the
   momentum portfolio by return AND by the share of positive days.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.rules import Rule, load_rules
from screener.backtest import (add_panel_columns, add_forward_returns, rule_hits,
                               bracket_returns, HORIZONS)
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
feat = feat.sort_values(["isin", "date"]).reset_index(drop=True)

# ---- the regime series, built from the universe itself, causally --------------
byd = feat.groupby("date")
ew_ret = byd["ret_1d"].mean() / 100                       # equal-weight daily return
idx = (1 + ew_ret.fillna(0)).cumprod()
sma200 = idx.rolling(200).mean(); sma50 = idx.rolling(50).mean()
breadth50 = byd.apply(lambda g: (g["adj_close"] > g["sma_50"]).mean() * 100)
regime = pd.DataFrame({"idx": idx, "sma200": sma200, "sma50": sma50, "breadth": breadth50})
regime["faber"] = regime["idx"] > regime["sma200"]
regime["fast"] = regime["idx"] > regime["sma50"]
regime["breadth_ok"] = regime["breadth"] > 50
regime["both"] = regime["faber"] & regime["breadth_ok"]
feat = feat.merge(regime[["faber", "fast", "breadth_ok", "both"]], left_on="date", right_index=True, how="left")
on_share = {k: regime.loc[regime.index >= SPLIT, k].mean() * 100 for k in ("faber", "fast", "breadth_ok", "both")}
print("REGIME - share of sessions 'on', out-of-sample window:",
      {k: f"{v:.0f}%" for k, v in on_share.items()})
print("        in-sample window:", {k: f"{regime.loc[(regime.index < SPLIT) & (regime.index >= regime.index[200]), k].mean()*100:.0f}%" for k in ("faber","fast","breadth_ok","both")})

IS, OOS = feat[feat["date"] < SPLIT], feat[feat["date"] >= SPLIT]
BASE = {r.name: r for r in load_rules("config/rules.yaml")}

def brk(df, mask, sl=3.0, rr=2.0, hold=22):
    t = bracket_returns(df, mask, rr=rr, max_bars=hold, risk="pct", risk_pct=sl)
    if t is None or t.empty or len(t) < 40: return None
    w, l = t.loc[t["ret"] > 0, "ret"], t.loc[t["ret"] <= 0, "ret"]
    return dict(n=len(t), avg=t["ret"].mean(), win=(t["ret"] > 0).mean() * 100,
                pf=(w.sum() / -l.sum()) if len(l) and l.sum() < 0 else np.nan)

print("\n=== 1. REGIME GATE on the 3%/6% bracket, one month hold ===")
print("Each screen: all sessions vs only when the gate is ON.  avg % per trade, IS | OOS\n")
print(f"{'screen':22s} {'gate':>11s} {'IS n':>6s} {'IS avg':>7s} | {'OOS n':>6s} {'OOS avg':>8s} {'win%':>5s} {'PF':>5s}")
print("-" * 82)
for name in ("ema21_pullback", "donchian_breakout", "narrow_cpr_breakout", "monthly_r1_breakout",
             "high_tight_breakout", "stage2_base_breakout", "rsi2_reversion", "minervini_trend_template"):
    base_mask_is, base_mask_oos = rule_hits(IS, BASE[name]), rule_hits(OOS, BASE[name])
    for gate in ("none", "faber", "both"):
        mi = base_mask_is if gate == "none" else base_mask_is & IS[gate].fillna(False)
        mo = base_mask_oos if gate == "none" else base_mask_oos & OOS[gate].fillna(False)
        a, b = brk(IS, mi), brk(OOS, mo)
        if a and b:
            tag = "  <-" if a["avg"] > 0 and b["avg"] > 0 else ""
            print(f"{name[:22]:22s} {gate:>11s} {a['n']:>6,d} {a['avg']:>+7.2f} | "
                  f"{b['n']:>6,d} {b['avg']:>+8.2f} {b['win']:>5.0f} {b['pf']:>5.2f}{tag}")
    print()

# ---- 2. Connors Double Seven ---------------------------------------------------
print("=== 2. CONNORS DOUBLE SEVEN: buy a 7-day closing low above the 200 DMA, sell on a 7-day closing high ===")
g = feat.groupby("isin")["adj_close"]
feat["lo7"] = g.transform(lambda s: s.shift(1).rolling(7).min())
feat["hi7"] = g.transform(lambda s: s.shift(1).rolling(7).max())
sig = (feat["adj_close"] < feat["lo7"]) & (feat["adj_close"] > feat["sma_200"]) & ~feat["contaminated"] \
      & (feat["turnover_median_20d"] >= 1e7) & (feat["bars_available"] >= 200)
c = feat["adj_close"].to_numpy(); o = feat["adj_open"].to_numpy(); hi7 = feat["hi7"].to_numpy()
codes = pd.factorize(feat["isin"])[0]
last = pd.Series(np.arange(len(feat))).groupby(codes).transform("max").to_numpy()
rows = []
for t in np.flatnonzero(sig.to_numpy()):
    if t + 1 > last[t]: continue
    e = o[t + 1]; k = t + 1; out = None
    while k <= min(last[t], t + 20):
        if c[k] > hi7[k]:                      # 7-day high close: exit next open
            out = (o[k + 1] if k + 1 <= last[t] else c[k]); break
        k += 1
    if out is None: out = c[min(k, last[t])]
    rows.append((feat["date"].iat[t], (out / e - 1) * 100, k - t))
d7 = pd.DataFrame(rows, columns=["date", "ret", "bars"])
for lab, w in (("IS", d7[d7["date"] < SPLIT]), ("OOS", d7[d7["date"] >= SPLIT])):
    ww = w.copy(); ww["mo"] = ww["date"].dt.to_period("M")
    m = ww.groupby("mo")["ret"].mean()
    wins, loss = w.loc[w.ret > 0, "ret"].sum(), -w.loc[w.ret <= 0, "ret"].sum()
    print(f"  {lab:3s}: {len(w):,} trades, avg {w['ret'].mean():+.2f}%, win {(w.ret>0).mean()*100:.0f}%, "
          f"PF {wins/loss:.2f}, avg hold {w['bars'].mean():.1f} bars, months up {(m>0).mean()*100:.0f}%, "
          f"worst month {m.min():+.2f}%")
d7g = d7.merge(regime[["faber"]], left_on="date", right_index=True, how="left")
for lab, w in (("IS, gate on", d7g[(d7g.date < SPLIT) & d7g.faber]), ("OOS, gate on", d7g[(d7g.date >= SPLIT) & d7g.faber])):
    if len(w) > 30:
        wins, loss = w.loc[w.ret > 0, "ret"].sum(), -w.loc[w.ret <= 0, "ret"].sum()
        print(f"  {lab:12s}: {len(w):,} trades, avg {w['ret'].mean():+.2f}%, win {(w.ret>0).mean()*100:.0f}%, PF {wins/loss:.2f}")

# ---- 3. Frog in the pan ------------------------------------------------------
print("\n=== 3. FROG IN THE PAN: momentum portfolio, plain vs smooth-path only ===")
feat["smooth_60"] = feat.groupby("isin")["ret_1d"].transform(lambda s: (s > 0).rolling(60).mean() * 100)
feat["mo"] = feat["date"].dt.to_period("M")
first = feat.groupby("mo")["date"].transform("min")
REB = feat[(feat["date"] == first) & feat["fwd_20d"].notna()].copy()
REB["bench"] = REB.groupby("date")["fwd_20d"].transform("mean")
def port(sub, extra=None, gate=None, n=20, sort="mom_12_1"):
    m = rule_hits(sub, BASE["momentum_leaders"])
    if extra is not None: m &= extra.reindex(sub.index).fillna(False)
    if gate is not None: m &= sub[gate].fillna(False)
    P, B = [], []
    for mo, g in sub[m].groupby("mo"):
        g = g.sort_values(sort, ascending=False).head(n)
        if len(g) < 3: continue
        P.append(g["fwd_20d"].mean()); B.append(g["bench"].iloc[0])
    return np.array(P), np.array(B)
print(f"{'variant':34s} {'IS mo':>6s} {'IS exc':>7s} | {'OOS mo':>7s} {'OOS exc':>8s} {'beat%':>6s}")
for lab, extra, gate in [("momentum, plain", None, None),
                         ("momentum, smooth >= 55% up days", REB["smooth_60"] >= 55, None),
                         ("momentum, smooth >= 60% up days", REB["smooth_60"] >= 60, None),
                         ("momentum, regime gate (Faber)", None, "faber"),
                         ("momentum, smooth 55 + gate", REB["smooth_60"] >= 55, "faber")]:
    a = port(REB[REB["date"] < SPLIT], extra, gate); b = port(REB[REB["date"] >= SPLIT], extra, gate)
    if len(a[0]) < 5 or len(b[0]) < 5: print(f"{lab:34s}  too few"); continue
    ia = (a[0] - a[1]).mean(); ob = b[0] - b[1]
    print(f"{lab:34s} {len(a[0]):>6d} {ia:>+7.2f} | {len(b[0]):>7d} {ob.mean():>+8.2f} {(ob>0).mean()*100:>5.0f}%")
