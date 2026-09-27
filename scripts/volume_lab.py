"""Volume-based breakout/reversal setups tested on 2018-2026 with our winning
trade mechanics: hard close-based 3-ATR stop, 7-day-high reversal exit, 20-bar
cap, market gate on, 10 slots by RS. Setups from classical trading literature.
"""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from multiprocessing import Pool
from screener.portfolio import simulate

t0 = time.time()
PANEL = "/tmp/claude-0/-home-user/397386f6-c392-5ccc-b9f1-52240babee74/scratchpad/bl/bottom_panel.parquet"
COST = 0.30
feat = pd.read_parquet(PANEL).sort_values(["isin", "date"]).reset_index(drop=True)
print(f"panel {len(feat):,} rows, {feat.date.min().date()}->{feat.date.max().date()}, {time.time()-t0:.0f}s")

# ---- volume features ----
g = feat.groupby("isin")
feat["vol_sma20"] = g["volume"].transform(lambda s: s.shift(1).rolling(20).mean())
feat["vol_ratio"] = feat["volume"] / feat["vol_sma20"]
feat["dry_5"] = g["vol_ratio"].transform(lambda s: s.shift(1).rolling(5).max()) # max vol ratio in last 5 sessions
feat["dry_20_med"] = g["vol_ratio"].transform(lambda s: s.shift(1).rolling(20).median())
feat["hi_20_prior"] = g["adj_high"].transform(lambda s: s.shift(1).rolling(20).max())
feat["lo_20_prior"] = g["adj_low"].transform(lambda s: s.shift(1).rolling(20).min())
feat["close_hi_20_prior"] = g["adj_close"].transform(lambda s: s.shift(1).rolling(20).max())
feat["close_lo_10"] = g["adj_close"].transform(lambda s: s.shift(1).rolling(10).max())  # for climax
feat["prev_close"] = g["adj_close"].shift(1)
feat["close_pos"] = (feat.adj_close - feat.adj_low) / (feat.adj_high - feat.adj_low + 1e-9)
# On-Balance Volume
feat["dv"] = np.where(feat.adj_close > feat.prev_close, feat.volume,
                      np.where(feat.adj_close < feat.prev_close, -feat.volume, 0))
feat["obv"] = g["dv"].cumsum()
feat["obv_hi_60"] = g["obv"].transform(lambda s: s.shift(1).rolling(60).max())
feat["close_hi_60"] = g["adj_close"].transform(lambda s: s.shift(1).rolling(60).max())
# Chaikin Money Flow (20)
feat["mfv"] = feat.volume * (2 * feat.close_pos - 1)  # money flow volume
feat["cmf_20"] = g["mfv"].transform(lambda s: s.rolling(20).sum()) / g["volume"].transform(lambda s: s.rolling(20).sum())
feat["cmf_prev"] = g["cmf_20"].shift(1)
# Pocket-pivot-style: today's UP-day volume > max of the last 10 sessions' DOWN-day volume
down_vol = feat.volume.where(feat.adj_close < feat.prev_close, 0)
feat["down_vol_10"] = down_vol.groupby(feat["isin"]).transform(lambda s: s.shift(1).rolling(10).max())
# recent 3-day dry-up
feat["v_3_max"] = g["vol_ratio"].transform(lambda s: s.shift(1).rolling(3).max())
# 3-day drop
feat["drop_5d_pct"] = (feat.adj_close / g["adj_close"].shift(5) - 1) * 100

C = feat.adj_close.to_numpy(float); O = feat.adj_open.to_numpy(float); L = feat.adj_low.to_numpy(float); H = feat.adj_high.to_numpy(float)
ATRP = feat.atr_pct.to_numpy(float) / 100
HI7 = g["adj_close"].transform(lambda s: s.shift(1).rolling(7).max()).to_numpy(float)
GATE = feat.mkt_gate_on.to_numpy(float)
DATES = feat["date"].to_numpy(); ISIN = feat["isin"].to_numpy()
LAST = pd.Series(np.arange(len(feat))).groupby(pd.factorize(feat["isin"])[0]).transform("max").to_numpy()
CAL = pd.DatetimeIndex(sorted(feat.date.unique())); CALPOS = {d: i for i, d in enumerate(CAL)}
RS = feat.rs_rank.to_numpy(float)

base = ((feat.turnover_median_20d >= 1e7) & (feat.bars_available >= 250) & (feat.adj_close >= 10) & (~feat.contaminated) & (GATE == 1)).to_numpy()

# ---- filter to Nifty Midcap 150 constituents (focused universe) ----
from screener.store import Store
mem = Store("data").read_membership()
mid = set(mem.loc[(mem["index_name"] == "NIFTY MIDCAP 150") & (mem["to_date"].isna()), "isin"])
print(f"Nifty Midcap 150 constituents in universe: {len(mid)}")
base = base & feat["isin"].isin(mid).to_numpy()
print(f"base signals available: {int(base.sum()):,}")


# ---- setup masks ----
S = {}
# 1. VCP Dry-up Breakout: base with vol median < 0.9x for 20 days, today new 20-day close high + 2x vol
S["VCP dry-up breakout"] = base & (feat.dry_20_med < 0.9).to_numpy() & (C > feat.close_hi_20_prior.to_numpy()) & (feat.vol_ratio >= 2.0).to_numpy() & (C > feat.sma_200.to_numpy())
# 2. Simple Volume Spike Breakout: new 20-day close high + 3x vol
S["Volume spike breakout (3x)"] = base & (C > feat.close_hi_20_prior.to_numpy()) & (feat.vol_ratio >= 3.0).to_numpy() & (C > feat.sma_200.to_numpy())
# 3. Pocket Pivot (self-computed): today's UP volume > max down-vol in last 10, close > SMA50, not extended > 15% above 21EMA
up_day = (C > feat.prev_close.to_numpy())
S["Pocket pivot"] = base & up_day & (feat.volume > feat.down_vol_10).to_numpy() & (C > feat.sma_50.to_numpy()) & (feat.dist_ema_21 <= 15).to_numpy() & (C > feat.sma_200.to_numpy())
# 4. OBV new high (Granville): OBV crosses to a new 60-day high, price still below its 60-day high (accumulation before markup)
S["OBV new high before price"] = base & (feat.obv > feat.obv_hi_60).to_numpy() & (C < feat.close_hi_60.to_numpy()) & (C > feat.sma_200.to_numpy())
# 5. Wyckoff Spring: today low < prior 20-day low, close back above it, vol >= 2x
S["Wyckoff spring"] = base & (L < feat.lo_20_prior.to_numpy()) & (C > feat.lo_20_prior.to_numpy()) & (feat.vol_ratio >= 2.0).to_numpy() & (C > feat.sma_200.to_numpy())
# 6. Selling climax reversal: 5-day drop <= -8%, today vol >= 3x, close in top 30% of range, close > prev close
S["Selling climax reversal"] = base & (feat.drop_5d_pct <= -8).to_numpy() & (feat.vol_ratio >= 3.0).to_numpy() & (feat.close_pos >= 0.7).to_numpy() & up_day
# 7. Volume dry-up at support: within 2% above 50-DMA, last 3-day max vol_ratio < 0.7, today up on vol_ratio >= 1.5
near_50 = (C >= feat.sma_50.to_numpy() * 0.98) & (C <= feat.sma_50.to_numpy() * 1.02)
S["Dry-up at 50-DMA support"] = base & near_50 & (feat.v_3_max < 0.7).to_numpy() & (feat.vol_ratio >= 1.5).to_numpy() & up_day & (C > feat.sma_200.to_numpy())
# 8. CMF zero cross bullish
S["CMF zero cross (bullish)"] = base & (feat.cmf_20 > 0).to_numpy() & (feat.cmf_prev <= 0).to_numpy() & (C > feat.sma_200.to_numpy())

E21 = feat["ema_21"].to_numpy(float)
def run(sig, family="reversal"):
    """family: 'reversal' = 7-day-high exit, 20-bar cap (mean-reversion);
              'trend' = close<21EMA trailing exit, 60-bar cap (trend-continuation)."""
    cap = 20 if family == "reversal" else 60
    rows = []; busy = {}
    for t in np.flatnonzero(sig & (DATES >= np.datetime64("2018-06-01"))):
        if t + 1 > LAST[t] or busy.get(ISIN[t], -1) >= t or not np.isfinite(ATRP[t]): continue
        e = O[t + 1]
        if not np.isfinite(e) or e <= 0: continue
        atr = ATRP[t] * e; sl = e - 3 * atr
        end = min(LAST[t], t + cap); k = t + 1; out = None; why = "time"
        while k <= end:
            if k > t + 1 and C[k] < sl and k + 1 <= LAST[t]: out = O[k+1]; k += 1; why = "stop"; break
            if family == "reversal" and np.isfinite(HI7[k]) and C[k] > HI7[k]:
                if k + 1 <= LAST[t]: out = O[k+1]; k += 1
                else: out = C[k]
                why = "rev7"; break
            if family == "trend" and np.isfinite(E21[k]) and C[k] < E21[k] and k + 1 <= LAST[t]:
                out = O[k+1]; k += 1; why = "trail"; break
            k += 1
        if out is None: out = C[k] if k <= LAST[t] else e
        if not np.isfinite(out): continue
        busy[ISIN[t]] = k
        rows.append((pd.Timestamp(DATES[t]), ISIN[t], CALPOS[pd.Timestamp(DATES[t+1])], CALPOS[pd.Timestamp(DATES[k])],
                     (out/e - 1) * 100 - COST, why, RS[t]))
    return pd.DataFrame(rows, columns=["date","isin","entry_i","exit_i","net","why","rs"])

print(f"\n{'Setup':32s}{'n':>6s}{'Win':>6s}{'Mean':>7s}{'Stop':>6s}{'CAGR':>7s}{'MaxDD':>7s}{'Score':>7s}{'H1 win':>8s}{'H2 win':>8s}")
print("-"*112)
results = []
FAMILY = {
    "VCP dry-up breakout": "trend",
    "Volume spike breakout (3x)": "trend",
    "Pocket pivot": "trend",
    "OBV new high before price": "trend",
    "CMF zero cross (bullish)": "trend",
    "Wyckoff spring": "reversal",
    "Selling climax reversal": "reversal",
    "Dry-up at 50-DMA support": "reversal",
}
for name, sig in S.items():
    t = run(sig, family=FAMILY[name])
    if len(t) < 100: 
        print(f"{name:32s} too few trades ({len(t)})"); continue
    p = simulate(t.sort_values(["entry_i","rs"], ascending=[True,False]).reset_index(drop=True), slots=10, sessions=CAL)
    h1 = t[t.date < "2022-06-30"]; h2 = t[t.date >= "2022-06-30"]
    w = (t.net > 0).mean()*100; wh1 = (h1.net > 0).mean()*100 if len(h1) else 0; wh2 = (h2.net > 0).mean()*100 if len(h2) else 0
    print(f"{name:32s}{len(t):>6d}{w:>5.1f}%{t.net.mean():>+7.2f}{(t.why=='stop').mean()*100:>5.0f}%{p['cagr']:>+6.1f}%{p['maxdd']:>+7.1f}{p['cagr']+p['maxdd']:>+7.1f}{wh1:>7.1f}%{wh2:>7.1f}%")
    results.append({"setup": name, "n": int(len(t)), "win": round(w,1), "mean": round(t.net.mean(),2), "cagr": round(p['cagr'],1), "maxdd": round(p['maxdd'],1), "score": round(p['cagr']+p['maxdd'],1), "win_h1": round(wh1,1), "win_h2": round(wh2,1)})
json.dump(results, open("data/screen/volume_lab.json","w"))
print(f"\nplaybook reference: 74% win, +32% CAGR, -9.5% DD, score +22.9  (SW Leader Dip Playbook 8-yr)")
print(f"done {time.time()-t0:.0f}s")
