"""Shared inputs for the regime study: a market-state row per day, and trade
lists tagged with everything known at the signal close.

Every feature here is AS OF the close of the date on the row. A trade's
signal is that close and its entry is the next open, so a feature can be used
as a filter without looking ahead. Anything that needs a future bar is
deliberately absent.

Outputs
  data/screen/market_state.parquet   one row per session, ~60 regime features
  data/screen/trades_tagged.parquet  one row per trade, for several strategies,
                                     with entry/exit, MAE/MFE, stock and market
                                     state at the signal
"""
import sys, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.rules import load_rules
from screener.backtest import add_panel_columns, rule_hits
from screener import quality
from screener.store import Store

t0 = time.time()
COST = 0.30
feat = pd.read_parquet("data/screen/features.parquet")
lat = pd.read_parquet("data/screen/latest_features.parquet")
feat = feat.merge(lat[["isin", "cap_band", "symbol"]].rename(columns={"symbol": "sym"}), on="isin", how="left")
st = Store("data")
b = feat[["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
feat["contaminated"] = quality.contamination_mask(b, quality.detect_price_jumps(b, st.read_adjustments())).to_numpy()
feat = add_panel_columns(feat).sort_values(["isin", "date"]).reset_index(drop=True)
g = feat.groupby("isin")
feat["ret1"] = g["adj_close"].pct_change()
feat["liquid"] = (feat["turnover_median_20d"] >= 1e7) & (feat["bars_available"] >= 250) & (~feat["contaminated"])
CAL = pd.DatetimeIndex(sorted(feat["date"].unique()))

# ---- market state from the published indices ------------------------------
ix = pd.read_parquet("data/raw/indices_daily.parquet")
px = ix.pivot(index="date", columns="index_name", values="close").reindex(CAL).ffill()
hi = ix.pivot(index="date", columns="index_name", values="high").reindex(CAL).ffill()
lo = ix.pivot(index="date", columns="index_name", values="low").reindex(CAL).ffill()
ms = pd.DataFrame(index=CAL)
def trend_block(prefix, s):
    for n in (20, 50, 200):
        ma = s.rolling(n).mean()
        ms[f"{prefix}_vs_{n}dma"] = (s / ma - 1) * 100
        ms[f"{prefix}_above_{n}dma"] = (s > ma).astype(int)
    ms[f"{prefix}_20dma_slope"] = (s.rolling(20).mean() / s.rolling(20).mean().shift(10) - 1) * 100
    ms[f"{prefix}_ret_5d"] = (s / s.shift(5) - 1) * 100
    ms[f"{prefix}_ret_20d"] = (s / s.shift(20) - 1) * 100
    ms[f"{prefix}_ret_60d"] = (s / s.shift(60) - 1) * 100
    ms[f"{prefix}_dd_252"] = (s / s.rolling(252).max() - 1) * 100
    ms[f"{prefix}_rvol_20"] = s.pct_change().rolling(20).std() * np.sqrt(252) * 100
trend_block("nifty", px["Nifty 50"])
trend_block("midcap", px["NIFTY MIDCAP 150"])
trend_block("bank", px["Nifty Bank"])
trend_block("next50", px["Nifty Next 50"])
ms["mid_vs_nifty_20d"] = ms["midcap_ret_20d"] - ms["nifty_ret_20d"]          # risk appetite
ms["next50_vs_nifty_20d"] = ms["next50_ret_20d"] - ms["nifty_ret_20d"]
v = px["India VIX"]
ms["vix"] = v
ms["vix_chg_5d"] = (v / v.shift(5) - 1) * 100
ms["vix_vs_20dma"] = (v / v.rolling(20).mean() - 1) * 100
ms["vix_pctile_1y"] = v.rolling(252).rank(pct=True) * 100
ms["vix_above_15"] = (v > 15).astype(int)
ms["vix_above_20"] = (v > 20).astype(int)
ms["vix_spike"] = (v > v.rolling(10).mean() * 1.2).astype(int)
ms["nifty_range_20d"] = ((hi["Nifty 50"].rolling(20).max() / lo["Nifty 50"].rolling(20).min()) - 1) * 100

# ---- breadth from our own panel --------------------------------------------
liq = feat[feat["liquid"]]
byd = liq.groupby("date")
ms["breadth_above_50"] = byd.apply(lambda x: (x["adj_close"] > x["sma_50"]).mean() * 100).reindex(CAL)
ms["breadth_above_200"] = byd.apply(lambda x: (x["adj_close"] > x["sma_200"]).mean() * 100).reindex(CAL)
ms["breadth_above_20"] = byd.apply(lambda x: (x["adj_close"] > x["sma_20"]).mean() * 100).reindex(CAL)
adv = byd["ret1"].apply(lambda r: (r > 0).mean()).reindex(CAL)
ms["adv_ratio"] = adv * 100
ms["breadth_thrust_10"] = adv.ewm(span=10).mean() * 100                    # Zweig-style
ms["pct_pos_20d"] = byd["ret_20d"].apply(lambda r: (r > 0).mean() * 100).reindex(CAL)
ms["new_hi_minus_lo"] = byd.apply(lambda x: ((x["pct_from_52w_high"] >= -0.5).mean()
                                             - (x["pct_from_52w_low"] <= 0.5).mean()) * 100).reindex(CAL)
ms["dispersion"] = byd["ret1"].std().reindex(CAL) * 100                   # cross-sectional
ms["ew_ret_1d"] = byd["ret1"].mean().reindex(CAL) * 100
ms["ew_ret_20d"] = ((1 + byd["ret1"].mean()).cumprod().reindex(CAL)).pct_change(20) * 100
ms["breadth_50_chg_10d"] = ms["breadth_above_50"] - ms["breadth_above_50"].shift(10)

# ---- calendar --------------------------------------------------------------
d = pd.Series(CAL, index=CAL)
ms["dow"] = d.dt.dayofweek
ms["month"] = d.dt.month
pos_in_month = d.groupby(d.dt.to_period("M")).cumcount()
n_in_month = d.groupby(d.dt.to_period("M")).transform("count")
ms["turn_of_month"] = ((pos_in_month <= 2) | (pos_in_month >= n_in_month - 2)).astype(int)
last_thu = d.groupby(d.dt.to_period("M")).apply(lambda s: s[s.dt.dayofweek == 3].max())
ms["expiry_week"] = d.apply(lambda x: int(any(0 <= (lt - x).days <= 4 for lt in last_thu.values if pd.notna(lt)))).values
ms["earnings_season"] = d.dt.day.between(12, 31).values & d.dt.month.isin([1, 4, 7, 10]).values | \
                        (d.dt.day <= 15).values & d.dt.month.isin([2, 5, 8, 11]).values
ms["earnings_season"] = ms["earnings_season"].astype(int)
ms.index.name = "date"
ms.reset_index().to_parquet("data/screen/market_state.parquet", index=False)
print(f"market_state: {len(ms)} sessions x {ms.shape[1]} features, {time.time()-t0:.0f}s", flush=True)

# ---- tagged trade lists ------------------------------------------------------
O = feat["adj_open"].to_numpy(float); H = feat["adj_high"].to_numpy(float)
L = feat["adj_low"].to_numpy(float);  C = feat["adj_close"].to_numpy(float)
HI7 = feat["hi7_prior"].to_numpy(float)
ISIN = feat["isin"].to_numpy(); DATES = feat["date"].to_numpy()
LAST = pd.Series(np.arange(len(feat))).groupby(pd.factorize(feat["isin"])[0]).transform("max").to_numpy()
CALPOS = {dt: i for i, dt in enumerate(CAL)}
STOCK_COLS = ["rs_rank", "atr_pct", "dist_ema_21", "dist_sma_50", "dist_sma_10", "sma_150_slope", "sma_200_slope",
              "smooth_60", "pct_from_52w_high", "pct_from_52w_low", "rsi_2", "rsi_14", "vol_vs_ema21", "close_pos",
              "gap_pct", "ret_5d", "ret_20d", "ret_60d", "bb_pct_b", "m_cpr_width_rank", "base_depth",
              "turnover_median_20d", "cap_band", "sym"]

def trades(sig, exit_spec, label):
    kind = exit_spec[0]; rows = []; busy = {}
    for t in np.flatnonzero(sig):
        if t + 1 > LAST[t] or busy.get(ISIN[t], -1) >= t:
            continue
        e = O[t + 1]
        if not np.isfinite(e) or e <= 0:
            continue
        k = t + 1; out = None; why = "time"
        if kind == "hold":
            k = min(t + exit_spec[1], LAST[t]); out = C[k]
        else:
            cap = min(LAST[t], t + 20)
            while k <= cap:
                if np.isfinite(HI7[k]) and C[k] > HI7[k]:
                    if k + 1 <= LAST[t]:
                        out = O[k + 1]; k += 1
                    else:
                        out = C[k]
                    why = "reversal"; break
                k += 1
            if out is None:
                k = min(t + 20, LAST[t]); out = C[k]
        if not np.isfinite(out):
            continue
        busy[ISIN[t]] = k
        seg_l = L[t + 1:k + 1]; seg_h = H[t + 1:k + 1]
        rows.append(dict(strategy=label, date=pd.Timestamp(DATES[t]), isin=ISIN[t], row=t,
                         entry_i=CALPOS[pd.Timestamp(DATES[t + 1])], exit_i=CALPOS[pd.Timestamp(DATES[k])],
                         entry=e, exit=out, net=(out / e - 1) * 100 - COST, bars=k - t, why=why,
                         mae=(seg_l.min() / e - 1) * 100 if len(seg_l) else 0.0,
                         mfe=(seg_h.max() / e - 1) * 100 if len(seg_h) else 0.0,
                         bars_to_mae=int(np.argmin(seg_l)) + 1 if len(seg_l) else 0,
                         bars_to_mfe=int(np.argmax(seg_h)) + 1 if len(seg_h) else 0))
    return rows

rules = {r.name: r for r in load_rules("config/rules.yaml")}
IN = (feat["date"] >= "2023-09-19").to_numpy()
SPECS = [("leader_dip", ("rev",), "Leader Dip | 7-day high"),
         ("leader_dip", ("hold", 20), "Leader Dip | hold 20"),
         ("minervini_trend_template", ("hold", 20), "Stage 2 Uptrend | hold 20"),
         ("momentum_leaders", ("rev",), "Momentum Leaders | 7-day high"),
         ("rsi2_reversion", ("rev",), "RSI(2) Snapback | 7-day high")]
allrows = []
for name, spec, label in SPECS:
    hits = rule_hits(feat, rules[name]).to_numpy() & IN
    allrows += trades(hits, spec, label)
    print(f"  {label}: {sum(1 for r in allrows if r['strategy']==label):,} trades", flush=True)
T = pd.DataFrame(allrows)
T = T.merge(feat[["isin", "date"] + STOCK_COLS], on=["isin", "date"], how="left")
T = T.merge(ms.reset_index(), on="date", how="left")
T.to_parquet("data/screen/trades_tagged.parquet", index=False)
print(f"\ntrades_tagged: {len(T):,} trades x {T.shape[1]} columns, {time.time()-t0:.0f}s")
print("strategies:", T["strategy"].value_counts().to_dict())
