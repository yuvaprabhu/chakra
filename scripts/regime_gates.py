"""Regime gates: when to be in the market for the swing screens.

For every regime signal (the market_state columns plus indicators computed
here from the raw indices and the stock panel) a binary RISK-ON gate is
defined. Thresholds that the literature does not fix are taken from the
FIRST half of the trade sample only (signal dates before SPLIT) and then
applied unchanged to the second half.

For each strategy and each half we compare trades taken with the gate ON
against the trades the gate would have REMOVED (gate OFF): count, mean net,
win rate, plus a date-clustered t-statistic for the difference (trades on
the same signal day are not independent).

A gate is PROMOTED when, in BOTH halves and for EVERY main strategy, the
removed trades have a lower mean net than the kept trades (equivalently the
kept mean beats the unfiltered mean), with at least MIN_OFF removed trades
in each cell. ROBUST additionally requires the pooled effect to be large
(>= 0.5 pct points), date-clustered |t| >= 2 in both halves, and the gate to
be on at least 40% of sessions.

Outputs
  data/screen/regime_gates.json
Reads (never writes)
  data/screen/market_state.parquet, data/screen/trades_tagged.parquet,
  data/raw/indices_daily.parquet, data/screen/features.parquet
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pandas as pd

t0 = time.time()
SPLIT = pd.Timestamp("2025-03-19")
MAIN = ["Leader Dip | 7-day high", "Momentum Leaders | 7-day high", "Stage 2 Uptrend | hold 20"]
MIN_OFF = 30          # removed trades needed per cell for the cell to count
MIN_TIM = 40.0        # % of sessions on, below which a gate is "not usable"
OUT = Path("data/screen/regime_gates.json")

ms = pd.read_parquet("data/screen/market_state.parquet").set_index("date").sort_index()
T = pd.read_parquet("data/screen/trades_tagged.parquet")
T["date"] = pd.to_datetime(T["date"])
T = T[T["strategy"].isin(MAIN + ["Leader Dip | hold 20", "RSI(2) Snapback | 7-day high"])].copy()
ALL_STRATS = sorted(T["strategy"].unique())
CAL = ms.index
H1_SESS = CAL[(CAL >= T["date"].min()) & (CAL < SPLIT)]     # sessions used to pick thresholds
H2_SESS = CAL[(CAL >= SPLIT) & (CAL <= T["date"].max())]

# ---------------------------------------------------------------------------
# 1. extra indicators from the raw indices (long history, so no warm-up gap)
# ---------------------------------------------------------------------------
ix = pd.read_parquet("data/raw/indices_daily.parquet")
ix["date"] = pd.to_datetime(ix["date"])
px = ix.pivot(index="date", columns="index_name", values="close").sort_index().ffill()
nifty, vix, mid = px["Nifty 50"], px["India VIX"], px["NIFTY MIDCAP 150"]
X = pd.DataFrame(index=px.index)
# Faber 10-month SMA, evaluated daily: mean of the last 9 completed month-end
# closes and today's close (the month-to-date convention practitioners use)
mend = nifty.groupby(nifty.index.to_period("M")).last()
mend_prev9 = mend.rolling(9).sum().shift(1)                        # 9 completed months, excluding the current
cur_per = nifty.index.to_period("M")
X["faber_sma10m"] = (mend_prev9.reindex(cur_per).to_numpy() + nifty.to_numpy()) / 10
X["nifty_gt_10m_sma"] = (nifty > X["faber_sma10m"]).astype(float)
X["nifty_ret_252"] = (nifty / nifty.shift(252) - 1) * 100         # 12-month absolute momentum
X["nifty_ret_126"] = (nifty / nifty.shift(126) - 1) * 100
X["nifty_ret_504"] = (nifty / nifty.shift(504) - 1) * 100         # Daniel-Moskowitz bear indicator window
mend_m = mid.groupby(mid.index.to_period("M")).last()
X["mid_gt_10m_sma"] = (mid > (mend_m.rolling(9).sum().shift(1).reindex(cur_per).to_numpy() + mid.to_numpy()) / 10).astype(float)
X["vix_vs_10dma"] = (vix / vix.rolling(10).mean() - 1) * 100
X["vix_vs_50dma"] = (vix / vix.rolling(50).mean() - 1) * 100
X["vix_roc_10d"] = (vix / vix.shift(10) - 1) * 100
X["vix_z_60"] = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()
X["nifty_rvol_10"] = nifty.pct_change().rolling(10).std() * np.sqrt(252) * 100
X["nifty_rvol_60"] = nifty.pct_change().rolling(60).std() * np.sqrt(252) * 100
X["nifty_sma50_gt_sma200"] = (nifty.rolling(50).mean() > nifty.rolling(200).mean()).astype(float)
X["nifty_ret_10d"] = (nifty / nifty.shift(10) - 1) * 100
X = X.reindex(CAL).ffill()
ms = ms.join(X, how="left")
ms["vrp"] = ms["vix"] - ms["nifty_rvol_20"]                         # implied minus realised (variance risk premium proxy)

# ---------------------------------------------------------------------------
# 2. breadth rebuilt from the stock panel with a turnover-only filter, so it
#    covers the whole trade window (market_state breadth starts 2024-06-11
#    because of its 250-bar history requirement)
# ---------------------------------------------------------------------------
cols = ["isin", "date", "adj_close", "sma_50", "sma_200", "ret_1d", "ret_20d", "pct_from_52w_high",
        "pct_from_52w_low", "turnover_median_20d"]
F = pd.read_parquet("data/screen/features.parquet", columns=cols)
F["date"] = pd.to_datetime(F["date"])
F = F[F["turnover_median_20d"] >= 1e7]
g = F.groupby("date")
B = pd.DataFrame(index=CAL)
r = F["ret_1d"]
B["adv"] = g["ret_1d"].apply(lambda s: (s > 0).sum()).reindex(CAL)
B["dec"] = g["ret_1d"].apply(lambda s: (s < 0).sum()).reindex(CAL)
B["n"] = g.size().reindex(CAL)
B["b50"] = g.apply(lambda x: (x["adj_close"] > x["sma_50"]).mean() * 100).reindex(CAL)
B["b200"] = g.apply(lambda x: (x["adj_close"] > x["sma_200"])[x["sma_200"].notna()].mean() * 100).reindex(CAL)
B["pos20"] = g["ret_20d"].apply(lambda s: (s > 0).mean() * 100).reindex(CAL)
B["nh"] = g["pct_from_52w_high"].apply(lambda s: (s >= -0.5).sum()).reindex(CAL)
B["nl"] = g["pct_from_52w_low"].apply(lambda s: (s <= 0.5).sum()).reindex(CAL)
B["disp"] = g["ret_1d"].std().reindex(CAL)
B["ew_ret"] = g["ret_1d"].mean().reindex(CAL) / 100
adr = B["adv"] / (B["adv"] + B["dec"])
B["zweig_ema"] = adr.ewm(span=10, adjust=False).mean()                 # Zweig: 10-day EMA of adv/(adv+dec)
rana = (B["adv"] - B["dec"]) / (B["adv"] + B["dec"]) * 1000             # ratio-adjusted net advances
B["mcc_osc"] = rana.ewm(span=19, adjust=False).mean() - rana.ewm(span=39, adjust=False).mean()
B["mcc_sum"] = B["mcc_osc"].cumsum()
B["ad_line"] = ((B["adv"] - B["dec"]) / (B["adv"] + B["dec"])).cumsum()
B["ad_line_sma50"] = B["ad_line"].rolling(50).mean()
B["nhnl_10"] = ((B["nh"] - B["nl"]) / B["n"] * 100).rolling(10).mean()
B["disp_20"] = B["disp"].rolling(20).mean()
ew = (1 + B["ew_ret"].fillna(0)).cumprod()
B["ew_valid_n"] = B["ew_ret"].notna().cumsum()          # 2025-04-26 (Saturday session) has no panel rows
B["ew_gt_sma50"] = (ew > ew.rolling(50).mean()).astype(float)
B["ew_gt_sma200"] = (ew > ew.rolling(200).mean()).astype(float)
B["b50_sma20"] = B["b50"].rolling(20, min_periods=15).mean()
B["nhnl_20"] = ((B["nh"] - B["nl"]) / B["n"] * 100).rolling(20, min_periods=15).mean()
# Schmitt-trigger version of the 50-DMA breadth: OFF once breadth drops below 40, back ON only above 50
b = B["b50"].to_numpy(); hyst = np.full(len(b), np.nan); state = np.nan
for i in range(len(b)):
    if np.isnan(b[i]):
        continue
    if np.isnan(state):
        state = 1.0 if b[i] > 50 else 0.0
    elif state == 1.0 and b[i] < 40:
        state = 0.0
    elif state == 0.0 and b[i] > 50:
        state = 1.0
    hyst[i] = state
B["b50_hyst"] = hyst
# "sustained" version: OFF only when breadth has been below 50 for 10 consecutive sessions
below = (B["b50"] < 50).astype(float).where(B["b50"].notna())
B["b50_sustained_on"] = (below.rolling(10).sum() < 10).astype(float).where(below.rolling(10).sum().notna())
B["ew_ret_20"] = (ew / ew.shift(20) - 1) * 100
B["b50_chg10"] = B["b50"] - B["b50"].shift(10)
B["b50_up"] = (B["b50_chg10"] > 0).astype(float)
# Zweig thrust: EMA goes from below 0.40 to above 0.615 within 10 sessions; stay on for 126 sessions after
ze = B["zweig_ema"].to_numpy(); thrust = np.zeros(len(ze))
for i in range(10, len(ze)):
    if ze[i] > 0.615 and np.nanmin(ze[i - 10:i]) < 0.40:
        thrust[i:i + 126] = 1
B["zweig_thrust_on"] = thrust
B["zweig_dates"] = 0
B.loc[[CAL[i] for i in range(10, len(ze)) if ze[i] > 0.615 and np.nanmin(ze[i - 10:i]) < 0.40], "zweig_dates"] = 1
ms = ms.join(B.add_prefix("p_"), how="left")
print(f"indicators built: {ms.shape[1]} columns, {time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------------------
# 3. gate definitions.  Each: (column, direction, threshold spec, source, note)
#    threshold spec: a number (fixed by the literature / construction) or
#    ("q", p) = the p-th percentile of the signal over H1 sessions.
# ---------------------------------------------------------------------------
GATES = {
    # --- index trend ---
    "faber_10m_sma":        ("nifty_gt_10m_sma", ">", 0.5, "Faber 2007", "Nifty above its 10-month SMA"),
    "nifty_above_200dma":   ("nifty_above_200dma", ">", 0.5, "Faber/Zakamulin; Minervini", "Nifty above 200-DMA"),
    "nifty_above_50dma":    ("nifty_above_50dma", ">", 0.5, "practitioner", "Nifty above 50-DMA"),
    "nifty_above_20dma":    ("nifty_above_20dma", ">", 0.5, "practitioner", "Nifty above 20-DMA"),
    "nifty_20dma_rising":   ("nifty_20dma_slope", ">", 0.0, "practitioner", "Nifty 20-DMA higher than 10 sessions ago"),
    "golden_cross":         ("nifty_sma50_gt_sma200", ">", 0.5, "Zenodo Nifty 2010-22 study (weak)", "Nifty 50-DMA above 200-DMA"),
    "abs_mom_12m":          ("nifty_ret_252", ">", 0.0, "Moskowitz-Ooi-Pedersen 2012 TSMOM", "Nifty 12-month return positive"),
    "abs_mom_6m":           ("nifty_ret_126", ">", 0.0, "TSMOM variant", "Nifty 6-month return positive"),
    "not_deep_drawdown":    ("nifty_dd_252", ">", -10.0, "Daniel-Moskowitz bear indicator proxy", "Nifty within 10% of its 252-day high"),
    "midcap_10m_sma":       ("mid_gt_10m_sma", ">", 0.5, "Faber applied to Midcap 150", "Midcap 150 above 10-month SMA"),
    "midcap_above_50dma":   ("midcap_above_50dma", ">", 0.5, "practitioner", "Midcap 150 above 50-DMA"),
    "risk_appetite_mid":    ("mid_vs_nifty_20d", ">", 0.0, "practitioner (mid/large ratio)", "Midcap beat Nifty over 20 sessions"),
    # --- volatility ---
    "vix_below_20dma":      ("vix_vs_20dma", "<", 0.0, "practitioner VIX-vs-trend", "VIX below its 20-day average"),
    "vix_below_10dma":      ("vix_vs_10dma", "<", 0.0, "Connors VIX reversal (inverse)", "VIX below its 10-day average"),
    "vix_below_50dma":      ("vix_vs_50dma", "<", 0.0, "practitioner", "VIX below its 50-day average"),
    "vix_below_15":         ("vix", "<", 15.0, "India VIX practitioner level", "India VIX below 15"),
    "vix_pctile_below_50":  ("vix_pctile_1y", "<", 50.0, "options.cafe VIX-rank filter", "VIX below its 1-year median"),
    "vix_falling_10d":      ("vix_roc_10d", "<", 0.0, "Samco VIX ROC (inverse)", "VIX lower than 10 sessions ago"),
    "vix_falling_5d":       ("vix_chg_5d", "<", 0.0, "practitioner", "VIX lower than 5 sessions ago"),
    "vix_z60_low":          ("vix_z_60", "<", ("q", 50), "z-score variant", "VIX z-score vs 60d below H1 median"),
    "no_vix_spike":         ("vix_spike", "<", 0.5, "Connors (spike = reversal buy) inverse", "VIX not >20% above its 10-day avg"),
    "vrp_high":             ("vrp", ">", ("q", 50), "Bollerslev-Tauchen-Zhou 2009", "VIX minus 20d realised vol above H1 median"),
    "rvol20_not_high":      ("nifty_rvol_20", "<", ("q", 75), "Barroso-SantaClara / Moreira-Muir vol targeting", "Nifty 20d realised vol below H1 75th pct"),
    "rvol60_not_high":      ("nifty_rvol_60", "<", ("q", 75), "vol targeting, slower", "Nifty 60d realised vol below H1 75th pct"),
    "range20_low":          ("nifty_range_20d", "<", ("q", 50), "practitioner", "Nifty 20d high-low range below H1 median"),
    # --- breadth (rebuilt here) ---
    "breadth50_gt_50":      ("p_b50", ">", 50.0, "stockcharts / Schwab breadth", "> 50% of liquid stocks above 50-DMA"),
    "breadth200_gt_50":     ("p_b200", ">", 50.0, "LPL / stockcharts breadth", "> 50% of liquid stocks above 200-DMA"),
    "breadth50_rising":     ("p_b50_up", ">", 0.5, "practitioner", "% above 50-DMA higher than 10 sessions ago"),
    "breadth50_sma20_gt_50":("p_b50_sma20", ">", 50.0, "slow variant of breadth50_gt_50", "20-day avg of % above 50-DMA > 50"),
    "breadth50_hysteresis": ("p_b50_hyst", ">", 0.5, "Schmitt-trigger variant", "off below 40%, back on only above 50%"),
    "breadth50_sustained":  ("p_b50_sustained_on", ">", 0.5, "slow variant", "off only after 10 straight sessions below 50%"),
    "nhnl20_positive":      ("p_nhnl_20", ">", 0.0, "slow variant of nhnl_positive", "20d avg of (new highs - new lows) > 0"),
    "zweig_ema_gt_50":      ("p_zweig_ema", ">", 0.5, "Zweig (level use)", "10d EMA of adv/(adv+dec) above 0.50"),
    "zweig_thrust_window":  ("p_zweig_thrust_on", ">", 0.5, "Zweig breadth thrust", "within 126 sessions of a 40%->61.5% thrust"),
    "mcclellan_osc_pos":    ("p_mcc_osc", ">", 0.0, "McClellan (CXO: weak)", "ratio-adjusted McClellan oscillator > 0"),
    "mcclellan_sum_rising": ("p_mcc_sum_rising", ">", 0.5, "McClellan summation", "summation index above its value 5 sessions ago"),
    "ad_line_gt_sma50":     ("p_ad_gt_sma", ">", 0.5, "A/D line trend", "A/D line above its 50-day SMA"),
    "nhnl_positive":        ("p_nhnl_10", ">", 0.0, "AAII new highs-lows", "10d avg of (new highs - new lows) > 0"),
    "pct_pos_20d_gt_50":    ("p_pos20", ">", 50.0, "breadth of 20d returns", "> 50% of stocks up over 20 sessions"),
    "ew_above_sma50":       ("p_ew_gt_sma50", ">", 0.5, "regime_and_literature.py", "equal-weight universe above 50-SMA"),
    "ew_above_sma200":      ("p_ew_gt_sma200", ">", 0.5, "regime_and_literature.py / Faber", "equal-weight universe above 200-SMA"),
    "dispersion_low":       ("p_disp_20", "<", ("q", 50), "Stivers-Sun 2010", "20d avg cross-sectional dispersion below H1 median"),
    # --- momentum-crash state ---
    "no_bear_rebound":      ("bear_rebound", "<", 0.5, "Daniel-Moskowitz 2016", "NOT (Nifty >10% off high AND up >5% in 20d)"),
    # --- calendar ---
    "not_expiry_week":      ("expiry_week", "<", 0.5, "Indian expiry-week studies (mixed)", "signal not in F&O expiry week"),
    "turn_of_month_only":   ("turn_of_month", ">", 0.5, "Nageswari et al (mixed for India)", "signal in last 3 / first 3 sessions"),
    "not_earnings_season":  ("earnings_season", "<", 0.5, "PEAD literature (stock-level)", "signal outside results season"),
}
ms["p_mcc_sum_rising"] = (ms["p_mcc_sum"] > ms["p_mcc_sum"].shift(5)).astype(float)
ms["p_ad_gt_sma"] = (ms["p_ad_line"] > ms["p_ad_line_sma50"]).astype(float)
ms["bear_rebound"] = ((ms["nifty_dd_252"] <= -10) & (ms["nifty_ret_20d"] >= 5)).astype(float)
# derived 0/1 columns must be undefined (NaN), not 0, where their inputs are undefined
for c, dep in (("p_mcc_sum_rising", ms["p_mcc_sum"].shift(5)), ("p_ad_gt_sma", ms["p_ad_line_sma50"]),
               ("p_b50_up", ms["p_b50_chg10"]), ("p_ew_gt_sma50", ms["p_ew_valid_n"].where(ms["p_ew_valid_n"] >= 50)),
               ("p_ew_gt_sma200", ms["p_ew_valid_n"].where(ms["p_ew_valid_n"] >= 200)), ("bear_rebound", ms["nifty_dd_252"])):
    ms.loc[dep.isna(), c] = np.nan

def resolve_threshold(col, spec):
    if isinstance(spec, tuple):
        return float(np.nanpercentile(ms.loc[H1_SESS, col].to_numpy(float), spec[1]))
    return float(spec)

def gate_series(col, op, thr):
    s = ms[col].astype(float)
    on = (s > thr) if op == ">" else (s < thr)
    return on.where(s.notna(), np.nan)          # NaN where the signal is undefined

# ---------------------------------------------------------------------------
# 4. evaluation
# ---------------------------------------------------------------------------
def cluster_t(y, x, cl):
    """t-stat of the slope in y = a + b*x with standard errors clustered by cl."""
    X = np.column_stack([np.ones(len(x)), x.astype(float)])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    codes = pd.factorize(cl)[0]
    S = np.zeros((2, 2))
    Xe = X * e[:, None]
    for c in np.unique(codes):
        u = Xe[codes == c].sum(axis=0)
        S += np.outer(u, u)
    V = XtX_inv @ S @ XtX_inv
    return float(beta[1] / np.sqrt(V[1, 1])) if V[1, 1] > 0 else float("nan")

def cell(df, on):
    """stats for one strategy x half given the boolean gate per trade."""
    k, r = df[on], df[~on]
    out = dict(n_all=int(len(df)), mean_all=round(df["net"].mean(), 3),
               n_on=int(len(k)), mean_on=round(k["net"].mean(), 3) if len(k) else None,
               win_on=round((k["net"] > 0).mean() * 100, 1) if len(k) else None,
               n_off=int(len(r)), mean_off=round(r["net"].mean(), 3) if len(r) else None,
               win_off=round((r["net"] > 0).mean() * 100, 1) if len(r) else None)
    out["diff"] = round(out["mean_on"] - out["mean_off"], 3) if len(k) and len(r) else None
    out["t_cluster"] = round(cluster_t(df["net"].to_numpy(float), on.to_numpy(), df["date"].to_numpy()), 2) \
        if len(k) >= 5 and len(r) >= 5 else None
    out["testable"] = bool(len(r) >= MIN_OFF and len(k) >= MIN_OFF)
    out["pass"] = bool(out["testable"] and out["mean_on"] > out["mean_off"])
    return out

def evaluate(on_by_date, name):
    """on_by_date: pd.Series over CAL with 1/0/NaN."""
    tr = T.merge(on_by_date.rename("gate").reset_index().rename(columns={"index": "date"}), on="date", how="left")
    tr = tr[tr["gate"].notna()].copy()
    tr["gate"] = tr["gate"].astype(bool)
    res = {}
    for s in ALL_STRATS + ["POOLED_MAIN"]:
        d = tr[tr["strategy"].isin(MAIN)] if s == "POOLED_MAIN" else tr[tr["strategy"] == s]
        res[s] = {"H1": cell(d[d["date"] < SPLIT], d.loc[d["date"] < SPLIT, "gate"]),
                  "H2": cell(d[d["date"] >= SPLIT], d.loc[d["date"] >= SPLIT, "gate"])}
    n_pass = sum(res[s]["H1"]["pass"] and res[s]["H2"]["pass"] for s in MAIN)
    n_testable = sum(res[s]["H1"]["testable"] and res[s]["H2"]["testable"] for s in MAIN)
    defined = on_by_date.notna()
    tim = {"all": round(on_by_date[defined].mean() * 100, 1),
           "H1": round(on_by_date.reindex(H1_SESS).mean() * 100, 1),
           "H2": round(on_by_date.reindex(H2_SESS).mean() * 100, 1)}
    pooled = res["POOLED_MAIN"]
    # regime-scale evidence: months where the gate was mostly on vs mostly off (pooled main strategies)
    dm = tr[tr["strategy"].isin(MAIN)].copy(); dm["m"] = dm["date"].dt.to_period("M")
    mon = dm.groupby("m").agg(net=("net", "mean"), n=("net", "size"), on=("gate", "mean"))
    mon = mon[mon["n"] >= 20]
    on_m, off_m = mon[mon["on"] >= 0.5], mon[mon["on"] < 0.5]
    month_stats = dict(months_mostly_on=int(len(on_m)), months_mostly_off=int(len(off_m)),
                       mean_of_month_means_on=round(on_m["net"].mean(), 3) if len(on_m) else None,
                       mean_of_month_means_off=round(off_m["net"].mean(), 3) if len(off_m) else None,
                       pct_off_months_negative=round((off_m["net"] < 0).mean() * 100, 0) if len(off_m) else None,
                       pct_on_months_negative=round((on_m["net"] < 0).mean() * 100, 0) if len(on_m) else None)
    # day-scale evidence: within each quarter, do kept trades beat removed trades?
    dm["q"] = dm["date"].dt.to_period("Q"); wq = []
    for q, x in dm.groupby("q"):
        a, b = x.loc[x["gate"], "net"], x.loc[~x["gate"], "net"]
        if len(a) >= 20 and len(b) >= 20:
            wq.append(round(float(a.mean() - b.mean()), 2))
    within_q = dict(quarters=len(wq), wins=int(sum(v > 0 for v in wq)), diffs=wq)
    promoted = bool(n_testable == 3 and n_pass == 3)
    robust = bool(promoted and tim["all"] >= MIN_TIM and
                  all(pooled[h]["diff"] is not None and pooled[h]["diff"] >= 0.5 and
                      pooled[h]["t_cluster"] is not None and pooled[h]["t_cluster"] >= 2.0 for h in ("H1", "H2")))
    return dict(time_in_market=tim, n_trades_evaluated=int(len(tr)),
                n_trades_dropped_undefined=int(len(T) - len(tr)), strategies_passing_both_halves=n_pass,
                strategies_testable=n_testable, promoted=promoted, robust=robust,
                usable=bool(tim["all"] >= MIN_TIM), month_level=month_stats, within_quarter=within_q, results=res)

results = {}
gate_bool = {}
for name, (col, op, spec, src, note) in GATES.items():
    thr = resolve_threshold(col, spec)
    on = gate_series(col, op, thr)
    gate_bool[name] = on
    r = evaluate(on, name)
    r.update(dict(signal=col, direction=op, threshold=round(thr, 4),
                  threshold_from="H1 percentile %d" % spec[1] if isinstance(spec, tuple) else "fixed",
                  source=src, definition=note))
    results[name] = r
print(f"{len(results)} gates evaluated, {time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------------------
# 5. combinations of the promoted gates (AND / OR)
# ---------------------------------------------------------------------------
promoted = [n for n, r in results.items() if r["promoted"] and r["usable"]]
combos = {}
for i, a in enumerate(promoted):
    for b in promoted[i + 1:]:
        ga, gb = gate_bool[a], gate_bool[b]
        both_def = ga.notna() & gb.notna()
        for mode in ("AND", "OR"):
            on = ((ga == 1) & (gb == 1)) if mode == "AND" else ((ga == 1) | (gb == 1))
            on = on.astype(float).where(both_def, np.nan)
            r = evaluate(on, f"{a} {mode} {b}")
            r.update(dict(gate_a=a, gate_b=b, mode=mode))
            combos[f"{a} {mode} {b}"] = r
# hand-specified literature combinations: slow uptrend regime AND a short-term washout ("buy the dip in an uptrend",
# Connors-style), plus the plain regime for reference. NOT(x) is written as ~x.
def G(n): return gate_bool[n]
LIT = {
    "ew_above_sma200 AND NOT breadth50_gt_50":      (("ew_above_sma200", "breadth50_gt_50"),   lambda a, b: (a == 1) & (b == 0)),
    "nifty_above_200dma AND NOT nifty_above_20dma": (("nifty_above_200dma", "nifty_above_20dma"), lambda a, b: (a == 1) & (b == 0)),
    "nifty_above_200dma AND vix_spike":             (("nifty_above_200dma", "no_vix_spike"),    lambda a, b: (a == 1) & (b == 0)),
    "zweig_thrust_window AND NOT breadth50_gt_50":  (("zweig_thrust_window", "breadth50_gt_50"), lambda a, b: (a == 1) & (b == 0)),
    "breadth50_hysteresis AND NOT breadth50_gt_50": (("breadth50_hysteresis", "breadth50_gt_50"), lambda a, b: (a == 1) & (b == 0)),
    "breadth50_hysteresis OR zweig_thrust_window":  (("breadth50_hysteresis", "zweig_thrust_window"), lambda a, b: (a == 1) | (b == 1)),
    "breadth50_sma20_gt_50 OR zweig_thrust_window": (("breadth50_sma20_gt_50", "zweig_thrust_window"), lambda a, b: (a == 1) | (b == 1)),
    "nhnl20_positive OR zweig_thrust_window":       (("nhnl20_positive", "zweig_thrust_window"), lambda a, b: (a == 1) | (b == 1)),
    "breadth50_hysteresis AND nhnl20_positive":     (("breadth50_hysteresis", "nhnl20_positive"), lambda a, b: (a == 1) & (b == 1)),
}
for k, ((pa, pb), fn) in LIT.items():
    ga, gb = gate_bool[pa], gate_bool[pb]
    on = fn(ga, gb).astype(float).where(ga.notna() & gb.notna(), np.nan)
    r = evaluate(on, k)
    r.update(dict(mode="LIT", gate_a=pa, gate_b=pb))
    combos[k] = r
print(f"{len(combos)} combinations evaluated, {time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------------------
# 6. summary + write
# ---------------------------------------------------------------------------
def pooled_min_diff(r):
    p = r["results"]["POOLED_MAIN"]
    return min(p["H1"]["diff"] if p["H1"]["diff"] is not None else -99,
               p["H2"]["diff"] if p["H2"]["diff"] is not None else -99)

ranked = sorted(results.items(), key=lambda kv: (-kv[1]["promoted"], -kv[1]["robust"], -pooled_min_diff(kv[1])))
combo_ranked = sorted(combos.items(), key=lambda kv: (-kv[1]["promoted"], -pooled_min_diff(kv[1])))
base = {s: {h: round(T.loc[(T["strategy"] == s) & ((T["date"] < SPLIT) if h == "H1" else (T["date"] >= SPLIT)), "net"].mean(), 3)
            for h in ("H1", "H2")} for s in ALL_STRATS}
summary = dict(
    split=str(SPLIT.date()), main_strategies=MAIN, n_trades=int(len(T)),
    h1_window=[str(H1_SESS.min().date()), str(H1_SESS.max().date())],
    h2_window=[str(H2_SESS.min().date()), str(H2_SESS.max().date())],
    unfiltered_mean_net=base, n_gates=len(results), n_promoted=len(promoted),
    promoted=promoted, robust=[n for n in promoted if results[n]["robust"]],
    promoted_partial_2of3=[n for n, r in results.items() if r["strategies_passing_both_halves"] == 2 and r["usable"]],
    best_combos=[k for k, v in combo_ranked[:5] if v["promoted"]],
    zweig_thrust_dates=[str(d.date()) for d in ms.index[ms["p_zweig_dates"] == 1]],
    not_computable=["VIX term structure (no India VIX futures / VIX3M series in the store)",
                    "FII daily flows (not in the store; midcap-vs-Nifty relative strength used as the risk-appetite proxy)"],
    multiple_comparisons_note="%d single gates and %d pairs tested; at a 5%% false-positive rate ~%d single gates "
                              "would pass one half by luck. Promotion requires both halves and all three strategies, "
                              "which cuts that sharply but not to zero; treat non-robust promotions as suggestive."
                              % (len(results), len(combos), round(len(results) * 0.05)))
json.dump(dict(summary=summary, gates=dict(ranked), combos=dict(combo_ranked)), open(OUT, "w"), indent=1, default=float)

# ---------------------------------------------------------------------------
# 7. console table
# ---------------------------------------------------------------------------
def fmt(c):
    if c["mean_on"] is None or c["mean_off"] is None:
        return f"{'n/a':>28s}"
    return f"{c['n_on']:>5d} {c['mean_on']:>+6.2f} | {c['n_off']:>5d} {c['mean_off']:>+6.2f} {c['t_cluster'] if c['t_cluster'] is not None else float('nan'):>+5.1f}"

print("\nunfiltered mean net (%):", {s: base[s] for s in MAIN})
print(f"\n{'gate':22s} {'TIM%':>5s} {'pass':>4s}  " + "  ".join(f"{s[:14]:^46s}" for s in MAIN))
print(f"{'':22s} {'':>5s} {'':>4s}  " + "  ".join(f"{'H1 on n mean | off n mean  t':^46s}" for s in MAIN))
for name, r in ranked:
    flag = "ROBUST" if r["robust"] else ("PROMO" if r["promoted"] else "")
    for h in ("H1", "H2"):
        lead = f"{name:22s} {r['time_in_market']['all']:>5.1f} {r['strategies_passing_both_halves']:>4d}" if h == "H1" else f"{'  '+h:22s} {'':>5s} {'':>4s}"
        print(f"{lead}  " + "  ".join(fmt(r["results"][s][h]) for s in MAIN) + (f"  {flag}" if h == "H1" else ""))
print("\nregime-scale vs day-scale evidence (pooled main strategies):")
print(f"{'gate':24s} {'mo_on':>5s} {'mo_off':>6s} {'mean_on':>7s} {'mean_off':>8s} {'off<0%':>6s} | {'within-Q wins':>13s}")
for name, r in ranked:
    if r["promoted"] or r["strategies_passing_both_halves"] >= 2:
        m, w = r["month_level"], r["within_quarter"]
        print(f"{name:24s} {m['months_mostly_on']:>5d} {m['months_mostly_off']:>6d} {m['mean_of_month_means_on'] if m['mean_of_month_means_on'] is not None else float('nan'):>+7.2f} "
              f"{m['mean_of_month_means_off'] if m['mean_of_month_means_off'] is not None else float('nan'):>+8.2f} {m['pct_off_months_negative'] if m['pct_off_months_negative'] is not None else float('nan'):>6.0f} | {w['wins']:>2d}/{w['quarters']:<2d} {w['diffs']}")
print("\npromoted:", promoted)
print("robust  :", summary["robust"])
print("\nbest combos:")
for k, v in combo_ranked[:10]:
    p = v["results"]["POOLED_MAIN"]
    print(f"  {k:60s} TIM {v['time_in_market']['all']:>5.1f}%  pass {v['strategies_passing_both_halves']}/3  "
          f"pooled diff H1 {p['H1']['diff']:+.2f} (t {p['H1']['t_cluster']:+.1f})  H2 {p['H2']['diff']:+.2f} (t {p['H2']['t_cluster']:+.1f})")
print(f"\nwrote {OUT}  ({time.time()-t0:.0f}s)")
