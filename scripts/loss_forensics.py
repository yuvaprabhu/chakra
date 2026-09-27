"""Loss forensics: what separates the bad trades from the good ones, BEFORE entry.

Input : data/screen/trades_tagged.parquet (one row per trade, stock + market
        state as-of the signal close, entry at next open).
Output: data/screen/loss_forensics.json

Method:
  * Every table is split into H1 (signal date < 2025-03-19) and H2 (>=).
    A relationship only counts if it has the same sign in both halves.
  * Quintile edges are fitted on H1 only and applied to H2.
  * Trades on the same signal day share the same market state, so the
    effective sample for market features is the number of DAYS, not trades.
    Every feature and every kept-vs-removed comparison therefore reports a
    t-statistic with standard errors clustered by signal date.
  * A feature "survives" only if its day-clustered |t| >= T_MIN in BOTH halves
    with the same sign. The null probability of that (and the expected number
    of false survivors given the number of features tested) is written to the
    JSON next to the observed count. The stricter "5 quintile means monotonic
    in both halves" flag is also reported as supplementary evidence.
  * Filters are built only from survivors: single thresholds at H1 deciles,
    then 2- and 3-way ANDs of the best threshold per feature. A filter passes
    only if, in both halves, it lifts win rate and mean net, removes good
    trades at a lower good-per-bad cost than random removal would, and still
    lets >= MIN_KEPT_PER_MONTH signals through. Each filter also reports
    months_active_share (regime concentration) and a per-quarter table.
  * MAE/MFE percentiles for winners/losers and the share of winners a stop at
    -X% would have knocked out.

Run: python scripts/loss_forensics.py  (about one minute)
"""
import json, time, itertools
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/screen/trades_tagged.parquet"
OUT = ROOT / "data/screen/loss_forensics.json"
SPLIT = pd.Timestamp("2025-03-19")
BAD, GOOD = -5.0, 3.0
BAD_DAY = -2.0
MIN_KEPT_PER_MONTH = 20        # per strategy, in each half
RHO_MONO = 0.9                 # Spearman over 5 quintile means; 1.0 = strictly monotonic
T_MIN = 2.0                    # day-clustered |t| required in EACH half, same sign

STOCK_NUM = ["rs_rank", "atr_pct", "dist_ema_21", "dist_sma_50", "dist_sma_10",
             "sma_150_slope", "sma_200_slope", "smooth_60", "pct_from_52w_high",
             "pct_from_52w_low", "rsi_2", "rsi_14", "vol_vs_ema21", "close_pos",
             "gap_pct", "ret_5d", "ret_20d", "ret_60d", "bb_pct_b", "m_cpr_width_rank",
             "base_depth", "turnover_median_20d", "n_signals_day", "n_signals_day_all"]
MARKET_NUM = ["nifty_vs_20dma", "nifty_vs_50dma", "nifty_vs_200dma", "nifty_20dma_slope",
              "nifty_ret_5d", "nifty_ret_20d", "nifty_ret_60d", "nifty_dd_252", "nifty_rvol_20",
              "midcap_vs_20dma", "midcap_vs_50dma", "midcap_vs_200dma", "midcap_20dma_slope",
              "midcap_ret_5d", "midcap_ret_20d", "midcap_ret_60d", "midcap_dd_252", "midcap_rvol_20",
              "bank_vs_20dma", "bank_vs_50dma", "bank_vs_200dma", "bank_20dma_slope",
              "bank_ret_5d", "bank_ret_20d", "bank_ret_60d", "bank_dd_252", "bank_rvol_20",
              "next50_vs_20dma", "next50_vs_50dma", "next50_vs_200dma", "next50_20dma_slope",
              "next50_ret_5d", "next50_ret_20d", "next50_ret_60d", "next50_dd_252", "next50_rvol_20",
              "mid_vs_nifty_20d", "next50_vs_nifty_20d", "vix", "vix_chg_5d", "vix_vs_20dma",
              "vix_pctile_1y", "nifty_range_20d", "breadth_above_50", "breadth_above_200",
              "breadth_above_20", "adv_ratio", "breadth_thrust_10", "pct_pos_20d",
              "new_hi_minus_lo", "dispersion", "ew_ret_1d", "ew_ret_20d", "breadth_50_chg_10d"]
BINARY = ["nifty_above_20dma", "nifty_above_50dma", "nifty_above_200dma",
          "midcap_above_20dma", "midcap_above_50dma", "midcap_above_200dma",
          "bank_above_20dma", "bank_above_50dma", "bank_above_200dma",
          "next50_above_20dma", "next50_above_50dma", "next50_above_200dma",
          "vix_above_15", "vix_above_20", "vix_spike", "turn_of_month", "expiry_week",
          "earnings_season"]
CATEG = ["cap_band", "dow", "month"]

t0 = time.time()
df = pd.read_parquet(SRC)
df["date"] = pd.to_datetime(df["date"])
df["half"] = np.where(df["date"] < SPLIT, "H1", "H2")
df["bad"] = df["net"] < BAD
df["good"] = df["net"] > GOOD
df["win"] = df["net"] > 0
# Crowding: how many signals fired that day (known at the signal close).
df["n_signals_day"] = df.groupby(["strategy", "date"])["net"].transform("size")
df["n_signals_day_all"] = df.groupby("date")["net"].transform("size")
STRATS = list(df["strategy"].unique())
print(f"{len(df):,} trades | {df['date'].min().date()} .. {df['date'].max().date()} | split {SPLIT.date()}", flush=True)


def r(x, k=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), k)


def cluster_t(y, x, g):
    """t-stat of slope in y = a + b*x with SEs clustered by g (signal date)."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    if len(y) < 10 or x.std() == 0:
        return np.nan
    X = np.column_stack([np.ones(len(x)), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    S = pd.DataFrame({"g": np.asarray(g), "e0": e, "e1": e * x}).groupby("g")[["e0", "e1"]].sum().to_numpy()
    V = XtX_inv @ (S.T @ S) @ XtX_inv
    return float(beta[1] / np.sqrt(V[1, 1])) if V[1, 1] > 0 else np.nan


def spearman5(vals):
    """Spearman rho between quintile index and the quintile statistic (5 points)."""
    v = np.asarray(vals, float)
    ok = np.isfinite(v)
    if ok.sum() < 4:
        return np.nan
    a = pd.Series(v[ok]).rank().to_numpy(); b = np.arange(ok.sum()) + 1.0
    return float(np.corrcoef(a, b)[0, 1])


def summarise(g):
    n = len(g)
    return {"n": int(n), "mean_net": r(g["net"].mean()), "win_rate": r(g["win"].mean()),
            "bad_share": r(g["bad"].mean()), "good_share": r(g["good"].mean()),
            "median_net": r(g["net"].median())}


def quintile_table(sub, feat, edges_from="H1"):
    """Quintile a feature with H1 edges; per-half tables + monotonicity flags."""
    x = sub[feat]
    h1 = sub[(sub["half"] == "H1") & x.notna()]
    if len(h1) < 100:
        return None
    qs = np.nanpercentile(h1[feat], [20, 40, 60, 80])
    if len(np.unique(qs)) < 4:                # near-constant / heavily tied feature
        return None
    q = pd.Series(np.searchsorted(qs, x.to_numpy(), side="right"), index=sub.index).where(x.notna())
    out = {"edges": [r(e, 4) for e in qs], "halves": {}, "flags": {}}
    means, bads, rho_m, rho_b = {}, {}, {}, {}
    for h in ("H1", "H2"):
        rows = []
        s = sub[sub["half"] == h]
        qq = q.loc[s.index]
        for k in range(5):
            g = s[qq == k]
            d = summarise(g) if len(g) else {"n": 0, "mean_net": None, "win_rate": None, "bad_share": None, "good_share": None, "median_net": None}
            d["q"] = k + 1
            rows.append(d)
        out["halves"][h] = rows
        means[h] = [d["mean_net"] if d["mean_net"] is not None else np.nan for d in rows]
        bads[h] = [d["bad_share"] if d["bad_share"] is not None else np.nan for d in rows]
        rho_m[h] = spearman5(means[h]); rho_b[h] = spearman5(bads[h])
        # trade-level spearman + day-clustered t, per half
        s2 = s[s[feat].notna()]
        if len(s2) > 30:
            out["halves"][h + "_rho_trade"] = r(s2[feat].rank().corr(s2["net"].rank()))
            out["halves"][h + "_t_cluster"] = r(cluster_t(s2["net"], s2[feat].rank(pct=True), s2["date"]), 2)
    mono_mean = (np.isfinite(rho_m["H1"]) and np.isfinite(rho_m["H2"])
                 and abs(rho_m["H1"]) >= RHO_MONO and abs(rho_m["H2"]) >= RHO_MONO
                 and np.sign(rho_m["H1"]) == np.sign(rho_m["H2"]))
    mono_bad = (np.isfinite(rho_b["H1"]) and np.isfinite(rho_b["H2"])
                and abs(rho_b["H1"]) >= RHO_MONO and abs(rho_b["H2"]) >= RHO_MONO
                and np.sign(rho_b["H1"]) == np.sign(rho_b["H2"]))
    t1 = out["halves"].get("H1_t_cluster"); t2 = out["halves"].get("H2_t_cluster")
    t_rule = (t1 is not None and t2 is not None and np.sign(t1) == np.sign(t2)
              and abs(t1) >= T_MIN and abs(t2) >= T_MIN)
    direction = "higher_better" if (t1 or 0) + (t2 or 0) > 0 else "lower_better"
    out["flags"] = {"t_rule_both": bool(t_rule), "min_abs_t": r(min(abs(t1 or 0), abs(t2 or 0)), 2),"rho_mean_H1": r(rho_m["H1"]), "rho_mean_H2": r(rho_m["H2"]),
                    "rho_bad_H1": r(rho_b["H1"]), "rho_bad_H2": r(rho_b["H2"]),
                    "monotonic_mean_both": bool(mono_mean), "monotonic_bad_both": bool(mono_bad),
                    "direction": direction,
                    # effect size: worst-quintile minus best-quintile bad share, per half
                    "bad_spread_H1": r(np.nanmax(bads["H1"]) - np.nanmin(bads["H1"])),
                    "bad_spread_H2": r(np.nanmax(bads["H2"]) - np.nanmin(bads["H2"])),
                    "mean_spread_H1": r(np.nanmax(means["H1"]) - np.nanmin(means["H1"])),
                    "mean_spread_H2": r(np.nanmax(means["H2"]) - np.nanmin(means["H2"]))}
    return out


def category_table(sub, feat):
    out = {"halves": {}}
    for h in ("H1", "H2"):
        s = sub[sub["half"] == h]
        rows = []
        for val, g in s.groupby(feat, dropna=True, observed=True):
            d = summarise(g); d["value"] = str(val) if not isinstance(val, (int, np.integer)) else int(val)
            rows.append(d)
        out["halves"][h] = rows
    # same-sign check for binaries: (1 minus 0) mean net
    if sub[feat].dropna().nunique() == 2:
        diff = {}
        for h in ("H1", "H2"):
            s = sub[sub["half"] == h]
            a = s[s[feat] == 1]["net"].mean(); b = s[s[feat] == 0]["net"].mean()
            diff[h] = r(a - b)
            s2 = s[s[feat].notna()]
            diff[h + "_t_cluster"] = r(cluster_t(s2["net"], s2[feat], s2["date"]), 2)
        diff["same_sign"] = bool(np.sign(diff["H1"] or 0) == np.sign(diff["H2"] or 0) and diff["H1"] not in (None, 0))
        out["diff_1_minus_0"] = diff
    return out


def null_prob_monotonic(rho=RHO_MONO):
    """P(|spearman| >= rho on 5 iid points) per half, then both halves same sign."""
    perms = list(itertools.permutations(range(5)))
    b = np.arange(5) + 1.0
    cnt = 0
    for p in perms:
        rr = np.corrcoef(np.array(p) + 1.0, b)[0, 1]
        if rr >= rho:
            cnt += 1
    p_one_dir = cnt / len(perms)
    return 2 * p_one_dir ** 2


# ---------------------------------------------------------------- filters
def eval_filter(sub, mask, base):
    """Per-half stats for keeping `mask`. base: per-half base rates for cost baseline."""
    res = {}
    for h in ("H1", "H2"):
        s = sub[sub["half"] == h]; m = mask.loc[s.index].fillna(False).to_numpy(bool)
        kept, rem = s[m], s[~m]
        n_bad = s["bad"].sum(); n_good = s["good"].sum()
        bad_rem = rem["bad"].sum(); good_rem = rem["good"].sum()
        months = max((s["date"].max() - s["date"].min()).days / 30.44, 1) if len(s) else 1
        d = {"n_all": int(len(s)), "n_kept": int(len(kept)), "kept_pct": r(len(kept) / max(len(s), 1)),
             "kept_per_month": r(len(kept) / months, 1),
             "wr_kept": r(kept["win"].mean()) if len(kept) else None,
             "mean_kept": r(kept["net"].mean()) if len(kept) else None,
             "median_kept": r(kept["net"].median()) if len(kept) else None,
             "bad_share_kept": r(kept["bad"].mean()) if len(kept) else None,
             "mean_removed": r(rem["net"].mean()) if len(rem) else None,
             "wr_removed": r(rem["win"].mean()) if len(rem) else None,
             "bad_removed_share": r(bad_rem / max(n_bad, 1)),
             "good_removed_share": r(good_rem / max(n_good, 1)),
             "cost_good_per_bad": r(good_rem / bad_rem, 2) if bad_rem > 0 else None,
             "cost_random_baseline": r(n_good / max(n_bad, 1), 2),
             "t_cluster_kept_vs_removed": r(cluster_t(s["net"], m.astype(float), s["date"]), 2)
             if 0 < m.sum() < len(s) else None}
        # regime concentration: share of calendar months in which the filter still lets >= 5 trades through
        mo_all = s["date"].dt.to_period("M"); mo_kept = kept["date"].dt.to_period("M").value_counts()
        d["months_active_share"] = r((mo_kept >= 5).sum() / max(mo_all.nunique(), 1))
        d["n_days_kept"] = int(kept["date"].nunique())
        res[h] = d
    return res


def filter_passes(res, base):
    """Helps in BOTH halves: higher WR, higher mean, cost below random baseline, enough flow."""
    ok = True
    for h in ("H1", "H2"):
        d = res[h]
        if d["wr_kept"] is None or d["mean_kept"] is None:
            return False
        ok &= d["wr_kept"] > base[h]["win_rate"]
        ok &= d["mean_kept"] > base[h]["mean_net"]
        ok &= d["cost_good_per_bad"] is not None and d["cost_good_per_bad"] < d["cost_random_baseline"]
        ok &= d["kept_per_month"] >= MIN_KEPT_PER_MONTH
    return bool(ok)


def score(res):
    """Rank: min win rate of kept across halves, then min mean_kept, then flow."""
    return (min(res["H1"]["wr_kept"], res["H2"]["wr_kept"]),
            min(res["H1"]["mean_kept"], res["H2"]["mean_kept"]),
            min(res["H1"]["kept_pct"], res["H2"]["kept_pct"]))


# ---------------------------------------------------------------- main loop
report = {"meta": {"trades": int(len(df)), "split": str(SPLIT.date()), "bad": BAD, "good": GOOD,
                   "bad_day": BAD_DAY, "min_kept_per_month": MIN_KEPT_PER_MONTH,
                   "null_prob_monotonic_both_halves": r(null_prob_monotonic(), 5),
                   "note": "Quintile edges from H1; monotonic flag = |spearman(q, stat)| >= %.2f in both halves, same sign." % RHO_MONO},
          "strategies": {}}

for strat in STRATS:
    ts = time.time()
    sub = df[df["strategy"] == strat].copy()
    S = {"base": {h: summarise(sub[sub["half"] == h]) for h in ("H1", "H2")}}
    S["base"]["all"] = summarise(sub)
    for h in ("H1", "H2"):
        s = sub[sub["half"] == h]
        S["base"][h]["days"] = int(s["date"].nunique())
        S["base"][h]["months"] = r((s["date"].max() - s["date"].min()).days / 30.44, 1)

    # (b) numeric quintiles, trade level
    S["numeric"] = {}
    for feat in STOCK_NUM + MARKET_NUM:
        t = quintile_table(sub, feat)
        if t is not None:
            t["level"] = "stock" if feat in STOCK_NUM else "market"
            S["numeric"][feat] = t
    # (c) binary / categorical
    S["categorical"] = {feat: category_table(sub, feat) for feat in BINARY + CATEG}

    # (d) DAY level: mean net per signal day vs market state that day
    day = sub.groupby("date").agg(net=("net", "mean"), n=("net", "size"),
                                  **{f: (f, "first") for f in MARKET_NUM + BINARY + ["n_signals_day", "n_signals_day_all"]}).reset_index()
    day["half"] = np.where(day["date"] < SPLIT, "H1", "H2")
    day["bad"] = day["net"] < BAD_DAY; day["good"] = day["net"] > GOOD; day["win"] = day["net"] > 0
    S["day_base"] = {h: {"days": int((day["half"] == h).sum()),
                         "bad_day_share": r(day[day["half"] == h]["bad"].mean()),
                         "mean_day_net": r(day[day["half"] == h]["net"].mean()),
                         "trades_per_day": r(day[day["half"] == h]["n"].mean(), 1)} for h in ("H1", "H2")}
    S["day_numeric"] = {}
    for feat in MARKET_NUM + ["n_signals_day", "n_signals_day_all"]:
        t = quintile_table(day, feat)
        if t is not None:
            S["day_numeric"][feat] = t
    S["day_binary"] = {feat: category_table(day, feat) for feat in BINARY}

    # survivors
    surv_trade = [f for f, t in S["numeric"].items() if t["flags"]["t_rule_both"]]
    surv_day = [f for f, t in S["day_numeric"].items() if t["flags"]["t_rule_both"]]
    mono_trade = [f for f, t in S["numeric"].items() if t["flags"]["monotonic_mean_both"] or t["flags"]["monotonic_bad_both"]]
    surv_bin = [f for f, t in S["categorical"].items() if t.get("diff_1_minus_0", {}).get("same_sign")
                and abs(t["diff_1_minus_0"]["H1_t_cluster"] or 0) >= T_MIN and abs(t["diff_1_minus_0"]["H2_t_cluster"] or 0) >= T_MIN]
    from math import erfc, sqrt
    sf = 0.5 * erfc(T_MIN / sqrt(2))              # one-sided normal tail
    p_t = 2 * sf * sf                              # P(both |t|>=T and same sign) under null
    S["survivors"] = {"trade_level": surv_trade, "day_level": surv_day, "binary_same_sign": surv_bin,
                      "monotonic_quintiles_trade": mono_trade,
                      "n_tests_trade": len(S["numeric"]), "n_tests_day": len(S["day_numeric"]),
                      "null_prob_t_rule": r(p_t, 5),
                      "expected_false_trade": r(len(S["numeric"]) * p_t, 2),
                      "expected_false_day": r(len(S["day_numeric"]) * p_t, 2),
                      "expected_false_monotonic": r(len(S["numeric"]) * report["meta"]["null_prob_monotonic_both_halves"] * 2, 2)}

    # (3) filters: single thresholds on survivors (+ binaries), then 2-3 way ANDs
    base = S["base"]
    singles = []
    cand_feats = sorted(set(surv_trade) | set(surv_day))
    h1 = sub[sub["half"] == "H1"]
    n_tests = 0
    for feat in cand_feats:
        t = S["numeric"].get(feat) or S["day_numeric"].get(feat)
        direction = t["flags"]["direction"]
        for p in (10, 20, 30, 40, 50, 60, 70, 80, 90):
            thr = float(np.nanpercentile(h1[feat], p))
            if direction == "higher_better":
                mask = sub[feat] >= thr; label = f"{feat} >= {thr:.4g}"
            else:
                mask = sub[feat] <= thr; label = f"{feat} <= {thr:.4g}"
            n_tests += 1
            res = eval_filter(sub, mask, base)
            if res["H1"]["n_kept"] < 30 or res["H2"]["n_kept"] < 30:
                continue
            singles.append({"filter": label, "feature": feat, "op": ">=" if direction == "higher_better" else "<=",
                            "thr": thr, "pctile_H1": p, "passes_both": filter_passes(res, base), "H1": res["H1"], "H2": res["H2"]})
    for feat in surv_bin:
        d = S["categorical"][feat]["diff_1_minus_0"]
        keep_val = 1 if d["H1"] > 0 else 0
        mask = sub[feat] == keep_val
        n_tests += 1
        res = eval_filter(sub, mask, base)
        singles.append({"filter": f"{feat} == {keep_val}", "feature": feat, "op": "==", "thr": keep_val,
                        "passes_both": filter_passes(res, base), "H1": res["H1"], "H2": res["H2"]})
    passing = [f for f in singles if f["passes_both"]]
    passing.sort(key=lambda f: score(f), reverse=True)
    # best threshold per feature, then combos of the top distinct features
    best_per_feat = {}
    for f in passing:
        if f["feature"] not in best_per_feat:
            best_per_feat[f["feature"]] = f
    top = list(best_per_feat.values())[:8]

    def mask_of(f):
        if f["op"] == ">=":
            return sub[f["feature"]] >= f["thr"]
        if f["op"] == "<=":
            return sub[f["feature"]] <= f["thr"]
        return sub[f["feature"]] == f["thr"]

    combos = []
    for k in (2, 3):
        for grp in itertools.combinations(top, k):
            m = mask_of(grp[0])
            for f in grp[1:]:
                m = m & mask_of(f)
            n_tests += 1
            res = eval_filter(sub, m, base)
            if res["H1"]["n_kept"] < 30 or res["H2"]["n_kept"] < 30:
                continue
            combos.append({"filter": " AND ".join(f["filter"] for f in grp), "features": [f["feature"] for f in grp],
                           "k": k, "passes_both": filter_passes(res, base), "H1": res["H1"], "H2": res["H2"]})
    leaderboard = [f for f in singles + combos if f["passes_both"]]
    leaderboard.sort(key=lambda f: score(f), reverse=True)
    # the user's budget: skip at most 70%, i.e. keep >= 30% in both halves
    budget = [f for f in leaderboard if f["H1"]["kept_pct"] >= 0.30 and f["H2"]["kept_pct"] >= 0.30]

    def mask_of_any(f):
        if "features" in f:                       # combo
            parts = [p.strip() for p in f["filter"].split(" AND ")]
            m = None
            for part in parts:
                feat, op, thr = part.split(" ")
                mm = (sub[feat] >= float(thr)) if op == ">=" else (sub[feat] <= float(thr)) if op == "<=" else (sub[feat] == float(thr))
                m = mm if m is None else (m & mm)
            return m
        return mask_of(f)

    def quarterly(f):
        m = mask_of_any(f).fillna(False)
        q = sub["date"].dt.to_period("Q").astype(str)
        rows = []
        for qq, g in sub.groupby(q):
            k = g[m.loc[g.index]]
            rows.append({"q": qq, "n_all": int(len(g)), "n_kept": int(len(k)), "wr_all": r(g["win"].mean()),
                         "wr_kept": r(k["win"].mean()) if len(k) else None, "mean_all": r(g["net"].mean()),
                         "mean_kept": r(k["net"].mean()) if len(k) else None})
        beats = [x for x in rows if x["wr_kept"] is not None and x["n_kept"] >= 10]
        return {"by_quarter": rows,
                "quarters_kept_beats_all_wr": f"{sum(x['wr_kept'] > x['wr_all'] for x in beats)}/{len(beats)}",
                "quarters_kept_beats_all_mean": f"{sum(x['mean_kept'] > x['mean_all'] for x in beats)}/{len(beats)}",
                "min_quarter_wr_kept": r(min(x["wr_kept"] for x in beats)) if beats else None}

    for f in leaderboard[:8] + budget[:8]:
        if "quarterly" not in f:
            f["quarterly"] = quarterly(f)
    S["filters"] = {"n_tests": n_tests, "n_single_tested": len(singles), "n_single_passing": len(passing),
                    "n_combo_tested": len(combos), "n_combo_passing": sum(f["passes_both"] for f in combos),
                    "leaderboard": leaderboard[:40],
                    "leaderboard_keep_ge_30pct": budget[:20],
                    "best_single_per_feature": top,
                    "all_singles": singles}

    # (6) MAE / MFE distributions
    pct = [10, 25, 50, 75, 90]
    mm = {}
    for name, m in (("winners", sub["net"] > 0), ("losers", sub["net"] <= 0),
                    ("good_gt3", sub["good"]), ("bad_lt_m5", sub["bad"])):
        g = sub[m]
        mm[name] = {"n": int(len(g)),
                    "mae_pct": {str(p): r(np.percentile(g["mae"], p), 2) for p in pct},
                    "mfe_pct": {str(p): r(np.percentile(g["mfe"], p), 2) for p in pct},
                    "bars_to_mae_median": r(g["bars_to_mae"].median(), 1),
                    "bars_to_mfe_median": r(g["bars_to_mfe"].median(), 1)}
    # stop survival: share of winners / losers whose MAE breached -X%
    stops = {}
    w = sub[sub["net"] > 0]; l = sub[sub["net"] <= 0]
    for X in (2, 3, 4, 5, 6, 7, 8, 10, 12):
        stops[str(X)] = {"winners_stopped": r((w["mae"] <= -X).mean()), "losers_stopped": r((l["mae"] <= -X).mean()),
                         "bad_stopped": r((sub[sub["bad"]]["mae"] <= -X).mean())}
    mm["stop_breach_share"] = stops
    S["mae_mfe"] = mm

    report["strategies"][strat] = S
    print(f"  {strat:32s} surv trade={len(surv_trade):2d} day={len(surv_day):2d} bin={len(surv_bin):2d} | "
          f"filters tested={n_tests} passing={len(leaderboard)} | {time.time()-ts:.1f}s", flush=True)

OUT.write_text(json.dumps(report, indent=1, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else (o.item() if hasattr(o, "item") else str(o))))
print(f"wrote {OUT} in {time.time()-t0:.0f}s")
