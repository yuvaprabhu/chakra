"""Which CONFIRMATION SIGNALS pick the best Double Seven trades?

The audited trade list (scripts/double_seven_audit.py -> double_seven_trades.csv,
one open position per name, 0.30% round-trip cost) is taken as the candidate
set. Every candidate feature is read AT THE SIGNAL CLOSE from the feature panel
(features.parquet rows are as-of that date; every column used is a rolling or
shifted quantity in screener/indicators.py, or a same-date cross-section).

Discipline, fixed before looking at any number:

* IS  = signal date <  2025-07-01, OOS = signal date >= 2025-07-01.
* Quintile edges come from the IS distribution only and are applied unchanged
  to OOS. Binary and categorical features use their natural bins.
* A feature is PROMOTED only if it is monotonic in BOTH periods:
    - Spearman(feature, net) has the same sign IS and OOS,
    - IS |rho| >= 0.03 with p < 0.05, and OOS |rho| >= 0.5 * IS |rho|,
    - the top-vs-bottom bin difference has the same sign in both periods and
      the bin means are ordered (Spearman across bin means >= 0.5) in both.
  For binary features the test is the 1-vs-0 difference: same sign in both
  periods, OOS magnitude >= 0.5 * IS, IS Welch t >= 2.
* One composite only: the equal-weight mean of the WITHIN-DAY percentile rank
  of the promoted features (2-4, best by min(|rho_IS|, |rho_OOS|)), sign-
  adjusted so higher = better. No weights are fitted. Per-day regime features
  are constant within a day, so they can act only as a gate, never in a
  within-day ranking, and are excluded from the composite by construction.
* Top-K per day for K = 3, 5, 10 with one open position per name, compared
  with 200 random draws of the same K on the same days.

Multiple comparisons: about 22 columns x 5 bins x 2 periods are examined.
With that many looks, a handful of bins WILL be at +/-1% by chance. Only the
cross-period consistency test above is used to promote anything, and even that
is a weak guard with ~3,000 OOS trades.

Run from the repo root:  python3 scripts/double_seven_selection.py
Writes data/screen/double_seven_selection.csv and
       data/screen/double_seven_topk.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from screener.backtest import add_panel_columns, concentration  # noqa: E402

SPLIT = pd.Timestamp("2025-07-01")
K_LIST = (3, 5, 10)
N_RANDOM = 200
SEED = 7
BIG_LOSS = -8.0          # "loses more than 8%" threshold on net return
N_BINS = 5
MAX_COMPOSITE = 4
MIN_COMPOSITE = 2

TRADES_CSV = "data/screen/double_seven_trades.csv"
FEATURES = "data/screen/features.parquet"
LATEST = "data/screen/latest_features.parquet"
OUT_CSV = "data/screen/double_seven_selection.csv"
OUT_JSON = "data/screen/double_seven_topk.json"

# (column, kind, hypothesis, description).  hypothesis is the sign we EXPECT
# for Spearman(feature, net) before looking; it is recorded, not used.
# kind: "q" = quintiles by IS edges, "b" = binary 0/1, "c" = ordered category.
CANDIDATES = [
    ("rs_rank",          "q", +1, "1. relative strength percentile on the day"),
    ("smooth_60",        "q", +1, "2. % up days over 60 (frog in the pan)"),
    ("dist_ema_21",      "q",  0, "3. % distance from EMA21 (pullback depth)"),
    ("dist_sma_10",      "q",  0, "3. % distance from SMA10 (pullback depth)"),
    ("rsi_2",            "q", -1, "4. RSI(2) at the signal close"),
    ("pb_vol_dry",       "q", -1, "5. 3d vol / prior 10d vol (dry pullback)"),
    ("vol_vs_ema21",     "q", -1, "5. day volume / EMA21 volume"),
    ("close_pos",        "q", +1, "6. close position in day range"),
    ("lower_wick",       "q", +1, "6. lower wick share of range"),
    ("hammer",           "b", +1, "6. hammer candle on signal day"),
    ("any_reversal",     "b", +1, "6. any reversal candle on signal day"),
    ("sma_150_slope",    "q", +1, "7. 20d slope of SMA150"),
    ("dist_sma_50",      "q",  0, "7. % distance from SMA50"),
    ("ema_stack",        "b", +1, "7. EMA21 > EMA50 > EMA200"),
    ("above_m_p",        "b", +1, "8. close above monthly pivot"),
    ("dist_m_p",         "q",  0, "8. % distance from monthly pivot"),
    ("m_cpr_width_rank", "q",  0, "8. monthly CPR width vs own 12m history"),
    ("breadth_50",       "q", +1, "9. REGIME: % universe above SMA50 (per day)"),
    ("ew_ret_20",        "q", +1, "9. REGIME: universe EW 20d return (per day)"),
    ("down_streak",      "q",  0, "10. consecutive down closes incl. signal day"),
    ("pb_days",          "q",  0, "10. sessions since 10d high"),
    ("mom_12_1",         "q", +1, "11. 12-1 month momentum"),
    ("cap_band",         "c",  0, "12. cap band (latest snapshot)"),
]
REGIME_COLS = {"breadth_50", "ew_ret_20"}
CAP_ORDER = ["Mid", "Large", "Mega"]


# --- data ------------------------------------------------------------------

def load_trades(path: str = TRADES_CSV) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["date"])
    d["mo"] = d["date"].dt.to_period("M")
    return d


def build_signal_features(trades: pd.DataFrame, feat_path: str = FEATURES,
                          latest_path: str = LATEST) -> pd.DataFrame:
    """Join every candidate feature at the signal (isin, date); add derived ones."""
    base = ["isin", "date", "adj_close", "sma_50", "ema_21", "ema_50", "ema_200",
            "ret_20d", "ret_60d", "ret_120d", "ret_250d", "atr_pct"]   # atr_pct: diagnostic only
    want = [c for c, _, _, _ in CANDIDATES
            if c not in ("rs_rank", "ema_stack", "breadth_50", "ew_ret_20", "down_streak", "cap_band")]
    cols = list(dict.fromkeys(base + want))
    f = pd.read_parquet(feat_path, columns=cols)
    f = add_panel_columns(f).reset_index(drop=True)          # rs_rank, bars_available
    f["pos"] = f.groupby("isin").cumcount()

    # Consecutive down closes including today. The first row of every isin
    # has a NaN diff -> False, which resets the run at the symbol boundary.
    down = (f["adj_close"] < f.groupby("isin")["adj_close"].shift(1)).fillna(False)
    f["down_streak"] = down.groupby((~down).cumsum()).cumsum().astype(int)

    f["ema_stack"] = ((f["ema_21"] > f["ema_50"]) & (f["ema_50"] > f["ema_200"])).astype("int8")

    # Regime: same-day cross-section of the whole panel. No forward data.
    by = f.groupby("date")
    reg = pd.DataFrame({
        "breadth_50": by.apply(lambda g: float((g["adj_close"] > g["sma_50"]).mean() * 100)
                               if g["sma_50"].notna().any() else np.nan),
        "ew_ret_20": by["ret_20d"].mean(),
    })

    keep = ["isin", "date", "pos", "down_streak", "ema_stack", "rs_rank", "atr_pct"] + want
    d = trades.merge(f[keep], on=["isin", "date"], how="left", validate="one_to_one")
    d = d.merge(reg, left_on="date", right_index=True, how="left")

    # Exit date from the symbol's own calendar: exit bar = signal bar + bars.
    cal = f[["isin", "pos", "date"]].rename(columns={"pos": "exit_pos", "date": "exit_date"})
    d["exit_pos"] = d["pos"] + d["bars"]
    d = d.merge(cal, on=["isin", "exit_pos"], how="left")

    lat = pd.read_parquet(latest_path, columns=["isin", "symbol", "cap_band"])
    d = d.merge(lat, on="isin", how="left")
    d["cap_band"] = pd.Categorical(d["cap_band"], categories=CAP_ORDER, ordered=True)
    return d


# --- statistics ------------------------------------------------------------

def spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    """Spearman rho, two-sided p (normal approx on the t statistic), n."""
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 10:
        return float("nan"), float("nan"), n
    rho = float(x[m].rank().corr(y[m].rank()))
    if not np.isfinite(rho) or abs(rho) >= 1:
        return rho, 0.0, n
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    p = math.erfc(abs(t) / math.sqrt(2))
    return rho, p, n


def bin_by_is(x: pd.Series, is_mask: pd.Series, n_bins: int = N_BINS) -> pd.Series:
    """Quintile labels 1..n_bins using edges from the IS rows ONLY.

    OOS values outside the IS range fall into the end bins. Ties collapse
    bins, in which case fewer labels are returned (still IS-defined).
    """
    edges = np.unique(x[is_mask].dropna().quantile(np.linspace(0, 1, n_bins + 1)).to_numpy())
    if len(edges) < 2:
        return pd.Series(np.where(x.notna(), 1, np.nan), index=x.index)
    edges[0], edges[-1] = -np.inf, np.inf
    return pd.cut(x, edges, labels=False, include_lowest=True) + 1


def _stats(net: pd.Series) -> dict:
    net = net.dropna()
    n = len(net)
    if n == 0:
        return {"n": 0, "net": np.nan, "win": np.nan, "pf": np.nan, "big_loss": np.nan}
    w, l = net[net > 0].sum(), -net[net <= 0].sum()
    return {"n": int(n), "net": float(net.mean()), "win": float((net > 0).mean() * 100),
            "pf": float(w / l) if l > 0 else float("inf"),
            "big_loss": float((net < BIG_LOSS).mean() * 100),
            "avg_loss": float(net[net <= 0].mean()) if (net <= 0).any() else 0.0}


def _welch_t(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def feature_table(d: pd.DataFrame, is_mask: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-feature-per-bin table (IS and OOS side by side) and a verdict table."""
    rows, verdicts = [], []
    for col, kind, hyp, desc in CANDIDATES:
        if col not in d.columns:
            continue
        x = d[col]
        if kind == "c":
            xnum = x.cat.codes.astype(float).where(x.notna())
            bins = xnum + 1
            labels = {i + 1: c for i, c in enumerate(x.cat.categories)}
        elif kind == "b":
            xnum = pd.to_numeric(x, errors="coerce")
            bins = xnum + 1
            labels = {1: "0", 2: "1"}
        else:
            xnum = pd.to_numeric(x, errors="coerce")
            bins = bin_by_is(xnum, is_mask)
            labels = {}
            for b in sorted(bins.dropna().unique()):
                v = xnum[(bins == b) & is_mask]
                labels[int(b)] = f"[{v.min():.3g}, {v.max():.3g}]"

        per = {}
        for lab, m in (("IS", is_mask), ("OOS", ~is_mask)):
            rho, p, n = spearman(xnum[m], d.loc[m, "net"])
            per[lab] = {"rho": rho, "p": p, "n": n}
            means = {}
            for b in sorted(bins.dropna().unique()):
                s = _stats(d.loc[m & (bins == b), "net"])
                means[int(b)] = s["net"]
                rows.append({"feature": col, "kind": kind, "hypothesis": hyp, "desc": desc,
                             "period": lab, "bin": int(b), "bin_range": labels.get(int(b), ""),
                             **s, "rho": rho, "rho_p": p})
            ks = sorted(means)
            per[lab]["bins"] = means
            per[lab]["top_minus_bottom"] = (means[ks[-1]] - means[ks[0]]) if len(ks) >= 2 else np.nan
            if len(ks) >= 3:
                per[lab]["bin_mono"] = float(pd.Series([means[k] for k in ks]).rank()
                                             .corr(pd.Series(ks, dtype=float).rank()))
            else:
                per[lab]["bin_mono"] = np.sign(per[lab]["top_minus_bottom"])
            if kind == "b":
                per[lab]["t"] = _welch_t(d.loc[m & (bins == 2), "net"], d.loc[m & (bins == 1), "net"])

        i, o = per["IS"], per["OOS"]
        # 5 quintiles: bin means ordered "mostly" (Spearman >= 0.5); a 3-level
        # category has too few bins for that to mean anything, so it must be
        # strictly ordered in both periods.
        mono_min = 0.99 if kind == "c" else 0.5
        if kind == "b":
            di, do = i["top_minus_bottom"], o["top_minus_bottom"]
            ok = (np.isfinite(di) and np.isfinite(do) and np.sign(di) == np.sign(do) != 0
                  and abs(i.get("t", 0)) >= 2 and abs(do) >= 0.5 * abs(di))
        else:
            ok = (np.isfinite(i["rho"]) and np.isfinite(o["rho"])
                  and np.sign(i["rho"]) == np.sign(o["rho"]) != 0
                  and abs(i["rho"]) >= 0.03 and i["p"] < 0.05
                  and abs(o["rho"]) >= 0.5 * abs(i["rho"])
                  and np.sign(i["top_minus_bottom"]) == np.sign(o["top_minus_bottom"]) != 0
                  and i["bin_mono"] * np.sign(i["rho"]) >= mono_min
                  and o["bin_mono"] * np.sign(o["rho"]) >= mono_min)
        direction = lambda z: "+" if z > 0 else ("-" if z < 0 else "0")   # noqa: E731
        if kind == "b":
            ok_is = np.isfinite(i["top_minus_bottom"]) and abs(i.get("t", 0)) >= 2
        else:
            ok_is = (np.isfinite(i["rho"]) and abs(i["rho"]) >= 0.03 and i["p"] < 0.05
                     and i["bin_mono"] * np.sign(i["rho"]) >= mono_min)
        verdicts.append({
            "feature": col, "kind": kind, "desc": desc, "hypothesis": hyp,
            "rho_is": i["rho"], "p_is": i["p"], "rho_oos": o["rho"], "p_oos": o["p"],
            "q_diff_is": i["top_minus_bottom"], "q_diff_oos": o["top_minus_bottom"],
            "mono_is": i["bin_mono"], "mono_oos": o["bin_mono"],
            "dir_is": direction(i["rho"] if kind != "b" else i["top_minus_bottom"]),
            "dir_oos": direction(o["rho"] if kind != "b" else o["top_minus_bottom"]),
            "promoted": bool(ok) and col not in REGIME_COLS,
            "passes_is_only": bool(ok_is) and col not in REGIME_COLS,
            "regime_gate": col in REGIME_COLS,
            "sign": int(np.sign(i["rho"])) if kind != "b" else int(np.sign(i["top_minus_bottom"])),
        })
    return pd.DataFrame(rows), pd.DataFrame(verdicts)


# --- composite and top-K ---------------------------------------------------

def composite_score(d: pd.DataFrame, feats: dict[str, int], date_col: str = "date") -> pd.Series:
    """Equal-weight mean of within-day percentile ranks, sign-adjusted.

    ``feats`` maps column -> +1 (higher is better) or -1 (lower is better).
    Rank-based: any monotone transform of a feature leaves the score unchanged.
    A trade with a missing feature gets the neutral 0.5 for that feature.
    """
    parts = []
    for col, sign in feats.items():
        x = d[col]
        if hasattr(x, "cat"):
            x = x.cat.codes.astype(float).where(x.notna())
        x = pd.to_numeric(x, errors="coerce") * sign
        r = x.groupby(d[date_col]).rank(pct=True)
        parts.append(r.fillna(0.5))
    if not parts:
        return pd.Series(0.5, index=d.index)
    return pd.concat(parts, axis=1).mean(axis=1)


def top_k_per_day(d: pd.DataFrame, score: pd.Series, k: int) -> pd.Series:
    """Boolean mask: the K highest-scoring signals each day, one open position per name.

    A name whose earlier selected trade has not exited yet (exit_date >= this
    signal date, same convention as the audit loop) is not eligible that day.
    """
    dates = d["date"].to_numpy()
    isin = d["isin"].to_numpy()
    exit_d = d["exit_date"].to_numpy()
    sc = score.to_numpy(dtype=float)
    order = np.lexsort((-sc, dates))           # by date, then score descending
    sel = np.zeros(len(d), dtype=bool)
    busy: dict = {}
    i = 0
    n = len(order)
    while i < n:
        j = i
        day = dates[order[i]]
        taken = 0
        while j < n and dates[order[j]] == day:
            idx = order[j]
            if taken < k:
                b = busy.get(isin[idx])
                if b is None or b < day:
                    sel[idx] = True
                    busy[isin[idx]] = exit_d[idx]
                    taken += 1
            j += 1
        i = j
    return pd.Series(sel, index=d.index)


def selection_metrics(w: pd.DataFrame) -> dict:
    s = _stats(w["net"])
    if s["n"] == 0:
        return {**s, "months_up": np.nan, "worst_month": np.nan, "trades_per_month": np.nan,
                "top10_share": np.nan}
    m = w.groupby("mo")["net"].mean()
    conc = concentration(w, col="net")
    return {**s, "months_up": float((m > 0).mean() * 100), "worst_month": float(m.min()),
            "worst_month_label": str(m.idxmin()),
            "trades_per_month": float(len(w) / w["mo"].nunique()),
            "top10_share": conc.get("top10_share"),
            "median_atr_pct": float(w["atr_pct"].median()) if "atr_pct" in w.columns else None}


def random_baseline(d: pd.DataFrame, k: int, is_mask: pd.Series, n_draws: int = N_RANDOM,
                    seed: int = SEED) -> dict:
    """Average metrics of ``n_draws`` random top-K selections on the same days."""
    rng = np.random.default_rng(seed)
    acc = {"IS": [], "OOS": []}
    for _ in range(n_draws):
        sel = top_k_per_day(d, pd.Series(rng.random(len(d)), index=d.index), k)
        acc["IS"].append(selection_metrics(d[sel & is_mask]))
        acc["OOS"].append(selection_metrics(d[sel & ~is_mask]))
    out = {}
    for lab, lst in acc.items():
        df = pd.DataFrame(lst)
        out[lab] = {"mean": df.mean(numeric_only=True).to_dict(),
                    "std_net": float(df["net"].std()),
                    "net_draws": df["net"].tolist()}
    return out


# --- main ------------------------------------------------------------------

def _fmt(v, nd=2, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "   -  "
    return f"{v:+.{nd}f}" if not pct else f"{v:.0f}"


def main() -> None:
    t0 = time.time()
    trades = load_trades()
    d = build_signal_features(trades)
    is_mask = d["date"] < SPLIT
    print(f"DOUBLE SEVEN confirmation study: {len(d):,} trades, IS {int(is_mask.sum()):,} "
          f"/ OOS {int((~is_mask).sum()):,}; features joined in {time.time()-t0:.0f}s\n")
    for lab, m in (("IS ", is_mask), ("OOS", ~is_mask)):
        s = selection_metrics(d[m])
        print(f"  ALL {lab}: n {s['n']:,} net {s['net']:+.2f}% win {s['win']:.0f}% PF {s['pf']:.2f} "
              f"months up {s['months_up']:.0f}% worst {s['worst_month']:+.2f}% >8% losers {s['big_loss']:.1f}%")

    table, verdicts = feature_table(d, is_mask)
    table.to_csv(OUT_CSV, index=False, float_format="%.4f")

    print("\nPER-FEATURE BINS (edges from IS only). net = mean net %/trade, win %, PF, n")
    for col, kind, hyp, desc in CANDIDATES:
        v = verdicts[verdicts.feature == col]
        if v.empty:
            continue
        v = v.iloc[0]
        print(f"\n{col:18s} {desc}\n"
              f"  Spearman IS {v.rho_is:+.3f} (p={v.p_is:.3f})  OOS {v.rho_oos:+.3f} (p={v.p_oos:.3f})"
              f"  top-bottom IS {v.q_diff_is:+.2f} OOS {v.q_diff_oos:+.2f}"
              f"  -> {'PROMOTED' if v.promoted else ('regime gate only' if v.regime_gate else 'not promoted')}")
        t = table[table.feature == col]
        print(f"  {'bin':>3s} {'range (IS)':>22s} | {'IS net':>7s} {'win':>4s} {'PF':>5s} {'n':>5s} | "
              f"{'OOS net':>7s} {'win':>4s} {'PF':>5s} {'n':>5s}")
        for b in sorted(t["bin"].unique()):
            i = t[(t.bin == b) & (t.period == "IS")].iloc[0]
            o_ = t[(t.bin == b) & (t.period == "OOS")]
            o_ = o_.iloc[0] if len(o_) else None
            rng_ = i.bin_range if kind == "q" else (CAP_ORDER[b - 1] if kind == "c" else str(b - 1))
            print(f"  {b:>3d} {rng_:>22s} | {_fmt(i.net):>7s} {_fmt(i.win, pct=True):>4s} {i.pf:5.2f} {i.n:5d} | "
                  + (f"{_fmt(o_.net):>7s} {_fmt(o_.win, pct=True):>4s} {o_.pf:5.2f} {o_.n:5d}" if o_ is not None else ""))

    print("\nVERDICTS")
    print(verdicts[["feature", "dir_is", "dir_oos", "rho_is", "rho_oos", "q_diff_is", "q_diff_oos",
                    "promoted"]].round(3).to_string(index=False))

    # --- composite ----------------------------------------------------------
    prom = verdicts[verdicts.promoted].copy()
    prom["strength"] = np.minimum(prom["rho_is"].abs().fillna(0), prom["rho_oos"].abs().fillna(0))
    # binary features have no rho on the same scale; use |q_diff| scaled to be comparable
    prom.loc[prom.kind == "b", "strength"] = np.minimum(prom["q_diff_is"].abs(), prom["q_diff_oos"].abs()) / 100
    prom = prom.sort_values("strength", ascending=False).head(MAX_COMPOSITE)
    feats = {r.feature: int(r.sign) for r in prom.itertuples()}

    # Strict holdout check. Promotion looked at OOS, so the OOS top-K numbers
    # are only a clean holdout if an IS-ONLY rule would have chosen the same
    # features. Rank IS-passing features by IS strength alone and compare.
    isq = verdicts[verdicts.passes_is_only].copy()
    isq["strength"] = isq["rho_is"].abs().fillna(0)
    isq.loc[isq.kind == "b", "strength"] = isq["q_diff_is"].abs() / 100
    is_only = {r.feature: int(r.sign) for r in isq.sort_values("strength", ascending=False)
               .head(MAX_COMPOSITE).itertuples()}
    same = is_only == feats
    print(f"\nIS-ONLY choice would have been {is_only}: {'SAME as composite' if same else 'DIFFERENT'}")
    if len(feats) >= 2:
        sub = d[list(feats)].apply(pd.to_numeric, errors="coerce")
        print("Spearman correlation among composite features (they are not independent):")
        print(sub.corr(method="spearman").round(2).to_string())
    if len(feats) < MIN_COMPOSITE:
        print(f"\nOnly {len(feats)} feature(s) promoted; composite uses that set as-is "
              f"(a composite needs >= {MIN_COMPOSITE} to be called a composite).")
    print(f"\nCOMPOSITE = mean of within-day pct-rank of {feats} (equal weights, no fitting)")
    score = composite_score(d, feats) if feats else pd.Series(np.nan, index=d.index)

    results = {"split": str(SPLIT.date()), "n_trades": int(len(d)), "n_is": int(is_mask.sum()),
               "n_oos": int((~is_mask).sum()), "composite": feats,
               "composite_definition": "equal-weight mean of within-day percentile ranks, sign-adjusted",
               "is_only_choice": is_only, "is_only_choice_matches": bool(same),
               "promotion_rule": "same-sign Spearman IS/OOS, IS |rho|>=0.03 p<0.05, OOS |rho|>=0.5*IS, "
                                 "ordered bins both periods; binary: same-sign diff, IS |t|>=2, OOS>=0.5*IS",
               "all_trades": {lab: selection_metrics(d[m]) for lab, m in (("IS", is_mask), ("OOS", ~is_mask))},
               "verdicts": verdicts.replace({np.nan: None}).to_dict(orient="records"),
               "topk": {}}

    print("\nTOP-K PER DAY: composite vs random (mean of 200 draws) vs all trades")
    hdr = (f"  {'K':>2s} {'per':>3s} {'who':>9s} | {'n':>5s} {'net':>6s} {'win':>4s} {'PF':>5s} "
           f"{'mo up':>5s} {'worst':>6s} {'tr/mo':>5s} {'top10':>5s} {'>8%L':>5s}")
    print(hdr)
    for k in K_LIST:
        if feats:
            sel = top_k_per_day(d, score, k)
        else:
            sel = pd.Series(False, index=d.index)
        rnd = random_baseline(d, k, is_mask)
        rec = {}
        for lab, m in (("IS", is_mask), ("OOS", ~is_mask)):
            comp = selection_metrics(d[sel & m])
            rest = selection_metrics(d[~sel & m])
            rmean = rnd[lab]["mean"]
            draws = np.array(rnd[lab]["net_draws"])
            pct_beat = float((draws < comp["net"]).mean() * 100) if np.isfinite(comp["net"]) else None
            rec[lab] = {"composite": comp, "unselected": rest, "random_mean": rmean,
                        "random_std_net": rnd[lab]["std_net"],
                        "composite_pct_of_random_draws_beaten": pct_beat,
                        "uplift_vs_random_net": (comp["net"] - rmean["net"]) if np.isfinite(comp["net"]) else None,
                        "big_loss_selected": comp["big_loss"], "big_loss_unselected": rest["big_loss"]}
            for who, s in (("composite", comp), ("random", rmean), ("unselect", rest)):
                print(f"  {k:>2d} {lab:>3s} {who:>9s} | {s['n']:5.0f} {_fmt(s['net']):>6s} {_fmt(s['win'], pct=True):>4s} "
                      f"{s['pf']:5.2f} {_fmt(s['months_up'], pct=True):>5s} {_fmt(s['worst_month']):>6s} "
                      f"{s['trades_per_month']:5.1f} {_fmt(s['top10_share'], pct=True):>5s} {s['big_loss']:5.1f}")
            print(f"     {lab} composite net beats {pct_beat if pct_beat is not None else '-'}% of random draws "
                  f"(random sd of net {rnd[lab]['std_net']:.2f}); composite worst month {comp['worst_month_label']}; "
                  f"avg loss sel {comp['avg_loss']:+.2f} vs unsel {rest['avg_loss']:+.2f}; "
                  f"median ATR% sel {comp['median_atr_pct']:.2f} vs unsel {rest['median_atr_pct']:.2f}")
        results["topk"][str(k)] = rec

    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o

    with open(OUT_JSON, "w") as fh:
        json.dump(_clean(results), fh, indent=1)
    print(f"\nwrote {OUT_CSV} and {OUT_JSON} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
