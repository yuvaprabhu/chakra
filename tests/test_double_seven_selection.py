"""Pure-function tests for scripts/double_seven_selection.py.

Nothing here reads the feature panel or the trade list; every input is a
small synthetic frame built in the test.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "double_seven_selection.py"
_spec = importlib.util.spec_from_file_location("double_seven_selection", _PATH)
sel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sel)


def _trades(n_days=6, per_day=4, seed=3):
    """Non-overlapping synthetic trades: distinct isins each day, 2-bar holds."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-06-20", periods=n_days)
    rows = []
    for i, day in enumerate(days):
        for j in range(per_day):
            rows.append({"isin": f"INE{i:03d}{j:02d}", "date": day,
                         "exit_date": day + pd.tseries.offsets.BDay(2),
                         "net": float(rng.normal(0.5, 3)), "f1": float(rng.random()),
                         "f2": float(rng.random())})
    d = pd.DataFrame(rows)
    d["mo"] = d["date"].dt.to_period("M")
    return d


def test_bin_by_is_uses_is_edges_only():
    x = pd.Series(np.concatenate([np.arange(100.0), 1000 + np.arange(100.0)]))
    is_mask = pd.Series([True] * 100 + [False] * 100)
    bins = sel.bin_by_is(x, is_mask, n_bins=5)
    # IS rows are split into five equal bins by their own distribution.
    assert bins[is_mask].value_counts().sort_index().tolist() == [20] * 5
    # OOS rows all sit above the IS range, so every one lands in the top bin.
    assert (bins[~is_mask] == 5).all()
    # Edges must not move when OOS values change: same IS -> same IS bins.
    x2 = x.copy(); x2[~is_mask] = -5000.0
    bins2 = sel.bin_by_is(x2, is_mask, n_bins=5)
    assert bins2[is_mask].tolist() == bins[is_mask].tolist()
    assert (bins2[~is_mask] == 1).all()


def test_composite_score_is_rank_based_and_sign_aware():
    d = _trades()
    base = sel.composite_score(d, {"f1": +1, "f2": -1})
    # A monotone transform of a feature must not change the score at all.
    t = d.copy(); t["f1"] = np.exp(5 * t["f1"]); t["f2"] = t["f2"] ** 3
    assert np.allclose(sel.composite_score(t, {"f1": +1, "f2": -1}), base)
    # Scores are within-day percentile ranks: bounded, and per day the best
    # f1 (single feature, +1) gets the top score.
    assert base.between(0, 1).all()
    s1 = sel.composite_score(d, {"f1": +1})
    for _, g in d.assign(s=s1).groupby("date"):
        assert g.loc[g["s"].idxmax(), "f1"] == g["f1"].max()
    # Flipping the sign reverses the within-day ordering.
    s1m = sel.composite_score(d, {"f1": -1})
    for _, g in d.assign(a=s1, b=s1m).groupby("date"):
        assert g["a"].rank().tolist() == g["b"].rank(ascending=False).tolist()


def test_top_k_per_day_caps_k_and_one_position_per_name():
    d = _trades(n_days=3, per_day=4)
    score = pd.Series(np.arange(len(d), dtype=float), index=d.index)
    sel_mask = sel.top_k_per_day(d, score, k=2)
    assert d[sel_mask].groupby("date").size().tolist() == [2, 2, 2]
    # Within a day the two highest scores are taken.
    for _, g in d.assign(s=score, sel=sel_mask).groupby("date"):
        assert set(g[g.sel]["s"]) == set(g["s"].nlargest(2))

    # Overlap: the same name signals again while its first trade is open.
    days = pd.bdate_range("2025-06-20", periods=2)
    o = pd.DataFrame({
        "isin":      ["A", "B", "A", "C"],
        "date":      [days[0], days[0], days[1], days[1]],
        "exit_date": [days[1] + pd.tseries.offsets.BDay(3), days[0] + pd.tseries.offsets.BDay(1),
                      days[1] + pd.tseries.offsets.BDay(3), days[1] + pd.tseries.offsets.BDay(1)],
        "net": [1.0, 1.0, 1.0, 1.0],
    })
    m = sel.top_k_per_day(o, pd.Series([1.0, 0.5, 9.0, 0.1]), k=1)
    # Day 1: A (score 1.0). Day 2: A is still open (exit after day 2), so C
    # is taken despite A's higher score.
    assert m.tolist() == [True, False, False, True]


def test_random_baseline_shape_and_counts():
    d = _trades(n_days=8, per_day=5)
    is_mask = d["date"] < d["date"].iloc[len(d) // 2]
    out = sel.random_baseline(d, k=2, is_mask=is_mask, n_draws=7, seed=1)
    assert set(out) == {"IS", "OOS"}
    for lab, m in (("IS", is_mask), ("OOS", ~is_mask)):
        assert len(out[lab]["net_draws"]) == 7
        # Every draw takes exactly K per day, so n is fixed across draws.
        assert out[lab]["mean"]["n"] == 2 * d.loc[m, "date"].nunique()
        assert np.isfinite(out[lab]["std_net"])
    # Two different seeds give different draws; the same seed reproduces.
    again = sel.random_baseline(d, k=2, is_mask=is_mask, n_draws=7, seed=1)
    assert again["IS"]["net_draws"] == out["IS"]["net_draws"]
    other = sel.random_baseline(d, k=2, is_mask=is_mask, n_draws=7, seed=2)
    assert other["IS"]["net_draws"] != out["IS"]["net_draws"]


def test_selection_metrics_known_values():
    w = pd.DataFrame({
        "isin": list("aabbc"), "net": [4.0, -2.0, 6.0, -10.0, 2.0],
        "mo": pd.PeriodIndex(["2025-01", "2025-01", "2025-02", "2025-02", "2025-03"], freq="M"),
    })
    s = sel.selection_metrics(w)
    assert s["n"] == 5 and s["win"] == pytest.approx(60.0)
    assert s["pf"] == pytest.approx(12.0 / 12.0)
    assert s["big_loss"] == pytest.approx(20.0)            # one trade below -8%
    assert s["months_up"] == pytest.approx(200 / 3)         # Jan +1, Feb -2, Mar +2
    assert s["worst_month"] == pytest.approx(-2.0)
    assert s["trades_per_month"] == pytest.approx(5 / 3)
