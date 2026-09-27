"""Data-quality gates: the bars themselves.

Everything else in this suite tests logic applied to prices. This file tests
the prices. A screener running quietly on bad data is worse than one that does
not run, and the failure is invisible from above: a moving average computed
across an unadjusted demerger is a number, it is just the wrong number.

What this exists because of: the corporate-action table is built from split and
bonus factors and therefore has no demergers in it. Nineteen unexplained
single-session moves of 25-67% sit in the panel - Vedanta, Tata Motors,
Siemens, Raymond, Aditya Birla Fashion, SKF, Quess - and a backtested trade
held across one of those dates books a loss that never happened.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener import quality


def bars(n=300, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, n))
    d = pd.DataFrame({
        "isin": "INE000A01001",
        "date": pd.bdate_range("2024-01-01", periods=n),
        "adj_close": close,
        "volume": rng.integers(1e5, 1e6, n).astype(float),
    })
    d["adj_open"] = np.r_[close[0], close[:-1]]
    d["adj_high"] = np.maximum(d["adj_open"], d["adj_close"]) * 1.01
    d["adj_low"] = np.minimum(d["adj_open"], d["adj_close"]) * 0.99
    return d


def demerge(d, at=200, factor=0.45):
    """Drop the price 55% on one day, the way an unadjusted spin-off looks."""
    d = d.copy()
    d.loc[d.index >= at, ["adj_open", "adj_high", "adj_low", "adj_close"]] *= factor
    return d


# --- detection --------------------------------------------------------------

def test_clean_history_raises_no_jumps():
    assert quality.detect_price_jumps(bars(), None).empty


def test_an_unadjusted_demerger_is_detected():
    j = quality.detect_price_jumps(demerge(bars()), None)
    assert len(j) == 1
    assert j["move_pct"].iloc[0] < -25
    assert not j["explained"].iloc[0]


def test_a_jump_matched_by_a_corporate_action_is_explained():
    d = demerge(bars())
    at = d["date"].iloc[200]
    actions = pd.DataFrame({"isin": ["INE000A01001"], "ex_date": [at],
                            "action": ["split_or_bonus"], "ratio": [2.0]})
    j = quality.detect_price_jumps(d, actions)
    assert j["explained"].all(), "a known split must not be reported as suspect"


def test_detection_does_not_depend_on_the_direction_of_the_move():
    d = bars()
    d.loc[d.index >= 150, ["adj_open", "adj_high", "adj_low", "adj_close"]] *= 1.6
    assert len(quality.detect_price_jumps(d, None)) == 1


# --- contamination ----------------------------------------------------------

def test_contamination_covers_the_window_after_the_jump():
    d = demerge(bars())
    j = quality.detect_price_jumps(d, None)
    m = quality.contamination_mask(d, j, sessions=250, forward=20)
    assert m.iloc[200:].all(), "every row after an unadjusted jump has bent windows"


def test_contamination_also_covers_the_rows_BEFORE_the_jump():
    """The half that is easy to forget. A hit twenty sessions before a demerger
    has a forward return that walks straight through it - which is how a
    backtest books a fictional 65% loss."""
    d = demerge(bars())
    j = quality.detect_price_jumps(d, None)
    m = quality.contamination_mask(d, j, sessions=250, forward=20)
    assert m.iloc[180:200].all()
    assert not m.iloc[:179].any(), "clean history must not be thrown away"


def test_an_explained_jump_contaminates_nothing():
    d = demerge(bars())
    at = d["date"].iloc[200]
    actions = pd.DataFrame({"isin": ["INE000A01001"], "ex_date": [at],
                            "action": ["split_or_bonus"], "ratio": [2.0]})
    j = quality.detect_price_jumps(d, actions)
    assert not quality.contamination_mask(d, j).any()


# --- structure --------------------------------------------------------------

def test_report_counts_structural_violations():
    d = bars()
    d.loc[5, "adj_high"] = d.loc[5, "adj_low"] - 1        # high below low
    d.loc[9, "adj_close"] = -3.0                          # negative price
    r = quality.report(d, None)
    assert r["ohlc_violations"] >= 1
    assert r["non_positive"] >= 1


def test_report_counts_duplicate_bars():
    d = pd.concat([bars(), bars().iloc[[7]]], ignore_index=True)
    assert quality.report(d, None)["duplicates"] == 1


def test_a_circuit_locked_bar_is_stale_but_not_junk():
    """O=H=L=C with real volume is an upper circuit - a genuine session. The
    same bar with zero volume is a listing where nothing traded."""
    d = bars()
    for c in ("adj_open", "adj_high", "adj_low"):
        d.loc[11, c] = d.loc[11, "adj_close"]
    assert quality.stale_bars(d).iloc[11]
    assert not quality.junk_bars(d).iloc[11]
    d.loc[11, "volume"] = 0
    assert quality.junk_bars(d).iloc[11]


# --- the gate is actually wired in ------------------------------------------

def test_a_contaminated_row_can_never_be_a_hit():
    from screener.rules import Rule, evaluate_rule
    df = pd.DataFrame({
        "isin": ["A", "B"], "symbol": ["A", "B"], "adj_close": [100.0, 100.0],
        "close": [100.0, 100.0], "sma_50": [90.0, 90.0],
        "turnover_median_20d": [1e9, 1e9], "bars_available": [500, 500],
        "contaminated": [False, True],
    })
    rule = Rule(name="t", expr="adj_close > sma_50", min_history=200)
    hits = evaluate_rule(df, rule)
    assert list(hits["isin"]) == ["A"], "the contaminated row must be excluded"
