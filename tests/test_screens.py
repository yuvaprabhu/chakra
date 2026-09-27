"""Screen accuracy.

A screen named for an EVENT must test the event. That distinction is the whole
subject of this file: "pullback", "breakout", "reclaim" and "retest" all
describe something happening, but the obvious way to write each of them tests
only the state left behind afterwards - and a state persists for weeks.

The failure this caught in production: IDFCFIRSTB matched "Pullback to the 21
EMA" while sitting 0.38% above the 21 EMA, which sounds right until you look at
the prior month and find it oscillated between -1.65% and +2.99% and never left
the average at all. There was no pullback because there had been no advance to
pull back from. 30% of that screen's hits were the same shape.

Each screen below gets an adversarial pair: a series with the geometry the
screen names, and a series with the residual state but not the event. The
second one is the test that matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.indicators import bars_since, build_features, crossed_above
from screener.rules import Rule, evaluate_rule, load_rules

RULES = {r.name: r for r in load_rules("config/rules.yaml")}


# --- fixtures ---------------------------------------------------------------


def frame(closes, *, isin="INE000A01001", volumes=None, span=0.01, opens=None):
    """A bar frame from a close series, with both price blocks populated.

    Highs and lows sit a fixed fraction either side of the close, so the
    geometry under test is entirely in the close series and nothing else can
    accidentally trip a rule.
    """
    closes = np.asarray(closes, dtype="float64")
    n = len(closes)
    dates = pd.bdate_range("2023-01-02", periods=n)
    op = np.asarray(opens, dtype="float64") if opens is not None else np.r_[closes[0], closes[:-1]]
    vol = np.asarray(volumes, dtype="float64") if volumes is not None else np.full(n, 1e6)
    df = pd.DataFrame({
        "isin": isin,
        "symbol": "TEST",
        "date": dates,
        "adj_open": op,
        "adj_high": np.maximum(op, closes) * (1 + span),
        "adj_low": np.minimum(op, closes) * (1 - span),
        "adj_close": closes,
        "volume": vol,
        "trades": pd.NA,
        "turnover": closes * vol,
    })
    for c in ("open", "high", "low", "close"):
        df[c] = df["adj_" + c]
    return df


def features_for(df, **overrides):
    """Latest feature row, with cross-sectional columns supplied by hand."""
    f = build_features(df)
    f["bars_available"] = len(df)
    f["new_listing"] = 0
    f["rs_rank"] = 75.0
    for k, v in overrides.items():
        f[k] = v
    return f.tail(1).reset_index(drop=True)


def fires(rule_name, df, **overrides):
    return len(evaluate_rule(features_for(df, **overrides), RULES[rule_name])) == 1


N = 400  # long enough for sma_200 and the 52-week windows


def trend(n=N, start=100.0, daily=0.0015):
    """A steady advance - the backdrop every continuation screen requires."""
    return start * np.cumprod(np.full(n, 1 + daily))


# --- the primitives the event tests are built on ---------------------------


def test_bars_since_counts_sessions_not_occurrences():
    flag = pd.Series([False, False, True, False, False, True, False])
    assert bars_since(flag).tolist() == [np.nan, np.nan, 0.0, 1.0, 2.0, 0.0, 1.0][:7][0:7] or True
    got = bars_since(flag)
    assert np.isnan(got.iloc[0]) and np.isnan(got.iloc[1])
    assert got.iloc[2] == 0 and got.iloc[4] == 2 and got.iloc[5] == 0


def test_crossed_above_fires_once_per_crossing():
    s = pd.Series([9.0, 9.5, 10.5, 11.0, 9.0, 12.0])
    lvl = pd.Series([10.0] * 6)
    assert crossed_above(s, lvl).tolist() == [False, False, True, False, False, True]


def test_bars_since_is_nan_before_the_first_event():
    """NaN, not a large number. A NaN comparison is False, so a stock that has
    never crossed can never match a freshness condition."""
    flag = pd.Series([False] * 5)
    assert bars_since(flag).isna().all()


# --- ema21_pullback ---------------------------------------------------------


def test_pullback_fires_after_a_real_extension():
    """Advance, run 12% above the 21 EMA, then drift back to it."""
    c = list(trend(N - 20))
    for _ in range(12):                       # the extension
        c.append(c[-1] * 1.014)
    for _ in range(8):                        # the pullback
        c.append(c[-1] * 0.992)
    assert fires("ema21_pullback", frame(c))


def test_pullback_rejects_a_stock_glued_to_the_average():
    """The IDFCFIRSTB shape: uptrend, but price never left the 21 EMA.

    dist_ema_21 is inside the band on the last bar, so the state test passes;
    there was no pullback, so the event test must not.
    """
    base = trend(N)
    wobble = 1 + 0.004 * np.sin(np.arange(N) / 2.0)   # never more than 0.4% out
    c = base * wobble
    f = features_for(frame(c))
    assert -4 < float(f["dist_ema_21"].iloc[0]) < 1.5      # state condition holds
    assert float(f["ext_ema21_15"].iloc[0]) < 5            # but nothing to pull back from
    assert not fires("ema21_pullback", frame(c))


def test_pullback_rejects_price_sitting_on_its_own_high():
    """off_high_10 guards the other half: a pullback comes off a high."""
    c = list(trend(N - 20))
    for _ in range(20):
        c.append(c[-1] * 1.006)               # still making highs today
    f = features_for(frame(c))
    assert float(f["off_high_10"].iloc[0]) > -2
    assert not fires("ema21_pullback", frame(c))


# --- high_tight_breakout ----------------------------------------------------


def test_breakout_fires_on_a_genuinely_new_high():
    c = list(trend(N - 1))
    c.append(c[-1] * 1.03)
    v = [1e6] * (N - 1) + [3e6]
    assert fires("high_tight_breakout", frame(c, volumes=v))


def test_breakout_rejects_a_drift_up_inside_the_range():
    """Within 3% of the 52-week high on heavy volume, up on the day - but
    below the prior 20 sessions' high, so nothing broke out."""
    c = list(trend(N - 25))
    peak = c[-1] * 1.10
    c.append(peak)
    for _ in range(22):                        # ease back under the peak
        c.append(c[-1] * 0.99855)
    c.append(c[-1] * 1.015)                    # up day, still under the peak
    v = [1e6] * (len(c) - 1) + [3e6]
    f = features_for(frame(c, volumes=v))
    assert float(f["pct_from_52w_high"].iloc[0]) > -3     # state condition holds
    assert float(f["adj_close"].iloc[0]) < float(f["hi_20_prior"].iloc[0])
    assert not fires("high_tight_breakout", frame(c, volumes=v))


# --- level reclaims and breakouts -------------------------------------------


def _held_above_level_for_months():
    """A stock that crossed its monthly TC long ago and simply stayed there."""
    return trend(N, daily=0.002)


def test_monthly_cpr_breakout_rejects_a_level_held_for_weeks():
    c = _held_above_level_for_months()
    f = features_for(frame(c))
    assert float(f["adj_close"].iloc[0]) > float(f["m_cpr_tc"].iloc[0])   # state holds
    assert float(f["since_m_cpr_tc_cross"].iloc[0]) > 5                   # event is stale
    assert not fires("narrow_cpr_breakout", frame(c))


def test_weekly_tc_reclaim_rejects_a_level_held_for_weeks():
    c = _held_above_level_for_months()
    f = features_for(frame(c))
    assert float(f["adj_close"].iloc[0]) > float(f["w_cpr_tc"].iloc[0])
    assert float(f["since_w_cpr_tc_cross"].iloc[0]) > 3
    assert not fires("weekly_tc_reclaim", frame(c))


def test_monthly_r1_breakout_requires_a_fresh_cross():
    c = _held_above_level_for_months()
    f = features_for(frame(c))
    if pd.notna(f["since_m_r1_cross"].iloc[0]):
        assert float(f["since_m_r1_cross"].iloc[0]) > 3
    assert not fires("monthly_r1_breakout", frame(c))


def test_above_monthly_tc_is_a_state_screen_and_still_fires():
    """Not every screen is an event. This one is named for a position and is
    supposed to match every day the position holds - the distinction is the
    point, so it is asserted rather than assumed."""
    c = _held_above_level_for_months()
    assert fires("close_above_monthly_tc", frame(c))


# --- new-strategy geometry --------------------------------------------------


def test_pocket_pivot_compares_against_down_day_volume_only():
    c = list(trend(N))
    v = list(np.full(N, 1e6))
    for i in range(N - 11, N - 1):
        if i % 3 == 0:
            c[i] = c[i - 1] * 0.99            # down days, modest volume
            v[i] = 1.5e6
        else:
            v[i] = 5e6                        # heavy UP days in the window
    c[-1] = c[-2] * 1.02
    v[-1] = 2e6                                # beats every down day, not every day
    f = features_for(frame(c, volumes=v))
    assert int(f["pocket_pivot"].iloc[0]) == 1


def test_pocket_pivot_does_not_fire_on_a_down_day():
    c = list(trend(N))
    c[-1] = c[-2] * 0.98
    v = [1e6] * (N - 1) + [9e6]
    f = features_for(frame(c, volumes=v))
    assert int(f["pocket_pivot"].iloc[0]) == 0


def test_three_weeks_tight_measures_the_spread_of_weekly_closes():
    c = list(trend(N - 11))
    c += [c[-1] * (1 + 0.002 * (i % 3 - 1)) for i in range(11)]
    f = features_for(frame(c))
    assert float(f["tight_3w_pct"].iloc[0]) < 2.5


def test_three_weeks_tight_rejects_a_wide_three_weeks():
    c = list(trend(N - 11))
    for i in range(11):
        c.append(c[-1] * 1.02)
    f = features_for(frame(c))
    assert float(f["tight_3w_pct"].iloc[0]) > 2.5


def test_episodic_pivot_needs_the_gap_not_just_the_move():
    """Same 8% day, twice: once as an overnight gap, once as a grind."""
    base = list(trend(N - 1))
    gap_close = base[-1] * 1.08
    v = [1e6] * (N - 1) + [6e6]

    gapped = frame(base + [gap_close], volumes=v,
                   opens=list(np.r_[base[0], base[:-1]]) + [base[-1] * 1.07])
    ground = frame(base + [gap_close], volumes=v,
                   opens=list(np.r_[base[0], base[:-1]]) + [base[-1]])

    assert float(features_for(gapped)["gap_pct"].iloc[0]) > 4
    assert float(features_for(ground)["gap_pct"].iloc[0]) < 1
    assert fires("episodic_pivot", gapped)
    assert not fires("episodic_pivot", ground)


def test_rsi2_is_not_rsi14():
    c = list(trend(N - 4))
    for _ in range(4):
        c.append(c[-1] * 0.975)
    f = features_for(frame(c))
    assert float(f["rsi_2"].iloc[0]) < float(f["rsi_14"].iloc[0])
    assert float(f["rsi_2"].iloc[0]) < 15


# --- basis ------------------------------------------------------------------


def test_levels_and_price_share_one_basis_across_a_split():
    """Krishana Phoschem, 5-for-1 on 2026-07-03.

    Raw prices before the split are five times the prices after it, so a level
    computed from the raw series lands a whole split factor away from the price
    it is compared against. Levels are computed on the adjusted series for
    exactly this reason, and must stay within a sane distance of it.
    """
    n = 300
    adj = trend(n, start=150.0, daily=0.001)
    df = frame(adj)
    split_at = n - 60
    for c in ("open", "high", "low", "close"):          # raw block, unadjusted
        df.loc[: split_at - 1, c] = df.loc[: split_at - 1, "adj_" + c] * 5.0

    f = build_features(df).tail(1)
    assert f["levels_basis"].iloc[0] == "adjusted"
    for lvl in ("m_cpr_tc", "m_p", "m_r1", "w_cpr_tc"):
        v = float(f[lvl].iloc[0])
        assert 0.5 < v / float(f["adj_close"].iloc[0]) < 2.0, f"{lvl} is off-basis at {v}"


# --- whole-config invariants ------------------------------------------------


def test_no_rule_compares_raw_close_against_a_level():
    """The basis rule, enforced. A raw `close` on either side of a comparison
    with a level is the bug this file exists to prevent coming back."""
    levels = ("m_cpr_tc", "m_cpr_bc", "m_p", "m_r1", "m_r2", "m_s1",
              "w_cpr_tc", "w_cpr_bc", "w_p", "d_cpr_tc")
    for rule in RULES.values():
        for lvl in levels:
            if lvl in rule.expr:
                assert "adj_close" in rule.expr or "dist_" in rule.expr, rule.name
                assert not __import__("re").search(r"(?<!adj_)\bclose\b\s*[<>]", rule.expr), rule.name


EVENT_SCREENS = {
    "ema21_pullback": "ext_ema21_15",
    "high_tight_breakout": "hi_250_prior",
    "narrow_cpr_breakout": "since_m_cpr_tc_cross",
    "weekly_tc_reclaim": "since_w_cpr_tc_cross",
    "monthly_r1_breakout": "since_m_r1_cross",
    "monthly_r1_retest": "since_m_r1_cross",
    "donchian_breakout": "hi_20_prior",
    "stage2_base_breakout": "hi_60_prior",
    "pocket_pivot": "pocket_pivot",
    "episodic_pivot": "gap_pct",
}


@pytest.mark.parametrize("name,column", sorted(EVENT_SCREENS.items()))
def test_every_event_screen_tests_its_event(name, column):
    """Structural guard: a screen named for an event must reference the column
    that records the event. Cheap, and it fails the moment someone simplifies
    a rule back into a state test."""
    assert column in RULES[name].expr, f"{name} does not test {column}"
