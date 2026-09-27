"""Definition tests: does each rule mean what its name and source mean?

The distinction from test_screens.py is the whole point of this file. That one
checks that a rule's expression is implemented correctly. This one checks that
the expression is the RIGHT EXPRESSION - that a screen named after a published
strategy encodes what that strategy actually says.

Every defect that reached the user came through this gap, not the other one:

* "Pullback to the 21 EMA" matched stocks that had never left the 21 EMA.
* "Volatility Contraction" matched a stock in a flat 8-11% range for three
  months, because the rule compared two windows instead of testing a sequence.
* "Breakout at 52-Week High" matched stocks that were not at a 52-week high,
  because the rule tested a 20-day high and a distance-from-high band.
* "20-Day Donchian Breakout" measured the channel on CLOSES; a Donchian channel
  is drawn on HIGHS, and half the hits were below the real channel top.
* "Stage 2 Base Breakout" used the 200 DMA; Weinstein's stage call is the
  30-week (150 DMA) average.
* "Three Weeks Tight" sampled every fifth session instead of weekly closes.
* "Momentum Leaders" cited the academic momentum literature while using a
  window that INCLUDES the most recent month, where short-term reversal lives.

An implementation test cannot catch any of those, because in each case the code
did exactly what the rule said. These assertions are written from the source
definitions instead.
"""

from __future__ import annotations

import re

import pytest

from screener.rules import load_rules

RULES = {r.name: r for r in load_rules("config/rules.yaml")}


def expr(name: str) -> str:
    return " ".join(RULES[name].expr.split())


# --- each requirement is a clause the canonical definition demands -----------

REQUIRED = {
    # Minervini's template is eight points; point 4 is 50 > 150 > 200.
    "minervini_trend_template": ["sma_50 > sma_150", "sma_150 > sma_200", "sma_200_slope > 0"],
    # A VCP is a SEQUENCE of contractions, each tighter than the last.
    "vcp_squeeze": ["contract_seq == 1", "contract_ratio", "vol_dryup"],
    # A pullback needs a prior extension to pull back from.
    "ema21_pullback": ["ext_ema21_15", "off_high_10", "dist_ema_21"],
    # A 52-week-high breakout must make a 52-week high, on prior HIGHS.
    "high_tight_breakout": ["hi_250_prior"],
    # Donchian's channel is the prior 20 sessions' highs.
    "donchian_breakout": ["hi_20_prior"],
    # Weinstein's stage call is the 30-week average, rising.
    "stage2_base_breakout": ["sma_150", "sma_150_slope > 0", "hi_60_prior"],
    # Academic momentum skips the most recent month.
    "momentum_leaders": ["mom_12_1"],
    # Connors' signal is a 2-period RSI, not a 14-period one.
    "rsi2_reversion": ["rsi_2"],
    # A pocket pivot is measured against DOWN-day volume, inside a base.
    "pocket_pivot": ["pocket_pivot == 1", "dist_sma_10"],
    # An episodic pivot is defined by the gap, not by the day's return.
    "episodic_pivot": ["gap_pct"],
    # A reclaim and a breakout are crossings, not positions.
    "narrow_cpr_breakout": ["since_m_cpr_tc_cross", "m_cpr_width_rank"],
    "weekly_tc_reclaim": ["since_w_cpr_tc_cross"],
    "monthly_r1_breakout": ["since_m_r1_cross"],
    "monthly_r1_retest": ["since_m_r1_cross", "low3_dist_m_r1"],
    # Three weeks tight is three WEEKLY closes, tightly.
    "three_weeks_tight": ["tight_3w_pct"],
    # The breakout and its volume must be the SAME session, which is what the
    # confirmed-cross column encodes - and the entry is a retest of one of two
    # levels, so both must appear.
    "r1_breakout_retest": ["since_r1_vol_breakout", "low3_dist_m_r1", "low3_dist_ema_9"],
    "leader_dip": ["rs_rank >= 80", "adj_close > sma_200", "new_lo7 == 1", "rsi_2 < 10"],
    # The playbook is the same dip plus the one filter that survived forensics:
    # the close still holds above the 21 EMA.
    "leader_dip_playbook": ["rs_rank >= 80", "adj_close > sma_200", "new_lo7 == 1", "rsi_2 < 10", "dist_ema_21 >= 2.6"],
    # The swing-lab entries are EVENTS (fresh flags) gated by market state.
    "trend_stack_swing": ["fresh_trend_stack == 1", "mkt_vix_pctile_1y >=", "mkt_nifty_range_20d >="],
    "momentum_swing": ["fresh_momentum_leaders == 1", "pct_from_52w_high >=", "mkt_vix_pctile_1y >=", "mkt_gate_on == 1"],
    # The groomed pullback swings: the base rule plus the lab's stock filters and the gate.
    "sw_leader_dip_concentrated": ["rs_rank >= 80", "adj_close > sma_200", "new_lo7 == 1", "rsi_2 < 10", "dist_ema_21 >= 2.6", "mkt_gate_on == 1"],
    "sw_rsi2_snapback": ["rsi_2 < 8", "adj_close > sma_200", "dist_sma_50 >= 8.7", "mkt_gate_on == 1"],
    "sw_double_seven": ["new_lo7 == 1", "adj_close > sma_200", "dist_ema_21 >= 1.0", "atr_pct >= 5.3", "mkt_gate_on == 1"],
    "sw_monthly_r1_retest": ["since_m_r1_cross >= 2", "low3_dist_m_r1", "dist_sma_200 >= 33", "atr_pct >= 4.3", "mkt_gate_on == 1"],
    "vl_volume_spike_breakout": ["vol_ratio >= 5", "adj_close > close_hi_20", "close_pos >= 0.7", "rs_rank >= 60", "adj_close > sma_200", "ema_21 > ema_50", "mkt_gate_on == 1"],
    "vl_long_base_breakout": ["adj_close > close_hi_500", "vol_ratio >= 2", "adj_close > sma_200", "mkt_gate_on == 1"],
    "vl_wyckoff_spring": ["spring == 1", "adj_close > sma_200", "vol_ratio >= 2", "mkt_gate_on == 1"],
    "vl_pocket_pivot": ["pocket_pivot == 1", "adj_close > sma_50", "adj_close > sma_200", "dist_ema_21 <= 15", "mkt_gate_on == 1"],
    # The shakeout must have HAPPENED and been reclaimed, and the stop must sit
    # outside the noise - a stop inside 1.5 ATR is taken by accident.
    "spring_reclaim": ["since_spring", "lo_20_prior", "atr_to_spring"],
    # "Held" is the whole claim: the lowest low since the break is still above
    # the level, and it has closed above it repeatedly.
    "held_breakout": ["brk_cushion", "closes_above_brk", "since_brk_60", "brk_vol"],
    # Asymmetry over twenty sessions, not one spike.
    "accumulation_thrust": ["ud_vol_20", "close_pos_3"],
}

FORBIDDEN = {
    # since_m_r1_cross is the plain cross; using it here would drop the volume
    # confirmation that is half the setup's name.
    "r1_breakout_retest": ["since_m_r1_cross >"],
    # vol_ratio is a single day's spike - the thing these screens exist to not
    # be fooled by. Sustained asymmetry is ud_vol_20.
    "accumulation_thrust": ["vol_ratio"],
    # These are the exact substitutions that produced shipped defects. Each one
    # is a plausible-looking clause that means something other than the name.
    "high_tight_breakout": ["hi_20_prior", "pct_from_52w_high"],
    "stage2_base_breakout": ["sma_200_slope"],
    "vcp_squeeze": ["range_ratio", "atr_ratio_60"],
    "momentum_leaders": ["ret_20d"],
}


@pytest.mark.parametrize("name,clauses", sorted(REQUIRED.items()))
def test_rule_encodes_its_canonical_definition(name, clauses):
    e = expr(name)
    for c in clauses:
        assert c in e, f"{name} does not encode {c!r}: {e}"


@pytest.mark.parametrize("name,clauses", sorted(FORBIDDEN.items()))
def test_rule_does_not_use_the_near_miss_it_used_to(name, clauses):
    e = expr(name)
    for c in clauses:
        assert c not in e, (
            f"{name} uses {c!r}, which is the near-miss this screen shipped with "
            f"and is not what its source defines: {e}")


def test_every_screen_names_a_source():
    """A screen claiming a published strategy has to say which one, so the
    definition can be checked against something other than my own opinion."""
    for r in RULES.values():
        assert r.origin.strip(), f"{r.name} cites no source"
        assert r.why.strip(), f"{r.name} does not say what it is trying to capture"


def test_thresholds_are_stated_in_the_description():
    """A number in the rule that is not in the description is a number nobody
    can audit. Every screen's prose has to name what it actually tests."""
    missing = []
    for r in RULES.values():
        nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", r.expr))
        # Bare comparisons against 0 and 1 are structural, not thresholds.
        nums -= {"0", "1"}
        if nums and not r.description.strip():
            missing.append(r.name)
    assert not missing, f"screens with thresholds but no description: {missing}"
