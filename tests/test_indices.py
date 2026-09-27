"""Index analytics: equal-weight levels, breadth, money flow, RRG."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.indices import (
    breadth, equal_weight_level, money_flow, quadrant, returns_from_level, rrg_point,
)


class TestEqualWeightLevel:
    def test_chains_mean_returns(self):
        """Both names double, so the index doubles - regardless of price level."""
        closes = pd.DataFrame({"a": [100.0, 200.0], "b": [10.0, 20.0]})
        lvl = equal_weight_level(closes)
        assert lvl.iloc[-1] == pytest.approx(200.0)

    def test_price_level_does_not_dominate(self):
        """A 20,000-rupee share must not outweigh a 40-rupee one. Averaging
        prices would give the expensive name ~500x the influence."""
        closes = pd.DataFrame({"big": [20000.0, 20000.0], "small": [40.0, 80.0]})
        lvl = equal_weight_level(closes)
        assert lvl.iloc[-1] == pytest.approx(150.0)   # mean of 0% and +100%

    def test_starts_at_base(self):
        closes = pd.DataFrame({"a": [100.0, 101.0]})
        assert equal_weight_level(closes, base=1000.0).iloc[0] == pytest.approx(1000.0)

    def test_tolerates_a_missing_member(self):
        closes = pd.DataFrame({"a": [100.0, 110.0, 121.0], "b": [np.nan, 50.0, 55.0]})
        lvl = equal_weight_level(closes)
        assert lvl.notna().all()
        assert lvl.iloc[-1] == pytest.approx(121.0)


class TestReturns:
    def test_windows(self):
        lvl = pd.Series(np.linspace(100, 200, 300))
        out = returns_from_level(lvl, {"1D": 1, "1M": 21})
        assert out["1D"] > 0 and out["1M"] > out["1D"]

    def test_short_history_is_nan_not_a_wrong_number(self):
        out = returns_from_level(pd.Series([100.0, 101.0]), {"1Y": 252})
        assert np.isnan(out["1Y"])


class TestBreadth:
    def _feat(self, n=10, above=6):
        return pd.DataFrame({
            "isin": [f"I{i}" for i in range(n)],
            "adj_close": [100.0] * n,
            "sma_20": [90.0] * n, "sma_50": [90.0] * n,
            "sma_200": [90.0] * above + [110.0] * (n - above),
            "ret_1d": [1.0] * above + [-1.0] * (n - above),
            "rsi_14": [55.0] * n,
            "pct_from_52w_high": [-0.5] * 2 + [-20.0] * (n - 2),
            "pct_from_52w_low": [50.0] * n,
        })

    def test_counts_members_above_averages(self):
        b = breadth(self._feat(), [f"I{i}" for i in range(10)])
        assert b["members"] == 10
        assert b["above_200"] == 6 and b["pct_above_200"] == 60.0
        assert b["advancing"] == 6 and b["declining"] == 4

    def test_new_highs(self):
        assert breadth(self._feat(), [f"I{i}" for i in range(10)])["new_highs"] == 2

    def test_empty_index(self):
        assert breadth(self._feat(), []) == {}


class TestMoneyFlow:
    def test_share_and_delta(self):
        """A sector taking a bigger slice of turnover than its own norm gives a
        positive flow delta - the EOD evidence of rotation."""
        idx = pd.date_range("2025-01-01", periods=60, freq="D")
        t = pd.DataFrame({"a": [100.0] * 60, "b": [100.0] * 60}, index=idx)
        t.loc[idx[-5:], "a"] = 300.0          # 'a' surges in the last week
        mf = money_flow(t, ["a"])
        assert mf["share_5d"] > mf["share_60d"]
        assert mf["flow_delta"] > 0

    def test_quiet_sector_has_negative_delta(self):
        idx = pd.date_range("2025-01-01", periods=60, freq="D")
        t = pd.DataFrame({"a": [100.0] * 60, "b": [100.0] * 60}, index=idx)
        t.loc[idx[-5:], "b"] = 400.0
        assert money_flow(t, ["a"])["flow_delta"] < 0

    def test_unknown_symbol_is_ignored(self):
        idx = pd.date_range("2025-01-01", periods=60, freq="D")
        t = pd.DataFrame({"a": [100.0] * 60}, index=idx)
        assert money_flow(t, ["zzz"]) == {}


class TestRRG:
    def _bench(self, n=400):
        return pd.Series(np.linspace(100, 120, n),
                         index=pd.date_range("2024-01-01", periods=n, freq="B"))

    def test_outperformer_lands_right_of_centre(self):
        b = self._bench()
        s = pd.Series(np.linspace(100, 180, len(b)), index=b.index)
        pt = rrg_point(s, b)
        assert pt["rs_ratio"] > 100
        assert pt["quadrant"] in ("leading", "weakening")

    def test_underperformer_lands_left_of_centre(self):
        b = self._bench()
        s = pd.Series(np.linspace(100, 80, len(b)), index=b.index)
        pt = rrg_point(s, b)
        assert pt["rs_ratio"] < 100
        assert pt["quadrant"] in ("lagging", "improving")

    def test_trail_is_returned_for_the_rotation_path(self):
        b = self._bench()
        s = pd.Series(np.linspace(100, 160, len(b)), index=b.index)
        pt = rrg_point(s, b)
        assert len(pt["trail"]) > 1
        assert pt["trail"][-1] == [pt["rs_ratio"], pt["rs_mom"]]

    def test_short_history_returns_nothing_rather_than_noise(self):
        b = self._bench(n=30)
        s = pd.Series(np.linspace(100, 110, 30), index=b.index)
        assert rrg_point(s, b) == {}

    def test_sampled_weekly_not_daily(self):
        """Weekly is the convention; daily sampling makes the tails unreadable."""
        b = self._bench()
        s = pd.Series(np.linspace(100, 160, len(b)), index=b.index)
        weekly = rrg_point(s, b)
        daily = rrg_point(s, b, freq="", n=60, m=15)
        assert weekly and daily
        # The weekly tail spans 8 weeks of data, not 8 sessions.
        assert len(weekly["trail"]) <= 8

    def test_quadrant_boundaries(self):
        assert quadrant(100, 100) == "leading"
        assert quadrant(100.1, 99.9) == "weakening"
        assert quadrant(99.9, 100.1) == "improving"
        assert quadrant(99, 99) == "lagging"

    def test_identical_series_sits_at_the_origin(self):
        """An index compared with itself has no relative strength to show."""
        b = self._bench()
        pt = rrg_point(b.copy(), b)
        assert pt == {} or (abs(pt["rs_ratio"] - 100) < 1e-6)


class TestMoneyFlowWindows:
    def _t(self, n=80):
        idx = pd.date_range("2025-01-01", periods=n, freq="B")
        return pd.DataFrame({"a": [100.0] * n, "b": [100.0] * n}, index=idx), idx

    def test_wow_detects_a_fresh_surge(self):
        """Share rising this week versus last week is a positive WoW."""
        t, idx = self._t()
        t.loc[idx[-5:], "a"] = 400.0
        mf = money_flow(t, ["a"])
        assert mf["wow_pp"] > 0

    def test_mom_and_wow_can_disagree(self):
        """Money that arrived during the month but has since left: MoM up while
        WoW is already negative. The two windows must not be redundant."""
        t, idx = self._t()
        t.loc[idx[-21:-6], "a"] = 400.0
        mf = money_flow(t, ["a"])
        assert mf["mom_pp"] > 0, mf
        assert mf["wow_pp"] < 0, mf

    def test_turnover_growth_is_reported_in_rupees_too(self):
        """Share can fall while rupees rise if the whole market grew."""
        t, idx = self._t()
        t.loc[idx[-5:], "a"] = 200.0
        t.loc[idx[-5:], "b"] = 900.0
        mf = money_flow(t, ["a"])
        assert mf["turnover_wow"] > 0      # 'a' traded more rupees
        assert mf["wow_pp"] < 0            # but lost share of the market

    def test_short_history_gives_none_not_a_wrong_number(self):
        idx = pd.date_range("2025-01-01", periods=4, freq="B")
        t = pd.DataFrame({"a": [100.0] * 4, "b": [100.0] * 4}, index=idx)
        mf = money_flow(t, ["a"])
        assert mf["mom_pp"] is None or mf["mom_pp"] == 0

    def test_dod_isolates_a_single_session(self):
        """Day-over-day must react to one session, where WoW averages it away."""
        t, idx = self._t()
        t.loc[idx[-1], "a"] = 900.0
        mf = money_flow(t, ["a"])
        assert mf["dod_pp"] > mf["wow_pp"]

    def test_dod_is_flat_when_nothing_changed(self):
        t, _ = self._t()
        assert money_flow(t, ["a"])["dod_pp"] == 0
