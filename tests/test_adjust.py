"""Deriving raw prices from an adjusted series plus a corporate-action table.

CLAUDE.md asks for synthetic bars with a known split, asserting the adjustment
factors come out right. That is the first class below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.adjust import cumulative_factors, derive_raw, derive_raw_all, verify


def actions(*rows):
    return pd.DataFrame([{"isin": i, "ex_date": pd.Timestamp(d), "action": "split_or_bonus",
                          "ratio": r, "purpose": ""} for i, d, r in rows])


def dates(n=10, start="2024-01-01"):
    return pd.DatetimeIndex(pd.bdate_range(start, periods=n))


class TestCumulativeFactors:
    def test_no_actions_is_all_ones(self):
        f = cumulative_factors(pd.DataFrame(), dates())
        assert (f == 1.0).all()

    def test_factor_applies_only_before_the_ex_date(self):
        """A 2:1 split on the 5th bar: earlier bars unwind by 2, the ex-date bar
        is already post-split and must not."""
        d = dates(10)
        f = cumulative_factors(actions(("I1", d[4], 2.0)), d)
        assert (f[:4] == 2.0).all()
        assert (f[4:] == 1.0).all()

    def test_two_actions_compound(self):
        d = dates(12)
        f = cumulative_factors(actions(("I1", d[3], 2.0), ("I1", d[8], 5.0)), d)
        assert f.iloc[0] == pytest.approx(10.0)   # both still ahead
        assert f.iloc[5] == pytest.approx(5.0)    # only the 5:1 ahead
        assert f.iloc[9] == pytest.approx(1.0)

    def test_fractional_bonus(self):
        """A 1:2 bonus is a 1.5 factor, not 2."""
        d = dates(6)
        f = cumulative_factors(actions(("I1", d[3], 1.5)), d)
        assert f.iloc[0] == pytest.approx(1.5)

    def test_zero_and_negative_ratios_are_ignored(self):
        d = dates(6)
        f = cumulative_factors(actions(("I1", d[2], 0.0), ("I1", d[4], -1.0)), d)
        assert (f == 1.0).all()


class TestDeriveRaw:
    def _bars(self, n=10, price=100.0):
        d = dates(n)
        return pd.DataFrame({
            "isin": "I1", "date": d,
            "adj_open": price, "adj_high": price * 1.02,
            "adj_low": price * 0.98, "adj_close": price,
            "open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan,
        })

    def test_synthetic_split_reconstructs_raw(self):
        """The assertion CLAUDE.md names: synthetic bars with a known split, and
        the factors must come out right. Pre-split raw is double the adjusted."""
        b = self._bars(10)
        out = derive_raw(b, actions(("I1", b["date"].iloc[5], 2.0)))
        assert out["close"].iloc[0] == pytest.approx(200.0)
        assert out["close"].iloc[4] == pytest.approx(200.0)
        assert out["close"].iloc[5] == pytest.approx(100.0)
        assert out["adj_factor"].iloc[0] == 2.0

    def test_all_four_price_columns_are_filled(self):
        b = self._bars(6)
        out = derive_raw(b, actions(("I1", b["date"].iloc[3], 2.0)))
        for raw, adj in (("open", "adj_open"), ("high", "adj_high"),
                         ("low", "adj_low"), ("close", "adj_close")):
            assert out[raw].iloc[0] == pytest.approx(out[adj].iloc[0] * 2)

    def test_observed_raw_is_never_overwritten(self):
        """A bhavcopy price is an observation; a derived one is arithmetic. The
        observation wins."""
        b = self._bars(6)
        b.loc[0, "close"] = 12345.0
        out = derive_raw(b, actions(("I1", b["date"].iloc[3], 2.0)))
        assert out["close"].iloc[0] == 12345.0
        assert out["close"].iloc[1] == pytest.approx(200.0)

    def test_no_actions_means_raw_equals_adjusted(self):
        b = self._bars(6)
        out = derive_raw(b, pd.DataFrame())
        assert (out["close"] == out["adj_close"]).all()

    def test_ohlc_ordering_survives(self):
        b = self._bars(8)
        out = derive_raw(b, actions(("I1", b["date"].iloc[4], 3.0)))
        assert (out["high"] >= out["low"]).all()
        assert (out["high"] >= out[["open", "close"]].max(axis=1)).all()

    def test_vectorised_across_symbols(self):
        a = self._bars(6).assign(isin="I1")
        c = self._bars(6, price=50.0).assign(isin="I2")
        out = derive_raw_all(pd.concat([a, c], ignore_index=True),
                             actions(("I1", a["date"].iloc[3], 2.0)))
        i1 = out[out["isin"] == "I1"].sort_values("date")
        i2 = out[out["isin"] == "I2"].sort_values("date")
        assert i1["close"].iloc[0] == pytest.approx(200.0)
        assert i2["close"].iloc[0] == pytest.approx(50.0)   # untouched


class TestVerifyAgainstObserved:
    def test_agreement_is_detected(self):
        d = dates(6)
        b = pd.DataFrame({"isin": "I1", "symbol": "X", "date": d,
                          "adj_close": [50.0] * 6, "close": [100.0] * 3 + [50.0] * 3})
        v = verify(b, actions(("I1", d[3], 2.0)))
        assert v["ok"].all()
        assert v["rel_err"].max() < 1e-9

    def test_a_missing_action_is_caught(self):
        """The point of the check: if the action table is incomplete, derived
        and observed diverge by exactly the missing factor."""
        d = dates(6)
        b = pd.DataFrame({"isin": "I1", "symbol": "X", "date": d,
                          "adj_close": [50.0] * 6, "close": [100.0] * 3 + [50.0] * 3})
        v = verify(b, pd.DataFrame())
        assert not v["ok"].iloc[0]
        assert v["rel_err"].iloc[0] == pytest.approx(0.5)

    def test_bars_without_both_blocks_are_skipped(self):
        d = dates(4)
        b = pd.DataFrame({"isin": "I1", "symbol": "X", "date": d,
                          "adj_close": [50.0] * 4, "close": [np.nan] * 4})
        assert verify(b, pd.DataFrame()).empty
