"""Indicator math, including the four assertions CLAUDE.md names by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.indicators import (
    add_daily_indicators, atr, bollinger, cpr, ema, period_levels, pivots, rsi, sma,
)


def series(vals):
    return pd.Series([float(v) for v in vals])


def bars(n=300, start="2024-01-01", seed=7):
    """Deterministic synthetic daily bars with both price blocks."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = np.clip(close * (1 + rng.normal(0, 0.006, n)), low, high)
    return pd.DataFrame({
        "isin": "INE1", "symbol": "TEST", "date": dates,
        "open": open_, "high": high, "low": low, "close": close,
        "adj_open": open_, "adj_high": high, "adj_low": low, "adj_close": close,
        "volume": rng.integers(1e5, 1e6, n), "turnover": close * rng.integers(1e5, 1e6, n),
    })


class TestPivotsByHand:
    def test_classic_pivots(self):
        """H=110 L=90 C=100 -> P=100, R1=110, S1=90, R2=120, S2=80."""
        p = pivots(110.0, 90.0, 100.0)
        assert p["p"] == pytest.approx(100.0)
        assert p["r1"] == pytest.approx(110.0)
        assert p["s1"] == pytest.approx(90.0)
        assert p["r2"] == pytest.approx(120.0)
        assert p["s2"] == pytest.approx(80.0)

    def test_asymmetric_pivots_by_hand(self):
        """H=120 L=90 C=115 -> P=108.3333, R1=126.6667, S1=96.6667."""
        p = pivots(120.0, 90.0, 115.0)
        assert p["p"] == pytest.approx(325 / 3)
        assert p["r1"] == pytest.approx(2 * (325 / 3) - 90)
        assert p["s1"] == pytest.approx(2 * (325 / 3) - 120)
        assert p["r2"] == pytest.approx((325 / 3) + 30)
        assert p["s2"] == pytest.approx((325 / 3) - 30)


class TestCprByHand:
    def test_known_values(self):
        """H=110 L=90 C=100 -> P=100, BC=100, TC=100, width 0."""
        c = cpr(series([110]), series([90]), series([100]))
        assert c["cpr_p"].iloc[0] == pytest.approx(100.0)
        assert c["cpr_bc"].iloc[0] == pytest.approx(100.0)
        assert c["cpr_tc"].iloc[0] == pytest.approx(100.0)
        assert c["cpr_width"].iloc[0] == pytest.approx(0.0)

    def test_close_above_midpoint(self):
        """H=120 L=100 C=118 -> P=112.6667, BC=110, TC=115.3333."""
        c = cpr(series([120]), series([100]), series([118]))
        assert c["cpr_p"].iloc[0] == pytest.approx(338 / 3)
        assert c["cpr_bc"].iloc[0] == pytest.approx(110.0)
        assert c["cpr_tc"].iloc[0] == pytest.approx(2 * (338 / 3) - 110)

    def test_tc_is_swapped_when_it_computes_below_bc(self):
        """A bar closing near its low puts 2P-BC BELOW BC. Unswapped, a
        'close above TC' rule silently inverts on exactly these bars."""
        c = cpr(series([110]), series([90]), series([92]))
        raw_tc = 2 * ((110 + 90 + 92) / 3) - ((110 + 90) / 2)
        assert raw_tc < 100.0  # the unsorted value really is below BC
        assert c["cpr_tc"].iloc[0] == pytest.approx(100.0)
        assert c["cpr_bc"].iloc[0] == pytest.approx(raw_tc)

    def test_tc_never_below_bc_over_random_bars(self):
        """The invariant, asserted across a wide sample."""
        rng = np.random.default_rng(0)
        low = pd.Series(rng.uniform(50, 150, 5000))
        high = low * (1 + rng.uniform(0.001, 0.25, 5000))
        close = low + (high - low) * rng.uniform(0, 1, 5000)
        c = cpr(high, low, close)
        assert (c["cpr_tc"] >= c["cpr_bc"] - 1e-12).all()
        assert (c["cpr_width"] >= -1e-12).all()


class TestBollingerPopulationSigma:
    def test_matches_hand_computed_population_sigma(self):
        """pandas defaults to ddof=1; charting platforms use ddof=0. With
        1..20 the mean is 10.5 and the POPULATION sigma is sqrt(33.25)."""
        s = series(range(1, 21))
        b = bollinger(s, 20, 2.0)
        mid = b["bb_mid"].iloc[-1]
        upper = b["bb_upper"].iloc[-1]
        assert mid == pytest.approx(10.5)
        pop_sigma = np.sqrt(np.mean((np.arange(1, 21) - 10.5) ** 2))
        assert pop_sigma == pytest.approx(np.sqrt(33.25))
        assert upper == pytest.approx(10.5 + 2 * pop_sigma)

    def test_differs_from_the_sample_sigma_default(self):
        s = series(range(1, 21))
        b = bollinger(s, 20, 2.0)
        sample_upper = 10.5 + 2 * pd.Series(range(1, 21)).astype(float).std(ddof=1)
        assert b["bb_upper"].iloc[-1] != pytest.approx(sample_upper)

    def test_pct_b_and_bandwidth(self):
        s = series(range(1, 21))
        b = bollinger(s, 20, 2.0)
        sd = np.sqrt(33.25)
        assert b["bb_pct_b"].iloc[-1] == pytest.approx((20 - (10.5 - 2 * sd)) / (4 * sd))
        assert b["bb_bandwidth"].iloc[-1] == pytest.approx(4 * sd / 10.5 * 100)

    def test_head_is_nan(self):
        b = bollinger(series(range(1, 31)), 20, 2.0)
        assert b["bb_mid"].iloc[:19].isna().all()
        assert b["bb_mid"].iloc[19:].notna().all()


class TestNoLookahead:
    def test_monthly_level_comes_from_the_previous_month(self):
        """Every trading day in September must carry AUGUST's monthly CPR."""
        df = bars(n=400, start="2024-01-01")
        out = period_levels(df[["date", "high", "low", "close"]], "ME", "m")
        out = pd.concat([df["date"], out], axis=1)

        aug = df[(df["date"] >= "2024-08-01") & (df["date"] <= "2024-08-31")]
        expected = cpr(
            series([aug["high"].max()]), series([aug["low"].min()]),
            series([aug["close"].iloc[-1]]),
        )["cpr_tc"].iloc[0]

        sep = out[(out["date"] >= "2024-09-01") & (out["date"] <= "2024-09-30")]
        assert sep["m_cpr_tc"].notna().all()
        assert sep["m_cpr_tc"].nunique() == 1
        assert sep["m_cpr_tc"].iloc[0] == pytest.approx(expected)

    def test_no_level_derives_from_a_period_ending_on_or_after_t(self):
        """The repaint assertion, checked on every bar.

        For each date t, recompute the level from the period t falls in. If the
        stored level ever equals that, an in-progress period leaked into it.
        """
        df = bars(n=500, start="2023-01-01")
        out = pd.concat([df["date"], period_levels(
            df[["date", "high", "low", "close"]], "ME", "m")], axis=1)

        s = df.set_index("date")
        current = s.resample("ME").agg({"high": "max", "low": "min", "close": "last"})
        cur_tc = cpr(current["high"], current["low"], current["close"])["cpr_tc"]
        cur_tc.index = cur_tc.index.to_period("M")

        leaked = 0
        for _, row in out.dropna(subset=["m_cpr_tc"]).iterrows():
            own_period = pd.Period(row["date"], freq="M")
            if own_period in cur_tc.index and np.isclose(row["m_cpr_tc"], cur_tc.loc[own_period]):
                leaked += 1
        assert leaked == 0

    def test_first_period_has_no_level(self):
        df = bars(n=400, start="2024-01-01")
        out = pd.concat([df["date"], period_levels(
            df[["date", "high", "low", "close"]], "ME", "m")], axis=1)
        jan = out[out["date"] <= "2024-01-31"]
        assert jan["m_cpr_tc"].isna().all()

    def test_weekly_levels_shift_too(self):
        df = bars(n=200, start="2024-01-01")
        out = pd.concat([df["date"], period_levels(
            df[["date", "high", "low", "close"]], "W", "w")], axis=1)
        wk = out[(out["date"] >= "2024-02-05") & (out["date"] <= "2024-02-09")]
        assert wk["w_cpr_tc"].nunique() == 1

    def test_rolling_indicators_have_no_lookahead(self):
        """Truncating the history must not change earlier values."""
        df = bars(n=300)
        full = add_daily_indicators(df)
        part = add_daily_indicators(df.iloc[:200].copy())
        cols = ["sma_20", "sma_50", "rsi_14", "atr_14", "bb_upper", "m_cpr_tc"]
        a = full.iloc[:200][cols].reset_index(drop=True)
        b = part[cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


class TestWilderSmoothing:
    def test_rsi_all_gains_is_100(self):
        assert rsi(series(range(1, 60)), 14).iloc[-1] == pytest.approx(100.0)

    def test_rsi_all_losses_is_zero(self):
        assert rsi(series(range(60, 1, -1)), 14).iloc[-1] == pytest.approx(0.0)

    def test_rsi_is_bounded(self):
        r = rsi(bars(n=400)["close"], 14).dropna()
        assert r.between(0, 100).all()

    def test_rsi_head_is_nan(self):
        r = rsi(series(range(1, 40)), 14)
        assert r.iloc[:14].isna().all() and r.iloc[14:].notna().all()

    def test_atr_uses_wilder_alpha(self):
        df = bars(n=100)
        got = atr(df["high"], df["low"], df["close"], 14)
        from screener.indicators import true_range
        tr = true_range(df["high"], df["low"], df["close"])
        expected = tr.ewm(alpha=1 / 14, adjust=False).mean()
        assert got.dropna().iloc[-1] == pytest.approx(expected.iloc[-1])

    def test_atr_is_positive(self):
        df = bars(n=200)
        assert (atr(df["high"], df["low"], df["close"], 14).dropna() > 0).all()


class TestMovingAverages:
    def test_sma_is_exact(self):
        assert sma(series(range(1, 21)), 20).iloc[-1] == pytest.approx(10.5)

    def test_sma_head_is_nan(self):
        s = sma(series(range(1, 31)), 20)
        assert s.iloc[:19].isna().all()

    def test_ema_blanks_the_unstable_head(self):
        """Seeded from the first value, so the first n-1 bars are not an EMA(n)
        in any meaningful sense and must not be read as one."""
        e = ema(series(range(1, 41)), 20)
        assert e.iloc[:19].isna().all()
        assert e.iloc[19:].notna().all()

    def test_ema_tracks_a_constant_series(self):
        assert ema(series([50] * 40), 20).iloc[-1] == pytest.approx(50.0)


class TestFeatureAssembly:
    def test_produces_expected_columns(self):
        out = add_daily_indicators(bars(n=400))
        for col in ("sma_200", "ema_50", "rsi_14", "atr_14", "bb_pct_b",
                    "m_cpr_tc", "w_cpr_tc", "d_cpr_tc", "vol_ratio", "ret_20d"):
            assert col in out.columns

    def test_levels_are_always_on_the_adjusted_basis(self):
        """Levels never use the raw block. The raw series breaks at a split,
        which would put last period's level a whole split factor away from the
        price it is compared against - see test_screens.py for the case."""
        out = add_daily_indicators(bars(n=300))
        assert (out["levels_basis"] == "adjusted").all()
        no_raw = bars(n=300)
        no_raw[["open", "high", "low", "close"]] = np.nan
        both = add_daily_indicators(no_raw)
        # Dropping the raw block entirely changes nothing about the levels.
        for col in ("m_cpr_tc", "w_cpr_tc", "m_r1", "d_cpr_tc"):
            pd.testing.assert_series_equal(out[col], both[col])

    def test_short_history_yields_nan_not_a_number_from_a_short_window(self):
        """Insufficient history must be NaN and filtered, never a number
        computed from a window shorter than the indicator's period."""
        out = add_daily_indicators(bars(n=120))
        assert out["sma_200"].isna().all()
        assert out["ret_250d"].isna().all()
        assert out["sma_50"].notna().iloc[-1]


class TestEmaSet:
    def test_chart_ema_lengths_present(self):
        out = add_daily_indicators(bars(n=400))
        for n in (9, 21, 50, 200):
            assert f"ema_{n}" in out.columns

    def test_shorter_ema_tracks_price_more_closely(self):
        """EMA(9) must sit nearer the close than EMA(50) on a trending series."""
        df = bars(n=400, seed=3)
        out = add_daily_indicators(df)
        last = out.iloc[-1]
        assert abs(last["adj_close"] - last["ema_9"]) <= abs(last["adj_close"] - last["ema_50"])

    def test_each_ema_blanks_its_own_head(self):
        out = add_daily_indicators(bars(n=400))
        for n in (9, 21, 50):
            assert out[f"ema_{n}"].iloc[: n - 1].isna().all()
            assert out[f"ema_{n}"].iloc[n - 1]  == out[f"ema_{n}"].iloc[n - 1]  # not NaN
