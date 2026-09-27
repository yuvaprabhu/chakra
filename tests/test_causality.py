"""No feature may use data from the future.

The third kind of check, and the one that caught the worst defect in this
project. The other two compare a rule against itself or against a definition;
neither can see a feature that peeks forward, because on the live screen there
IS no forward data and everything looks correct. It only corrupts the backtest,
silently, in the direction of flattering it.

The test is mechanical: recompute the feature stack from bars truncated at T,
and compare the row at T with the same row computed from the full history. A
causal indicator cannot tell the difference.

What this caught: `turnover_est` chose its formula with
``if turn.isna().all()`` - a decision over the WHOLE series, future included.
The exchange file covers only recent sessions, so on the historical panel the
reconstruction never fired, turnover stayed NaN, the liquidity gate read NaN as
zero, and 598 of 818 backtest sessions were discarded without a word. The
published study was running on two quarters while claiming three years.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.indicators import build_features

BASE = pd.Timestamp("2024-01-01")


def synth(n=420, seed=3):
    """One symbol with a full price history, corporate-action free."""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0006, 0.017, n))
    dates = pd.bdate_range(BASE, periods=n)
    hi = close * (1 + rng.uniform(0.001, 0.02, n))
    lo = close * (1 - rng.uniform(0.001, 0.02, n))
    op = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({
        "isin": "INE000A01001", "symbol": "TEST", "date": dates,
        "adj_open": op, "adj_high": hi, "adj_low": lo, "adj_close": close,
        "volume": rng.integers(2e5, 4e6, n).astype(float), "trades": pd.NA,
        "turnover": pd.NA,          # the exchange file covers only recent dates
    })
    for c in ("open", "high", "low", "close"):
        df[c] = df["adj_" + c]
    return df


def numeric_columns(f):
    return [c for c in f.columns if f[c].dtype.kind in "fi" and c not in ("volume", "trades")]


@pytest.mark.parametrize("cut_back", [1, 30, 120])
def test_no_feature_changes_when_the_future_is_removed(cut_back):
    bars = synth()
    full = build_features(bars)
    cut = bars["date"].iloc[-cut_back]
    trunc = build_features(bars[bars["date"] <= cut])

    a = full[full["date"] == cut].iloc[0]
    b = trunc[trunc["date"] == cut].iloc[0]
    leaks = [c for c in numeric_columns(full)
             if not np.isclose(float(a[c]), float(b[c]), rtol=1e-9, atol=1e-9, equal_nan=True)]
    assert not leaks, f"features that see the future at {cut.date()}: {leaks}"


def test_turnover_is_reconstructed_per_row_not_per_series():
    """The exact shape of the shipped bug: turnover present only at the END of
    the history must not change how EARLIER rows are computed."""
    bars = synth()
    bars["turnover"] = pd.NA
    full_nan = build_features(bars)

    late = bars.copy()
    tail = late.index[-40:]
    late.loc[tail, "turnover"] = late.loc[tail, "adj_close"] * late.loc[tail, "volume"]
    with_late = build_features(late)

    early = full_nan["date"] < late.loc[tail[0], "date"]
    pd.testing.assert_series_equal(
        full_nan.loc[early, "turnover_median_20d"],
        with_late.loc[early, "turnover_median_20d"],
        check_names=False,
    )


def test_liquidity_is_computable_across_the_whole_history():
    """Missing turnover must be reconstructed, never treated as zero. Read as
    zero it fails the liquidity floor, which silently deletes whole years of
    backtest without any error being raised."""
    bars = synth()
    bars["turnover"] = pd.NA
    f = build_features(bars)
    usable = f["turnover_median_20d"].notna()
    # Only the first 9 bars (below the rolling min_periods) may be blank.
    assert usable.iloc[10:].all(), (
        f"{int((~usable).iloc[10:].sum())} rows have no liquidity figure; "
        "the gate would read those as zero and drop them")
    assert (f.loc[usable, "turnover_median_20d"] > 0).all()
