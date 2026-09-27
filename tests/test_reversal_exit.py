"""The reversal exit is the only one that kept an edge, so it gets its own tests."""
import sys; sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import pytest
from screener.backtest import reversal_trades, reversal_stats


def frame(closes, opens=None, hi7=None, isin="X"):
    n = len(closes)
    opens = opens if opens is not None else closes
    hi7 = hi7 if hi7 is not None else [np.inf] * n
    return pd.DataFrame({
        "isin": [isin] * n,
        "date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "adj_open": np.array(opens, float),
        "adj_close": np.array(closes, float),
        "hi7_prior": np.array(hi7, float),
    })


def test_entry_is_the_next_open_not_the_signal_close():
    d = frame(closes=[100, 100, 100], opens=[100, 90, 95], hi7=[np.inf, np.inf, 0])
    t = reversal_trades(d, [True, False, False], cost=0)
    assert len(t) == 1
    assert t["entry"].iloc[0] == 90          # bar 1's open, not bar 0's close


def test_exit_is_the_open_after_the_first_close_above_the_prior_seven():
    # bar 2 closes above its hi7, so the exit is bar 3's open
    d = frame(closes=[100, 101, 120, 130], opens=[100, 100, 110, 115],
              hi7=[np.inf, np.inf, 110, np.inf])
    t = reversal_trades(d, [True, False, False, False], cost=0)
    assert t["why"].iloc[0] == "reversal"
    assert t["exit"].iloc[0] == 115
    assert t["bars"].iloc[0] == 3


def test_time_stop_applies_when_the_high_never_comes():
    n = 30
    d = frame(closes=[100] * n, opens=[100] * n, hi7=[np.inf] * n)
    t = reversal_trades(d, [True] + [False] * (n - 1), cost=0, max_bars=5)
    assert t["why"].iloc[0] == "time"
    assert t["bars"].iloc[0] == 5


def test_cost_is_a_full_round_trip_off_the_return():
    d = frame(closes=[100, 100, 110], opens=[100, 100, 110], hi7=[np.inf, np.inf, 0])
    free = reversal_trades(d, [True, False, False], cost=0)["net"].iloc[0]
    paid = reversal_trades(d, [True, False, False], cost=0.30)["net"].iloc[0]
    assert paid == pytest.approx(free - 0.30)


def test_one_open_position_per_name():
    """A second signal while the first trade is still open is skipped, which is
    the only way the result could be traded with one slot per stock."""
    n = 12
    d = frame(closes=[100] * n, opens=[100] * n, hi7=[np.inf] * n)
    t = reversal_trades(d, [True, True, True] + [False] * (n - 3), cost=0, max_bars=5)
    assert len(t) == 1


def test_two_names_do_not_block_each_other():
    a = frame([100] * 8, [100] * 8, [np.inf] * 8, isin="A")
    b = frame([100] * 8, [100] * 8, [np.inf] * 8, isin="B")
    d = pd.concat([a, b], ignore_index=True)
    mask = ([True] + [False] * 7) * 2
    t = reversal_trades(d, mask, cost=0, max_bars=3)
    assert set(t["isin"]) == {"A", "B"}


def test_a_signal_on_the_last_bar_cannot_be_entered():
    d = frame([100, 100], [100, 100], [np.inf, np.inf])
    assert len(reversal_trades(d, [False, True], cost=0)) == 0


def test_stats_are_self_consistent():
    t = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-02-02", "2024-03-04"]),
                      "net": [5.0, -2.0, 1.0], "bars": [3, 9, 4],
                      "why": ["reversal", "time", "reversal"]})
    s = reversal_stats(t)
    assert s["n"] == 3
    assert s["win"] == pytest.approx(200 / 3)
    assert s["profit_factor"] == pytest.approx(6.0 / 2.0)
    assert s["payoff"] == pytest.approx(3.0 / 2.0)
    assert s["on_reversal"] == pytest.approx(200 / 3)


def test_mask_length_is_checked():
    d = frame([100, 100, 100])
    with pytest.raises(ValueError):
        reversal_trades(d, [True, False])


def test_index_never_leaves_the_symbol_on_a_final_bar_reversal():
    """A reversal close on the symbol's last bar has no next open to fill at.
    The loop must stop there rather than stepping into the next symbol's rows."""
    a = frame(closes=[100, 100, 130], opens=[100, 100, 100], hi7=[np.inf, np.inf, 110], isin="A")
    b = frame(closes=[999, 999, 999], opens=[999, 999, 999], hi7=[np.inf] * 3, isin="B")
    d = pd.concat([a, b], ignore_index=True)
    t = reversal_trades(d, [False, True, False, False, False, False], cost=0)
    assert len(t) == 1
    assert t["exit"].iloc[0] == 130          # A's own close, never B's 999
    assert t["bars"].iloc[0] == 1


def test_the_mask_follows_the_callers_row_order():
    """The frame is re-sorted inside; a mask given in the caller's order must
    still select the row the caller meant."""
    d = frame(closes=[100, 100, 120], opens=[100, 90, 110], hi7=[np.inf, np.inf, 110])
    flipped = d.iloc[::-1].reset_index(drop=True)
    t = reversal_trades(flipped, [False, False, True], cost=0)
    assert len(t) == 1
    assert t["entry"].iloc[0] == 90
