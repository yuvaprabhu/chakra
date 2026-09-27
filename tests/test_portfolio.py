"""A slot portfolio is where per-trade averages meet reality, so it gets tests."""
import sys; sys.path.insert(0, ".")
import pandas as pd
import pytest
from screener.portfolio import simulate


def trades(rows):
    return pd.DataFrame(rows, columns=["isin", "entry_i", "exit_i", "net"])


def test_one_trade_moves_equity_by_its_share_of_the_book():
    """One slot of ten earning 10% is 1% on the account, not 10%."""
    r = simulate(trades([("A", 0, 5, 10.0)]), slots=10)
    assert r["total"] == pytest.approx(1.0)
    assert r["n_taken"] == 1


def test_profit_lands_on_the_exit_not_the_entry():
    """Equity must be flat while the position is open, then step at the exit."""
    r = simulate(trades([("A", 0, 4, 20.0)]), slots=1)
    eq = r["equity"]
    assert eq.iloc[0] == pytest.approx(100.0)
    assert eq.iloc[3] == pytest.approx(100.0)      # still open
    assert r["total"] == pytest.approx(20.0)


def test_a_slot_is_blocked_for_the_whole_holding_period():
    """Two signals on the same session with one slot: the second is declined."""
    r = simulate(trades([("A", 0, 20, 5.0), ("B", 0, 3, 5.0)]), slots=1)
    assert r["n_offered"] == 2 and r["n_taken"] == 1


def test_the_slot_frees_the_session_after_the_exit():
    r = simulate(trades([("A", 0, 3, 0.0), ("B", 4, 6, 10.0)]), slots=1)
    assert r["n_taken"] == 2


def test_a_signal_during_an_open_trade_in_the_same_name_is_declined():
    r = simulate(trades([("A", 0, 10, 0.0), ("A", 2, 5, 50.0)]), slots=5)
    assert r["n_taken"] == 1


def test_holding_period_is_sessions_not_calendar_days():
    """Exit index 10 means ten sessions; a slot must not free up at day 7."""
    r = simulate(trades([("A", 0, 10, 0.0), ("B", 8, 12, 0.0)]), slots=1)
    assert r["n_taken"] == 1                        # B falls inside A's hold


def test_losses_compound_downward():
    r = simulate(trades([("A", 0, 2, -50.0), ("A", 3, 5, -50.0)]), slots=1)
    assert r["total"] == pytest.approx(-75.0)


def test_drawdown_is_measured_on_the_curve():
    r = simulate(trades([("A", 0, 2, -20.0), ("B", 3, 5, 100.0)]), slots=1)
    assert r["maxdd"] == pytest.approx(-20.0)
    assert r["total"] == pytest.approx(60.0)


def test_equity_is_never_invented_while_a_trade_is_open():
    """A winner still running must not lift the curve before it closes."""
    eq = simulate(trades([("A", 0, 30, 100.0)]), slots=1)["equity"]
    assert eq.iloc[:-1].max() == pytest.approx(100.0)


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError):
        simulate(pd.DataFrame({"isin": ["A"], "net": [1.0]}))


def test_cagr_uses_a_trading_year():
    """252 sessions is one year, so doubling over 252 sessions is +100% a year."""
    r = simulate(trades([("A", 0, 251, 100.0)]), slots=1)
    assert r["years"] == pytest.approx(1.0, abs=0.01)
    assert r["cagr"] == pytest.approx(100.0, abs=1.0)


def test_the_caller_decides_who_gets_a_contested_slot():
    """Two names, one slot, same session. The engine must take whichever the
    caller listed first, not whichever sorts first alphabetically."""
    a_first = trades([("AAA", 0, 3, 1.0), ("ZZZ", 0, 3, 50.0)])
    z_first = trades([("ZZZ", 0, 3, 50.0), ("AAA", 0, 3, 1.0)])
    assert simulate(a_first, slots=1)["total"] == pytest.approx(1.0)
    assert simulate(z_first, slots=1)["total"] == pytest.approx(50.0)
