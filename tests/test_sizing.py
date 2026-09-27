"""Risk-based sizing is where a stop width becomes an account outcome, so it gets tests."""
import sys; sys.path.insert(0, ".")
import pandas as pd
import pytest
from screener.sizing import position_size, simulate_risk


def trades(rows):
    return pd.DataFrame(rows, columns=["isin", "entry_i", "exit_i", "net", "risk_pct"])


def test_position_is_risk_budget_over_stop_distance():
    """1% of equity at risk on a 5% stop is a 20% position; on a 2% stop, 50%."""
    assert position_size(100.0, 1.0, 5.0) == pytest.approx(20.0)
    assert position_size(100.0, 1.0, 2.0) == pytest.approx(50.0)
    assert position_size(100.0, 1.0, 0.0) == 0.0


def test_a_full_stop_out_loses_exactly_the_risk_budget():
    """A trade that hits its 5% stop must cost the account 1%, not 5%."""
    r = simulate_risk(trades([("A", 0, 3, -5.0, 5.0)]), risk_per_trade=1.0)
    assert r["total"] == pytest.approx(-1.0)
    assert r["n_taken"] == 1


def test_a_winner_is_scaled_by_the_same_position():
    """+10% on a 20% position (1% risk, 5% stop) is +2% on the account."""
    r = simulate_risk(trades([("A", 0, 3, 10.0, 5.0)]), risk_per_trade=1.0)
    assert r["total"] == pytest.approx(2.0)


def test_no_leverage_a_tight_stop_runs_out_of_cash():
    """1% risk on a 2% stop is half the account a trade: the third signal
    on the same session cannot be paid for and is declined."""
    t = trades([("A", 0, 5, 0.0, 2.0), ("B", 0, 5, 0.0, 2.0), ("C", 0, 5, 0.0, 2.0)])
    r = simulate_risk(t, risk_per_trade=1.0, max_positions=10)
    assert r["n_taken"] == 2 and r["declined_cash"] == 1
    assert r["max_weight"] == pytest.approx(50.0)


def test_open_risk_is_the_sum_of_position_risks():
    """Two positions each risking 1% of equity put 2% of the account at risk."""
    t = trades([("A", 0, 5, 0.0, 5.0), ("B", 0, 5, 0.0, 10.0)])
    r = simulate_risk(t, risk_per_trade=1.0, max_positions=2)
    assert r["max_open_risk"] == pytest.approx(2.0)
    assert r["risk_when_loaded"] == pytest.approx(2.0)


def test_profit_lands_on_the_exit_and_max_positions_is_enforced():
    t = trades([("A", 0, 4, 20.0, 10.0), ("B", 0, 4, 20.0, 10.0)])
    r = simulate_risk(t, risk_per_trade=1.0, max_positions=1)
    eq = r["equity"]
    assert r["n_taken"] == 1
    assert eq.iloc[0] == pytest.approx(100.0) and eq.iloc[3] == pytest.approx(100.0)
    assert r["total"] == pytest.approx(2.0)


def test_missing_stop_column_is_rejected():
    with pytest.raises(ValueError):
        simulate_risk(pd.DataFrame({"isin": ["A"], "entry_i": [0], "exit_i": [1], "net": [1.0]}))
