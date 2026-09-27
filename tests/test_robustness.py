import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.robustness import stress

CAL = pd.date_range("2024-01-01", periods=600, freq="B")
SPLIT = CAL[300]


def make(params):
    """A synthetic strategy whose return scales with a parameter."""
    rows = []
    for i in range(0, 580, 10):
        rows.append(("A", CAL[i], i, i + 5, params["edge"], 90.0))
        rows.append(("B", CAL[i], i, i + 5, params["edge"] * 0.5, 50.0))
    return pd.DataFrame(rows, columns=["isin", "date", "entry_i", "exit_i", "net", "rs"])


def test_reports_both_halves_and_neighbours():
    r = stress(make, {"edge": 2.0}, sessions=CAL, split=SPLIT, slots=1, n_random=20)
    assert r["h1"]["cagr"] > 0 and r["h2"]["cagr"] > 0
    assert set(r["neighbours"]) == {"edge=1.6", "edge=2.4"}
    assert r["robust"] is True


def test_random_orderings_expose_slot_choice():
    """One slot, two signals a day, one twice as good: ordering must matter."""
    r = stress(make, {"edge": 4.0}, sessions=CAL, split=SPLIT, slots=1, n_random=40)
    assert r["random_order"]["p95"] > r["random_order"]["p5"]
    assert r["full"]["cagr"] >= r["random_order"]["mean"]      # rs ordering picks A


def test_a_losing_strategy_is_not_robust():
    r = stress(make, {"edge": -1.0}, sessions=CAL, split=SPLIT, slots=1, n_random=10)
    assert r["robust"] is False
