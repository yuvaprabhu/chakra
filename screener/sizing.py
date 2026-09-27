"""Risk-based position sizing: the account decides the size, the stop decides the risk.

``portfolio.simulate`` commits equity / slots to every trade regardless of where
its stop sits, so a trade with a 3% stop risks a third of what a trade with a
9% stop risks. Fixed-fractional sizing inverts that: every trade risks the same
slice of equity, and the position is whatever size makes the stop distance
equal to that slice:

    position = equity * risk_per_trade / stop_distance

A 1% risk budget on a 5% stop is a 20% position; on a 2.5% stop it is a 40%
position. That second number is the whole story of tight stops - the loss per
trade is fixed, but the money committed to a single name is not, and a gap
through the stop is a loss on the position, not on the budget.

The engine keeps the conventions that ``portfolio.simulate`` is tested for: a
slot is measured in sessions, profit lands on the exit, equity is marked at
cost while a trade is open, and the caller's row order decides who gets a
contested slot. There is no leverage: a position that cannot be paid for from
cash is declined (or, with ``allow_partial``, cut to the cash that is there).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_YEAR = 252


def position_size(equity: float, risk_per_trade: float, stop_pct: float) -> float:
    """Money to commit so that a stop at ``stop_pct`` below entry loses
    ``risk_per_trade`` per cent of ``equity``. Both percentages are in per cent."""
    if stop_pct <= 0 or risk_per_trade <= 0 or equity <= 0:
        return 0.0
    return equity * (risk_per_trade / stop_pct)


def simulate_risk(trades: pd.DataFrame, *, risk_per_trade: float = 1.0, max_positions: int = 10,
                  sessions: pd.DatetimeIndex | None = None, allow_partial: bool = False,
                  max_weight: float | None = None) -> dict:
    """Fixed-fractional portfolio over a trade list.

    ``trades`` needs ``entry_i`` and ``exit_i`` (integer session indices into a
    shared calendar), ``net`` (per cent on the position, costs deducted),
    ``isin`` and ``risk_pct`` (the initial stop distance in per cent of entry,
    the same column ``backtest.bracket_returns`` produces).

    ``risk_per_trade`` is the per cent of current equity a full stop-out loses.
    ``max_weight`` optionally caps any single position at that fraction of
    equity (0.25 = a quarter of the account), which is the only defence against
    a very tight stop turning into a very large position.

    Returns the equity curve and the headline numbers, plus what the sizing
    did: the peak and average per cent of equity at risk across open
    positions, the largest single position weight, and the worst calendar
    month when ``sessions`` is given.
    """
    need = {"entry_i", "exit_i", "net", "isin", "risk_pct"}
    missing = need - set(trades.columns)
    if missing:
        raise ValueError(f"simulate_risk needs {sorted(missing)}")
    if max_positions < 1:
        raise ValueError("max_positions must be at least 1")
    if risk_per_trade <= 0:
        raise ValueError("risk_per_trade must be positive")
    if not len(trades):
        return {"n_taken": 0, "n_offered": 0, "equity": pd.Series(dtype=float)}

    d = trades.sort_values("entry_i", kind="stable").reset_index(drop=True)
    first, last = int(d["entry_i"].min()), int(d["exit_i"].max())
    by_entry: dict[int, list] = {}
    for r in d.itertuples():
        by_entry.setdefault(int(r.entry_i), []).append(r)

    cash = 100.0
    open_pos: list[tuple[int, float, float, str, float]] = []   # exit_i, committed, net, isin, stop_pct
    curve = np.empty(last - first + 1, dtype=float)
    open_risk = np.zeros(last - first + 1, dtype=float)          # % of equity at risk, per session
    n_open = np.zeros(last - first + 1, dtype=int)
    taken = 0
    declined_cash = 0
    max_w = 0.0
    held = set()

    for i in range(first, last + 1):
        still = []
        for ex, amt, net, isin, sp in open_pos:
            if ex < i:
                cash += amt * (1 + net / 100.0)
                held.discard(isin)
            else:
                still.append((ex, amt, net, isin, sp))
        open_pos = still
        equity = cash + sum(p[1] for p in open_pos)
        for r in by_entry.get(i, []):
            if len(open_pos) >= max_positions or r.isin in held:
                continue
            sp = float(r.risk_pct)
            want = position_size(equity, risk_per_trade, sp)
            if max_weight is not None:
                want = min(want, equity * max_weight)
            if want <= 0:
                continue
            if want > cash:
                if not allow_partial or cash <= 0:
                    declined_cash += 1
                    continue
                want = cash
            open_pos.append((int(r.exit_i), want, float(r.net), r.isin, sp))
            held.add(r.isin)
            cash -= want
            taken += 1
            max_w = max(max_w, want / equity)
        equity = cash + sum(p[1] for p in open_pos)
        curve[i - first] = equity
        open_risk[i - first] = sum(p[1] * p[4] / 100.0 for p in open_pos) / equity * 100.0
        n_open[i - first] = len(open_pos)

    for ex, amt, net, isin, sp in open_pos:
        cash += amt * (1 + net / 100.0)
    final = cash
    idx = (sessions[first:last + 1] if sessions is not None
           else pd.RangeIndex(first, last + 1))
    eq = pd.Series(curve, index=idx)
    eq.iloc[-1] = final
    years = len(eq) / TRADING_DAYS_YEAR
    dd = float((eq / eq.cummax() - 1).min() * 100)
    worst_month = None
    months_up = None
    if isinstance(idx, pd.DatetimeIndex) and len(eq) > 1:
        m = eq.groupby(eq.index.to_period("M")).last()
        m = pd.concat([pd.Series([100.0]), m.reset_index(drop=True)]).pct_change().dropna() * 100
        if len(m):
            worst_month = float(m.min())
            months_up = float((m > 0).mean() * 100)
    loaded = n_open >= max_positions
    return {
        "n_offered": int(len(d)), "n_taken": taken, "declined_cash": int(declined_cash),
        "total": float(final - 100.0),
        "cagr": float((final / 100.0) ** (1 / max(years, 1e-9)) * 100 - 100),
        "maxdd": dd, "years": float(years), "equity": eq,
        "worst_month": worst_month, "months_up": months_up,
        "max_open_risk": float(open_risk.max()),
        "avg_open_risk": float(open_risk.mean()),
        "risk_when_loaded": float(open_risk[loaded].mean()) if loaded.any() else None,
        "sessions_loaded": int(loaded.sum()),
        "avg_open": float(n_open.mean()),
        "max_weight": float(max_w * 100),
    }
