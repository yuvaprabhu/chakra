"""Turn a trade list into what an account would actually have done.

A per-trade average says nothing about an account. Ten slots that each recycle
into the next signal is a different thing from an unlimited-capital average:
slots are scarce, so on a busy day most signals are declined, and a trade that
runs for six weeks blocks a slot for six weeks.

Two rules matter and both are easy to get wrong:

* A slot is measured in SESSIONS, not calendar days. A trade held eleven
  sessions blocks its slot for eleven sessions, which is about fifteen days.
  Using calendar arithmetic frees slots early and invents trades.
* A trade's profit lands on its EXIT, not its entry. Compounding at entry
  front-runs the gain and hides the drawdown while the position is open.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_YEAR = 252


def simulate(trades: pd.DataFrame, *, slots: int = 10, sessions: pd.DatetimeIndex | None = None,
             allow_partial: bool = False) -> dict:
    """Equal-weight slot portfolio over a trade list.

    ``trades`` needs ``entry_i`` and ``exit_i`` (integer session indices into a
    shared calendar), ``net`` (per cent, costs already deducted) and ``isin``.
    Returns the equity curve plus the headline numbers.

    Capital is split ``slots`` ways at each entry. While a position is open its
    slot holds the amount committed; the profit is credited on the exit session.
    Equity is therefore marked at cost during a trade, which understates the
    peak but never invents one.
    """
    need = {"entry_i", "exit_i", "net", "isin"}
    missing = need - set(trades.columns)
    if missing:
        raise ValueError(f"simulate needs {sorted(missing)}")
    if slots < 1:
        raise ValueError("slots must be at least 1")
    if not len(trades):
        return {"n_taken": 0, "n_offered": 0, "equity": pd.Series(dtype=float)}

    # Sort on the entry session ONLY, stably, so the caller decides who gets a
    # slot when a day offers more signals than there are slots. Sorting by isin
    # here would hand every contested slot to the alphabetically first name and
    # silently make that arbitrary choice part of the result.
    d = trades.sort_values("entry_i", kind="stable").reset_index(drop=True)
    first, last = int(d["entry_i"].min()), int(d["exit_i"].max())
    by_entry: dict[int, list] = {}
    for r in d.itertuples():
        by_entry.setdefault(int(r.entry_i), []).append(r)

    cash = 100.0
    open_pos: list[tuple[int, float, float, str]] = []      # exit_i, committed, net, isin
    curve = np.empty(last - first + 1, dtype=float)
    taken = 0
    held = set()

    for i in range(first, last + 1):
        # Yesterday's exits free their slot and settle before today's entries.
        still = []
        for ex, amt, net, isin in open_pos:
            if ex < i:
                cash += amt * (1 + net / 100.0)
                held.discard(isin)
            else:
                still.append((ex, amt, net, isin))
        open_pos = still
        equity = cash + sum(p[1] for p in open_pos)
        for r in by_entry.get(i, []):
            if len(open_pos) >= slots or r.isin in held:
                continue
            want = equity / slots
            amt = want if want <= cash else (cash if allow_partial else 0.0)
            if amt <= 0:
                continue
            open_pos.append((int(r.exit_i), amt, float(r.net), r.isin))
            held.add(r.isin)
            cash -= amt
            taken += 1
        curve[i - first] = cash + sum(p[1] for p in open_pos)

    for ex, amt, net, isin in open_pos:                      # settle what is still open
        cash += amt * (1 + net / 100.0)
    final = cash
    idx = (sessions[first:last + 1] if sessions is not None
           else pd.RangeIndex(first, last + 1))
    eq = pd.Series(curve, index=idx)
    eq.iloc[-1] = final
    years = len(eq) / TRADING_DAYS_YEAR
    dd = float((eq / eq.cummax() - 1).min() * 100)
    return {
        "n_offered": int(len(d)), "n_taken": taken,
        "total": float(final - 100.0),
        "cagr": float((final / 100.0) ** (1 / max(years, 1e-9)) * 100 - 100),
        "maxdd": dd, "years": float(years), "equity": eq,
    }
