"""Stress a strategy before believing it.

A single backtest number is a point. This turns it into a region: how it did
in each half of history, how much depends on which signal got a slot on a busy
day, and whether the neighbouring parameter settings agree with the chosen one.
A result that only holds at one exact setting, in one half, with one lucky
ordering is a curve fit; this is where that gets caught.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .portfolio import simulate


def _port(trades: pd.DataFrame, slots: int, sessions: pd.DatetimeIndex, order_col: str | None) -> dict:
    if trades is None or len(trades) < 10:
        return {"cagr": np.nan, "maxdd": np.nan, "total": np.nan, "n_taken": 0}
    t = trades.sort_values(["entry_i", order_col], ascending=[True, False]) if order_col else trades
    r = simulate(t, slots=slots, sessions=sessions)
    return {k: r[k] for k in ("cagr", "maxdd", "total", "n_taken")}


def stress(make_trades: Callable[[dict], pd.DataFrame], params: dict, *, sessions: pd.DatetimeIndex,
           split: pd.Timestamp, slots: int = 10, order_col: str | None = "rs",
           n_random: int = 100, neighbour_pct: float = 0.2, seed: int = 7) -> dict:
    """Run the strategy at ``params`` and around it.

    ``make_trades(params)`` must return a trade frame with entry_i, exit_i, net,
    isin, date (signal date) and, if ``order_col`` is given, that column.
    Numeric parameters are each nudged by +/- ``neighbour_pct`` in turn.
    """
    rng = np.random.default_rng(seed)
    base = make_trades(params)
    out = {"params": dict(params), "n_trades": int(len(base))}
    out["full"] = _port(base, slots, sessions, order_col)
    out["h1"] = _port(base[base["date"] < split], slots, sessions, order_col)
    out["h2"] = _port(base[base["date"] >= split], slots, sessions, order_col)

    tot = []
    for _ in range(n_random):
        sh = base.iloc[rng.permutation(len(base))].sort_values("entry_i", kind="stable")
        tot.append(simulate(sh, slots=slots, sessions=sessions)["cagr"])
    tot = np.array(tot)
    out["random_order"] = {"mean": float(tot.mean()), "p5": float(np.percentile(tot, 5)),
                           "p95": float(np.percentile(tot, 95)), "worst": float(tot.min())}

    neigh = {}
    for k, v in params.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        for sign in (-1, 1):
            p2 = dict(params)
            nv = v * (1 + sign * neighbour_pct)
            p2[k] = int(round(nv)) if isinstance(v, int) else nv
            if p2[k] == v:
                continue
            t2 = make_trades(p2)
            neigh[f"{k}={p2[k]}"] = _port(t2, slots, sessions, order_col)["cagr"]
    out["neighbours"] = neigh
    vals = [x for x in neigh.values() if np.isfinite(x)]
    out["neighbour_min"] = float(min(vals)) if vals else np.nan
    out["neighbour_mean"] = float(np.mean(vals)) if vals else np.nan
    # The verdict: both halves up, random orderings mostly up, and the
    # neighbourhood not collapsing. "Robust" is a high bar on purpose.
    out["robust"] = bool(out["h1"]["cagr"] > 0 and out["h2"]["cagr"] > 0
                         and out["random_order"]["p5"] > 0
                         and (not vals or min(vals) > 0))
    return out
