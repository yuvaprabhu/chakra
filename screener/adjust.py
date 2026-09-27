"""Corporate-action adjustment: deriving raw prices from an adjusted series.

Every free provider that covers NSE deeply (Upstox, Yahoo) serves prices
already back-adjusted for splits and bonuses. The exchange's own bhavcopy is
the only true raw source, and it is the least reliable thing to depend on
daily.

The way out is arithmetic. A back-adjusted close divided by the cumulative
factor of every action *after* that date is the price the exchange printed:

    adj[t] = raw[t] / prod(ratio for actions with ex_date > t)
    raw[t] = adj[t] * prod(ratio for actions with ex_date > t)

So a corporate-action table plus an adjusted series reconstructs raw prices for
the entire history, and the bhavcopy becomes a *verification* source rather
than a dependency.

The factor table is rebuildable from scratch at any time, which is the property
the storage design asked for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema as S

RAW_FROM_ADJ = {"open": "adj_open", "high": "adj_high", "low": "adj_low", "close": "adj_close"}


def cumulative_factors(actions: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Factor per date: the product of every action strictly after that date.

    A bar on the ex-date itself is already post-action, so the comparison is
    strict: only actions dated later still need unwinding.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    if actions is None or actions.empty:
        return pd.Series(1.0, index=dates, name="factor")
    a = actions.dropna(subset=["ex_date", "ratio"]).sort_values("ex_date")
    a = a[a["ratio"] > 0]
    if a.empty:
        return pd.Series(1.0, index=dates, name="factor")

    ex = pd.DatetimeIndex(pd.to_datetime(a["ex_date"])).normalize()
    ratios = a["ratio"].to_numpy(dtype="float64")
    # Suffix products: factor for date d is the product of ratios with ex > d.
    suffix = np.concatenate([np.cumprod(ratios[::-1])[::-1], [1.0]])
    pos = np.searchsorted(ex.to_numpy(), dates.to_numpy(), side="right")
    return pd.Series(suffix[pos], index=dates, name="factor")


def derive_raw(bars: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Fill the raw OHLC block from the adjusted block for one symbol.

    Only fills where raw is missing; an observed bhavcopy price always wins over
    a derived one.
    """
    if bars.empty:
        return bars
    out = bars.sort_values("date").copy()
    out["date"] = pd.to_datetime(out["date"])
    factor = cumulative_factors(actions, pd.DatetimeIndex(out["date"]))
    f = out["date"].map(factor)
    out["adj_factor"] = f.to_numpy()
    for raw_col, adj_col in RAW_FROM_ADJ.items():
        derived = out[adj_col] * out["adj_factor"]
        if raw_col in out.columns:
            out[raw_col] = out[raw_col].where(out[raw_col].notna(), derived)
        else:
            out[raw_col] = derived
    return out


def derive_raw_all(bars: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Vectorised across symbols."""
    if bars.empty:
        return bars
    by_isin = {k: v for k, v in actions.groupby("isin")} if len(actions) else {}
    frames = [derive_raw(g, by_isin.get(isin, pd.DataFrame(columns=S.ADJ_COLUMNS)))
              for isin, g in bars.groupby("isin", sort=False)]
    return pd.concat(frames, ignore_index=True)


def verify(bars: pd.DataFrame, actions: pd.DataFrame, *, tol: float = 0.005) -> pd.DataFrame:
    """Compare DERIVED raw against OBSERVED bhavcopy raw, where both exist.

    This is the check that decides whether the derivation can stand in for the
    exchange file. Returns one row per bar that carries both.
    """
    both = bars[bars["close"].notna() & bars["adj_close"].notna()].copy()
    if both.empty:
        return both
    by_isin = {k: v for k, v in actions.groupby("isin")} if len(actions) else {}
    rows = []
    for isin, g in both.groupby("isin", sort=False):
        g = g.sort_values("date").copy()
        g["date"] = pd.to_datetime(g["date"])
        factor = cumulative_factors(by_isin.get(isin, pd.DataFrame(columns=S.ADJ_COLUMNS)),
                                    pd.DatetimeIndex(g["date"]))
        g["adj_factor"] = g["date"].map(factor).to_numpy()
        g["derived_close"] = g["adj_close"] * g["adj_factor"]
        g["abs_err"] = (g["derived_close"] - g["close"]).abs()
        g["rel_err"] = g["abs_err"] / g["close"]
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    out["ok"] = out["rel_err"] <= tol
    return out
