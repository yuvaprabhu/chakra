"""Quality gates. Fail loud.

A screener quietly running on bad data is worse than one that does not run.

The gate that matters most here is the price-jump check. The corporate-action
table is built from split and bonus factors, which covers most events - but NOT
demergers, and India has had a lot of them: Vedanta, Tata Motors, Siemens,
Raymond, Aditya Birla Fashion, SKF, Quess. When a parent spins off a subsidiary
its price drops by the value of what left, and an unadjusted series records
that as a -65% day.

Nothing downstream can tell that apart from a crash. The moving averages bend,
the 52-week high is wrong, relative strength collapses, and any backtested
trade held across the date books a catastrophic loss that never happened.

This module does NOT invent an adjustment factor to paper over it. Deriving one
from the price gap would silently convert every genuine crash into a corporate
action. It flags the event and marks the window where indicators are untrustworthy,
so the affected rows can be excluded and counted rather than quietly used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A real single-session move beyond this is rare enough, and a corporate action
# common enough, that the balance of probability favours investigating.
JUMP_PCT = 25.0
# The longest lookback any indicator uses. After an unadjusted event, every
# window spanning it is wrong, and 250 sessions is the last of them to clear.
CONTAMINATION_SESSIONS = 250


def detect_price_jumps(bars: pd.DataFrame, actions: pd.DataFrame | None = None,
                       *, threshold: float = JUMP_PCT,
                       window_days: int = 4) -> pd.DataFrame:
    """Single-session moves beyond ``threshold`` with no corporate action to explain them."""
    b = bars.sort_values(["isin", "date"]).copy()
    b["move_pct"] = b.groupby("isin")["adj_close"].pct_change() * 100
    sus = b.loc[b["move_pct"].abs() > threshold, ["isin", "date", "adj_close", "move_pct"]].copy()
    if sus.empty:
        return sus.assign(explained=pd.Series(dtype=bool))

    explained = np.zeros(len(sus), dtype=bool)
    if actions is not None and len(actions):
        for i, (isin, date) in enumerate(zip(sus["isin"], sus["date"])):
            a = actions[(actions["isin"] == isin)
                        & (actions["ex_date"].sub(date).abs() <= pd.Timedelta(days=window_days))]
            explained[i] = len(a) > 0
    sus["explained"] = explained
    return sus.reset_index(drop=True)


def contamination_mask(bars: pd.DataFrame, jumps: pd.DataFrame,
                       *, sessions: int = CONTAMINATION_SESSIONS,
                       forward: int = 20) -> pd.Series:
    """True for rows whose indicators - or forward return - span an unexplained jump.

    Two windows, because a jump poisons the data in both directions. Rows AFTER
    it have moving averages and 52-week ranges computed across a discontinuity.
    Rows BEFORE it have a forward return that walks straight through one, which
    is what turns a backtest hit into a fictional 65% loss.
    """
    b = bars.sort_values(["isin", "date"])
    mask = pd.Series(False, index=b.index)
    bad = jumps[~jumps["explained"]] if "explained" in jumps.columns else jumps
    if bad.empty:
        return mask.reindex(bars.index).fillna(False)

    for isin, grp in bad.groupby("isin"):
        rows = b.index[b["isin"] == isin]
        dates = b.loc[rows, "date"].to_numpy()
        for d in grp["date"]:
            i = int(np.searchsorted(dates, np.datetime64(d)))
            lo, hi = max(0, i - forward), min(len(rows), i + sessions + 1)
            mask.loc[rows[lo:hi]] = True
    return mask.reindex(bars.index).fillna(False)


def stale_bars(bars: pd.DataFrame) -> pd.Series:
    """Bars where open, high, low and close are all identical.

    Legitimate on a circuit-locked day, so this is reported rather than
    rejected - but a run of them with zero volume is a dead or junk listing.
    """
    o, h, l, c = bars["adj_open"], bars["adj_high"], bars["adj_low"], bars["adj_close"]
    return (o == h) & (h == l) & (l == c)


def junk_bars(bars: pd.DataFrame) -> pd.Series:
    """Zero-volume bars that also carry no price movement: nothing traded."""
    return stale_bars(bars) & (bars["volume"].fillna(0) <= 0)


def report(bars: pd.DataFrame, actions: pd.DataFrame | None = None) -> dict:
    """Everything the gate found, as numbers a caller can print or assert on."""
    jumps = detect_price_jumps(bars, actions)
    unexplained = jumps[~jumps["explained"]] if len(jumps) else jumps
    contam = contamination_mask(bars, jumps) if len(jumps) else pd.Series(False, index=bars.index)
    junk = junk_bars(bars)
    o, h, l, c = bars["adj_open"], bars["adj_high"], bars["adj_low"], bars["adj_close"]
    return {
        "bars": int(len(bars)),
        "symbols": int(bars["isin"].nunique()),
        "ohlc_violations": int((~((h >= l) & (h >= o - 1e-9) & (h >= c - 1e-9)
                                  & (l <= o + 1e-9) & (l <= c + 1e-9))).sum()),
        "non_positive": int((~((o > 0) & (h > 0) & (l > 0) & (c > 0))).sum()),
        "duplicates": int(bars.duplicated(["isin", "date"]).sum()),
        "jumps_total": int(len(jumps)),
        "jumps_unexplained": int(len(unexplained)),
        "jump_symbols": sorted(unexplained["isin"].unique().tolist()) if len(unexplained) else [],
        "contaminated_rows": int(contam.sum()),
        "contaminated_pct": round(float(contam.mean() * 100), 2),
        "stale_bars": int(stale_bars(bars).sum()),
        "junk_bars": int(junk.sum()),
    }
