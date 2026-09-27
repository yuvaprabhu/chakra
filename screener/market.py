"""Market-state columns and fresh-entry flags, attached to the feature panel.

The swing lab found that for trend entries the filters that survive both
halves are market-level, not stock-level: where India VIX sits in its one-year
range, how wide the Nifty's 20-day range is, and whether the breadth gate is
on.  A rule expression can only see columns of the feature frame, so those
series are broadcast onto every row by date, prefixed ``mkt_``.

``fresh_*`` flags turn a state screen into an event: the first session the
rule is true after at least QUIET sessions of being false.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INDICES = Path("data/raw/indices_daily.parquet")
QUIET = 10                       # sessions of "false" before a new "true" counts as fresh
MARKET_COLS = ["mkt_vix", "mkt_vix_pctile_1y", "mkt_nifty_range_20d", "mkt_breadth_50", "mkt_gate_on"]


def index_series(path: Path = INDICES) -> pd.DataFrame:
    """Per-date VIX percentile (1y) and Nifty 50 20-day range %, from the index store."""
    if not Path(path).exists():
        return pd.DataFrame(columns=["date", "mkt_vix", "mkt_vix_pctile_1y", "mkt_nifty_range_20d"])
    ix = pd.read_parquet(path)
    v = ix[ix["index_name"] == "India VIX"].set_index("date")["close"].sort_index()
    n = ix[ix["index_name"] == "Nifty 50"].set_index("date").sort_index()
    out = pd.DataFrame({"mkt_vix": v, "mkt_vix_pctile_1y": v.rolling(252).rank(pct=True) * 100})
    out["mkt_nifty_range_20d"] = ((n["high"].rolling(20).max() / n["low"].rolling(20).min()) - 1) * 100
    return out.reset_index().rename(columns={"index": "date"})


def gate_series(feat: pd.DataFrame) -> pd.DataFrame:
    """Breadth (% above 50-DMA) with 50/40 hysteresis, OR a Zweig thrust window.

    Same construction as scripts/regime_gates.py and scripts/build_gate.py.
    """
    g = feat.groupby("date")
    cal = pd.DatetimeIndex(sorted(feat["date"].unique()))
    b = pd.DataFrame(index=cal)
    ret = feat["adj_close"] / feat.groupby("isin")["adj_close"].shift(1) - 1
    b["adv"] = ret.gt(0).groupby(feat["date"]).sum().reindex(cal)
    b["dec"] = ret.lt(0).groupby(feat["date"]).sum().reindex(cal)
    above = (feat["adj_close"] > feat["sma_50"]).where(feat["sma_50"].notna())
    b["b50"] = above.groupby(feat["date"]).mean().reindex(cal) * 100
    ze = (b["adv"] / (b["adv"] + b["dec"])).ewm(span=10, adjust=False).mean().to_numpy()
    b50 = b["b50"].to_numpy(); hyst = np.full(len(b50), np.nan); state = np.nan
    for i in range(len(b50)):
        if np.isnan(b50[i]):
            continue
        if np.isnan(state):
            state = 1.0 if b50[i] > 50 else 0.0
        elif state == 1.0 and b50[i] < 40:
            state = 0.0
        elif state == 0.0 and b50[i] > 50:
            state = 1.0
        hyst[i] = state
    thrust = np.zeros(len(ze))
    for i in range(10, len(ze)):
        if ze[i] > 0.615 and np.nanmin(ze[i - 10:i]) < 0.40:
            thrust[i:i + 126] = 1
    b["mkt_breadth_50"] = b50
    b["mkt_gate_on"] = ((hyst == 1) | (thrust == 1)).astype(int)
    return b[["mkt_breadth_50", "mkt_gate_on"]].reset_index().rename(columns={"index": "date"})


def attach_market(feat: pd.DataFrame, indices: Path = INDICES) -> pd.DataFrame:
    """Broadcast the market columns onto every row of the panel by date."""
    feat = feat.drop(columns=[c for c in MARKET_COLS if c in feat.columns])
    m = index_series(indices).merge(gate_series(feat), on="date", how="outer")
    return feat.merge(m, on="date", how="left")


def add_fresh_flags(feat: pd.DataFrame, exprs: dict[str, str], quiet: int = QUIET) -> pd.DataFrame:
    """``fresh_<name>`` = 1 on the first session ``expr`` is true after ``quiet`` false sessions."""
    feat = feat.sort_values(["isin", "date"])
    for name, expr in exprs.items():
        state = feat.eval(expr).fillna(False).astype(bool)
        prev_true = (state.astype(int).groupby(feat["isin"])
                     .transform(lambda s: s.shift(1).rolling(quiet, min_periods=quiet).sum()))
        feat[f"fresh_{name}"] = (state & (prev_true == 0)).astype(int)
    return feat
