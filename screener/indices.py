"""Index-level analytics: returns, breadth, money flow and relative rotation.

Three questions this answers, in the order a trader asks them:

1. *What moved?*      -> ``index_table``: returns over 1D/1W/1M/3M/6M/1Y.
2. *Is it broad?*     -> breadth: share of members above their 20/50/200-day
                         averages, advancers vs decliners, new highs and lows.
3. *Where is money
   actually going?*   -> turnover share now versus its own 60-day norm, plus a
                         Relative Rotation Graph against a benchmark.

The RRG construction follows the public description of JdK RS-Ratio and
RS-Momentum (the exact coefficients are proprietary): a sector's price is
divided by the benchmark's, normalised against its own recent mean and
deviation, and the momentum axis is that series normalised again. Leading /
Weakening / Lagging / Improving are then the four quadrants around (100, 100).

**Index levels here are equal-weighted and built from *current* membership**,
because every constituent list we hold is a snapshot taken today. That is
survivorship-biased for history and is fine for reading the last few months of
rotation, but it is not a substitute for the real index level.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .store import Store
from .universe import Universe

log = logging.getLogger(__name__)

RET_WINDOWS = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}


def _pivot(bars: pd.DataFrame, value: str = "adj_close") -> pd.DataFrame:
    return bars.pivot_table(index="date", columns="isin", values=value, aggfunc="last").sort_index()


def equal_weight_level(closes: pd.DataFrame, base: float = 100.0) -> pd.Series:
    """Chain the cross-sectional mean daily return into an index level.

    Chaining mean returns rather than averaging prices keeps a 20,000-rupee
    share from dominating a 40-rupee one.
    """
    rets = closes.pct_change()
    mean_ret = rets.mean(axis=1, skipna=True).fillna(0.0)
    return base * (1.0 + mean_ret).cumprod()


def returns_from_level(level: pd.Series, windows: dict[str, int] = RET_WINDOWS) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, n in windows.items():
        if len(level) > n:
            prev = level.iloc[-1 - n]
            out[label] = float(level.iloc[-1] / prev - 1) * 100 if prev else np.nan
        else:
            out[label] = np.nan
    return out


def breadth(features: pd.DataFrame, isins: list[str]) -> dict[str, float]:
    """Participation stats for one index on the latest bar."""
    f = features[features["isin"].isin(isins)]
    n = len(f)
    if not n:
        return {}
    above = lambda col: int((f["adj_close"] > f[col]).sum()) if col in f else 0
    return {
        "members": n,
        "above_20": above("sma_20"), "above_50": above("sma_50"), "above_200": above("sma_200"),
        "pct_above_50": round(above("sma_50") / n * 100, 1),
        "pct_above_200": round(above("sma_200") / n * 100, 1),
        "advancing": int((f["ret_1d"] > 0).sum()),
        "declining": int((f["ret_1d"] < 0).sum()),
        "new_highs": int((f["pct_from_52w_high"] > -1).sum()),
        "new_lows": int((f["pct_from_52w_low"] < 1).sum()),
        "median_rsi": round(float(f["rsi_14"].median()), 1) if f["rsi_14"].notna().any() else None,
        "median_ret_1d": round(float(f["ret_1d"].median()), 2),
    }


def money_flow(turnover: pd.DataFrame, isins: list[str], *, fast: int = 5, slow: int = 60) -> dict[str, float]:
    """Share of market turnover, and how that share is changing.

    Three different questions, deliberately kept apart:

    * ``flow_delta``  - last week's share against its own 60-day norm. "Is this
      sector busier than it usually is?"
    * ``dod_pp`` / ``wow_pp`` / ``mom_pp`` - today against yesterday, this week
      against last week, this month against last month. Three windows on the
      same question: "is the money arriving or leaving right now?" Day is the
      earliest and the noisiest; month is the slowest and the most reliable.
    * ``turnover_wow`` / ``turnover_mom`` - the same windows in rupees. Share can
      fall while rupees rise if the whole market grew, and a trader needs to
      know which of the two happened.

    Note this is turnover, not price. A sector can rise while its share of
    turnover falls, which is a rally nobody is participating in.
    """
    cols = [c for c in isins if c in turnover.columns]
    if not cols or turnover.empty:
        return {}
    total = turnover.sum(axis=1, skipna=True)
    mine = turnover[cols].sum(axis=1, skipna=True)
    share = (mine / total.replace(0, np.nan)) * 100

    def window(series: pd.Series, n: int, back: int = 0) -> float:
        """Mean of n points ending `back` points before the last one."""
        end = len(series) - back
        seg = series.iloc[max(end - n, 0):end]
        return float(seg.mean()) if len(seg) else float("nan")

    def change(series: pd.Series, n: int) -> tuple[float, float]:
        now, prev = window(series, n), window(series, n, back=n)
        return now, prev

    share_d, share_d_prev = change(share, 1)
    share_w, share_w_prev = change(share, 5)
    share_m, share_m_prev = change(share, 21)
    turn_w, turn_w_prev = change(mine, 5)
    turn_m, turn_m_prev = change(mine, 21)
    fast_share, slow_share = window(share, fast), window(share, slow)

    def pp(a: float, b: float) -> float | None:
        return None if (pd.isna(a) or pd.isna(b)) else round(a - b, 2)

    def growth(a: float, b: float) -> float | None:
        return None if (pd.isna(a) or pd.isna(b) or not b) else round((a / b - 1) * 100, 1)

    return {
        "turnover_cr": round(float(mine.iloc[-1]) / 1e7, 1),
        "share_pct": round(float(share.iloc[-1]), 2),
        "share_5d": round(fast_share, 2),
        "share_60d": round(slow_share, 2),
        "flow_delta": pp(fast_share, slow_share),
        "dod_pp": pp(share_d, share_d_prev),
        "wow_pp": pp(share_w, share_w_prev),
        "mom_pp": pp(share_m, share_m_prev),
        "turnover_wow": growth(turn_w, turn_w_prev),
        "turnover_mom": growth(turn_m, turn_m_prev),
    }


def rrg_point(
    level: pd.Series,
    bench: pd.Series,
    *,
    n: int = 14,
    m: int = 5,
    freq: str = "W-FRI",
    trail: int = 5,
) -> dict[str, float]:
    """JdK-style RS-Ratio and RS-Momentum for one index against a benchmark.

    Sampled WEEKLY, which is how RRG is conventionally read. Daily sampling
    makes the normalised series whip around and the rotation tails turn into
    spaghetti that says nothing; weekly bars give the smooth clockwise arc the
    chart exists to show.

    Returns the latest point plus a tail, so the UI can draw the rotation path
    that makes a quadrant crossing readable.
    """
    joined = pd.concat([level.rename("s"), bench.rename("b")], axis=1).dropna()
    if freq:
        joined = joined.resample(freq).last().dropna()
    if len(joined) < n + m + 3:
        return {}
    rs = 100.0 * joined["s"] / joined["b"]
    sd = rs.rolling(n).std(ddof=0)
    rs_ratio = 100 + (rs - rs.rolling(n).mean()) / sd.replace(0, np.nan)
    sd2 = rs_ratio.rolling(m).std(ddof=0)
    rs_mom = 100 + (rs_ratio - rs_ratio.rolling(m).mean()) / sd2.replace(0, np.nan)
    tail = pd.concat([rs_ratio.rename("x"), rs_mom.rename("y")], axis=1).dropna().tail(trail)
    if tail.empty:
        return {}
    x, y = float(tail["x"].iloc[-1]), float(tail["y"].iloc[-1])
    return {
        "rs_ratio": round(x, 2), "rs_mom": round(y, 2),
        "quadrant": quadrant(x, y),
        "trail": [[round(a, 2), round(b, 2)] for a, b in zip(tail["x"], tail["y"])],
    }


def quadrant(x: float, y: float) -> str:
    if x >= 100 and y >= 100:
        return "leading"
    if x >= 100:
        return "weakening"
    if y >= 100:
        return "improving"
    return "lagging"


def build_index_analytics(
    store: Store,
    features: pd.DataFrame,
    index_names: list[str],
    *,
    benchmark: str = "NIFTY 500",
    as_of=None,
    lookback_days: int = 800,
) -> tuple[list[dict], dict]:
    """One row per index: returns, breadth, money flow and rotation."""
    u = Universe(store)
    as_of = pd.Timestamp(as_of or features["date"].max())
    start = as_of - pd.Timedelta(days=int(lookback_days * 1.5))

    members = {name: u.isins_on(name, as_of) for name in index_names}
    all_isins = sorted({i for v in members.values() for i in v})
    bars = store.read_bars(start=start, end=as_of, isins=all_isins,
                           columns=["isin", "date", "adj_close", "volume", "turnover"])
    bars["date"] = pd.to_datetime(bars["date"])
    closes = _pivot(bars, "adj_close")
    # Bhavcopy turnover is only present for recent sessions; fall back to
    # price x volume so the flow series covers the whole window.
    bars["turnover_est"] = bars["turnover"].fillna(bars["adj_close"] * bars["volume"])
    turnover = _pivot(bars, "turnover_est")

    levels = {name: equal_weight_level(closes[[c for c in members[name] if c in closes.columns]])
              for name in index_names if members[name]}
    bench_level = levels.get(benchmark)

    empty = [n for n in index_names if not members[n]]
    if empty:
        # Loud, because the usual cause is a snapshot dated to a calendar day
        # that is not a trading session, which makes every index silently vanish.
        log.warning("%d index/indices have no members on %s: %s",
                    len(empty), as_of.date(), empty[:6])

    rows = []
    for name in index_names:
        isins = members[name]
        if not isins or name not in levels:
            continue
        lvl = levels[name]
        row = {"index_name": name, "members": len(isins)}
        row.update({f"ret_{k}": (None if pd.isna(v) else round(v, 2))
                    for k, v in returns_from_level(lvl).items()})
        row.update(breadth(features, isins))
        row.update(money_flow(turnover, isins))
        if bench_level is not None and name != benchmark:
            row.update(rrg_point(lvl, bench_level))
        rows.append(row)

    ctx = {
        "benchmark": benchmark,
        "as_of": str(as_of.date()),
        "universe": len(all_isins),
        "total_turnover_cr": round(float(turnover.iloc[-1].sum()) / 1e7, 1) if len(turnover) else None,
    }
    return rows, ctx
