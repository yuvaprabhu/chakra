"""Indicator math. Pure functions over a DataFrame, vectorised across symbols.

No lookahead: every value at date ``t`` uses only data up to ``t``.

Two price bases, never mixed within one calculation:

* **adjusted** (``adj_close`` and friends) for moving averages, Bollinger, RSI
  and ATR — a split would otherwise register as a 50% crash.
* **raw** for pivots and CPR, which are levels a trader reads off a chart.

Where a raw block is absent the level functions fall back to the adjusted
series and say so via ``levels_basis``. That fallback is self-consistent: the
adjusted series is continuous and entirely on today's price basis, so
``close > tc`` compares like with like. It is only the *printed level* that can
differ from a trader's chart, and only for a symbol with a corporate action
inside the lookback window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "sma", "ema", "rsi", "atr", "bollinger", "pivots", "cpr",
    "bars_since", "crossed_above",
    "period_levels", "add_daily_indicators", "build_features",
]


# --- trend ------------------------------------------------------------------


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    """EMA seeded from the first value, with the first n-1 bars blanked.

    TradingView seeds from SMA(n). Rather than reconcile two conventions, the
    unstable head is NaN'd out so nothing downstream reads a number that came
    from a window shorter than n.
    """
    out = s.ewm(span=n, adjust=False).mean()
    return out.mask(s.notna().cumsum() < n)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI. Wilder smoothing is ewm(alpha=1/n, adjust=False)."""
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    # A window with no losses is RSI 100, not NaN from the divide.
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return out.mask(s.notna().cumsum() <= n)


def bars_since(flag: pd.Series) -> pd.Series:
    """Sessions since ``flag`` was last True; NaN before it ever was.

    This is what turns a state test into an event test. "Breakout" and
    "reclaim" describe a crossing, but a rule that only asks whether price is
    above a level matches every day it stays there - which is how 59% of the
    narrow-CPR hits were names that had been above TC for weeks.
    """
    idx = pd.Series(np.arange(len(flag)), index=flag.index, dtype="float64")
    last = idx.where(flag.fillna(False)).ffill()
    return idx - last


def crossed_above(series: pd.Series, level: pd.Series) -> pd.Series:
    """True on the session a series closes above a level it was at or below."""
    above = series > level
    return above & ~above.shift(1, fill_value=False)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    out = tr.ewm(alpha=1 / n, adjust=False).mean()
    return out.mask(close.notna().cumsum() <= n)


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Bollinger bands using POPULATION sigma.

    pandas defaults to ddof=1; every charting platform uses ddof=0, and the
    bands will not match any of them until this is set explicitly.
    """
    mid = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + k * sd
    lower = mid - k * sd
    width = (upper - lower) / mid * 100
    pct_b = (s - lower) / (upper - lower)
    return pd.DataFrame({
        "bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
        "bb_bandwidth": width, "bb_pct_b": pct_b,
    })


# --- levels -----------------------------------------------------------------


def pivots(high, low, close) -> dict[str, float | pd.Series]:
    """Classic floor-trader pivots."""
    p = (high + low + close) / 3
    return {
        "p": p,
        "r1": 2 * p - low,
        "s1": 2 * p - high,
        "r2": p + (high - low),
        "s2": p - (high - low),
        "r3": high + 2 * (p - low),
        "s3": low - 2 * (high - p),
    }


def cpr(high, low, close) -> dict[str, pd.Series]:
    """Central Pivot Range.

    TC as written (2P - BC) can compute BELOW BC — on a bar closing near its
    low, for instance. The pair is sorted so tc is always the upper line;
    without that, a 'close above TC' rule silently inverts on those bars.
    """
    p = (high + low + close) / 3
    bc = (high + low) / 2
    tc_raw = 2 * p - bc
    tc = pd.concat([tc_raw, bc], axis=1).max(axis=1) if hasattr(tc_raw, "index") else max(tc_raw, bc)
    bc_sorted = pd.concat([tc_raw, bc], axis=1).min(axis=1) if hasattr(tc_raw, "index") else min(tc_raw, bc)
    width = (tc - bc_sorted) / p * 100
    return {"cpr_p": p, "cpr_tc": tc, "cpr_bc": bc_sorted, "cpr_width": width}


def period_levels(
    df: pd.DataFrame,
    freq: str,
    prefix: str,
    *,
    high="high", low="low", close="close", date="date",
) -> pd.DataFrame:
    """Pivots and CPR from the LAST FULLY CLOSED period, on the daily grid.

    The repaint rule. Every trading day in September carries August's monthly
    levels. Implemented as resample -> compute -> shift(1) -> forward-fill, so
    a level can never derive from a period that has not finished. If an
    in-progress period ever reaches a rule, the backtest is fiction.
    """
    s = df.set_index(date).sort_index()
    agg = s.resample(freq).agg({high: "max", low: "min", close: "last"}).dropna()
    if agg.empty:
        return pd.DataFrame(index=df.index)

    piv = pivots(agg[high], agg[low], agg[close])
    cp = cpr(agg[high], agg[low], agg[close])
    levels = pd.DataFrame({**piv, **cp}, index=agg.index)

    # The shift is the whole point: period N's levels apply from period N+1.
    levels = levels.shift(1)
    levels.columns = [f"{prefix}_{c}" for c in levels.columns]

    # Map each daily bar onto the period it falls in, then read that period's
    # (already shifted) levels.
    period_of_day = s.index.to_period(_freq_to_period(freq))
    levels.index = levels.index.to_period(_freq_to_period(freq))
    out = levels.reindex(period_of_day)
    out.index = s.index
    return out.reindex(df.set_index(date).index).reset_index(drop=True).set_index(df.index)


def _freq_to_period(freq: str) -> str:
    f = freq.upper()
    if f.startswith("W"):
        return "W"
    if f.startswith("M"):
        return "M"
    if f.startswith("Q"):
        return "Q"
    if f.startswith("Y") or f.startswith("A"):
        return "Y"
    raise ValueError(f"unsupported period frequency: {freq}")


# --- per-symbol assembly ----------------------------------------------------


def add_daily_indicators(g: pd.DataFrame) -> pd.DataFrame:
    """All indicators for ONE symbol's bar history, oldest first."""
    g = g.sort_values("date").reset_index(drop=True)

    # Adjusted basis for anything statistical.
    a = g["adj_close"]
    ah, al = g["adj_high"], g["adj_low"]
    for n in (10, 20, 50, 150, 200):
        g[f"sma_{n}"] = sma(a, n)
    # Weinstein works on the 30-WEEK average. 150 sessions is that average;
    # the 200 is a different line and a different stage call.
    g["sma_150_slope"] = (g["sma_150"] / g["sma_150"].shift(20) - 1) * 100
    g["dist_sma_10"] = (a / g["sma_10"] - 1) * 100
    # 9/21/50 are the chart's EMA set; 200 is kept for the long-trend filter.
    for n in (9, 21, 50, 200):
        g[f"ema_{n}"] = ema(a, n)
    g["rsi_14"] = rsi(a, 14)
    g["atr_14"] = atr(ah, al, a, 14)
    g["atr_pct"] = g["atr_14"] / a * 100
    # The playbook's disaster stop: three ATRs under the close. Tighter stops
    # were taken by normal noise in every test.
    g["stop_3atr"] = a * (1 - 3 * g["atr_pct"] / 100)
    # Momentum swing: a 4 ATR stop with a 1:3 target twelve ATRs up.
    g["stop_4atr"] = a * (1 - 4 * g["atr_pct"] / 100)
    g["tgt_3r_4atr"] = a * (1 + 12 * g["atr_pct"] / 100)
    g = pd.concat([g, bollinger(a, 20, 2.0)], axis=1)

    # Returns and momentum, adjusted basis.
    for n in (1, 5, 20, 60, 120, 250):
        g[f"ret_{n}d"] = a.pct_change(n) * 100
    # Cross-sectional momentum as the literature defines it: twelve months
    # MINUS the most recent one. The skip is not a detail - the last month is
    # where short-term reversal lives, and including it measures the opposite
    # effect from the one the anomaly describes.
    g["mom_12_1"] = (a.shift(22) / a.shift(252) - 1) * 100
    g["high_52w"] = ah.rolling(250, min_periods=60).max()
    g["low_52w"] = al.rolling(250, min_periods=60).min()
    g["pct_from_52w_high"] = (a / g["high_52w"] - 1) * 100
    g["pct_from_52w_low"] = (a / g["low_52w"] - 1) * 100

    # --- event geometry ---------------------------------------------------
    # A setup named for an event has to test the event, not the state it leaves
    # behind. These columns record what happened, not merely where price is.

    # Pullback: how extended was price above the 21 EMA before it came back?
    # Without this a stock glued to its average for a month reads as a pullback.
    d21 = (a / g["ema_21"] - 1) * 100
    g["ext_ema21_15"] = d21.shift(1).rolling(15, min_periods=5).max()
    g["off_high_10"] = (a / ah.rolling(10, min_periods=5).max() - 1) * 100

    # Breakout: the prior high, excluding today, so "new high" means new.
    # Measured on HIGHS. A Donchian channel, a Darvas box and a base's
    # resistance are all drawn at the highest price traded, not the highest
    # close - using closes let half the "breakouts" through below the level
    # they were supposed to be clearing.
    g["hi_20_prior"] = ah.shift(1).rolling(20, min_periods=10).max()
    g["hi_60_prior"] = ah.shift(1).rolling(60, min_periods=30).max()
    g["hi_250_prior"] = ah.shift(1).rolling(250, min_periods=60).max()

    # Where in the day's range the close landed. 1.0 is a close on the high,
    # which is what separates an absorbed gap from a faded one.
    rng = (ah - al).replace(0, np.nan)
    g["close_pos"] = ((a - al) / rng).clip(0, 1)

    # Overnight gap, adjusted basis. An episodic pivot is defined by the gap,
    # not by the day's return - a stock that grinds up 8% is a different event.
    g["gap_pct"] = (g["adj_open"] / a.shift(1) - 1) * 100

    # Pocket pivot (O'Neil / Morales): an up day whose volume exceeds every
    # DOWN day's volume in the prior ten sessions. The comparison is against
    # selling volume specifically, which is the whole point of the signal.
    vol = g["volume"].astype("float64")
    down_vol = vol.where(a < a.shift(1))
    g["down_vol_max_10"] = down_vol.shift(1).rolling(10, min_periods=3).max()
    g["pocket_pivot"] = (
        (a > a.shift(1)) & (vol > g["down_vol_max_10"])
    ).astype("int8")

    # Three weeks tight (Minervini): three consecutive WEEKLY closes within a
    # percent or so of each other. Sampling every fifth session was a proxy for
    # that and a poor one - it drifts off Friday and measures an arbitrary
    # three-day-apart triple. Computed on real weeks, and on COMPLETED weeks
    # only: the current week's close is not known until the week ends, so using
    # it would be lookahead of exactly the kind the repaint rule forbids.
    wk = (g.set_index("date")["adj_close"].resample("W-FRI").last().dropna())
    spread = ((wk.rolling(3).max() - wk.rolling(3).min()) / wk.rolling(3).max() * 100).shift(1)
    per = g["date"].dt.to_period("W-FRI")
    spread.index = spread.index.to_period("W-FRI")
    g["tight_3w_pct"] = spread.reindex(per).to_numpy()

    # Connors' RSI(2), for the mean-reversion screen. A 2-period RSI is a
    # different instrument from RSI(14): it saturates, and that is intended.
    g["rsi_2"] = rsi(a, 2)

    # --- setup geometry -------------------------------------------------
    # Is the long-term trend actually rising, not merely being sat above?
    g["sma_200_slope"] = (g["sma_200"] / g["sma_200"].shift(20) - 1) * 100
    g["sma_50_slope"] = (g["sma_50"] / g["sma_50"].shift(10) - 1) * 100

    # Distance to the short average, for pullback setups. Negative means price
    # has come back under it.
    g["dist_ema_21"] = (a / g["ema_21"] - 1) * 100
    g["dist_sma_200"] = (a / g["sma_200"] - 1) * 100
    g["dist_sma_50"] = (a / g["sma_50"] - 1) * 100

    # Volatility contraction: today's ATR against its own level a quarter ago.
    # Below 1 means the stock is coiling rather than expanding.
    g["atr_ratio_60"] = g["atr_14"] / g["atr_14"].shift(60)

    # Range contraction: the last two weeks' span against the two before it.
    span = ah.rolling(10).max() - al.rolling(10).min()
    g["range_ratio"] = span / span.shift(10)

    # A volatility contraction pattern is a SEQUENCE - two to four pullbacks,
    # each tighter than the one before, volume drying into the pivot. A single
    # ratio of "now" against "then" is not that test: it passes whenever the
    # earlier window happened to be wide, which is how a stock oscillating in a
    # flat 8-11% range for three months read as a contraction.
    #
    # Three consecutive ten-session windows, oldest to newest, each measured as
    # a span in percent of its own price. The pattern requires c < b < a.
    win = (ah.rolling(10).max() - al.rolling(10).min()) / a.rolling(10).mean() * 100
    g["contract_a"] = win.shift(20)
    g["contract_b"] = win.shift(10)
    g["contract_c"] = win
    g["contract_seq"] = ((g["contract_c"] < g["contract_b"])
                         & (g["contract_b"] < g["contract_a"])).astype("int8")
    g["contract_ratio"] = g["contract_c"] / g["contract_a"]
    # How tight the FINAL contraction is. The stop sits under it, so this is
    # the number that decides whether the setup is worth taking at all.
    g["tightness"] = g["contract_c"]

    # Base depth: how far below the recent high price has pulled back. A tight
    # base is shallow; a broken one is not.
    g["base_depth"] = (a / ah.rolling(40, min_periods=20).max() - 1) * 100

    # Dry-up: volume during the base against the prior month's.
    g["vol_dryup"] = g["volume"].rolling(10).mean() / g["volume"].rolling(40).mean()

    # Inside bar - today's range wholly inside yesterday's.
    g["inside_bar"] = ((ah <= ah.shift(1)) & (al >= al.shift(1))).astype("int8")

    # Liquidity, on raw turnover where present, else reconstructed ROW BY ROW.
    #
    # The series-level `if turn.isna().all()` this replaces was a lookahead bug
    # and a silent-exclusion bug at once. The choice of formula depended on
    # whether turnover existed ANYWHERE in the series, including after date t,
    # so the value at t moved when later data arrived. And because the exchange
    # file only covers recent sessions, the reconstruction never fired for the
    # historical panel: turnover stayed NaN, the liquidity gate read NaN as
    # zero, and 598 of 818 backtest sessions were quietly thrown away.
    #
    # Per-row, each date uses the best figure available FOR THAT DATE and
    # nothing else.
    raw_turn = g["turnover"] if "turnover" in g.columns else pd.Series(pd.NA, index=g.index)
    turn = pd.to_numeric(raw_turn, errors="coerce")
    turn = turn.where(turn.notna(), a * g["volume"])
    g["turnover_est"] = turn
    g["turnover_median_20d"] = turn.rolling(20, min_periods=10).median()
    g["vol_sma_20"] = g["volume"].rolling(20, min_periods=10).mean()
    g["vol_ratio"] = g["volume"] / g["vol_sma_20"]

    # Levels are computed on the ADJUSTED series, and only on the adjusted
    # series. The raw block is not continuous: a 5-for-1 split puts last
    # month's raw high five times above this month's raw close, so a raw level
    # is on a different scale from the price it is compared against for every
    # period that spans a corporate action. Krishana Phoschem split on
    # 2026-07-03 and its June pivots printed near 900 against a 180 stock.
    #
    # The adjusted series has no such break, it is entirely on today's basis,
    # and it is what the chart draws - so the level a rule tests, the level the
    # chart plots and the candles under it are all the same number.
    lh, ll, lc = "adj_high", "adj_low", "adj_close"
    g["levels_basis"] = "adjusted"

    src = g[["date", lh, ll, lc]].rename(columns={lh: "high", ll: "low", lc: "close"})
    for freq, prefix in (("W", "w"), ("ME", "m")):
        g = pd.concat([g, period_levels(src, freq, prefix)], axis=1)

    vol_f = pd.to_numeric(g["volume"], errors="coerce").astype("float64")
    g["vol_ema_21"] = vol_f.ewm(span=21, adjust=False).mean().mask(vol_f.notna().cumsum() < 21)
    g["vol_vs_ema21"] = vol_f / g["vol_ema_21"]

    # Three-month base geometry. A "bottom consolidation" is a stock that has
    # gone sideways in a tight range for a quarter AFTER a decline - not one
    # coiling near its highs. Both need the range; only the first is near the
    # lows, so the two are separated by where in the 52-week range it sits.
    base_hi = ah.shift(1).rolling(60, min_periods=40).max()
    base_lo = al.shift(1).rolling(60, min_periods=40).min()
    g["base_range_60"] = (base_hi - base_lo) / a * 100
    g["base_hi_60"] = base_hi
    # How much of the last quarter was spent inside the top half of that range.
    # A real base drifts; a falling knife spends its time at the bottom.
    g["time_in_upper_half"] = (
        (a > (base_hi + base_lo) / 2).rolling(60, min_periods=40).mean() * 100)

    # --- anti-fakeout geometry -------------------------------------------
    # Three different ways to tell a real move from one that is about to trap
    # you. None of them is "a breakout happened"; each tests what happened
    # AFTER, which is the only place the difference shows up.

    # 1. The shakeout that failed (Wyckoff spring / Raschke's Turtle Soup).
    #    Price takes out an obvious 20-day low - where every stop is resting -
    #    and closes back above it the same session. The stop run is the signal,
    #    not the risk: supply below has just been absorbed in public, and the
    #    low it happened on is a floor that has been tested rather than assumed.
    g["lo_20_prior"] = al.shift(1).rolling(20, min_periods=10).min()
    spring = (al < g["lo_20_prior"]) & (a > g["lo_20_prior"]) & (g["close_pos"] > 0.5)
    g["spring"] = spring.astype("int8")
    g["since_spring"] = bars_since(spring)
    g["spring_low"] = al.where(spring).ffill()
    g["dist_spring_low"] = (a / g["spring_low"] - 1) * 100

    # 2. A breakout that SURVIVED time. A fakeout reverses within a session or
    #    three; the tell is that the lowest low since the break is still above
    #    the level, meaning nobody who bought it has been stopped out yet.
    #    cumsum on the cross flag numbers each breakout, so cummin inside that
    #    group is "the worst it has traded since this particular break".
    if "hi_60_prior" in g.columns:
        brk = crossed_above(a, g["hi_60_prior"])
        grp = brk.cumsum()
        g["brk_level"] = g["hi_60_prior"].where(brk).ffill()
        g["low_since_brk"] = al.groupby(grp).cummin()
        g["closes_above_brk"] = (a > g["brk_level"]).groupby(grp).cumsum()
        g["since_brk_60"] = bars_since(brk)
        g["brk_vol"] = (vol_f / g["vol_ema_21"]).where(brk).ffill()
        # Positive means the break has never been given back.
        g["brk_cushion"] = (g["low_since_brk"] / g["brk_level"] - 1) * 100

    # 3. Accumulation, measured as volume asymmetry rather than a single spike.
    #    One heavy day is news; twenty days of buyers outweighing sellers is a
    #    position being built, and that is what does not reverse on you.
    up_day = a > a.shift(1)
    up_v = vol_f.where(up_day, 0.0).rolling(20, min_periods=15).sum()
    dn_v = vol_f.where(~up_day, 0.0).rolling(20, min_periods=15).sum()
    g["ud_vol_20"] = up_v / dn_v.replace(0, np.nan)
    g["close_pos_3"] = g["close_pos"].rolling(3, min_periods=2).mean()

    # How far the invalidation sits in units of this stock's own daily noise.
    # A stop closer than about 1.5 ATR is inside the wiggle and gets taken by
    # accident rather than by the thesis being wrong.
    g["atr_to_spring"] = (a - g["spring_low"]) / g["atr_14"]

    # --- from the literature ----------------------------------------------
    # Connors' Double Seven: a new 7-session closing low is the entry, a new
    # 7-session closing high is the exit. Both windows exclude today, so
    # "new low" means lower than every one of the prior seven closes.
    g["lo7_prior"] = a.shift(1).rolling(7).min()
    g["hi7_prior"] = a.shift(1).rolling(7).max()
    g["new_lo7"] = (a < g["lo7_prior"]).astype("int8")
    g["new_hi7"] = (a > g["hi7_prior"]).astype("int8")
    # Frog in the pan (Da, Gurun, Warachka 2014): the share of up sessions in
    # the momentum window. A smooth path is information arriving gradually,
    # which the market under-reacts to; one jump is information that arrived
    # all at once, which it does not.
    g["smooth_60"] = (a.diff() > 0).rolling(60, min_periods=40).mean() * 100

    # --- confirmation primitives ------------------------------------------
    # The pieces a discretionary trader adds on top of a setup before pulling
    # the trigger: a reversal candle at the low, a pullback on drying volume,
    # the monthly pivot underneath, a structural level to put the stop under.
    o = g["adj_open"]
    body = (a - o).abs()
    rng_ = (ah - al).replace(0, np.nan)
    prev_o, prev_c = o.shift(1), a.shift(1)
    g["lower_wick"] = (np.minimum(o, a) - al) / rng_          # share of range below the body
    g["body_pct"] = body / rng_

    # Bullish engulfing: a red bar, then a green bar whose body covers it.
    g["bull_engulf"] = ((prev_c < prev_o) & (a > o)
                        & (o <= prev_c) & (a >= prev_o)).astype("int8")
    # Hammer / pin bar: long lower wick, close in the top of the range.
    g["hammer"] = ((g["lower_wick"] >= 0.5) & (g["close_pos"] >= 0.6)).astype("int8")
    # Doji: almost no body.
    g["doji"] = (g["body_pct"] <= 0.1).astype("int8")
    # A reversal bar that closes above the prior bar's HIGH - the strongest
    # single-bar confirmation there is.
    g["close_above_prev_high"] = (a > ah.shift(1)).astype("int8")
    g["any_reversal"] = ((g["bull_engulf"] == 1) | (g["hammer"] == 1)
                         | (g["close_above_prev_high"] == 1)).astype("int8")
    # The stop goes under the confirmation candle: its low, in percent.
    g["confirm_low_pct"] = (al / a - 1) * 100
    g["low3_pct"] = (al.rolling(3, min_periods=1).min() / a - 1) * 100

    # Pullback volume dry-up: the last three sessions against the ten before.
    # A healthy pullback is quiet; one on breakout-sized volume is distribution.
    g["pb_vol_dry"] = vol_f.rolling(3).mean() / vol_f.shift(3).rolling(10).mean()
    # RSI turning up off the pullback low.
    g["rsi_turn_up"] = (g["rsi_14"] > g["rsi_14"].shift(1)).astype("int8")
    # Sessions since the 10-day high - how long the pullback has run.
    g["pb_days"] = bars_since(ah >= ah.rolling(10).max())
    # Structure intact: today's low above the prior 20-session low.
    g["higher_low"] = (al > al.shift(1).rolling(20).min()).astype("int8")
    # Position inside the monthly pivot frame.
    if "m_p" in g.columns:
        g["above_m_p"] = (a > g["m_p"]).astype("int8")
    # Two consecutive up closes - momentum has actually turned.
    g["two_up"] = ((a > a.shift(1)) & (a.shift(1) > a.shift(2))).astype("int8")

    # --- volume-confirmed breakout, then a retest -------------------------
    # The breakout and its volume have to be the SAME session. "Crossed R1 at
    # some point" and "had heavy volume at some point" are two facts that can
    # be weeks apart; only their coincidence is a confirmed breakout.
    if "m_r1" in g.columns:
        confirmed = crossed_above(a, g["m_r1"]) & (vol_f > g["vol_ema_21"])
        g["since_r1_vol_breakout"] = bars_since(confirmed)

    # Distance to the 9 EMA, and how close the last three sessions' lows came
    # to it. The shallow cousin of the level retest: a strong breakout often
    # never gives the level back and pulls in to the 9 EMA instead.
    g["dist_ema_9"] = (a / g["ema_9"] - 1) * 100
    g["low3_dist_ema_9"] = (al.rolling(3, min_periods=1).min() / g["ema_9"] - 1) * 100

    # How many sessions since price crossed above each level. A "reclaim" that
    # happened six weeks ago is not a reclaim.
    for lvl in ("m_cpr_tc", "w_cpr_tc", "m_p", "m_r1"):
        if lvl in g.columns:
            g[f"since_{lvl}_cross"] = bars_since(crossed_above(a, g[lvl]))

    # Distance to each level in percent, on the level's own price basis, plus
    # how close the last three sessions' lows came to it. A retest is a touch,
    # not merely a nearby close.
    lbasis, llow = a, al
    for lvl in ("m_cpr_tc", "m_p", "m_r1", "m_r2", "w_cpr_tc"):
        if lvl in g.columns:
            g[f"dist_{lvl}"] = (lbasis / g[lvl] - 1) * 100
            g[f"low3_dist_{lvl}"] = (llow.rolling(3, min_periods=1).min() / g[lvl] - 1) * 100

    # "Narrow" has to be narrow FOR THIS STOCK. A fixed 2% threshold passes a
    # name whose CPR is always 0.4% and permanently excludes one whose CPR is
    # always 3%, so the absolute test is paired with a relative one: where this
    # month's width sits against the last twelve months of its own widths.
    if "m_cpr_width" in g.columns:
        per_m = g["date"].dt.to_period("M")
        monthly = g.groupby(per_m)["m_cpr_width"].first()
        rk = monthly.rolling(12, min_periods=6).apply(
            lambda x: float((x.iloc[:-1] < x.iloc[-1]).mean()), raw=False)
        g["m_cpr_width_rank"] = rk.reindex(per_m).to_numpy()

    # Daily CPR from the previous session, for the intraday view.
    prev = g[["adj_high", "adj_low", "adj_close"]].shift(1)
    prev.columns = ["high", "low", "close"]
    d = cpr(prev["high"], prev["low"], prev["close"])
    for k, v in d.items():
        g[f"d_{k}"] = v
    return g


def build_features(bars: pd.DataFrame, *, min_bars: int = 2) -> pd.DataFrame:
    """Feature frame for every symbol in ``bars``.

    ``min_bars`` is deliberately tiny: a freshly listed stock belongs in the
    universe from day one, and every window-based indicator already returns NaN
    until it has the bars it needs. Filtering the symbol out instead would hide
    a 1-lakh-crore IPO for three months; letting the indicators be NaN hides
    only the numbers that genuinely do not exist yet.
    """
    out = []
    for isin, g in bars.groupby("isin", sort=False):
        if len(g) < min_bars:
            continue
        out.append(add_daily_indicators(g))
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)
