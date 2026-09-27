"""Forward-return harness.

**It runs the same rule expressions the live screener runs, against the same
feature frame.** That is the whole point: a backtest built on a re-implementation
of the indicators tells you about the re-implementation.

What it measures, for every session in history and every rule:

    hit at date t  ->  adj_close[t+k] / adj_close[t] - 1   for k in 1, 5, 20

and the same number for the equal-weight universe on that date. The difference
is the excess return, and it is the only figure worth reading. A screen that
returns +3% in a month when everything returned +3% has found nothing.

Three things this cannot tell you, stated here rather than in a footnote:

* **Hits overlap.** The same stock matching on Monday and Tuesday is nearly the
  same observation twice, so the sample is smaller than the hit count suggests
  and the win rate is not a binomial draw. Treat n as an upper bound.
* **No costs.** No brokerage, no STT, no slippage, no impact. On a mid-cap at
  the liquidity floor, a round trip is not free.
* **Survivorship.** The universe is today's listed names above the market-cap
  floor. Anything delisted or fallen below it is absent, which flatters every
  long screen equally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rules import Rule, liquidity_mask

HORIZONS = (1, 5, 20)


def add_panel_columns(feat: pd.DataFrame) -> pd.DataFrame:
    """Add the columns the rules need that are cross-sectional or as-of.

    ``rs_rank`` is a percentile against the other stocks *on that date*, which
    is what makes it relative strength rather than absolute return. Computing
    it once on the latest snapshot and reusing it across history would leak
    today's ranking into 2023.
    """
    df = feat.sort_values(["isin", "date"]).copy()

    # Bars available AS OF each date, not in total.
    df["bars_available"] = df.groupby("isin").cumcount() + 1

    for col in ("ret_60d", "ret_120d", "ret_250d"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    by_date = df.groupby("date")
    score = (0.4 * by_date["ret_60d"].rank(pct=True)
             + 0.2 * by_date["ret_120d"].rank(pct=True)
             + 0.4 * by_date["ret_250d"].rank(pct=True))
    df["rs_score"] = score
    df["rs_rank"] = (df.groupby("date")["rs_score"].rank(pct=True) * 99).round(0)
    return df


def add_forward_returns(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Forward returns in percent, and the equal-weight universe's, per date.

    The benchmark is the cross-sectional mean forward return of every name in
    the universe on that date - an equal-weight index of exactly the stocks the
    screen could have chosen from. Comparing against a cap-weighted index would
    mostly measure the size factor instead of the screen.
    """
    df = df.sort_values(["isin", "date"]).copy()
    g = df.groupby("isin")["adj_close"]
    for k in horizons:
        df[f"fwd_{k}d"] = (g.shift(-k) / df["adj_close"] - 1) * 100
        bench = df.groupby("date")[f"fwd_{k}d"].transform("mean")
        df[f"bench_{k}d"] = bench
        df[f"exc_{k}d"] = df[f"fwd_{k}d"] - bench
    return df


def rule_hits(df: pd.DataFrame, rule: Rule) -> pd.Series:
    """Boolean mask of every (symbol, date) this rule fired on."""
    eligible = liquidity_mask(df, rule)
    if "bars_available" in df.columns:
        eligible &= df["bars_available"].fillna(0) >= rule.min_history
    if "contaminated" in df.columns:
        eligible &= ~df["contaminated"].fillna(False).astype(bool)
    matched = df.eval(rule.expr)
    if not isinstance(matched, pd.Series) or matched.dtype != bool:
        raise ValueError(f"rule {rule.name}: expression is not boolean")
    return eligible & matched.fillna(False)


def _stats(exc: pd.Series, raw: pd.Series) -> dict:
    exc = exc.dropna()
    if len(exc) < 5:
        return {"n": int(len(exc))}
    return {
        "n": int(len(exc)),
        "win": round(float((exc > 0).mean() * 100), 1),
        "mean": round(float(exc.mean()), 2),
        "median": round(float(exc.median()), 2),
        "p25": round(float(exc.quantile(0.25)), 2),
        "p75": round(float(exc.quantile(0.75)), 2),
        "std": round(float(exc.std()), 2),
        # Raw return alongside the excess, so a screen that made money in a
        # rising market cannot be confused with one that beat the market.
        "raw_mean": round(float(raw.dropna().mean()), 2) if raw.notna().any() else None,
        "raw_win": round(float((raw.dropna() > 0).mean() * 100), 1) if raw.notna().any() else None,
    }


BINS = [-np.inf, -15, -10, -6, -3, -1, 0, 1, 3, 6, 10, 15, np.inf]


def _histogram(exc: pd.Series) -> list[int]:
    v = exc.dropna()
    if v.empty:
        return [0] * (len(BINS) - 1)
    return [int(x) for x in np.histogram(v.clip(-40, 40), bins=BINS)[0]]


def concentration(hits: pd.DataFrame, col: str = "exc_20d", top: int = 10) -> dict:
    """How much of a screen's result comes from a handful of names.

    A mean is only a useful summary when the distribution behind it is not
    dominated by a few observations. Episodic Pivot returns +2.57% average
    excess - and ten of its 258 names produce 80% of that, while the median
    trade in 2026 lost money. Ranked by mean it sits top of the table; what it
    actually has is a lottery-ticket payoff, which is a different instrument
    from an edge and should not be presented as the same thing.
    """
    v = hits[[c for c in ("isin", col) if c in hits.columns]].dropna()
    if len(v) < 20 or "isin" not in v.columns:
        return {}
    by_name = v.groupby("isin")[col].sum().sort_values(ascending=False)
    # Normalise against the GROSS POSITIVE contribution, not the net sum. The
    # net is winners minus losers and can be near zero or negative, which makes
    # a "share of total" ratio explode past 100% or flip sign - it reported
    # 798% for one screen before this was fixed. Gross positive is bounded and
    # comparable across screens.
    gross_up = float(by_name[by_name > 0].sum())
    if gross_up <= 0:
        return {"names": int(v["isin"].nunique()), "top10_share": None,
                "median": round(float(v[col].median()), 2)}
    return {
        "names": int(v["isin"].nunique()),
        "top10_share": round(float(by_name.head(top).clip(lower=0).sum()) / gross_up * 100, 1),
        "winners": int((by_name > 0).sum()),
        "median": round(float(v[col].median()), 2),
    }


def backtest_rule(df: pd.DataFrame, rule: Rule, horizons=HORIZONS) -> dict:
    hits = df[rule_hits(df, rule)]
    sessions = df["date"].nunique()
    out = {
        "name": rule.name,
        "title": rule.title or rule.name,
        "hits": int(len(hits)),
        "names": int(hits["isin"].nunique()) if len(hits) else 0,
        "per_session": round(len(hits) / sessions, 1) if sessions else 0,
        "horizons": {},
        "by_cap": {},
        "by_year": {},
        "hist": _histogram(hits["exc_20d"]) if len(hits) else [0] * (len(BINS) - 1),
        "conc": concentration(hits) if len(hits) else {},
    }
    for k in horizons:
        out["horizons"][str(k)] = _stats(hits[f"exc_{k}d"], hits[f"fwd_{k}d"])

    if len(hits) and "cap_band" in hits.columns:
        for band, grp in hits.groupby("cap_band", observed=True):
            out["by_cap"][str(band)] = {
                str(k): _stats(grp[f"exc_{k}d"], grp[f"fwd_{k}d"]) for k in horizons
            }
    if len(hits):
        for year, grp in hits.groupby(hits["date"].dt.year):
            out["by_year"][str(year)] = {
                str(k): _stats(grp[f"exc_{k}d"], grp[f"fwd_{k}d"]) for k in horizons
            }
        yrs = [v.get("20", {}).get("mean") for v in out["by_year"].values()]
        yrs = [y for y in yrs if y is not None]
        out["years_positive"] = int(sum(1 for y in yrs if y > 0))
        out["years_total"] = int(len(yrs))
    return out


def run(df: pd.DataFrame, rules: list[Rule], horizons=HORIZONS) -> dict:
    """Backtest every rule. ``df`` must already carry panel and forward columns."""
    rows = [backtest_rule(df, r, horizons) for r in rules]
    return {
        "bins": [None if np.isinf(b) else float(b) for b in BINS],
        "sessions": int(df["date"].nunique()),
        "universe": int(df["isin"].nunique()),
        "start": str(df["date"].min().date()),
        "end": str(df["date"].max().date()),
        "horizons": [str(k) for k in horizons],
        "rules": rows,
    }


# --- bracket exits: a real trade, with a stop and a target --------------------
#
# The fixed-holding study above answers "does this screen pick stocks that go
# up". It does not answer "can this be traded", because nobody holds a losing
# position for twenty sessions without a stop. This section runs each hit as an
# actual bracket order and reports what the trade did.
#
# Four decisions here decide whether the numbers are honest:
#
# 1. **Entry is the NEXT session's open.** The screen runs on the close, so the
#    close it fired on is not a price you could have paid. Using it is the most
#    common way a backtest invents returns that were never available.
# 2. **When a bar touches BOTH the stop and the target, the stop wins.** Daily
#    bars record the high and the low but not their order. Assuming the target
#    came first is the single largest source of fake edge in bracket backtests,
#    and on a wide bar it is a coin flip being called in your favour every time.
# 3. **Gaps fill at the open, not at the level.** If a bar opens below the stop
#    the fill is the open. A stop is an instruction, not a guarantee, and
#    pretending otherwise removes exactly the losses that hurt most.
# 4. **Unresolved trades are closed at the time stop**, at that bar's close, and
#    counted separately - they are neither wins nor losses.

def bracket_returns(
    df: pd.DataFrame,
    hit_mask: pd.Series,
    *,
    risk: str = "atr",
    risk_pct: float = 3.0,
    rr: float = 2.0,
    max_bars: int = 20,
    atr_mult: float = 1.0,
    stop_price_col: str | None = None,
) -> pd.DataFrame:
    """Simulate a stop/target bracket for every hit.

    ``risk='atr'`` sizes the stop at ``atr_mult`` x ATR(14) as a percentage of
    the entry - a 0.8% ATR name and a 6% ATR name should not be given the same
    stop. ``risk='pct'`` uses a flat ``risk_pct`` for every trade. The target is
    always ``rr`` times the stop distance, so rr=2 is the 1:2 the trade is
    usually described by.
    """
    d = df.sort_values(["isin", "date"]).reset_index(drop=True)
    hit = hit_mask.reindex(df.index).fillna(False).to_numpy()
    hit = pd.Series(hit, index=df.index).reindex(
        df.sort_values(["isin", "date"]).index).to_numpy()

    o = d["adj_open"].to_numpy(float)
    h = d["adj_high"].to_numpy(float)
    lo = d["adj_low"].to_numpy(float)
    c = d["adj_close"].to_numpy(float)

    # Last row index belonging to each symbol, so a window never walks into the
    # next stock's prices.
    codes = pd.factorize(d["isin"])[0]
    last = pd.Series(np.arange(len(d))).groupby(codes).transform("max").to_numpy()

    idx = np.flatnonzero(hit)
    idx = idx[idx + 1 <= last[idx]]              # need at least one bar to enter
    if not len(idx):
        return pd.DataFrame()

    entry = o[idx + 1]
    if risk == "atr":
        r_pct = (d["atr_14"].to_numpy(float)[idx] / c[idx] * 100) * atr_mult
    elif risk == "price":
        # A STRUCTURAL stop: an absolute price per signal row - the low of the
        # confirmation candle, the pivot the trade was built on - rather than
        # a fixed distance. The risk is whatever the structure makes it.
        sp = d[stop_price_col].to_numpy(float)[idx]
        r_pct = (entry - sp) / entry * 100
    else:
        r_pct = np.full(len(idx), float(risk_pct))
    ok = np.isfinite(entry) & np.isfinite(r_pct) & (r_pct > 0.2) & (r_pct < 20) & (entry > 0)
    idx, entry, r_pct = idx[ok], entry[ok], r_pct[ok]
    if not len(idx):
        return pd.DataFrame()

    stop = entry * (1 - r_pct / 100)
    target = entry * (1 + rr * r_pct / 100)

    n, N = len(idx), max_bars
    win = idx[:, None] + 1 + np.arange(N)[None, :]
    valid = win <= last[idx][:, None]
    safe = np.where(valid, win, 0)

    H, L, O, C = h[safe], lo[safe], o[safe], c[safe]
    H = np.where(valid, H, -np.inf)
    L = np.where(valid, L, np.inf)

    hit_stop = L <= stop[:, None]
    hit_tgt = H >= target[:, None]
    BIG = N + 10
    first_stop = np.where(hit_stop.any(1), hit_stop.argmax(1), BIG)
    first_tgt = np.where(hit_tgt.any(1), hit_tgt.argmax(1), BIG)

    # Ties go to the stop: a daily bar does not say which came first. The
    # `< BIG` guard is load-bearing - without it a trade that touched NEITHER
    # level has first_stop == first_tgt == BIG, satisfies `<=`, and is booked
    # as a full stop-out. That alone turned every screen's expectancy negative.
    stopped = (first_stop <= first_tgt) & (first_stop < BIG)
    won = first_tgt < first_stop
    resolved = stopped | won
    exit_bar = np.where(stopped, first_stop, np.where(won, first_tgt, BIG))

    rows = np.arange(n)
    last_valid = valid.sum(1) - 1
    tb = np.where(resolved, np.minimum(exit_bar, N - 1), np.maximum(last_valid, 0))

    px = np.empty(n)
    # A gapped stop fills at the open, below the level. Same, inverted, for a
    # gapped target - which is why a gap is not symmetric good news.
    px[stopped] = np.minimum(stop[stopped], O[rows[stopped], tb[stopped]])
    px[won] = np.maximum(target[won], O[rows[won], tb[won]])
    unres = ~resolved
    px[unres] = C[rows[unres], tb[unres]]

    ret = (px / entry - 1) * 100
    return pd.DataFrame({
        "row": idx,
        "date": d["date"].to_numpy()[idx],
        "isin": d["isin"].to_numpy()[idx],
        "entry": entry,
        "exit": px,
        "risk_pct": r_pct,
        "ret": ret,
        "r_multiple": ret / r_pct,
        "bars_held": tb + 1,
        "outcome": np.where(stopped, "stop", np.where(won, "target", "time")),
    })


def bracket_stats(t: pd.DataFrame) -> dict:
    """Summarise a set of bracket trades the way a trader reads a system."""
    if t is None or t.empty:
        return {"n": 0}
    n = len(t)
    wins = t[t["ret"] > 0]["ret"]
    losses = t[t["ret"] <= 0]["ret"]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    counts = t["outcome"].value_counts()
    return {
        "n": int(n),
        "target_pct": round(float(counts.get("target", 0) / n * 100), 1),
        "stop_pct": round(float(counts.get("stop", 0) / n * 100), 1),
        "time_pct": round(float(counts.get("time", 0) / n * 100), 1),
        "win": round(float((t["ret"] > 0).mean() * 100), 1),
        "avg": round(float(t["ret"].mean()), 2),
        "avg_win": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else 0.0,
        # Expectancy in R: what one unit of risk returns on average. This is the
        # number that decides whether a system is worth trading, not win rate.
        "expectancy_r": round(float(t["r_multiple"].mean()), 3),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "avg_bars": round(float(t["bars_held"].mean()), 1),
        "avg_risk": round(float(t["risk_pct"].mean()), 2),
    }

# --- reversal exit -----------------------------------------------------------

REV_COST = 0.30
REV_CAP = 20


def reversal_trades(df: pd.DataFrame, mask, *, cost: float = REV_COST,
                    max_bars: int = REV_CAP) -> pd.DataFrame:
    """Trades exited on the first close above the previous seven closes.

    The bracket and the fixed hold both cut a trade at a distance or a date
    that has nothing to do with the stock. This exit lets the trade run until
    the move it was bought for has happened - a fresh short-term high - which
    is the only exit that kept an edge on this universe. Entry is the next
    open, one open position per name at a time, and ``cost`` is the full round
    trip. ``max_bars`` is the time stop for a trade whose high never comes.

    Returns one row per trade: date, isin, entry, exit, net, bars, why.
    """
    need = {"isin", "date", "adj_open", "adj_close", "hi7_prior"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"reversal_trades needs {sorted(missing)}")
    order = df.sort_values(["isin", "date"]).index
    d = df.loc[order].reset_index(drop=True)
    # The mask is the CALLER's row order. Re-order it the same way rather than
    # trusting the two to coincide, which silently returns nothing when they do
    # not (and every caller happens to pre-sort, which is how that stays hidden).
    m = mask if isinstance(mask, pd.Series) else pd.Series(np.asarray(mask, dtype=bool), index=df.index)
    if len(m) != len(d):
        raise ValueError("mask length does not match the frame")
    sig = m.loc[order].to_numpy(dtype=bool)
    o = d["adj_open"].to_numpy(float)
    c = d["adj_close"].to_numpy(float)
    h7 = d["hi7_prior"].to_numpy(float)
    isin = d["isin"].to_numpy()
    dates = d["date"].to_numpy()
    last = pd.Series(np.arange(len(d))).groupby(pd.factorize(d["isin"])[0]).transform("max").to_numpy()

    rows = []
    busy: dict = {}
    for t in np.flatnonzero(sig):
        if t + 1 > last[t] or busy.get(isin[t], -1) >= t:
            continue
        entry = o[t + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        k = t + 1
        out = None
        why = "time"
        while k <= min(last[t], t + max_bars):
            if np.isfinite(h7[k]) and c[k] > h7[k]:
                if k + 1 <= last[t]:
                    out = o[k + 1]; k += 1          # filled at the next open
                else:
                    out = c[k]                      # no next bar for this symbol
                why = "reversal"
                break
            k += 1
        if out is None:
            # The time stop is max_bars sessions AFTER ENTRY, so a 20-bar stop
            # exits on the twentieth held session, not the twenty-first.
            k = min(t + max_bars, last[t])
            out = c[k]
        if not np.isfinite(out):
            continue
        busy[isin[t]] = k
        rows.append((dates[t], isin[t], entry, out, (out / entry - 1) * 100 - cost, k - t, why))
    return pd.DataFrame(rows, columns=["date", "isin", "entry", "exit", "net", "bars", "why"])


def reversal_stats(t: pd.DataFrame) -> dict:
    """Headline numbers for a reversal-exit trade list."""
    if not len(t):
        return {"n": 0}
    net = t["net"]
    win, loss = net[net > 0], net[net <= 0]
    mo = net.groupby(pd.to_datetime(t["date"]).dt.to_period("M")).mean()
    return {
        "n": int(len(t)),
        "net": float(net.mean()),
        "win": float((net > 0).mean() * 100),
        "avg_win": float(win.mean()) if len(win) else 0.0,
        "avg_loss": float(loss.mean()) if len(loss) else 0.0,
        "payoff": float(win.mean() / -loss.mean()) if len(loss) and loss.mean() < 0 else 0.0,
        "profit_factor": float(win.sum() / -loss.sum()) if len(loss) and loss.sum() < 0 else 0.0,
        "avg_bars": float(t["bars"].mean()),
        "on_reversal": float((t["why"] == "reversal").mean() * 100),
        "big_loss": float((net < -8).mean() * 100),
        "months_up": float((mo > 0).mean() * 100) if len(mo) else 0.0,
        "worst_month": float(mo.min()) if len(mo) else 0.0,
    }
