"""Canonical column schemas.

Every table in the store is keyed on ISIN, never on trading symbol: symbols get
renamed and reused, and a rename silently splits one stock's history into two
series if the symbol is the key.
"""

from __future__ import annotations

import pandas as pd

# --- bars ------------------------------------------------------------------

BAR_KEY = ["isin", "date"]

# A bar carries two parallel price blocks that are never mixed:
#
#   raw (open/high/low/close)         - what a trader saw on the chart that day.
#                                       Pivots and CPR are computed on these.
#                                       Source: NSE UDiFF bhavcopy.
#   adjusted (adj_open/.../adj_close) - back-adjusted for splits and bonuses.
#                                       Moving averages, Bollinger, RSI and ATR
#                                       are computed on these.
#                                       Source: Upstox historical candles.
#
# Adjusted needs its own OHLC, not just a close: ATR is a high/low/close
# calculation, and feeding it raw highs against adjusted closes is the same
# split-factor bug in a different place.
#
# Where both blocks are present for a bar, close / adj_close IS the cumulative
# adjustment factor for that date - which is how the corporate-action stage
# gets an empirical factor curve rather than a modelled one.
BAR_DTYPES: dict[str, str] = {
    "isin": "string",
    "symbol": "string",
    "series": "string",
    "date": "datetime64[ns]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "prev_close": "float64",
    "adj_open": "float64",
    "adj_high": "float64",
    "adj_low": "float64",
    "adj_close": "float64",
    # Nullable: an adjusted-only write must leave a stored bhavcopy volume
    # alone, and 0 is a real value that would win a column-wise merge.
    "volume": "Int64",
    "turnover": "float64",
    "trades": "Int64",
    "source": "string",       # provenance of the raw block
    "adj_source": "string",   # provenance of the adjusted block
}

BAR_COLUMNS = list(BAR_DTYPES)

# Only the key is mandatory: a bar may arrive raw-only (bhavcopy) or
# adjusted-only (Upstox), and the two legs merge into one row on (isin, date).
BAR_REQUIRED = ["isin", "date"]

RAW_PRICE_COLUMNS = ["open", "high", "low", "close", "prev_close"]
ADJ_PRICE_COLUMNS = ["adj_open", "adj_high", "adj_low", "adj_close"]
PRICE_COLUMNS = RAW_PRICE_COLUMNS + ADJ_PRICE_COLUMNS

# --- minute bars -----------------------------------------------------------

# Upstox intraday candles are back-adjusted, so these are the adjusted series.
# Kept in their own table: ~3.9M rows per month for the Nifty 500.
MINUTE_DTYPES: dict[str, str] = {
    "isin": "string",
    "ts": "datetime64[ns]",   # IST wall clock, tz stripped
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "Int64",
    "source": "string",
}
MINUTE_COLUMNS = list(MINUTE_DTYPES)
MINUTE_KEY = ["isin", "ts"]

# --- adjustment factors (populated by stage 3, stored here) ----------------

ADJ_DTYPES: dict[str, str] = {
    "isin": "string",
    "ex_date": "datetime64[ns]",
    "action": "string",     # split | bonus | dividend | rights | consolidation
    "ratio": "float64",     # price multiplier applied to bars BEFORE ex_date
    "purpose": "string",    # raw NSE corporate-action text, kept for audit
}
ADJ_COLUMNS = list(ADJ_DTYPES)
ADJ_KEY = ["isin", "ex_date", "action"]

# --- symbol master ---------------------------------------------------------

SYMBOL_DTYPES: dict[str, str] = {
    "isin": "string",
    "symbol": "string",
    "name": "string",
    "series": "string",
    "first_seen": "datetime64[ns]",
    "last_seen": "datetime64[ns]",
}
SYMBOL_COLUMNS = list(SYMBOL_DTYPES)

# Every (isin, symbol) interval ever observed. to_date is NaT while current.
SYMBOL_HISTORY_DTYPES: dict[str, str] = {
    "isin": "string",
    "symbol": "string",
    "from_date": "datetime64[ns]",
    "to_date": "datetime64[ns]",
}
SYMBOL_HISTORY_COLUMNS = list(SYMBOL_HISTORY_DTYPES)

# --- fundamentals (shares outstanding -> market cap) -----------------------

# Shares outstanding move only on corporate actions, so they are fetched rarely
# and cached. Market cap is derived as shares x our own close rather than taken
# from the vendor, so the cap agrees with every other price on the screen.
FUNDAMENTALS_DTYPES: dict[str, str] = {
    "isin": "string",
    "symbol": "string",
    "shares": "float64",
    "vendor_mcap": "float64",
    "source": "string",
    "fetched_at": "datetime64[ns]",
}
FUNDAMENTALS_COLUMNS = list(FUNDAMENTALS_DTYPES)

# --- index membership ------------------------------------------------------

# Half-open interval [from_date, to_date): a name is a member on date d when
# from_date <= d < to_date. to_date is NaT while the name is still in the index.
MEMBERSHIP_DTYPES: dict[str, str] = {
    "index_name": "string",
    "isin": "string",
    "symbol": "string",
    "name": "string",
    "from_date": "datetime64[ns]",
    "to_date": "datetime64[ns]",
}
MEMBERSHIP_COLUMNS = list(MEMBERSHIP_DTYPES)

# --- ingest log ------------------------------------------------------------

INGEST_LOG_DTYPES: dict[str, str] = {
    "date": "datetime64[ns]",
    "source": "string",
    "status": "string",     # ok | no_data | error
    "rows": "Int64",
    "message": "string",
    "run_at": "datetime64[ns]",
}
INGEST_LOG_COLUMNS = list(INGEST_LOG_DTYPES)
INGEST_LOG_KEY = ["date", "source"]


def empty_frame(dtypes: dict[str, str]) -> pd.DataFrame:
    """An empty DataFrame carrying the right columns and dtypes."""
    return pd.DataFrame({col: pd.Series([], dtype=dt) for col, dt in dtypes.items()})


def coerce(df: pd.DataFrame, dtypes: dict[str, str], *, strict: bool = False) -> pd.DataFrame:
    """Reindex to the schema's columns and cast dtypes.

    Missing columns are added as nulls unless ``strict``. Integer columns are
    filled with 0 rather than cast through NaN, which would fail.
    """
    out = df.copy()
    missing = [c for c in dtypes if c not in out.columns]
    if missing and strict:
        raise ValueError(f"missing columns: {missing}")
    for col in missing:
        out[col] = pd.Series([pd.NA] * len(out), index=out.index)
    out = out[list(dtypes)]
    for col, dt in dtypes.items():
        if dt[0] == "I":  # nullable Int64 — NA must survive, never become 0
            out[col] = pd.to_numeric(out[col], errors="coerce").round().astype(dt)
        elif dt.startswith("int"):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(dt)
        elif dt.startswith("datetime"):
            out[col] = pd.to_datetime(out[col], errors="coerce")
        elif dt == "float64":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        else:
            out[col] = out[col].astype(dt)
    return out
