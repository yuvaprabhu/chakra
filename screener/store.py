"""Parquet store, partitioned by year, read through duckdb.

Layout under ``root``::

    bars/year=2024/bars.parquet     one file per calendar year
    universe/symbols.parquet
    universe/symbol_history.parquet
    universe/membership.parquet
    meta/adjustments.parquet
    meta/ingest_log.parquet

Writes are upserts keyed on the table's primary key, so re-running an ingest is
a no-op rather than a duplicate. Every file is written to a temp path and
renamed, so a crash mid-write cannot leave a half-parquet behind.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from . import schema as S


class StoreError(Exception):
    """Raised when a write would corrupt the store."""


@dataclass(frozen=True)
class WriteResult:
    inserted: int
    updated: int
    years: tuple[int, ...]

    @property
    def total(self) -> int:
        return self.inserted + self.updated


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    df.to_parquet(tmp, index=False, engine="pyarrow", compression="snappy")
    os.replace(tmp, path)


def _upsert(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    key: list[str],
    *,
    column_wise: bool = False,
) -> tuple[pd.DataFrame, int, int]:
    """Merge ``new`` over ``existing`` on ``key``. Returns (frame, inserted, updated).

    With ``column_wise``, a null in ``new`` leaves the stored value alone. That is
    what lets the raw block (bhavcopy) and the adjusted block (Upstox) land in
    the same row from two independent writes without either one erasing the
    other's columns. Without it, the whole row is replaced.
    """
    if existing.empty:
        return new.reset_index(drop=True), len(new), 0
    old_keys = set(map(tuple, existing[key].itertuples(index=False, name=None)))
    new_keys = list(map(tuple, new[key].itertuples(index=False, name=None)))
    updated = sum(1 for k in new_keys if k in old_keys)
    inserted = len(new_keys) - updated

    if not column_wise:
        merged = pd.concat([existing, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=key, keep="last").sort_values(key).reset_index(drop=True)
        return merged, inserted, updated

    left = existing.set_index(key)
    right = new.set_index(key)
    right = right[~right.index.duplicated(keep="last")]
    left = left[~left.index.duplicated(keep="last")]
    # combine_first takes right's non-null values and falls back to left.
    merged = right.combine_first(left)
    merged = merged.reset_index().sort_values(key).reset_index(drop=True)
    return merged, inserted, updated


def _validate_ohlc_block(df: pd.DataFrame, cols: tuple[str, str, str, str], label: str) -> None:
    """Check one OHLC block, ignoring rows where the block is absent."""
    o, h, l, c = cols
    if not all(col in df.columns for col in cols):
        return
    block = df[list(cols)]
    present = block.notna().all(axis=1)
    if not present.any():
        return
    sub = df[present]

    bad_hl = sub[h] < sub[l]
    if bad_hl.any():
        sample = sub.loc[bad_hl, S.BAR_KEY].head(3).to_dict("records")
        raise StoreError(f"{int(bad_hl.sum())} {label} bars with high < low, e.g. {sample}")
    nonpos = (sub[list(cols)] <= 0).any(axis=1)
    if nonpos.any():
        raise StoreError(f"{int(nonpos.sum())} {label} bars with non-positive prices")
    outside = (sub[h] < sub[[o, c]].max(axis=1)) | (sub[l] > sub[[o, c]].min(axis=1))
    if outside.any():
        sample = sub.loc[outside, S.BAR_KEY].head(3).to_dict("records")
        raise StoreError(
            f"{int(outside.sum())} {label} bars where open/close sit outside the "
            f"high/low range, e.g. {sample}"
        )


def validate_bars(df: pd.DataFrame) -> None:
    """Reject anything that would poison downstream indicator math.

    A bar may legitimately carry only the raw block or only the adjusted block,
    so each is validated where it is present rather than demanded outright.
    """
    missing = [c for c in S.BAR_REQUIRED if c not in df.columns]
    if missing:
        raise StoreError(f"bars missing required columns: {missing}")
    if df.empty:
        return
    for col in ("isin", "date"):
        if df[col].isna().any():
            raise StoreError(f"bars have null {col} — key columns must be complete")
    if df["isin"].astype("string").str.len().min() == 0:
        raise StoreError("bars have empty-string isin")
    dupes = df.duplicated(subset=S.BAR_KEY).sum()
    if dupes:
        raise StoreError(f"{dupes} duplicate (isin, date) rows in one write")

    _validate_ohlc_block(df, ("open", "high", "low", "close"), "raw")
    _validate_ohlc_block(df, ("adj_open", "adj_high", "adj_low", "adj_close"), "adjusted")

    if "volume" in df.columns and (df["volume"].fillna(0) < 0).any():
        raise StoreError("bars with negative volume")

    have_raw = df[["open", "high", "low", "close"]].notna().all(axis=1) if "close" in df.columns else pd.Series(False, index=df.index)
    have_adj = df[["adj_open", "adj_high", "adj_low", "adj_close"]].notna().all(axis=1) if "adj_close" in df.columns else pd.Series(False, index=df.index)
    if not (have_raw | have_adj).any():
        raise StoreError("no bar carries a complete raw or adjusted OHLC block")


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.bars_dir = self.root / "bars"
        self.universe_dir = self.root / "universe"
        self.meta_dir = self.root / "meta"
        for d in (self.bars_dir, self.universe_dir, self.meta_dir):
            d.mkdir(parents=True, exist_ok=True)

    # --- paths -------------------------------------------------------------

    def _year_path(self, year: int) -> Path:
        return self.bars_dir / f"year={year}" / "bars.parquet"

    def years(self) -> list[int]:
        return sorted(
            int(p.name.split("=", 1)[1])
            for p in self.bars_dir.glob("year=*")
            if (p / "bars.parquet").exists()
        )

    @property
    def bars_glob(self) -> str:
        return str(self.bars_dir / "year=*" / "bars.parquet")

    @staticmethod
    def _date_predicate(start, end) -> tuple[list[str], list]:
        """Date bounds plus matching bounds on the ``year`` hive key, so duckdb
        skips whole partitions instead of opening every file."""
        where: list[str] = []
        params: list = []
        if start is not None:
            ts = pd.Timestamp(start)
            where.append("date >= ?")
            params.append(ts.to_pydatetime())
            where.append("year >= ?")
            params.append(int(ts.year))
        if end is not None:
            ts = pd.Timestamp(end)
            where.append("date <= ?")
            params.append(ts.to_pydatetime())
            where.append("year <= ?")
            params.append(int(ts.year))
        return where, params

    # --- bars --------------------------------------------------------------

    def write_bars(self, df: pd.DataFrame) -> WriteResult:
        """Upsert bars keyed on (isin, date), routed to the right year partition."""
        if df is None or len(df) == 0:
            return WriteResult(0, 0, ())
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        validate_bars(df)

        supplied = [c for c in df.columns if c in S.BAR_DTYPES]
        df = S.coerce(df, S.BAR_DTYPES)
        # Columns the caller did not supply stay null so they cannot overwrite
        # the other source's block. adj_close is never silently faked from close.
        for col in S.BAR_DTYPES:
            if col not in supplied:
                df[col] = pd.NA
        df = S.coerce(df, S.BAR_DTYPES)

        inserted = updated = 0
        touched: list[int] = []
        for year, chunk in df.groupby(df["date"].dt.year, sort=True):
            path = self._year_path(int(year))
            existing = (
                pd.read_parquet(path, engine="pyarrow")
                if path.exists()
                else S.empty_frame(S.BAR_DTYPES)
            )
            existing = S.coerce(existing, S.BAR_DTYPES)
            merged, ins, upd = _upsert(existing, chunk, S.BAR_KEY, column_wise=True)
            _atomic_write_parquet(S.coerce(merged, S.BAR_DTYPES), path)
            inserted += ins
            updated += upd
            touched.append(int(year))
        return WriteResult(inserted, updated, tuple(touched))

    def read_bars(
        self,
        *,
        start=None,
        end=None,
        isins=None,
        symbols=None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read bars with predicate pushdown across year partitions.

        ``symbols`` filters on the symbol recorded on each bar; for anything
        historical, resolve to ISINs via ``universe`` first — a symbol filter
        will miss the pre-rename half of a renamed stock's history.
        """
        if not self.years():
            return S.empty_frame(S.BAR_DTYPES)[columns or S.BAR_COLUMNS]

        cols = ", ".join(columns) if columns else "*"
        where, params = self._date_predicate(start, end)
        if isins is not None:
            isins = list(isins)
            if not isins:
                return S.empty_frame(S.BAR_DTYPES)[columns or S.BAR_COLUMNS]
            where.append(f"isin IN ({', '.join('?' * len(isins))})")
            params.extend(isins)
        if symbols is not None:
            symbols = list(symbols)
            if not symbols:
                return S.empty_frame(S.BAR_DTYPES)[columns or S.BAR_COLUMNS]
            where.append(f"symbol IN ({', '.join('?' * len(symbols))})")
            params.extend(symbols)

        clause = f"WHERE {' AND '.join(where)}" if where else ""
        sql = (
            f"SELECT {cols} FROM read_parquet(?, union_by_name=true, hive_partitioning=true) "
            f"{clause} ORDER BY isin, date"
        )
        with duckdb.connect() as con:
            out = con.execute(sql, [self.bars_glob, *params]).fetch_df()
        # ``year`` is the hive partition key, used above for pruning only.
        if columns is None and "year" in out.columns:
            out = out.drop(columns=["year"])
        return out

    def available_dates(self, start=None, end=None, *, require: str | None = None) -> pd.DatetimeIndex:
        """Distinct trading dates present in the store.

        ``require`` narrows to dates that actually carry a given price block:
        ``"raw"`` or ``"adjusted"``. Without it, a date backfilled from Upstox
        alone counts as present — which would tell the bhavcopy gap detector
        there is nothing to fetch even though no raw price exists yet.
        """
        if not self.years():
            return pd.DatetimeIndex([], name="date")
        where, params = self._date_predicate(start, end)
        if require == "raw":
            where.append("close IS NOT NULL")
        elif require == "adjusted":
            where.append("adj_close IS NOT NULL")
        elif require is not None:
            raise ValueError(f"require must be 'raw', 'adjusted' or None, got {require!r}")
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        sql = (
            "SELECT DISTINCT date FROM read_parquet(?, union_by_name=true, "
            f"hive_partitioning=true) {clause} ORDER BY date"
        )
        with duckdb.connect() as con:
            out = con.execute(sql, [self.bars_glob, *params]).fetch_df()
        return pd.DatetimeIndex(pd.to_datetime(out["date"]), name="date")

    def bar_counts_by_date(self, start=None, end=None) -> pd.Series:
        """Rows per trading date — the first thing a quality gate looks at."""
        if not self.years():
            return pd.Series([], dtype="int64", name="bars")
        where, params = self._date_predicate(start, end)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        sql = (
            "SELECT date, COUNT(*) AS bars FROM read_parquet(?, union_by_name=true, "
            f"hive_partitioning=true) {clause} GROUP BY date ORDER BY date"
        )
        with duckdb.connect() as con:
            out = con.execute(sql, [self.bars_glob, *params]).fetch_df()
        return pd.Series(
            out["bars"].to_numpy(dtype="int64"),
            index=pd.DatetimeIndex(pd.to_datetime(out["date"]), name="date"),
            name="bars",
        )

    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        """Escape hatch: run SQL against the store.

        Exposes the views ``bars`` and, when present, ``minutes``.
        """
        # duckdb cannot prepare a CREATE VIEW, so the glob is inlined. It is
        # built from the store root, never from caller input; quotes are
        # escaped anyway so a path with an apostrophe cannot break the DDL.
        bars = self.bars_glob.replace("'", "''")
        minutes = self.minutes_glob.replace("'", "''")
        with duckdb.connect() as con:
            con.execute(
                f"CREATE VIEW bars AS SELECT * EXCLUDE (year) FROM "
                f"read_parquet('{bars}', union_by_name=true, hive_partitioning=true)"
            )
            if self.minute_months():
                con.execute(
                    f"CREATE VIEW minutes AS SELECT * EXCLUDE (year, month) FROM "
                    f"read_parquet('{minutes}', union_by_name=true, hive_partitioning=true)"
                )
            return con.execute(sql, params or []).fetch_df()

    # --- minute bars -------------------------------------------------------

    def _month_path(self, year: int, month: int) -> Path:
        return self.minutes_dir / f"year={year}" / f"month={month:02d}" / "minutes.parquet"

    @property
    def minutes_dir(self) -> Path:
        d = self.root / "minutes"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def minutes_glob(self) -> str:
        return str(self.minutes_dir / "year=*" / "month=*" / "minutes.parquet")

    def minute_months(self) -> list[tuple[int, int]]:
        out = []
        for yp in sorted(self.minutes_dir.glob("year=*")):
            for mp in sorted(yp.glob("month=*")):
                if (mp / "minutes.parquet").exists():
                    out.append((int(yp.name.split("=")[1]), int(mp.name.split("=")[1])))
        return out

    def write_minutes(self, df: pd.DataFrame, *, replace_month: bool = False) -> WriteResult:
        """Upsert minute bars keyed on (isin, ts), partitioned by year/month.

        Month partitions run to millions of rows, so callers should batch a whole
        month in one call rather than writing symbol by symbol — each call
        rewrites the partitions it touches.
        """
        if df is None or len(df) == 0:
            return WriteResult(0, 0, ())
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"])
        if df["isin"].isna().any() or df["ts"].isna().any():
            raise StoreError("minute bars have null isin or ts")
        df = S.coerce(df, S.MINUTE_DTYPES)
        df = df.drop_duplicates(subset=S.MINUTE_KEY, keep="last")

        inserted = updated = 0
        touched: list[int] = []
        for (year, month), chunk in df.groupby([df["ts"].dt.year, df["ts"].dt.month], sort=True):
            path = self._month_path(int(year), int(month))
            if replace_month or not path.exists():
                merged, ins, upd = chunk.reset_index(drop=True), len(chunk), 0
            else:
                existing = S.coerce(pd.read_parquet(path, engine="pyarrow"), S.MINUTE_DTYPES)
                merged, ins, upd = _upsert(existing, chunk, S.MINUTE_KEY)
            _atomic_write_parquet(S.coerce(merged, S.MINUTE_DTYPES), path)
            inserted += ins
            updated += upd
            touched.append(int(year))
        return WriteResult(inserted, updated, tuple(sorted(set(touched))))

    def read_minutes(self, *, start=None, end=None, isins=None, columns=None) -> pd.DataFrame:
        if not self.minute_months():
            return S.empty_frame(S.MINUTE_DTYPES)[columns or S.MINUTE_COLUMNS]
        cols = ", ".join(columns) if columns else "* EXCLUDE (year, month)"
        where, params = [], []
        if start is not None:
            ts = pd.Timestamp(start)
            where.append("ts >= ?")
            params.append(ts.to_pydatetime())
        if end is not None:
            ts = pd.Timestamp(end)
            where.append("ts <= ?")
            params.append(ts.to_pydatetime())
        if isins is not None:
            isins = list(isins)
            if not isins:
                return S.empty_frame(S.MINUTE_DTYPES)[columns or S.MINUTE_COLUMNS]
            where.append(f"isin IN ({', '.join('?' * len(isins))})")
            params.extend(isins)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        sql = (
            f"SELECT {cols} FROM read_parquet(?, union_by_name=true, hive_partitioning=true) "
            f"{clause} ORDER BY isin, ts"
        )
        with duckdb.connect() as con:
            return con.execute(sql, [self.minutes_glob, *params]).fetch_df()

    def minute_summary(self) -> pd.DataFrame:
        """Rows, symbols and time span per month partition."""
        if not self.minute_months():
            return pd.DataFrame(columns=["year", "month", "rows", "symbols", "first_ts", "last_ts"])
        sql = (
            "SELECT year, month, COUNT(*) AS rows, COUNT(DISTINCT isin) AS symbols, "
            "MIN(ts) AS first_ts, MAX(ts) AS last_ts "
            "FROM read_parquet(?, union_by_name=true, hive_partitioning=true) "
            "GROUP BY year, month ORDER BY year, month"
        )
        with duckdb.connect() as con:
            return con.execute(sql, [self.minutes_glob]).fetch_df()

    # --- generic table helpers --------------------------------------------

    def _read_table(self, path: Path, dtypes: dict[str, str]) -> pd.DataFrame:
        if not path.exists():
            return S.empty_frame(dtypes)
        return S.coerce(pd.read_parquet(path, engine="pyarrow"), dtypes)

    def _write_table(self, path: Path, df: pd.DataFrame, dtypes: dict[str, str]) -> None:
        _atomic_write_parquet(S.coerce(df, dtypes), path)

    # --- adjustment factors ------------------------------------------------

    @property
    def adjustments_path(self) -> Path:
        return self.meta_dir / "adjustments.parquet"

    def read_adjustments(self) -> pd.DataFrame:
        return self._read_table(self.adjustments_path, S.ADJ_DTYPES)

    def write_adjustments(self, df: pd.DataFrame, *, replace: bool = False) -> WriteResult:
        """Upsert corporate actions. ``replace=True`` rebuilds the table from scratch,
        which is the point of keeping factors separate from prices."""
        new = S.coerce(df, S.ADJ_DTYPES)
        existing = S.empty_frame(S.ADJ_DTYPES) if replace else self.read_adjustments()
        merged, ins, upd = _upsert(existing, new, S.ADJ_KEY)
        self._write_table(self.adjustments_path, merged, S.ADJ_DTYPES)
        return WriteResult(ins, upd, ())

    # --- ingest log --------------------------------------------------------

    @property
    def ingest_log_path(self) -> Path:
        return self.meta_dir / "ingest_log.parquet"

    def read_ingest_log(self) -> pd.DataFrame:
        return self._read_table(self.ingest_log_path, S.INGEST_LOG_DTYPES)

    def log_ingest(self, date, source: str, status: str, rows: int, message: str = "") -> None:
        """Record the outcome of one (date, source) ingest attempt.

        This is what makes backfill cheap on re-runs: a date already logged as
        ``no_data`` is a known holiday, not a gap to keep retrying forever.
        """
        row = pd.DataFrame(
            [{
                "date": pd.Timestamp(date).normalize(),
                "source": source,
                "status": status,
                "rows": int(rows),
                "message": str(message)[:500],
                "run_at": pd.Timestamp.now("UTC").tz_localize(None),
            }]
        )
        merged, _, _ = _upsert(self.read_ingest_log(), S.coerce(row, S.INGEST_LOG_DTYPES), S.INGEST_LOG_KEY)
        self._write_table(self.ingest_log_path, merged, S.INGEST_LOG_DTYPES)

    def ingest_status(self, source: str | None = None) -> pd.Series:
        """date -> status, for gap detection."""
        log = self.read_ingest_log()
        if source is not None:
            log = log[log["source"] == source]
        if log.empty:
            return pd.Series([], dtype="string", name="status")
        log = log.sort_values("run_at").drop_duplicates(subset=["date"], keep="last")
        return pd.Series(
            log["status"].to_numpy(),
            index=pd.DatetimeIndex(log["date"], name="date"),
            name="status",
        ).astype("string").sort_index()

    @property
    def fundamentals_path(self) -> Path:
        return self.meta_dir / "fundamentals.parquet"

    def read_fundamentals(self) -> pd.DataFrame:
        return self._read_table(self.fundamentals_path, S.FUNDAMENTALS_DTYPES)

    def write_fundamentals(self, df: pd.DataFrame) -> WriteResult:
        new = S.coerce(df, S.FUNDAMENTALS_DTYPES)
        merged, ins, upd = _upsert(self.read_fundamentals(), new, ["isin"])
        self._write_table(self.fundamentals_path, merged, S.FUNDAMENTALS_DTYPES)
        return WriteResult(ins, upd, ())

    # --- quarantine --------------------------------------------------------

    @property
    def quarantine_path(self) -> Path:
        return self.meta_dir / "quarantine.parquet"

    def read_quarantine(self) -> pd.DataFrame:
        if not self.quarantine_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.quarantine_path, engine="pyarrow")

    def quarantine(self, df: pd.DataFrame, reason: str, source: str) -> int:
        """Park rows the store refused, with why, instead of dropping them silently.

        A screener that quietly discards bad bars is indistinguishable from one
        that never saw them. This keeps the evidence queryable.
        """
        if df is None or len(df) == 0:
            return 0
        rows = df.copy()
        rows["quarantine_reason"] = reason
        rows["quarantine_source"] = source
        rows["quarantined_at"] = pd.Timestamp.now("UTC").tz_localize(None)
        for col in rows.columns:
            if str(rows[col].dtype) == "object":
                rows[col] = rows[col].astype("string")
        existing = self.read_quarantine()
        merged = pd.concat([existing, rows], ignore_index=True) if len(existing) else rows
        _atomic_write_parquet(merged, self.quarantine_path)
        return len(rows)

    # --- universe tables (written by universe.py) --------------------------

    @property
    def symbols_path(self) -> Path:
        return self.universe_dir / "symbols.parquet"

    @property
    def symbol_history_path(self) -> Path:
        return self.universe_dir / "symbol_history.parquet"

    @property
    def membership_path(self) -> Path:
        return self.universe_dir / "membership.parquet"

    def read_symbols(self) -> pd.DataFrame:
        return self._read_table(self.symbols_path, S.SYMBOL_DTYPES)

    def write_symbols(self, df: pd.DataFrame) -> None:
        self._write_table(self.symbols_path, df, S.SYMBOL_DTYPES)

    def read_symbol_history(self) -> pd.DataFrame:
        return self._read_table(self.symbol_history_path, S.SYMBOL_HISTORY_DTYPES)

    def write_symbol_history(self, df: pd.DataFrame) -> None:
        self._write_table(self.symbol_history_path, df, S.SYMBOL_HISTORY_DTYPES)

    def read_membership(self) -> pd.DataFrame:
        return self._read_table(self.membership_path, S.MEMBERSHIP_DTYPES)

    def write_membership(self, df: pd.DataFrame) -> None:
        self._write_table(self.membership_path, df, S.MEMBERSHIP_DTYPES)

    # --- summary -----------------------------------------------------------

    def summary(self) -> dict:
        years = self.years()
        dates = self.available_dates()
        members = self.read_membership()
        with duckdb.connect() as con:
            if years:
                n_rows, n_isin = con.execute(
                    "SELECT COUNT(*), COUNT(DISTINCT isin) FROM read_parquet(?, union_by_name=true)",
                    [self.bars_glob],
                ).fetchone()
            else:
                n_rows, n_isin = 0, 0
        return {
            "root": str(self.root),
            "years": years,
            "bars": int(n_rows),
            "symbols": int(n_isin),
            "first_date": dates.min() if len(dates) else None,
            "last_date": dates.max() if len(dates) else None,
            "trading_days": len(dates),
            "indices": sorted(members["index_name"].dropna().unique().tolist()) if not members.empty else [],
            "adjustments": len(self.read_adjustments()),
        }
