"""Bhavcopy fetch and ingestion.

NSE blocks naked requests: you need browser-like headers and a session cookie
warmed by hitting the homepage first, and the cookie goes stale. ``NSESession``
is a retryable object that owns that dance, so callers never build a raw
``requests.get``.

The pre-July-2024 equity bhavcopy is discontinued — this fetches
``CM-UDiFF Common Bhavcopy Final (zip)``.
"""

from __future__ import annotations

import io
import logging
import random
import time
import zipfile
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path

import pandas as pd
import requests

from . import schema as S
from .store import Store
from .universe import Universe

log = logging.getLogger(__name__)

BASE_URL = "https://www.nseindia.com"
ARCHIVE_URL = "https://nsearchives.nseindia.com"
UDIFF_PATH = "/content/cm/BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip"

# UDiFF carries every CM instrument. Equity series only; the trade-to-trade
# variants are kept so a stock moving EQ -> BE does not leave a hole.
DEFAULT_SERIES = ("EQ", "BE", "BZ")

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


class FetchError(Exception):
    """Network or server-side failure that retrying did not fix."""


class NotPublished(Exception):
    """No bhavcopy exists for this date — a holiday, a weekend, or not out yet."""


class NSESession:
    """A requests session that keeps an NSE cookie warm and retries with backoff."""

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        max_retries: int = 4,
        backoff: float = 1.5,
        cookie_ttl: float = 600.0,
        sleep=time.sleep,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.cookie_ttl = cookie_ttl
        self._sleep = sleep
        self._warmed_at: float | None = None
        self.session = requests.Session()
        self.session.headers.update(BROWSER_HEADERS)

    # --- cookies -----------------------------------------------------------

    @property
    def warm(self) -> bool:
        return (
            self._warmed_at is not None
            and (time.monotonic() - self._warmed_at) < self.cookie_ttl
            and len(self.session.cookies) > 0
        )

    def warm_up(self, *, force: bool = False) -> None:
        """Hit the homepage to pick up the cookies the archive host demands."""
        if self.warm and not force:
            return
        for url in (BASE_URL, f"{BASE_URL}/market-data/securities-available-for-trading"):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                log.debug("warm-up %s -> %s", url, resp.status_code)
            except requests.RequestException as exc:  # pragma: no cover - network
                log.debug("warm-up %s failed: %s", url, exc)
        self._warmed_at = time.monotonic()

    def clear_cookies(self) -> None:
        self.session.cookies.clear()
        self._warmed_at = None

    # --- requests ----------------------------------------------------------

    def get(self, url: str, *, referer: str = BASE_URL, headers: dict | None = None) -> requests.Response:
        """GET with warm cookies, retry/backoff, and a cookie re-warm on a block.

        A 403/401 means the cookie went stale, so it is retried with fresh
        cookies rather than counted as a hard failure. A 404 is raised as
        ``NotPublished`` immediately — retrying a file that does not exist just
        wastes the backoff budget.
        """
        self.warm_up()
        req_headers = {"Referer": referer, "Sec-Fetch-Site": "same-origin", **(headers or {})}
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            if attempt:
                delay = self.backoff ** attempt + random.uniform(0, 0.3)
                log.debug("retry %s in %.1fs (%s)", url, delay, last_error)
                self._sleep(delay)
            try:
                resp = self.session.get(url, headers=req_headers, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                continue

            if resp.status_code == 404:
                raise NotPublished(f"404 for {url}")
            if resp.status_code in (401, 403):
                last_error = FetchError(f"{resp.status_code} for {url} (cookie stale or blocked)")
                self.clear_cookies()
                self.warm_up(force=True)
                continue
            if resp.status_code >= 500 or resp.status_code == 429:
                last_error = FetchError(f"{resp.status_code} for {url}")
                continue
            resp.raise_for_status()
            return resp

        raise FetchError(f"giving up on {url} after {self.max_retries} attempts: {last_error}")


# --- URLs and parsing -------------------------------------------------------


def udiff_url(day) -> str:
    day = pd.Timestamp(day)
    return ARCHIVE_URL + UDIFF_PATH.format(yyyymmdd=day.strftime("%Y%m%d"))


def fetch_bhavcopy(day, session: NSESession | None = None) -> bytes:
    """Download the UDiFF zip for ``day``. Raises NotPublished on a holiday."""
    session = session or NSESession()
    url = udiff_url(day)
    resp = session.get(
        url,
        referer=f"{BASE_URL}/all-reports",
        headers={"Accept": "application/zip,application/octet-stream,*/*"},
    )
    body = resp.content
    if not body:
        raise NotPublished(f"empty body for {url}")
    # NSE serves an HTML error page with a 200 when it dislikes the request.
    if body[:2] != b"PK":
        head = body[:200].decode("utf-8", "replace")
        raise FetchError(f"{url} did not return a zip; got: {head!r}")
    return body


def _read_zip(payload: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise FetchError(f"no CSV inside zip; members: {zf.namelist()}")
        return zf.read(names[0]).decode("utf-8-sig")


# UDiFF -> canonical
_UDIFF_MAP = {
    "TradDt": "date",
    "ISIN": "isin",
    "TckrSymb": "symbol",
    "SctySrs": "series",
    "OpnPric": "open",
    "HghPric": "high",
    "LwPric": "low",
    "ClsPric": "close",
    "PrvsClsgPric": "prev_close",
    "TtlTradgVol": "volume",
    "TtlTrfVal": "turnover",
    "TtlNbOfTxsExctd": "trades",
}


def parse_udiff(payload: bytes | str, *, series=DEFAULT_SERIES) -> pd.DataFrame:
    """Parse a UDiFF bhavcopy (zip bytes or CSV text) into canonical bars."""
    if isinstance(payload, bytes):
        text = _read_zip(payload) if payload[:2] == b"PK" else payload.decode("utf-8-sig")
    else:
        text = payload

    raw = pd.read_csv(io.StringIO(text), dtype=str).rename(columns=lambda c: c.strip())
    missing = [c for c in ("TradDt", "ISIN", "TckrSymb", "ClsPric") if c not in raw.columns]
    if missing:
        raise FetchError(f"not a UDiFF bhavcopy — missing {missing}; got {list(raw.columns)[:12]}")

    # Equities only: UDiFF also carries futures, options, ETFs, debt and index rows.
    if "FinInstrmTp" in raw.columns:
        raw = raw[raw["FinInstrmTp"].str.strip().str.upper().isin({"STK", "EQ"})]
    raw = raw[raw["SctySrs"].str.strip().str.upper().isin({s.upper() for s in series})]
    raw = raw[raw["ISIN"].notna() & (raw["ISIN"].str.strip() != "")]

    df = raw.rename(columns=_UDIFF_MAP)[[c for c in _UDIFF_MAP.values() if c in raw.rename(columns=_UDIFF_MAP).columns]].copy()
    for col in ("isin", "symbol", "series"):
        df[col] = df[col].str.strip()
    df["date"] = _parse_dates(df["date"])
    df["source"] = "bhavcopy_udiff"
    df = S.coerce(df, S.BAR_DTYPES)
    # The bhavcopy is the RAW series and must never populate the adjusted block.
    # Copying close into adj_close here would overwrite the genuine
    # back-adjusted values from Upstox and silently force the implied
    # adjustment factor to 1.0 for every bar.
    for col in S.ADJ_PRICE_COLUMNS:
        df[col] = pd.NA
    df = S.coerce(df, S.BAR_DTYPES)

    # A stock suspended for the day prints zero volume and stale prices; keep it
    # (the gap matters) but drop rows with no usable close at all.
    df = df[df["close"].notna() & (df["close"] > 0)]
    df = df.drop_duplicates(subset=S.BAR_KEY, keep="last")
    return df.sort_values(["symbol"]).reset_index(drop=True)


def _parse_dates(col: pd.Series) -> pd.Series:
    """UDiFF has shipped both YYYY-MM-DD and DD-MM-YYYY."""
    s = col.astype("string").str.strip()
    parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any():
        alt = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
        parsed = parsed.fillna(alt)
    if parsed.isna().any():
        alt = pd.to_datetime(s, errors="coerce")
        parsed = parsed.fillna(alt)
    if parsed.isna().any():
        bad = s[parsed.isna()].head(3).tolist()
        raise FetchError(f"unparseable TradDt values: {bad}")
    return parsed.dt.normalize()


# --- ingestion --------------------------------------------------------------


@dataclass
class IngestResult:
    date: pd.Timestamp
    status: str          # ok | no_data | error | skipped
    rows: int = 0
    message: str = ""
    renames: list = None

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "skipped")

    def __str__(self) -> str:
        return f"{self.date.date()} {self.status} rows={self.rows}" + (
            f" ({self.message})" if self.message else ""
        )


def ingest_date(
    store: Store,
    day,
    *,
    session: NSESession | None = None,
    series=DEFAULT_SERIES,
    force: bool = False,
    raw_dir: str | Path | None = None,
) -> IngestResult:
    """Fetch, parse and upsert one trading day. Safe to re-run.

    Every outcome is written to the ingest log, so a holiday is recorded as
    ``no_data`` and backfill stops retrying it, while an ``error`` stays a gap.
    """
    day = pd.Timestamp(day).normalize()
    source = "bhavcopy_udiff"

    if not force:
        status = store.ingest_status(source)
        if day in status.index and status.loc[day] in ("ok", "no_data"):
            return IngestResult(day, "skipped", 0, f"already {status.loc[day]}")

    try:
        payload = fetch_bhavcopy(day, session=session)
    except NotPublished as exc:
        store.log_ingest(day, source, "no_data", 0, str(exc))
        return IngestResult(day, "no_data", 0, str(exc))
    except (FetchError, requests.RequestException) as exc:
        store.log_ingest(day, source, "error", 0, str(exc))
        return IngestResult(day, "error", 0, str(exc))

    if raw_dir is not None:
        raw_path = Path(raw_dir) / f"BhavCopy_NSE_CM_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(payload)

    try:
        bars = parse_udiff(payload, series=series)
    except Exception as exc:
        store.log_ingest(day, source, "error", 0, f"parse failed: {exc}")
        return IngestResult(day, "error", 0, f"parse failed: {exc}")

    if bars.empty:
        store.log_ingest(day, source, "no_data", 0, "no equity rows in file")
        return IngestResult(day, "no_data", 0, "no equity rows in file")

    # The file's own TradDt is authoritative; a mismatch means NSE served a
    # different day's file, which would silently misdate a whole session.
    file_dates = set(bars["date"].unique())
    if file_dates != {day}:
        msg = f"file reports TradDt {sorted(str(pd.Timestamp(d).date()) for d in file_dates)}, requested {day.date()}"
        store.log_ingest(day, source, "error", 0, msg)
        return IngestResult(day, "error", 0, msg)

    result = store.write_bars(bars)
    renames = Universe(store).observe_symbols(bars[["isin", "symbol", "series", "date"]])
    store.log_ingest(day, source, "ok", result.total, f"+{result.inserted}/~{result.updated}")
    return IngestResult(day, "ok", result.total, f"+{result.inserted}/~{result.updated}", renames)


def expected_sessions(start, end, holidays=None) -> pd.DatetimeIndex:
    """Weekdays in range, minus known holidays.

    A proper NSE trading calendar lands with the quality gates; until then this
    over-estimates on holidays, which is the safe direction — an extra candidate
    date costs one 404, a missed one costs a silent gap.
    """
    days = pd.bdate_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())
    if holidays is not None and len(holidays):
        days = days.difference(pd.DatetimeIndex(pd.to_datetime(list(holidays))).normalize())
    return days


def missing_dates(store: Store, start, end, *, holidays=None, retry_errors: bool = True) -> list[pd.Timestamp]:
    """Sessions in range with no bars stored and no ``no_data`` verdict logged."""
    expected = expected_sessions(start, end, holidays)
    # Only raw-bearing dates count: the adjusted block is filled independently
    # from Upstox and must not mask a missing bhavcopy.
    have = set(store.available_dates(start, end, require="raw"))
    status = store.ingest_status("bhavcopy_udiff")
    known_empty = set(status[status == "no_data"].index)
    errored = set(status[status == "error"].index)

    out = []
    for day in expected:
        if day in have or day in known_empty:
            continue
        if day in errored and not retry_errors:
            continue
        out.append(day)
    return out


def backfill(
    store: Store,
    start,
    end=None,
    *,
    session: NSESession | None = None,
    series=DEFAULT_SERIES,
    holidays=None,
    pause: float = 0.7,
    sleep=time.sleep,
    on_result=None,
) -> list[IngestResult]:
    """Ingest every missing session in range, oldest first.

    Gaps are detected and filled rather than skipped, so a run that dies
    halfway leaves the next run exactly the work that remains.
    """
    end = pd.Timestamp(end or _date.today()).normalize()
    session = session or NSESession()
    todo = missing_dates(store, start, end, holidays=holidays)
    results: list[IngestResult] = []
    for i, day in enumerate(todo):
        if i and pause:
            sleep(pause)
        res = ingest_date(store, day, session=session, series=series)
        results.append(res)
        log.info("%s", res)
        if on_result is not None:
            on_result(res)
    return results
