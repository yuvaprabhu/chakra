"""Upstox historical-candle provider.

Free, unauthenticated, and keyed on ISIN (``NSE_EQ|INE002A01018``), which is the
key this project uses everywhere. Supplies:

  * daily candles back to 2000, written to the ``adj_*`` block
  * 1-minute candles back to Jan 2022, written to the minute table

**These prices are back-adjusted for corporate actions.** Verified against
Reliance's 1:1 bonus (ex-date 2024-10-28): a raw series must show a ~50% gap
there and this one does not. So it fills the adjusted block only — raw prices
come from the NSE bhavcopy in ``fetch.py``.

API shape: ``{base}/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}``
(to-date first). Candles arrive newest-first as
``[ts, open, high, low, close, volume, open_interest]``.

Observed limits: 10 years per daily request, 1 month per minute request, no
rate limiting at ~10 req/s. A default ``User-Agent`` such as urllib's is
rejected with 403, so one is always set.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from urllib.parse import quote
import requests

from . import schema as S
from .store import Store

log = logging.getLogger(__name__)

API_BASE = "https://api.upstox.com/v3/historical-candle"
INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"

# Anything that is not the stdlib default works; Upstox 403s ``Python-urllib``.
USER_AGENT = "nse-screener/0.1 (+historical EOD research)"

MINUTE_HISTORY_START = pd.Timestamp("2022-01-01")
DAILY_HISTORY_START = pd.Timestamp("2000-01-01")
SESSION_MINUTES = 375  # 09:15 -> 15:29 inclusive


class UpstoxError(Exception):
    pass


def instrument_key(isin: str) -> str:
    return f"NSE_EQ|{isin}"


# --- instrument master ------------------------------------------------------


def load_instruments(path: str | Path | None = None, *, download: bool = False) -> pd.DataFrame:
    """NSE equity instruments with their ISIN parsed out of the instrument key."""
    if download or path is None:
        resp = requests.get(INSTRUMENTS_URL, headers={"User-Agent": USER_AGENT}, timeout=120)
        resp.raise_for_status()
        import gzip, io
        text = gzip.decompress(resp.content).decode("utf-8")
        raw = pd.read_csv(io.StringIO(text), dtype=str)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
    else:
        raw = pd.read_csv(path, dtype=str)

    eq = raw[(raw["exchange"] == "NSE_EQ") & (raw["instrument_type"] == "EQUITY")].copy()
    eq["isin"] = eq["instrument_key"].str.split("|").str[1]
    # Equity ISINs start INE/INF; IN0/IN2/IN9 are SDLs, T-bills and other debt
    # that NSE lists in the same segment.
    eq = eq[eq["isin"].str.match(r"^IN[EF][A-Z0-9]{9}$", na=False)]
    # Corporate bonds carry an INE ISIN too, so the ISIN test alone lets ~1,500
    # of them through. Their tickers encode a coupon and so begin with a digit
    # (737IRFC29, 805ABCL28); no equity ticker does.
    eq = eq[~eq["tradingsymbol"].str.match(r"^\d", na=False)]
    # Characters 8-9 of an Indian ISIN are the security type: 01-06 are equity
    # classes, 07 upwards are debt. Without this, a bond listed under the
    # company's own ticker (MOTHERSON -> INE775A08105) masquerades as the share
    # and its ISIN never matches the one NSE uses in its index files.
    eq = eq[eq["isin"].str[7:9].isin({"01", "02", "03", "04", "05", "06"})]
    return eq[["isin", "instrument_key", "tradingsymbol", "name"]].reset_index(drop=True)


# --- client -----------------------------------------------------------------


class UpstoxClient:
    """Thread-safe client with retry and backoff. One session per thread."""

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        max_retries: int = 4,
        backoff: float = 1.6,
        sleep=time.sleep,
        session_factory=None,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._sleep = sleep
        # Sessions are per-thread because ingestion fans out over a thread pool.
        # session_factory makes that injectable: a stub set on the calling
        # thread would otherwise be invisible to the workers.
        self._session_factory = session_factory or self._default_session
        self._local = threading.local()

    @staticmethod
    def _default_session() -> requests.Session:
        s = requests.Session()
        s.headers.update({"Accept": "application/json", "User-Agent": USER_AGENT})
        return s

    @property
    def session(self):
        s = getattr(self._local, "session", None)
        if s is None:
            s = self._session_factory()
            self._local.session = s
        return s

    def candles(self, isin: str, unit: str, interval: int, start, end) -> list[list]:
        """One request. ``start``/``end`` are inclusive dates; returns newest-first."""
        start = pd.Timestamp(start).strftime("%Y-%m-%d")
        end = pd.Timestamp(end).strftime("%Y-%m-%d")
        key = requests.utils.quote(instrument_key(isin), safe="")
        url = f"{API_BASE}/{key}/{unit}/{interval}/{end}/{start}"

        last: Exception | None = None
        for attempt in range(self.max_retries):
            if attempt:
                self._sleep(self.backoff ** attempt + random.uniform(0, 0.3))
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                last = exc
                continue
            if resp.status_code == 400:
                # Range too wide, or the instrument has no data at all. Neither
                # is retryable; the caller chunks ranges before calling.
                raise UpstoxError(f"400 for {isin} {unit}/{interval} {start}..{end}: {resp.text[:160]}")
            if resp.status_code in (429,) or resp.status_code >= 500:
                last = UpstoxError(f"{resp.status_code} for {url}")
                continue
            if resp.status_code != 200:
                raise UpstoxError(f"{resp.status_code} for {url}: {resp.text[:160]}")
            try:
                payload = resp.json()
            except ValueError as exc:
                last = UpstoxError(f"non-JSON response for {url}: {exc}")
                continue
            if payload.get("status") != "success":
                raise UpstoxError(f"upstream status {payload.get('status')} for {url}: {str(payload)[:200]}")
            return payload.get("data", {}).get("candles", []) or []
        raise UpstoxError(f"giving up on {url} after {self.max_retries} attempts: {last}")


# --- candle -> frame --------------------------------------------------------

_CANDLE_COLS = ["ts", "open", "high", "low", "close", "volume", "open_interest"]


def _empty_candle_frame() -> pd.DataFrame:
    """Typed empty frame — an untyped one turns ``ts`` into object dtype on
    concat, which breaks ``.dt`` for any symbol listed after the window start."""
    return pd.DataFrame({
        "ts": pd.Series([], dtype="datetime64[ns]"),
        "open": pd.Series([], dtype="float64"),
        "high": pd.Series([], dtype="float64"),
        "low": pd.Series([], dtype="float64"),
        "close": pd.Series([], dtype="float64"),
        "volume": pd.Series([], dtype="float64"),
        "isin": pd.Series([], dtype="string"),
    })


def candles_to_frame(candles: list[list], isin: str) -> pd.DataFrame:
    if not candles:
        return _empty_candle_frame()
    df = pd.DataFrame(candles, columns=_CANDLE_COLS[: len(candles[0])])
    # Timestamps arrive as +05:30; store IST wall clock with the offset stripped
    # so every date comparison is in exchange-local time.
    ts = pd.to_datetime(df["ts"], format="ISO8601", utc=True).dt.tz_convert("Asia/Kolkata")
    df["ts"] = ts.dt.tz_localize(None)
    df["isin"] = isin
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    return df.sort_values("ts").reset_index(drop=True)


def _month_windows(start, end) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Calendar-month windows; the minute endpoint rejects anything wider."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    out = []
    cur = start.replace(day=1)
    while cur <= end:
        nxt = (cur + pd.offsets.MonthBegin(1)).normalize()
        out.append((max(cur, start), min(nxt - pd.Timedelta(days=1), end)))
        cur = nxt
    return [(a, b) for a, b in out if a <= b]


def _decade_windows(start, end) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """<=10 year windows; the daily endpoint returns nothing beyond that."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    out, cur = [], start
    while cur <= end:
        stop = min(cur + pd.DateOffset(years=9, months=11), end)
        out.append((cur, stop))
        cur = stop + pd.Timedelta(days=1)
    return out


# --- daily ------------------------------------------------------------------


def fetch_daily(client: UpstoxClient, isin: str, start, end) -> pd.DataFrame:
    """Adjusted daily bars, mapped onto the ``adj_*`` block."""
    frames = []
    for a, b in _decade_windows(max(pd.Timestamp(start), DAILY_HISTORY_START), end):
        frames.append(candles_to_frame(client.candles(isin, "days", 1, a, b), isin))
    # Drop empty windows before concat: a company listed in 2015 has nothing for
    # the 2000-2009 window, and concatenating empties loses the ts dtype.
    frames = [f for f in frames if len(f)]
    if not frames:
        return S.empty_frame(S.BAR_DTYPES)
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return S.empty_frame(S.BAR_DTYPES)
    out = pd.DataFrame({
        "isin": df["isin"],
        "date": df["ts"].dt.normalize(),
        "adj_open": df["open"],
        "adj_high": df["high"],
        "adj_low": df["low"],
        "adj_close": df["close"],
        "volume": df["volume"],
        "adj_source": "upstox",
    })
    return out.drop_duplicates(subset=["isin", "date"], keep="last").reset_index(drop=True)


OHLC_ADJ = ["adj_open", "adj_high", "adj_low", "adj_close"]

INT32_WRAP = 2 ** 32


def repair_volume(df: pd.DataFrame, col: str = "volume") -> tuple[pd.DataFrame, int]:
    """Undo signed 32-bit overflow in reported volume.

    Upstox occasionally serves a negative volume for a very heavily traded
    session: the true figure exceeded 2**31 and wrapped. Adding 2**32 recovers
    it exactly — e.g. IDEA on 2024-08-30 reports -81,259,413, which is
    4,213,707,883 wrapped, and sits between neighbouring days of 827M and 686M.

    Only applied where the result becomes positive; anything still negative is
    left for the caller to quarantine.
    """
    if col not in df.columns or df.empty:
        return df, 0
    vol = pd.to_numeric(df[col], errors="coerce")
    wrapped = vol < 0
    if not wrapped.any():
        return df, 0
    fixed = vol.where(~wrapped, vol + INT32_WRAP)
    repaired = int((wrapped & (fixed > 0)).sum())
    out = df.copy()
    out[col] = fixed
    return out, repaired


def split_invalid(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate unusable bars from good ones. Returns (clean, rejected).

    Two defects appear in real Upstox history, both rare and both fatal
    downstream:

    * OHLC printed as 0.00 with non-zero volume (mostly 2003-04). A zero close
      destroys every indicator that divides by it.
    * open or close outside the high/low range — frozen quotes on halted or
      suspended stocks, where the close is a settlement price rather than a
      traded one. These break any true-range or pivot calculation.
    """
    o, h, l, c = cols
    present = df[cols].notna().all(axis=1)
    positive = (df[cols] > 0).all(axis=1)
    if "volume" in df.columns:
        positive &= df["volume"].fillna(0) >= 0
    consistent = (
        (df[h] >= df[l])
        & (df[h] >= df[[o, c]].max(axis=1))
        & (df[l] <= df[[o, c]].min(axis=1))
    )
    good = present & positive & consistent
    return df[good].reset_index(drop=True), df[~good].reset_index(drop=True)


@dataclass
class FetchStat:
    isin: str
    rows: int = 0
    error: str = ""
    dropped: int = 0


def ingest_daily(
    store: Store,
    isins: list[str],
    start,
    end,
    *,
    client: UpstoxClient | None = None,
    workers: int = 8,
    on_progress=None,
) -> list[FetchStat]:
    """Fetch adjusted daily bars for many symbols in parallel and write once.

    Only the adjusted block is written, so a later bhavcopy ingest fills the raw
    block of the same rows without either clobbering the other.
    """
    client = client or UpstoxClient()
    stats: list[FetchStat] = []
    frames: list[pd.DataFrame] = []

    rejects: list[pd.DataFrame] = []
    repaired = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_daily, client, isin, start, end): isin for isin in isins}
        for i, fut in enumerate(as_completed(futures), 1):
            isin = futures[fut]
            try:
                df = fut.result()
                if len(df):
                    df, nrep = repair_volume(df)
                    if nrep:
                        repaired += nrep
                        log.info("repaired %d wrapped volume value(s) for %s", nrep, isin)
                clean, bad = split_invalid(df, OHLC_ADJ) if len(df) else (df, df)
                frames.append(clean)
                if len(bad):
                    rejects.append(bad)
                stats.append(FetchStat(isin, len(clean), "", len(bad)))
            except Exception as exc:
                stats.append(FetchStat(isin, 0, str(exc)[:200]))
                log.warning("daily fetch failed for %s: %s", isin, exc)
            if on_progress:
                on_progress(i, len(isins), isin)

    if rejects:
        n = store.quarantine(
            pd.concat(rejects, ignore_index=True), "unusable adjusted OHLC (non-positive, missing, or open/close outside high/low)",
            "upstox_daily",
        )
        log.warning("quarantined %d daily bars with unusable prices", n)

    if repaired:
        log.warning("repaired %d wrapped (int32-overflowed) volume values", repaired)

    frames = [f for f in frames if len(f)]
    if frames:
        store.write_bars(pd.concat(frames, ignore_index=True))
    return stats


# --- minutes ----------------------------------------------------------------


def fetch_minutes(client: UpstoxClient, isin: str, start, end) -> pd.DataFrame:
    frames = []
    for a, b in _month_windows(max(pd.Timestamp(start), MINUTE_HISTORY_START), end):
        frames.append(candles_to_frame(client.candles(isin, "minutes", 1, a, b), isin))
    frames = [f for f in frames if len(f)]
    if not frames:
        return S.empty_frame(S.MINUTE_DTYPES)
    df = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({
        "isin": df["isin"],
        "ts": df["ts"],
        "open": df["open"],
        "high": df["high"],
        "low": df["low"],
        "close": df["close"],
        "volume": df["volume"],
        "source": "upstox",
    })
    return out.drop_duplicates(subset=["isin", "ts"], keep="last").reset_index(drop=True)


def ingest_minutes_month(
    store: Store,
    isins: list[str],
    year: int,
    month: int,
    *,
    client: UpstoxClient | None = None,
    workers: int = 12,
    on_progress=None,
) -> tuple[int, list[FetchStat]]:
    """Fetch one calendar month for every symbol, then write the partition once.

    Batched this way because a month partition holds millions of rows: writing
    per symbol would rewrite the same file once per symbol.
    """
    client = client or UpstoxClient()
    start = pd.Timestamp(year=year, month=month, day=1)
    end = (start + pd.offsets.MonthBegin(1)) - pd.Timedelta(days=1)
    if start < MINUTE_HISTORY_START:
        return 0, []

    stats: list[FetchStat] = []
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_minutes, client, isin, start, end): isin for isin in isins}
        for i, fut in enumerate(as_completed(futures), 1):
            isin = futures[fut]
            try:
                df = fut.result()
                if len(df):
                    frames.append(df)
                stats.append(FetchStat(isin, len(df)))
            except Exception as exc:
                stats.append(FetchStat(isin, 0, str(exc)[:200]))
                log.warning("minute fetch failed for %s %s-%02d: %s", isin, year, month, exc)
            if on_progress:
                on_progress(i, len(isins), isin)

    rows = 0
    if frames:
        allm = pd.concat(frames, ignore_index=True)
        res = store.write_minutes(allm, replace_month=False)
        rows = res.total
    errors = sum(1 for s in stats if s.error)
    store.log_ingest(
        start, "upstox_minute",
        "ok" if not errors else "error",
        rows, f"{len(frames)}/{len(isins)} symbols, {errors} errors",
    )
    return rows, stats


def backfill_minutes(
    store: Store,
    isins: list[str],
    start,
    end,
    *,
    client: UpstoxClient | None = None,
    workers: int = 12,
    skip_done: bool = True,
    on_month=None,
) -> dict[tuple[int, int], int]:
    """Walk calendar months oldest-first. Re-runs skip months already logged ok."""
    client = client or UpstoxClient()
    start = max(pd.Timestamp(start).normalize(), MINUTE_HISTORY_START)
    end = pd.Timestamp(end).normalize()
    done = store.ingest_status("upstox_minute") if skip_done else pd.Series(dtype="string")

    out: dict[tuple[int, int], int] = {}
    cur = start.replace(day=1)
    while cur <= end:
        key = (cur.year, cur.month)
        if skip_done and cur in done.index and done.loc[cur] == "ok":
            log.info("skip %s-%02d (already ingested)", *key)
        else:
            rows, stats = ingest_minutes_month(
                store, isins, cur.year, cur.month, client=client, workers=workers
            )
            out[key] = rows
            if on_month:
                on_month(key, rows, stats)
        cur = (cur + pd.offsets.MonthBegin(1)).normalize()
    return out

# --- today (intraday endpoint) ----------------------------------------------
# The historical endpoint stops at the previous session. Today's candles live
# on a separate intraday endpoint, both as one daily bar and as minutes.
INTRADAY_BASE = "https://api.upstox.com/v3/historical-candle/intraday"


def fetch_today(client: "UpstoxClient", isin: str, unit: str) -> pd.DataFrame:
    """Today's candles for one symbol: unit 'days' -> one bar, 'minutes' -> 1-min bars."""
    url = f"{INTRADAY_BASE}/{quote(instrument_key(isin), safe='')}/{unit}/1"
    resp = client.session.get(url, timeout=client.timeout)
    if resp.status_code != 200:
        raise UpstoxError(f"{resp.status_code} for {url}")
    payload = resp.json()
    if payload.get("status") != "success":
        raise UpstoxError(f"upstream status {payload.get('status')} for {url}")
    return candles_to_frame(payload["data"].get("candles", []), isin)


def ingest_today(store: Store, isins: list[str], *, client: "UpstoxClient | None" = None,
                 workers: int = 12) -> dict:
    """Write today's daily bar (adjusted block) and minute bars for every symbol."""
    client = client or UpstoxClient()
    daily, mins, errors = [], [], 0
    def one(isin):
        d = fetch_today(client, isin, "days"); m = fetch_today(client, isin, "minutes")
        return isin, d, m
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(one, i) for i in isins]):
            try:
                isin, d, m = fut.result()
            except Exception as exc:
                errors += 1; log.warning("today fetch failed: %s", exc); continue
            if len(d):
                daily.append(pd.DataFrame({"isin": d["isin"], "date": d["ts"].dt.normalize(),
                    "adj_open": d["open"], "adj_high": d["high"], "adj_low": d["low"], "adj_close": d["close"],
                    "volume": d["volume"], "adj_source": "upstox"}))
            if len(m):
                mins.append(pd.DataFrame({"isin": m["isin"], "ts": m["ts"], "open": m["open"], "high": m["high"],
                    "low": m["low"], "close": m["close"], "volume": m["volume"], "source": "upstox"}))
    out = {"symbols": len(isins), "errors": errors, "daily_rows": 0, "minute_rows": 0}
    if daily:
        df = pd.concat(daily, ignore_index=True)
        clean, bad = split_invalid(df, OHLC_ADJ)
        store.write_bars(clean); out["daily_rows"] = len(clean); out["quarantined"] = len(bad)
    if mins:
        res = store.write_minutes(pd.concat(mins, ignore_index=True), replace_month=False)
        out["minute_rows"] = res.total
    return out
