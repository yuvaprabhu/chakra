"""Upstox provider. Every data defect asserted here was found in live data."""

from __future__ import annotations

import pandas as pd
import pytest
import requests

from screener.upstox import (
    INT32_WRAP, MINUTE_HISTORY_START, OHLC_ADJ, UpstoxClient, UpstoxError,
    _decade_windows, _month_windows, candles_to_frame, fetch_daily, fetch_minutes,
    ingest_daily, instrument_key, load_instruments, repair_volume, split_invalid,
)

from .conftest import StubResponse


def candle(ts, o=100.0, h=105.0, l=99.0, c=104.0, v=1000):
    return [ts, o, h, l, c, v, 0]


class StubJsonResponse(StubResponse):
    def __init__(self, status_code=200, payload=None, text=""):
        super().__init__(status_code, b"")
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def ok(candles):
    return StubJsonResponse(200, {"status": "success", "data": {"candles": candles}})


class StubSession:
    def __init__(self, queue):
        self.queue = list(queue)
        self.headers = {}
        self.urls = []
        self._lock = __import__("threading").Lock()

    def get(self, url, timeout=None):
        with self._lock:
            self.urls.append(url)
            item = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        if isinstance(item, Exception):
            raise item
        return item


def client_with(queue, **kw):
    """Inject via session_factory so thread-pool workers get the stub too."""
    shared = StubSession(queue)
    c = UpstoxClient(sleep=lambda _: None, session_factory=lambda: shared, **kw)
    c.session  # bind on this thread as well
    return c


class TestInstrumentKey:
    def test_is_isin_native(self):
        """The key is the ISIN, which is why this provider fits the schema."""
        assert instrument_key("INE002A01018") == "NSE_EQ|INE002A01018"


class TestWindows:
    def test_minute_windows_are_calendar_months(self):
        w = _month_windows("2025-08-20", "2025-10-05")
        assert w[0] == (pd.Timestamp("2025-08-20"), pd.Timestamp("2025-08-31"))
        assert w[1] == (pd.Timestamp("2025-09-01"), pd.Timestamp("2025-09-30"))
        assert w[-1][1] == pd.Timestamp("2025-10-05")

    def test_no_minute_window_exceeds_a_month(self):
        """The endpoint 400s on wider ranges."""
        for a, b in _month_windows("2022-01-01", "2026-09-18"):
            assert (b - a).days <= 31

    def test_no_daily_window_exceeds_ten_years(self):
        """The endpoint silently returns nothing beyond ten years."""
        for a, b in _decade_windows("2000-01-01", "2026-09-18"):
            assert (b - a).days <= 3653

    def test_windows_cover_the_range_without_gaps(self):
        w = _decade_windows("2000-01-01", "2026-09-18")
        assert w[0][0] == pd.Timestamp("2000-01-01")
        assert w[-1][1] == pd.Timestamp("2026-09-18")
        for (_, end), (start, _) in zip(w, w[1:]):
            assert start == end + pd.Timedelta(days=1)


class TestCandleParsing:
    def test_strips_ist_offset_to_wall_clock(self):
        df = candles_to_frame([candle("2025-09-15T09:15:00+05:30")], "INE1")
        assert df.iloc[0]["ts"] == pd.Timestamp("2025-09-15 09:15:00")
        assert df["ts"].dt.tz is None

    def test_sorted_oldest_first(self):
        """The API returns newest-first."""
        df = candles_to_frame([
            candle("2025-09-15T15:29:00+05:30"), candle("2025-09-15T09:15:00+05:30")
        ], "INE1")
        assert df.iloc[0]["ts"] < df.iloc[1]["ts"]

    def test_empty_response_keeps_datetime_dtype(self):
        """An untyped empty frame turns ts into object on concat, which breaks
        every symbol listed after the window start."""
        df = candles_to_frame([], "INE1")
        assert df.empty
        assert str(df["ts"].dtype) == "datetime64[ns]"


class TestClientRetry:
    def test_retries_server_errors(self):
        c = client_with([StubJsonResponse(500), ok([candle("2025-09-15T09:15:00+05:30")])])
        assert len(c.candles("INE1", "days", 1, "2025-09-01", "2025-09-15")) == 1

    def test_400_is_not_retried(self):
        """A 400 means the range is too wide or the instrument has no data;
        retrying just burns the backoff budget."""
        c = client_with([StubJsonResponse(400, text="range too wide")])
        with pytest.raises(UpstoxError, match="400"):
            c.candles("INE1", "minutes", 1, "2025-01-01", "2025-09-15")
        assert len(c.session.urls) == 1

    def test_retries_connection_errors(self):
        c = client_with([requests.ConnectionError("reset"), ok([])])
        assert c.candles("INE1", "days", 1, "2025-09-01", "2025-09-15") == []

    def test_gives_up_eventually(self):
        c = client_with([StubJsonResponse(503)], max_retries=3)
        with pytest.raises(UpstoxError, match="giving up"):
            c.candles("INE1", "days", 1, "2025-09-01", "2025-09-15")
        assert len(c.session.urls) == 3

    def test_url_puts_to_date_before_from_date(self):
        c = client_with([ok([])])
        c.candles("INE002A01018", "minutes", 1, "2025-09-01", "2025-09-30")
        assert c.session.urls[0].endswith("/minutes/1/2025-09-30/2025-09-01")


class TestFetchDaily:
    def test_lands_in_the_adjusted_block_only(self):
        """Upstox is back-adjusted, so it must never write raw close."""
        c = client_with([ok([candle("2025-09-15T00:00:00+05:30", c=104.0)])])
        df = fetch_daily(c, "INE1", "2025-09-01", "2025-09-15")
        assert df.iloc[0]["adj_close"] == 104.0
        assert df.iloc[0]["adj_source"] == "upstox"
        assert "close" not in df.columns or pd.isna(df.iloc[0].get("close"))

    def test_survives_empty_early_windows(self):
        """A company listed in 2015 has nothing for the 2000-2009 window."""
        c = client_with([ok([]), ok([]), ok([candle("2025-09-15T00:00:00+05:30")])])
        df = fetch_daily(c, "INE1", "2000-01-01", "2026-09-18")
        assert len(df) == 1

    def test_all_windows_empty_returns_empty(self):
        c = client_with([ok([])])
        assert fetch_daily(c, "INE1", "2000-01-01", "2026-09-18").empty


class TestVolumeRepair:
    def test_undoes_int32_overflow(self):
        """IDEA on 2024-08-30 reports -81,259,413; the true figure is
        4,213,707,883, which wrapped past 2**31."""
        df = pd.DataFrame({"volume": [-81_259_413]})
        out, n = repair_volume(df)
        assert n == 1
        assert out.iloc[0]["volume"] == 4_213_707_883
        assert out.iloc[0]["volume"] == -81_259_413 + INT32_WRAP

    def test_leaves_normal_volume_alone(self):
        df = pd.DataFrame({"volume": [1000, 8_453_266_674]})
        out, n = repair_volume(df)
        assert n == 0
        assert list(out["volume"]) == [1000, 8_453_266_674]

    def test_unrepairable_stays_negative_for_quarantine(self):
        df = pd.DataFrame({"volume": [-(INT32_WRAP + 5000)]})
        out, n = repair_volume(df)
        assert n == 0 and out.iloc[0]["volume"] < 0


class TestSplitInvalid:
    def _frame(self, **kw):
        base = {"adj_open": 100.0, "adj_high": 105.0, "adj_low": 99.0,
                "adj_close": 104.0, "volume": 1000}
        base.update(kw)
        return pd.DataFrame([base])

    def test_keeps_good_bars(self):
        clean, bad = split_invalid(self._frame(), OHLC_ADJ)
        assert len(clean) == 1 and bad.empty

    def test_rejects_zero_prices(self):
        """Upstox deep history prints 0.00 OHLC with non-zero volume."""
        clean, bad = split_invalid(
            self._frame(adj_open=0.0, adj_high=0.0, adj_low=0.0, adj_close=0.0), OHLC_ADJ
        )
        assert clean.empty and len(bad) == 1

    def test_rejects_close_outside_high_low(self):
        """Frozen quotes on halted stocks carry a settlement close outside the
        traded range, which breaks true-range and pivot maths."""
        clean, bad = split_invalid(
            self._frame(adj_open=280.5, adj_high=280.5, adj_low=280.5, adj_close=295.0),
            OHLC_ADJ,
        )
        assert clean.empty and len(bad) == 1

    def test_rejects_high_below_low(self):
        clean, bad = split_invalid(self._frame(adj_high=90.0, adj_low=95.0), OHLC_ADJ)
        assert clean.empty and len(bad) == 1

    def test_rejects_missing_prices(self):
        clean, bad = split_invalid(self._frame(adj_close=float("nan")), OHLC_ADJ)
        assert clean.empty and len(bad) == 1

    def test_rejects_still_negative_volume(self):
        clean, bad = split_invalid(self._frame(volume=-5), OHLC_ADJ)
        assert clean.empty and len(bad) == 1


class TestIngestDaily:
    def test_quarantines_bad_bars_and_stores_the_rest(self, store):
        good = candle("2025-09-15T00:00:00+05:30", c=104.0)
        zero = candle("2025-09-16T00:00:00+05:30", o=0, h=0, l=0, c=0)
        c = client_with([ok([good, zero])])
        stats = ingest_daily(store, ["INE1"], "2025-09-01", "2025-09-30", client=c, workers=1)
        assert stats[0].rows == 1 and stats[0].dropped == 1
        assert len(store.read_bars()) == 1
        q = store.read_quarantine()
        assert len(q) == 1 and "unusable" in q.iloc[0]["quarantine_reason"]

    def test_failure_on_one_symbol_does_not_lose_the_others(self, store):
        class Flaky(StubSession):
            def get(self, url, timeout=None):
                self.urls.append(url)
                if "INEBAD" in url:
                    return StubJsonResponse(400, text="no data")
                return ok([candle("2025-09-15T00:00:00+05:30")])

        shared = Flaky([])
        c = UpstoxClient(sleep=lambda _: None, session_factory=lambda: shared)
        stats = ingest_daily(store, ["INE1", "INEBAD"], "2025-09-01", "2025-09-30",
                             client=c, workers=1)
        assert {s.isin: bool(s.error) for s in stats} == {"INE1": False, "INEBAD": True}
        assert len(store.read_bars()) == 1


class TestMinuteHistoryBound:
    def test_does_not_request_before_january_2022(self):
        """1-minute history does not exist before 2022; asking wastes calls."""
        c = client_with([ok([])])
        fetch_minutes(c, "INE1", "2018-01-01", "2022-02-28")
        for url in c.session.urls:
            tail = url.rsplit("/", 1)[-1]
            assert pd.Timestamp(tail) >= MINUTE_HISTORY_START
