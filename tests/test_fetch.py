"""Bhavcopy fetch and ingestion. No test here touches the network."""

from __future__ import annotations

import pandas as pd
import pytest
import requests

from screener import fetch
from screener.fetch import (
    FetchError, IngestResult, NotPublished, NSESession,
    backfill, expected_sessions, ingest_date, missing_dates, parse_udiff, udiff_url,
)
from screener.universe import Universe

from .conftest import StubResponse, StubTransport, default_rows, udiff_csv, udiff_row, udiff_zip

ARCH = "https://nsearchives.nseindia.com"


def session_with(responses=None, default=None, **kw):
    s = NSESession(sleep=lambda _: None, **kw)
    s.session = StubTransport(responses, default)
    return s


class TestUrl:
    def test_udiff_filename(self):
        assert udiff_url("2024-08-01").endswith(
            "/content/cm/BhavCopy_NSE_CM_0_0_0_20240801_F_0000.csv.zip"
        )


class TestParse:
    def test_maps_udiff_to_canonical(self):
        df = parse_udiff(udiff_zip(default_rows()))
        rel = df[df["symbol"] == "RELIANCE"].iloc[0]
        assert rel["isin"] == "INE002A01018"
        assert rel["date"] == pd.Timestamp("2024-08-01")
        assert rel["open"] == 2900 and rel["close"] == 2940
        assert rel["volume"] == 5_000_000
        assert rel["turnover"] == pytest.approx(1.47e10)
        assert rel["trades"] == 120_000
        assert rel["source"] == "bhavcopy_udiff"

    def test_filters_non_equity_instruments(self):
        """UDiFF carries futures, options, index and ETF rows in the same file."""
        df = parse_udiff(udiff_zip(default_rows()))
        assert set(df["symbol"]) == {"RELIANCE", "INFY", "SMALLCO"}
        assert "NIFTY" not in set(df["symbol"])
        assert "NIFTYBEES" not in set(df["symbol"])

    def test_drops_rows_without_isin(self):
        df = parse_udiff(udiff_zip(default_rows()))
        assert "NOISIN" not in set(df["symbol"])
        assert df["isin"].notna().all()

    def test_series_filter_is_configurable(self):
        df = parse_udiff(udiff_zip(default_rows()), series=("EQ",))
        assert set(df["symbol"]) == {"RELIANCE", "INFY"}

    def test_never_populates_the_adjusted_block(self):
        """The bhavcopy is the raw series. Copying close into adj_close would
        overwrite Upstox's genuine back-adjusted values and force every implied
        adjustment factor to exactly 1.0."""
        df = parse_udiff(udiff_zip(default_rows()))
        assert df["adj_close"].isna().all()
        assert df["adj_open"].isna().all()

    def test_accepts_plain_csv_text(self):
        df = parse_udiff(udiff_csv(default_rows()))
        assert len(df) == 3

    def test_accepts_ddmmyyyy_dates(self):
        rows = [udiff_row(trad_dt="01-08-2024", isin="INE002A01018", symbol="RELIANCE")]
        assert parse_udiff(udiff_csv(rows)).iloc[0]["date"] == pd.Timestamp("2024-08-01")

    def test_rejects_a_file_that_is_not_a_bhavcopy(self):
        with pytest.raises(FetchError, match="not a UDiFF"):
            parse_udiff("col_a,col_b\n1,2\n")

    def test_drops_zero_price_rows(self):
        rows = default_rows() + [
            udiff_row(trad_dt="2024-08-01", isin="INE999Z01011", symbol="ZEROCO",
                      open_=0, high=0, low=0, close=0)
        ]
        assert "ZEROCO" not in set(parse_udiff(udiff_csv(rows))["symbol"])


class TestSession:
    def test_warms_cookies_before_the_archive(self):
        s = session_with({ARCH: [StubResponse(200, udiff_zip(default_rows()))]})
        s.get(f"{ARCH}/x.zip")
        assert s.session.calls[0][0].startswith("https://www.nseindia.com")
        assert len(s.session.cookies) > 0

    def test_404_is_not_retried(self):
        s = session_with({ARCH: [StubResponse(404, b"")]})
        with pytest.raises(NotPublished):
            s.get(f"{ARCH}/x.zip")
        assert len(s.session.archive_calls) == 1

    def test_403_rewarms_and_retries(self):
        """A 403 means the cookie went stale, not that the file is missing."""
        s = session_with({ARCH: [
            StubResponse(403, b"denied"), StubResponse(200, b"PK\x03\x04ok")
        ]})
        resp = s.get(f"{ARCH}/x.zip")
        assert resp.status_code == 200
        assert len(s.session.archive_calls) == 2

    def test_gives_up_after_max_retries(self):
        s = session_with({ARCH: [StubResponse(500, b"")]}, max_retries=3)
        with pytest.raises(FetchError, match="giving up"):
            s.get(f"{ARCH}/x.zip")
        assert len(s.session.archive_calls) == 3

    def test_retries_connection_errors(self):
        s = session_with({ARCH: [
            requests.ConnectionError("reset"), StubResponse(200, b"PK ok")
        ]})
        assert s.get(f"{ARCH}/x.zip").status_code == 200

    def test_backoff_grows(self):
        delays = []
        s = NSESession(sleep=delays.append, max_retries=4, backoff=2.0)
        s.session = StubTransport({ARCH: [StubResponse(500, b"")]})
        with pytest.raises(FetchError):
            s.get(f"{ARCH}/x.zip")
        assert len(delays) == 3
        assert delays[0] < delays[1] < delays[2]

    def test_cookies_are_reused_across_calls(self):
        s = session_with({ARCH: [StubResponse(200, b"PK ok")]})
        s.get(f"{ARCH}/a.zip")
        n = len(s.session.calls)
        s.get(f"{ARCH}/b.zip")
        assert len(s.session.calls) == n + 1  # no second warm-up


class TestFetchBhavcopy:
    def test_returns_zip_bytes(self):
        payload = udiff_zip(default_rows())
        s = session_with({ARCH: [StubResponse(200, payload)]})
        assert fetch.fetch_bhavcopy("2024-08-01", session=s) == payload

    def test_html_error_page_with_200_is_an_error(self):
        """NSE serves an Akamai 'Access Denied' page with HTTP 200."""
        s = session_with({ARCH: [StubResponse(200, b"<HTML><HEAD><TITLE>Access Denied")]})
        with pytest.raises(FetchError, match="did not return a zip"):
            fetch.fetch_bhavcopy("2024-08-01", session=s)

    def test_empty_body_is_not_published(self):
        s = session_with({ARCH: [StubResponse(200, b"")]})
        with pytest.raises(NotPublished):
            fetch.fetch_bhavcopy("2024-08-01", session=s)


class TestIngest:
    def test_writes_bars_and_logs_ok(self, store):
        s = session_with({ARCH: [StubResponse(200, udiff_zip(default_rows()))]})
        res = ingest_date(store, "2024-08-01", session=s)
        assert res.status == "ok" and res.rows == 3
        assert len(store.read_bars()) == 3
        assert store.ingest_status("bhavcopy_udiff").iloc[0] == "ok"

    def test_rerun_is_skipped(self, store):
        s = session_with({ARCH: [StubResponse(200, udiff_zip(default_rows()))]})
        ingest_date(store, "2024-08-01", session=s)
        before = len(s.session.archive_calls)
        res = ingest_date(store, "2024-08-01", session=s)
        assert res.status == "skipped"
        assert len(s.session.archive_calls) == before  # no refetch
        assert len(store.read_bars()) == 3

    def test_force_refetches(self, store):
        s = session_with({ARCH: [StubResponse(200, udiff_zip(default_rows()))]})
        ingest_date(store, "2024-08-01", session=s)
        res = ingest_date(store, "2024-08-01", session=s, force=True)
        assert res.status == "ok"
        assert len(store.read_bars()) == 3

    def test_holiday_is_recorded_as_no_data(self, store):
        s = session_with({ARCH: [StubResponse(404, b"")]})
        res = ingest_date(store, "2024-08-15", session=s)
        assert res.status == "no_data"
        assert store.ingest_status("bhavcopy_udiff").iloc[0] == "no_data"

    def test_network_failure_is_an_error_not_a_holiday(self, store):
        s = session_with({ARCH: [StubResponse(500, b"")]}, max_retries=2)
        res = ingest_date(store, "2024-08-01", session=s)
        assert res.status == "error"
        assert store.ingest_status("bhavcopy_udiff").iloc[0] == "error"

    def test_wrong_trading_date_in_file_is_rejected(self, store):
        """If NSE serves another day's file, storing it would misdate a whole
        session. The file's own TradDt is authoritative."""
        s = session_with({ARCH: [StubResponse(200, udiff_zip(default_rows("2024-07-31")))]})
        res = ingest_date(store, "2024-08-01", session=s)
        assert res.status == "error" and "TradDt" in res.message
        assert store.read_bars().empty

    def test_updates_symbol_master(self, store):
        s = session_with({ARCH: [StubResponse(200, udiff_zip(default_rows()))]})
        ingest_date(store, "2024-08-01", session=s)
        symbols = store.read_symbols()
        assert set(symbols["symbol"]) == {"RELIANCE", "INFY", "SMALLCO"}
        assert Universe(store).symbol_on("INE002A01018", "2024-08-01") == "RELIANCE"

    def test_saves_raw_payload_when_asked(self, store, tmp_path):
        s = session_with({ARCH: [StubResponse(200, udiff_zip(default_rows()))]})
        ingest_date(store, "2024-08-01", session=s, raw_dir=tmp_path / "raw")
        assert (tmp_path / "raw" / "BhavCopy_NSE_CM_0_0_0_20240801_F_0000.csv.zip").exists()


class TestGapDetection:
    def test_expected_sessions_skips_weekends(self):
        days = expected_sessions("2024-08-01", "2024-08-07")
        assert pd.Timestamp("2024-08-03") not in days  # Saturday
        assert pd.Timestamp("2024-08-04") not in days  # Sunday
        assert len(days) == 5

    def test_holidays_are_excluded(self):
        days = expected_sessions("2024-08-01", "2024-08-07", holidays=["2024-08-02"])
        assert pd.Timestamp("2024-08-02") not in days

    def test_missing_dates_finds_the_gap(self, store):
        s = session_with({ARCH: [StubResponse(200, udiff_zip(default_rows("2024-08-01")))]})
        ingest_date(store, "2024-08-01", session=s)
        missing = missing_dates(store, "2024-08-01", "2024-08-07")
        assert pd.Timestamp("2024-08-01") not in missing
        assert pd.Timestamp("2024-08-02") in missing

    def test_known_holidays_are_not_reported_as_gaps(self, store):
        store.log_ingest("2024-08-02", "bhavcopy_udiff", "no_data", 0, "404")
        assert pd.Timestamp("2024-08-02") not in missing_dates(store, "2024-08-01", "2024-08-07")

    def test_errors_are_retried_by_default(self, store):
        store.log_ingest("2024-08-02", "bhavcopy_udiff", "error", 0, "500")
        assert pd.Timestamp("2024-08-02") in missing_dates(store, "2024-08-01", "2024-08-07")
        assert pd.Timestamp("2024-08-02") not in missing_dates(
            store, "2024-08-01", "2024-08-07", retry_errors=False
        )


class TestBackfill:
    def test_fills_only_the_missing_days(self, store):
        rows = {d: udiff_zip(default_rows(d)) for d in
                ("2024-08-01", "2024-08-02", "2024-08-05")}

        class DateAware(StubTransport):
            def get(self, url, headers=None, timeout=None):
                self.calls.append((url, dict(headers or {})))
                if url.startswith("https://www.nseindia.com"):
                    self.cookies.set("nsit", "tok")
                    return StubResponse(200, b"<html>")
                for d, payload in rows.items():
                    if d.replace("-", "") in url:
                        return StubResponse(200, payload)
                return StubResponse(404, b"")

        s = NSESession(sleep=lambda _: None)
        s.session = DateAware()
        results = backfill(store, "2024-08-01", "2024-08-05", session=s,
                           pause=0, sleep=lambda _: None)
        by_status = {r.status for r in results}
        assert "ok" in by_status
        assert len(store.read_bars()) == 9  # 3 symbols x 3 days
        assert set(store.available_dates()) == {
            pd.Timestamp("2024-08-01"), pd.Timestamp("2024-08-02"), pd.Timestamp("2024-08-05")
        }

    def test_second_run_does_no_work(self, store):
        s = session_with({ARCH: [StubResponse(404, b"")]})
        backfill(store, "2024-08-01", "2024-08-02", session=s, pause=0, sleep=lambda _: None)
        results = backfill(store, "2024-08-01", "2024-08-02", session=s, pause=0, sleep=lambda _: None)
        assert results == []


class TestRawAwareGapDetection:
    """Regression: the adjusted block is backfilled from Upstox across every
    date since 2000. If gap detection counts those as present, the bhavcopy
    ingest silently never runs and no raw price is ever stored."""

    def test_adjusted_only_date_still_counts_as_missing(self, store):
        store.write_bars(pd.DataFrame([{
            "isin": "INE1", "date": "2024-08-01", "adj_open": 10.0, "adj_high": 11.0,
            "adj_low": 9.0, "adj_close": 10.5, "adj_source": "upstox",
        }]))
        assert pd.Timestamp("2024-08-01") in missing_dates(store, "2024-08-01", "2024-08-02")

    def test_raw_bearing_date_is_not_missing(self, store):
        s = session_with({ARCH: [StubResponse(200, udiff_zip(default_rows("2024-08-01")))]})
        ingest_date(store, "2024-08-01", session=s)
        assert pd.Timestamp("2024-08-01") not in missing_dates(store, "2024-08-01", "2024-08-02")

    def test_available_dates_require_filter(self, store):
        store.write_bars(pd.DataFrame([{
            "isin": "INE1", "date": "2024-08-01", "adj_open": 10.0, "adj_high": 11.0,
            "adj_low": 9.0, "adj_close": 10.5,
        }]))
        assert len(store.available_dates()) == 1
        assert len(store.available_dates(require="adjusted")) == 1
        assert len(store.available_dates(require="raw")) == 0
