"""Storage layer: partitioning, idempotency, two-source merge, validation."""

from __future__ import annotations

import pandas as pd
import pytest

from screener import schema as S
from screener.store import Store, StoreError


def bar(isin="INE002A01018", date="2024-08-01", **kw):
    row = {
        "isin": isin, "symbol": "RELIANCE", "series": "EQ", "date": date,
        "open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0,
        "volume": 1000, "source": "bhavcopy_udiff",
    }
    row.update(kw)
    return row


def frame(*rows):
    return pd.DataFrame(list(rows))


class TestWriteAndRead:
    def test_roundtrip(self, store):
        res = store.write_bars(frame(bar()))
        assert (res.inserted, res.updated) == (1, 0)
        out = store.read_bars()
        assert len(out) == 1
        assert out.iloc[0]["close"] == 104.0
        assert out.iloc[0]["symbol"] == "RELIANCE"

    def test_partitioned_by_year(self, store):
        store.write_bars(frame(
            bar(date="2023-06-01"), bar(date="2024-06-01"), bar(date="2025-06-01")
        ))
        assert store.years() == [2023, 2024, 2025]
        assert (store.bars_dir / "year=2024" / "bars.parquet").exists()

    def test_hive_partition_key_does_not_leak(self, store):
        store.write_bars(frame(bar()))
        assert "year" not in store.read_bars().columns
        assert list(store.read_bars().columns) == S.BAR_COLUMNS

    def test_date_range_filter(self, store):
        store.write_bars(frame(
            bar(date="2023-06-01"), bar(date="2024-06-01"), bar(date="2025-06-01")
        ))
        out = store.read_bars(start="2024-01-01", end="2024-12-31")
        assert len(out) == 1
        assert out.iloc[0]["date"] == pd.Timestamp("2024-06-01")

    def test_isin_filter(self, store):
        store.write_bars(frame(bar(isin="INE001"), bar(isin="INE002")))
        assert store.read_bars(isins=["INE001"]).iloc[0]["isin"] == "INE001"
        assert store.read_bars(isins=[]).empty

    def test_empty_store_reads_clean(self, store):
        assert store.read_bars().empty
        assert len(store.available_dates()) == 0
        assert store.summary()["bars"] == 0

    def test_available_dates(self, store):
        store.write_bars(frame(
            bar(date="2024-08-01"), bar(isin="INE9", date="2024-08-01"), bar(date="2024-08-02")
        ))
        assert list(store.available_dates()) == [
            pd.Timestamp("2024-08-01"), pd.Timestamp("2024-08-02")
        ]

    def test_bar_counts_by_date(self, store):
        store.write_bars(frame(
            bar(date="2024-08-01"), bar(isin="INE9", date="2024-08-01"), bar(date="2024-08-02")
        ))
        counts = store.bar_counts_by_date()
        assert counts.loc[pd.Timestamp("2024-08-01")] == 2
        assert counts.loc[pd.Timestamp("2024-08-02")] == 1


class TestIdempotency:
    def test_rerun_is_not_a_duplicate(self, store):
        store.write_bars(frame(bar()))
        res = store.write_bars(frame(bar()))
        assert (res.inserted, res.updated) == (0, 1)
        assert len(store.read_bars()) == 1

    def test_rewrite_updates_value(self, store):
        store.write_bars(frame(bar(close=104.0)))
        store.write_bars(frame(bar(close=101.0)))
        out = store.read_bars()
        assert len(out) == 1 and out.iloc[0]["close"] == 101.0

    def test_partial_day_rerun_merges(self, store):
        store.write_bars(frame(bar(isin="INE1"), bar(isin="INE2")))
        store.write_bars(frame(bar(isin="INE2", close=101.0), bar(isin="INE3")))
        out = store.read_bars().set_index("isin")
        assert set(out.index) == {"INE1", "INE2", "INE3"}
        assert out.loc["INE2", "close"] == 101.0


class TestTwoSourceMerge:
    """Raw (bhavcopy) and adjusted (Upstox) legs must land in one row without
    either erasing the other — the whole point of the column-wise upsert."""

    def test_adjusted_then_raw(self, store):
        store.write_bars(frame({
            "isin": "INE1", "date": "2024-08-01", "adj_open": 10.0, "adj_high": 11.0,
            "adj_low": 9.0, "adj_close": 10.5, "adj_source": "upstox",
        }))
        store.write_bars(frame(bar(isin="INE1", open=20.0, high=22.0, low=18.0, close=21.0)))
        out = store.read_bars()
        assert len(out) == 1
        row = out.iloc[0]
        assert row["adj_close"] == 10.5 and row["close"] == 21.0
        assert row["adj_source"] == "upstox" and row["source"] == "bhavcopy_udiff"

    def test_raw_then_adjusted(self, store):
        store.write_bars(frame(bar(isin="INE1", open=20.0, high=22.0, low=18.0, close=21.0)))
        store.write_bars(frame({
            "isin": "INE1", "date": "2024-08-01", "adj_open": 10.0, "adj_high": 11.0,
            "adj_low": 9.0, "adj_close": 10.5, "adj_source": "upstox",
        }))
        row = store.read_bars().iloc[0]
        assert row["close"] == 21.0 and row["adj_close"] == 10.5
        assert row["symbol"] == "RELIANCE"

    def test_adjusted_write_does_not_zero_volume(self, store):
        """volume is nullable Int64 for exactly this reason: 0 is a real value
        and would win a column-wise merge against a stored 1000."""
        store.write_bars(frame(bar(isin="INE1", volume=1000)))
        store.write_bars(frame({
            "isin": "INE1", "date": "2024-08-01", "adj_open": 1.0, "adj_high": 2.0,
            "adj_low": 0.5, "adj_close": 1.5,
        }))
        assert store.read_bars().iloc[0]["volume"] == 1000

    def test_implied_adjustment_factor(self, store):
        """Where both legs exist, close / adj_close is the cumulative factor."""
        store.write_bars(frame(bar(isin="INE1", open=1900.0, high=2100.0, low=1800.0, close=2000.0)))
        store.write_bars(frame({
            "isin": "INE1", "date": "2024-08-01", "adj_open": 900.0, "adj_high": 1100.0,
            "adj_low": 800.0, "adj_close": 1000.0,
        }))
        row = store.read_bars().iloc[0]
        assert row["close"] / row["adj_close"] == pytest.approx(2.0)

    def test_raw_write_cannot_clobber_a_stored_adjusted_price(self, store):
        """Regression: a bhavcopy write arriving after an Upstox write must
        leave adj_close alone, or the implied factor collapses to 1.0."""
        store.write_bars(frame({
            "isin": "INE1", "date": "2024-08-01", "adj_open": 900.0, "adj_high": 1100.0,
            "adj_low": 800.0, "adj_close": 1000.0, "adj_source": "upstox",
        }))
        store.write_bars(frame(bar(
            isin="INE1", open=1900.0, high=2100.0, low=1800.0, close=2000.0,
        )))
        row = store.read_bars().iloc[0]
        assert row["adj_close"] == 1000.0
        assert row["close"] / row["adj_close"] == pytest.approx(2.0)

    def test_never_fabricates_adj_close_from_close(self, store):
        """Raw-only writes must leave adjusted null, not copy close into it —
        a fake adj_close is indistinguishable from a real one downstream."""
        store.write_bars(frame(bar(close=104.0)))
        assert pd.isna(store.read_bars().iloc[0]["adj_close"])


class TestValidation:
    def test_rejects_high_below_low(self, store):
        with pytest.raises(StoreError, match="high < low"):
            store.write_bars(frame(bar(high=90.0, low=95.0)))

    def test_rejects_close_outside_range(self, store):
        with pytest.raises(StoreError, match="outside the high/low"):
            store.write_bars(frame(bar(high=105.0, low=99.0, close=120.0)))

    def test_rejects_non_positive_price(self, store):
        with pytest.raises(StoreError, match="non-positive"):
            store.write_bars(frame(bar(low=0.0, close=0.0)))

    def test_rejects_duplicate_keys_in_one_write(self, store):
        with pytest.raises(StoreError, match="duplicate"):
            store.write_bars(frame(bar(), bar()))

    def test_rejects_null_isin(self, store):
        with pytest.raises(StoreError, match="null isin"):
            store.write_bars(frame(bar(isin=None)))

    def test_rejects_negative_volume(self, store):
        with pytest.raises(StoreError, match="negative volume"):
            store.write_bars(frame(bar(volume=-5)))

    def test_validates_adjusted_block_too(self, store):
        with pytest.raises(StoreError, match="adjusted bars with high < low"):
            store.write_bars(frame({
                "isin": "INE1", "date": "2024-08-01", "adj_open": 10.0,
                "adj_high": 5.0, "adj_low": 9.0, "adj_close": 7.0,
            }))

    def test_rejects_row_with_neither_block(self, store):
        with pytest.raises(StoreError, match="complete raw or adjusted"):
            store.write_bars(frame({"isin": "INE1", "date": "2024-08-01", "volume": 10}))

    def test_bad_write_leaves_store_untouched(self, store):
        store.write_bars(frame(bar()))
        with pytest.raises(StoreError):
            store.write_bars(frame(bar(date="2024-08-02", high=1.0, low=99.0)))
        assert len(store.read_bars()) == 1


class TestIngestLog:
    def test_records_and_dedupes_by_date(self, store):
        store.log_ingest("2024-08-01", "bhavcopy_udiff", "error", 0, "boom")
        store.log_ingest("2024-08-01", "bhavcopy_udiff", "ok", 1800, "+1800")
        status = store.ingest_status("bhavcopy_udiff")
        assert status.loc[pd.Timestamp("2024-08-01")] == "ok"
        assert len(status) == 1

    def test_sources_are_independent(self, store):
        store.log_ingest("2024-08-01", "bhavcopy_udiff", "ok", 10)
        store.log_ingest("2024-08-01", "upstox_minute", "error", 0)
        assert store.ingest_status("bhavcopy_udiff").iloc[0] == "ok"
        assert store.ingest_status("upstox_minute").iloc[0] == "error"


class TestQuarantine:
    def test_keeps_rejected_rows_with_reason(self, store):
        bad = frame(bar(close=0.0))
        n = store.quarantine(bad, "zero close", "upstox_daily")
        assert n == 1
        q = store.read_quarantine()
        assert q.iloc[0]["quarantine_reason"] == "zero close"
        assert q.iloc[0]["quarantine_source"] == "upstox_daily"

    def test_appends_across_calls(self, store):
        store.quarantine(frame(bar()), "a", "s")
        store.quarantine(frame(bar(date="2024-08-02")), "b", "s")
        assert len(store.read_quarantine()) == 2


class TestMinutes:
    def _minutes(self, isin="INE1", day="2025-09-15", n=375):
        ts = pd.date_range(f"{day} 09:15", periods=n, freq="min")
        return pd.DataFrame({
            "isin": isin, "ts": ts, "open": 1.0, "high": 2.0, "low": 0.5,
            "close": 1.5, "volume": 100, "source": "upstox",
        })

    def test_write_and_read(self, store):
        res = store.write_minutes(self._minutes())
        assert res.inserted == 375
        assert len(store.read_minutes()) == 375

    def test_idempotent(self, store):
        store.write_minutes(self._minutes())
        res = store.write_minutes(self._minutes())
        assert (res.inserted, res.updated) == (0, 375)
        assert len(store.read_minutes()) == 375

    def test_month_partitioned(self, store):
        store.write_minutes(self._minutes(day="2025-08-15", n=10))
        store.write_minutes(self._minutes(day="2025-09-15", n=10))
        assert store.minute_months() == [(2025, 8), (2025, 9)]

    def test_partition_keys_do_not_leak(self, store):
        store.write_minutes(self._minutes(n=5))
        assert list(store.read_minutes().columns) == S.MINUTE_COLUMNS

    def test_time_filter(self, store):
        store.write_minutes(self._minutes(n=60))
        out = store.read_minutes(start="2025-09-15 09:30", end="2025-09-15 09:39")
        assert len(out) == 10

    def test_rejects_null_key(self, store):
        m = self._minutes(n=3)
        m.loc[0, "ts"] = pd.NaT
        with pytest.raises(StoreError, match="null isin or ts"):
            store.write_minutes(m)


class TestQueryView:
    def test_bars_view(self, store):
        store.write_bars(frame(bar(isin="INE1"), bar(isin="INE2")))
        out = store.query("SELECT COUNT(*) AS n FROM bars")
        assert out.iloc[0]["n"] == 2

    def test_view_excludes_partition_key(self, store):
        store.write_bars(frame(bar()))
        assert "year" not in store.query("SELECT * FROM bars").columns

    def test_minutes_view(self, store):
        ts = pd.date_range("2025-09-15 09:15", periods=5, freq="min")
        store.write_minutes(pd.DataFrame({
            "isin": "INE1", "ts": ts, "open": 1.0, "high": 2.0, "low": 0.5,
            "close": 1.5, "volume": 10, "source": "upstox",
        }))
        store.write_bars(frame(bar()))
        assert store.query("SELECT COUNT(*) AS n FROM minutes").iloc[0]["n"] == 5

    def test_parameterised_query(self, store):
        store.write_bars(frame(bar(isin="INE1"), bar(isin="INE2")))
        out = store.query("SELECT * FROM bars WHERE isin = ?", ["INE1"])
        assert len(out) == 1
