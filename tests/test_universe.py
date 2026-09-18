"""Symbol master and index membership.

The property that matters most: a rebalance applied today must not change what
``members_on`` returns for a past date. Everything else here is in service of
keeping ISIN, not symbol, as the identity of a stock.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from screener.universe import Universe, UniverseError, parse_constituents_csv


CSV = """Company Name,Industry,Symbol,Series,ISIN Code
Reliance Industries Ltd.,Oil Gas & Consumable Fuels,RELIANCE,EQ,INE002A01018
Infosys Ltd.,Information Technology,INFY,EQ,INE009A01021
HDFC Bank Ltd.,Financial Services,HDFCBANK,EQ,INE040A01034
"""


@pytest.fixture
def universe(store):
    return Universe(store)


def cons(*triples):
    return pd.DataFrame(
        [{"isin": i, "symbol": s, "name": n} for i, s, n in triples]
    )


class TestParseConstituents:
    def test_parses_nse_columns(self):
        df = parse_constituents_csv(io.StringIO(CSV))
        assert len(df) == 3
        assert set(df.columns) == {"isin", "symbol", "name", "series", "industry"}
        assert df.iloc[0]["isin"] == "INE002A01018"
        assert df.iloc[0]["symbol"] == "RELIANCE"

    def test_quarantines_placeholder_isin(self):
        """NSE ships rows like 'Dummy HEG Ltd.' with ISIN DUM545A01024 during
        demergers. Refusing to start on those would be worse than skipping."""
        text = CSV + "Dummy HEG Ltd.,Metals & Mining,DUMMYHEG,EQ,DUM545A01024\n"
        df = parse_constituents_csv(io.StringIO(text))
        assert len(df) == 3
        assert df.attrs["excluded"].iloc[0]["symbol"] == "DUMMYHEG"

    def test_strict_mode_raises_instead(self):
        text = CSV + "Dummy HEG Ltd.,Metals & Mining,DUMMYHEG,EQ,DUM545A01024\n"
        with pytest.raises(UniverseError, match="non-equity ISIN"):
            parse_constituents_csv(io.StringIO(text), strict=True)

    def test_rejects_missing_columns(self):
        with pytest.raises(UniverseError, match="missing columns"):
            parse_constituents_csv(io.StringIO("Company Name,Symbol\nFoo,BAR\n"))

    def test_rejects_duplicate_isin(self):
        text = CSV + "Reliance Again,Oil,RELIANCE2,EQ,INE002A01018\n"
        with pytest.raises(UniverseError, match="duplicate ISIN"):
            parse_constituents_csv(io.StringIO(text))


class TestPointInTimeMembership:
    def test_rebalance_does_not_rewrite_the_past(self, universe):
        """The survivorship-bias guard. HDFC was a member in 2023; adding INFY
        and dropping HDFC in 2024 must leave the 2023 answer alone."""
        universe.apply_snapshot("NIFTY500", "2023-01-01", cons(
            ("INE001A01036", "HDFC", "HDFC Ltd"),
            ("INE002A01018", "RELIANCE", "Reliance"),
        ))
        before = universe.isins_on("NIFTY500", "2023-06-01")

        universe.apply_snapshot("NIFTY500", "2024-01-01", cons(
            ("INE002A01018", "RELIANCE", "Reliance"),
            ("INE009A01021", "INFY", "Infosys"),
        ))

        assert universe.isins_on("NIFTY500", "2023-06-01") == before
        assert set(before) == {"INE001A01036", "INE002A01018"}
        assert set(universe.isins_on("NIFTY500", "2024-06-01")) == {
            "INE002A01018", "INE009A01021"
        }

    def test_interval_is_half_open(self, universe):
        """from_date inclusive, to_date exclusive: a name leaving on D is a
        member on D-1 and not on D."""
        universe.apply_snapshot("NIFTY500", "2023-01-01", cons(("INE1", "A", "A Ltd")))
        universe.apply_snapshot("NIFTY500", "2024-01-01", cons(("INE2", "B", "B Ltd")))
        assert "INE1" in universe.isins_on("NIFTY500", "2023-12-31")
        assert "INE1" not in universe.isins_on("NIFTY500", "2024-01-01")
        assert "INE2" in universe.isins_on("NIFTY500", "2024-01-01")

    def test_before_first_snapshot_is_empty(self, universe):
        universe.apply_snapshot("NIFTY500", "2023-01-01", cons(("INE1", "A", "A Ltd")))
        assert universe.isins_on("NIFTY500", "2022-06-01") == []

    def test_reapplying_same_snapshot_changes_nothing(self, universe):
        snap = cons(("INE1", "A", "A Ltd"), ("INE2", "B", "B Ltd"))
        universe.apply_snapshot("NIFTY500", "2023-01-01", snap)
        res = universe.apply_snapshot("NIFTY500", "2023-02-01", snap)
        assert not res.changed
        assert res.unchanged == 2
        assert len(universe.store.read_membership()) == 2

    def test_rejoining_creates_a_second_interval(self, universe):
        universe.apply_snapshot("NIFTY500", "2023-01-01", cons(("INE1", "A", "A Ltd")))
        universe.apply_snapshot("NIFTY500", "2024-01-01", cons(("INE2", "B", "B Ltd")))
        universe.apply_snapshot("NIFTY500", "2025-01-01", cons(
            ("INE1", "A", "A Ltd"), ("INE2", "B", "B Ltd")
        ))
        assert "INE1" in universe.isins_on("NIFTY500", "2023-06-01")
        assert "INE1" not in universe.isins_on("NIFTY500", "2024-06-01")
        assert "INE1" in universe.isins_on("NIFTY500", "2025-06-01")

    def test_indices_are_independent(self, universe):
        universe.apply_snapshot("NIFTY500", "2023-01-01", cons(("INE1", "A", "A Ltd")))
        universe.apply_snapshot("NIFTY50", "2023-01-01", cons(("INE2", "B", "B Ltd")))
        assert universe.isins_on("NIFTY500", "2023-06-01") == ["INE1"]
        assert universe.isins_on("NIFTY50", "2023-06-01") == ["INE2"]

    def test_backdated_snapshot_is_refused(self, universe):
        universe.apply_snapshot("NIFTY500", "2024-01-01", cons(("INE1", "A", "A Ltd")))
        with pytest.raises(UniverseError, match="chronological order"):
            universe.apply_snapshot("NIFTY500", "2023-01-01", cons(("INE2", "B", "B Ltd")))

    def test_membership_changes_audit(self, universe):
        universe.apply_snapshot("NIFTY500", "2023-01-01", cons(("INE1", "A", "A Ltd")))
        universe.apply_snapshot("NIFTY500", "2024-01-01", cons(("INE2", "B", "B Ltd")))
        ch = universe.membership_changes("NIFTY500")
        events = set(zip(ch["event"], ch["isin"]))
        assert ("join", "INE1") in events
        assert ("leave", "INE1") in events
        assert ("join", "INE2") in events


class TestSymbolIdentity:
    def test_rename_keeps_one_isin_series(self, universe):
        """A rename must not split one stock's history into two series."""
        universe.observe_symbols(pd.DataFrame([
            {"isin": "INE1", "symbol": "OLDNAME", "date": "2024-01-01"},
            {"isin": "INE1", "symbol": "OLDNAME", "date": "2024-06-01"},
        ]))
        universe.observe_symbols(pd.DataFrame([
            {"isin": "INE1", "symbol": "NEWNAME", "date": "2025-01-01"},
        ]))
        assert len(universe.store.read_symbols()) == 1
        assert universe.symbol_on("INE1", "2024-03-01") == "OLDNAME"
        assert universe.symbol_on("INE1", "2025-06-01") == "NEWNAME"

    def test_rename_is_reported(self, universe):
        universe.observe_symbols(pd.DataFrame([
            {"isin": "INE1", "symbol": "OLD", "date": "2024-01-01"}]))
        renames = universe.observe_symbols(pd.DataFrame([
            {"isin": "INE1", "symbol": "NEW", "date": "2025-01-01"}]))
        assert renames == [("INE1", "OLD", "NEW")]

    def test_symbol_reuse_across_isins_is_surfaced(self, universe):
        """NSE recycles symbols. Two ISINs under one symbol must be visible,
        not silently merged."""
        universe.observe_symbols(pd.DataFrame([
            {"isin": "INE1", "symbol": "SHARED", "date": "2020-01-01"}]))
        universe.observe_symbols(pd.DataFrame([
            {"isin": "INE2", "symbol": "SHARED", "date": "2024-01-01"}]))
        reused = universe.reused_symbols()
        assert set(reused["isin"]) == {"INE1", "INE2"}
        assert sorted(universe.isin_for_symbol("SHARED")) == ["INE1", "INE2"]

    def test_isin_for_symbol_at_a_date(self, universe):
        universe.observe_symbols(pd.DataFrame([
            {"isin": "INE1", "symbol": "OLD", "date": "2024-01-01"}]))
        universe.observe_symbols(pd.DataFrame([
            {"isin": "INE1", "symbol": "NEW", "date": "2025-01-01"}]))
        assert universe.isin_for_symbol("OLD", "2024-06-01") == ["INE1"]
        assert universe.isin_for_symbol("OLD", "2025-06-01") == []

    def test_tracks_first_and_last_seen(self, universe):
        universe.observe_symbols(pd.DataFrame([
            {"isin": "INE1", "symbol": "A", "date": "2024-01-01"},
            {"isin": "INE1", "symbol": "A", "date": "2024-12-31"},
        ]))
        row = universe.store.read_symbols().iloc[0]
        assert row["first_seen"] == pd.Timestamp("2024-01-01")
        assert row["last_seen"] == pd.Timestamp("2024-12-31")

    def test_snapshot_carries_rename_onto_live_row(self, universe):
        universe.apply_snapshot("NIFTY500", "2024-01-01", cons(("INE1", "OLD", "X Ltd")))
        res = universe.apply_snapshot("NIFTY500", "2025-01-01", cons(("INE1", "NEW", "X Ltd")))
        assert res.renamed == [("INE1", "OLD", "NEW")]
        assert len(universe.store.read_membership()) == 1
        assert universe.members_on("NIFTY500", "2025-06-01").iloc[0]["symbol"] == "NEW"
