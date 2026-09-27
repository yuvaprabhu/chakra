"""Index constituents, ISIN mapping and membership history.

Everything is keyed on ISIN. A membership row is a half-open interval
``[from_date, to_date)`` so a rebalance applied today cannot retroactively
change what the screener would have seen on a past date — the single most
important property here, because a backtest that screens today's Nifty 500
over 2018 is survivorship-biased fiction.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import schema as S
from .store import Store

log = logging.getLogger(__name__)

# NSE publishes ind_nifty500list.csv with these headers.
_CSV_COLUMNS = {
    "company name": "name",
    "industry": "industry",
    "symbol": "symbol",
    "series": "series",
    "isin code": "isin",
}

# Tradable equity ISINs are INE (companies) or INF (funds/ETFs). NSE also
# ships placeholder rows such as "Dummy HEG Ltd." with ISIN DUM545A01024
# during demergers; those are quarantined, not loaded.
_ISIN_RE = re.compile(r"^IN[EF][A-Z0-9]{9}$")


class UniverseError(Exception):
    pass


@dataclass
class SnapshotResult:
    """What one constituent-list snapshot changed."""

    index_name: str
    as_of: pd.Timestamp
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    renamed: list[tuple[str, str, str]] = field(default_factory=list)  # (isin, old, new)
    unchanged: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed or self.renamed)

    def __str__(self) -> str:
        return (
            f"{self.index_name} @ {self.as_of.date()}: "
            f"+{len(self.added)} -{len(self.removed)} "
            f"~{len(self.renamed)} renamed, {self.unchanged} unchanged"
        )


def parse_constituents_csv(
    path_or_buf, *, index_name: str | None = None, strict: bool = False
) -> pd.DataFrame:
    """Parse an NSE ``ind_nifty500list.csv`` into (isin, symbol, name, series, industry).

    Rows whose ISIN is not a real equity ISIN are dropped and listed on
    ``df.attrs["excluded"]`` (as records) rather than failing the load — NSE routinely ships
    one or two placeholder rows and a screener that refuses to start on a
    normal trading day is worse than one that skips a dummy. Pass ``strict`` to
    raise instead.
    """
    raw = pd.read_csv(path_or_buf, dtype=str).fillna("")
    raw.columns = [c.strip().lower() for c in raw.columns]
    missing = [c for c in ("symbol", "isin code") if c not in raw.columns]
    if missing:
        raise UniverseError(
            f"constituent CSV missing columns {missing}; got {list(raw.columns)}"
        )
    out = raw.rename(columns=_CSV_COLUMNS)
    for col in ("isin", "symbol", "name", "series", "industry"):
        out[col] = out[col].str.strip() if col in out.columns else ""
    out = out[["isin", "symbol", "name", "series", "industry"]]

    valid = out["isin"].str.match(_ISIN_RE)
    excluded = out[~valid].reset_index(drop=True)
    if len(excluded):
        if strict:
            raise UniverseError(
                f"{len(excluded)} rows with non-equity ISIN, e.g. "
                f"{excluded.head(3)[['symbol', 'isin']].to_dict('records')}"
            )
        log.warning(
            "dropping %d constituent row(s) with non-equity ISIN: %s",
            len(excluded), excluded[["symbol", "isin"]].to_dict("records"),
        )
    out = out[valid]

    dupes = out[out.duplicated(subset=["isin"], keep=False)]
    if len(dupes):
        raise UniverseError(f"duplicate ISINs in constituent list: {sorted(set(dupes['isin']))}")
    if index_name is not None:
        out.insert(0, "index_name", index_name)
    out = out.reset_index(drop=True)
    # Records, not a DataFrame: pandas compares .attrs when concatenating and a
    # DataFrame there raises "truth value is ambiguous", which breaks any
    # concat of parsed constituent lists.
    out.attrs["excluded"] = excluded.to_dict("records")
    return out


class Universe:
    def __init__(self, store: Store):
        self.store = store

    # --- symbol master -----------------------------------------------------

    def observe_symbols(self, obs: pd.DataFrame) -> list[tuple[str, str, str]]:
        """Record (isin, symbol, name, date) sightings; return detected renames.

        Called on every ingest with that day's bhavcopy rows, so the symbol
        master tracks reality rather than whatever the last index CSV said.
        """
        if obs is None or len(obs) == 0:
            return []
        obs = obs.copy()
        obs["date"] = pd.to_datetime(obs["date"])
        for col in ("name", "series"):
            if col not in obs.columns:
                obs[col] = pd.NA
        obs = obs.dropna(subset=["isin", "symbol"])
        # Keep the last sighting per (isin, symbol) plus overall bounds per ISIN.
        obs = obs.sort_values("date")

        symbols = self.store.read_symbols().set_index("isin")
        history = self.store.read_symbol_history()
        renames: list[tuple[str, str, str]] = []

        hist_rows = history.to_dict("records")
        sym_rows = {isin: row for isin, row in symbols.iterrows()}

        for isin, grp in obs.groupby("isin", sort=True):
            first, last = grp["date"].iloc[0], grp["date"].iloc[-1]
            latest = grp.iloc[-1]
            new_symbol = str(latest["symbol"])
            prior = sym_rows.get(isin)

            if prior is None:
                sym_rows[isin] = pd.Series({
                    "symbol": new_symbol,
                    "name": latest.get("name"),
                    "series": latest.get("series"),
                    "first_seen": first,
                    "last_seen": last,
                })
                hist_rows.append({
                    "isin": isin, "symbol": new_symbol,
                    "from_date": first, "to_date": pd.NaT,
                })
                continue

            old_symbol = str(prior["symbol"])
            if new_symbol != old_symbol and last >= pd.Timestamp(prior["last_seen"]):
                # Rename: close the old interval, open a new one. The ISIN — and
                # therefore the price history — is untouched.
                for row in hist_rows:
                    if row["isin"] == isin and row["symbol"] == old_symbol and pd.isna(row["to_date"]):
                        row["to_date"] = last
                hist_rows.append({
                    "isin": isin, "symbol": new_symbol,
                    "from_date": last, "to_date": pd.NaT,
                })
                renames.append((isin, old_symbol, new_symbol))
                prior["symbol"] = new_symbol
            elif new_symbol == old_symbol:
                if not any(
                    r["isin"] == isin and r["symbol"] == new_symbol for r in hist_rows
                ):
                    hist_rows.append({
                        "isin": isin, "symbol": new_symbol,
                        "from_date": first, "to_date": pd.NaT,
                    })

            if pd.isna(prior["first_seen"]) or first < pd.Timestamp(prior["first_seen"]):
                prior["first_seen"] = first
            if pd.isna(prior["last_seen"]) or last > pd.Timestamp(prior["last_seen"]):
                prior["last_seen"] = last
            for col in ("name", "series"):
                val = latest.get(col)
                if pd.notna(val) and str(val):
                    prior[col] = val
            sym_rows[isin] = prior

        out = pd.DataFrame.from_dict(sym_rows, orient="index")
        out.index.name = "isin"
        out = out.reset_index()
        self.store.write_symbols(out.sort_values("isin"))
        hist = pd.DataFrame(hist_rows)
        hist = hist.drop_duplicates(subset=["isin", "symbol", "from_date"], keep="last")
        self.store.write_symbol_history(hist.sort_values(["isin", "from_date"]))
        return renames

    def symbol_on(self, isin: str, date) -> str | None:
        """The symbol this ISIN traded under on ``date``."""
        date = pd.Timestamp(date)
        hist = self.store.read_symbol_history()
        rows = hist[hist["isin"] == isin]
        if rows.empty:
            return None
        live = rows[
            (rows["from_date"] <= date)
            & (rows["to_date"].isna() | (rows["to_date"] > date))
        ]
        if live.empty:
            # Before the first sighting: fall back to the earliest known symbol.
            return str(rows.sort_values("from_date")["symbol"].iloc[0])
        return str(live.sort_values("from_date")["symbol"].iloc[-1])

    def isin_for_symbol(self, symbol: str, date=None) -> list[str]:
        """ISINs that traded under ``symbol``.

        Returns a list because NSE reuses symbols: more than one result means
        the symbol was recycled, and anything keyed on it is ambiguous.
        """
        hist = self.store.read_symbol_history()
        rows = hist[hist["symbol"] == symbol]
        if rows.empty:
            return []
        if date is not None:
            date = pd.Timestamp(date)
            rows = rows[
                (rows["from_date"] <= date)
                & (rows["to_date"].isna() | (rows["to_date"] > date))
            ]
        return sorted(set(rows["isin"].tolist()))

    def reused_symbols(self) -> pd.DataFrame:
        """Symbols mapped to more than one ISIN — each one is a trap for any
        symbol-keyed join, so surface them rather than silently merging."""
        hist = self.store.read_symbol_history()
        if hist.empty:
            return hist
        counts = hist.groupby("symbol")["isin"].nunique()
        offenders = counts[counts > 1].index
        return hist[hist["symbol"].isin(offenders)].sort_values(["symbol", "from_date"])

    # --- index membership --------------------------------------------------

    def apply_snapshot(
        self,
        index_name: str,
        as_of,
        constituents: pd.DataFrame,
        *,
        allow_backdate: bool = False,
    ) -> SnapshotResult:
        """Fold one constituent list, observed on ``as_of``, into membership history.

        Names not in the snapshot get ``to_date = as_of`` (they were last members
        the day before); new names get ``from_date = as_of``. Re-applying the same
        snapshot is a no-op.
        """
        as_of = pd.Timestamp(as_of).normalize()
        cons = constituents.copy()
        if "isin" not in cons.columns or "symbol" not in cons.columns:
            raise UniverseError("constituents need isin and symbol columns")
        cons = cons.drop_duplicates(subset=["isin"], keep="first")
        if "name" not in cons.columns:
            cons["name"] = pd.NA

        members = self.store.read_membership()
        mine = members[members["index_name"] == index_name]
        others = members[members["index_name"] != index_name]

        if not mine.empty and not allow_backdate:
            latest = pd.concat([mine["from_date"], mine["to_date"].dropna()]).max()
            if pd.notna(latest) and as_of < latest:
                raise UniverseError(
                    f"snapshot for {as_of.date()} predates recorded history to "
                    f"{pd.Timestamp(latest).date()}; snapshots must be applied in "
                    "chronological order (pass allow_backdate=True only when rebuilding)"
                )

        open_rows = mine["to_date"].isna()
        current = mine[open_rows]
        current_isins = set(current["isin"])
        snap_isins = set(cons["isin"])

        added = sorted(snap_isins - current_isins)
        removed = sorted(current_isins - snap_isins)
        retained = snap_isins & current_isins

        mine = mine.copy()
        # Close out departures.
        if removed:
            mask = mine["to_date"].isna() & mine["isin"].isin(removed)
            mine.loc[mask, "to_date"] = as_of

        # Carry symbol changes onto the live row; ISIN stays the join key.
        renamed: list[tuple[str, str, str]] = []
        snap_by_isin = cons.set_index("isin")
        for isin in sorted(retained):
            mask = mine["to_date"].isna() & (mine["isin"] == isin)
            if not mask.any():
                continue
            old_symbol = str(mine.loc[mask, "symbol"].iloc[0])
            new_symbol = str(snap_by_isin.loc[isin, "symbol"])
            if old_symbol != new_symbol:
                mine.loc[mask, "symbol"] = new_symbol
                renamed.append((isin, old_symbol, new_symbol))
            name = snap_by_isin.loc[isin, "name"]
            if pd.notna(name) and str(name):
                mine.loc[mask, "name"] = name

        new_rows = pd.DataFrame(
            [
                {
                    "index_name": index_name,
                    "isin": isin,
                    "symbol": snap_by_isin.loc[isin, "symbol"],
                    "name": snap_by_isin.loc[isin, "name"],
                    "from_date": as_of,
                    "to_date": pd.NaT,
                }
                for isin in added
            ]
        )

        merged = pd.concat(
            [df for df in (others, mine, new_rows) if len(df)], ignore_index=True
        )
        merged = S.coerce(merged, S.MEMBERSHIP_DTYPES).sort_values(
            ["index_name", "isin", "from_date"]
        )
        self.store.write_membership(merged)

        # Snapshot rows are also symbol sightings.
        self.observe_symbols(
            cons.assign(date=as_of)[["isin", "symbol", "name", "date"]]
        )

        return SnapshotResult(
            index_name=index_name,
            as_of=as_of,
            added=added,
            removed=removed,
            renamed=renamed,
            unchanged=len(retained) - len(renamed),
        )

    def load_constituents_file(
        self, path: str | Path, index_name: str, as_of, **kwargs
    ) -> SnapshotResult:
        cons = parse_constituents_csv(path)
        return self.apply_snapshot(index_name, as_of, cons, **kwargs)

    def members_on(self, index_name: str, date) -> pd.DataFrame:
        """Constituents as of ``date`` — point-in-time, no survivorship bias."""
        date = pd.Timestamp(date).normalize()
        members = self.store.read_membership()
        if members.empty:
            return members
        rows = members[
            (members["index_name"] == index_name)
            & (members["from_date"] <= date)
            & (members["to_date"].isna() | (members["to_date"] > date))
        ]
        return rows.sort_values("symbol").reset_index(drop=True)

    def isins_on(self, index_name: str, date) -> list[str]:
        return self.members_on(index_name, date)["isin"].tolist()

    def membership_changes(self, index_name: str) -> pd.DataFrame:
        """Every join and leave event, for auditing a rebalance."""
        members = self.store.read_membership()
        rows = members[members["index_name"] == index_name]
        if rows.empty:
            return pd.DataFrame(columns=["date", "event", "isin", "symbol"])
        joins = rows[["from_date", "isin", "symbol"]].rename(columns={"from_date": "date"})
        joins["event"] = "join"
        leaves = rows.dropna(subset=["to_date"])[["to_date", "isin", "symbol"]].rename(
            columns={"to_date": "date"}
        )
        leaves["event"] = "leave"
        out = pd.concat([joins, leaves], ignore_index=True)
        return out[["date", "event", "isin", "symbol"]].sort_values(["date", "event", "symbol"]).reset_index(drop=True)
