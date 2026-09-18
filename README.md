# NSE 500 Technical Screener — stages 1 & 2

Storage, symbol master, and ingestion. Indicators, rules, quality gates and the
backtest harness are not built yet: stage 3 (corporate action adjustment) is the
next one and is deliberately not started.

## Data sources

Two providers, deliberately. Neither one alone is sufficient.

| | Upstox historical-candle | NSE UDiFF bhavcopy |
|---|---|---|
| Fills | `adj_*` block + all minute bars | raw `open/high/low/close` |
| Prices | **back-adjusted** for splits/bonuses | **raw** — what the chart showed |
| Auth | none | none, but needs a warmed cookie |
| Daily history | 2000-01-03 onwards | day-by-day from ingest |
| Minute history | Jan 2022 onwards, 375/session | n/a |
| Reliability here | excellent | intermittent (Akamai) |
| Key | ISIN-native (`NSE_EQ\|INE002A01018`) | ISIN column |

**Why both.** Upstox is back-adjusted — verified against Reliance's 1:1 bonus
(ex-date 2024-10-28), where a raw series must show a ~50% gap and this one does
not. Adjusted prices are right for moving averages, Bollinger, RSI and ATR, and
wrong for pivots and CPR: after a corporate action, a monthly CPR built from
adjusted bars is off against an unadjusted close by exactly the split factor.
So raw prices come from the bhavcopy and the two land in the same row.

Where a bar carries both blocks, `close / adj_close` **is** the cumulative
adjustment factor for that date. Stage 3 gets an empirical factor curve rather
than a modelled one.

**They agree.** Across 11,500 overlapping bars the two providers match to the
last decimal on 99.66% of them, and every deviation falls on the current
trading day — the Upstox candle for today is provisional, while the bhavcopy
close is the official NSE figure (a 30-minute VWAP, not the last traded price).

Rejected: **Kite Connect** (₹2000/mo plus a daily token, no benefit here);
**yfinance** (1-minute data only for the last 7 days).

## Layout

```
screener/
  schema.py     canonical column schemas and dtype coercion
  store.py      parquet/duckdb, bars partitioned by year, minutes by year/month
  universe.py   index constituents, ISIN mapping, membership history
  fetch.py      NSE bhavcopy session, UDiFF parsing, gap-filling backfill
  upstox.py     Upstox provider: adjusted daily + 1-minute candles
scripts/
  backfill_daily.py       adjusted daily history for an index universe
  backfill_minutes.py     1-minute history, month by month
  ingest_bhavcopy.py      raw daily bars for a date range
  inspect_data.py         read-only data report (start here)
tests/
```

## What is loaded

| | |
|---|---|
| Daily bars | 1,945,279 rows · 2,959 ISINs · 2000-01-03 → 2026-09-18 · 6,643 sessions |
| Minute bars | 92,267,587 rows · 500 symbols · 2024-09 → 2026-09 · 1.2 GB |
| Universe | Nifty 500, point-in-time membership |
| Quarantined | 556 daily bars the store refused |

```bash
python3 scripts/inspect_data.py     # full report
python3 -m pytest                   # 121 tests, network hard-blocked
```

## Design notes

**Everything is keyed on ISIN.** Symbols get renamed and reused; keying on the
symbol splits one stock's history into two series and silently merges two
stocks into one. Live data already contains five symbols mapped to more than
one ISIN — `Universe.reused_symbols()` surfaces them instead of merging them.

**Membership is point-in-time.** A membership row is a half-open interval
`[from_date, to_date)`, so applying today's rebalance cannot change what
`members_on` returns for a past date. Screening today's Nifty 500 over 2018 is
survivorship-biased fiction, and this is the guard against it.

**Writes are column-wise upserts.** A null in an incoming frame leaves the
stored value alone, which is what lets the raw and adjusted legs arrive in any
order from two independent jobs without either erasing the other. `volume` is a
nullable `Int64` for the same reason: `0` is a real value and would win a merge
against a stored `1000`.

**Ingestion is idempotent and gap-filling.** Every `(date, source)` attempt is
logged. A holiday records `no_data` and is never retried; a network failure
records `error` and stays a gap. Gap detection asks whether the *raw* block is
present, not whether any bar exists — otherwise the Upstox backfill (which
covers every date since 2000) would mask every missing bhavcopy.

**The store is a hard gate; bad rows are quarantined, not dropped.** Validation
rejects `high < low`, non-positive prices, open/close outside the high/low
range, duplicate keys and negative volume. Rows that fail are parked in
`meta/quarantine.parquet` with a reason, because a screener that quietly
discards bad bars is indistinguishable from one that never saw them.

## Real data defects found and handled

Each of these came out of live data and has a regression test.

- **Zero-price bars.** 529 bars print `0.00` OHLC with non-zero volume, mostly
  2003–04. Quarantined.
- **Frozen quotes on halted stocks.** 27 bars carry a settlement close outside
  the traded high/low range. Quarantined.
- **Signed 32-bit volume overflow.** IDEA on 2024-08-30 reports
  `-81,259,413`; the true figure is `4,213,707,883`, which wrapped past 2³¹.
  Repaired by adding 2³², and only where the result becomes positive.
- **Placeholder constituents.** NSE ships rows such as `Dummy HEG Ltd.` with
  ISIN `DUM545A01024` during demergers. Quarantined rather than fatal — a
  screener that refuses to start on a normal trading day is worse.
- **Akamai 200-with-HTML.** NSE serves an "Access Denied" page with HTTP 200.
  Treated as an error, not as data.
- **Muhurat trading.** 2024-11-01 has a single 60-minute evening session at
  18:00–18:59 IST. Genuine NSE behaviour: any "375 candles per day" quality
  gate must special-case it.

## Known limitations

- **Raw prices exist only for dates the bhavcopy was ingested.** Historical raw
  OHLC is absent before that. Stage 3 is where it gets reconstructed by
  un-adjusting, using the empirical `close / adj_close` factor curve.
- **NSE is intermittent from a datacenter IP.** The retry and cookie re-warm
  recover most of the time; failed days stay gaps and are retried next run. It
  is reliable from a residential IP.
- **Minute bars are back-adjusted**, since that is what Upstox serves. They
  have no raw counterpart.
- **No trading-holiday calendar yet.** `expected_sessions` uses weekdays, which
  over-estimates. That is the safe direction: a spurious candidate date costs
  one 404, a missed one is a silent gap.
- **Today's adjusted bar is provisional** until the session settles.
