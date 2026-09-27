# Chakra Terminal — NSE technical screener

End-of-day technical screener for NSE equities. Adjusted (Upstox) basis,
DuckDB/parquet store, ISIN keys. Ships with a single-file HTML dashboard
(`dashboard/index.html`) that renders 40+ swing/momentum/mean-reversion
screens plus a full strategy audit portfolio.

Live dashboard (private artifact): https://claude.ai/artifact/4XaB2txyq2RhRh1b3xe2Yu

---

## What's in this repo

```
nse-screener/
├── screener/           Core library: schema, store, indicators, rules engine,
│                       market gate, portfolio, quality gate
├── scripts/            60+ CLI scripts:
│   ├── daily_update.sh   the daily pipeline (fetch → screen → export → publish)
│   ├── run_screen.py     compute features + evaluate every rule
│   ├── strategy_audit.py THE source of truth for real-money edge per screen
│   ├── export_dashboard.py builds dashboard/data.json + series/hourly shards
│   ├── volume_lab.py, bottom_lab.py, harden_playbook.py … research labs
│   └── (many more, mostly historical studies)
├── config/rules.yaml   Every screen: expression, description, why, manage text
├── tests/              15 test modules, ~120 tests, network-blocked in CI
├── dashboard/          Single-file HTML app + JSON data shards + charts lib
├── data/
│   ├── bars/           Daily OHLC per year (81MB)
│   ├── raw/            Bhavcopy dumps, corporate actions, instrument master
│   ├── universe/       Point-in-time index constituents
│   ├── meta/           Quarantine log, ingest attempts
│   ├── screen/         Feature panel + all screen outputs + audit JSONs
│   └── backtest_panels/ Extended 8-year panel for strategy_audit.py (485MB)
├── CLAUDE.md           15 lessons from backtesting mistakes — READ FIRST
└── requirements.txt    Python deps (pandas, pyarrow, duckdb, requests, pyyaml…)
```

Not included in this tarball (regeneratable):
- `data/minutes/` — 2GB of 1-minute bars (rebuild with `scripts/backfill_minutes.py`)
- `logs/`, `__pycache__/`, `.pytest_cache/`

---

## Getting it running locally

**Prerequisites**
- Python 3.11 (or 3.10+)
- `pip install -r requirements.txt`
- No API tokens. Upstox historical-candle is free/unauthenticated; NSE
  bhavcopy uses a session cookie warmed automatically.

**Just look at what's here (no refresh):**
```bash
tar -xzf nse-screener-full.tar.gz
cd nse-screener
python3 -m http.server 8000 --directory dashboard
# open http://localhost:8000 in a browser
```
The dashboard loads with today's screens, portfolio audit, backtest views.

**Refresh with today's data:**
```bash
bash scripts/daily_update.sh
```
This: pulls today's bars → recomputes features → evaluates all rules →
exports dashboard JSON. No auth needed. Optional: `MCAP_FLOOR_CR=2000` or
`DAYS=15` env vars.

**Re-run the strategy audit (real-money source of truth):**
```bash
python3 scripts/strategy_audit.py
```
Uses `data/backtest_panels/bottom_panel.parquet` (8yr) + `data/screen/features.parquet`
(3yr) → writes `data/screen/strategy_audit.json`. Takes ~5 min.

**Run tests:**
```bash
python3 -m pytest
```

---

## Key files by purpose

### The data
- `data/screen/features.parquet` — 1.05M rows × ~80 columns. Every stock, every
  day, ~3 years back. Every indicator (SMA/EMA/RSI/ATR/Bollinger/pivots/CPR/RS).
  Every screen queries this file. **This is the master data source.**
- `data/screen/latest_features.parquet` — the most recent day sliced out.
- `data/screen/hits.parquet` — which stocks matched which screens today.
- `data/screen/gate_r3.parquet` — daily market gate (breadth + Zweig thrust).
- `data/screen/market_state.parquet` — 74 daily market features.
- `data/screen/trades_tagged.parquet` — 34,857 historical trades for loss forensics.
- `data/backtest_panels/bottom_panel.parquet` — 2.5M rows, 2017-2026, extended
  panel with market gate baked in for the strategy audit.

### The screens
- `config/rules.yaml` — 40+ rules. Each has a df.eval expression, description,
  `why` (rationale), `manage` (execution text with stop/exit rules).
- Prefixes:
  - `SW - …` = swing-lab-validated, hard stops, tradeable
  - `VL - …` = volume-lab, breakout/spring family
  - no prefix = watchlist / reference / legacy

### The audit
- `scripts/strategy_audit.py` — for every tradeable rule: contamination drop,
  shipped features only, fresh-event detection (20 quiet), family-matched
  exit, close-based ATR stop from fill, 10-slot concurrent portfolio,
  0.50% RT cost, daily equity mark-to-market. Writes to
  `data/screen/strategy_audit.json`. The Portfolio tab of the dashboard
  renders this exact file.

### The dashboard
- `dashboard/index.html` — single-file app. Reads `data.json` + `series_*.json`
  + `hourly_*.json`. Views: Market, Screens, Chart, Money Flow, Watchlist,
  **Backtest → Portfolio** (the star-rated audit), Console.
- `scripts/export_dashboard.py` — builds all the JSON shards from parquet.

---

## Reading `CLAUDE.md` before making changes

15 hard-won lessons from real backtesting mistakes. The critical ones:
- **#1** Match EXIT to SETUP FAMILY (mean-reversion vs trend-continuation).
- **#2** Test on the FULL 8-year panel, not just the recent window.
- **#7** Close-based ATR stop, always — wicks are false shakeouts.
- **#13** Max DD from DAILY equity mark-to-market, not event-time compounding.
- **#14** Grooming numbers in `rules.yaml` are STALE; `strategy_audit.json`
  is the real-money source of truth.
- **#15** Setup family and exit mechanic must not fight each other.

Do not ship a new screen without an audit entry.

---

## Data sources

Two providers, deliberately:
- **Upstox historical-candle API** — back-adjusted daily + 1-minute bars.
  Needs `UPSTOX_ACCESS_TOKEN`. Provides the adjusted OHLC block.
- **NSE UDiFF bhavcopy** — raw (unadjusted) daily bars. No auth but needs a
  warmed cookie. Provides the raw OHLC block used for pivots and CPR.

Where a bar has both, `close / adj_close` = the empirical cumulative
adjustment factor (used to reconstruct historical raw prices).

Rejected: Kite Connect (₹2000/mo + daily token, no benefit), yfinance
(1-minute is only last 7 days).

---

## Universe

- Today's NSE equities with own market cap ≥ ₹1,000 Cr
  (`MCAP_FLOOR_CR` env, default 1000). About 1,474 names as of Sep 2026.
- Point-in-time index membership (`screener/universe.py`) — applying today's
  rebalance never changes what `members_on(<past_date>)` returns.

---

## Design notes (from stage 1-2, still true)

- **Everything is keyed on ISIN.** Symbols get renamed and reused.
- **Writes are column-wise upserts.** Nulls don't overwrite stored values;
  the raw and adjusted legs can arrive in any order.
- **Ingestion is idempotent and gap-filling.** Every (date, source) attempt
  is logged. A holiday records `no_data` and never retries.
- **Bad rows are quarantined, not dropped.** `high < low`, non-positive
  prices, wrapped signed volumes, placeholder demerger ISINs — all parked
  in `meta/quarantine.parquet` with a reason.

## Real data defects handled (with regression tests)
- Zero-price bars (529 bars, mostly 2003-04) → quarantined
- Frozen quotes on halted stocks (27 bars) → quarantined
- Signed 32-bit volume overflow (IDEA 2024-08-30: -81M → +4.2B) → repaired
- Placeholder constituents during demergers (`Dummy HEG Ltd.`) → quarantined
- NSE Akamai 200-with-HTML "Access Denied" → treated as error, not data
- Muhurat trading (18:00-18:59 evening session) → recognized

---

## License

Personal use. Not for redistribution.
