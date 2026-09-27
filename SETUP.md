# SETUP — Restore Chakra Terminal from this repo

Instructions for a Claude Code session (or any developer) to clone this
repo and get it running exactly like the cloud session it came from.

## Prerequisites

- Python 3.11 (3.10+ works)
- git, curl
- ~4GB free disk (data files) or ~1GB for code-only checkout
- **No API tokens needed.** The Upstox historical-candle API used here is
  free and unauthenticated (see `screener/upstox.py` docstring). NSE
  bhavcopy is also free but needs a warmed session cookie (handled
  automatically by `screener/fetch.py`).

## Setup path A: snapshot restore (fastest, ~5 min)

If you have the snapshot tarball or a clone with the shipped parquet
files, skip to path A. Everything is ready to run.

```bash
# 1. Clone or extract the snapshot
git clone https://github.com/yuvaprabhu/chakra.git   # if pushed
cd chakra

# 2. If features.parquet / bottom_panel.parquet were not in the clone,
#    download them from the GitHub release
mkdir -p data/screen data/backtest_panels
curl -L -o data/screen/features.parquet \
  https://github.com/yuvaprabhu/chakra/releases/download/v0-snapshot/features.parquet
curl -L -o data/backtest_panels/bottom_panel.parquet \
  https://github.com/yuvaprabhu/chakra/releases/download/v0-snapshot/bottom_panel.parquet

# 3. Python env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4. Sanity check
python3 -c "import pandas as pd; f = pd.read_parquet('data/screen/features.parquet'); print(f'features: {len(f):,} rows, last={f.date.max().date()}')"

# 5. Serve the dashboard (uses snapshot state)
python3 -m http.server 8000 --directory dashboard
# open http://localhost:8000
```

## Setup path B: fresh cold start (no data included, ~3-5 hours)

If you only have the code and nothing else, do this. Every step is
idempotent and resumable, so a network hiccup does not restart the run.
NO tokens or API keys needed at any step.

```bash
# 0. Extract code (from bundle or tarball), install deps
git clone <your-clone-source> chakra   # or tar -xzf; or bundle
cd chakra
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Instrument master (~5 sec). Downloads the NSE symbol list from
#    Upstox and writes data/raw/upstox_nse_instruments.csv. The universe
#    scan reads this to know which ISINs to hit.
python3 -c "from screener.upstox import load_instruments; load_instruments('data/raw/upstox_nse_instruments.csv', refresh=True); print('instruments loaded')"

# 2. Scan market cap for every NSE equity (~30-60 min).
#    Yahoo Finance is rate-limited — 3 workers, ~5 req/s. Resumable:
#    stops write a checkpoint, next run picks up where it left off.
#    Writes data/raw/all_nse_mcap.parquet (5137 equities).
python3 scripts/scan_all_mcap.py

# 3. Promote scan into the fundamentals table (~1 sec). The universe
#    filter (>=Rs 1000 Cr by default) reads from this table.
python3 scripts/sync_fundamentals.py

# 4. Backfill daily bars for every universe name from 2017-01-01
#    onwards (~2-4 hours, 1474 names × ~2500 sessions each).
#    Both raw (bhavcopy) and adjusted (Upstox) blocks.
#    Idempotent: names with enough history are skipped on rerun.
python3 scripts/sync_universe.py --start 2017-01-01 --skip-minutes

# 5. Fill anything the sync missed for the last 30 days (~2 min).
python3 scripts/sync_today.py --days 30

# 6. Index bars: NIFTY, VIX, sector indices (~1 min).
python3 scripts/ingest_indices.py

# 7. Corporate actions — used by the quality gate to mark
#    contaminated windows around unadjusted splits/bonuses (~1 min).
python3 scripts/fetch_corporate_actions.py

# 8. Compute features and evaluate every rule (~5 min).
#    Writes data/screen/features.parquet + hits.parquet + latest_features.
python3 scripts/run_screen.py

# 9. Build the 8-year extended backtest panel (~30 min). This is what
#    scripts/strategy_audit.py needs for its 8yr coverage rows.
python3 scripts/bottom_lab.py

# 10. Run the strategy audit (~5 min).
#     Writes data/screen/strategy_audit.json.
python3 scripts/strategy_audit.py

# 11. Export dashboard JSON shards (~30 sec).
python3 scripts/export_dashboard.py

# 12. Serve the dashboard.
python3 -m http.server 8000 --directory dashboard
# open http://localhost:8000
```

Total: 3-5 hours of mostly-unattended time. Do the long steps (2, 4, 9)
overnight or in the background — they can all be killed and resumed.

If Yahoo blocks step 2 with 429s, cut concurrency (edit `WORKERS = 1`
in `scripts/scan_all_mcap.py`), wait an hour, retry. If NSE bhavcopy
gets Akamai-blocked in step 4, the script logs errors and continues;
missed days retry next run.

## Refresh data with today's session (optional)

```bash
bash scripts/daily_update.sh
```
This: pulls today's bars → recomputes features → evaluates every rule →
rebuilds dashboard JSON. Takes ~5 minutes with warm cache. No auth needed.

Optional env vars:
- `MCAP_FLOOR_CR=2000` — restrict universe to ≥₹2000 Cr (default 1000)
- `DAYS=15` — catch up after a break (default 5-day lookback)

## Re-run the strategy audit (5-10 min)

```bash
python3 scripts/strategy_audit.py
```
Writes `data/screen/strategy_audit.json`. The dashboard's Portfolio tab
reads this file — refresh the browser to see updated numbers.

## Directory reference (for a Claude session that has never seen this repo)

- `screener/` — core library, imported by every script. Read
  `rules.py`, `indicators.py`, `market.py` first.
- `scripts/run_screen.py` — the daily screener. Reads bars, computes
  features, evaluates rules from `config/rules.yaml`, writes
  `data/screen/features.parquet` and `data/screen/hits.parquet`.
- `scripts/strategy_audit.py` — the real-money audit. Reads
  `data/backtest_panels/bottom_panel.parquet` (8yr) and/or
  `data/screen/features.parquet` (3yr), writes
  `data/screen/strategy_audit.json`. This file is the source of truth
  for the Portfolio tab.
- `scripts/export_dashboard.py` — builds `dashboard/data.json` and the
  series/hourly shards from the parquet files.
- `dashboard/index.html` — single-file HTML app. Reads JSON shards next
  to it. Chakra artifact URL:
  https://claude.ai/artifact/4XaB2txyq2RhRh1b3xe2Yu
- `CLAUDE.md` — 15 lessons from backtesting mistakes. READ FIRST before
  changing anything about backtesting, screens, or exits. Lessons #1
  (exit matches setup family), #13 (Max DD from daily equity), #14
  (audit is truth over grooming numbers), and #15 (entry ≠ exit) are
  the ones most easily broken by well-meaning edits.

## What state was this project in at snapshot?

- 40+ rules in `config/rules.yaml`
- 15 SW/VL swing-lab-validated screens with hard stops
- Strategy audit run over 41 rules → 36 traded, 5 watchlist
- Dashboard v50 published, Portfolio tab with star ratings
- Chakra artifact: https://claude.ai/artifact/4XaB2txyq2RhRh1b3xe2Yu
- Universe: 1,474 NSE equities ≥ ₹1,000 Cr as of Sep 2026

## Common tasks

**Add a new screen:**
1. Add rule to `config/rules.yaml` (expression, why, manage text)
2. Add canonical clauses to `tests/test_definitions.py`
3. Run `python3 -m pytest tests/test_definitions.py`
4. Run `python3 scripts/run_screen.py --rules config/rules.yaml`
5. Run `python3 scripts/strategy_audit.py` to get the new screen's
   real-money numbers into the Portfolio tab
6. Update `manage:` text with the audit numbers (win%, CAGR, MaxDD)
7. Never quote grooming-window numbers in `manage:` text (lesson #14)

**Debug a screen showing zero hits:**
- `python3 -c "import pandas as pd; f = pd.read_parquet('data/screen/hits.parquet'); print(f[f.rule=='sw_rsi2_snapback'])"`
- If empty, the rule expression is filtering everything out — check
  each clause against `data/screen/latest_features.parquet` columns.
- Common causes: typo in column name, market gate off, feature not on
  today's panel.

**Recompute the 8-year backtest panel** (needed if data/backtest_panels
gets corrupted):
```bash
python3 scripts/bottom_lab.py
```
Takes ~30 min. Writes to `data/backtest_panels/bottom_panel.parquet`.

## Restrictions

- Never edit files under `data/` by hand; they're regenerated. Data
  changes go through the pipeline.
- Never quote a model identifier in commits, PR bodies, or artifact
  content.
- Do not push data files > 100MB to git (GitHub blocks them). Big
  parquets go as release assets — see `scripts/publish_release.sh`
  (if it exists) or use `gh release upload`.
