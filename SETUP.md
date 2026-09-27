# SETUP — Restore Chakra Terminal from this repo

Instructions for a Claude Code session (or any developer) to clone this
repo and get it running exactly like the cloud session it came from.

## Prerequisites

- Python 3.11 (3.10+ works)
- git, git-lfs, curl
- ~4GB free disk (data files) or ~1GB for code-only checkout
- An Upstox account with an API access token (only needed if you want to
  refresh data with today's session). https://upstox.com/developer/apps
  → create app → grab access token. Set `UPSTOX_ACCESS_TOKEN` env var.

## One-shot setup (copy-paste)

```bash
# 1. Clone the repo (code + small data files, ~150MB)
git clone https://github.com/yuvaprabhu/chakra.git
cd chakra

# 2. Download the two big parquet files from the initial release
mkdir -p data/screen data/backtest_panels
curl -L -o data/screen/features.parquet \
  https://github.com/yuvaprabhu/chakra/releases/download/v0-snapshot/features.parquet
curl -L -o data/backtest_panels/bottom_panel.parquet \
  https://github.com/yuvaprabhu/chakra/releases/download/v0-snapshot/bottom_panel.parquet

# 3. Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Sanity check
python3 -c "import pandas as pd; f = pd.read_parquet('data/screen/features.parquet'); print(f'features loaded: {len(f):,} rows, cols={len(f.columns)}, last={f.date.max().date()}')"

# 5. View the dashboard immediately (uses the snapshot data)
python3 -m http.server 8000 --directory dashboard
# open http://localhost:8000
```

If the dashboard opens with the Portfolio audit visible and today's
screens listed, you're synced with the cloud state at snapshot time.

## Refresh data with today's session (optional)

```bash
export UPSTOX_ACCESS_TOKEN="…"
bash scripts/daily_update.sh
```
This: pulls today's bars → recomputes features → evaluates every rule →
rebuilds dashboard JSON. Takes ~5 minutes with warm cache.

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
