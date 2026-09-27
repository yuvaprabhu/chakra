# Chakra Terminal — Handover

**For:** the next AI agent (Cursor) or developer picking up this project.
**From:** a Claude Code session on 2026-09-27 that pulled Chakra out of a cloud environment onto a local Mac and prepared it for automation.
**Owner:** yuvaprabhu (yuva.prabhu@nuplay.ai), personal-repo `github.com/yuvaprabhu/chakra`.

Read this file first, then read `PLAN.md`, then `CLAUDE.md`.

---

## 1. What Chakra is (one paragraph)

An end-of-day technical screener for NSE (India) equities. Pulls ~10 years of adjusted daily bars from Upstox (free API, no auth), ~1,500 stocks in the ≥₹1,000 Cr market-cap universe, computes ~80 rolling indicators, evaluates 40+ curated rules from `config/rules.yaml`, and exports a static-JSON dashboard (`dashboard/index.html`) that shows charts, screens, a strategy audit, and a paper-trade console. Core screening + backtest audit logic is production-quality; `CLAUDE.md` has 15 lessons from real backtesting mistakes that must be respected when editing anything in the audit path.

## 2. Environment (as this handover ships)

- **Host:** macOS (Darwin 24.5.0), zsh, Python 3.11.9 via pyenv.
- **Project root:** `/Users/yuva/personal/chakra` (moved from `~/Downloads/chakra` at end of session; the launchd plist in `PLAN.md` Phase 3 is already updated to this path). If you move it again, update `scripts/install_launchd.sh` and re-run.
- **Virtualenv:** `/Users/yuva/personal/chakra/.venv` (already set up, `pip install -r requirements.txt` complete + `yfinance` installed manually)
- **GitHub remote:** `https://github.com/yuvaprabhu/chakra` (default branch: `main`)
- **Dashboard server:** `python3 -m http.server 4747 --directory dashboard` running in background at time of handover. Port 4747 is the standard for this project.
- **Data range:** 2017-01-02 → 2026-09-25, 2.57M daily bars, 1,517 names (≥₹1,000 Cr as of 2026-09-27).

## 3. What was done in this session

### Bootstrap (getting the cloud snapshot to run locally)

1. Reassembled `chakra.bundle` from `chakra.bundle.part-{00..03}` in `~/Downloads/` (each 27 MB), then `git clone chakra.bundle chakra`.
2. Pushed local `master` → GitHub `main` at `github.com/yuvaprabhu/chakra`. Required a classic PAT (fine-grained PAT was scope-limited and rejected).
3. Extracted `nse-screener-code-v2.tar.gz` overwrite of the repo contents (fixed docs + excluded parquets). Committed and pushed as commit `74fe346`.
4. Full cold-start pipeline (`SETUP.md` "path B") — hit and fixed 5 bugs (see §4). Total wall-clock: ~35 min on a MacBook Pro M-series, not the 3–5 h the doc estimated.
5. Dashboard renders at http://localhost:4747. Star rating column was rewritten to a manual whitelist of the top-3 risk-adjusted screens.

### Bugs found and fixed during cold-start (all committed)

Documented in `SETUP.md` under **"Known cold-start bugs and fixes"**:

1. **`yfinance` missing from `requirements.txt`** — `scripts/scan_all_mcap.py` needed it. Installed via `pip install yfinance`; needs to be added to the requirements file (still a TODO).
2. **`scripts/sync_universe.py:31`** — crashed on empty `data/bars/` with `_duckdb.IOException`. Wrapped the query in try/except.
3. **`scripts/bottom_lab.py` multiprocessing** — macOS spawn method caused module re-import recursion because the script kicks off a `Pool()` at module scope. Fixed with `mp.set_start_method("fork", force=True)`.
4. **Hardcoded cloud paths (all fixed at ~ the line numbers below; grep for the string, not the line number, since line numbers drift):**
   - `scripts/bottom_lab.py` — `--panel` default was `/tmp/claude-0/.../scratchpad/bl/bottom_panel.parquet`. Now `data/backtest_panels/bottom_panel.parquet`. (Search: `BOTTOM_LAB_PANEL`.)
   - `scripts/strategy_audit.py` — `ROOT` was `Path("/home/user/nse-screener")`. Now `Path(__file__).resolve().parent.parent`. (Search: `ROOT = Path`.)
   - Three other scripts still have hardcoded scratchpad `PANEL` constants (not on the daily path, deferred): `scripts/harden_playbook.py`, `scripts/volume_lab.py`, `scripts/ma_pivot_retest_study.py`. Grep: `/tmp/claude-0`.
5. **`data/raw/index_catalog.csv` missing** — `scripts/export_dashboard.py` requires it. Created empty stub with headers only. All sector/broad labels fall back to "sector". Not a functional problem; if you want real labels, download NSE `ind_*list.csv` files from niftyindices.com into `data/raw/` and run `python3 scripts/load_indices.py`.
6. **`dashboard/lightweight-charts.js` missing** — v2 tarball excluded it. Chart pane rendered "Charting library did not load ... file is missing." Restored from git history (v4.2.3, 160 KB, Apache-2.0). If ever lost again: `curl -o dashboard/lightweight-charts.js https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js`.
7. **`dashboard/index.html` no `<meta charset="utf-8">`** — Python `http.server` sends bare `text/html` and the browser guessed wrong, rendering star characters (★) as `â˜…` mojibake. Added `<meta charset="utf-8">` right after `<title>`. Cmd-Shift-R to bypass browser cache after any change.
8. **Star rating logic rewritten** — was a 3-tier auto-computed rating (Elite/Solid/Positive) based on CAGR/DD/win%. Replaced with a manual whitelist of the top 3 risk-adjusted screens (see §5). Tooltip and legend updated to match.

### Data sources — actual endpoints (from the code, verified)

| Source | URL | Purpose | Auth |
|---|---|---|---|
| Upstox historical-candle | `https://api.upstox.com/v3/historical-candle` | Adjusted daily OHLCV (2017 → today), per ISIN | none |
| Upstox instruments | `https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz` | NSE symbol master (ISIN ↔ symbol ↔ instrument key) | none |
| Upstox intraday | `https://api.upstox.com/v3/historical-candle/intraday` | 1-min bars for the current session | none |
| NSE mainsite | `https://www.nseindia.com` | Bhavcopy for raw (unadjusted) OHLC used for pivots/CPR | session cookie (auto) |
| NSE archives | `https://nsearchives.nseindia.com` | Historical bhavcopy CSVs | session cookie (auto) |
| Yahoo Finance (via `yfinance`) | `finance.yahoo.com` | Market caps for the ≥₹1,000 Cr universe filter | none |
| niftyindices.com | manual browser download | `ind_*list.csv` index constituent files | none |

All free. No API keys anywhere in the pipeline.

### Data storage — absolute paths

Everything under `/Users/yuva/personal/chakra/data/`, total ~1.5 GB after cold start:

| Path | Size | Contents |
|---|---|---|
| `data/bars/year=YYYY/bars.parquet` | 42 MB | Daily OHLCV, hive-partitioned |
| `data/minutes/` | 249 MB | 1-min bars for current session |
| `data/backtest_panels/bottom_panel.parquet` | 499 MB | 8-year audit panel |
| `data/screen/features.parquet` | ~700 MB | Rolling indicators per (isin, date) |
| `data/screen/latest_features.parquet` | small | Most recent day slice |
| `data/screen/hits.parquet` | small | Rule matches by day |
| `data/screen/strategy_audit.json` | small | Per-screen audit numbers |
| `data/screen/market_state.parquet` | small | Daily market gate features |
| `data/screen/gate_r3.parquet` | small | Market gate state |
| `data/screen/indices.json` | small | Index performance for dashboard |
| `data/screen/{backtest,bracket,reversal,bottom_lab}.json` | small | Study outputs |
| `data/meta/*.parquet` | 100 KB | fundamentals, adjustments, ingest_log, quarantine |
| `data/raw/upstox_nse_instruments.csv` | small | Symbol master |
| `data/raw/all_nse_mcap.parquet` | small | Yahoo mcap scan |
| `data/raw/index_catalog.csv` | 21 B stub | Empty (see §3 bug 5) |
| `data/universe/` | 168 KB | Index membership |
| `dashboard/*.json` | ~3.5 MB | Static shards read by the browser |

## 4. Current state — what runs, what doesn't

### Runs today (as of handover)

- `bash scripts/daily_update.sh` — one-shot data + features + screens + dashboard export.
- `python3 -m http.server 4747 --directory dashboard` — the dashboard, at http://localhost:4747. Renders: Market, Screens, Chart, Money Flow, Watchlist, Backtest (with Portfolio tab), Console.
- Full cold start (path B in `SETUP.md`) works end-to-end after the bugs in §3 were fixed.

### Not built yet (the next phases — see `PLAN.md`)

- DB connection (need Astra API endpoint from user — see §7).
- `screener/trader.py` — the paper-trader state machine (entries / pending / exits).
- `chakra sync` shell alias + launchd job.
- Console tab rewrite to read from DB.
- Chart buy/sell markers from DB.
- Dockerize.

## 5. Key decisions locked in this session

Everything below is decided. Do not re-litigate unless the user asks to change it.

### Whitelisted strategies (only these three are active)

| Screen | Family | Stop | Exit | Position sizing |
|---|---|---|---|---|
| **SW - Leader Dip Concentrated** | MR (mean reversion) | fill − 3×ATR(14), close-based | close > highest close of last 7 sessions **or** 20th session close | ₹1,00,000 per trade |
| **SW - Monthly R1 Retest** | MR | fill − 3×ATR(14), close-based | close > highest close of last 7 sessions **or** 60th session close | ₹1,00,000 per trade |
| **SW - Trend Stack** | TC (trend continuation) | fill − 3×ATR(14), close-based | close < EMA50 **or** 60th session close | ₹1,00,000 per trade |

Sizing formula: `qty = floor(100000 / entry_price)`. No lot-size rounding beyond that.

### Duplicate-signal tie-break

Same ISIN fires on two or more screens the same day → take exactly ONE trade, in this fixed priority order (highest audited win% first):
1. Leader Dip Concentrated (70.3% win)
2. Monthly R1 Retest (69.7% win)
3. Trend Stack (38.1% win)

### Slot cap

**None.** Every fresh signal on sync day gets a ₹1L trade regardless of how many other trades are open. This diverges from the audit (which assumed 10 slots per screen). Divergence is intentional — the goal of this run is to see what the no-cap version does in practice.

### Fresh-entry timing rules

- Entries execute ONLY on the day of sync. Fresh signals are never taken on past dates during a gap resync.
- A signal fires today (T) → we insert a `pending_entry` for T+1. The actual `trade` is created when we can read T+1's open price.
- Multi-day gap (missed N days, syncing on day N+1):
  - Open trades: replay every daily close from `gap_start` → `today` against stop & exit trigger. Close at first hit, `exit_date` = the actual day the trigger was hit.
  - Pending entries with `for_date` inside the gap: execute at that date's open.
  - Fresh signals from days inside the gap (except the current sync day): **discarded**.

### Paper trading only

No real broker. All "trades" are documents in Astra DB + markers on the dashboard chart. Nothing routes anywhere.

### Star rating (dashboard Portfolio tab)

Old logic (3-tier auto-computed Elite/Solid/Positive) is replaced. New logic: a hardcoded whitelist of exactly the three screens above. All others show `-` in the star column. Their audit numbers (Win%, CAGR, Max DD) still render — no star ≠ broken screen.

## 6. Where credentials live

### Astra DB

`~/personal/chakra/.env` (gitignored, local only). At time of handover it contains:

```
ASTRA_DB_APPLICATION_TOKEN=AstraCS:FecCIXdtZfRJuOtyeiwscldG:<...>
ASTRA_DB_CLIENT_ID=FecCIXdtZfRJuOtyeiwscldG
ASTRA_DB_CLIENT_SECRET=<redacted>
ASTRA_DB_API_ENDPOINT=            # ← STILL NEEDED (see §7)
```

Astra DB free tier, region `us-east-2` (AWS Ohio — only free region shown at signup). 80 GB storage, 25M reads/mo, 12.5M writes/mo, permanent. Docs: https://docs.datastax.com/en/astra-db-serverless/index.html

**Never** commit `.env` or paste the token into any tracked file. `.gitignore` already excludes it; do not weaken that.

### GitHub PATs used this session (need rotation)

Two tokens were pasted into this chat transcript by the user during the initial push. Both need to be revoked at https://github.com/settings/tokens:
- One fine-grained PAT (prefix `github_pat_11AE63...`) — was scope-limited, did not work.
- One classic PAT (prefix `ghp_1ppCZk...`) — was used to push commits `74fe346` and later.

Full token strings are NOT recorded here (would trip GitHub push protection and are useless anyway once rotated). If the user needs the exact values, they are in this session's chat history on the user's machine.

## 7. Blockers for the next session

**Only one blocker:** the Astra DB API endpoint URL is missing from `.env`. Ask the user for it:

> "Go to https://astra.datastax.com → click your `chakra` DB → Connect tab → copy the API Endpoint URL. It looks like `https://<db-id>-us-east-2.apps.astra.datastax.com`."

Paste it into `ASTRA_DB_API_ENDPOINT=` in `.env`. Then Phase 0 → Phase 1 → Phase 2 can start (see `PLAN.md`).

## 8. How to run this locally, from a clean checkout

```bash
git clone https://github.com/yuvaprabhu/chakra
cd chakra
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install yfinance   # missing from requirements.txt as of handover

# Cold start (~35 min end-to-end). Reads SETUP.md path B.
python3 -c "from screener.upstox import load_instruments; load_instruments('data/raw/upstox_nse_instruments.csv', download=True)"
python3 scripts/scan_all_mcap.py                             # ~26 min
python3 scripts/sync_fundamentals.py                         # ~1 s
python3 scripts/sync_universe.py --start 2017-01-01 --skip-minutes   # ~21 s (Upstox is generous)
python3 scripts/sync_today.py --days 30                      # ~2 min
python3 scripts/ingest_indices.py                            # ~1 min
python3 scripts/fetch_corporate_actions.py                   # ~2 min
python3 scripts/run_screen.py                                # ~1 min
python3 scripts/bottom_lab.py                                # ~4 min
python3 scripts/strategy_audit.py                            # ~13 s
printf "index_name,kind,size\n" > data/raw/index_catalog.csv # unblock export
python3 scripts/export_dashboard.py                          # ~5 s
python3 -m http.server 4747 --directory dashboard            # opens the app

# For subsequent days, one command does it all:
bash scripts/daily_update.sh
```

Once the automation (`chakra sync`, launchd) is built, that last line will be replaced.

## 9. Files to read first, in order

1. **`HANDOVER.md`** — this file. Current state.
2. **`PLAN.md`** — the detailed phase-by-phase plan for what to build next.
3. **`CLAUDE.md`** — project rules + goal + the 15 lessons. Especially lessons #1 (exit matches setup family), #13 (Max DD from daily equity), #14 (audit is truth over grooming), #15 (entry ≠ exit).
4. **`SETUP.md`** — cold-start instructions with the "Known cold-start bugs" section.
5. **`README.md`** — high-level overview + design notes.
6. **`config/rules.yaml`** — all 40+ screener definitions. The 3 whitelisted ones are `sw_leader_dip_concentrated`, `sw_monthly_r1_retest`, `trend_stack_swing`.
7. **`screener/store.py`** — parquet + DuckDB store abstraction. Read this before any data access change.
8. **`screener/rules.py`** — how rules from YAML get compiled to `df.eval` expressions.
9. **`scripts/strategy_audit.py`** — the audit engine. This is what determines "does a screen work" per the 15 lessons.
10. **`dashboard/index.html`** — single-file frontend, ~2500 lines. Section markers with comments make it navigable.

## 10. Constraints and conventions (read before changing anything)

- **Never edit files under `data/` by hand.** They are pipeline-generated. Modify the source (script or rule) instead.
- **Symbols get renamed and reused. Key everything on ISIN.** The store uses ISIN throughout.
- **Column-wise upserts, not full replaces.** Raw and adjusted OHLC arrive on independent schedules; a null in one column must not overwrite a stored value.
- **Bad rows are quarantined, not dropped.** `data/meta/quarantine.parquet` is the record; every reason is a string tag.
- **Never quote a model identifier** (Claude version, GPT version, etc.) in commits, PR bodies, or artifact metadata. Deployment-agnostic.
- **Never quote grooming-window numbers in `manage:` text.** Grooming numbers are stale; `strategy_audit.json` is the real-money source of truth. Lesson #14.
- **Do not weaken `.gitignore` to commit `.env` or a token.** See §6.

## 11. What the user cares about (product goals, in their words)

Direct quotes and paraphrases from the session:

1. "Automate the paper trading for a few months and see how the trades have gone through."
2. "One command — `chakra sync` — starts the server, syncs data, refreshes filters, sees if there is any next-day fresh buying or past-day exits."
3. "Trigger at 8 PM or 9 PM daily. Also whenever I open my laptop it should fire."
4. "You will not take trades on past dates, only on the current day."
5. "If sync missed 5 days, on day 6 you replay past bars against stop/target for OPEN trades. You do NOT take fresh trades in the past."
6. "One lakh worth of money in each trade."
7. "Free NoSQL database with huge storage." → Astra DB picked (80 GB free).
8. "Console page: remove capital constraint. Show each trade taken, no of days held, SL %/amount, target %/amount, quantity, and reasoning field."
9. "Green ▲ buy marker, red ▼ sell marker on the chart at the exact bar."
10. "Write tests for both DB/API integration and frontend interaction. Loop until fix — don't hand off half-done."
11. "Dockerize once v1 works locally."

## 12. Test methodology (from user)

- Every test asserts an **observable outcome**: a doc exists in Astra, a row exists in the DOM, a marker is drawn on canvas. Not "did it throw".
- Frontend E2E via `pytest-playwright` (headless Chromium).
- DB integration tests use a separate keyspace `chakra_test`, wiped in a `conftest.py` fixture before every run.
- **Loop-until-fix contract:** if a test fails, diagnose root cause and fix source, don't blind-retry. If the same fix fails twice, stop and ask.
- No phase is complete until its tests are green **and** the assertions have been documented in a way a human can sanity-check.

## 13. Session port assignments and background processes

At handover the following are running or expected to run:
- **Port 4747** — dashboard http.server (`python3 -m http.server 4747 --directory dashboard`). Chosen to avoid clashes with common services on 3000/3001/4200/5000/8080/8000.
- **No other ports bound** by this project.
- Future: `chakra sync` background server also uses 4747. If a different service already binds it, the script must fail loudly (do not silently pick a different port — the user's browser bookmark and launchd guard file assume 4747).

## 14. Frequently missed details

- **Star column rewrite (`dashboard/index.html`):** the `stars` function (grep `const stars=r=>`) no longer computes tiers from audit numbers. It's now a hardcoded `Set` called `STARRED` (grep `const STARRED`). If audit numbers change, the star column does not automatically re-rate — it's manual. This is intentional.
- **Charset meta tag:** required for star and ₹ symbol rendering under Python's `http.server`. Do not remove.
- **Multi-day gap logic:** the `manage_open_trades` function replays bar-by-bar; the stop is close-based (wicks are ignored) per lesson #7. If a stop hits on day 3 of a 5-day gap, the trade closes with `exit_date = day 3`, not `exit_date = day 5`.
- **`mkt_gate_on` in signal expressions:** two of the three whitelisted screens require the market gate. If the gate is off (bearish regime), those screens fire zero signals. This is by design; don't try to "override" the gate to force trades.
- **`fresh_trend_stack` for Trend Stack:** this is a *fresh event* flag — it fires only on the first day the trend stack condition becomes true, not every day it remains true. Same pattern for the other screens' first-day triggers. See `screener/indicators.py` for the fresh-event derivation.
- **Position sizing rounding:** `qty = floor(100000 / entry_price)`. For a ₹5,000 stock that's 20 shares = ₹100,000 exactly. For a ₹7,000 stock that's 14 shares = ₹98,000 (not rounded up to ₹100,000). Under-allocation is preferred over over-allocation. If `entry_price > 100000` → `qty = 0` → pending is dropped with `reason: "below_min_qty"`.
- **Universe floor is locked at `MCAP_FLOOR_CR=1000`.** Do NOT change this without user sign-off — every audit and every strategy's expected win% assumes the ≥₹1,000 Cr universe. Environment variable exists to override for one-off tests only.
- **`data/screen/trades_tagged.parquet` (34,857 historical trades for loss forensics)** is NOT migrated to Astra. Remains local, read-only, for research. It predates the paper-trader and is a different data model (backtest labels, not live paper trades).
- **Dashboard "Portfolio" is a sub-tab under Backtest,** not a top-level view. `dashboard/index.html` has main tabs Market/Screens/Chart/Money Flow/Watchlist/Backtest/Console; the star-rated audit table appears inside Backtest → Portfolio.
- **`chakra_test` namespace wipe scope.** The `conftest.py` `astra_test_client` fixture is `scope="function"` — wipes and reseeds before EVERY test. This is slower but eliminates order-dependence flakiness. If tests get too slow (>2 min), consider `scope="session"` + per-test collection-name namespacing.

## 15. Escalate to the user if

- The Astra API endpoint is not in `.env` and cannot be inferred. **Do not guess.**
- Any file under `data/` needs to be deleted or overwritten by hand.
- A pipeline step fails with a network error and does not self-recover after one retry.
- You'd need to change the tie-break order, slot cap, or sizing rule — those are locked in §5.
- A new screen looks worth adding to the active whitelist.
- Any credential appears in a tracked file (README, docs, code).

Everything else is fair game to decide and document in a PR.
