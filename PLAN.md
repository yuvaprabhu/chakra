# Chakra Terminal — Development Plan

**Owner:** yuvaprabhu (yuva.prabhu@nuplay.ai)
**Status:** Phase 0 blocked on Astra DB API endpoint. All prior local-setup work complete.
**Last updated:** 2026-09-27

This document defines what to build next. Read [HANDOVER.md](./HANDOVER.md) first for the current state of the repo.

---

## 1. What Chakra is

An NSE end-of-day technical screener that shipped as a cloud-only tool and is being pulled down to run locally + automate paper trading.

Today, `bash scripts/daily_update.sh` fetches ~10 years of NSE data from Upstox (free, unauthenticated) + NSE bhavcopy, computes ~80 rolling indicators per symbol, evaluates 40+ curated rules from `config/rules.yaml`, and exports JSON shards that a static HTML dashboard renders (charts, screens, portfolio audit, watchlist).

The core screening + audit logic is production-quality (see `CLAUDE.md` — 15 lessons from real backtesting mistakes). The gap is that:
1. Everything is local files. No persistence across a machine wipe. No cross-device.
2. There is no automation — you must remember to run `daily_update.sh` daily.
3. The dashboard has a paper-trading module driven by browser localStorage — not the same thing as a real audit trail.

## 2. Goal

**Automate paper trading against the 3 whitelisted strategies for a few months. Log every trade so we can review how they actually performed in live (walk-forward) conditions, not just in the historical audit.**

Concrete outcomes:
- One command (`chakra sync`) fetches today's data, runs the screens, decides entries/exits, and updates the dashboard.
- macOS `launchd` runs it daily at 20:00 IST **and** on wake/login. Missing a day (or five) is safe.
- Every paper trade is a document in a cloud NoSQL database, persists across laptop wipes.
- The dashboard reads trades from the DB and draws buy/sell markers on the chart.

## 3. The 3 strategies (locked in)

Only these three are active. Every other rule in `config/rules.yaml` gets archived to the `strategies` collection with `is_active: false`.

| Screen | Family | Entry signal | Entry | Stop | Exit |
|---|---|---|---|---|---|
| **SW - Leader Dip Concentrated** | MR | `rs_rank≥80 AND price>SMA200 AND new_lo7==1 AND rsi_2<10 AND dist_ema_21≥2.6 AND mkt_gate_on==1` | T+1 open | fill − 3·ATR(14), close-based | next open after close > highest close of prev 7 sessions, **or** close of 20th session |
| **SW - Monthly R1 Retest** | MR | dist to monthly R1 in (0, 3%), 3-day low near R1, R1 cross 2-15 sessions ago, dist_sma_200≥33, atr_pct≥4.3, mkt_gate_on==1 | T+1 open | fill − 3·ATR(14), close-based | next open after close > highest close of prev 7 sessions, **or** close of 60th session |
| **SW - Trend Stack** | TC | `fresh_trend_stack==1 AND mkt_vix_pctile_1y≥63.7 AND mkt_nifty_range_20d≥6.855` | T+1 open | fill − 3·ATR(14), close-based | next open after close < EMA50, **or** close of 60th session |

Position sizing: **₹1,00,000 per trade**. `qty = floor(100000 / entry_price)`. Under-allocation is preferred over rounding up. `entry_price` and `exit_price` use `adj_open` from `data/bars/year=YYYY/bars.parquet` (adjusted, since stops/exits are ATR-based on the same basis; mixing raw and adjusted breaks corp-action-affected trades — see CLAUDE.md lesson #1 on single-basis discipline). Close-based stop/exit checks use `adj_close`.

**Cost model (paper P&L is NET, not gross):** subtract 0.30% round-trip on every closed trade (per CLAUDE.md #12: 0.30% floor for large caps, 0.50% for 1000–5000 Cr band; we use 0.30% flat as the universe is ≥₹1,000 Cr and mostly liquid). `pnl_pct = ((exit - entry)/entry - 0.003) * 100`. Store both `pnl_gross_pct` and `pnl_pct` (net) on the trade doc.

**Fresh-signal suppression for non-`fresh_*` screens.** Two of the three whitelisted screens (`sw_leader_dip_concentrated`, `sw_monthly_r1_retest`) evaluate a *state* — they will fire every day the state holds. The trader must NOT open a new ₹1L trade each such day. Rule:
- If a signal fires for `(isin, screen)` today, and the same `(isin, screen)` was in the previous day's hits, this is NOT a fresh signal — suppress.
- After a trade on `(isin, screen)` closes, suppress new signals for that pair for **K=5** trading sessions (cooldown).
- If a trade is currently OPEN on that `isin` in ANY screen, no new signal for the same `isin` fires regardless of cooldown.
- `trend_stack_swing` uses `fresh_trend_stack==1` which is already a fresh-event flag (first day the trend stack becomes true) — for that screen the "not in yesterday's hits" check is redundant but should still be applied (belt and braces).

Duplicate-signal tie-break (same ISIN fires on 2+ screens same day) — take exactly one trade, in this priority order (highest audited win% first):
1. Leader Dip Concentrated (70.3% win)
2. Monthly R1 Retest (69.7% win)
3. Trend Stack (38.1% win)

Slot cap: **none**. Every fresh signal on sync day gets a ₹1L trade. This diverges from the audit's 10-slot assumption — that divergence is intentional and is part of what this run measures.

Fresh entries execute **only on the day of sync**. Never backfilled onto past dates. A T+1 pending queued on day T does execute at T+1's open even if T+1 is inside an offline gap.

Multi-day gap handling (missed 5 days, syncing on day 6):
- Open trades: replay bars from day 1 → day 6 against stop AND exit trigger. Close at first hit, mark exit on the actual day. **If both a stop and an exit trigger fire on the same replayed close, stop wins** (extension of CLAUDE.md #12 — capital preservation takes precedence).
- Pending entries created before the gap: if their `for_date` is inside the gap, execute at that date's `adj_open`. If `adj_open` for `for_date` is missing (stock halted, suspended, delisted, or holiday), advance `for_date` to the next trading day and retry. If 4 consecutive trading sessions pass without a valid open, drop the pending with `status: "abandoned_no_open"`.
- Fresh signals from days 1–5: **discarded**. Only day-6 fresh signals get pending_entries for day-7.

**NSE trading calendar (holiday handling).** No external file. A "trading day" is defined as any date present in `data/bars/year=YYYY/bars.parquet` where at least 100 distinct ISINs have a row (excludes Saturdays, Sundays, NSE holidays, and thin muhurat sessions). Wrap this in `screener/calendar.py::trading_days_between(start, end)`. Every gap-replay loop, T+1 lookup, and cooldown counter uses this — never `pd.date_range` on calendar days.

**Timezone.** All dates in `pending_entries.for_date`, `trades.entry_date`, `trades.exit_date`, and `sync_runs.run_id` are Asia/Kolkata (IST) trading dates. `created_at` and `completed_at` are ISO-8601 with `+05:30` offset. Never store bare UTC — this is an India-market-only app.

## 4. Data plan — what moves to DB, what stays local

### Stays local (parquet, regeneratable)

| Path | Regenerate with |
|---|---|
| `data/bars/year=*/bars.parquet` | `sync_universe.py` |
| `data/minutes/` | `sync_today.py` |
| `data/backtest_panels/bottom_panel.parquet` | `bottom_lab.py` |
| `data/screen/features.parquet` | `run_screen.py` |
| `data/screen/latest_features.parquet` | `run_screen.py` |
| `data/screen/market_state.parquet` | `run_screen.py` |
| `data/screen/gate_r3.parquet` | `run_screen.py` |
| `data/meta/*.parquet` | pipeline |
| `data/raw/*` | pipeline / manual |
| `dashboard/*.json` | `export_dashboard.py` |
| `config/rules.yaml` | git (source of truth), also mirrored to DB |

### Moves to Astra DB

| Collection | Source | Growth | Purpose |
|---|---|---|---|
| `strategies` | `config/rules.yaml` | ~50 docs, static | Curated screen definitions. 3 active + 38 archived. |
| `strategy_audits` | `data/screen/strategy_audit.json` | ~50 × N runs | Historical backtest snapshot per audit run. |
| `strategy_lessons` | `CLAUDE.md` "15 lessons" | 15, static | Engineering wisdom, referenced when editing screens. |
| `trades` | *new* | ~200/yr expected | THE operational collection. Every paper trade. |
| `pending_entries` | *new* | ~30/day, ephemeral | T+1 flags. Cleared on execution or expiry. |
| `sync_runs` | *new* | 1/day | Per-`chakra sync` audit trail. |
| `screener_hits` | `data/screen/hits.parquet` (last run only, today) | ~1500/day | Historical daily rule matches, for "why was X on the list on day Y". |
| `watchlist` | *browser localStorage today* | ~50 docs | User-tracked names. Persists cross-device. |
| `market_snapshots` | subset of `data/screen/market_state.parquet` | 1/day | gate_on / breadth / VIX / nifty_range_20d — for joining trades against market regime. |

### DB choice

**Astra DB (DataStax)** — 80 GB free tier, permanent, no card. Region: `us-east-2` (only free region shown at signup). `~250 ms` round-trip from India — fine for nightly batch.

**Access mode:** Astra **Data API** (JSON, MongoDB-shaped), NOT CQL. Client: `astrapy>=1.5.0,<2.0` (pin — 0.x had a different API surface).

**Database and namespace layout.** ONE Astra database named `chakra`, with TWO keyspaces (Astra calls them "namespaces" in the Data API):
- `chakra` — production data.
- `chakra_test` — wiped before every test session (see Phase 0 conftest).

Both live in the same physical DB. Astra free tier supports multiple namespaces per DB (limit: 10). Do not create two separate databases — that wastes the free-tier database slot count and complicates the token model.

Endpoint URL is copied from **Astra dashboard → your `chakra` DB → Connect → "API Endpoint"**. Pattern: `https://<uuid>-us-east-2.apps.astra.datastax.com`. Store in `.env` as `ASTRA_DB_API_ENDPOINT`. The token in `ASTRA_DB_APPLICATION_TOKEN` has "Database Administrator" role and works for both namespaces.

### Schema example — `strategies`

```json
{
  "_id": "sw_leader_dip_concentrated",
  "title": "SW - Leader Dip Concentrated",
  "family": "MR",
  "is_active": true,
  "star": true,
  "tie_break_rank": 1,
  "expr": "rs_rank >= 80 and adj_close > sma_200 and new_lo7 == 1 and rsi_2 < 10 and dist_ema_21 >= 2.6 and mkt_gate_on == 1",
  "entry_rule": "next_open",
  "stop_rule": {"kind": "atr", "atr_mult": 3, "basis": "close"},
  "exit_rules": [
    {"kind": "trail_high", "lookback": 7, "basis": "close"},
    {"kind": "time", "n_sessions": 20}
  ],
  "why": "Across 3,050 Playbook signals from 2018-2026 the shipped version wins 69 percent...",
  "manage_text": "Buy next open. Stop at fill minus 3 ATR(14)...",
  "audit_ref": "audit_2026_09_27"
}
```

### Schema example — `trades`

```json
{
  "_id": "trade_20260925_INE001A01036_sw_leader_dip_concentrated",
  "isin": "INE001A01036",
  "symbol": "HDFCBANK",
  "screen": "sw_leader_dip_concentrated",
  "entry_date": "2026-09-25",
  "entry_price": 1523.40,
  "qty": 65,
  "capital_deployed": 99021.00,
  "stop_price": 1425.60,
  "stop_pct": -6.42,
  "exit_rule_active": "trail_7d_high_close",
  "time_stop_session": 20,
  "status": "open",
  "days_held": 3,
  "last_close": 1547.20,
  "last_close_date": "2026-09-27",
  "cost_pct_rt": 0.30,
  "pnl_gross_pct": 1.56,
  "pnl_pct": 1.26,
  "pnl_rupees": 1247.00,
  "sync_run_id": "run_20260925_ist",
  "reasoning": {
    "clauses_fired": {"rs_rank": 84, "rsi_2": 4.2, "dist_ema_21": 3.1, "new_lo7": 1, "adj_close_gt_sma_200": true, "mkt_gate_on": true},
    "market_context": {"gate_on": true, "vix_pctile": 68, "nifty_range_20d": 7.1},
    "manage_snapshot": "Buy next open. Stop at fill minus 3 ATR(14), close-based. Exit next open after close > highest close of previous 7 sessions, or close of 20th session."
  }
}
```

- `_id` includes `entry_date` — the same `(isin, screen)` re-triggering months later (after cooldown) creates a new doc with a different `entry_date` prefix, so no collision.
- `sync_run_id` uses the scheduled trigger date (IST), not wall-clock time. A `RunAtLoad` fire at 22:15 IST on 2026-09-25 has `sync_run_id = "run_20260925_ist"`, not `"run_20260925_2215_ist"`. Second run same day is deduped by the `.chakra/last_sync_date` guard.
- `capital_deployed = qty * entry_price` (informational, not a source of truth for sizing).
- `time_stop_session`: 20 for Leader Dip, 60 for R1 Retest / Trend Stack.
- `pnl_pct` on OPEN trades is `mark-to-market minus cost/2` (assume half the round-trip is realized at entry). On CLOSED trades it's full realized net.
- `exit_rule_active` values: `"trail_7d_high_close"` (Leader Dip, R1 Retest), `"close_below_ema50"` (Trend Stack). On close, `exit_reason` is one of: `stop_hit`, `trail_exit`, `ema50_exit`, `time_stop`.

### Schema example — `pending_entries`

```json
{
  "_id": "pending_20260926_INE001A01036",
  "isin": "INE001A01036",
  "symbol": "HDFCBANK",
  "screen": "sw_leader_dip_concentrated",
  "for_date": "2026-09-26",
  "created_at": "2026-09-25T20:00:00+05:30",
  "reasoning": { "...same as trade.reasoning..." }
}
```

## 5. Phased delivery

Each phase is independently shippable. If we stop after phase N, the app still works — just less complete. Every phase has tests before it's marked done. See §6 for the loop-until-fix contract.

### Phase 0 — Prep

**Blocked on user:** need `ASTRA_DB_API_ENDPOINT` in `.env`.

- [ ] Create Astra DB namespaces via Data API: `chakra` (prod) + `chakra_test` (test). Both inside the single `chakra` database.
- [ ] `.env` has `ASTRA_DB_APPLICATION_TOKEN` and `ASTRA_DB_API_ENDPOINT`. Verify with `python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); assert os.getenv('ASTRA_DB_API_ENDPOINT')"`.
- [ ] `requirements.txt` gains: `astrapy>=1.5.0,<2.0`, `python-dotenv>=1.0`, `pytest-playwright>=0.5`, `yfinance>=0.2` (missing today; installed by hand).
- [ ] `playwright install chromium` (downloads ~130 MB browser bundle; skip if E2E tests will run only in CI). Do NOT commit the browser cache.
- [ ] `scripts/db_ping.py` connects to `chakra_test`, writes one doc, reads back, asserts equal, deletes. Exit code 0 on success.
- [ ] `.chakra/` directory created at repo root for local runtime state (PID files, guard files). Added to `.gitignore`.

### Phase 1 — DB foundation + seed loader

**Deliverables**
- `screener/db.py` — one wrapper around `astrapy`. All DB calls go through here.
- `scripts/seed_db.py` — reads `config/rules.yaml` + `data/screen/strategy_audit.json` + parses the "15 lessons" section of `CLAUDE.md` → upserts `strategies`, `strategy_audits`, `strategy_lessons`. Idempotent (re-run leaves DB unchanged).

**Idempotency contract (important — different for each collection):**
- `strategies`: **upsert by `_id`**. Re-running with the same `rules.yaml` leaves the collection unchanged. Editing a rule's `expr` or `manage` and re-running updates the existing doc. Count stays at ~41.
- `strategy_lessons`: **upsert by `_id`** (`lesson_01` … `lesson_15`). Count always 15.
- `strategy_audits`: **INSERT ONLY**. Each run's `_id` is `audit_<YYYYMMDD>` (the date the audit ran). Re-running the audit script on the same day overwrites *that* day's doc. Running on a new day appends a new doc. Collection accumulates over time — this is by design. Do NOT test that count stays fixed across days; DO test that count stays fixed across same-day re-runs.

**Tests (`tests/test_seed.py`)**
- After first seed run, `strategies` has exactly 3 docs with `is_active=true` and expected `_id`s: `sw_leader_dip_concentrated`, `sw_monthly_r1_retest`, `trend_stack_swing` (note: NOT `sw_trend_stack` — the rule ID in `rules.yaml` is `trend_stack_swing`, while its title is "SW - Trend Stack").
- Each active strategy has `stop_rule.atr_mult == 3` and `stop_rule.basis == "close"`.
- Each active strategy has a non-empty `reasoning_manage_text` field (copy of the original `manage:` from YAML — preserved verbatim for the dashboard tooltip).
- `strategy_lessons` has 15 docs.
- Re-running seed twice **on the same day** produces identical DB state (no duplicates in any collection).
- Re-running the audit loader with `audit_<today>` overwrites today's doc but leaves prior-day audit docs untouched.

### Phase 2 — Trader state machine

**Deliverables**
- `screener/calendar.py` — `trading_days_between(start, end)` reading from `data/bars/`, `next_trading_day(d)`, `is_trading_day(d)`. Used by every date arithmetic below.
- `screener/trader.py`:
  - `read_fresh_signals(as_of_date)` — reads `data/screen/hits.parquet`, filters to the 3 active screens, applies the fresh-signal suppression rules (not in yesterday's hits for the ISIN×screen pair, cooldown of K=5 sessions post-close, no re-entry while trade open on ISIN in any screen). Returns a DataFrame of `(isin, symbol, screen, ...clauses)`. Reads yesterday's hits from `data/screen/hits.parquet` filtered to `date == as_of_date - 1 trading day`.
  - `evaluate_entries(as_of_date)` — takes `read_fresh_signals()` output. For each fresh signal, insert `pending_entry` for `for_date = next_trading_day(as_of_date)`. Apply duplicate-signal tie-break (Leader Dip > R1 Retest > Trend Stack).
  - `execute_pending(as_of_date)` — for each `pending_entry.for_date <= as_of_date`, resolve `adj_open` at `for_date` from `data/bars/`. If missing, advance `for_date` by one trading day, retry up to 4 times, then mark `status: "abandoned_no_open"` on the pending. On success: insert `trade` doc with `entry_price=adj_open`, `qty=floor(100000/entry_price)`, `stop_price=entry_price - 3*atr14_at_entry_date`, delete the pending.
  - `manage_open_trades(as_of_date, gap_start_date)` — for every trade with `status="open"`:
    - Load `adj_close` and derived `hi7_prior_close`, `ema50_close` for every trading day between `max(trade.entry_date+1, gap_start_date)` and `as_of_date`.
    - For each day in order: check `adj_close <= stop_price` → mark `exit_reason="stop_hit"`, `exit_date=this_day`, `exit_price=next_trading_day(this_day).adj_open`. Then check exit trigger. If both fire on same day, stop wins.
    - If the trade hits `entry_date + time_stop_session` trading days without an exit, force-close at `adj_open` of session `time_stop_session+1` with `exit_reason="time_stop"`.
    - Compute `pnl_gross_pct`, `pnl_pct` (net), `pnl_rupees`. Set `status="closed"`.
  - `write_sync_run(as_of_date, results)` — `sync_runs` insert with the counts (entries taken, pendings, exits, errors).
  - Fresh entries NEVER taken on past dates. `evaluate_entries` only fires for `as_of_date == today`. On a gap-recovery sync, `evaluate_entries` runs once (for today); `execute_pending` and `manage_open_trades` cover the historical replay.
- `scripts/trade.py` — CLI: `python3 scripts/trade.py --date 2026-09-25` (default: today, taken from IST clock). Also: `--dry-run` prints intended DB writes without executing them; useful for tests.

**Tests (`tests/test_trader.py`)** — using fixture parquets under `tests/fixtures/`:
- `test_fresh_signal_creates_pending` — one signal → one `pending_entry` for T+1.
- `test_state_signal_not_fresh_next_day_suppressed` — Leader Dip fires day T (fresh, creates pending) → same signal day T+1 → no new pending (was in yesterday's hits).
- `test_post_close_cooldown` — trade on `(isin, screen)` closes on day X. Signal fires again on day X+3 → suppressed. Signal on day X+6 → allowed.
- `test_open_trade_blocks_other_screen_same_isin` — open trade on ISIN under Leader Dip. Same ISIN fires on R1 Retest → suppressed.
- `test_duplicate_signal_tie_break_prefers_leader_dip` — same ISIN fires on Leader Dip + R1 same day → exactly 1 pending, screen=leader_dip.
- `test_pending_executes_at_next_open` — pending for date D → trade doc with `entry_price = adj_open at D`.
- `test_halted_stock_pending_retries` — pending for date D, `adj_open` missing on D and D+1 → executes at D+2's open. `entry_date == D+2`.
- `test_halted_stock_pending_abandoned_after_4_sessions` — no `adj_open` for 4 trading days → pending marked `abandoned_no_open`, no trade created.
- `test_gap_replay_closes_at_stop_hit_day` — seed open trade, close on day 3 of gap hits stop → `exit_date=day 3`, `exit_price = adj_open at day 4`, `exit_reason="stop_hit"`.
- `test_gap_replay_closes_at_trail_exit_day` — close on day 3 exceeds prev-7-close high → `exit_date=day 3`, `exit_reason="trail_exit"`.
- `test_gap_replay_stop_wins_over_trail_same_day` — day 3 satisfies both stop and trail → `exit_reason="stop_hit"`.
- `test_time_stop_fires` — Leader Dip trade with no earlier exit → closes at open of session 21 with `exit_reason="time_stop"`.
- `test_no_fresh_entries_on_past_dates` — signals fire on days T-3..T-1, sync on day T → 0 new pending_entries for those past days. Only T's signals become pendings for T+1.
- `test_position_size_1L_floors_correctly` — entry=₹500 → qty=200. Entry=₹7000 → qty=14 (capital_deployed=₹98,000, NOT rounded up).
- `test_position_size_one_share_min` — entry=₹120,000 (rare, high-priced stock) → qty=0 → trade skipped with `reason="below_min_qty"`.
- `test_pnl_includes_cost` — closed trade entry=100 exit=110 → `pnl_gross_pct=10.0`, `pnl_pct=9.7` (10 - 0.3 cost).
- `test_sync_run_written` — after a trade cycle, `sync_runs` has one doc with correct counts.

### Phase 3 — `chakra sync` + launchd (runs in parallel with Phase 4)

**Deliverables**
- `scripts/chakra_sync.sh` (bash, `set -euo pipefail`):
  1. `cd "$(dirname "$0")/.."` — anchor to repo root regardless of caller cwd.
  2. `source .venv/bin/activate` — launchd runs without shell env, so this is required. Fail fast if `.venv` missing.
  3. `export TZ=Asia/Kolkata` — pin timezone so all IST-date logic works regardless of Mac's system TZ (traveller-safe).
  4. Guard: if `.chakra/last_sync_date` == today's IST date → log "already synced today, skipping" and exit 0.
  5. `bash scripts/daily_update.sh` — bars → features → screens → export.
  6. `python3 scripts/trade.py` — trader state machine (Phase 2).
  7. Ensure dashboard server on :4747. Check with `lsof -ti:4747`; if empty, `nohup python3 -m http.server 4747 --directory dashboard > .chakra/server.log 2>&1 &` and write PID to `.chakra/server.pid`. **If :4747 is bound by a NON-chakra process (PID doesn't match `.chakra/server.pid`), fail loudly with a message — never silently pick a different port.** The user's launchd job, browser bookmark, and Playwright tests all assume 4747.
  8. Write today's IST date to `.chakra/last_sync_date`.
  9. Print summary to stdout: `chakra sync OK 2026-09-25 IST | entries=N pending_T+1=N exits=N errors=N | dashboard http://localhost:4747`.
- `scripts/install_alias.sh` — appends `alias chakra="$HOME/personal/chakra/scripts/chakra_sync.sh"` to `~/.zshrc`. Idempotent (grep before append). Also writes `chakra-logs` alias → `tail -f $HOME/personal/chakra/.chakra/server.log`.
- `~/Library/LaunchAgents/com.chakra.sync.plist`:
  - `Label`: `com.chakra.sync`
  - `ProgramArguments`: `["/bin/bash", "/Users/yuva/personal/chakra/scripts/chakra_sync.sh"]`
  - `StartCalendarInterval`: `{Hour: 20, Minute: 0}` — 20:00.
  - `EnvironmentVariables`: `{TZ: "Asia/Kolkata"}` — pins the "20:00" to IST regardless of system TZ.
  - `RunAtLoad: true` — fires on login (and thus after wake, on the first login).
  - `StandardOutPath`: `/Users/yuva/personal/chakra/.chakra/launchd.stdout.log`
  - `StandardErrorPath`: `/Users/yuva/personal/chakra/.chakra/launchd.stderr.log`
- `scripts/install_launchd.sh` — copies plist to `~/Library/LaunchAgents/`, runs `launchctl unload` (ignore error) then `launchctl load`. Also runs `plutil -lint` on the plist first and refuses to install if invalid.

**Path is load-bearing.** The plist and the shell alias both hardcode `/Users/yuva/personal/chakra`. If the repo is moved (or if a different user clones it), regenerate the plist and re-run `install_alias.sh`. Docker phase (Phase 5) also mounts this path.

**Tests (`tests/test_chakra_sync.py`)**
- `test_sync_idempotent` — run twice same day, second run exits fast without re-processing.
- `test_server_starts_when_not_running` — `curl http://localhost:4747/data.json` returns 200 with valid JSON.
- `test_launchd_plist_valid` — `plutil -lint com.chakra.sync.plist` exits 0.

### Phase 4 — Console rewrite + chart markers (runs in parallel with Phase 3)

**Deliverables**
- `scripts/export_dashboard.py` also writes `dashboard/trades.json` and `dashboard/watchlist.json` from Astra collections.
- `dashboard/index.html`:
  - Rewrite `#viewConsole` panel per spec: single table with Symbol, Strategy, Entry date, Entry ₹, Qty, Stop ₹/%, Exit trigger, Days held, Status, Exit date/₹, P&L ₹/%, Reasoning.
  - Delete `PT.capital`, settings modal capital field, `Trade` button on chart, localStorage paper-book.
  - Chart marker layer: for each trade where `trade.isin == currentIsin`, draw green ▲ at `entry_date` bar, red ▼ at `exit_date` bar. **The vendored library is `lightweight-charts` v4.2.3, so the API is `candleSeries.setMarkers([...])` — NOT `createSeriesMarkers` (that was renamed in v5). This will silently fail if you use the v5 API.** Marker fields: `{time, position: "belowBar"|"aboveBar", color: "#26A69A"|"#EF5350", shape: "arrowUp"|"arrowDown", text: "BUY"|"SELL"}`.
  - Read trade data from `dashboard/trades.json` (new artifact produced by `export_dashboard.py`, sourced from Astra `chakra.trades` namespace).
  - `dashboard/watchlist.json` (new artifact) similarly for the Watchlist tab.

**Tests (`tests/test_dashboard_e2e.py`)** — pytest-playwright, headless Chromium.
- `test_console_lists_trades` — seed 2 fake trades in Astra → run `export_dashboard.py` → open dashboard → click Console tab → both rows visible with correct Symbol, Qty, Stop%.
- `test_chart_markers_render` — seed a trade → open `/#/chart/<ISIN>` → assert a green marker element exists on the chart at the expected date.
- `test_watchlist_add_persists` — click add-to-watchlist → assert `watchlist` collection has the doc.
- `test_closed_trade_shows_pnl` — seed closed trade entry=100 exit=110 → row shows "+10.0%".

### Phase 5 — Docker

**Deliverables**
- `Dockerfile` — Python 3.11-slim + `requirements.txt` + code + `chakra_sync.sh` as entrypoint.
- `docker-compose.yml` — one service; volumes for `~/personal/chakra/data` (persistent) and `.env` (bind-mounted read-only).
- `docker compose run chakra pytest` runs the full suite inside the container. Green there = works on any machine.
- launchd job on host calls `docker compose run chakra` instead of the local venv.

### Parallelization

```
Phase 0  →  Phase 1  →  Phase 2  ┬→  Phase 3
                                 └→  Phase 4   (parallel)
                                     ↓
                                 Phase 5
```

## 6. Loop-until-fix contract (testing methodology)

For every test suite added:
1. Write the test with an **observable outcome** assertion (a doc exists in Astra, a row exists in the DOM, a marker is drawn on canvas). Never "did the function throw".
2. Run it.
3. If red: read the actual failure (stack trace, DOM snapshot, DB dump), find the root cause, fix source, re-run.
4. Repeat until green. If the same fix pattern fails twice, pause and ask.
5. Only mark a phase done when tests are green **and** the assertions have been documented so a human can sanity-check what the test actually proves.

### Test stack

| Layer | Tool |
|---|---|
| Backend / logic | `pytest` + `pytest-asyncio` |
| DB integration | `pytest` + `chakra_test` keyspace (wiped in `conftest.py` fixture) |
| Frontend E2E | `pytest-playwright` (headless Chromium) |

## 7. Non-goals (for now)

- Real broker integration (Zerodha Kite, etc.). Paper only.
- Real-time / intraday signals. End-of-day only.
- Multi-user auth. Single-user local tool.
- Mobile app.
- The 41 archived screens. Only the 3 whitelisted run for entries.
- Multi-cloud DB failover. One Astra DB, region `us-east-2`.

## 8. Total effort estimate

| Phase | Time | Depends on |
|---|---|---|
| 0 — Prep | 30 min | User pastes API endpoint |
| 1 — DB + seed loader | 2 hr | Phase 0 |
| 2 — Trader state machine | 4 hr | Phase 1 |
| 3 — `chakra sync` + launchd | 2 hr | Phase 2 |
| 4 — Console + markers | 3 hr | Phase 2 |
| 5 — Docker | 2 hr | Phase 3 |
| **Total** | **~14 hr** |

Phases 3 and 4 run concurrent.
