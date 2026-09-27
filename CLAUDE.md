# Chakra Terminal — project rules for Claude

## Current goal (2026-09-27)

Automate paper trading against the 3 whitelisted strategies for a few months
and log every trade so we can review how they actually performed in live
(walk-forward) conditions, not just in the historical audit.

- Signal source: SW - Trend Stack, SW - Monthly R1 Retest, SW - Leader Dip
  Concentrated. No others until we review the data.
- Sizing: ₹1,00,000 per trade, `qty = floor(100000 / entry_price)`.
- Execution: paper only. Nothing routes to a real broker. Trades exist as DB
  records and chart markers.
- Duplicate signals: if the same ISIN fires on two strategies same day,
  keep exactly one trade (not two). Tie-break by highest historical win% —
  fixed priority: Leader Dip Concentrated (70.3%) > Monthly R1 Retest (69.7%)
  > Trend Stack (38.1%).
- Slot cap: NONE. Every fresh signal on sync day gets a ₹1L trade,
  regardless of how many other trades are open. This diverges from the
  audit (which assumed 10 slots per screen) — the point of this run is to
  learn how a no-cap version behaves in practice.
- Entries fire ONLY on the day of sync (never backfilled). Pending T+1
  entries queued on day T do execute at T+1's open even if T+1 is inside a
  laptop-offline gap, and open trades replay stop/target across the gap.

NSE end-of-day technical screener. Parquet store via DuckDB, adjusted (Upstox)
basis, ISIN keys. The dashboard is a single-file HTML artifact; the daily
pipeline is `scripts/daily_update.sh`.

---

## Backtesting: mistakes I have made — do not repeat

Every one of these is a lesson from something that actually went wrong in this
project. Read this section before writing any backtest, filter test, or screen
proposal.

### 1. Match the EXIT to the SETUP FAMILY, not to a favourite exit

The single biggest error I have made: applying the Playbook's 7-day-high
reversal exit (a MEAN-REVERSION exit) to trend-continuation setups (VCP,
pocket pivot, OBV new high, CMF, volume-spike breakouts). Result: those
setups looked broken when in fact I had cut every winner short.

Rule: pick the exit from the setup's INTENT, not the tester's habit.
- **Mean-reversion / dip setups** (Leader Dip, RSI(2), Double Seven, Wyckoff
  spring, selling climax, dry-up at support): use the 7-day-high reversal
  exit (first close above the prior 7 closes → next open), cap 20 sessions.
  You are catching a bounce, not a trend.
- **Trend-continuation / breakout setups** (VCP, pocket pivot, volume spike,
  new-high breakouts, OBV, CMF, EMA/pivot retests in uptrend, trend-stack
  entries): use a trailing exit (close below the 21-EMA or 50-EMA → next
  open), cap 60 sessions. You are riding a move; a fast reversion exit will
  print false weakness.
- **Momentum leaders / fresh trend**: 4 ATR stop, 1:3 target, cap 60. Payoff
  matters more than accuracy; expect 35-45% win at +2%/trade.

Corollary: do NOT compare a mean-reversion CAGR/DD score to a
trend-continuation one and call the second "worse". They are different tools
with different structural win rates. A 74% win rate is only achievable for
mean-reversion setups because you cut on the first bounce. A 40% win rate at
+2%/trade with 60-day holds is the trend-continuation equivalent of the same
edge.

### 2. Test on the FULL 8-year panel (2018-2026), not just the recent window

`/tmp/claude-0/.../scratchpad/bl/bottom_panel.parquet` has 2.5M rows,
2017-01-02 → 2026-09-22. It contains 2018-19 midcap bear, 2020 COVID crash,
2022, 2025Q1 and 2026Q1. Use it as the PRIMARY window; use the recent 2024-06
onward window only for comparison.

The Playbook looked like +48% CAGR / -7% DD on the 2-year window. On 8 years
it is +20-30% CAGR / -25% DD. Both true. Report the 8-year first.

### 3. Regime filters (VIX percentile, Nifty range, breadth thrust) are 3-episode bets

If a filter that survives grooming is a market-regime switch that was ON in
only 3-4 windows of the whole sample (typically post-sell-off rebounds), the
effective evidence is 3 episodes, not 600 trades. Say so explicitly. Do NOT
report the clustered t-stat as if the 600 trades were independent.

Watch for: `mkt_vix_pctile_1y >=`, `mkt_nifty_range_20d >=`,
`mkt_breadth_thrust_10 == 1` — all of these tend to be regime picks.

### 4. Survivorship bias — bound it, then say so

The universe is today's ≥₹1,000 Cr names. Stocks that bottomed and never came
back, or delisted, are missing. This flatters every bottom-buying result, and
adds ~1-3% CAGR to every trend result.

Bound it: require `mcap_at_signal = today_shares * signal_close >= 1000 Cr`
and report how much CAGR falls when you do. Then state that the missing
delisted names cannot be bounded from this data.

### 5. "Fresh event" vs "state" for continuous rules

State rules (`weekly_tc_reclaim`, `close_above_monthly_tc`, momentum leaders,
trend stack) fire on EVERY day the state holds. They generate 20-30k trades
with ~0 per-trade edge. Convert to events with `first_after_quiet(mask, N)`:
the first day the state is true after N sessions of being false. That is what
`screener/market.py::add_fresh_flags` does and what `fresh_trend_stack` /
`fresh_momentum_leaders` are.

### 6. Signal filter vs execution rule — put each in its right place

- **Signal filter** (goes in the rule expression): trigger conditions, RS
  rank, distance from MAs, volume, price levels. Changes which rows appear
  on the screen.
- **Execution rule** (goes in the `manage:` text, not the expression): stop
  mechanic (intraday vs close-based), sizing, entry timing (open vs limit),
  slot allocation. Changes how you trade the signal, not what the signal is.

Do not put "close-based stop" in a rule expression. Do put "both triggers on
the same day" (7-day low AND RSI(2)<10) in a rule expression.

### 7. Close-based 3 ATR stop is a strict improvement over intraday

Verified on all 6 SW screens (Leader Dip Playbook, Double Seven, RSI(2)
Snapback, Monthly R1 Retest, Trend Stack, Momentum Leader) and on the 8-year
Playbook stream. Wicks below -3 ATR that close green are false shakeouts;
only close < -3 ATR should stop you (exit next open). This raises win rate
1-3 pts, cuts stopped% from ~13% to ~8%, and lifts CAGR ~7-10 pts at the
same drawdown. There is no known case where the intraday stop wins.

### 8. "Retest" screens must check that TODAY's close is still near the level

Bug I shipped and then fixed: `sw_monthly_r1_retest` required only that the
3-day low touched R1 and price was currently above it. This let signals show
stocks that had retested days ago and drifted 20% above R1 (KABRAEXTRU
+23.5% "retest"). Fix: also require `dist_m_r1 < 3.0` so the current close
is still within 3% above the level. Same principle applies to any retest,
reclaim or bounce setup — always cap how far the current close can be from
the reference level.

### 9. Portfolio circuit-breakers that sound smart but aren't

Tested and rejected on the Playbook stream:
- **"Pause on rolling win rate"**: 32/32 quarters flat. The Playbook's rolling
  win rate is always fine on average and terrible right when you need to be
  in.
- **"Halve slots in drawdown > 10%"**: cuts CAGR from +20% to +8% while
  saving only 3-4 pts of DD.
- **"Top 3 RS names per day"**: −3 pts CAGR, no DD improvement.

What actually works (tested on 8-yr Playbook):
- **Pause 10 sessions after 3 consecutive stops**: CAGR +20.6 → +22.3, DD
  −25 → −23, worst quarter −14 → −9. Free lunch.
- **Breadth-scaled slots (10 if breadth>60, 5 if 45-60, 0 if <45)**: worst
  quarter −14 → −9. Real DD relief.
- **Both triggers required (7-day low AND RSI2<10)**: fewer trades (24% of
  Playbook), 72% win vs 69%, mean +2.84% vs +1.60%, DD −14% vs −25%.

### 10. Entry timing: LIMIT beats CHASE, but not by "waiting for a shakeout"

Tested on 8-yr Playbook:
- **Next open** (shipped): +20.6% CAGR, -25% DD (baseline)
- **Limit -2% valid 3 sessions** (buy the retest): +24.9% CAGR, -22% DD.
  Better on both metrics. Fills 66% of signals; the missed 34% would have
  been the smaller bounces anyway.
- **Wait for a shakeout bar** (undercut and close green): +3% CAGR. TRAP.
  The bounces that actually happen don't need a shakeout; the shakeouts you
  see are already-failed setups.
- **Wait for a close above prior day's high** (confirmation candle): +0.6%
  CAGR. Same trap.

Rule: limit at a small discount is fine. "Wait for confirmation" is a
category of trading advice that costs money in the data.

### 11. Recurring code bugs

- `df.isin` is a pandas method, not a column accessor. Always use `df["isin"]`.
  Same for `.date`, `.name`, `.index`, `.columns` when those are column names.
- `pd.DataFrame.eval()` returns object dtype when a boolean condition uses
  numpy operators. Cast: `feat.eval(expr).fillna(False).astype(bool).to_numpy()`.
- `groupby(...).transform(lambda s: s.rolling(N).X())` is 10-100x slower than
  precomputed rolling. Precompute for hot loops.

### 12. Bar mechanics

- Entry is always the NEXT open after the signal close, never the signal
  close itself.
- Stop check FIRST each bar. Gap-through-stop fills at the open, not at the
  stop level.
- One open position per name (isin) at a time.
- Cost 0.30% round trip (rough NSE brokerage + STT + slippage floor).
- When a daily bar touches both stop and target, STOP WINS. Never assume you
  got the target first; that is the largest source of fake edge in bracket
  backtests.
- Close-based exits (trail, close-below-EMA) trigger on today's close and
  fill at NEXT open.

### 13. Max DD must be tracked on DAILY equity, not at trade events

The bug I shipped and got called out on: reporting max drawdown by
compounding trades in the order they close and taking the min of that
event-time equity curve. This under-counts DD massively because it never
sees the trough of a position that recovered by exit.

Concrete example: a trade opens, drops -25% mid-life, recovers to +2% at
exit. Event-time equity moves +0.2% (2%/10 slots) — DD looks like nothing.
Daily-tracking equity dips 2.5% at the -25% intra-trade trough. The daily
number is the one you'd actually sit through.

Rule: build a per-date equity vector, mark-to-market open positions each
session (or at minimum book realized P&L on the exit date and let paper
positions be revealed as they close in DATE order), then take
`(equity / cummax(equity) - 1).min()`. Report THAT as max DD. Event-time
sums are fine for CAGR and total return but never for DD.

Also: contaminated rows (unadjusted corporate-action windows,
`contaminated == 1`) MUST be excluded before running any backtest — even
if the rule expression doesn't reference the affected fields, the entry
and exit prices around a split will lie by the split factor. Drop them
first, then compute features (never after; recomputing on the reduced
row set silently changes what `.shift(N)` references).

Cost: 0.30% RT is the floor for liquid ≥₹5000 Cr names. For mid- and
small-caps in the 1000-5000 Cr band, real slippage adds 20-70 bp, so
audit runs should use 0.50% and note the split. Any strategy whose edge
disappears between 0.30% and 0.50% is not shippable.

### 14. Grooming numbers in rules.yaml are STALE; strategy_audit.json is truth

The `why:` and `manage:` blocks in rules.yaml were written during grooming
runs on the 2-3 year recent window with event-time equity tracking, and
they QUOTE numbers like "74 percent win, -8 percent drawdown, 30 percent a
year on ten slots". Those numbers are optimistic. The full-panel audit in
`scripts/strategy_audit.py` → `data/screen/strategy_audit.json` shows the
same screens on 8 years with daily equity tracking and contamination drop:

- SW - RSI(2) Snapback: shipped text says -8% DD → audit shows -32%
- SW - Double Seven: shipped is silent → audit shows -51%
- Leader Dip Playbook: shipped implies clean → audit shows -37%
- Momentum Leaders: shipped shows +40% CAGR → audit shows -3% CAGR
- EMA21 Pullback: never groomed → audit shows -17% CAGR / -59% DD

Rule: `strategy_audit.json` is the source of truth for real-money sizing
and expectations. When a `why:`/`manage:` block conflicts, trust the audit.
When shipping a new screen, update its docstring numbers from the audit
BEFORE marketing it as tradeable.

Corollary: a "recent-window grooming" number is a starting point, not a
verdict. Every SW/VL screen needs a full 8-year audit entry before it
appears on the Portfolio tab of the Backtest page as tradeable evidence.

### 15. Setup family and exit mechanic must not fight each other

`ema21_pullback` audit shows -16.5% CAGR / -59% DD. Reason: the rule buys
when price pulls INTO the 21-EMA (dist in [-4%, +1.5%]), then the trend-
continuation exit is close < 21-EMA. So every entry is a coin-flip on the
next close: half the time you exit the next bar for a small loss. The
setup is a mean-reversion pullback masquerading as trend-continuation.

Rule: if the entry sits ON the trigger level, the exit must be elsewhere.
Either widen the entry (only if price has BOUNCED off the 21-EMA and
closed a bar or two above it) OR change the exit (7-day-high reversal for
a bounce, or close < 50-EMA for a slower trend trail). "Buy the trigger,
trail the trigger" is a self-defeating loop.

Watch for: any rule whose entry level and exit level are the same MA or
the same pivot. `ma_pivot_retest`, any "retest the 50-DMA" rule, any
"tag the 21-EMA and buy" rule.

---

## Universe, data, files (facts you need often)

- Universe = today's NSE equities with own market cap ≥ ₹1,000 Cr
  (`MCAP_FLOOR_CR` env, default 1000). 1,474 names as of 2026-09-25.
- Daily pipeline: `scripts/daily_update.sh` runs fundamentals → gap backfill →
  today's bars → indices (VIX, Nifty) → screens → export → publish.
- Adjusted (Upstox) basis for indicators; a quality gate excludes 20-session
  contamination windows around unadjusted corporate actions.
- Feature panel: `data/screen/features.parquet` (production, last ~3 years).
  Extended panel for backtests: `/tmp/claude-0/.../scratchpad/bl/bottom_panel.parquet`
  (2017-01-02 → present, 2.5M rows).
- Market state: `data/screen/market_state.parquet` (74 features, from
  `scripts/build_regime_data.py`).
- Trades tagged: `data/screen/trades_tagged.parquet` (34,857 trades across
  5 strategies for loss forensics).
- Gate: `data/screen/gate_r3.parquet` (breadth with hysteresis, Zweig thrust
  window). Same construction as `screener/market.py::gate_series` — use ONE
  definition, not two.
- Rules: `config/rules.yaml`, one entry per screen. Tests:
  `tests/test_definitions.py` (canonical clauses each rule must contain) and
  `tests/test_screens.py`.

## Screen categories and prefixes

- **SW – <name>**: swing-lab-validated screens with hard stops. Everything
  new that we would trade goes here.
- No prefix: legacy raw screens (Leader Dip, Momentum Leaders, Minervini
  Trend Template, etc.) — kept as reference/watchlist views, not for direct
  trading.
- Every SW screen has a `stop_3atr` or `stop_4atr` column and states in its
  `manage:` text that the stop is CLOSE-BASED (wicks are ignored).

## Publish

- Dashboard is `dashboard/index.html`; support files sharded by size cap.
- 16 MB per file, 64 MB per publish, 255 files per artifact.
- Artifact URL: https://claude.ai/artifact/4XaB2txyq2RhRh1b3xe2Yu.
- Never introduce a hardcoded shard count; always shard by size (see
  `scripts/export_dashboard.py::shard`).

## Model attribution

Follow the session's attribution reminder for commits / PRs; do not include
any model identifier in the code, in commit messages beyond the trailer, or
in artifact content.
