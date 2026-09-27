"""Backfill anything the universe is missing: full daily history, then recent minutes.

Idempotent. Run it after lowering the market-cap floor, after a long gap, or
any time `inspect_data.py` shows short histories. Names that already have
enough bars are skipped, so a second run costs seconds.

  python3 scripts/sync_universe.py                      # floor from screener/config.py
  MCAP_FLOOR_CR=1000 python3 scripts/sync_universe.py   # override the floor
  python3 scripts/sync_universe.py --skip-minutes       # daily bars only
"""
import argparse, sys, time
sys.path.insert(0, ".")
import pandas as pd
from screener.store import Store
from screener.config import MCAP_FLOOR_CR, MIN_HISTORY_BARS, HOURLY_BACKFILL_MONTHS, universe_isins
from screener.upstox import UpstoxClient, ingest_daily, ingest_minutes_month

ap = argparse.ArgumentParser()
ap.add_argument("--min-mcap-cr", type=float, default=MCAP_FLOOR_CR)
ap.add_argument("--start", default="2000-01-01")
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--months", type=int, default=HOURLY_BACKFILL_MONTHS)
ap.add_argument("--skip-minutes", action="store_true")
a = ap.parse_args()

store = Store("data")
end = pd.Timestamp.today().normalize()
isins = universe_isins(store, a.min_mcap_cr)
print(f"universe at Rs {a.min_mcap_cr:,.0f} Cr: {len(isins)} names", flush=True)

counts = store.query("SELECT isin, count(*) n FROM bars GROUP BY isin").set_index("isin")["n"]
need = [i for i in isins if counts.get(i, 0) < MIN_HISTORY_BARS]
print(f"daily: {len(need)} name(s) below {MIN_HISTORY_BARS} bars", flush=True)
t0 = time.time()
if need:
    def prog(i, n, isin):
        if i % 50 == 0 or i == n:
            print(f"  daily {i}/{n}  {time.time()-t0:.0f}s", flush=True)
    stats = ingest_daily(store, need, a.start, end, client=UpstoxClient(), workers=a.workers, on_progress=prog)
    ok = [s for s in stats if not s.error]; bad = [s for s in stats if s.error]
    print(f"  {len(ok)} ok ({sum(s.rows for s in ok):,} rows), {len(bad)} failed", flush=True)
    for s in bad[:10]:
        print("   FAIL", s.isin, s.error[:110])

if not a.skip_minutes:
    months = pd.date_range(end - pd.DateOffset(months=a.months - 1), end, freq="MS")
    since = (end - pd.DateOffset(months=1)).normalize()
    seen = set(store.query(f"SELECT DISTINCT isin FROM minutes WHERE ts >= '{since.date()}'")["isin"])
    gap = [i for i in isins if i not in seen]
    print(f"minutes: {len(gap)} name(s) with no recent bars, {len(months)} month(s) each", flush=True)
    if gap:
        client = UpstoxClient()
        for m in months:
            rows, ms = ingest_minutes_month(store, gap, m.year, m.month, client=client, workers=a.workers)
            print(f"  {m.year}-{m.month:02d}: {rows:,} rows, {sum(1 for s in ms if s.error)} errors, "
                  f"{time.time()-t0:.0f}s", flush=True)

print(f"\ndone in {time.time()-t0:.0f}s. store last_date: {store.summary()['last_date']}", flush=True)
