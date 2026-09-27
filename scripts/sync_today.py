"""Bring the store up to today for the screener universe: adjusted daily bars
and 1-minute bars from Upstox for every name in the latest screen run.
Usage: python3 scripts/sync_today.py [--days 7]"""
import argparse, sys, time
sys.path.insert(0, ".")
import pandas as pd
from screener.store import Store
from screener.config import MCAP_FLOOR_CR, universe_isins
from screener.upstox import UpstoxClient, ingest_daily, ingest_minutes_month, ingest_today

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=int, default=7)
ap.add_argument("--min-mcap-cr", type=float, default=MCAP_FLOOR_CR)
a = ap.parse_args()
store = Store("data")
isins = universe_isins(store, a.min_mcap_cr)
print(f"universe: {len(isins)} names at or above Rs {a.min_mcap_cr:,.0f} Cr", flush=True)
end = pd.Timestamp.today().normalize(); start = end - pd.Timedelta(days=a.days)
client = UpstoxClient()
t0 = time.time()
stats = ingest_daily(store, isins, start, end, client=client, workers=12)
ok = [s for s in stats if not s.error]; bad = [s for s in stats if s.error]
print(f"daily: {len(ok)} ok ({sum(s.rows for s in ok):,} rows), {len(bad)} failed in {time.time()-t0:.0f}s", flush=True)
for s in bad[:8]: print("  FAIL", s.isin, s.error[:100])
months = sorted({(d.year, d.month) for d in pd.date_range(start, end)})
for y, m in months:
    rows, ms = ingest_minutes_month(store, isins, y, m, client=client, workers=12)
    print(f"minutes {y}-{m:02d}: {rows:,} rows, {sum(1 for s in ms if s.error)} errors, {time.time()-t0:.0f}s", flush=True)
# The historical endpoint stops at the previous session, so today's bar comes
# from the intraday one. Check coverage per name: a global "last date" is
# already today the moment ONE name has been fetched, which would skip the rest.
if end.weekday() < 5:
    done = set(store.query(
        f"SELECT DISTINCT isin FROM bars WHERE date = '{end.date()}' AND adj_close IS NOT NULL")["isin"])
    todo = [i for i in isins if i not in done]
    print(f"today: {len(done)} name(s) already have {end.date()}, fetching {len(todo)}", flush=True)
    if todo:
        r = ingest_today(store, todo, client=client)
        print(f"  intraday endpoint: {r}", flush=True)
print("store last_date:", store.summary()["last_date"])
