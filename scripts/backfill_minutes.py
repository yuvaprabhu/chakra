"""Backfill 1-minute bars month by month from Upstox."""
import argparse, logging, sys, time
sys.path.insert(0, ".")
import pandas as pd
from screener.store import Store
from screener.universe import Universe
from screener.upstox import UpstoxClient, backfill_minutes

ap = argparse.ArgumentParser()
ap.add_argument("--start", default=None, help="default: 2 years back")
ap.add_argument("--end", default=None)
ap.add_argument("--index", default="NIFTY500")
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--root", default="data")
a = ap.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
store = Store(a.root)
end = pd.Timestamp(a.end or pd.Timestamp.today().normalize())
start = pd.Timestamp(a.start) if a.start else (end - pd.DateOffset(years=2)).replace(day=1)
isins = Universe(store).isins_on(a.index, end)
print(f"{len(isins)} symbols, {start.date()} .. {end.date()}", flush=True)

t0 = time.time()
total = 0
def on_month(key, rows, stats):
    global total
    total += rows
    errs = [s for s in stats if s.error]
    print(f"  {key[0]}-{key[1]:02d}: {rows:>9,} rows  ({len(errs)} errors)  "
          f"cum={total:>11,}  {time.time()-t0:6.0f}s", flush=True)
    for s in errs[:3]:
        print(f"      FAIL {s.isin}: {s.error[:110]}", flush=True)

backfill_minutes(store, isins, start, end, client=UpstoxClient(), workers=a.workers, on_month=on_month)
print(f"\ntotal {total:,} minute bars in {time.time()-t0:.0f}s", flush=True)
print(store.minute_summary().to_string(index=False), flush=True)
