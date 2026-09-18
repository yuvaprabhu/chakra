"""Ingest raw NSE UDiFF bhavcopy for a date range (fills the raw price block)."""
import argparse, logging, sys, time
sys.path.insert(0, ".")
import pandas as pd
from screener.store import Store
from screener.fetch import NSESession, backfill, missing_dates

ap = argparse.ArgumentParser()
ap.add_argument("--start", required=True)
ap.add_argument("--end", default=None)
ap.add_argument("--pause", type=float, default=2.5, help="seconds between requests; NSE throttles hard")
ap.add_argument("--root", default="data")
a = ap.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("urllib3").setLevel(logging.WARNING)
store = Store(a.root)
end = pd.Timestamp(a.end or pd.Timestamp.today().normalize())

todo = missing_dates(store, a.start, end)
print(f"{len(todo)} session(s) to fetch: {[str(d.date()) for d in todo[:8]]}{'...' if len(todo)>8 else ''}", flush=True)

t0 = time.time()
results = backfill(store, a.start, end, session=NSESession(max_retries=4), pause=a.pause,
                   on_result=lambda r: print(f"  {r}", flush=True))
by = {}
for r in results:
    by[r.status] = by.get(r.status, 0) + 1
print(f"\n{time.time()-t0:.0f}s: {by}", flush=True)
print(store.summary(), flush=True)
