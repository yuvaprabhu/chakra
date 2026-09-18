"""Backfill adjusted daily bars for an index universe from Upstox."""
import argparse, logging, sys, time
sys.path.insert(0, ".")
import pandas as pd
from screener.store import Store
from screener.universe import Universe
from screener.upstox import UpstoxClient, ingest_daily

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2000-01-01")
ap.add_argument("--end", default=None)
ap.add_argument("--index", default="NIFTY500")
ap.add_argument("--workers", type=int, default=10)
ap.add_argument("--root", default="data")
a = ap.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
store = Store(a.root)
end = a.end or pd.Timestamp.today().normalize()
isins = Universe(store).isins_on(a.index, end)
print(f"{len(isins)} constituents of {a.index} as of {pd.Timestamp(end).date()}", flush=True)

t0 = time.time()
def progress(i, n, isin):
    if i % 25 == 0 or i == n:
        print(f"  {i}/{n}  {time.time()-t0:6.1f}s", flush=True)

stats = ingest_daily(store, isins, a.start, end, client=UpstoxClient(), workers=a.workers, on_progress=progress)
ok = [s for s in stats if not s.error]
bad = [s for s in stats if s.error]
print(f"\ndone in {time.time()-t0:.1f}s: {len(ok)} ok ({sum(s.rows for s in ok):,} rows), {len(bad)} failed")
for s in bad[:15]:
    print("  FAIL", s.isin, s.error[:140])
store.log_ingest(end, "upstox_daily", "ok" if not bad else "error", sum(s.rows for s in ok), f"{len(ok)}/{len(stats)} symbols")
print(store.summary())
