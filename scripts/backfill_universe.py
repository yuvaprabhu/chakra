"""Daily backfill for the union of all loaded indices, above a market-cap floor."""
import argparse, logging, sys, time
sys.path.insert(0, ".")
import pandas as pd
from screener.store import Store
from screener.upstox import UpstoxClient, ingest_daily

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2000-01-01")
ap.add_argument("--min-mcap-cr", type=float, default=5000.0)
ap.add_argument("--workers", type=int, default=12)
a = ap.parse_args()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

store = Store("data")
mem = store.read_membership()
live = mem[mem["to_date"].isna()]
fund = store.read_fundamentals()
fund["mcap_cr"] = fund["vendor_mcap"] / 1e7

isins = sorted(set(live["isin"]))
keep = set(fund.loc[fund["mcap_cr"] >= a.min_mcap_cr, "isin"])
target = [i for i in isins if i in keep]
print(f"{len(isins)} names in indices -> {len(target)} at or above Rs {a.min_mcap_cr:,.0f} Cr", flush=True)

t0 = time.time()
def prog(i, n, isin):
    if i % 50 == 0 or i == n:
        print(f"  {i}/{n}  {time.time()-t0:.0f}s", flush=True)

stats = ingest_daily(store, target, a.start, pd.Timestamp.today().normalize(),
                     client=UpstoxClient(), workers=a.workers, on_progress=prog)
ok = [s for s in stats if not s.error]
bad = [s for s in stats if s.error]
print(f"\n{len(ok)} ok ({sum(s.rows for s in ok):,} rows), {len(bad)} failed, "
      f"{sum(s.dropped for s in ok):,} quarantined, in {time.time()-t0:.0f}s")
for s in bad[:10]:
    print("  FAIL", s.isin, s.error[:110])
print(store.summary())
