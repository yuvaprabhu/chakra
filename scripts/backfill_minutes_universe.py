import argparse, logging, sys, time
sys.path.insert(0, ".")
import pandas as pd
from screener.store import Store
from screener.upstox import UpstoxClient, backfill_minutes

ap = argparse.ArgumentParser()
ap.add_argument("--years", type=float, default=2.0)
ap.add_argument("--min-mcap-cr", type=float, default=5000.0)
ap.add_argument("--workers", type=int, default=14)
a = ap.parse_args()
logging.basicConfig(level=logging.WARNING)

store = Store("data")
mem = store.read_membership(); live = mem[mem["to_date"].isna()]
fund = store.read_fundamentals(); fund["mcap_cr"] = fund["vendor_mcap"] / 1e7
keep = set(fund.loc[fund["mcap_cr"] >= a.min_mcap_cr, "isin"])
target = sorted(set(live["isin"]) & keep)
end = pd.Timestamp.today().normalize()
start = (end - pd.DateOffset(years=a.years)).replace(day=1)
print(f"{len(target)} symbols, {start.date()} .. {end.date()}", flush=True)

t0 = time.time(); total = 0
def on_month(key, rows, stats):
    global total; total += rows
    errs = sum(1 for s in stats if s.error)
    print(f"  {key[0]}-{key[1]:02d}: {rows:>10,}  cum={total:>12,}  err={errs}  {time.time()-t0:5.0f}s", flush=True)

backfill_minutes(store, target, start, end, client=UpstoxClient(),
                 workers=a.workers, skip_done=False, on_month=on_month)
print(f"\ntotal {total:,} minute bars in {time.time()-t0:.0f}s", flush=True)
