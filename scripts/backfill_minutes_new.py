"""Minute backfill for universe members that have no minute history yet."""
import sys, time
sys.path.insert(0, ".")
import pandas as pd
from screener.store import Store
from screener.upstox import UpstoxClient, ingest_minutes_month

store = Store("data")
lat = pd.read_parquet("data/screen/latest_features.parquet", columns=["isin"])
have = set(store.query("SELECT DISTINCT isin FROM minutes")["isin"])
need = [i for i in lat["isin"].unique() if i not in have]
print(f"{len(need)} names need minute history", flush=True)
if not need:
    raise SystemExit(0)

end = pd.Timestamp.today().normalize()
start = (end - pd.DateOffset(years=2)).replace(day=1)
client = UpstoxClient()
t0 = time.time(); total = 0
cur = start
while cur <= end:
    rows, stats = ingest_minutes_month(store, need, cur.year, cur.month, client=client, workers=10)
    total += rows
    errs = sum(1 for s in stats if s.error)
    print(f"  {cur.year}-{cur.month:02d}: {rows:>9,}  cum={total:>11,}  err={errs}  {time.time()-t0:5.0f}s", flush=True)
    cur = (cur + pd.offsets.MonthBegin(1)).normalize()
print(f"\ntotal {total:,} minute bars in {time.time()-t0:.0f}s", flush=True)
