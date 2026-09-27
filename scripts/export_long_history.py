"""Weekly bars 2018-2024 for every stock in the universe, sharded by ISIN.
Loaded on-demand by the dashboard when the user zooms to W/M timeframe.
Compact JSON (rounded to 2dp, integer volume), same shape as the existing series."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
import pandas as pd
from screener.store import Store

t0 = time.time()
lat = pd.read_parquet("data/screen/latest_features.parquet")
isins = lat["isin"].tolist()
print(f"universe: {len(isins)} stocks", flush=True)

st = Store("data")
bars = st.read_bars(isins=isins, start="2018-01-01", end="2024-06-01")
print(f"bars loaded: {len(bars):,} rows in {time.time()-t0:.0f}s", flush=True)

# resample to weekly
weekly = (bars.set_index("date").groupby("isin")
          .resample("W-FRI").agg(o=("adj_open", "first"), h=("adj_high", "max"),
                                  l=("adj_low", "min"), c=("adj_close", "last"),
                                  v=("volume", "sum"))
          .dropna(subset=["c"]).reset_index())
print(f"weekly bars: {len(weekly):,}, {weekly.date.min().date()}->{weekly.date.max().date()}", flush=True)

SHARDS = 15
def shard_of(isin):
    return sum(ord(c) for c in isin) % SHARDS

buckets = [{} for _ in range(SHARDS)]
for isin, g in weekly.groupby("isin"):
    buckets[shard_of(isin)][isin] = {
        "t": [d.strftime("%Y-%m-%d") for d in g.date],
        "o": [round(x, 2) for x in g.o],
        "h": [round(x, 2) for x in g.h],
        "l": [round(x, 2) for x in g.l],
        "c": [round(x, 2) for x in g.c],
        "v": [int(v) if pd.notna(v) else 0 for v in g.v],
    }

total_mb = 0
Path("dashboard").mkdir(exist_ok=True)
for sid, data in enumerate(buckets):
    fn = f"dashboard/long_{sid:02d}.json"
    with open(fn, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    mb = Path(fn).stat().st_size / 1024 / 1024
    total_mb += mb
    print(f"  {fn}: {mb:.2f} MB, {len(data)} stocks", flush=True)

print(f"\ntotal: {total_mb:.1f} MB across {SHARDS} shards, {time.time()-t0:.0f}s")
