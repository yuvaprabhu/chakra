"""Find NSE equities above the market-cap floor that are in NONE of our indices.

Index membership is the universe gate, and NSE only admits a new listing at a
periodic review, so a large recent IPO can be tradable for months before any
index carries it. This quantifies that blind spot.
"""
import sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, ".")
import pandas as pd
import yfinance as yf
from screener.store import Store
from screener.upstox import load_instruments

store = Store("data")
mem = store.read_membership()
in_index = set(mem[mem["to_date"].isna()]["isin"])
inst = load_instruments("data/raw/upstox_nse_instruments.csv")
off = inst[~inst["isin"].isin(in_index)].to_dict("records")
print(f"{len(inst)} NSE equities on Upstox; {len(in_index)} sit in an index; {len(off)} do not", flush=True)

out = []
t0 = time.time()

def one(r):
    try:
        fi = yf.Ticker(f"{r['tradingsymbol']}.NS").fast_info
        mc = getattr(fi, "market_cap", None)
        return {"isin": r["isin"], "symbol": r["tradingsymbol"], "name": r["name"],
                "mcap_cr": (float(mc) / 1e7) if mc else None}
    except Exception:
        return None

with ThreadPoolExecutor(max_workers=16) as pool:
    futs = [pool.submit(one, r) for r in off]
    for i, f in enumerate(as_completed(futs), 1):
        r = f.result()
        if r and r["mcap_cr"]:
            out.append(r)
        if i % 500 == 0:
            print(f"  {i}/{len(off)}  {time.time()-t0:.0f}s  priced={len(out)}", flush=True)

df = pd.DataFrame(out)
df.to_parquet("data/raw/offindex_mcap.parquet", index=False)
big = df[df["mcap_cr"] >= 5000].sort_values("mcap_cr", ascending=False)
print(f"\n{len(df)} priced; {len(big)} of them are >= Rs 5,000 Cr but in NO tracked index")
print(big.head(25)[["symbol", "name", "mcap_cr"]].to_string(index=False))
