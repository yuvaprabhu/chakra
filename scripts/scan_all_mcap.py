"""Market cap for every NSE equity. Rate-limited, resumable.

The universe gate moves from index membership to market cap, so this has to
cover the whole exchange rather than the ~750 names that sit in an index.
Yahoo throttles aggressively at high concurrency, so this runs few workers with
backoff and checkpoints as it goes - a stall costs minutes, not the whole run.
"""
import random, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, ".")
import pandas as pd
import yfinance as yf
from screener.upstox import load_instruments

OUT = Path("data/raw/all_nse_mcap.parquet")
WORKERS, RETRIES = 3, 4
PACE = 0.25   # seconds between requests per worker; Yahoo throttles above ~5/s

inst = load_instruments("data/raw/upstox_nse_instruments.csv")
done: dict[str, dict] = {}
if OUT.exists():
    prev = pd.read_parquet(OUT)
    done = {r["isin"]: r for r in prev.to_dict("records") if r.get("mcap_cr") is not None}
    print(f"resuming: {len(done)} already priced", flush=True)

todo = [r for r in inst.to_dict("records") if r["isin"] not in done]
print(f"{len(inst)} NSE equities, {len(todo)} to fetch, {WORKERS} workers", flush=True)

def one(r):
    for attempt in range(RETRIES):
        try:
            time.sleep(PACE + random.uniform(0, 0.15))
            fi = yf.Ticker(f"{r['tradingsymbol']}.NS").fast_info
            mc = getattr(fi, "market_cap", None)
            sh = getattr(fi, "shares", None)
            if mc:
                return {"isin": r["isin"], "symbol": r["tradingsymbol"], "name": r["name"],
                        "mcap_cr": float(mc) / 1e7, "shares": float(sh) if sh else None}
            return None
        except Exception:
            time.sleep((2.5 ** attempt) + random.uniform(0, 1.2))
    return None

rows = list(done.values())
t0 = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futs = {pool.submit(one, r): r for r in todo}
    for i, f in enumerate(as_completed(futs), 1):
        r = f.result()
        if r:
            rows.append(r)
        if i % 250 == 0:
            pd.DataFrame(rows).to_parquet(OUT, index=False)
            rate = i / max(time.time() - t0, 1)
            print(f"  {i}/{len(todo)}  priced={len(rows)}  {rate:.1f}/s  "
                  f"eta {int((len(todo)-i)/max(rate,0.01)/60)}m", flush=True)

df = pd.DataFrame(rows).drop_duplicates("isin")
df.to_parquet(OUT, index=False)
big = df[df["mcap_cr"] >= 5000]
print(f"\n{len(df)} priced of {len(inst)} ({len(df)/len(inst)*100:.1f}%) in {time.time()-t0:.0f}s")
print(f"{len(big)} at or above Rs 5,000 Cr")
print(big["mcap_cr"].describe(percentiles=[.25, .5, .75]).round(0).to_string())
