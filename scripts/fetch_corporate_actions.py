"""Corporate actions (splits and bonuses) per ISIN, from Yahoo.

This is what lets raw prices be DERIVED from the adjusted series instead of
scraped: a back-adjusted close multiplied by the cumulative factor of every
action after that date returns the price the exchange actually printed.
"""
import random, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, ".")
import pandas as pd
import yfinance as yf
from screener.store import Store

store = Store("data")
# Every name in the universe, which is now gated on market cap rather than
# index membership - a missing action shows up as a derived price that
# disagrees with the exchange file, so this has to cover all of them.
fund = store.read_fundamentals()
fund["mcap_cr"] = fund["vendor_mcap"] / 1e7
rows = (fund.loc[fund["mcap_cr"] >= 5000, ["isin", "symbol"]]
        .dropna().drop_duplicates("isin").to_dict("records"))
print(f"fetching corporate actions for {len(rows)} names", flush=True)

out, fails = [], []
t0 = time.time()

def one(r):
    try:
        time.sleep(0.2 + random.uniform(0, 0.15))   # Yahoo throttles above ~5/s
        sp = yf.Ticker(f"{r['symbol']}.NS").splits
        recs = []
        for ts, ratio in sp.items():
            if ratio and ratio > 0:
                recs.append({"isin": r["isin"], "symbol": r["symbol"],
                             "ex_date": pd.Timestamp(ts).tz_localize(None).normalize(),
                             "action": "split_or_bonus", "ratio": float(ratio),
                             "purpose": f"yahoo split factor {ratio}"})
        return recs
    except Exception as exc:
        fails.append((r["symbol"], str(exc)[:60]))
        return []

with ThreadPoolExecutor(max_workers=4) as pool:
    futs = [pool.submit(one, r) for r in rows]
    for i, f in enumerate(as_completed(futs), 1):
        out.extend(f.result())
        if i % 100 == 0:
            print(f"  {i}/{len(rows)}  {time.time()-t0:.0f}s  actions={len(out)}", flush=True)

df = pd.DataFrame(out)
# UPSERT, never replace. A partially throttled fetch that replaces the table
# silently deletes actions it failed to re-read - which is how ZFCVINDIA lost
# its 6:1 split and its derived prices drifted 83% from the exchange file.
# A full rebuild is still possible, but it has to be asked for explicitly and
# only when the fetch came back clean.
rebuild = "--rebuild" in sys.argv
if len(df):
    if rebuild and not fails:
        store.write_adjustments(df, replace=True)
        print("table rebuilt from scratch (clean fetch)")
    else:
        if rebuild and fails:
            print(f"refusing to rebuild: {len(fails)} fetch failures would delete good rows")
        store.write_adjustments(df, replace=False)
print(f"\n{len(df)} corporate actions across {df['isin'].nunique() if len(df) else 0} names, "
      f"{len(fails)} fetch failures, in {time.time()-t0:.0f}s")
if len(df):
    print(df.sort_values("ex_date").tail(8).to_string(index=False))
