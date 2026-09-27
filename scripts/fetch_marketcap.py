"""Shares outstanding per ISIN from Yahoo, cached to parquet.

Market cap is then computed as shares x OUR OWN close, rather than taken from
Yahoo directly: shares outstanding move only on corporate actions, while the
price moves daily, so this keeps the cap consistent with the prices every other
number on the screen is built from.
"""
import sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, ".")
import pandas as pd
import yfinance as yf

uni = pd.read_csv("data/raw/union_universe.csv", dtype=str)
out, fails = [], []
t0 = time.time()

def one(row):
    tk = f"{row['symbol']}.NS"
    try:
        fi = yf.Ticker(tk).fast_info
        sh = getattr(fi, "shares", None)
        mc = getattr(fi, "market_cap", None)
        return {"isin": row["isin"], "symbol": row["symbol"],
                "shares": float(sh) if sh else None,
                "yf_mcap": float(mc) if mc else None}
    except Exception as exc:
        return {"isin": row["isin"], "symbol": row["symbol"], "shares": None,
                "yf_mcap": None, "err": str(exc)[:80]}

rows = uni.to_dict("records")
with ThreadPoolExecutor(max_workers=12) as pool:
    futs = {pool.submit(one, r): r for r in rows}
    for i, f in enumerate(as_completed(futs), 1):
        r = f.result()
        (fails if r["shares"] is None else out).append(r)
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}  {time.time()-t0:.0f}s  ok={len(out)} fail={len(fails)}", flush=True)

df = pd.DataFrame(out + fails)
df.to_parquet("data/raw/shares_outstanding.parquet", index=False)
print(f"\n{len(out)} with shares, {len(fails)} without, in {time.time()-t0:.0f}s")
if fails:
    print("sample failures:", [f["symbol"] for f in fails[:12]])
