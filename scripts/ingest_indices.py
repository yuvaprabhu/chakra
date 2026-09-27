"""Daily history for the market indices and India VIX, from Upstox.

Screens are measured against an equal-weight index built from our own bars,
which is right for stock selection but cannot say "the Nifty was under its
50-day that week" or "implied volatility was elevated". A regime filter needs
the real published series, so this fetches them the same way the equities are
fetched, year by year plus today's bar from the intraday endpoint.
"""
import sys, time
from pathlib import Path
from urllib.parse import quote
sys.path.insert(0, ".")
import pandas as pd, requests

WANT = ["Nifty 50", "Nifty Bank", "Nifty Next 50", "Nifty 500", "NIFTY MIDCAP 150",
        "NIFTY MIDCAP 100", "Nifty Midcap 50", "Nifty Smallcap 500", "India VIX"]
BASE = "https://api.upstox.com/v3/historical-candle"
OUT = Path("data/raw/indices_daily.parquet")

sess = requests.Session(); sess.headers["User-Agent"] = "chakra/1.0"
rows = []
for name in WANT:
    key = quote(f"NSE_INDEX|{name}", safe="")
    got = []
    for y in range(pd.Timestamp.today().year, 1999, -1):
        r = sess.get(f"{BASE}/{key}/days/1/{y}-12-31/{y}-01-01", timeout=30)
        if r.status_code == 200:
            got += r.json().get("data", {}).get("candles", [])
        time.sleep(0.05)
    r = sess.get(f"{BASE}/intraday/{key}/days/1", timeout=30)
    if r.status_code == 200:
        got += r.json().get("data", {}).get("candles", [])
    if not got:
        print(f"  {name}: nothing", flush=True); continue
    d = pd.DataFrame(got).iloc[:, :6]
    d.columns = ["ts", "open", "high", "low", "close", "volume"]
    d["date"] = (pd.to_datetime(d["ts"], format="ISO8601", utc=True).dt.tz_convert("Asia/Kolkata")
                 .dt.tz_localize(None).dt.normalize())
    d["index_name"] = name
    d = d.drop(columns=["ts"]).drop_duplicates(subset=["index_name", "date"], keep="last")
    rows.append(d)
    print(f"  {name}: {len(d):,} sessions, {d['date'].min().date()} to {d['date'].max().date()}", flush=True)
out = pd.concat(rows, ignore_index=True).sort_values(["index_name", "date"]).reset_index(drop=True)
out.to_parquet(OUT, index=False)
print(f"\nwrote {OUT}: {len(out):,} rows, {out['index_name'].nunique()} series")
