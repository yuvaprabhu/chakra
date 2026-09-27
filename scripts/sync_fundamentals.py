"""Load the market-cap scan into the fundamentals table the universe is built from.

`scan_all_mcap.py` writes data/raw/all_nse_mcap.parquet, a Yahoo scan of every
NSE equity. That file is the raw material; this promotes the rows at or above
the floor into the store's fundamentals table, which is what
`screener.config.universe_isins` reads. Idempotent - `write_fundamentals`
upserts on ISIN.

  python3 scripts/sync_fundamentals.py                    # floor from config
  MCAP_FLOOR_CR=1000 python3 scripts/sync_fundamentals.py
"""
import argparse, sys
sys.path.insert(0, ".")
import pandas as pd
from screener.store import Store
from screener.config import MCAP_FLOOR_CR

ap = argparse.ArgumentParser()
ap.add_argument("--min-mcap-cr", type=float, default=MCAP_FLOOR_CR)
ap.add_argument("--scan", default="data/raw/all_nse_mcap.parquet")
a = ap.parse_args()

store = Store("data")
scan = pd.read_parquet(a.scan)
keep = scan[scan["mcap_cr"] >= a.min_mcap_cr].copy()
before = set(store.read_fundamentals()["isin"])
rows = pd.DataFrame({
    "isin": keep["isin"], "symbol": keep["symbol"], "shares": keep["shares"],
    "vendor_mcap": keep["mcap_cr"] * 1e7, "source": "yfinance_scan",
    "fetched_at": pd.Timestamp.today().normalize(),
})
res = store.write_fundamentals(rows)
after = store.read_fundamentals()
after["mcap_cr"] = after["vendor_mcap"] / 1e7
print(f"scan holds {len(scan):,} names; {len(keep):,} at or above Rs {a.min_mcap_cr:,.0f} Cr")
print(f"fundamentals: {res.inserted} inserted, {res.updated} updated "
      f"({len(set(keep['isin']) - before)} new to the store)")
print(f"universe now: {int((after['mcap_cr'] >= a.min_mcap_cr).sum()):,} names at this floor")
