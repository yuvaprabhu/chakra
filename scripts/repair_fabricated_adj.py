"""One-off repair: null out adjusted prices that a bhavcopy write fabricated.

An earlier version of parse_udiff copied close into adj_close. Those values are
raw prices wearing an adjusted label, and they overwrote genuine Upstox values.
They are identifiable because only Upstox ever sets adj_source.
"""
import sys
sys.path.insert(0, ".")
import pandas as pd
from screener import schema as S
from screener.store import Store, _atomic_write_parquet

store = Store(sys.argv[1] if len(sys.argv) > 1 else "data")
total = 0
for year in store.years():
    path = store._year_path(year)
    df = S.coerce(pd.read_parquet(path, engine="pyarrow"), S.BAR_DTYPES)
    bad = df["adj_close"].notna() & df["adj_source"].isna()
    n = int(bad.sum())
    if not n:
        continue
    df.loc[bad, S.ADJ_PRICE_COLUMNS] = pd.NA
    _atomic_write_parquet(S.coerce(df, S.BAR_DTYPES), path)
    print(f"  {year}: cleared {n:,} fabricated adjusted prices")
    total += n
print(f"total cleared: {total:,}")
