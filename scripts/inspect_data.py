"""Data inspection report for stages 1-2. Read-only."""
import sys
sys.path.insert(0, ".")
import pandas as pd
from screener.store import Store
from screener.universe import Universe

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

root = sys.argv[1] if len(sys.argv) > 1 else "data"
st = Store(root)
u = Universe(st)


def head(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


head("STORE SUMMARY")
s = st.summary()
for k, v in s.items():
    print(f"  {k:15s} {v if not isinstance(v, list) or len(v) < 12 else f'{len(v)} items: {v[0]}..{v[-1]}'}")

head("UNIVERSE (point-in-time membership)")
members = u.members_on("NIFTY500", pd.Timestamp.today())
print(f"  NIFTY500 members today: {len(members)}")
print(members[["symbol", "isin", "name", "from_date"]].head(5).to_string(index=False))
sym = st.read_symbols()
print(f"\n  symbol master: {len(sym)} ISINs")
print(f"  symbol history rows: {len(st.read_symbol_history())}")
reused = u.reused_symbols()
print(f"  symbols mapped to >1 ISIN (reuse traps): {reused['symbol'].nunique() if len(reused) else 0}")
if len(reused):
    print(reused.head(6).to_string(index=False))

head("DAILY BARS — COVERAGE BY SOURCE")
cov = st.query("""
    SELECT
      COUNT(*)                                        AS rows,
      COUNT(DISTINCT isin)                            AS symbols,
      SUM(CASE WHEN close     IS NOT NULL THEN 1 ELSE 0 END) AS has_raw,
      SUM(CASE WHEN adj_close IS NOT NULL THEN 1 ELSE 0 END) AS has_adjusted,
      SUM(CASE WHEN close IS NOT NULL AND adj_close IS NOT NULL THEN 1 ELSE 0 END) AS has_both
    FROM bars
""")
print(cov.to_string(index=False))

print("\n  rows per year (first/last 5):")
per_year = st.query("SELECT year(date) AS yr, COUNT(*) AS rows, COUNT(DISTINCT isin) AS symbols FROM bars GROUP BY 1 ORDER BY 1")
print(pd.concat([per_year.head(5), per_year.tail(5)]).to_string(index=False))

head("TWO-SOURCE MERGE — implied adjustment factor (close / adj_close)")
both = st.query("""
    SELECT isin, symbol, date, close, adj_close, close / adj_close AS factor
    FROM bars WHERE close IS NOT NULL AND adj_close IS NOT NULL
""")
if len(both):
    print(f"  {len(both):,} bars carry BOTH raw and adjusted prices")
    print(f"  factor: min={both['factor'].min():.4f} median={both['factor'].median():.4f} max={both['factor'].max():.4f}")
    off = both[(both["factor"] - 1).abs() > 0.01]
    print(f"  bars where raw != adjusted (i.e. a corporate action sits in between): {len(off)}")
    if len(off):
        print(off.nlargest(8, "factor")[["symbol", "date", "close", "adj_close", "factor"]].to_string(index=False))
    print("\n  sample of matched bars:")
    print(both.head(5).to_string(index=False))
else:
    print("  none yet — run scripts/ingest_bhavcopy.py to fill the raw block")

head("CROSS-SOURCE AGREEMENT (bhavcopy raw vs Upstox adjusted)")
agree = st.query("""
    SELECT symbol, date, close, adj_close, ABS(close / adj_close - 1) AS dev
    FROM bars WHERE close IS NOT NULL AND adj_close IS NOT NULL
""")
if len(agree):
    exact = (agree["dev"] <= 1e-9).mean()
    print(f"  {len(agree):,} bars carry both sources")
    print(f"  exact to the last decimal: {exact * 100:.2f}%")
    print("  Two independent providers agreeing exactly is the strongest")
    print("  evidence available that both are right.")
    dev = agree[agree["dev"] > 1e-3]
    if len(dev):
        print(f"\n  {len(dev)} bar(s) deviate >0.1%, by date:")
        print(dev.groupby("date").size().to_string())
        today = pd.Timestamp.today().normalize()
        if set(dev["date"]) <= {today}:
            print("\n  NOTE: all deviations are on today's date. The Upstox daily")
            print("  candle for the current session is provisional, while the")
            print("  bhavcopy close is the official NSE figure (a 30-minute VWAP,")
            print("  not the last traded price). Re-fetch today's adjusted bar")
            print("  after settlement, or prefer the bhavcopy close for today.")
        print(dev.nlargest(6, "dev").to_string(index=False))
else:
    print("  no overlapping bars yet")

head("RAW-ONLY / ADJ-ONLY BREAKDOWN BY DATE (last 10 sessions)")
recent = st.query("""
    SELECT date,
           COUNT(*) AS bars,
           SUM(CASE WHEN close IS NOT NULL THEN 1 ELSE 0 END) AS raw,
           SUM(CASE WHEN adj_close IS NOT NULL THEN 1 ELSE 0 END) AS adjusted
    FROM bars GROUP BY 1 ORDER BY 1 DESC LIMIT 10
""")
print(recent.to_string(index=False))

head("MINUTE BARS")
ms = st.minute_summary()
if len(ms):
    print(f"  {ms['rows'].sum():,} rows across {len(ms)} month partitions")
    print(pd.concat([ms.head(3), ms.tail(3)]).to_string(index=False))
    day = st.query("SELECT 1") is not None
    per_day = st.read_minutes(start="2026-09-17 00:00", end="2026-09-17 23:59")
    if len(per_day):
        n = per_day.groupby("isin").size()
        print(f"\n  2026-09-17: {len(per_day):,} candles, {per_day['isin'].nunique()} symbols")
        print(f"  candles per symbol: min={n.min()} median={int(n.median())} max={n.max()} (a full session is 375)")
else:
    print("  none yet")

head("INGEST LOG")
log = st.read_ingest_log()
if len(log):
    print(log.groupby(["source", "status"]).agg(days=("date", "count"), rows=("rows", "sum")).to_string())
    bad = log[log["status"] == "error"]
    if len(bad):
        print(f"\n  {len(bad)} error day(s) — these stay gaps and are retried on the next run:")
        print(bad[["date", "source", "message"]].head(6).to_string(index=False))

head("QUARANTINE (rows the store refused)")
q = st.read_quarantine()
if len(q):
    print(f"  {len(q)} rows")
    print(q.groupby(["quarantine_source", "quarantine_reason"]).size().to_string())
    print("\n  sample:")
    cols = [c for c in ["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"] if c in q.columns]
    print(q[cols].head(6).to_string(index=False))
else:
    print("  empty")

head("SANITY CHECKS")
checks = []
d = st.query("SELECT COUNT(*) c FROM bars WHERE high < low").iloc[0]["c"]
checks.append(("no bar has high < low", d == 0, d))
d = st.query("SELECT COUNT(*) c FROM bars WHERE close <= 0 OR adj_close <= 0").iloc[0]["c"]
checks.append(("no non-positive prices", d == 0, d))
d = st.query("SELECT COUNT(*) c FROM bars WHERE volume < 0").iloc[0]["c"]
checks.append(("no negative volume", d == 0, d))
d = st.query("SELECT COUNT(*) c FROM (SELECT isin, date FROM bars GROUP BY 1,2 HAVING COUNT(*)>1)").iloc[0]["c"]
checks.append(("(isin, date) is unique", d == 0, d))
d = st.query("SELECT COUNT(*) c FROM bars WHERE close IS NULL AND adj_close IS NULL").iloc[0]["c"]
checks.append(("every bar has at least one price block", d == 0, d))
mem = st.read_membership()
d = int((mem["to_date"].notna() & (mem["to_date"] <= mem["from_date"])).sum())
checks.append(("membership intervals are non-empty", d == 0, d))
for name, ok, n in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f"  ({n} violations)"))
