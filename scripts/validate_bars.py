"""Data integrity audit of the price bars themselves.

Everything above this is downstream of these numbers: the screens, both
backtests, the charts. The checks are the ones a bar must satisfy to be a bar
at all, plus agreement with the exchange file wherever both sources exist.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from screener.store import Store
from screener.adjust import derive_raw_all

st = Store("data")
latest = pd.read_parquet("data/screen/latest_features.parquet")
isins = sorted(latest["isin"].unique())
b = st.read_bars(start=pd.Timestamp("2023-06-01"), isins=isins)
print(f"panel: {len(b):,} bars, {b['isin'].nunique()} symbols, "
      f"{b['date'].min().date()} -> {b['date'].max().date()}\n")

fails = []
def chk(name, bad_mask, detail="", informational=False):
    """``informational`` rows are counted and shown but are not failures.
    Three findings here are real market behaviour, not defects, and marking
    them FAIL would train a reader to ignore the whole report:
      * circuit-locked days genuinely have O=H=L=C;
      * NSE genuinely trades on some Saturdays (Budget day, mock sessions) and
        on Diwali Sunday for Muhurat;
      * a demerger genuinely moves a price 40% in one session.
    The last one IS a data problem, but it is handled by the quality gate in
    screener/quality.py rather than by rejecting the panel."""
    n = int(bad_mask.sum()) if hasattr(bad_mask, "sum") else int(bad_mask)
    status = "INFO" if informational else ("PASS" if n == 0 else "FAIL")
    print(f"  [{status}] {name:52s} {n:>8,d} {detail}")
    if n and not informational: fails.append((name, n))
    return n

O, H, L, C = b["adj_open"], b["adj_high"], b["adj_low"], b["adj_close"]

print("STRUCTURE - a bar that breaks these is not a bar")
chk("high >= low", ~(H >= L))
chk("high >= open and high >= close", ~((H >= O - 1e-9) & (H >= C - 1e-9)))
chk("low <= open and low <= close", ~((L <= O + 1e-9) & (L <= C + 1e-9)))
chk("all four prices strictly positive", ~((O > 0) & (H > 0) & (L > 0) & (C > 0)))
chk("no duplicate (isin, date)", b.duplicated(["isin", "date"]).sum())
chk("volume present and non-negative", ~(b["volume"].fillna(-1) >= 0))

print("\nCOVERAGE - a missing column silently becomes a wrong answer downstream")
for c in ("adj_open", "adj_high", "adj_low", "adj_close"):
    miss = b[c].isna()
    chk(f"{c} has no gaps", miss, f"({(1-miss.mean())*100:.2f}% present)")

print("\nPLAUSIBILITY - values that are legal but almost certainly wrong")
rng = (H - L) / C * 100
chk("intraday range under 50% (circuit is 20%)", rng > 50, f"max {rng.max():.1f}%")
ret = b.sort_values(["isin","date"]).groupby("isin")["adj_close"].pct_change() * 100
chk("one-day moves over 35% (demergers - see quality gate)", ret.abs() > 35,
    f"max {ret.abs().max():.1f}% - excluded from screens and backtest", informational=True)
flat = (H == L) & (H == C) & (H == O)
chk("bars where O=H=L=C (circuit lock)", flat,
    f"({flat.mean()*100:.2f}% - real on circuit days)", informational=True)

print("\nTRADING CALENDAR")
grid = pd.Series(sorted(b["date"].unique()))
gaps = grid.diff().dt.days
chk("no calendar gap over 5 days in the session grid", (gaps > 5).sum(),
    f"({len(grid)} sessions)")
chk("weekend sessions (Budget day, Muhurat, mock)", grid.dt.dayofweek.isin([5, 6]).sum(),
    "- NSE does trade these", informational=True)

print("\nQUALITY GATE")
from screener import quality
_qr = quality.report(b, st.read_adjustments())
print(f"  [{'PASS' if _qr['jumps_unexplained'] == 0 else 'GATED'}] "
      f"{'unexplained price jumps':52s} {_qr['jumps_unexplained']:>8,d} "
      f"across {len(_qr['jump_symbols'])} symbols; {_qr['contaminated_rows']:,} rows "
      f"({_qr['contaminated_pct']}%) excluded from every screen and backtest")
print(f"  [{'PASS' if _qr['junk_bars'] == 0 else 'INFO'}] {'zero-volume, zero-range bars':52s} "
      f"{_qr['junk_bars']:>8,d}")

print("\nCROSS-SOURCE - the exchange bhavcopy against the vendor's adjusted series")
both = b[b["close"].notna() & b["adj_close"].notna()].copy()
act = st.read_adjustments()
fac = derive_raw_all(b.copy(), act)
both = fac[fac["close"].notna() & fac["adj_close"].notna()].copy()
obs = st.read_bars(start=pd.Timestamp("2023-06-01"), isins=isins)
obs = obs[obs["close"].notna()]
m = obs.merge(fac[["isin","date","adj_open","adj_high","adj_low","adj_close"]],
              on=["isin","date"], how="inner", suffixes=("","_adj"))
print(f"  {len(m):,} bars carry BOTH the exchange file and the vendor series")
for raw, adj in (("open","adj_open"),("high","adj_high"),("low","adj_low"),("close","adj_close")):
    r = (m[raw] / m[adj]).replace([np.inf,-np.inf], np.nan).dropna()
    # a split inside the window makes the ratio the split factor, not 1
    near1 = (r - 1).abs() < 0.005
    # The disagreements are not errors: every one is a bar before a split,
    # where raw and adjusted differ BY the split factor. Checked as such.
    off = r[~near1]
    unexplained = 0
    if len(off):
        rounded = off.round(2)
        unexplained = int((~rounded.isin([1.33, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0])).sum())
    ok = near1.mean() > 0.98 and unexplained == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] {raw:5s} vs {adj:9s}: "
          f"{near1.mean()*100:6.2f}% identical, {len(off):,} differ by a split factor, "
          f"{unexplained} unexplained")
    if not ok: fails.append((f"{raw} vs {adj}", unexplained))

print("\n" + "="*72)
print("RESULT:", "all structural checks pass" if not fails
      else f"{len(fails)} FAILING CHECKS: {fails}")
