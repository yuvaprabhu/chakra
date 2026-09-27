"""Compute features for the whole universe, evaluate rules, build index analytics."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from screener.config import MCAP_FLOOR_CR
from screener.store import Store
from screener.universe import Universe
from screener.adjust import derive_raw_all, verify
from screener.indicators import build_features
from screener.indices import build_index_analytics
from screener.rules import load_rules, run_rules

ap = argparse.ArgumentParser()
ap.add_argument("--rules", default="config/rules.yaml")
ap.add_argument("--min-mcap-cr", type=float, default=MCAP_FLOOR_CR)
ap.add_argument("--benchmark", default="NIFTY 500")
ap.add_argument("--root", default="data")
ap.add_argument("--history-days", type=int, default=800)
ap.add_argument("--out", default="data/screen")
a = ap.parse_args()

st = Store(a.root)
u = Universe(st)
as_of = st.available_dates().max()

# The universe is EVERY NSE equity above the market-cap floor. Index membership
# was the gate until it turned out that names like AstraZeneca and P&G Hygiene
# sit in no Nifty index at all; it is now only a filter dimension in the UI.
mem = st.read_membership()
live = mem[mem["to_date"].isna()]
fund = st.read_fundamentals()
fund["mcap_cr"] = fund["vendor_mcap"] / 1e7
isins = sorted(fund.loc[fund["mcap_cr"] >= a.min_mcap_cr, "isin"].unique())
in_index = len(set(isins) & set(live["isin"]))
print(f"universe: {len(isins)} NSE equities >= Rs {a.min_mcap_cr:,.0f} Cr "
      f"({in_index} in an index, {len(isins)-in_index} outside one), as-of {as_of.date()}", flush=True)

t0 = time.time()
bars = st.read_bars(start=as_of - pd.Timedelta(days=int(a.history_days * 1.5)), end=as_of, isins=isins)
print(f"loaded {len(bars):,} bars in {time.time()-t0:.1f}s", flush=True)

# Fill the raw price block from the adjusted series plus the corporate-action
# table, so pivots and CPR compute on raw prices across the whole history rather
# than only the handful of sessions the bhavcopy happens to cover. Observed
# bhavcopy prices are left untouched and are used to check the derivation.
actions = st.read_adjustments()
observed = int(bars["close"].notna().sum())
chk = verify(bars, actions)
if len(chk):
    agree = chk["ok"].mean() * 100
    print(f"derivation check: {len(chk):,} bars carry both sources, "
          f"{agree:.3f}% agree within 0.5% (max err {chk['rel_err'].max()*100:.3f}%)", flush=True)
    if agree < 99.0:
        print("  WARNING: derivation disagrees with the exchange file; "
              "the corporate-action table is probably incomplete", flush=True)
bars = derive_raw_all(bars, actions)
print(f"raw block: {observed:,} observed + {int(bars['close'].notna().sum())-observed:,} derived "
      f"= {int(bars['close'].notna().sum()):,} of {len(bars):,} bars", flush=True)

t0 = time.time()
# Quality gate before anything is screened. An unadjusted corporate action
# bends every window that spans it, and nothing downstream can tell that apart
# from a real move - so the affected rows are excluded and counted, never
# quietly used. See screener/quality.py.
from screener import quality
qr = quality.report(bars, actions)
print(f"quality gate: {qr['jumps_unexplained']} unexplained price jumps across "
      f"{len(qr['jump_symbols'])} symbols | {qr['contaminated_rows']:,} rows "
      f"({qr['contaminated_pct']}%) excluded | {qr['junk_bars']} junk bars | "
      f"OHLC violations {qr['ohlc_violations']}, duplicates {qr['duplicates']}", flush=True)
if qr["ohlc_violations"] or qr["non_positive"] or qr["duplicates"]:
    raise SystemExit("ABORT: the price panel is structurally invalid")

feat = build_features(bars)
feat["bars_available"] = feat.groupby("isin")["date"].transform("size")
_jumps = quality.detect_price_jumps(bars, actions)
feat["contaminated"] = quality.contamination_mask(
    feat[["isin", "date"]].assign(**{c: 0 for c in ()}), _jumps).to_numpy()
# Short history is a fact about the listing, not a defect. Flagged so the
# app can say "new listing" rather than show blank indicators unexplained.
feat["new_listing"] = (feat["bars_available"] < 60).astype(int)
print(f"features for {feat['isin'].nunique()} symbols in {time.time()-t0:.1f}s", flush=True)

# Market state (VIX percentile, Nifty range, breadth gate) and fresh-entry
# flags, so a rule can say "first day in a Stage 2 stack while the tape is
# volatile". See screener/market.py.
from screener.market import attach_market, add_fresh_flags
feat = attach_market(feat)
_rules_for_fresh = {r.name: r.expr for r in load_rules(a.rules) if r.name in ("trend_stack", "momentum_leaders")}
feat = add_fresh_flags(feat, _rules_for_fresh)
# Volume-lab features: multi-year breakout highs (2-year and 3-month close highs).
_g = feat.groupby("isin")
feat["close_hi_500"] = _g["adj_close"].transform(lambda s: s.shift(1).rolling(500, min_periods=250).max())
feat["close_hi_20"] = _g["adj_close"].transform(lambda s: s.shift(1).rolling(20).max())
feat["range_500_pct"] = (_g["adj_close"].transform(lambda s: s.shift(1).rolling(500).max())
                          / _g["adj_close"].transform(lambda s: s.shift(1).rolling(500).min()) - 1) * 100
print(f"market columns attached; gate {'ON' if int(feat.loc[feat['date'] == as_of, 'mkt_gate_on'].iloc[0]) else 'OFF'}, "
      f"VIX pctile {feat.loc[feat['date'] == as_of, 'mkt_vix_pctile_1y'].iloc[0]:.0f}, "
      f"Nifty 20d range {feat.loc[feat['date'] == as_of, 'mkt_nifty_range_20d'].iloc[0]:.1f}%", flush=True)

latest = feat[feat["date"] == as_of].copy()
# Names come from the index files where available, else the instrument master.
info = live.drop_duplicates("isin").set_index("isin")
from screener.upstox import load_instruments
inst = load_instruments("data/raw/upstox_nse_instruments.csv").drop_duplicates("isin").set_index("isin")
latest["name"] = latest["isin"].map(info["name"]).fillna(latest["isin"].map(inst["name"]))
latest["symbol"] = (latest["isin"].map(info["symbol"])
                    .fillna(latest["isin"].map(inst["tradingsymbol"]))
                    .fillna(latest["symbol"]))
latest = latest.merge(fund[["isin", "shares", "mcap_cr"]], on="isin", how="left")
# Market cap from OUR close, so it agrees with every other price on the screen.
latest["mcap"] = np.where(latest["shares"].notna(),
                          latest["shares"] * latest["adj_close"] / 1e7, latest["mcap_cr"])
CAP_BANDS = [(1_00_000, "Mega"), (20_000, "Large"), (5_000, "Mid"), (0, "Small")]
latest["cap_band"] = "Small"
for lo, lab in CAP_BANDS:
    latest.loc[latest["mcap"] >= lo, "cap_band"] = lab
    break_ = None
latest["cap_band"] = pd.cut(latest["mcap"], [0, 20_000, 1_00_000, np.inf],
                            labels=["Mid", "Large", "Mega"], right=False).astype(str)

# The floor has to be applied to the SAME number the screen displays. The
# vendor's market cap and ours (shares x our own close) disagree badly for
# names whose share count has not caught up with a split, which is how a
# sub-floor company slipped into a universe defined by that floor.
gap = (latest["mcap_cr"] / latest["mcap"]).replace([np.inf, -np.inf], np.nan)
stale = latest.loc[gap > 3, ["symbol", "mcap", "mcap_cr"]].sort_values("mcap")
if len(stale):
    print(f"market cap disagreement: {len(stale)} name(s) where the vendor figure is more than "
          f"3x ours - the vendor share count is stale, e.g. "
          f"{', '.join(f'{r.symbol} ({r.mcap:,.0f} vs {r.mcap_cr:,.0f} Cr)' for r in stale.head(4).itertuples())}",
          flush=True)
below = latest["mcap"] < a.min_mcap_cr
if below.any():
    names = ", ".join(latest.loc[below].nsmallest(5, "mcap")["symbol"].astype(str))
    print(f"dropping {int(below.sum())} name(s) below Rs {a.min_mcap_cr:,.0f} Cr on our own "
          f"market cap: {names}{' ...' if below.sum() > 5 else ''}", flush=True)
    latest = latest.loc[~below].reset_index(drop=True)
print(f"latest snapshot: {len(latest)} rows | cap bands: {latest['cap_band'].value_counts().to_dict()}", flush=True)

# Relative strength as a cross-sectional percentile, the way IBD and Minervini
# use it: a stock is strong relative to its peers today, not in the abstract.
# Weighted to favour recent quarters, then ranked 0-99.
for col, w in (("ret_60d", 0.4), ("ret_120d", 0.2), ("ret_250d", 0.4)):
    latest[col] = pd.to_numeric(latest[col], errors="coerce")
latest["rs_score"] = (0.4 * latest["ret_60d"].rank(pct=True)
                      + 0.2 * latest["ret_120d"].rank(pct=True)
                      + 0.4 * latest["ret_250d"].rank(pct=True))
latest["rs_rank"] = (latest["rs_score"].rank(pct=True) * 99).round(0)
print(f"RS rank computed across {int(latest['rs_rank'].notna().sum())} names", flush=True)

rules = load_rules(a.rules)
hits = run_rules(latest, rules)
print("\nrule hits:")
for r in rules:
    print(f"  {r.name:24s} {len(hits[r.name]):>4d}")

t0 = time.time()
index_names = sorted(live["index_name"].unique())
rows, ctx = build_index_analytics(st, latest, index_names, benchmark=a.benchmark, as_of=as_of)
print(f"\nindex analytics for {len(rows)} indices in {time.time()-t0:.1f}s", flush=True)
top = sorted([r for r in rows if r.get("ret_1D") is not None], key=lambda r: -r["ret_1D"])[:5]
for r in top:
    print(f"  {r['index_name']:26s} 1D {r['ret_1D']:+6.2f}%  flow {r.get('flow_delta',0):+5.2f}pp  {r.get('quadrant','-')}")

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
latest.to_parquet(out / "latest_features.parquet", index=False)
feat.to_parquet(out / "features.parquet", index=False)
pd.concat([h for h in hits.values() if len(h)], ignore_index=True).to_parquet(out / "hits.parquet", index=False)
(out / "indices.json").write_text(json.dumps({"context": ctx, "rows": rows}, default=str))
print(f"\nwritten to {out}/")
