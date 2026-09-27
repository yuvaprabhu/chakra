"""Export a compact JSON payload for the dashboard."""
import json, math, sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from screener.store import Store
from screener.universe import Universe
from screener.rules import load_rules

ROOT = "data"
OUT = Path("dashboard")
OUT.mkdir(exist_ok=True)
CHART_DAYS = 500

st = Store(ROOT)
u = Universe(st)
latest = pd.read_parquet("data/screen/latest_features.parquet")
index_payload = json.loads(Path("data/screen/indices.json").read_text())
catalog = pd.read_csv("data/raw/index_catalog.csv")
KIND = dict(zip(catalog["index_name"], catalog["kind"]))
hits = pd.read_parquet("data/screen/hits.parquet")
rules = load_rules("config/rules.yaml")
as_of = pd.Timestamp(latest["date"].iloc[0])

def r(x, nd=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    return round(float(x), nd)

# ---- price series on a shared date grid ----
isins = latest["isin"].tolist()
start = as_of - pd.Timedelta(days=int(CHART_DAYS * 1.55))
bars = st.read_bars(start=start, end=as_of, isins=isins,
                    columns=["isin", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"])
bars["date"] = pd.to_datetime(bars["date"])
grid = sorted(bars["date"].unique())[-CHART_DAYS:]
gidx = {d: i for i, d in enumerate(grid)}
bars = bars[bars["date"].isin(gidx)]

# EMA 9/21/50 and the volume ratio are computed in the browser now, because the
# chart has five timeframes and each needs its own. Over 500 daily sessions a
# client-side EMA(50) has long converged, so nothing is lost; the SMA(200) the
# SCREENS use still comes from the feature frame and is never recomputed here.
# Monthly pivots travel with the series: they are levels, not indicators.
PIVOT_COLS = ["m_p", "m_r1", "m_r2", "m_s1", "m_s2", "m_cpr_tc", "m_cpr_bc"]
feat = pd.read_parquet(
    "data/screen/features.parquet",
    columns=["isin", "date"] + PIVOT_COLS,
)
feat["date"] = pd.to_datetime(feat["date"])
feat = feat[feat["date"].isin(gidx)]
ma_by_isin = {k: v for k, v in feat.groupby("isin", sort=False)}
PIVOT_MONTHS = 3

series = {}
for isin, g in bars.groupby("isin", sort=False):
    g = g.sort_values("date")
    pos = g["date"].map(gidx).to_numpy()
    n = len(grid)
    o = [None]*n; h = [None]*n; l = [None]*n; c = [None]*n; v = [None]*n
    for i, p in enumerate(pos):
        o[p] = r(g["adj_open"].iloc[i]); h[p] = r(g["adj_high"].iloc[i])
        l[p] = r(g["adj_low"].iloc[i]);  c[p] = r(g["adj_close"].iloc[i])
        v[p] = int(g["volume"].iloc[i]) if pd.notna(g["volume"].iloc[i]) else None
    rec = {"o": o, "h": h, "l": l, "c": c, "v": v}
    mg = ma_by_isin.get(isin)
    if mg is not None:
        mg = mg.sort_values("date")
        # Monthly pivots as SEGMENTS, one per calendar month. Each month's levels
        # are constant across it because they come from the previous closed
        # month, so a segment is the honest shape: a level that held for a month,
        # not a line drifting through it.
        segs = []
        for (_, _), grp in list(mg.groupby([mg["date"].dt.year, mg["date"].dt.month],
                                           sort=True))[-PIVOT_MONTHS:]:
            row = grp.dropna(subset=["m_p"]).head(1)
            if row.empty:
                continue
            row = row.iloc[0]
            seg = {"a": int(gidx[grp["date"].iloc[0]]), "b": int(gidx[grp["date"].iloc[-1]])}
            for col in PIVOT_COLS:
                seg[col.replace("m_cpr_", "").replace("m_", "")] = r(row[col])
            segs.append(seg)
        if segs:
            rec["piv"] = segs
    series[isin] = rec

# ---- hourly bars from the minute store, aggregated in DuckDB ----
# Buckets are anchored at 09:15 so a bar is 09:15-10:15 ... 15:15-15:29, the
# way an Indian intraday chart is read. Timestamps are emitted as if the naive
# IST clock were UTC, which is what makes the chart library print 09:15
# instead of 03:45.
HOURLY_SESSIONS = 40
import duckdb
h_start = pd.Timestamp(grid[-HOURLY_SESSIONS]) if len(grid) >= HOURLY_SESSIONS else pd.Timestamp(grid[0])
hourly = {}
try:
    ph = ", ".join("?" * len(isins))
    sql = f"""
        SELECT isin,
               time_bucket(INTERVAL 60 MINUTE, ts, TIMESTAMP '2000-01-01 09:15:00') AS b,
               first(open ORDER BY ts) AS o, max(high) AS h, min(low) AS l,
               last(close ORDER BY ts) AS c, sum(volume) AS v
        FROM read_parquet(?, union_by_name=true, hive_partitioning=true)
        WHERE ts >= ? AND isin IN ({ph})
        GROUP BY isin, b ORDER BY isin, b
    """
    with duckdb.connect() as con:
        hb = con.execute(sql, [st.minutes_glob, h_start.to_pydatetime(), *isins]).fetch_df()
    hb = hb.dropna(subset=["c"])
    hb["b"] = pd.to_datetime(hb["b"])
    # Keep only bars inside the session; a stray 18:00 print is not a bar.
    hb = hb[(hb["b"].dt.hour * 60 + hb["b"].dt.minute).between(9 * 60 + 15, 15 * 60 + 15)]
    epoch = pd.Timestamp("1970-01-01")
    for isin, g in hb.groupby("isin", sort=False):
        g = g.sort_values("b")
        if len(g) < 14:
            continue
        hourly[isin] = {
            "t": [int((t - epoch).total_seconds()) for t in g["b"]],
            "o": [r(x) for x in g["o"]], "h": [r(x) for x in g["h"]],
            "l": [r(x) for x in g["l"]], "c": [r(x) for x in g["c"]],
            "v": [int(x) if pd.notna(x) else 0 for x in g["v"]],
        }
    print(f"hourly: {len(hourly)} symbols, {HOURLY_SESSIONS} sessions from {h_start.date()}")
except Exception as exc:
    print(f"hourly: skipped ({exc})")

# ---- per-symbol feature payload ----
FIELDS = [
    "mcap", "rs_rank", "dist_ema_21", "atr_ratio_60", "range_ratio",
    "vol_dryup", "base_depth", "sma_200_slope", "sma_150",
    "adj_close", "ret_1d", "ret_5d", "ret_20d", "ret_60d", "ret_120d", "ret_250d",
    "rsi_14", "atr_14", "atr_pct", "sma_20", "sma_50", "sma_200",
    "ema_9", "ema_21", "ema_50",
    "bb_upper", "bb_lower", "bb_mid", "bb_bandwidth", "bb_pct_b",
    "m_cpr_tc", "m_cpr_bc", "m_cpr_width", "m_p", "m_r1", "m_s1", "m_r2", "m_s2",
    "vol_ratio",
    "w_cpr_tc", "w_cpr_bc", "w_cpr_width", "d_cpr_tc", "d_cpr_bc", "d_cpr_width",
    "high_52w", "low_52w", "pct_from_52w_high", "pct_from_52w_low",
    "vol_ratio", "vol_sma_20", "turnover_median_20d", "bars_available", "volume",
    # event geometry - the columns that make a screen test its event
    "ext_ema21_15", "off_high_10", "hi_20_prior", "hi_60_prior", "hi_250_prior",
    "since_m_cpr_tc_cross", "since_w_cpr_tc_cross", "since_m_p_cross", "since_m_r1_cross",
    "dist_m_r1", "low3_dist_m_r1", "dist_m_cpr_tc", "dist_sma_50",
    "pocket_pivot", "down_vol_max_10", "tight_3w_pct", "gap_pct", "close_pos", "rsi_2",
    "new_listing",
    "since_r1_vol_breakout", "vol_vs_ema21", "vol_ema_21",
    "dist_ema_9", "low3_dist_ema_9",
    "spring", "since_spring", "spring_low", "dist_spring_low", "atr_to_spring", "lo_20_prior",
    "brk_level", "brk_cushion", "closes_above_brk", "since_brk_60", "brk_vol",
    "ud_vol_20", "close_pos_3", "lo7_prior", "hi7_prior", "new_lo7", "smooth_60",
    "stop_3atr", "stop_4atr", "tgt_3r_4atr",
    "dist_sma_200",
    "mkt_vix_pctile_1y", "mkt_nifty_range_20d", "mkt_gate_on", "mkt_breadth_50",
]
# Which indices each name belongs to, so the screener can filter by index and a
# stock's detail can show its memberships.
mem = st.read_membership()
mem = mem[mem["to_date"].isna()]
member_of: dict[str, list[str]] = {}
for isin, grp in mem.groupby("isin"):
    member_of[isin] = sorted(grp["index_name"].tolist())

symbols = {}
for _, row in latest.iterrows():
    d = {"sym": row["symbol"], "name": (row["name"] if pd.notna(row["name"]) else row["symbol"]),
         "basis": row["levels_basis"], "vol": int(row["volume"]) if pd.notna(row["volume"]) else None,
         "cap": row.get("cap_band"), "idx": member_of.get(row["isin"], [])}
    for f in FIELDS:
        if f in latest.columns:
            d[f] = r(row[f], 4 if f.endswith("width") else 2)
    symbols[row["isin"]] = d

# ---- screens ----
screens = []
for rule in rules:
    h = hits[hits["rule"] == rule.name]
    screens.append({
        "name": rule.name, "title": rule.title,
        "description": " ".join(rule.description.split()),
        "horizon": rule.horizon, "origin": rule.origin,
        "why": " ".join(rule.why.split()), "manage": " ".join(rule.manage.split()),
        "expr": " ".join(rule.expr.split()),
        "count": len(h),
        "emit": [e for e in rule.emit if e in latest.columns],
        "sort_by": rule.sort_by, "ascending": bool(rule.ascending),
        "min_turnover": rule.min_turnover, "min_price": rule.min_price,
        "isins": h["isin"].tolist(),
    })

# ---- breadth & market context ----
above = lambda col: int((latest["adj_close"] > latest[col]).sum())
adv = int((latest["ret_1d"] > 0).sum()); dec = int((latest["ret_1d"] < 0).sum())
breadth = {
    "above_sma20": above("sma_20"), "above_sma50": above("sma_50"),
    "above_sma200": above("sma_200"), "total": len(latest),
    "advancing": adv, "declining": dec,
    "unchanged": len(latest) - adv - dec,
    "median_rsi": r(latest["rsi_14"].median()),
    "new_highs": int((latest["pct_from_52w_high"] > -1).sum()),
    "new_lows": int((latest["pct_from_52w_low"] < 1).sum()),
    "median_ret_1d": r(latest["ret_1d"].median()),
    "median_ret_20d": r(latest["ret_20d"].median()),
}
rsi_hist, rsi_edges = np.histogram(latest["rsi_14"].dropna(), bins=10, range=(0, 100))
ret_hist, ret_edges = np.histogram(latest["ret_20d"].dropna().clip(-40, 40), bins=16, range=(-40, 40))
breadth["rsi_hist"] = rsi_hist.tolist()
breadth["ret20_hist"] = ret_hist.tolist()
breadth["ret20_edges"] = [r(x, 1) for x in ret_edges]

summary = st.summary()
log = st.read_ingest_log()

# The forward-return study, if it has been run. Optional: the screener works
# without it, it just cannot tell you which screen has edge.
bt_path = Path("data/screen/backtest.json")
backtest = json.loads(bt_path.read_text()) if bt_path.exists() else None
if backtest:
    print(f"backtest: {len(backtest['rules'])} rules over {backtest['sessions']} sessions "
          f"({backtest['start']} to {backtest['end']})")

# The same hits run as real bracket trades, with a stop and a target.
br_path = Path("data/screen/bracket.json")
bracket = json.loads(br_path.read_text()) if br_path.exists() else None
rv_path = Path("data/screen/reversal.json")
reversal = json.loads(rv_path.read_text()) if rv_path.exists() else None
# Verified strategy audit — the real-money source of truth for each screen.
sa_path = Path("data/screen/strategy_audit.json")
strategy_audit = json.loads(sa_path.read_text()) if sa_path.exists() else None
if strategy_audit:
    traded = [r for r in strategy_audit if r.get('family') != 'watchlist']
    print(f"strategy audit: {len(strategy_audit)} rules ({len(traded)} traded), "
          f"CAGR range {min(r['cagr_pct'] for r in traded):+.1f}% to "
          f"{max(r['cagr_pct'] for r in traded):+.1f}%")
if reversal:
    print(f"reversal exit: {len(reversal['rules'])} rules, {reversal['cost']}% round trip, "
          f"split {reversal['split']}")
if bracket:
    print(f"bracket: 1:{bracket['rr']:g} with a {bracket['max_bars']}-session time stop, "
          f"{len(bracket['setups'])} stop widths")
payload = {
    "meta": {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "index": "NIFTY 500",
        "universe": len(latest),
        "generated": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "daily_bars": int(summary["bars"]),
        "minute_bars": int(st.minute_summary()["rows"].sum()) if len(st.minute_summary()) else 0,
        "first_date": str(summary["first_date"].date()),
        "trading_days": int(summary["trading_days"]),
        "quarantined": int(len(st.read_quarantine())),
        "levels_basis": latest["levels_basis"].mode().iloc[0],
        "raw_days": int(len(st.available_dates(require="raw"))),
        "chart_days": len(grid),
        "min_mcap_cr": 5000,
        "cap_bands": latest["cap_band"].value_counts().to_dict(),
        "indices_tracked": len(index_payload["rows"]),
    },
    "dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in grid],
    "breadth": breadth,
    "screens": screens,
    "indices": [
        {**r, "kind": KIND.get(r["index_name"], "sector")}
        for r in index_payload["rows"]
    ],
    "index_context": index_payload["context"],
    "backtest": backtest,
    "bracket": bracket,
    "reversal": reversal,
    "strategy_audit": strategy_audit,
    "symbols": symbols,
}

# The artifact host caps a single file at 16 MB. Rather than hard-coding two
# shards, split each block into as many pieces as its own size needs, so the
# universe can grow without anyone remembering to re-shard by hand.
LIMIT_MB = 16
TARGET_MB = 7.0          # aim well under the cap; JSON size per name varies


def _jsonable(o):
    """Missing values reach here as pandas NA/NaT for names whose vendor data has
    gaps; anything else that is not JSON is a real bug and should still raise."""
    try:
        if o is pd.NaT or pd.isna(o):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f"cannot serialise {type(o).__name__}: {o!r}")


def dump(name, obj):
    q = OUT / name
    q.write_text(json.dumps(obj, separators=(",", ":"), default=_jsonable))
    mb = q.stat().st_size / 1e6
    flag = "  <-- OVER 16 MB" if mb > LIMIT_MB else ""
    print(f"wrote {q}  {mb:.2f} MB{flag}")
    return mb


def shard(prefix, obj):
    """Write obj across as many files as it takes to stay under the cap."""
    keys = sorted(obj)
    if not keys:
        return dump(f"{prefix}_a.json", {}) and [f"{prefix}_a.json"] or [f"{prefix}_a.json"]
    probe = len(json.dumps(obj, separators=(",", ":")).encode()) / 1e6
    n = max(1, math.ceil(probe / TARGET_MB))
    names = []
    for i in range(n):
        part = {k: obj[k] for k in keys[i::n]}          # round-robin keeps parts even
        fn = f"{prefix}_{chr(ord('a') + i)}.json"
        dump(fn, part)
        names.append(fn)
    return names


dump("data.json", payload)
series_files = shard("series", series)
hourly_files = shard("hourly", hourly)
payload["files"] = {"series": series_files, "hourly": hourly_files}
dump("data.json", payload)      # rewrite now that the shard names are known
print(f"  symbols={len(symbols)} series={len(series)} hourly={len(hourly)} dates={len(grid)}")
print(f"  screens: " + ", ".join(f"{s['title']}={s['count']}" for s in screens))
