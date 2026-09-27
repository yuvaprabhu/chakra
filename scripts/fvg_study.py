"""Fair Value Gap (FVG) and Inverse FVG (IFVG), tested long-only.

FVG (ICT): three candles where candle 1's high is below candle 3's low - the
middle candle moved so fast it left a range nobody traded. The zone between
high[t-2] and low[t] is expected to be revisited and to act as support.
  entry  first later bar whose low enters the zone: limit at the zone top
         (fills at the open if the bar opens inside the zone; skipped if it
         opens below the zone - the gap was blown through)
  stop   zone bottom minus a buffer (0.25 ATR daily, 0.1% hourly)
  exit   2R target, or the time cap, whichever first; stop checked first

IFVG: a BEARISH gap (high[t] < low[t-2]) that price then closes ABOVE.  The
failed bearish gap flips to support ("inversion"); entry on the first retrace
into it, same stop/target logic.

Daily: last 2 years and the last month.  Hourly (60-minute bars from the
minute store): the last month, which is the timeframe FVG traders use.
"""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd, duckdb
from screener.store import Store
from screener import quality

t0 = time.time()
COST_D, COST_H = 0.30, 0.30
feat = pd.read_parquet("data/screen/features.parquet",
                       columns=["isin","symbol","date","adj_open","adj_high","adj_low","adj_close","volume","atr_pct","sma_200","ret_60d","ret_120d","ret_250d","turnover_median_20d"])
from screener.backtest import add_panel_columns
feat = add_panel_columns(feat)
st = Store("data")
b = feat[["isin","date","adj_open","adj_high","adj_low","adj_close","volume"]]
feat["contaminated"] = quality.contamination_mask(b, quality.detect_price_jumps(b, st.read_adjustments())).to_numpy()
feat = feat.sort_values(["isin","date"]).reset_index(drop=True)
LAST_DATE = feat.date.max()
print("panel", len(feat), "last", LAST_DATE.date(), f"{time.time()-t0:.0f}s", flush=True)

def simulate(o, h, l, c, atrp, ok, *, kind, valid_bars, cap, buf_atr=None, buf_pct=None, rr=2.0, min_gap_atr=None, min_gap_pct=None):
    """Return list of (setup_i, entry_i, exit_i, entry, exit, why, R) for one symbol."""
    n = len(c); out = []; busy = -1
    for t in range(2, n):
        if not ok[t]: continue
        if kind == "fvg":
            if not (l[t] > h[t-2]): continue
            bot, top = h[t-2], l[t]
            if not (c[t-1] > o[t-1]): continue           # the displacement candle is up
            start = t + 1
        else:  # ifvg: bearish gap, then a close above it
            if not (h[t] < l[t-2]): continue
            bot, top = h[t], l[t-2]
            inv = None
            for k in range(t+1, min(n, t+1+valid_bars)):
                if c[k] > top: inv = k; break
            if inv is None: continue
            start = inv + 1
        gap = top - bot
        ref = atrp[t] * c[t] / 100 if atrp is not None else None
        if min_gap_atr is not None and (not np.isfinite(ref) or gap < min_gap_atr * ref): continue
        if min_gap_pct is not None and gap / c[t] * 100 < min_gap_pct: continue
        buf = buf_atr * ref if buf_atr is not None else buf_pct / 100 * top
        stop = bot - buf
        if start <= busy: continue
        # first retrace into the zone
        e = None
        for k in range(start, min(n, start + valid_bars)):
            if c[k] < bot: break                              # zone blown through before a fill: setup dead
            if l[k] <= top:
                if o[k] < bot: break                          # opens below the zone: no fill
                e = min(o[k], top); ei = k; break
        if e is None: continue
        risk = e - stop
        if risk <= 0 or risk / e > 0.2: continue
        tgt = e + rr * risk
        out_px = None; why = "time"; xi = min(n-1, ei + cap)
        for k in range(ei, min(n, ei + cap + 1)):
            lo_k = l[k] if k > ei else min(l[k], e)          # entry bar: after the fill only the low matters
            if lo_k <= stop and k > ei or (k == ei and l[k] <= stop):
                out_px = stop if k > ei or o[k] > stop else min(o[k], stop); why = "stop"; xi = k; break
            if h[k] >= tgt and k > ei:
                out_px = tgt; why = "target"; xi = k; break
            if k == ei and h[k] >= tgt and c[k] >= tgt:      # same-bar target only if it closed there (conservative)
                out_px = tgt; why = "target"; xi = k; break
        if out_px is None: out_px = c[xi]
        busy = xi
        out.append((t, ei, xi, e, out_px, why, (out_px - e) / risk))
    return out

def run_daily(start, label, trend=False, rs=None, kind="fvg"):
    rows = []
    for isin, g in feat[feat.date >= pd.Timestamp(start) - pd.Timedelta(days=40)].groupby("isin", sort=False):
        if len(g) < 30: continue
        o,h,l,c = (g[k].to_numpy(float) for k in ("adj_open","adj_high","adj_low","adj_close"))
        atrp = g["atr_pct"].to_numpy(float)
        ok = (g.date >= pd.Timestamp(start)).to_numpy() & (~g.contaminated.to_numpy()) & (g.turnover_median_20d.to_numpy() >= 1e7)
        if trend: ok &= (g.adj_close > g.sma_200).to_numpy()
        if rs is not None: ok &= (g.rs_rank.to_numpy() >= rs)
        for (ti, ei, xi, e, x, why, R) in simulate(o,h,l,c,atrp,ok,kind=kind,valid_bars=10,cap=10,buf_atr=0.25,min_gap_atr=0.25):
            rows.append((isin, g.symbol.iloc[0], g.date.iloc[ti], g.date.iloc[ei], g.date.iloc[xi], e, x, why, R, (x/e-1)*100 - COST_D, xi-ei))
    return pd.DataFrame(rows, columns=["isin","symbol","setup","entry","exit","e","x","why","R","net","bars"])

def summarise(t, label):
    if len(t) == 0: return {"label": label, "n": 0}
    d = {"label": label, "n": int(len(t)), "win": round((t.net>0).mean()*100,1), "mean_net": round(t.net.mean(),2),
         "mean_R": round(t.R.mean(),2), "target": round((t.why=="target").mean()*100,0), "stopped": round((t.why=="stop").mean()*100,0),
         "timed": round((t.why=="time").mean()*100,0), "avg_bars": round(t.bars.mean(),1), "pf": round(t.net[t.net>0].sum()/max(1e-9,-t.net[t.net<=0].sum()),2)}
    print(f"{label:52s} n={d['n']:5d} win {d['win']:5.1f}% mean {d['mean_net']:+.2f}% ({d['mean_R']:+.2f}R) target {d['target']:.0f}% stop {d['stopped']:.0f}% time {d['timed']:.0f}% PF {d['pf']:.2f}", flush=True)
    return d

res = {"daily": [], "hourly": []}
import os
RUN_DAILY = os.environ.get("FVG_DAILY","1")=="1"
if RUN_DAILY: print("\nDAILY, 2R target, 10-bar cap, stop under the zone (0.25 ATR), gap >= 0.25 ATR, cost 0.30%")
for kind in (("fvg", "ifvg") if RUN_DAILY else ()):
    for start, lab in (("2024-09-22", "last 2y"), ("2026-08-22", "last 1m")):
        for trend, rs, f in ((False, None, "all"), (True, None, "above 200-SMA"), (True, 80, "above 200-SMA & RS>=80")):
            t = run_daily(start, lab, trend, rs, kind)
            res["daily"].append(summarise(t, f"{kind.upper()} daily {lab} | {f}"))
            if kind == "fvg" and lab == "last 2y" and f == "all":
                print(t.sort_values("entry").head(5).to_string(index=False))

# ---- hourly, last month ----
grid = sorted(feat.date.unique()); h_start = pd.Timestamp(grid[-22])
isins = feat.loc[feat.date == LAST_DATE, "isin"].tolist()
sql = f"""SELECT isin, time_bucket(INTERVAL 60 MINUTE, ts, TIMESTAMP '2000-01-01 09:15:00') AS b,
          first(open ORDER BY ts) o, max(high) h, min(low) l, last(close ORDER BY ts) c, sum(volume) v
          FROM read_parquet(?, union_by_name=true, hive_partitioning=true)
          WHERE ts >= ? AND isin IN ({", ".join("?"*len(isins))}) GROUP BY isin, b ORDER BY isin, b"""
with duckdb.connect() as con:
    hb = con.execute(sql, [st.minutes_glob, h_start.to_pydatetime(), *isins]).fetch_df()
hb = hb.dropna(subset=["c"]); hb["b"] = pd.to_datetime(hb["b"])
hb = hb[(hb["b"].dt.hour*60 + hb["b"].dt.minute).between(9*60+15, 15*60+15)]
hb["sess"] = hb["b"].dt.normalize()
lastf = feat[feat.date == LAST_DATE].set_index("isin")
trend_ok = (lastf.adj_close > lastf.sma_200); rs80 = lastf.rs_rank >= 80
liquid = (lastf.turnover_median_20d >= 1e7) & (~lastf.contaminated)
print(f"\nHOURLY bars: {len(hb):,} rows, {hb['isin'].nunique()} names, {hb.sess.nunique()} sessions from {h_start.date()}")
print("HOURLY, last month, 2R target, 20-bar (~3 session) cap, stop 0.1% under the zone, gap >= 0.15%, cost 0.30%")
for kind in ("fvg", "ifvg"):
    for f, mask in (("all", liquid), ("above 200-SMA", liquid & trend_ok), ("above 200-SMA & RS>=80", liquid & trend_ok & rs80)):
        rows = []
        for isin, g in hb.groupby("isin", sort=False):
            if isin not in mask.index or not mask.loc[isin] or len(g) < 30: continue
            o,h,l,c = (g[k].to_numpy(float) for k in "ohlc")
            ok = np.ones(len(g), bool)
            for (ti, ei, xi, e, x, why, R) in simulate(o,h,l,c,None,ok,kind=kind,valid_bars=20,cap=20,buf_pct=0.1,min_gap_pct=0.15):
                rows.append((isin, g.b.iloc[ti], g.b.iloc[ei], g.b.iloc[xi], e, x, why, R, (x/e-1)*100 - COST_H, xi-ei))
        t = pd.DataFrame(rows, columns=["isin","setup","entry","exit","e","x","why","R","net","bars"])
        d = summarise(t, f"{kind.upper()} hourly last 1m | {f}")
        if len(t): d["gross_mean"] = round((t.net + COST_H).mean(), 2); print(f"{'':52s} gross mean {d['gross_mean']:+.2f}%")
        res["hourly"].append(d)
json.dump(res, open("data/screen/fvg_study.json","w"), indent=1)
print(f"\ndone {time.time()-t0:.0f}s")
