"""Bottom lab: is there a bottom-finding strategy?

Ten popular "buy the bottom" setups (capitulation reversal, Connors exhaustion,
RSI(14) oversold turn, Bollinger re-entry, 200-DMA reclaim, undercut-and-rally,
double bottom, extreme distance from the 200-DMA, weekly 10-week SMA reclaim,
52-week-low bounce), each with a HARD stop, each also in a "bottom in a leader"
(rs_rank >= 60) and a "market washout" (breadth < 30% within 20 sessions) form,
backtested on two windows:

  * EXTENDED  signals 2018-06-01 .. 2026-09-22, halves split 2022-06-30
              (2018-19 midcap bear, 2020 crash, 2022 correction, 2024-25, 2026Q1)
  * RECENT    signals 2024-06-01 .. 2026-09-22, halves split 2025-03-19
              (the window every other lab used; Leader Dip is the bar to beat)

Panel: built from the raw daily store for 2017-01-01 .. 2026-09-22 over today's
>= Rs 1,000 Cr universe (screener.config.universe_isins), quality-gated with
screener.quality, features from screener.indicators.build_features, rs_rank /
bars_available from screener.backtest.add_panel_columns, market columns from
screener.market.attach_market.  BOTH WINDOWS CARRY SURVIVORSHIP BIAS: the
universe is today's survivors, so names that bottomed and never came back are
missing.  A market-cap-at-signal variant bounds that.

Trade engine: scripts/stop_reward_study.py / swing_lab_pullbacks.py run()
conventions, unchanged: entry next open, stop checked first each bar, gap
through the stop fills at the open, close-based exits fill at the next open,
one open position per name, 0.30% cost, time cap.  The per-trade loop is
vectorised here for speed and checked against a verbatim copy of the original
loop at start-up (see --check).

Output: data/screen/bottom_lab.json
Run:    python scripts/bottom_lab.py --build      (panel cache, ~5 min)
        python scripts/bottom_lab.py              (both windows: matrix, grooming, stress, quarterly)
        python scripts/bottom_lab.py --window ext | --window recent
"""
import json, sys, time, itertools, os, argparse
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--build", action="store_true")
ap.add_argument("--check", action="store_true", help="engine validation + sanity trades only")
ap.add_argument("--window", default="both", choices=["both", "ext", "recent"])
ap.add_argument("--panel", default=os.environ.get("BOTTOM_LAB_PANEL",
                "/tmp/claude-0/-home-user/397386f6-c392-5ccc-b9f1-52240babee74/scratchpad/bl/bottom_panel.parquet"))
ap.add_argument("--procs", type=int, default=4)
ARGS = ap.parse_args()

COST, SLOTS = 0.30, 10
PANEL_START, PANEL_END = pd.Timestamp("2017-01-01"), pd.Timestamp("2026-09-22")
WINDOWS = {"ext": {"start": pd.Timestamp("2018-06-01"), "split": pd.Timestamp("2022-06-30"),
                   "quarters": ("2018Q3", "2026Q3")},
           "recent": {"start": pd.Timestamp("2024-06-01"), "split": pd.Timestamp("2025-03-19"),
                      "quarters": ("2024Q3", "2026Q3")}}
CAPS = [60, 20]
MIN_N = 100                       # trades per half
MIN_KEPT_PER_MONTH = 15
T_MIN = 2.0
Q_MIN_SHARE, Q_WORST = 0.65, -8.0
MIN_TURNOVER, MIN_BARS, MIN_PRICE = 1e7, 250, 10.0
FAMILIES = ["A", "B", "C", "D", "E", "E2", "F", "G", "H", "I", "J"]
TWISTS = ["", "_ldr", "_wash"]
EXITS = [("rev7",), ("trail", "ema21"), ("trail", "ema50"), ("target", 2.0), ("target", 3.0), ("none",)]
ATR_STOPS = [("atr", 3.0), ("atr", 4.0)]
STRUCT_OF = {f: "bar_low" for f in FAMILIES}
STRUCT_OF["G"] = "low10"
TITLE = {"A": "Capitulation reversal", "B": "Connors exhaustion", "C": "RSI14 oversold turn",
         "D": "Bollinger re-entry", "E": "200-DMA reclaim", "E2": "200-DMA reclaim, slope>=0",
         "F": "Undercut and rally", "G": "Double bottom", "H": "25% under 200-DMA, RSI turn",
         "I": "Weekly 10w-SMA reclaim", "J": "52w-low bounce"}
DEFS = {"A": "close <= 30% below 52w high; volume >= 3x 20d avg; close in top 30% of range; close > prev close",
        "B": ">= 5 consecutive lower closes ending yesterday; RSI(2) yesterday < 5; today close > prev close",
        "C": "RSI(14) < 30 on one of the last 5 sessions, < 30 yesterday, >= 30 today",
        "D": "close < lower Bollinger(20,2) yesterday (so within the last 3 sessions), first close back inside today",
        "E": "close < 200-DMA for >= 40 consecutive sessions ending yesterday; close > 200-DMA today",
        "E2": "E and the 200-DMA 20-session slope >= 0",
        "F": "today's low < lowest low of prior 20 sessions (lo_20_prior); close > that low",
        "G": "swing low (11-bar centre) 15-60 sessions ago, >= 25% under its 52w high; min low of last 10 sessions within +/-3% of it and >= 25% under the 52w high; today's close > max high between the lows; no G signal in the prior 5 sessions",
        "H": "close >= 25% below the 200-DMA; RSI(14) > RSI(14) 3 sessions ago",
        "I": "last session of the week; close > 50-day SMA; the previous >= 8 week-end closes were all below it",
        "J": "yesterday's close within 5% of the 52w low; today close >= +4%; volume >= 2x 20d avg",
        "_ldr": "twist: rs_rank >= 60 at the signal", "_wash": "twist: min mkt_breadth_50 over the last 20 sessions < 30",
        "base": f"turnover_median_20d >= {MIN_TURNOVER:.0e}, bars_available >= {MIN_BARS}, adj_close >= {MIN_PRICE}, not contaminated",
        "struct_stop": "reversal bar's low (G: min low of last 10 sessions), accepted only if 3-20% below entry"}
STOCK_FEATS = ["rs_rank", "pct_from_52w_high", "dist_sma_200", "dist_sma_50", "atr_pct", "vol_ratio", "ret_20d",
               "ret_60d", "ret_250d", "turnover_median_20d", "bars_since_52w_low", "dd_depth_52w"]
MARKET_NUM = ["mkt_breadth_50", "mkt_vix", "mkt_vix_pctile_1y", "mkt_nifty_range_20d", "mkt_washout"]
MARKET_BIN = ["mkt_gate_on"]
KEEP = ["isin", "symbol", "date", "adj_open", "adj_high", "adj_low", "adj_close", "volume", "sma_50", "sma_200",
        "sma_200_slope", "ema_21", "ema_50", "rsi_14", "rsi_2", "atr_pct", "bb_lower", "ret_1d", "ret_20d", "ret_60d",
        "ret_120d", "ret_250d", "high_52w", "low_52w", "pct_from_52w_high", "pct_from_52w_low", "close_pos", "vol_ratio",
        "lo_20_prior", "spring", "hi7_prior", "dist_sma_200", "dist_sma_50", "dist_ema_21", "turnover_median_20d",
        "contaminated"]
t0 = time.time()


def jdump(o):
    return None if (isinstance(o, float) and not np.isfinite(o)) else (o.item() if hasattr(o, "item") else str(o))


# ------------------------------------------------------------------ panel build
def _feat_chunk(bars):
    from screener.indicators import build_features
    f = build_features(bars)
    return f[[c for c in KEEP if c in f.columns]]


def build_panel(path):
    from screener.store import Store
    from screener.config import universe_isins
    from screener.backtest import add_panel_columns
    from screener.market import attach_market
    from screener import quality
    st = Store("data"); isins = universe_isins(st, 1000)
    bars = st.read_bars(start=PANEL_START, end=PANEL_END, isins=isins)
    print(f"{len(bars):,} bars for {bars['isin'].nunique()} names {bars['date'].min().date()}..{bars['date'].max().date()} | {time.time()-t0:.0f}s", flush=True)
    actions = st.read_adjustments()
    jumps = quality.detect_price_jumps(bars, actions)
    bars["contaminated"] = quality.contamination_mask(bars, jumps).to_numpy()
    print(f"quality: {int((~jumps['explained']).sum()) if len(jumps) else 0} unexplained jumps, "
          f"{bars['contaminated'].mean()*100:.2f}% rows contaminated | {time.time()-t0:.0f}s", flush=True)
    groups = [g for _, g in bars.groupby("isin", sort=False)]
    chunks = [pd.concat(groups[i:i + 25]) for i in range(0, len(groups), 25)]
    with Pool(ARGS.procs) as pool:
        parts = pool.map(_feat_chunk, chunks, chunksize=1)
    feat = pd.concat(parts, ignore_index=True)
    print(f"features {len(feat):,} rows | {time.time()-t0:.0f}s", flush=True)
    feat = add_panel_columns(feat)
    feat = attach_market(feat).sort_values(["isin", "date"]).reset_index(drop=True)
    feat.to_parquet(path, index=False)
    print(f"panel cached at {path} ({os.path.getsize(path)/1e6:.0f} MB) | {time.time()-t0:.0f}s", flush=True)
    return feat


if ARGS.build:
    build_panel(ARGS.panel); sys.exit(0)
if not Path(ARGS.panel).exists():
    build_panel(ARGS.panel)

# ------------------------------------------------------------------ panel load, derived columns
feat = pd.read_parquet(ARGS.panel).sort_values(["isin", "date"]).reset_index(drop=True)
lat = pd.read_parquet("data/screen/latest_features.parquet")[["isin", "cap_band", "shares", "symbol"]]
feat = feat.drop(columns=["symbol"]).merge(lat, on="isin", how="left")
N = len(feat)
CAL = pd.DatetimeIndex(sorted(feat["date"].unique()))
CALPOS = {d: i for i, d in enumerate(CAL)}
END = CAL[-1]
IC = pd.factorize(feat["isin"])[0]
ISIN = feat["isin"].to_numpy(); SYM = feat["symbol"].to_numpy(); DATES = feat["date"].to_numpy()
LAST = pd.Series(np.arange(N)).groupby(IC).transform("max").to_numpy()


def lag(a, k):
    """a shifted k rows back, NaN where that crosses into another name."""
    a = np.asarray(a, float); out = np.full(N, np.nan)
    out[k:] = a[:-k]
    out[k:][IC[k:] != IC[:-k]] = np.nan
    return out


def runlen(flag):
    """Consecutive-True count ending at each row, within name."""
    f = pd.Series(np.asarray(flag, bool))
    return f.astype(int).groupby([pd.Series(IC), (~f).cumsum()]).cumsum().to_numpy()


O = feat["adj_open"].to_numpy(float); H = feat["adj_high"].to_numpy(float)
L = feat["adj_low"].to_numpy(float); C = feat["adj_close"].to_numpy(float)
ATRP = feat["atr_pct"].to_numpy(float) / 100.0
E21 = feat["ema_21"].to_numpy(float); E50 = feat["ema_50"].to_numpy(float)
HI7 = feat["hi7_prior"].to_numpy(float); RS = feat["rs_rank"].to_numpy(float)
SMA50 = feat["sma_50"].to_numpy(float); SMA200 = feat["sma_200"].to_numpy(float)
RSI14 = feat["rsi_14"].to_numpy(float); RSI2 = feat["rsi_2"].to_numpy(float)
BBL = feat["bb_lower"].to_numpy(float)
C1 = lag(C, 1)
feat["low10_incl"] = pd.Series(L).groupby(IC).transform(lambda s: s.rolling(10, min_periods=5).min()).to_numpy()
LOW10 = feat["low10_incl"].to_numpy(float)
STRUCT = {"bar_low": L, "low10": LOW10}
# liquidity / eligibility at the signal date
feat["liquid"] = ((feat["turnover_median_20d"] >= MIN_TURNOVER) & (feat["bars_available"] >= MIN_BARS)
                  & (~feat["contaminated"].astype(bool)))
BASE = (feat["liquid"] & (feat["adj_close"] >= MIN_PRICE) & np.isfinite(ATRP)).to_numpy()
# market washout: min breadth over the last 20 sessions (date-level)
b50 = feat.groupby("date")["mkt_breadth_50"].first().reindex(CAL)
feat["mkt_washout"] = feat["date"].map(b50.rolling(20, min_periods=5).min()).to_numpy()
# bars since the 52-week low, and the 52-week washout depth
newlow = np.where(L <= feat["low_52w"].to_numpy(float) + 1e-9, np.arange(N, dtype=float), np.nan)
lastlow = pd.Series(newlow).groupby(IC).ffill().to_numpy()
feat["bars_since_52w_low"] = np.clip(np.arange(N) - lastlow, 0, 250)
feat["dd_depth_52w"] = (feat["low_52w"] / feat["high_52w"] - 1) * 100
# market cap at the signal: today's share count x the signal close (splits are in adj_close)
feat["mcap_at_signal"] = feat["shares"] * feat["adj_close"] / 1e7
# week-end flag: the session before an ISO-week change
wk = CAL.isocalendar().week.to_numpy() + 100 * CAL.isocalendar().year.to_numpy()
week_end = np.zeros(len(CAL), bool); week_end[:-1] = wk[1:] != wk[:-1]; week_end[-1] = True
WE = feat["date"].map(pd.Series(week_end, index=CAL)).to_numpy()
mkt_ret = feat.loc[feat["liquid"]].groupby("date")["ret_1d"].mean().reindex(CAL).fillna(0) / 100
BENCH = (1 + mkt_ret).cumprod()
MS = feat.groupby("date")[["mkt_vix", "mkt_vix_pctile_1y", "mkt_breadth_50", "mkt_gate_on", "mkt_nifty_range_20d", "mkt_washout"]].first().reindex(CAL)
print(f"panel {N:,} rows, {feat['isin'].nunique()} names | {CAL[0].date()} to {END.date()} | liquid rows {BASE.sum():,} | {time.time()-t0:.0f}s", flush=True)


# ------------------------------------------------------------------ setups
def raw_setups():
    s = {}
    s["A"] = ((feat["pct_from_52w_high"] <= -30) & (feat["vol_ratio"] >= 3) & (feat["close_pos"] >= 0.7)).to_numpy() & (C > C1)
    down = lag(runlen(feat["ret_1d"].to_numpy() < 0), 1)
    s["B"] = (down >= 5) & (lag(RSI2, 1) < 5) & (C > C1)
    r1 = lag(RSI14, 1)
    s["C"] = (r1 < 30) & (RSI14 >= 30)                       # r1 < 30 already implies "< 30 within the last 5 sessions"
    s["D"] = (C1 < lag(BBL, 1)) & (C >= BBL)
    below200 = lag(runlen(C < SMA200), 1)
    s["E"] = (below200 >= 40) & (C > SMA200)
    s["E2"] = s["E"] & (feat["sma_200_slope"].to_numpy(float) >= 0)
    lo20 = feat["lo_20_prior"].to_numpy(float)
    s["F"] = (L < lo20) & (C > lo20)
    # G: double bottom
    swing = pd.Series(L).groupby(IC).transform(lambda x: x.rolling(11, center=True, min_periods=11).min()).to_numpy()
    is_swing = np.isfinite(swing) & (L <= swing + 1e-9)
    h52 = feat["high_52w"].to_numpy(float)
    deep_prior = is_swing & (L / h52 - 1 <= -0.25)
    second_deep = (LOW10 / h52 - 1) <= -0.25
    hmax_prev = np.full(N, np.nan)                        # rolling max of H over the prior k-1 bars, built incrementally
    g = np.zeros(N, bool)
    Hl = lag(H, 1)
    run_max = Hl.copy()
    for k in range(2, 61):
        run_max = np.fmax(run_max, lag(H, k - 1)) if k > 2 else run_max   # max of H[t-1..t-(k-1)]
        if k >= 15:
            dp = lag(deep_prior.astype(float), k) == 1
            lk = lag(L, k)
            g |= dp & (np.abs(LOW10 / lk - 1) <= 0.03) & second_deep & (C > run_max)
    # dedupe: no G signal in the prior 5 sessions
    gi = g.astype(float)
    recent = np.zeros(N, bool)
    for k in range(1, 6):
        recent |= lag(gi, k) == 1
    s["G"] = g & ~recent
    s["H"] = (feat["dist_sma_200"].to_numpy(float) <= -25) & (RSI14 > lag(RSI14, 3))
    # I: weekly reclaim
    we_idx = np.flatnonzero(WE)
    below_w = (C[we_idx] < SMA50[we_idx])
    fw = pd.Series(below_w.astype(int)); icw = pd.Series(IC[we_idx])
    run_w = fw.groupby([icw, (~pd.Series(below_w)).cumsum()]).cumsum()
    prev_w = run_w.groupby(icw).shift(1).to_numpy()
    sI = np.zeros(N, bool)
    sI[we_idx] = (prev_w >= 8) & (C[we_idx] > SMA50[we_idx])
    s["I"] = sI
    s["J"] = (lag(feat["pct_from_52w_low"].to_numpy(float), 1) <= 5) & (feat["ret_1d"].to_numpy(float) >= 4) & (feat["vol_ratio"].to_numpy(float) >= 2)
    for k in s:
        s[k] = np.nan_to_num(s[k]).astype(bool) & BASE
    return s


RAW = raw_setups()
LDR = np.nan_to_num(RS >= 60).astype(bool)
WASH = np.nan_to_num(feat["mkt_washout"].to_numpy(float) < 30).astype(bool)
SIG, IN_WIN, START, SPLIT, WNAME, QUARTERS = {}, None, None, None, None, None


def set_window(name):
    global SIG, IN_WIN, START, SPLIT, WNAME, QUARTERS
    w = WINDOWS[name]; START, SPLIT, WNAME = w["start"], w["split"], name
    IN_WIN = ((feat["date"] >= START) & (feat["date"] <= END)).to_numpy()
    SIG = {}
    for f in FAMILIES:
        SIG[f] = RAW[f] & IN_WIN
        SIG[f + "_ldr"] = RAW[f] & IN_WIN & LDR
        SIG[f + "_wash"] = RAW[f] & IN_WIN & WASH
    QUARTERS = [str(p) for p in pd.period_range(w["quarters"][0], w["quarters"][1], freq="Q")]


# ------------------------------------------------------------------ engine
def _risk(t, e, stop):
    if stop[0] == "pct":
        risk = e * stop[1] / 100.0
    elif stop[0] == "atr":
        risk = stop[1] * ATRP[t] * e
    else:
        lvl = STRUCT[stop[1]][t]
        if not np.isfinite(lvl):
            return None
        risk = e - lvl
    rpct = risk / e * 100.0
    if stop[0] == "struct":
        if not (3.0 <= rpct <= 20.0):
            return None
    elif not (0.2 < rpct < 20):
        return None
    return risk, rpct


def run(sig, stop, reward, cap):
    """One trade per signal, with a hard initial stop; vectorised per trade.

    stop:   ('atr', m) m x ATR(14)% at the signal bar | ('struct', 'bar_low'|'low10') a level known at the signal close
    reward: ('target', rr) | ('trail', 'ema21'|'ema50') exit next open after a close below
            ('rev7',) exit next open after the first close above the prior 7 closes | ('none',) stop + cap only
    """
    rows = []; busy = {}
    for t in np.flatnonzero(sig):
        if t + 1 > LAST[t] or busy.get(ISIN[t], -1) >= t or not np.isfinite(ATRP[t]):
            continue
        e = O[t + 1]
        if not np.isfinite(e) or e <= 0:
            continue
        rr = _risk(t, e, stop)
        if rr is None:
            continue
        risk, rpct = rr
        sl = e - risk
        end = min(LAST[t], t + cap); n = end - t
        Ls = L[t + 1:end + 1]; Hs = H[t + 1:end + 1]; Cs = C[t + 1:end + 1]
        hit = np.flatnonzero(Ls <= sl); ks = hit[0] if len(hit) else n
        kt = n; kx = n
        if reward[0] == "target":
            tgt = e + reward[1] * risk
            ht = np.flatnonzero(Hs >= tgt); kt = ht[0] if len(ht) else n
        elif reward[0] in ("trail", "rev7"):
            if reward[0] == "trail":
                lvl = (E21 if reward[1] == "ema21" else E50)[t + 1:end + 1]
                cond = np.isfinite(lvl) & (Cs < lvl)
            else:
                lvl = HI7[t + 1:end + 1]
                cond = np.isfinite(lvl) & (Cs > lvl)
            cond &= (np.arange(n) + t + 2) <= LAST[t]
            hx = np.flatnonzero(cond); kx = hx[0] if len(hx) else n
        kmin = min(ks, kt, kx)
        if kmin == n:
            k = end; out = C[end]; why = "time"; incl = n
        elif kmin == ks:
            k = t + 1 + ks; out = min(O[k], sl); why = "stop"; incl = ks + 1
        elif kmin == kt:
            k = t + 1 + kt; out = max(O[k], tgt); why = "target"; incl = kt + 1
        else:
            k = t + 2 + kx; out = O[k]; why = reward[0]; incl = kx + 1
        if not np.isfinite(out):
            continue
        lo = np.nanmin(Ls[:incl]) if incl else np.inf; hi = np.nanmax(Hs[:incl]) if incl else -np.inf
        busy[ISIN[t]] = k
        ei, xi = CALPOS[pd.Timestamp(DATES[t + 1])], CALPOS[pd.Timestamp(DATES[k])]
        rows.append((pd.Timestamp(DATES[t]), ISIN[t], t, ei, xi, e, out, (out / e - 1) * 100 - COST, k - t,
                     rpct, RS[t], why, (min(lo, out) / e - 1) * 100, (max(hi, out) / e - 1) * 100))
    return pd.DataFrame(rows, columns=["date", "isin", "row", "entry_i", "exit_i", "entry", "exit", "net", "bars",
                                       "risk_pct", "rs", "why", "mae", "mfe"])


def run_ref(sig, stop, reward, cap):
    """Verbatim port of swing_lab_pullbacks.run() (bar-by-bar loop), for validation only."""
    rows = []; busy = {}
    for t in np.flatnonzero(sig):
        if t + 1 > LAST[t] or busy.get(ISIN[t], -1) >= t or not np.isfinite(ATRP[t]):
            continue
        e = O[t + 1]
        if not np.isfinite(e) or e <= 0:
            continue
        rr = _risk(t, e, stop)
        if rr is None:
            continue
        risk, rpct = rr
        sl = e - risk
        tgt = e + reward[1] * risk if reward[0] == "target" else np.inf
        end = min(LAST[t], t + cap)
        k = t + 1; lo = np.inf; hi = -np.inf
        out = None; why = "time"
        while k <= end:
            lo = min(lo, L[k]); hi = max(hi, H[k])
            if L[k] <= sl:
                out = min(O[k], sl); why = "stop"; break
            if reward[0] == "target" and H[k] >= tgt:
                out = max(O[k], tgt); why = "target"; break
            if reward[0] == "trail":
                lvl = E21[k] if reward[1] == "ema21" else E50[k]
                if np.isfinite(lvl) and C[k] < lvl and k + 1 <= LAST[t]:
                    out = O[k + 1]; why = "trail"; k += 1; break
            elif reward[0] == "rev7":
                if np.isfinite(HI7[k]) and C[k] > HI7[k] and k + 1 <= LAST[t]:
                    out = O[k + 1]; why = "rev7"; k += 1; break
            k += 1
        if out is None:
            k = min(k, end); out = C[k]
        if not np.isfinite(out):
            continue
        busy[ISIN[t]] = k
        ei, xi = CALPOS[pd.Timestamp(DATES[t + 1])], CALPOS[pd.Timestamp(DATES[k])]
        rows.append((pd.Timestamp(DATES[t]), ISIN[t], t, ei, xi, e, out, (out / e - 1) * 100 - COST, k - t,
                     rpct, RS[t], why, (min(lo, out) / e - 1) * 100, (max(hi, out) / e - 1) * 100))
    return pd.DataFrame(rows, columns=["date", "isin", "row", "entry_i", "exit_i", "entry", "exit", "net", "bars",
                                       "risk_pct", "rs", "why", "mae", "mfe"])


# ------------------------------------------------------------------ stats, labels, cells
def stats(t):
    if len(t) < MIN_N:
        return None
    n = t["net"]; w = n[n > 0]; l = n[n <= 0]
    if not len(w) or not len(l):
        return None
    why = t["why"].value_counts(normalize=True) * 100
    return {"n": int(len(t)), "win": float((n > 0).mean() * 100), "net": float(n.mean()), "median": float(n.median()),
            "avg_win": float(w.mean()), "avg_loss": float(l.mean()), "payoff": float(w.mean() / -l.mean()),
            "pf": float(w.sum() / -l.sum()), "stopped": float(why.get("stop", 0)), "bars": float(t["bars"].mean()),
            "exp_r": float((n / t["risk_pct"]).mean()), "risk": float(t["risk_pct"].mean()),
            "mae_med": float(t["mae"].median()), "mfe_med": float(t["mfe"].median()),
            "days": int(t["date"].nunique()), "per_month": float(len(t) / max((t["date"].max() - t["date"].min()).days / 30.44, 1))}


def label(stop, reward, cap):
    s = f"{stop[1]:g} ATR" if stop[0] == "atr" else {"bar_low": "bar low", "low10": "10d low"}[stop[1]]
    r = (f"target 1:{reward[1]:g}" if reward[0] == "target" else
         {"ema21": "close<21EMA", "ema50": "close<50EMA"}[reward[1]] if reward[0] == "trail"
         else "7d-high reversal" if reward[0] == "rev7" else "stop only")
    return s, f"{r} cap{cap}"


def family_of(setup):
    return setup.replace("_ldr", "").replace("_wash", "")


def cell(job):
    setup, stop, reward, cap = job
    tr = run(SIG[setup], stop, reward, cap)
    if not len(tr):
        return None
    a, s1, s2 = stats(tr), stats(tr[tr["date"] < SPLIT]), stats(tr[tr["date"] >= SPLIT])
    if not (a and s1 and s2):
        return None
    sl, rl = label(stop, reward, cap)
    return {"setup": setup, "family": family_of(setup), "stop": sl, "exit": rl, "stop_spec": list(stop),
            "exit_spec": list(reward), "cap": cap, "all": a, "h1": s1, "h2": s2,
            "both": bool(s1["net"] > 0 and s2["net"] > 0),
            "worse_net": float(min(s1["net"], s2["net"])), "worse_win": float(min(s1["win"], s2["win"]))}


ATTACH_COLS = STOCK_FEATS + ["cap_band", "mcap_at_signal", "rsi_14", "close_pos"] + MARKET_NUM + MARKET_BIN


def attach(tr):
    """Signal-bar stock features and same-day market state, for grooming."""
    if not len(tr):
        return tr
    f = feat.iloc[tr["row"].to_numpy()][ATTACH_COLS].reset_index(drop=True)
    tr = pd.concat([tr.reset_index(drop=True), f], axis=1)
    tr["half"] = np.where(tr["date"] < SPLIT, "H1", "H2")
    tr["win"] = tr["net"] > 0
    return tr


# ------------------------------------------------------------------ grooming (swing_lab_pullbacks conventions)
def cluster_t(y, x, grp):
    y = np.asarray(y, float); x = np.asarray(x, float)
    if len(y) < 10 or x.std() == 0:
        return np.nan
    X = np.column_stack([np.ones(len(x)), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    S = pd.DataFrame({"g": np.asarray(grp), "e0": e, "e1": e * x}).groupby("g")[["e0", "e1"]].sum().to_numpy()
    V = XtX_inv @ (S.T @ S) @ XtX_inv
    return float(beta[1] / np.sqrt(V[1, 1])) if V[1, 1] > 0 else np.nan


def r(x, k=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), k)


def eval_filter(sub, mask):
    res = {}
    for h in ("H1", "H2"):
        s = sub[sub["half"] == h]; m = mask.loc[s.index].fillna(False).to_numpy(bool)
        kept, rem = s[m], s[~m]
        months = max((s["date"].max() - s["date"].min()).days / 30.44, 1) if len(s) else 1
        n = kept["net"]; w = n[n > 0]; l = n[n <= 0]
        res[h] = {"n_all": int(len(s)), "n_kept": int(len(kept)), "kept_pct": r(len(kept) / max(len(s), 1)),
                  "kept_per_month": r(len(kept) / months, 1),
                  "wr_kept": r(kept["win"].mean()) if len(kept) else None,
                  "mean_kept": r(kept["net"].mean()) if len(kept) else None,
                  "pf_kept": r(w.sum() / -l.sum(), 2) if len(l) and l.sum() < 0 else None,
                  "stopped_kept": r((kept["why"] == "stop").mean()) if len(kept) else None,
                  "wr_removed": r(rem["win"].mean()) if len(rem) else None,
                  "mean_removed": r(rem["net"].mean()) if len(rem) else None,
                  "t_cluster": r(cluster_t(s["net"], m.astype(float), s["date"]), 2) if 0 < m.sum() < len(s) else None}
    return res


def passes(res, base):
    ts = []
    for h in ("H1", "H2"):
        d = res[h]
        if d["wr_kept"] is None or d["mean_kept"] is None or d["t_cluster"] is None:
            return False
        if not (d["wr_kept"] > base[h]["win"] and d["mean_kept"] > base[h]["net"]):
            return False
        if d["kept_per_month"] < MIN_KEPT_PER_MONTH:
            return False
        ts.append(d["t_cluster"])
    return bool(max(ts) >= T_MIN and min(ts) > 0)


def score(res):
    return (min(res["H1"]["wr_kept"], res["H2"]["wr_kept"]), min(res["H1"]["mean_kept"], res["H2"]["mean_kept"]),
            min(res["H1"]["kept_pct"], res["H2"]["kept_pct"]))


def mask_of(sub, f):
    if f["op"] == ">=":
        return sub[f["feature"]] >= f["thr"]
    if f["op"] == "<=":
        return sub[f["feature"]] <= f["thr"]
    if f["op"] == "in":
        return sub[f["feature"]].isin(f["thr"])
    return sub[f["feature"]] == f["thr"]


def months_active(tr, mask):
    mo_all = tr["date"].dt.to_period("M").nunique()
    mo_kept = tr.loc[mask, "date"].dt.to_period("M").value_counts()
    return float((mo_kept >= 5).sum() / max(mo_all, 1)), int(mo_all)


def parts_of(f):
    return f["parts"] if "parts" in f else [f]


def groom(sub):
    base = {h: {"win": float(sub.loc[sub["half"] == h, "win"].mean()), "net": float(sub.loc[sub["half"] == h, "net"].mean())}
            for h in ("H1", "H2")}
    h1 = sub[sub["half"] == "H1"]
    singles = []; n_tests = 0
    for f in STOCK_FEATS + MARKET_NUM:
        x = h1[f].dropna()
        if len(x) < 100:
            continue
        edges = np.unique(np.nanpercentile(x, [20, 40, 60, 80]))
        for thr in edges:
            for op in (">=", "<="):
                spec = {"feature": f, "op": op, "thr": float(thr), "level": "market" if f in MARKET_NUM else "stock"}
                m = mask_of(sub, spec); n_tests += 1
                res = eval_filter(sub, m)
                if res["H1"]["n_kept"] < 30 or res["H2"]["n_kept"] < 30:
                    continue
                spec.update({"filter": f"{f} {op} {thr:.4g}", "passes": passes(res, base), "H1": res["H1"], "H2": res["H2"]})
                singles.append(spec)
    bins = [{"feature": f, "op": "==", "thr": v, "filter": f"{f} == {v}", "level": "market"} for f in MARKET_BIN for v in (1, 0)]
    bins += [{"feature": "cap_band", "op": "in", "thr": v, "filter": f"cap_band in {v}", "level": "stock"}
             for v in (["Mid"], ["Large", "Mega"], ["Mega"])]
    for spec in bins:
        m = mask_of(sub, spec); n_tests += 1
        res = eval_filter(sub, m)
        if res["H1"]["n_kept"] < 30 or res["H2"]["n_kept"] < 30:
            continue
        spec.update({"passes": passes(res, base), "H1": res["H1"], "H2": res["H2"]})
        singles.append(spec)
    passing = sorted([f for f in singles if f["passes"]], key=lambda f: score(f), reverse=True)
    best_per_feat = {}
    for f in passing:
        best_per_feat.setdefault(f["feature"], f)
    top = list(best_per_feat.values())[:8]
    combos = []
    for a, bb in itertools.combinations(top, 2):
        m = mask_of(sub, a) & mask_of(sub, bb); n_tests += 1
        res = eval_filter(sub, m)
        if res["H1"]["n_kept"] < 30 or res["H2"]["n_kept"] < 30:
            continue
        combos.append({"filter": f"{a['filter']} AND {bb['filter']}", "parts": [a, bb], "passes": passes(res, base),
                       "H1": res["H1"], "H2": res["H2"], "level": "market" if "market" in (a["level"], bb["level"]) else "stock"})
    board = sorted([f for f in singles + combos if f["passes"]], key=lambda f: score(f), reverse=True)
    for f in board[:25]:                                   # regime concentration of each surviving filter
        mk = pd.Series(True, index=sub.index)
        for q in parts_of(f):
            mk &= mask_of(sub, q).fillna(False)
        share, n_mo = months_active(sub, mk)
        f["months_active_share"] = r(share); f["n_months"] = n_mo
        f["regime_flag"] = bool(f["level"] == "market" and share < 0.5)
    slim = lambda f: {k: v for k, v in f.items() if k != "parts"} | ({"parts": [{kk: vv for kk, vv in p.items() if kk in ("feature", "op", "thr", "filter", "level")} for p in f["parts"]]} if "parts" in f else {})
    return {"base": base, "n_tests": n_tests, "n_single_passing": len(passing), "n_combo_passing": sum(f["passes"] for f in combos),
            "leaderboard": [slim(f) for f in board[:25]]}


def groom_job(job):
    setup, stop, reward, cap = job
    tr = attach(run(SIG[setup], stop, reward, cap))
    out = groom(tr)
    out["cell"] = {"setup": setup, "stop_spec": list(stop), "exit_spec": list(reward), "cap": cap}
    return out


# ------------------------------------------------------------------ portfolio
from screener.robustness import stress
from screener.portfolio import simulate
_CACHE = {}


def trades_for(spec):
    stop = tuple(spec["stop_spec"]); reward = tuple(spec["exit_spec"]); cap = int(spec["cap"])
    key = (WNAME, spec["setup"], stop, reward, cap)
    if key not in _CACHE:
        _CACHE[key] = attach(run(SIG[spec["setup"]], stop, reward, cap))
    return _CACHE[key]


def make_trades_from(spec):
    """spec: setup, stop_spec, exit_spec, cap, filters [{feature, op, thr}], gate, mcap_floor."""
    def mk(p):
        stop = ("atr", p["atr_mult"]) if spec["stop_spec"][0] == "atr" else tuple(spec["stop_spec"])
        reward = ("target", p["rr"]) if spec["exit_spec"][0] == "target" else tuple(spec["exit_spec"])
        tr = trades_for({**spec, "stop_spec": list(stop), "exit_spec": list(reward), "cap": int(p["cap"])})
        m = pd.Series(True, index=tr.index)
        for i, f in enumerate(spec["filters"]):
            thr = p.get(f"thr{i}", f["thr"])
            m &= mask_of(tr, {**f, "thr": thr}).fillna(False)
        if spec.get("gate"):
            m &= tr["mkt_gate_on"] == 1
        if spec.get("mcap_floor"):
            m &= tr["mcap_at_signal"] >= spec["mcap_floor"]
        return tr[m][["date", "isin", "entry_i", "exit_i", "net", "rs", "why", "bars", "mae", "mfe", "mcap_at_signal", "risk_pct"]].reset_index(drop=True)
    return mk


def params_of(spec):
    p = {"cap": int(spec["cap"])}
    if spec["stop_spec"][0] == "atr":
        p["atr_mult"] = float(spec["stop_spec"][1])
    if spec["exit_spec"][0] == "target":
        p["rr"] = float(spec["exit_spec"][1])
    for i, f in enumerate(spec["filters"]):
        if f["op"] in (">=", "<=") and isinstance(f["thr"], float) and f["thr"] != 0:
            p[f"thr{i}"] = float(f["thr"])
    return p


def trade_summary(sub):
    n = sub["net"]
    return {"n": int(len(sub)), "win": r(float((n > 0).mean() * 100), 1) if len(sub) else None,
            "net": r(float(n.mean()), 2) if len(sub) else None,
            "per_month": r(len(sub) / max((sub["date"].max() - sub["date"].min()).days / 30.44, 1), 1) if len(sub) else None,
            "stopped": r(float((sub["why"] == "stop").mean() * 100), 0) if len(sub) else None}


def mae_report(tr):
    """How far winners dip before working, and what a tighter stop would have cost."""
    w = tr[tr["net"] > 0]; l = tr[tr["net"] <= 0]
    if not len(w):
        return None
    q = lambda s, p: r(float(np.percentile(s, p)), 1)
    return {"winners_n": int(len(w)), "winners_mae_p25": q(w["mae"], 25), "winners_mae_med": q(w["mae"], 50),
            "winners_mae_p75": q(w["mae"], 75), "winners_mae_p90": q(w["mae"], 90),
            "losers_mae_med": q(l["mae"], 50) if len(l) else None, "losers_mfe_med": q(l["mfe"], 50) if len(l) else None,
            "winners_mfe_med": q(w["mfe"], 50), "risk_med": q(tr["risk_pct"], 50),
            "winners_dipping_beyond": {f"{x}%": r(float((w["mae"] <= -x).mean() * 100), 1) for x in (3, 5, 8, 10, 15)}}


def survivorship(tr):
    """Signals in names that were small at the time, and the result with a Rs 1,000 Cr floor at the signal."""
    m = tr["mcap_at_signal"]
    out = {"share_mcap_known": r(float(m.notna().mean())), "share_below_500cr": r(float((m < 500).mean())),
           "share_below_1000cr": r(float((m < 1000).mean()))}
    big = tr[m >= 1000]
    for h, sub in (("all", big), ("H1", big[big["date"] < SPLIT]), ("H2", big[big["date"] >= SPLIT])):
        out[f"floor1000_{h}"] = trade_summary(sub)
    small = tr[m < 1000]
    out["below1000_all"] = trade_summary(small)
    return out


def stress_job(spec):
    p = params_of(spec)
    mk = make_trades_from(spec)
    res = stress(mk, p, sessions=CAL, split=SPLIT, slots=SLOTS, order_col="rs", n_random=100)
    tr = mk(p)
    for h, sub in (("all", tr), ("H1", tr[tr["date"] < SPLIT]), ("H2", tr[tr["date"] >= SPLIT])):
        res[f"trades_{h}"] = trade_summary(sub)
    res["label"] = spec["label"]; res["setup"] = spec["setup"]; res["family"] = family_of(spec["setup"]); res["kind"] = spec.get("kind")
    res["spec"] = {k: v for k, v in spec.items() if k != "label"}
    res["mae"] = mae_report(tr)
    res["survivorship"] = survivorship(tr)
    res["quarterly"] = quarter_table(tr)
    return res


# ------------------------------------------------------------------ quarterly
def quarter_table(tr):
    """Per calendar quarter of the SIGNAL date: trades, win %, mean net, and the
    10-slot account's return over that quarter (rs-ordered simulation, marked at
    cost, profit on the exit session)."""
    t = tr.sort_values(["entry_i", "rs"], ascending=[True, False])
    eq = simulate(t, slots=SLOTS, sessions=CAL)["equity"] if len(t) >= 1 else pd.Series(dtype=float)
    qe = eq.groupby(eq.index.to_period("Q").astype(str)).last() if len(eq) else pd.Series(dtype=float)
    rows = []; prev = 100.0
    for q in QUARTERS:
        sub = tr[tr["date"].dt.to_period("Q").astype(str) == q]
        n = sub["net"]; pr = None
        if q in qe.index:
            pr = float((qe[q] / prev - 1) * 100); prev = float(qe[q])
        rows.append({"q": q, "n": int(len(sub)), "win": r(float((n > 0).mean() * 100), 1) if len(sub) else None,
                     "net": r(float(n.mean()), 2) if len(sub) else None, "port_ret": r(pr, 2),
                     "stopped": r(float((sub["why"] == "stop").mean() * 100), 0) if len(sub) else None})
    test = [x for x in rows if x["port_ret"] is not None]
    pos = sum(x["port_ret"] > 0 for x in test)
    worst = min((x["port_ret"] for x in test), default=None)
    lost = [x["q"] for x in test if x["port_ret"] <= 0]
    return {"table": rows, "quarters_tested": len(test), "quarters_positive": pos, "share_positive": r(pos / max(len(test), 1)),
            "worst_quarter": r(worst, 2), "worst_q": min(test, key=lambda x: x["port_ret"])["q"] if test else None,
            "lost_quarters": lost,
            "quarter_pass": bool(len(test) and pos / len(test) >= Q_MIN_SHARE and worst is not None and worst > Q_WORST)}


def market_context():
    out = []
    for q in QUARTERS:
        p = pd.Period(q, freq="Q"); lo, hi = p.start_time, p.end_time
        bh = BENCH.loc[lo:hi]; m_ = MS.loc[lo:hi]
        prev = BENCH.loc[:lo - pd.Timedelta(days=1)]
        base = float(prev.iloc[-1]) if len(prev) else float(bh.iloc[0])
        out.append({"q": q, "universe_ret": r((float(bh.iloc[-1]) / base - 1) * 100, 1) if len(bh) else None,
                    "universe_maxdd": r(float((bh / bh.cummax() - 1).min() * 100), 1) if len(bh) else None,
                    "vix_mean": r(float(m_["mkt_vix"].mean()), 1), "breadth50_mean": r(float(m_["mkt_breadth_50"].mean()), 0),
                    "breadth50_min": r(float(m_["mkt_breadth_50"].min()), 0), "gate_on_share": r(float(m_["mkt_gate_on"].mean()), 2)})
    return out


# ------------------------------------------------------------------ printing
def show(rows, title):
    print(f"\n{title}")
    print(f"{'setup':28s}{'stop':>8s} {'exit':24s}{'n':>6s}{'win':>5s}{'net':>6s}{'aW':>6s}{'aL':>6s}{'pay':>5s}{'PF':>5s}{'stp%':>5s}{'bars':>5s}"
          f" | {'H1 n':>5s}{'win':>4s}{'net':>6s} | {'H2 n':>5s}{'win':>4s}{'net':>6s}")
    for x in rows:
        a, h1, h2 = x["all"], x["h1"], x["h2"]
        print(f"{x['setup']:28s}{x['stop']:>8s} {x['exit']:24s}{a['n']:>6,d}{a['win']:>5.0f}{a['net']:>+6.2f}{a['avg_win']:>6.2f}"
              f"{a['avg_loss']:>6.2f}{a['payoff']:>5.2f}{a['pf']:>5.2f}{a['stopped']:>5.0f}{a['bars']:>5.0f}"
              f" | {h1['n']:>5d}{h1['win']:>4.0f}{h1['net']:>+6.2f} | {h2['n']:>5d}{h2['win']:>4.0f}{h2['net']:>+6.2f}"
              f"  {'BOTH' if x['both'] else ''}")


def sanity(setup, stop, reward, cap, k=5, since=None):
    tr = run(SIG[setup], stop, reward, cap)
    if since is not None:
        tr = tr[tr["date"] >= since]
    tr = tr.drop_duplicates("isin")
    sl, rl = label(stop, reward, cap)
    print(f"\nSANITY: {setup} ({TITLE[family_of(setup)]}), {sl} stop, {rl} - {k} trades")
    for x in tr.head(k).itertuples():
        t = x.row
        print(f"  {str(SYM[t]):12s} sig {x.date.date()} close {C[t]:.2f} low {L[t]:.2f} (52wH {feat['high_52w'].iat[t]:.2f}, "
              f"{feat['pct_from_52w_high'].iat[t]:+.0f}%, vol x{feat['vol_ratio'].iat[t]:.1f}, pos {feat['close_pos'].iat[t]:.2f}, rsi14 {RSI14[t]:.0f})")
        print(f"      entry {x.entry:.2f} (open of {pd.Timestamp(DATES[t+1]).date()}) stop {x.entry*(1-x.risk_pct/100):.2f} ({x.risk_pct:.1f}%) "
              f"exit {x.exit:.2f} on {pd.Timestamp(DATES[t+x.bars]).date()} why={x.why} bars={x.bars} net={x.net:+.2f} mae={x.mae:.1f} mfe={x.mfe:.1f}")
        if x.why in ("trail", "rev7"):
            kk = t + x.bars - 1
            lvl = HI7[kk] if x.why == "rev7" else (E21[kk] if reward[1] == "ema21" else E50[kk])
            print(f"      trigger bar {pd.Timestamp(DATES[kk]).date()}: close {C[kk]:.2f} vs level {lvl:.2f}; next open {O[kk+1]:.2f}")
        elif x.why == "stop":
            kk = t + x.bars
            print(f"      stop bar {pd.Timestamp(DATES[kk]).date()}: open {O[kk]:.2f} low {L[kk]:.2f} <= stop; fill {x.exit:.2f}")
    return tr


def validate_engine():
    """The vectorised engine must reproduce the bar-by-bar reference loop exactly."""
    checks = [("A", ("atr", 3.0), ("rev7",), 20), ("C", ("atr", 4.0), ("trail", "ema21"), 60), ("D", ("struct", "bar_low"), ("target", 2.0), 60),
              ("F", ("atr", 3.0), ("none",), 20), ("G", ("struct", "low10"), ("trail", "ema50"), 60), ("B", ("atr", 3.0), ("target", 3.0), 20)]
    for setup, stop, reward, cap in checks:
        a = run(SIG[setup], stop, reward, cap); b = run_ref(SIG[setup], stop, reward, cap)
        same = len(a) == len(b) and a[["row", "exit_i", "why"]].equals(b[["row", "exit_i", "why"]]) and \
            np.allclose(a[["entry", "exit", "net", "mae", "mfe", "risk_pct"]].to_numpy(float), b[["entry", "exit", "net", "mae", "mfe", "risk_pct"]].to_numpy(float), equal_nan=True)
        print(f"  engine check {setup:3s} {stop} {reward} cap{cap}: {len(a)} vs {len(b)} trades -> {'IDENTICAL' if same else 'MISMATCH'}", flush=True)
        if not same:
            raise SystemExit("engine mismatch")


def signal_counts():
    print(f"\nSIGNALS ({WNAME}: {START.date()}..{END.date()})")
    for f in FAMILIES:
        print(f"  {f:3s} {TITLE[f]:30s} plain {SIG[f].sum():>7,d} ({feat.loc[SIG[f], 'date'].nunique():>4d} days)"
              f"  leader {SIG[f+'_ldr'].sum():>6,d}  washout {SIG[f+'_wash'].sum():>6,d}")
    return {n: {"signals": int(s.sum()), "days": int(feat.loc[s, "date"].nunique())} for n, s in SIG.items()}


# ------------------------------------------------------------------ one window
def run_window(name):
    set_window(name)
    tw = time.time()
    print(f"\n{'='*100}\nWINDOW {name}: signals {START.date()}..{END.date()}, split {SPLIT.date()}\n{'='*100}", flush=True)
    counts = signal_counts()
    # A. matrix
    jobs = [(n, s, e, c) for n in SIG for s in ATR_STOPS + [("struct", STRUCT_OF[family_of(n)])] for e in EXITS for c in CAPS]
    print(f"\n{len(jobs)} cells ...", flush=True)
    with Pool(ARGS.procs) as pool:
        MATRIX = [x for x in pool.map(cell, jobs, chunksize=4) if x]
    print(f"{len(MATRIX)} cells with >= {MIN_N} trades per half | {time.time()-tw:.0f}s", flush=True)
    BEST = {}
    for fam in FAMILIES:
        for tw_ in TWISTS:
            n = fam + tw_
            rows = sorted([x for x in MATRIX if x["setup"] == n], key=lambda x: (-x["worse_net"], -x["worse_win"]))
            if not rows:
                continue
            show(rows[:4], f"A. {n} {TITLE[fam]}{DEFS.get(tw_, '')} - top 4 cells by worse-half mean net (of {len(rows)})")
            picks = rows[:2]
            by_win = sorted(rows, key=lambda x: (-x["worse_win"], -x["worse_net"]))
            if by_win[0] not in picks:
                picks = picks + [by_win[0]]
            BEST[n] = picks
    # B. grooming on the best cells
    gjobs = [(n, tuple(x["stop_spec"]), tuple(x["exit_spec"]), x["cap"]) for n in BEST for x in BEST[n]]
    print(f"\ngrooming {len(gjobs)} cells ...", flush=True)
    with Pool(ARGS.procs) as pool:
        GROOM = pool.map(groom_job, gjobs, chunksize=1)
    print(f"grooming done | {time.time()-tw:.0f}s", flush=True)
    for gr in GROOM:
        c = gr["cell"]; b = gr["base"]
        sl, rl = label(tuple(c["stop_spec"]), tuple(c["exit_spec"]), c["cap"])
        print(f"\nB. {c['setup']} | {sl} | {rl}  base H1 win {b['H1']['win']*100:.0f}% net {b['H1']['net']:+.2f} | "
              f"H2 win {b['H2']['win']*100:.0f}% net {b['H2']['net']:+.2f}  tests={gr['n_tests']} singles pass={gr['n_single_passing']} combos pass={gr['n_combo_passing']}")
        for f in gr["leaderboard"][:6]:
            print(f"   {f['filter'][:58]:58s} H1 {f['H1']['n_kept']:>5d} {f['H1']['wr_kept']*100:>3.0f}% {f['H1']['mean_kept']:>+5.2f} t{f['H1']['t_cluster']:>+5.1f} "
                  f"{f['H1']['kept_per_month']:>4.0f}/mo | H2 {f['H2']['n_kept']:>5d} {f['H2']['wr_kept']*100:>3.0f}% {f['H2']['mean_kept']:>+5.2f} t{f['H2']['t_cluster']:>+5.1f} {f['H2']['kept_per_month']:>4.0f}/mo"
                  f"  active {f['months_active_share']*100:.0f}% of months{'  REGIME' if f['regime_flag'] else ''}")
    # C. portfolio stress: per family, best plain/ldr/wash cell as baseline; best groomed; groomed + gate; baseline with mcap floor
    specs = []
    for fam in FAMILIES:
        variants = [n for n in BEST if family_of(n) == fam]
        if not variants:
            continue
        for n in variants:
            best_cell = BEST[n][0]
            specs.append({"label": f"{n} | baseline", "setup": n, "stop_spec": best_cell["stop_spec"], "exit_spec": best_cell["exit_spec"],
                          "cap": best_cell["cap"], "filters": [], "gate": False, "kind": "baseline"})
            specs.append({"label": f"{n} | baseline, mcap>=1000Cr at signal", "setup": n, "stop_spec": best_cell["stop_spec"], "exit_spec": best_cell["exit_spec"],
                          "cap": best_cell["cap"], "filters": [], "gate": False, "kind": "baseline_mcap", "mcap_floor": 1000})
            cands = [gr for gr in GROOM if gr["cell"]["setup"] == n and gr["leaderboard"]]
            if cands:
                gr = max(cands, key=lambda gr: score(gr["leaderboard"][0]))
                f = gr["leaderboard"][0]
                filt = [{"feature": p["feature"], "op": p["op"], "thr": p["thr"]} for p in parts_of(f)]
                c = gr["cell"]
                specs.append({"label": f"{n} | groomed: {f['filter']}", "setup": n, "stop_spec": c["stop_spec"], "exit_spec": c["exit_spec"],
                              "cap": c["cap"], "filters": filt, "gate": False, "kind": "groomed", "regime_flag": f["regime_flag"],
                              "months_active_share": f["months_active_share"]})
                specs.append({"label": f"{n} | groomed + gate", "setup": n, "stop_spec": c["stop_spec"], "exit_spec": c["exit_spec"],
                              "cap": c["cap"], "filters": filt, "gate": True, "kind": "groomed_gate"})
            else:
                specs.append({"label": f"{n} | baseline + gate", "setup": n, "stop_spec": best_cell["stop_spec"], "exit_spec": best_cell["exit_spec"],
                              "cap": best_cell["cap"], "filters": [], "gate": True, "kind": "baseline_gate"})
    print(f"\nstressing {len(specs)} portfolios ...", flush=True)
    with Pool(ARGS.procs) as pool:
        PORT = pool.map(stress_job, specs, chunksize=1)
    print(f"stress done | {time.time()-tw:.0f}s", flush=True)
    print(f"\nC. PORTFOLIO ({name}), 10 slots, rs order, 100 random orderings, +/-20% neighbours")
    print(f"{'strategy':70s}{'n':>6s}{'win':>5s}{'net':>6s}{'/mo':>5s} | {'CAGR':>6s}{'maxDD':>6s} | {'H1':>6s}{'H2':>6s} | {'rnd p5':>7s}{'nbr min':>8s} robust | q+ worst")
    for p in PORT:
        ta = p["trades_all"]; qt = p["quarterly"]
        print(f"{p['label'][:70]:70s}{ta['n']:>6d}{ta['win'] or 0:>5.0f}{ta['net'] or 0:>+6.2f}{ta['per_month'] or 0:>5.0f} | "
              f"{p['full']['cagr']:>+6.1f}{p['full']['maxdd']:>6.1f} | {p['h1']['cagr']:>+6.1f}{p['h2']['cagr']:>+6.1f} | "
              f"{p['random_order']['p5']:>+7.1f}{p['neighbour_min']:>+8.1f} {'YES' if p['robust'] else 'no ':3s}  | {qt['quarters_positive']:>2d}/{qt['quarters_tested']:<2d} {qt['worst_quarter'] if qt['worst_quarter'] is not None else float('nan'):>+6.1f}")
    ctx = market_context()
    print(f"\nMARKET BY QUARTER ({name})")
    print(f"{'q':8s}{'univ ret':>9s}{'univ DD':>8s}{'VIX':>6s}{'b50':>5s}{'b50min':>7s}{'gate':>6s}")
    for c_ in ctx:
        print(f"{c_['q']:8s}{c_['universe_ret']:>+9.1f}{c_['universe_maxdd']:>8.1f}{c_['vix_mean']:>6.1f}{c_['breadth50_mean']:>5.0f}{c_['breadth50_min']:>7.0f}{c_['gate_on_share']:>6.2f}")
    # D. verdicts per family: best variant by the bar
    RANK = {"TRADE": 2, "MAYBE": 1, "DROP": 0}
    VERD = {}
    for p in PORT:
        ta, qt = p["trades_all"], p["quarterly"]
        win = ta["win"] or 0
        halves = (p["trades_H1"]["net"] or -1) > 0 and (p["trades_H2"]["net"] or -1) > 0
        small_dd = p["full"]["maxdd"] >= -15
        p["verdict"] = ("TRADE" if (win >= 65 and halves and p["robust"] and qt["quarter_pass"] and small_dd)
                        else "MAYBE" if (win >= 60 and halves) else "DROP")
    for fam in FAMILIES:
        vs = [p for p in PORT if p["family"] == fam]
        if not vs:
            continue
        best = max(vs, key=lambda p: (RANK[p["verdict"]], p["trades_all"]["win"] or 0, p["quarterly"]["worst_quarter"] or -99))
        VERD[fam] = {"verdict": best["verdict"], "variant": best["label"], "kind": best["kind"], "win_all": best["trades_all"]["win"],
                     "net_all": best["trades_all"]["net"], "n": best["trades_all"]["n"], "per_month": best["trades_all"]["per_month"],
                     "h1": best["trades_H1"], "h2": best["trades_H2"], "cagr": r(best["full"]["cagr"], 1), "maxdd": r(best["full"]["maxdd"], 1),
                     "robust": best["robust"], "quarters_positive": f"{best['quarterly']['quarters_positive']}/{best['quarterly']['quarters_tested']}",
                     "worst_quarter": best["quarterly"]["worst_quarter"], "worst_q": best["quarterly"]["worst_q"],
                     "lost_quarters": best["quarterly"]["lost_quarters"], "regime_flag": best["spec"].get("regime_flag"),
                     "spec": best["spec"], "survivorship": best["survivorship"], "mae": best["mae"]}
    print(f"\nD. VERDICTS ({name}) - TRADE = win>=65%, both halves>0, stress robust, >= {Q_MIN_SHARE*100:.0f}% quarters>0, no quarter < {Q_WORST}%, maxDD >= -15%")
    for fam, v in VERD.items():
        print(f"  {fam:3s} {TITLE[fam]:30s} {v['verdict']:6s} win {v['win_all']:.0f}% net {v['net_all']:+.2f} n {v['n']} CAGR {v['cagr']:+.1f} DD {v['maxdd']:.1f} "
              f"q {v['quarters_positive']} worst {v['worst_quarter']:+.1f} ({v['worst_q']})  <- {v['variant'][:70]}")
    for p in PORT:
        p.pop("keys", None)
    print(f"\nwindow {name} done in {time.time()-tw:.0f}s", flush=True)
    return {"window": {"start": str(START.date()), "split": str(SPLIT.date()), "end": str(END.date()), "quarters": QUARTERS},
            "signals": counts, "matrix": MATRIX, "best_cells": BEST, "grooming": GROOM, "portfolio": PORT,
            "market_context": ctx, "verdicts": VERD}


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    set_window("ext")
    print("\nENGINE VALIDATION (vectorised run vs verbatim bar-by-bar loop)")
    validate_engine()
    sanity("A", ("atr", 3.0), ("rev7",), 20, since=pd.Timestamp("2020-03-01"))
    sanity("G", ("struct", "low10"), ("trail", "ema21"), 60, since=pd.Timestamp("2022-01-01"), k=3)
    sanity("E", ("atr", 4.0), ("trail", "ema50"), 60, since=pd.Timestamp("2023-01-01"), k=2)
    if ARGS.check:
        sys.exit(0)
    OUT = {"note": "Bottom lab. BOTH windows carry survivorship bias: the universe is today's >= Rs 1,000 Cr names, so "
                   "stocks that bottomed and never recovered (or delisted) are absent, which flatters every bottom-buying result. "
                   "See portfolio[*].survivorship and the baseline_mcap portfolios for the bound.",
           "definitions": DEFS, "titles": TITLE, "cost": COST, "slots": SLOTS, "caps": CAPS, "min_n_per_half": MIN_N,
           "min_kept_per_month": MIN_KEPT_PER_MONTH, "t_min": T_MIN, "quarter_rule": {"min_share_positive": Q_MIN_SHARE, "worst_floor": Q_WORST},
           "exits": [list(e) for e in EXITS], "atr_stops": [list(s) for s in ATR_STOPS], "struct_stop": STRUCT_OF,
           "groom_features": {"stock": STOCK_FEATS + ["cap_band"], "market": MARKET_NUM + MARKET_BIN},
           "panel": {"rows": int(N), "names": int(feat["isin"].nunique()), "from": str(CAL[0].date()), "to": str(END.date())}}
    try:
        J = json.load(open("data/screen/swing_lab_pullbacks.json"))
        OUT["leader_dip_benchmark_recent"] = {"portfolio": [p for p in J["portfolio"] if p["setup"] == "leader_dip_final"][0],
                                             "quarterly_verdict": J.get("quarterly", {}).get("verdicts", {}).get("leader_dip_final")}
    except Exception as ex:
        OUT["leader_dip_benchmark_recent"] = str(ex)
    for w in (["ext", "recent"] if ARGS.window == "both" else [ARGS.window]):
        OUT[w] = run_window(w)
        Path("data/screen/bottom_lab.json").write_text(json.dumps(OUT, default=jdump))
        print(f"wrote data/screen/bottom_lab.json | {time.time()-t0:.0f}s", flush=True)
