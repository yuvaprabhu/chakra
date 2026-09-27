"""What a fixed-slot account would have made trading ranked Double Seven signals.

Rules: N equal slots. Each day, with free slots, take the highest-scoring
signals (the composite from double_seven_selection.py), one position per
name, entry next open, exit per the audited trade list, 0.30% cost included
in ``net``. A slot is re-used the day after its exit. Equity is cash plus
open positions at cost; a position's P&L lands on its exit date.
Compared with random picks (same slots, same days) and equal-weight
buy-and-hold of the universe.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scripts.double_seven_selection import (load_trades, build_signal_features,
                                            composite_score, SPLIT)

FEATS = {"rs_rank": 1, "dist_sma_10": -1, "sma_150_slope": 1, "dist_sma_50": 1}


def simulate(d: pd.DataFrame, score: np.ndarray, slots: int, start=None, end=None):
    d = d.copy(); d["sc"] = score
    if start is not None: d = d[d.date >= start]
    if end is not None: d = d[d.date < end]
    d = d.sort_values(["date", "sc"], ascending=[True, False])
    days = np.sort(d["date"].unique())
    cash = 100.0; open_pos = []            # (exit_date, alloc, net, isin)
    eq = []; trades = 0
    groups = dict(tuple(d.groupby("date")))
    for day in days:
        # close positions that exited before today
        still = []
        for ex, alloc, net, isin in open_pos:
            if ex < day: cash += alloc * (1 + net / 100)
            else: still.append((ex, alloc, net, isin))
        open_pos = still
        held = {p[3] for p in open_pos}
        equity = cash + sum(p[1] for p in open_pos)
        free = slots - len(open_pos)
        for r in groups[day].itertuples():
            if free <= 0: break
            if r.isin in held or pd.isna(r.exit_date): continue
            alloc = min(equity / slots, cash)
            if alloc <= 0: break
            open_pos.append((r.exit_date, alloc, r.net, r.isin)); held.add(r.isin)
            cash -= alloc; free -= 1; trades += 1
        eq.append((day, cash + sum(p[1] for p in open_pos)))
    # flush
    for ex, alloc, net, isin in open_pos: cash += alloc * (1 + net / 100)
    eq.append((days[-1], cash))
    e = pd.Series(dict(eq)).sort_index()
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    cagr = (e.iloc[-1] / 100) ** (1 / yrs) * 100 - 100
    dd = (e / e.cummax() - 1).min() * 100
    m = e.resample("ME").last().pct_change().dropna() * 100
    return dict(total=e.iloc[-1] - 100, cagr=cagr, maxdd=dd, months_up=(m > 0).mean() * 100,
                worst_month=m.min(), best_month=m.max(), trades=trades, years=yrs), e


def main():
    trades = load_trades()
    d = build_signal_features(trades)
    sc = composite_score(d, FEATS).to_numpy()
    rng = np.random.default_rng(7)
    feat = pd.read_parquet("data/screen/features.parquet", columns=["isin", "date", "adj_close"])
    piv = feat.pivot(index="date", columns="isin", values="adj_close")

    periods = [("IS  2023-06 to 2025-06", None, SPLIT), ("OOS 2025-07 to 2026-09", SPLIT, None),
               ("FULL 2023-06 to 2026-09", None, None)]
    for label, a, b in periods:
        print(f"\n== {label} ==")
        for slots in (5, 10):
            r, e = simulate(d, sc, slots, a, b)
            rr = [simulate(d, rng.random(len(d)), slots, a, b)[0] for _ in range(30)]
            rand_cagr = np.mean([x["cagr"] for x in rr]); rand_dd = np.mean([x["maxdd"] for x in rr])
            print(f"  {slots:2d} slots ranked : total {r['total']:+6.1f}%  CAGR {r['cagr']:+5.1f}%/yr  "
                  f"maxDD {r['maxdd']:5.1f}%  months up {r['months_up']:.0f}%  worst mo {r['worst_month']:+.1f}%  "
                  f"trades {r['trades']}  ({r['years']:.1f} yrs)")
            print(f"  {slots:2d} slots random : CAGR {rand_cagr:+5.1f}%/yr  maxDD {rand_dd:5.1f}%  (mean of 30 draws)")
        # benchmark: equal-weight universe buy and hold over the same window
        p = piv.loc[e.index[0]:e.index[-1]]
        ew = (p / p.iloc[0]).mean(axis=1)
        yrs = (p.index[-1] - p.index[0]).days / 365.25
        bdd = (ew / ew.cummax() - 1).min() * 100
        print(f"  universe equal-weight buy & hold: total {ew.iloc[-1]*100-100:+6.1f}%  "
              f"CAGR {ew.iloc[-1]**(1/yrs)*100-100:+5.1f}%/yr  maxDD {bdd:5.1f}%")
    r, e = simulate(d, sc, 10)
    y = e.resample("YE").last(); y = pd.concat([pd.Series([100.0], index=[e.index[0]]), y])
    print("\nYear by year, 10 slots ranked:", {str(i.year): f"{v:+.1f}%" for i, v in (y.pct_change().dropna() * 100).items()})


if __name__ == "__main__":
    main()
