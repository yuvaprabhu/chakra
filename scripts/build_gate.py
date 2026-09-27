"""The market gate the playbook uses, as one reusable series.

R3 = breadth-with-hysteresis OR Zweig-thrust-window, exactly as in
scripts/regime_gates.py.  Written to data/screen/gate_r3.parquet with columns
date, b50 (% of universe above the 50-DMA), hyst (0/1), zweig_ema, thrust (0/1),
R3 (0/1).  Any study that wants "only trade when the gate is on" reads this file
rather than re-deriving it, so every study uses the identical gate.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
feat = pd.read_parquet("data/screen/features.parquet", columns=["isin", "date", "adj_close", "sma_50", "ret_1d"])
CAL = pd.DatetimeIndex(sorted(feat["date"].unique()))
g = feat.groupby("date")
B = pd.DataFrame(index=CAL)
B["adv"] = g["ret_1d"].apply(lambda s: (s > 0).sum()).reindex(CAL)
B["dec"] = g["ret_1d"].apply(lambda s: (s < 0).sum()).reindex(CAL)
B["b50"] = g.apply(lambda x: (x["adj_close"] > x["sma_50"]).mean() * 100).reindex(CAL)
B["zweig_ema"] = (B["adv"] / (B["adv"] + B["dec"])).ewm(span=10, adjust=False).mean()
b = B["b50"].to_numpy(); hyst = np.full(len(b), np.nan); state = np.nan
for i in range(len(b)):
    if np.isnan(b[i]): continue
    if np.isnan(state): state = 1.0 if b[i] > 50 else 0.0
    elif state == 1.0 and b[i] < 40: state = 0.0
    elif state == 0.0 and b[i] > 50: state = 1.0
    hyst[i] = state
B["hyst"] = hyst
ze = B["zweig_ema"].to_numpy(); thrust = np.zeros(len(ze))
for i in range(10, len(ze)):
    if ze[i] > 0.615 and np.nanmin(ze[i - 10:i]) < 0.40:
        thrust[i:i + 126] = 1
B["thrust"] = thrust
B["R3"] = ((B["hyst"] == 1) | (B["thrust"] == 1)).astype(int)
out = B[["b50", "hyst", "zweig_ema", "thrust", "R3"]].reset_index().rename(columns={"index": "date"})
out.to_parquet("data/screen/gate_r3.parquet", index=False)
print(out.tail(3).to_string(index=False)); print("R3 on share since 2024-06:", out[out.date >= "2024-06-01"].R3.mean().round(3))
