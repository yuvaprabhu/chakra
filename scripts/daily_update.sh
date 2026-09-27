#!/usr/bin/env bash
# End-of-day refresh. Run after 16:00 IST on a trading day.
#
#   bash scripts/daily_update.sh              # normal day
#   DAYS=15 bash scripts/daily_update.sh      # catch up after a break
#   MCAP_FLOOR_CR=2000 bash scripts/daily_update.sh   # different universe
#
# Publishing dashboard/index.html plus its data files to the artifact is the
# only step this cannot do; ask Claude, or run it yourself from the web UI.
set -euo pipefail
cd "$(dirname "$0")/.."
export MCAP_FLOOR_CR="${MCAP_FLOOR_CR:-1000}"
echo "== universe floor: Rs ${MCAP_FLOOR_CR} Cr =="
echo "== fundamentals =="; python3 scripts/sync_fundamentals.py
echo "== backfill gaps =="; python3 scripts/sync_universe.py
echo "== today ==";        python3 scripts/sync_today.py --days "${DAYS:-5}"
echo "== indices (VIX, Nifty) =="; python3 scripts/ingest_indices.py | tail -1
echo "== screens =="; python3 scripts/run_screen.py 2>&1 | grep -v "m_cpr_width_rank\|Warning\|warnings.warn"
echo "== export ==";  python3 scripts/export_dashboard.py 2>&1 | grep -iE "wrote|screens:|symbols="
python3 - <<'PY'
import json, pandas as pd
d = json.load(open("dashboard/data.json")); h = pd.read_parquet("data/screen/hits.parquet")
print(f"dashboard as_of {d['meta']['as_of']} | {len(d['symbols'])} names | "
      f"hits {h['date'].max().date()} across {h['isin'].nunique()} names")
ld = h[h['rule'] == 'leader_dip']
print("Leader Dip:", ", ".join(sorted(ld['symbol'])) or "none today")
PY
echo "done. Publish dashboard/index.html with every dashboard/*.json file."
