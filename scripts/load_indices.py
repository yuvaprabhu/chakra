"""Load every NSE index constituent list into point-in-time membership."""
import glob, logging, sys
sys.path.insert(0, ".")
import pandas as pd
from screener.store import Store
from screener.universe import Universe, parse_constituents_csv

logging.basicConfig(level=logging.ERROR)
AS_OF = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp.today().normalize()

# Display names keyed by the slug in the filename.
NAMES = {
    "nifty50": "NIFTY 50", "niftynext50": "NIFTY NEXT 50", "nifty100": "NIFTY 100",
    "nifty200": "NIFTY 200", "nifty500": "NIFTY 500", "niftytotalmarket": "NIFTY TOTAL MARKET",
    "niftymidcap150": "NIFTY MIDCAP 150", "niftysmallcap250": "NIFTY SMALLCAP 250",
    "niftymicrocap250": "NIFTY MICROCAP 250", "niftymidsmallcap400": "NIFTY MIDSMALLCAP 400",
    "niftybank": "NIFTY BANK", "niftyit": "NIFTY IT", "niftyauto": "NIFTY AUTO",
    "niftypharma": "NIFTY PHARMA", "niftyfmcg": "NIFTY FMCG", "niftymetal": "NIFTY METAL",
    "niftyrealty": "NIFTY REALTY", "niftyenergy": "NIFTY ENERGY",
    "niftypsubank": "NIFTY PSU BANK", "niftyprivatebank": "NIFTY PRIVATE BANK",
    "niftyfinancialservices": "NIFTY FINANCIAL SERVICES", "niftymedia": "NIFTY MEDIA",
    "niftyinfra": "NIFTY INFRASTRUCTURE", "niftyconsumerdurables": "NIFTY CONSUMER DURABLES",
    "niftyhealthcare": "NIFTY HEALTHCARE", "niftyoilgas": "NIFTY OIL & GAS",
    "niftycommodities": "NIFTY COMMODITIES",
}
BROAD = {"NIFTY 50", "NIFTY NEXT 50", "NIFTY 100", "NIFTY 200", "NIFTY 500",
         "NIFTY TOTAL MARKET", "NIFTY MIDCAP 150", "NIFTY SMALLCAP 250",
         "NIFTY MICROCAP 250", "NIFTY MIDSMALLCAP 400"}

store = Store("data")
u = Universe(store)
loaded = []
for f in sorted(glob.glob("data/raw/ind_*list.csv")):
    slug = f.split("ind_")[1].replace("list.csv", "").replace("_", "")
    name = NAMES.get(slug)
    if not name:
        print(f"  skip unknown index file {f}")
        continue
    cons = parse_constituents_csv(f)
    res = u.apply_snapshot(name, AS_OF, cons)
    kind = "broad" if name in BROAD else "sector"
    loaded.append((name, kind, len(cons)))
    print(f"  {name:26s} {kind:6s} {len(cons):>4d}  {res}")

pd.DataFrame(loaded, columns=["index_name", "kind", "size"]).to_csv("data/raw/index_catalog.csv", index=False)
print(f"\n{len(loaded)} indices loaded as of {AS_OF.date()}")
print(f"membership rows: {len(store.read_membership())}")
