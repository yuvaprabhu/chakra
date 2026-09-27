"""Settings more than one script needs, in one place.

The market-cap floor defines the tradable universe. Every script that builds a
universe reads it from here, so raising or lowering it is one edit - or one
environment variable - instead of a hunt through argparse defaults.
"""
import os

MCAP_FLOOR_CR = float(os.environ.get("MCAP_FLOOR_CR", 1000))

# What a name needs before a screen may look at it.
MIN_HISTORY_BARS = 250          # indicators need a 200-day average plus slope
HOURLY_BACKFILL_MONTHS = 4      # the chart shows 40 sessions of hourly bars


def universe_isins(store, min_mcap_cr=None) -> list[str]:
    """Every NSE equity at or above the market-cap floor, by ISIN."""
    floor = MCAP_FLOOR_CR if min_mcap_cr is None else float(min_mcap_cr)
    fund = store.read_fundamentals()
    mcap_cr = fund["vendor_mcap"] / 1e7
    return sorted(fund.loc[mcap_cr >= floor, "isin"].dropna().unique())
