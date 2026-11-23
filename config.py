"""Central settings and every modelling assumption, in one place so they are easy to inspect and change."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_TICKETS = ROOT / "data" / "raw" / "October-Delhi.csv"
GTFS_DIR = ROOT / "data" / "gtfs"
OUT_DIR = ROOT / "data" / "processed"

# ---------------------------------------------------------------- expansion (unticketed riders)
# The dataset holds ~100k e-tickets per weekday (10%-discounted app/QR fares). Delhi's DTC + cluster
# buses carry on the order of 40 lakh boardings per day, including pink-ticket (free) women riders
# and pass holders, who never appear in this data. We scale ticketed flows up by one city-wide factor:
#   EXPANSION = SYSTEM_DAILY_BOARDINGS / mean ticketed boardings on a normal weekday
# Override with the CR_SYSTEM_DAILY_BOARDINGS environment variable to test sensitivity.
SYSTEM_DAILY_BOARDINGS = 4_000_000

# Public holidays inside October 2026 that we keep out of the "normal day" profiles.
HOLIDAYS = {"2026-10-02": "Gandhi Jayanti", "2026-10-20": "Dussehra"}

DAYTYPES = ["weekday", "saturday", "sunday"]

# ---------------------------------------------------------------- bus capacity
# 12 m standard / low-floor buses (DTC & cluster, CNG and electric) vs 9 m DEVI feeder e-buses
# (routes with a "D-" prefix in the ticket data).
CAPACITY = {
    "12m": {"seats": 38, "nominal": 70, "crush": 90},
    "9m": {"seats": 23, "nominal": 40, "crush": 50},
}

# Crowding levels, as a share of seats / nominal capacity. Shown to commuters with one word + colour.
LEVELS = [
    # key, label, upper bound as load / capacity reference, in-vehicle time multiplier
    ("seats", "Seats available", ("seats", 0.85), 1.0),
    ("standing", "Standing room", ("nominal", 0.75), 1.3),
    ("crowded", "Crowded", ("nominal", 1.0), 1.6),
    ("packed", "Packed", (None, None), 2.1),
]
# Multipliers: how much worse a minute feels at each level. They follow the range reported in the
# crowding-valuation literature (Wardman & Whelan 2011 meta-analysis: ~1.0 seated at low load up to
# ~2 when standing in crush conditions).

# Load varies bus-to-bus inside an hour (bunching, irregular headways). Coefficient of variation of
# per-bus load around the hourly mean, used when computing the chance a bus is full.
PER_BUS_LOAD_CV = 0.35

# ---------------------------------------------------------------- planner
WALK_SPEED_KMH = 4.5
WALK_DETOUR = 1.25          # straight line -> walked distance
MAX_ACCESS_WALK_M = 900
MAX_TRANSFER_WALK_M = 250
TRANSFER_PENALTY_MIN = 5
MAX_WAIT_MIN = 30
WAIT_WEIGHT = 1.5           # waiting feels worse than riding
WALK_WEIGHT = 1.3

# Routes with no ticket data: we cannot show their crowding, so the planner assumes "between standing and
# crowded" rather than letting an unmeasured route win just because nothing is known about it.
UNKNOWN_MULT = 1.45

# Sparse route-hours (few real tickets) are noisy once multiplied by the expansion factor. Their load is
# shrunk toward the route's usual (rider-weighted) load: w = n / (n + SHRINK_K), n = tickets in the month.
SHRINK_K = 40
