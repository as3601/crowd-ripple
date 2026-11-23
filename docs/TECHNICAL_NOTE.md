# Crowd Ripple: technical note

**Problem 2: crowdedness-aware journey planner** · Transport Stack Hackathon 2026, Phase 1

Crowd Ripple estimates how full every Delhi bus is, by route, direction, stretch between stops, day type and
hour, from the October 2026 e-ticketing data. It puts that next to every option in a journey planner, so a
commuter can trade minutes for comfort. A second layer, carried over from our earlier project *Ripple*
(cascading second-order effects of disruptions), shows operators how one disruption spreads as crowding:
to the next bus, to parallel routes and to the return trip.

## 1. Data

| Input | What we use |
|---|---|
| `October-Delhi.csv`: 2,559,899 ticket records, 1–31 Oct 2026 | boarding stop, alighting stop, route + direction, time, passengers, AC flag, vehicle number (51% of rows) |
| Static GTFS (DTC + DIMTS, 2,964 routes, 6,342 stops, 78,515 trips) | stop order per route, stop coordinates, scheduled run times, scheduled buses per hour |

The fares (₹4.5 / 9 / 13.5 / 18 / 22.5) are the standard fares less 10%, the discount for app and QR tickets.
So this is the **digital-ticket sample**: about 102,000 riders on a normal weekday, a small share of all
bus riders. That shapes the method (§3).

The GTFS we could fetch publicly is the DTC/DIMTS feed archived on Mobility Database (service dates 2022–2024).
It is older than the tickets, so some routes have changed. The pipeline accepts the current GTFS from
delhi.transportstack.in as a drop-in replacement (`data/gtfs/`).

## 2. From tickets to on-board load

1. **Route match.** Ticket route codes (`534UP`, `717ADOWN`, …) are matched to GTFS `route_long_name` after
   normalising case and punctuation. This covers **85.7%** of passengers.
2. **Stop match.** Each ticket's boarding and alighting stop is placed on that route's canonical stop pattern
   (its most common GTFS stop sequence). Names are written differently in the two sources
   ("Mahipal Pur (Nh8)" vs "Mahipal Pur Village"), so matching runs through a ladder of methods:
   exact normalised name → main part / containment → fuzzy (difflib ≥ 0.78) → **geographic snap**. In the
   last step, if the name exists anywhere in GTFS, we take its coordinates and use the nearest stop on this
   route within 700 m. Result: **71.8%** of all passengers have both stops placed. Tickets whose alighting
   stop comes before the boarding stop (0.8%) are dropped.
3. **Running count.** For each route and day we add a ticket's passengers at its boarding stop and subtract
   them at its alighting stop; a cumulative sum along the route gives **riders on every stretch** (stop k →
   k+1), by boarding hour. This is done for every date (31 × 24 × stretches per route), so we also keep the
   **day-to-day spread**.
4. **Day types.** Weekday / Saturday / Sunday profiles are means over dates. Gandhi Jayanti (2 Oct, 45% of a
   normal weekday) and Dussehra (20 Oct, 55%) are kept out.

## 3. From the ticket sample to everyone on board

**Unticketed riders.** Pink-ticket (free) women riders, pass holders and cash/ETM riders are not in this data.
We scale every ticketed flow by one city-wide **expansion factor**:

> E = system boardings per day ÷ ticketed boardings on a normal weekday = 40 lakh ÷ 102,303 ≈ **39.1**

40 lakh is the order of magnitude of DTC + cluster daily ridership. It is one number in `config.py`
(`SYSTEM_DAILY_BOARDINGS`, or the `CR_SYSTEM_DAILY_BOARDINGS` env var), and the validation report shows how
the results move at 30 and 50 lakh.
*Assumption:* unticketed riders travel at the same places and times as app users. This is the biggest
uncertainty in the model (see §7).

**Buses per hour.** For each route, stretch and hour we count scheduled buses from GTFS `stop_times` (the time
each trip reaches each stop) and take the larger of that and the number of **distinct vehicles seen in the
tickets** that hour (a hard lower bound on buses that actually ran).

**Physical consistency.** Tickets are people who *did* board, so the buses that ran must have had room for
them. If the implied load beats crush capacity, more buses must have run than the 2023 schedule lists. We
then raise the bus count until load ≤ crush and flag the route-hour (`bus_shortfall`, 3% of weekday
route-hours). Operators can use the flags as a list of routes whose schedule looks out of date.

**Sparse route-hours.** Multiplying a handful of tickets by 39 is noisy. Load is shrunk toward the route's
usual rider-weighted load: `load = w·raw + (1−w)·route_mean`, with `w = n / (n + 40)` and `n` = real tickets
behind that route-hour in the month. Each estimate also carries a **confidence** label: high (n ≥ 150),
medium (≥ 40), low (< 40). These are 67% / 25% / 8% of weekday rider-weighted estimates.

## 4. Load → crowding a commuter understands

Bus size: 12 m standard/low-floor buses (38 seats, 70 nominal, 90 crush) and 9 m DEVI feeder e-buses (route
codes `D-…`; 23 / 40 / 50).

| Level | Load per bus (12 m) | Perceived-time multiplier |
|---|---|---|
| 🟢 Seats available | ≤ 32 (85% of seats) | 1.0 |
| 🟡 Standing room | ≤ 52 | 1.3 |
| 🟠 Crowded | ≤ 70 | 1.6 |
| 🔴 Packed | > 70 | 2.1 |

The multipliers sit in the range of the crowding-valuation literature (Wardman & Whelan 2011 meta-analysis:
about 1.0 seated at low load, up to about 2 standing in crush conditions).

Load also varies from bus to bus within an hour (bunching). Per-bus load is modelled as lognormal around the
hourly mean, with coefficient of variation √(0.35² + (day-to-day CV)²). From that we compute:
- **seat chance**: P(load < seats) along the ride;
- **P(bus arrives full)**: P(load > crush) at the boarding stop. This is the core of the *crowd ripple* in the
  planner: if a bus is full you wait for the next one, so the expected wait grows by p/(1−p) × headway.
  Crowding becomes extra travel time, not just discomfort.

## 5. Journey planner

`GET /journeys?from=…&to=…&depart=…&comfort=0..1` (stop names or `lat,lon`).

- Stops within 900 m of origin and destination; direct routes plus one transfer (same stop or ≤ 250 m walk).
- In-vehicle time comes from GTFS run times for that departure hour; wait is headway/2 plus the full-bus
  ripple above.
- Each option's crowding is computed **stretch by stretch at the hour the bus actually reaches it**.
- Ranking: perceived minutes = walk×1.3 + wait×1.5 + Σ stretch minutes × (1 + comfort × (multiplier − 1))
  + 5 min per transfer. The slider moves the user between "fastest" and "avoid crowds".
- Output tags: *Fastest*, *Most comfortable*, *Best for you*. A **leave 20/40 min later** option is added when
  it is clearly less crowded. Each answer includes a comparison with the same trip at a quieter hour.
- Routes with no ticket data are shown as "No crowding data". When ranking we assume the cautious "between
  standing and crowded", so an unmeasured route never wins just because nothing is known about it.

## 6. Ripple: how one disruption spreads (operators)

From our Ripple project: a disruption is pushed through a dependency graph and returned as a time-ordered
list of 1st/2nd/3rd-order effects, each with an action. Here the graph is the bus network: buses on a route
follow one another, routes share stretches of road, and a bus that breaks down also misses its return trip.

| Event | 1st order | 2nd order | 3rd order |
|---|---|---|---|
| **Bus breakdown** on route R at hour H | The next bus takes both loads; riders who do not fit are left behind. The backlog drains by (crush − usual load) per following bus | 35% of stranded riders switch to parallel routes (routes sharing consecutive stops), split by frequency; their level change is shown | The same bus misses its return trip one run-time later; crowding on the opposite direction |
| **Demand surge** at a place (metro disruption, event) | Extra riders split across nearby routes by capacity; level change per route | Riders who cannot board anywhere: queue size | Usual riders further down those routes find fuller buses |
| **Add a bus** | Peak load, level and headway before → after | | |

**Where would one more bus help most?** For every route, day type and hour we compute crowded
passenger-minutes (Σ riders × stretch minutes × (multiplier − 1)) and how much one extra bus removes. The
operator console ranks route-hours by that relief.

## 7. Where the estimates are least reliable

1. **The expansion factor is global.** If app adoption differs between routes (e.g. AC vs non-AC, or office vs
   industrial corridors), loads on low-adoption routes are under-estimated and high-adoption routes
   over-estimated. With route-level ETM (conductor) counts the same pipeline would fit one factor per route.
2. **Pink-ticket riders** may travel more off-peak than app users; peak crowding could be slightly overstated.
3. **Planned vs actual alighting.** We use the stop on the ticket. Over-riding or getting off early is not
   observable here.
4. **Old GTFS.** 14% of tickets are on routes not in the 2022–24 feed (e.g. `801-IND-IGI`, many `D-` DEVI
   routes), and some stops have moved. These count in the expansion factor but are not mapped.
5. **Hourly bins.** Riders are assigned to their boarding hour; within-hour peaks (e.g. 8:45–9:15) are smoothed,
   though the lognormal per-bus spread partly covers this.
6. **Early-morning and late-night hours** have few tickets; shrinkage and the confidence label handle this, but
   treat "low" estimates as indicative.

## 8. Validation (full report: `docs/VALIDATION.md`)

- Morning peak 09:00 (5.2× midday), evening peak 18:00 (3.5×). Saturday 79% and Sunday 49% of a weekday.
- Crowded or packed share of riding: 45% at 9am, 10% at 2pm, 38% at 6pm; Sunday 9am 14%.
- **Hold-out:** loads built from three October weeks predict the fourth with Spearman 0.70; same level for
  63% of riders, within one level for 91%.
- Week-to-week rank correlation of route-hour ridership: 0.82–0.85.

## 9. Credits

Python, pandas, NumPy, SciPy (cKDTree), FastAPI, Uvicorn, Leaflet, OpenStreetMap tiles. GTFS mirror:
Mobility Database (feed mdb-3139, IIIT-Delhi open data licence). Crowding multipliers: Wardman, M. &
Whelan, G. (2011), *Twenty years of rail crowding valuation studies*, Transport Reviews 31(3).
