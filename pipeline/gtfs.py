"""Static GTFS -> one canonical stop pattern per route, with run times and buses-per-hour at every stop."""
import numpy as np
import pandas as pd

import config


def _secs(s: pd.Series) -> np.ndarray:
    p = s.str.split(":", expand=True).astype(int)
    return (p[0] * 3600 + p[1] * 60 + p[2]).to_numpy()


def norm_route(name: str) -> str:
    return "".join(ch for ch in str(name).upper() if ch.isalnum())


def load_gtfs():
    d = config.GTFS_DIR
    stops = pd.read_csv(d / "stops.txt", dtype={"stop_id": int})
    routes = pd.read_csv(d / "routes.txt")
    trips = pd.read_csv(d / "trips.txt")
    st = pd.read_csv(d / "stop_times.txt", usecols=["trip_id", "arrival_time", "stop_id", "stop_sequence"])
    st = st.sort_values(["trip_id", "stop_sequence"], kind="stable")
    st["t"] = _secs(st["arrival_time"])
    st = st.merge(trips[["trip_id", "route_id"]], on="trip_id")
    return stops, routes, st


def build_patterns():
    """Return dict route_id -> pattern info.

    pattern = {name, stops [stop_id...], cum [24 x nstops] minutes from first stop by departure hour,
               buses [24 x nstops] scheduled buses passing each stop per hour, n_trips}
    """
    stops, routes, st = load_gtfs()
    names = dict(zip(routes.route_id, routes.route_long_name.astype(str).str.strip()))
    out = {}
    for rid, g in st.groupby("route_id", sort=False):
        seqs = g.groupby("trip_id", sort=False)["stop_id"].agg(tuple)
        if seqs.empty:
            continue
        counts = seqs.value_counts()
        top = counts.max()
        pattern = max((s for s, c in counts.items() if c == top), key=len)
        n = len(pattern)
        if n < 3:
            continue
        ptrips = seqs.index[seqs.map(lambda s: s == pattern)]
        gt = g[g.trip_id.isin(ptrips)]
        times = gt["t"].to_numpy().reshape(len(ptrips), n)
        start = times[:, 0]
        rel = (times - start[:, None]) / 60.0
        hour0 = (start // 3600) % 24
        overall = np.median(rel, axis=0)
        cum = np.tile(overall, (24, 1))
        for h in np.unique(hour0):
            cum[h] = np.median(rel[hour0 == h], axis=0)
        # buses passing stop k during hour h, scaled up for trips of the route on other patterns
        scale = len(seqs) / len(ptrips)
        hours_at = (times // 3600) % 24
        buses = np.zeros((24, n))
        for k in range(n):
            buses[:, k] = np.bincount(hours_at[:, k].astype(int), minlength=24)[:24] * scale
        out[int(rid)] = {"name": names.get(rid, str(rid)), "stops": list(map(int, pattern)),
                         "cum": cum.astype(np.float32), "buses": buses.astype(np.float32),
                         "n_trips": int(len(seqs))}
    stop_tab = stops.set_index("stop_id")[["stop_name", "stop_lat", "stop_lon"]]
    return out, stop_tab
