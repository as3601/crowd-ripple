"""Running on-board counts: ticket (boarding stop, alighting stop) -> people crossing every stretch of a route."""
import numpy as np


def route_flows(o, d, day, hour, pax, nstops, ndays):
    """Return array [ndays, 24, nseg] of ticketed riders on each stretch (stop k -> k+1), by boarding hour.

    Every ticket adds its passengers at the boarding stop and removes them at the alighting stop;
    a cumulative sum along the route then gives the running on-board count (the 'load profile').
    """
    arr = np.zeros((ndays, 24, nstops + 1), dtype=np.float32)
    np.add.at(arr, (day, hour, o), pax)
    np.add.at(arr, (day, hour, d), -pax)
    return np.cumsum(arr, axis=2)[:, :, : nstops - 1]


def by_route(t):
    """Split a mapped-ticket table (dict of arrays sorted or not) into per-route index arrays."""
    order = np.argsort(t["route"], kind="stable")
    r = t["route"][order]
    cuts = np.flatnonzero(np.diff(r)) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(r)]
    for s, e in zip(starts, ends):
        yield int(r[s]), order[s:e]
