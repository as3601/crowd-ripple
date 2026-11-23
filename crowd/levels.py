"""On-board load -> a crowding level a commuter understands in one second, plus the chance a bus is full."""
import math

import config

_LEVELS = [{"key": k, "label": lab, "ref": ref, "mult": m, "rank": i}
           for i, (k, lab, ref, m) in enumerate(config.LEVELS)]
BY_KEY = {lv["key"]: lv for lv in _LEVELS}


def thresholds(cap):
    c = config.CAPACITY[cap]
    return [c[ref] * frac for (ref, frac) in (lv["ref"] for lv in _LEVELS[:-1])]


def level_for(load, cap="12m"):
    for lv, th in zip(_LEVELS, thresholds(cap)):
        if load <= th:
            return lv
    return _LEVELS[-1]


def p_over(mean_load, limit, day_std=0.0):
    """Chance a given bus carries more than `limit`, combining bus-to-bus scatter (CV) and day-to-day spread.

    Per-bus load ~ lognormal with mean `mean_load` and total CV = sqrt(PER_BUS_CV^2 + (day_std/mean)^2).
    """
    if mean_load <= 0:
        return 0.0
    cv = math.sqrt(config.PER_BUS_LOAD_CV ** 2 + (day_std / mean_load) ** 2)
    s2 = math.log(1 + cv * cv)
    mu = math.log(mean_load) - s2 / 2
    z = (math.log(limit) - mu) / math.sqrt(s2)
    return 0.5 * math.erfc(z / math.sqrt(2))


def confidence(tickets_in_month):
    """How much to trust a route-hour estimate, from the number of real tickets behind it."""
    if tickets_in_month >= 150:
        return "high"
    if tickets_in_month >= 40:
        return "medium"
    return "low"
