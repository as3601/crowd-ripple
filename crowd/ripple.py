"""Crowd Ripple: how one disruption spreads through the bus network as crowding, over time.

Adapted from our Ripple project (cascading second-order effects of a disruption through a dependency
graph). Here the graph is the bus network: buses on a route follow each other, routes share stretches
of road, and a bus that breaks down also misses its return trip. A disruption is pushed through these
links and returned as a time-ordered list of 1st/2nd/3rd-order effects, each with a suggested action.

Events
  cancel  - a bus on route R breaks down / a trip is cancelled at hour H
  surge   - extra riders appear at a place (metro line disruption, stadium event, market closing)
  add     - operator adds one bus on route R at hour H (the fix, to compare against)
"""
import numpy as np

import config
from crowd import levels
from crowd.store import DT_INDEX

SWITCH_SHARE = 0.35      # share of riders stranded by a full bus who switch to a parallel route
MAX_FOLLOWERS = 6


def _lvl(load, cap):
    return levels.level_for(load, cap)


def _impact(order, t_min, title, detail, people, severity, action, route=None, stretch=None, path=None):
    return {"order": order, "t_min": round(t_min), "title": title, "detail": detail, "people": int(round(people)),
            "severity": round(severity, 2), "action": action, "route": route, "stretch": stretch, "path": path or []}


class Ripple:
    def __init__(self, model):
        self.m = model

    def _path(self, rid, a, b):
        st = self.m.routes[rid]["stops"]
        return [self.m.coord(s) for s in st[a:b + 1]]

    def _stretch_name(self, rid, a, b):
        st = self.m.routes[rid]["stops"]
        return f"{self.m.stop_name.get(st[a])} → {self.m.stop_name.get(st[min(b + 1, len(st) - 1)])}"

    def _parallel(self, rid, ks):
        """Routes that run along the same stretches (same consecutive stop pair, either stop id or <=250 m)."""
        st = self.m.routes[rid]["stops"]
        pairs = {(st[k], st[k + 1]) for k in ks}
        out = {}
        for (a, b) in pairs:
            for r2, p in self.m.serving.get(a, []):
                if r2 == rid:
                    continue
                s2 = self.m.routes[r2]["stops"]
                if p + 1 < len(s2) and (s2[p + 1] == b or s2[p + 1] in {n for n, _ in self.m.transfer_neighbours(b)}):
                    out.setdefault(r2, set()).add(p)
        return out

    # ------------------------------------------------------------------ events
    def simulate(self, kind, dt, hour, route=None, lat=None, lon=None, extra_riders=0, radius_m=600):
        if kind in ("cancel", "add"):
            rid = self.m.find_route(route)
            if rid is None or self.m.routes[rid]["flow"] is None:
                return {"error": f"No crowding data for route '{route}'"}
            return self._cancel(rid, dt, hour) if kind == "cancel" else self._add(rid, dt, hour)
        if kind == "surge":
            return self._surge(dt, hour, lat, lon, extra_riders, radius_m)
        return {"error": f"unknown event '{kind}'"}

    def _cancel(self, rid, dt, hour):
        m, d = self.m, DT_INDEX[dt]
        r = m.routes[rid]
        cap = config.CAPACITY[r["cap"]]
        L = r["load"][d, hour].astype(float)                 # usual load per bus, by stretch
        b = float(max(r["buses"][d, hour].max(), 1))
        headway = 60 / b
        impacts = []
        # 1st order: the cancelled bus's riders pile onto the next bus; whoever does not fit waits for the one after.
        # Each following bus only has (crush - usual load) spare room, so the backlog drains bus by bus.
        demand = 2 * L
        k = int(np.argmax(demand))
        bad = np.flatnonzero(demand > cap["nominal"])
        a_, z_ = (int(bad.min()), int(bad.max())) if len(bad) else (k, k)
        lv = _lvl(float(min(demand[k], cap["crush"])), r["cap"])
        impacts.append(_impact(
            1, headway, f"Next {r['ticket_name']} bus overloaded",
            f"It must also carry the missing bus's riders: about {int(demand[k])} people want it at the busiest point "
            f"(room for {cap['crush']}). {lv['label']}" + (f" from {self._stretch_name(rid, a_, z_)}." if len(bad) else "."),
            float(L.max()), 0.9 if len(bad) else 0.5,
            "Dispatch a spare bus from the nearest depot or short-turn a bus to fill the gap.",
            r["ticket_name"], self._stretch_name(rid, a_, z_), self._path(rid, a_, z_ + 1)))
        backlog = np.maximum(demand - cap["crush"], 0)
        first_left = float(backlog.max())
        stranded_total, wait_extra, n = 0.0, 0.0, 0
        while backlog.max() >= 1 and n < MAX_FOLLOWERS:
            stranded_total += float(backlog.max())
            wait_extra += float(backlog.max()) * headway
            spare = np.maximum(cap["crush"] - L, 0)
            backlog = np.maximum(backlog - spare, 0)
            n += 1
        if first_left >= 1:
            kk = int(np.argmax(np.maximum(demand - cap["crush"], 0)))
            clears = (f"The backlog clears after about {n} more buses (~{round(n * headway)} min)."
                      if backlog.max() < 1 else
                      f"Following buses are nearly full too, so the backlog does not clear within {round(n * headway)} min.")
            impacts.append(_impact(
                1, headway * 2, f"~{int(first_left)} riders left behind",
                f"The overloaded bus passes them by at {self._stretch_name(rid, kk, kk)}; each waits at least {round(headway)} min more. {clears}",
                first_left, 0.8, "Show waiting riders (stop display / app) which parallel route has room.",
                r["ticket_name"], self._stretch_name(rid, kk, kk), self._path(rid, kk, kk + 1)))
        delay_pax_min = float(L.max()) * headway + wait_extra
        first_packed = bad
        # 2nd order: some stranded riders switch to parallel routes on the same road
        ks = list(first_packed) if first_packed is not None and len(first_packed) else [int(np.argmax(L))]
        par = self._parallel(rid, ks)
        movers = max(stranded_total, float(L.max()) * 0.5) * SWITCH_SHARE
        tot_freq = sum(float(m.routes[r2]["buses"][d, hour].max()) for r2 in par) or 1
        for r2, ps in sorted(par.items(), key=lambda kv: -float(m.routes[kv[0]]["buses"][d, hour].max()))[:4]:
            R2 = m.routes[r2]
            if R2["flow"] is None:
                continue
            share = float(R2["buses"][d, hour].max()) / tot_freq
            b2 = max(float(R2["buses"][d, hour].max()), 1)
            add_per_bus = movers * share / b2
            p = min(ps)
            before = float(R2["load"][d, hour, p])
            after = before + add_per_bus
            lv0, lv1 = _lvl(before, R2["cap"]), _lvl(after, R2["cap"])
            if add_per_bus < 0.5:
                continue
            impacts.append(_impact(
                2, headway * 1.5, f"Spill-over onto {R2['ticket_name']}",
                f"Riders switching from {r['ticket_name']} add ~{max(1, round(add_per_bus))} per bus on the shared stretch "
                f"{self._stretch_name(r2, p, p)}: {lv0['label']} → {lv1['label']}.",
                movers * share, 0.6 if lv1["rank"] > lv0["rank"] else 0.3,
                f"Good alternative while {r['ticket_name']} recovers." if lv1["rank"] <= 1 else
                f"{R2['ticket_name']} is also near capacity: add a trip here too.",
                R2["ticket_name"], self._stretch_name(r2, p, p), self._path(r2, p, p + 1)))
        # 3rd order: the broken bus also misses its return trip in the opposite direction
        run = float(r["cum"][hour][-1])
        back_h = int(hour + max(1, round(run / 60))) % 24
        opp = self._opposite(rid)
        if opp is not None and m.routes[opp]["flow"] is not None:
            R3 = m.routes[opp]
            L3 = R3["load"][d, back_h]
            b3 = max(float(R3["buses"][d, back_h].max()), 1)
            k3 = int(np.argmax(L3))
            after = float(L3[k3]) * 2
            impacts.append(_impact(
                3, run + 60 / b3, f"Return trip {R3['ticket_name']} missing around {back_h:02d}:00",
                f"The same bus would have run back. Riders on {R3['ticket_name']} see a {round(60 / b3)} min longer gap and the "
                f"next bus reaches ~{int(after)} on board ({_lvl(after, R3['cap'])['label']}) near {self._stretch_name(opp, k3, k3)}.",
                float(L3[k3]), 0.5, "Swap in a depot spare for the return run before it is due.",
                R3["ticket_name"], self._stretch_name(opp, k3, k3), self._path(opp, k3, k3 + 1)))
        fix = self._add(rid, dt, hour, summary_only=True)
        return self._wrap("cancel", rid, dt, hour, impacts, delay_pax_min, fix)

    def _add(self, rid, dt, hour, summary_only=False):
        m, d = self.m, DT_INDEX[dt]
        r = m.routes[rid]
        relief = float(r["relief"][d, hour])
        b = float(max(r["buses"][d, hour].max(), 1))
        L = r["load"][d, hour]
        k = int(np.argmax(L))
        before, after = float(L[k]), float(L[k]) * b / (b + 1)
        out = {"route": r["ticket_name"], "relief_pax_min": round(relief), "peak_before": round(before),
               "peak_after": round(after), "level_before": _lvl(before, r["cap"])["label"],
               "level_after": _lvl(after, r["cap"])["label"], "headway_before": round(60 / b, 1),
               "headway_after": round(60 / (b + 1), 1), "stretch": self._stretch_name(rid, k, k)}
        if summary_only:
            return out
        imp = [_impact(1, 0, f"One more {r['ticket_name']} bus at {hour:02d}:00",
                       f"Peak load {out['peak_before']} → {out['peak_after']} per bus on {out['stretch']} "
                       f"({out['level_before']} → {out['level_after']}); gap between buses {out['headway_before']} → {out['headway_after']} min.",
                       float(r["flow"][d, hour].max()), 0.4, "Schedule it in the peak window shown.",
                       r["ticket_name"], out["stretch"], self._path(rid, k, k + 1))]
        return self._wrap("add", rid, dt, hour, imp, -relief, out)

    def _surge(self, dt, hour, lat, lon, extra, radius_m):
        """Extra riders at a place (e.g. a metro line stops and its riders take the bus)."""
        m, d = self.m, DT_INDEX[dt]
        stops = m.near(lat, lon, radius_m)
        if not stops:
            return {"error": "No bus stops near that point"}
        served = {}
        for sid, _ in stops:
            for rid, pos in m.serving[sid]:
                R = m.routes[rid]
                if R["flow"] is not None and pos < len(R["stops"]) - 1 and rid not in served:
                    served[rid] = pos
        if not served:
            return {"error": "No routes with crowding data near that point"}
        cap_tot = {rid: float(m.routes[rid]["buses"][d, hour, p]) * config.CAPACITY[m.routes[rid]["cap"]]["nominal"]
                   for rid, p in served.items()}
        tot = sum(cap_tot.values()) or 1
        impacts, overflow, delay = [], 0.0, 0.0
        rows = []
        for rid, p in served.items():
            R = m.routes[rid]
            b = max(float(R["buses"][d, hour, p]), 1)
            add = extra * cap_tot[rid] / tot / b                 # extra riders per bus boarding at the stop
            # spread their trips over the next stops like usual riders from here (half get off within ~6 stops)
            n = min(len(R["stops"]) - 1, p + 12)       # the next dozen stretches, where surge riders still are
            ks = np.arange(p, n)
            stay = np.exp(-(ks - p) / 6.0)
            L0 = R["load"][d, hour, p:n]
            L1 = L0 + add * stay
            crush = config.CAPACITY[R["cap"]]["crush"]
            left = max(float(L1[0]) - crush, 0) * b
            overflow += left
            delay += left * 60 / b
            k = int(np.argmax(L1))
            rows.append((rid, p, k + p, float(L0[k]), float(L1[k]), left, add * b))
        rows.sort(key=lambda x: (-(_lvl(x[4], self.m.routes[x[0]]['cap'])['rank'] - _lvl(x[3], self.m.routes[x[0]]['cap'])['rank']), -x[6]))
        for rid, p, k, before, after, left, riders in rows[:8]:
            R = m.routes[rid]
            lv0, lv1 = _lvl(before, R["cap"]), _lvl(after, R["cap"])
            impacts.append(_impact(
                1, 5, f"{R['ticket_name']}: {lv0['label']} → {lv1['label']}",
                f"Takes ~{int(riders)} of the extra riders per hour; peak ~{int(after)} on board at {self._stretch_name(rid, k, k)}.",
                riders, 0.85 if lv1["rank"] >= 3 else 0.6 if lv1["rank"] > lv0["rank"] else 0.3,
                "Run short-turn shuttles from the affected stop" if lv1["rank"] >= 2 else "Has room: direct riders here.",
                R["ticket_name"], self._stretch_name(rid, k, k), self._path(rid, p, k + 1)))
        if overflow >= 1:
            impacts.append(_impact(2, 15, f"~{int(overflow)} riders per hour cannot board",
                                   "Every bus leaving the area is full at the first stops, so queues build at the stop.",
                                   overflow, 0.9, "Queue marshals and bus bridge (dedicated shuttles to the next metro station).",
                                   path=[(lat, lon)]))
        # 3rd order: the routes that take the surge get crowded further down the line, where usual riders wait
        down = []
        for rid, p, k, before, after, left, riders in rows[:8]:
            R = m.routes[rid]
            n = len(R["stops"]) - 1
            for kk in range(p + 3, min(p + 15, n)):
                L0 = float(R["load"][d, hour, kk])
                L1 = L0 + riders / max(float(R["buses"][d, hour, p]), 1) * np.exp(-(kk - p) / 6.0)
                if _lvl(L1, R["cap"])["rank"] >= 2 > _lvl(L0, R["cap"])["rank"]:
                    down.append((rid, kk, L0, L1))
                    break
        for rid, kk, L0, L1 in down[:3]:
            R = m.routes[rid]
            impacts.append(_impact(3, 30, f"Usual {R['ticket_name']} riders further along feel it",
                                   f"Riders boarding near {self._stretch_name(rid, kk, kk)} find the bus {_lvl(L1, R['cap'])['label'].lower()} "
                                   f"instead of {_lvl(L0, R['cap'])['label'].lower()}.",
                                   L0, 0.4, "Warn riders on the app; suggest leaving 15–20 min later.",
                                   R["ticket_name"], self._stretch_name(rid, kk, kk), self._path(rid, kk, kk + 1)))
        return self._wrap("surge", None, dt, hour, impacts, delay, {"extra_riders": extra, "stops": len(stops),
                                                                    "routes": len(served), "lat": lat, "lon": lon})

    def _opposite(self, rid):
        name = str(self.m.routes[rid]["ticket_name"]).upper()
        if name.endswith("UP"):
            return self.m.find_route(name[:-2] + "DOWN")
        if name.endswith("DOWN"):
            return self.m.find_route(name[:-4] + "UP")
        return None

    def _wrap(self, kind, rid, dt, hour, impacts, pax_min, extra):
        impacts.sort(key=lambda x: (x["order"], x["t_min"]))
        by_order = {o: sum(1 for i in impacts if i["order"] == o) for o in (1, 2, 3)}
        return {"event": kind, "route": self.m.routes[rid]["ticket_name"] if rid is not None else None,
                "daytype": dt, "hour": hour, "impacts": impacts, "counts": by_order,
                "passenger_minutes_lost": round(max(pax_min, 0)), "people_affected": int(sum(i["people"] for i in impacts)),
                "fix": extra}
