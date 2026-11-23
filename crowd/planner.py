"""Bus journey planner on static GTFS that ranks options on travel time AND expected crowding.

Search: every route within walking distance of the origin x every route within walking distance of the
destination, direct or with one transfer (same stop or a short walk). Each option is scored by
  perceived minutes = walk*1.3 + wait*1.5 + sum(stretch minutes * crowding multiplier^comfort) + 5/transfer
where `comfort` (0..1) is the user's slider: 0 = only time matters, 1 = crowding fully counts.
"""
from datetime import datetime

import config
from crowd.store import DT_INDEX, walk_min  # noqa: F401

LEVEL_RANK = {k: i for i, (k, *_r) in enumerate(config.LEVELS)}
LEVEL_RANK["unknown"] = -1
LEVEL_LABEL = {k: lab for (k, lab, *_r) in config.LEVELS}
LEVEL_LABEL["unknown"] = "No data"
MULT = {k: m for (k, _l, _r, m) in config.LEVELS}


def daytype_of(dt: datetime):
    return "saturday" if dt.weekday() == 5 else "sunday" if dt.weekday() == 6 else "weekday"


def _fmt(t_min):
    t = int(round(t_min)) % (24 * 60)
    return f"{t // 60:02d}:{t % 60:02d}"


class Planner:
    def __init__(self, model):
        self.m = model

    # ---------------------------------------------------------------- candidates
    def _access(self, lat, lon):
        out = {}
        for sid, dist in self.m.near(lat, lon, config.MAX_ACCESS_WALK_M):
            for rid, pos in self.m.serving[sid]:
                w = walk_min(dist)
                if rid not in out or w < out[rid][0][1]:
                    out.setdefault(rid, [])
                    out[rid] = [(pos, w, sid)] + [x for x in out[rid] if x[0] != pos]
                else:
                    out[rid].append((pos, w, sid))
        return out

    def _candidates(self, O, D):
        """Yield (rid1, i, walk_o, rid2|None, t_pos_on_r1, t_pos_on_r2, transfer_walk, j, walk_d)."""
        m = self.m
        # direct
        for rid in set(O) & set(D):
            best = None
            for i, wo, _ in O[rid]:
                for j, wd, _ in D[rid]:
                    if j > i:
                        ride = m.routes[rid]["cum"][8][j] - m.routes[rid]["cum"][8][i]
                        c = wo + wd + ride
                        if best is None or c < best[0]:
                            best = (c, i, wo, j, wd)
            if best:
                _, i, wo, j, wd = best
                yield (rid, i, wo, None, None, None, 0.0, j, wd)
        # one transfer: r1 from origin, r2 to destination
        down = {}
        for r1, lst in O.items():
            i, wo, _ = min(lst, key=lambda x: x[1] + x[0] * 0.01)
            stops = m.routes[r1]["stops"]
            down[r1] = (i, wo, {s: p for p, s in enumerate(stops) if p > i})
        up = {}
        for r2, lst in D.items():
            j, wd, _ = min(lst, key=lambda x: x[1])
            stops = m.routes[r2]["stops"]
            up[r2] = (j, wd, {s: p for p, s in enumerate(stops) if p < j})
        for r1, (i, wo, dmap) in down.items():
            if r1 in D:
                continue
            for r2, (j, wd, umap) in up.items():
                if r2 == r1 or r2 in O:
                    continue
                best = None
                for s, p1 in dmap.items():
                    if s in umap:
                        cand = [(s, 0.0)]
                    else:
                        cand = [(n, d) for n, d in self.m.transfer_neighbours(s) if n in umap]
                    for s2, dist in cand:
                        tw = walk_min(dist)
                        r1c, r2c = m.routes[r1]["cum"][8], m.routes[r2]["cum"][8]
                        c = (r1c[p1] - r1c[i]) + tw + (r2c[j] - r2c[umap[s2]])
                        if best is None or c < best[0]:
                            best = (c, p1, umap[s2], tw)
                if best:
                    _, p1, p2, tw = best
                    yield (r1, i, wo, r2, p1, p2, tw, j, wd)

    # ---------------------------------------------------------------- evaluation
    def _bus_leg(self, rid, i, j, dt, t):
        w = self.m.wait_at(rid, i, dt, t)
        if w is None:
            return None
        board = t + w["wait"]
        c = self.m.leg_crowding(rid, i, j, dt, board)
        r = self.m.routes[rid]
        return {"type": "bus", "route": str(r["ticket_name"]), "route_id": rid, "cap": r["cap"],
                "ac_share": round(r["ac_share"], 2),
                "from": self.m.stop_name.get(r["stops"][i]), "to": self.m.stop_name.get(r["stops"][j]),
                "from_xy": self.m.coord(r["stops"][i]), "to_xy": self.m.coord(r["stops"][j]),
                "path": [self.m.coord(s) for s in r["stops"][i:j + 1]],
                "board_at": _fmt(board), "alight_at": _fmt(board + c["ride_min"]),
                "stops": j - i, **w, **c, "_board": board, "_end": board + c["ride_min"]}

    def _evaluate(self, cand, dt, t0, origin, dest):
        r1, i, wo, r2, p1, p2, tw, j, wd = cand
        legs = []
        t = t0
        o_stop = self.m.routes[r1]["stops"][i]
        legs.append({"type": "walk", "min": round(wo, 1), "to": self.m.stop_name.get(o_stop),
                     "path": [origin, self.m.coord(o_stop)]})
        t += wo
        if r2 is None:
            b = self._bus_leg(r1, i, j, dt, t)
            if not b:
                return None
            legs.append(b)
            t = b["_end"]
        else:
            b1 = self._bus_leg(r1, i, p1, dt, t)
            if not b1:
                return None
            legs.append(b1)
            t = b1["_end"]
            if tw > 0:
                s_from, s_to = self.m.routes[r1]["stops"][p1], self.m.routes[r2]["stops"][p2]
                legs.append({"type": "walk", "min": round(tw, 1), "to": self.m.stop_name.get(s_to),
                             "path": [self.m.coord(s_from), self.m.coord(s_to)], "transfer": True})
                t += tw
            b2 = self._bus_leg(r2, p2, j, dt, t)
            if not b2:
                return None
            legs.append(b2)
            t = b2["_end"]
        last = legs[-1]
        d_stop = self.m.routes[last["route_id"]]["stops"][j]
        legs.append({"type": "walk", "min": round(wd, 1), "to": "Destination", "path": [self.m.coord(d_stop), dest]})
        t += wd
        buses = [lg for lg in legs if lg["type"] == "bus"]
        walk = sum(lg["min"] for lg in legs if lg["type"] == "walk")
        wait = sum(lg["wait"] for lg in buses)
        ride = sum(lg["ride_min"] for lg in buses)
        worst = max((lg["worst"] for lg in buses if lg["worst"]), key=lambda s: LEVEL_RANK[s["level"]] * 1000 + s["load"],
                    default=None)
        mins_by = {}
        for lg in buses:
            for k, v in lg["minutes_by_level"].items():
                mins_by[k] = round(mins_by.get(k, 0) + v, 1)
        known = [lg["seat_chance"] for lg in buses if lg["seat_chance"] is not None]
        conf_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        for lg in buses:
            for k in ("_board", "_end"):
                lg.pop(k, None)
        return {"legs": legs, "total_min": round(t - t0), "arrive": _fmt(t), "depart": _fmt(t0),
                "walk_min": round(walk), "wait_min": round(wait), "ride_min": round(ride),
                "left_behind_extra_min": round(sum(lg["left_behind_extra"] for lg in buses), 1),
                "transfers": len(buses) - 1, "routes": [lg["route"] for lg in buses],
                "crowding": {"worst_level": worst["level"] if worst else "unknown",
                             "worst_label": LEVEL_LABEL[worst["level"] if worst else "unknown"],
                             "worst_stretch": f"{worst['from']} → {worst['to']}" if worst else None,
                             "minutes_by_level": mins_by,
                             "seat_chance": round(min(known), 2) if known else None,
                             "confidence": min((lg["confidence"] for lg in buses), key=lambda c: conf_order[c])},
                "_walk": walk, "_wait": wait, "_ride_legs": buses, "_transfers": len(buses) - 1, "_cand": cand}

    def _score(self, opt, comfort):
        s = opt["_walk"] * config.WALK_WEIGHT + opt["_wait"] * config.WAIT_WEIGHT
        s += opt["_transfers"] * config.TRANSFER_PENALTY_MIN
        for lg in opt["_ride_legs"]:
            for st in lg["stretches"]:
                mult = MULT.get(st["level"], config.UNKNOWN_MULT)
                s += st["min"] * (1 + comfort * (mult - 1))
        return s

    # ---------------------------------------------------------------- public
    def plan(self, origin, dest, depart: datetime, comfort=0.5, k=4, later=True):
        dt = daytype_of(depart)
        t0 = depart.hour * 60 + depart.minute
        O = self._access(*origin)
        D = self._access(*dest)
        if not O or not D:
            return {"options": [], "reason": "No bus stop within walking distance of the "
                                              + ("origin" if not O else "destination")}
        opts = []
        for cand in self._candidates(O, D):
            o = self._evaluate(cand, dt, t0, origin, dest)
            if o:
                opts.append(o)
        if not opts:
            return {"options": [], "reason": "No direct or one-transfer bus journey found"}
        # keep the best few by each objective, then rank by the user's preference
        fastest = sorted(opts, key=lambda o: o["total_min"])[:12]
        comfy = sorted(opts, key=lambda o: self._score(o, 1.0))[:12]
        pool = {tuple(o["routes"]): o for o in fastest + comfy}.values()
        pool = [o for o in pool if o["total_min"] <= min(x["total_min"] for x in opts) * 1.6 + 10]
        for o in pool:
            o["score"] = round(self._score(o, comfort), 1)
        ranked = sorted(pool, key=lambda o: o["score"])[:k]
        if ranked:
            f = min(ranked, key=lambda o: o["total_min"])
            c = min(ranked, key=lambda o: (LEVEL_RANK[o["crowding"]["worst_level"]], self._score(o, 1.0)))
            f.setdefault("tags", []).append("Fastest")
            if c is not f:
                c.setdefault("tags", []).append("Most comfortable")
            ranked[0].setdefault("tags", []).insert(0, "Best for you")
            for o in ranked:
                if o["crowding"]["worst_level"] == "unknown":
                    o.setdefault("tags", []).append("No crowding data")
        # "leave a bit later" options: same routes, later departure, if it is clearly less crowded
        extra = []
        if later and ranked:
            base = ranked[0]
            for dmin in (20, 40):
                o2 = self._evaluate(base["_cand"], dt, t0 + dmin, origin, dest)
                if o2 and LEVEL_RANK[o2["crowding"]["worst_level"]] < LEVEL_RANK[base["crowding"]["worst_level"]]:
                    o2["score"] = round(self._score(o2, comfort) + dmin * 0.5, 1)
                    o2["tags"] = [f"Leave {dmin} min later"]
                    extra.append(o2)
                    break
        # same top journey at a contrasting hour, so the user sees how much crowding depends on timing
        compare = None
        if ranked:
            alt_h = 13 if not 11 <= depart.hour <= 15 else 9
            q = self._evaluate(ranked[0]["_cand"], dt, alt_h * 60, origin, dest)
            if q:
                compare = {"hour": f"{alt_h:02d}:00", "total_min": q["total_min"], "wait_min": q["wait_min"],
                           "worst_level": q["crowding"]["worst_level"], "worst_label": q["crowding"]["worst_label"],
                           "seat_chance": q["crowding"]["seat_chance"]}
        result = ranked + extra
        for o in result:
            for key in ("_walk", "_wait", "_ride_legs", "_transfers", "_cand"):
                o.pop(key, None)
        return {"daytype": dt, "depart": _fmt(t0), "comfort": comfort, "options": result, "compare": compare}
