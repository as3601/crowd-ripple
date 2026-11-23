"""Loads the built model and answers fast questions about stops, routes and crowding."""
import math
import pickle
from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree

import config
from crowd import levels

DT_INDEX = {d: i for i, d in enumerate(config.DAYTYPES)}
_LAT0 = 28.6


def _xy(lat, lon):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return np.c_[lon * 111320 * math.cos(math.radians(_LAT0)), lat * 110540]


def walk_min(meters):
    return meters * config.WALK_DETOUR / (config.WALK_SPEED_KMH * 1000 / 60)


class Model:
    def __init__(self, path=config.OUT_DIR / "model.pkl"):
        with open(path, "rb") as fh:
            m = pickle.load(fh)
        self.meta, self.routes, stops = m["meta"], m["routes"], m["stops"]
        self.stop_ids = stops.index.to_numpy()
        self.stop_row = {int(s): i for i, s in enumerate(self.stop_ids)}
        self.stop_name = stops["stop_name"].astype(str).to_dict()
        self.lat = stops["stop_lat"].to_numpy()
        self.lon = stops["stop_lon"].to_numpy()
        self.tree = cKDTree(_xy(self.lat, self.lon))
        # stop -> [(route_id, position)]
        self.serving = {}
        for rid, r in self.routes.items():
            for pos, s in enumerate(r["stops"]):
                self.serving.setdefault(s, []).append((rid, pos))
        self.by_name = {}
        for rid, r in self.routes.items():
            for nm in {r["name"].upper(), str(r["ticket_name"]).upper()}:
                self.by_name.setdefault(nm, rid)
        self._precompute()

    # ------------------------------------------------------------------ geometry
    def coord(self, sid):
        i = self.stop_row[sid]
        return float(self.lat[i]), float(self.lon[i])

    def near(self, lat, lon, radius_m):
        idx = self.tree.query_ball_point(_xy([lat], [lon])[0], radius_m)
        out = []
        for i in idx:
            sid = int(self.stop_ids[i])
            if sid in self.serving:
                d = float(np.hypot(*(_xy([self.lat[i]], [self.lon[i]])[0] - _xy([lat], [lon])[0])))
                out.append((sid, d))
        return sorted(out, key=lambda x: x[1])

    @lru_cache(maxsize=20000)
    def transfer_neighbours(self, sid):
        lat, lon = self.coord(sid)
        return tuple(self.near(lat, lon, config.MAX_TRANSFER_WALK_M))

    def search_stops(self, q, limit=8):
        q = q.strip().lower()
        if len(q) < 2:
            return []
        seen, out = set(), []
        for sid, nm in self.stop_name.items():
            if sid in self.serving and q in nm.lower() and nm.lower() not in seen:
                seen.add(nm.lower())
                lat, lon = self.coord(sid)
                out.append({"stop_id": sid, "name": nm, "lat": lat, "lon": lon,
                            "score": (not nm.lower().startswith(q), len(nm))})
        out.sort(key=lambda x: x["score"])
        for o in out:
            o.pop("score")
        return out[:limit]

    def find_route(self, name):
        rid = self.by_name.get(str(name).upper())
        if rid is None and str(name).isdigit() and int(name) in self.routes:
            rid = int(name)
        return rid

    # ------------------------------------------------------------------ crowding
    def _precompute(self):
        """Per route: load per bus, level index, discomfort; per (daytype, hour): add-a-bus relief ranking."""
        th = {cap: np.array(levels.thresholds(cap)) for cap in config.CAPACITY}
        mults = np.array([m for (_, _, _, m) in config.LEVELS])
        for r in self.routes.values():
            cum = r["cum"]
            seg = np.clip(np.diff(cum, axis=1), 0.2, None)               # [24, nseg] minutes per stretch
            r["seg_min"] = seg
            if r["flow"] is None:
                continue
            b = np.maximum(r["buses"], 1e-6)
            load = r["load"] if r.get("load") is not None else r["flow"] / b      # shrunk in the pipeline
            r["lvl"] = np.searchsorted(th[r["cap"]], load, side="left").astype(np.int8)
            # discomfort = extra perceived passenger-minutes compared with everyone seated
            r["discomfort"] = (r["flow"] * seg[None] * (mults[r["lvl"]] - 1)).sum(axis=2)          # [3, 24]
            load1 = load * b / (b + 1)
            lvl1 = np.searchsorted(th[r["cap"]], load1, side="left")
            r["relief"] = r["discomfort"] - (r["flow"] * seg[None] * (mults[lvl1] - 1)).sum(axis=2)

    def leg_crowding(self, rid, i, j, dt, t_min):
        """Crowding on route `rid` from position i to j, boarding at minute-of-day t_min."""
        r = self.routes[rid]
        d = DT_INDEX[dt]
        h0 = int(t_min // 60) % 24
        cum = r["cum"][h0]
        stretches, total_seat, total, weighted = [], 0.0, 0.0, 0.0
        has_data = r["flow"] is not None
        for k in range(i, j):
            h = int((t_min + cum[k] - cum[i]) // 60) % 24
            mins = float(max(cum[k + 1] - cum[k], 0.2))
            if has_data:
                L = float(r["load"][d, h, k])
                sd = float(r["flow_std"][d, h, k] / max(r["buses"][d, h, k], 1e-6))
                lv = levels.level_for(L, r["cap"])
                p_stand = levels.p_over(L, config.CAPACITY[r["cap"]]["seats"], sd)
            else:
                L, sd, lv, p_stand = None, 0.0, None, None
            stretches.append({"k": k, "from": self.stop_name.get(r["stops"][k]), "to": self.stop_name.get(r["stops"][k + 1]),
                              "min": round(mins, 1), "load": None if L is None else round(L, 1),
                              "level": lv["key"] if lv else "unknown"})
            total += mins
            if lv:
                weighted += mins * lv["mult"]
                total_seat += mins * (1 - p_stand)
            else:
                weighted += mins * config.UNKNOWN_MULT
        conf = levels.confidence(float(r["tickets"][d, h0])) if has_data else "none"
        worst = max((s for s in stretches if s["load"] is not None), key=lambda s: s["load"], default=None)
        mins_by = {}
        for s in stretches:
            mins_by[s["level"]] = round(mins_by.get(s["level"], 0) + s["min"], 1)
        return {"stretches": stretches, "ride_min": round(total, 1), "felt_min": round(weighted, 1),
                "worst": worst, "minutes_by_level": mins_by, "confidence": conf,
                "seat_chance": round(total_seat / total, 2) if has_data and total else None}

    def wait_at(self, rid, i, dt, t_min):
        """Average wait for route `rid` at position i, including the 'crowd ripple': full buses passing you by."""
        r = self.routes[rid]
        d = DT_INDEX[dt]
        h = int(t_min // 60) % 24
        b = float(r["buses"][d, h, min(i, r["buses"].shape[2] - 1)])
        if b < 0.5:
            return None
        headway = 60 / b
        base = min(headway / 2, config.MAX_WAIT_MIN)
        p_full = 0.0
        if r["flow"] is not None:
            L = float(r["load"][d, h, i])
            sd = float(r["flow_std"][d, h, i] / max(r["buses"][d, h, i], 1e-6))
            p_full = min(levels.p_over(L, config.CAPACITY[r["cap"]]["crush"], sd), 0.9)
        extra = p_full / (1 - p_full) * headway       # expected number of full buses that pass you x headway
        return {"headway": round(headway, 1), "wait": round(base + extra, 1), "base_wait": round(base, 1),
                "left_behind_extra": round(extra, 1), "p_full": round(p_full, 2)}

    @lru_cache(maxsize=128)
    def network(self, dt, hour, top=3500):
        """Busiest stretches citywide for the map: riders/hour summed over routes, worst level among them."""
        d = DT_INDEX[dt]
        agg = {}
        for rid, r in self.routes.items():
            if r["flow"] is None:
                continue
            f = r["flow"][d, hour]
            lv = r["lvl"][d, hour]
            st = r["stops"]
            for k in np.flatnonzero(f > 0):
                key = (st[k], st[k + 1])
                ld = float(r["load"][d, hour, k])
                a = agg.get(key)
                if a is None:
                    agg[key] = [float(f[k]), float(f[k]) * ld, r["ticket_name"], ld, int(lv[k])]
                else:
                    a[0] += float(f[k])
                    a[1] += float(f[k]) * ld
                    if lv[k] > a[4] or (lv[k] == a[4] and ld > a[3]):
                        a[2], a[3], a[4] = r["ticket_name"], ld, int(lv[k])
        items = sorted(agg.items(), key=lambda kv: -kv[1][0])[:top]
        keys = [lv[0] for lv in config.LEVELS]
        out = []
        for (a, b), (riders, wload, route, load, wl) in items:
            avg = wload / riders                      # what the typical rider on this stretch experiences
            out.append({"a": [round(v, 5) for v in self.coord(a)], "b": [round(v, 5) for v in self.coord(b)],
                        "riders": round(riders), "level": levels.level_for(avg)["key"], "avg_load": round(avg),
                        "route": route, "load": round(load), "worst_level": keys[wl],
                        "from": self.stop_name.get(a), "to": self.stop_name.get(b)})
        return out

    @lru_cache(maxsize=128)
    def hotspots(self, dt, hour, top=15):
        d = DT_INDEX[dt]
        rows = []
        for rid, r in self.routes.items():
            if r["flow"] is None or r["relief"][d, hour] <= 0:
                continue
            k = int(np.argmax(r["load"][d, hour]))
            rows.append({"route": r["ticket_name"], "route_id": rid,
                         "relief_pax_min": round(float(r["relief"][d, hour])),
                         "discomfort_pax_min": round(float(r["discomfort"][d, hour])),
                         "peak_load": round(float(r["load"][d, hour, k])),
                         "buses_per_hour": round(float(r["buses"][d, hour, k]), 1),
                         "worst_from": self.stop_name.get(r["stops"][k]), "worst_to": self.stop_name.get(r["stops"][k + 1]),
                         "confidence": levels.confidence(float(r["tickets"][d, hour]))})
        rows.sort(key=lambda x: -x["relief_pax_min"])
        return rows[:top]

    def route_profile(self, rid, dt):
        r = self.routes[rid]
        d = DT_INDEX[dt]
        keys = [lv[0] for lv in config.LEVELS]
        prof = {"route": r["ticket_name"], "route_id": rid, "cap": r["cap"], "ac_share": round(r["ac_share"], 2),
                "stops": [{"id": s, "name": self.stop_name.get(s), "lat": self.coord(s)[0], "lon": self.coord(s)[1]}
                          for s in r["stops"]]}
        if r["flow"] is None:
            prof["has_crowding"] = False
            return prof
        prof.update({"has_crowding": True,
                     "load": np.round(r["load"][d], 1).tolist(),
                     "level": [[keys[v] for v in row] for row in r["lvl"][d]],
                     "buses": np.round(r["buses"][d].max(axis=1), 1).tolist(),
                     "tickets": r["tickets"][d].astype(int).tolist(),
                     "confidence": [levels.confidence(float(x)) for x in r["tickets"][d]]})
        return prof
