"""End-to-end pipeline: raw tickets + static GTFS -> crowding model (data/processed/model.pkl + crowding.sqlite).

Run:  python -m pipeline.build
"""
import difflib
import json
import os
import pickle
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from crowd import flows, levels  # noqa: E402
from pipeline.gtfs import build_patterns, norm_route  # noqa: E402

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def norm_stop(s):
    s = str(s).lower()
    s = re.sub(r"\b(bus stop|bus stand|stop)\b", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def load_tickets():
    df = pd.read_csv(config.RAW_TICKETS, na_values=["NULL"],
                     dtype={"vehicle_number": "string", "route": "string"})
    df["ts"] = pd.to_datetime(df["created_at"].str[:19])
    df["date"] = df["ts"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["ts"].dt.hour.astype(np.int8)
    df["dow"] = df["ts"].dt.dayofweek
    return df


def daytype_of(date_str):
    dow = pd.Timestamp(date_str).dayofweek
    return "saturday" if dow == 5 else "sunday" if dow == 6 else "weekday"


def map_routes(df, patterns):
    by_name = {}
    for rid, p in patterns.items():
        k = norm_route(p["name"])
        if k not in by_name or p["n_trips"] > patterns[by_name[k]]["n_trips"]:
            by_name[k] = rid
    keys = df["route"].map(norm_route)
    return keys.map(by_name)


def main_part(s):
    """'Mahipal Pur (Nh8)' -> 'mahipalpur';  'Hamdard Nagar / Sangam Vihar' -> 'hamdardnagar'."""
    return norm_stop(re.split(r"[(/]", str(s))[0])


def _hav_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))


class StopResolver:
    """Finds where a ticket's stop name sits on a route's GTFS pattern.

    Ladder: exact name -> main-part / containment -> fuzzy name -> geographic snap (the name is found
    anywhere in the GTFS stop list, and the nearest stop on this route within SNAP_M is used).
    """
    SNAP_M = 700

    def __init__(self, patterns, stops):
        self.patterns = patterns
        self.stops = stops
        self.names = stops["stop_name"].to_dict()
        g = stops.assign(n=stops.stop_name.map(norm_stop), m=stops.stop_name.map(main_part))
        self.coords_by_norm = g.groupby("n")[["stop_lat", "stop_lon"]].apply(lambda x: x.to_numpy()).to_dict()
        self.coords_by_main = g.groupby("m")[["stop_lat", "stop_lon"]].apply(lambda x: x.to_numpy()).to_dict()
        self.all_norm = list(self.coords_by_norm)
        self._geo = {}
        self._pat = {}
        self.how = {"exact": 0, "partial": 0, "fuzzy": 0, "geo": 0, "none": 0}

    def coords(self, name):
        if name in self._geo:
            return self._geo[name]
        n, m = norm_stop(name), main_part(name)
        c = self.coords_by_norm.get(n)
        if c is None:
            c = self.coords_by_main.get(m)
        if c is None:
            hit = difflib.get_close_matches(n, self.all_norm, n=1, cutoff=0.85)
            c = self.coords_by_norm[hit[0]] if hit else None
        self._geo[name] = c
        return c

    def pattern(self, rid):
        if rid not in self._pat:
            ids = self.patterns[rid]["stops"]
            self._pat[rid] = ([norm_stop(self.names.get(s, "")) for s in ids],
                              [main_part(self.names.get(s, "")) for s in ids],
                              self.stops.loc[ids, ["stop_lat", "stop_lon"]].to_numpy())
        return self._pat[rid]

    def __call__(self, rid, name):
        normed, mains, xy = self.pattern(rid)
        n, m = norm_stop(name), main_part(name)
        if n in normed:
            self.how["exact"] += 1
            return normed.index(n)
        for i, (pn, pm) in enumerate(zip(normed, mains)):
            if (m and m == pm) or (len(m) >= 6 and (m in pn or (len(pm) >= 6 and pm in n))):
                self.how["partial"] += 1
                return i
        hit = difflib.get_close_matches(n, normed, n=1, cutoff=0.78)
        if hit:
            self.how["fuzzy"] += 1
            return normed.index(hit[0])
        c = self.coords(name)
        if c is not None:
            dist = _hav_m(xy[:, 0][None, :], xy[:, 1][None, :], c[:, 0][:, None], c[:, 1][:, None]).min(axis=0)
            i = int(dist.argmin())
            if dist[i] <= self.SNAP_M:
                self.how["geo"] += 1
                return i
        self.how["none"] += 1
        return None


def map_stops(df, patterns):
    """Position of the boarding and alighting stop on the route's GTFS pattern."""
    resolver = StopResolver(patterns, STOPS)
    cache = {}

    def lookup(rid, name):
        key = (rid, name)
        if key not in cache:
            cache[key] = resolver(rid, name)
        return cache[key]

    pairs = df[["rid", "start_stop_name", "end_stop_name"]].drop_duplicates()
    log(f"matching {len(pairs):,} unique (route, stop pair) combinations")
    pairs["o"] = [lookup(r, s) for r, s in zip(pairs.rid, pairs.start_stop_name)]
    pairs["d"] = [lookup(r, s) for r, s in zip(pairs.rid, pairs.end_stop_name)]
    log(f"stop match methods (unique route-stop names): {resolver.how}")
    MATCH_HOW.update(resolver.how)
    return df.merge(pairs, on=["rid", "start_stop_name", "end_stop_name"], how="left")


def main():
    global STOPS
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("reading GTFS")
    patterns, STOPS = build_patterns()
    log(f"{len(patterns):,} route patterns")

    log("reading tickets")
    df = load_tickets()
    dates = sorted(df["date"].unique())
    date_idx = {d: i for i, d in enumerate(dates)}
    df["day"] = df["date"].map(date_idx).astype(np.int16)
    date_type = {d: ("holiday" if d in config.HOLIDAYS else daytype_of(d)) for d in dates}

    # ---- expansion factor: ticketed sample -> all riders
    daily = df.groupby("date")["passenger_count"].sum()
    normal_weekdays = [d for d in dates if date_type[d] == "weekday"]
    ticketed_weekday = float(daily[normal_weekdays].mean())
    system = float(os.environ.get("CR_SYSTEM_DAILY_BOARDINGS", config.SYSTEM_DAILY_BOARDINGS))
    expansion = system / ticketed_weekday
    log(f"ticketed riders per weekday {ticketed_weekday:,.0f} -> expansion factor {expansion:.1f}")

    # ---- route-level table for ALL ticket routes (operator heatmaps, incl. routes missing from GTFS)
    df["daytype"] = df["date"].map(date_type)
    rh = (df.groupby(["route", "daytype", "hour"])["passenger_count"].sum().rename("pax").reset_index())
    ndays = pd.Series(date_type).value_counts()
    rh["boardings_per_day"] = rh["pax"] / rh["daytype"].map(ndays) * expansion

    # ---- link tickets to GTFS patterns
    df["rid"] = map_routes(df, patterns)
    matched = df.dropna(subset=["rid"]).copy()
    matched["rid"] = matched["rid"].astype(int)
    matched = map_stops(matched, patterns)
    ok = matched.dropna(subset=["o", "d"]).copy()
    ok["o"] = ok["o"].astype(int)
    ok["d"] = ok["d"].astype(int)
    forward = ok["o"] < ok["d"]
    ok = ok[forward]
    total = df["passenger_count"].sum()
    coverage = {
        "tickets_total_pax": int(total),
        "route_matched_pax": int(matched["passenger_count"].sum()),
        "stop_matched_pax": int(ok["passenger_count"].sum()),
        "dropped_wrong_direction_pax": int(matched.loc[matched.o.notna() & matched.d.notna(), "passenger_count"].sum()
                                           - ok["passenger_count"].sum()),
    }
    log("coverage " + json.dumps({k: f"{v / total:.1%}" for k, v in coverage.items()}))

    np.savez_compressed(config.OUT_DIR / "tickets_mapped.npz",
                        route=ok["rid"].to_numpy(np.int32), o=ok["o"].to_numpy(np.int16),
                        d=ok["d"].to_numpy(np.int16), day=ok["day"].to_numpy(np.int16),
                        hour=ok["hour"].to_numpy(np.int8), pax=ok["passenger_count"].to_numpy(np.float32),
                        is_ac=ok["is_ac"].to_numpy(bool))

    # observed distinct buses per route / day / hour (a lower bound on buses actually running)
    veh = (ok.dropna(subset=["vehicle_number"]).groupby(["rid", "day", "hour"])["vehicle_number"]
           .nunique().rename("veh").reset_index())

    # ---- per-route crowding arrays
    t = {k: ok[c].to_numpy() for k, c in
         [("route", "rid"), ("o", "o"), ("d", "d"), ("day", "day"), ("hour", "hour"), ("pax", "passenger_count")]}
    ac_share = ok.groupby("rid")["is_ac"].mean().to_dict()
    route_names_raw = ok.groupby("rid")["route"].agg(lambda s: s.mode().iat[0]).to_dict()
    dt_days = {dt: np.array([date_idx[d] for d in dates if date_type[d] == dt]) for dt in config.DAYTYPES}
    nd = len(dates)
    model_routes = {}
    veh_g = {r: g for r, g in veh.groupby("rid")}
    for rid, idx in flows.by_route(t):
        p = patterns[rid]
        n = len(p["stops"])
        f = flows.route_flows(t["o"][idx], t["d"][idx], t["day"][idx], t["hour"][idx], t["pax"][idx], n, nd)
        obs = np.zeros((nd, 24), dtype=np.float32)
        if rid in veh_g:
            g = veh_g[rid]
            obs[g["day"].to_numpy(), g["hour"].to_numpy()] = g["veh"].to_numpy()
        boards = np.zeros((nd, 24), dtype=np.float32)
        np.add.at(boards, (t["day"][idx], t["hour"][idx]), t["pax"][idx])
        mean = np.zeros((3, 24, n - 1), np.float32)
        std = np.zeros_like(mean)
        tickets = np.zeros((3, 24), np.float32)
        buses = np.zeros((3, 24, n - 1), np.float32)
        for i, dt in enumerate(config.DAYTYPES):
            days = dt_days[dt]
            mean[i] = f[days].mean(axis=0) * expansion
            std[i] = f[days].std(axis=0) * expansion
            tickets[i] = boards[days].sum(axis=0)
            obs_dt = obs[days].mean(axis=0)   # distinct buses seen in tickets: a lower bound, never scaled up
            sched = p["buses"][:, : n - 1]
            buses[i] = np.maximum(sched, obs_dt[:, None])
        buses = np.where((buses < 1) & (mean > 0), 1, buses)
        cap = "9m" if str(route_names_raw.get(rid, "")).upper().startswith("D-") else "12m"
        # Physical consistency: tickets are people who DID board, so the buses that ran must have had room
        # for them. If the implied load beats crush capacity, more buses ran than the (2023) schedule lists.
        crush = config.CAPACITY[cap]["crush"]
        listed = buses.max(axis=2)
        needed = mean.max(axis=2) / crush
        shortfall = np.maximum(needed - listed, 0).astype(np.float32)
        buses = np.maximum(buses, needed[..., None])
        # empirical-Bayes style shrinkage of sparse route-hours toward the route's usual load
        raw = mean / np.maximum(buses, 1e-6)
        tot = mean.sum(axis=(1, 2))
        prior = np.where(tot > 0, (mean * raw).sum(axis=(1, 2)) / np.maximum(tot, 1e-6), 0)
        w = (tickets / (tickets + config.SHRINK_K))[..., None]
        load = np.where(mean > 0, w * raw + (1 - w) * prior[:, None, None], 0).astype(np.float32)
        model_routes[rid] = {
            "name": p["name"], "ticket_name": route_names_raw.get(rid, p["name"]), "stops": p["stops"],
            "cum": p["cum"], "flow": mean, "flow_std": std, "buses": buses, "tickets": tickets,
            "cap": cap, "ac_share": float(ac_share.get(rid, 0)), "bus_shortfall": shortfall, "load": load,
        }
    # routes in GTFS without tickets still help the planner (travel time only, crowding unknown)
    for rid, p in patterns.items():
        if rid not in model_routes:
            n = len(p["stops"])
            model_routes[rid] = {"name": p["name"], "ticket_name": p["name"], "stops": p["stops"], "cum": p["cum"],
                                 "flow": None, "flow_std": None,
                                 "buses": np.repeat(p["buses"][None, :, : n - 1], 3, axis=0),
                                 "tickets": np.zeros((3, 24), np.float32), "cap": "12m", "ac_share": 0.0}
    log(f"{sum(r['flow'] is not None for r in model_routes.values()):,} routes with crowding, "
        f"{len(model_routes):,} routes in planner")

    meta = {"expansion": expansion, "ticketed_weekday": ticketed_weekday, "system_daily": system,
            "dates": dates, "date_type": date_type, "days_per_type": {k: int(v) for k, v in ndays.items()},
            "coverage": coverage, "stop_match_methods": MATCH_HOW, "built": time.strftime("%Y-%m-%d %H:%M"),
            "daily_ticketed": {d: int(v) for d, v in daily.items()}}
    with open(config.OUT_DIR / "model.pkl", "wb") as fh:
        pickle.dump({"meta": meta, "routes": model_routes, "stops": STOPS}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # ---- queryable export: one row per route / stretch / day type / hour
    log("writing crowding.sqlite")
    rows = []
    sn = STOPS["stop_name"].to_dict()
    for rid, r in model_routes.items():
        if r["flow"] is None:
            continue
        load = r["load"]
        for i, dt in enumerate(config.DAYTYPES):
            hs, ks = np.nonzero(r["flow"][i] > 0)
            for h, k in zip(hs, ks):
                L = float(load[i, h, k])
                rows.append((r["ticket_name"], rid, dt, int(h), int(k), sn.get(r["stops"][k]), sn.get(r["stops"][k + 1]),
                             round(float(r["flow"][i, h, k]), 1), round(float(r["buses"][i, h, k]), 1), round(L, 1),
                             levels.level_for(L, r["cap"])["key"], int(r["tickets"][i, h])))
    db = config.OUT_DIR / "crowding.sqlite"
    db.unlink(missing_ok=True)
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE segment_crowding(route TEXT, gtfs_route_id INT, daytype TEXT, hour INT, seq INT,
                   from_stop TEXT, to_stop TEXT, riders_per_hour REAL, buses_per_hour REAL, load_per_bus REAL,
                   level TEXT, tickets_in_month INT)""")
    con.executemany("INSERT INTO segment_crowding VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.execute("CREATE INDEX ix_seg ON segment_crowding(route, daytype, hour)")
    rh.to_sql("route_hour_boardings", con, index=False)
    con.commit()
    con.close()
    log(f"done: {len(rows):,} stretch-hour rows")


STOPS = None
MATCH_HOW = {}
if __name__ == "__main__":
    main()
