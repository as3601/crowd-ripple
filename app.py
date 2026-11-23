"""Crowd Ripple API + web screens.  Run:  python app.py   then open http://127.0.0.1:8000"""
import json
import os
import re
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from crowd.planner import Planner
from crowd.ripple import Ripple
from crowd.store import Model

app = FastAPI(
    title="Crowd Ripple API",
    description="Crowdedness-aware bus journey planner for Delhi, built on October 2026 e-ticketing data and static GTFS. "
                "Crowding levels: seats / standing / crowded / packed. Ripple simulations show how a disruption "
                "spreads as crowding through the network.",
    version="1.0",
)
MODEL = Model()
PLANNER = Planner(MODEL)
RIPPLE = Ripple(MODEL)
STATIC = config.ROOT / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_LATLON = re.compile(r"^\s*(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)\s*$")


def _place(text):
    """'28.64,77.22' or a stop name -> (lat, lon, label)."""
    m = _LATLON.match(text or "")
    if m:
        return float(m.group(1)), float(m.group(3)), text
    hits = MODEL.search_stops(text, 1)
    if not hits:
        raise HTTPException(404, f"No stop matching '{text}'")
    return hits[0]["lat"], hits[0]["lon"], hits[0]["name"]


def _dt(dt):
    if dt not in config.DAYTYPES:
        raise HTTPException(400, f"daytype must be one of {config.DAYTYPES}")
    return dt


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/operator", include_in_schema=False)
def operator():
    return FileResponse(STATIC / "operator.html")


@app.get("/journeys", summary="Plan bus journeys ranked on time and expected crowding")
def journeys(
    from_: str = Query(..., alias="from", description="'lat,lon' or a stop name, e.g. 'Anand Vihar ISBT Terminal'"),
    to: str = Query(..., description="'lat,lon' or a stop name"),
    depart: Optional[str] = Query(None, description="ISO datetime, e.g. 2026-10-27T18:00 (default: now)"),
    comfort: float = Query(0.5, ge=0, le=1, description="0 = fastest only, 1 = avoid crowding as much as possible"),
    k: int = Query(4, ge=1, le=6),
):
    o, d = _place(from_), _place(to)
    when = datetime.fromisoformat(depart) if depart else datetime.now()
    res = PLANNER.plan(o[:2], d[:2], when, comfort=comfort, k=k)
    res.update({"from": {"lat": o[0], "lon": o[1], "label": o[2]}, "to": {"lat": d[0], "lon": d[1], "label": d[2]},
                "date": when.date().isoformat()})
    return res


@app.get("/api/meta", summary="Model facts: expansion factor, coverage, levels, capacities")
def meta():
    mt = MODEL.meta
    return {"expansion": round(mt["expansion"], 1), "ticketed_weekday": round(mt["ticketed_weekday"]),
            "system_daily": mt["system_daily"], "coverage": mt["coverage"], "built": mt["built"],
            "days_per_type": mt["days_per_type"], "holidays": config.HOLIDAYS, "daily_ticketed": mt["daily_ticketed"],
            "levels": [{"key": k, "label": lab, "multiplier": mu} for (k, lab, _r, mu) in config.LEVELS],
            "capacity": config.CAPACITY,
            "routes_with_crowding": sum(r["flow"] is not None for r in MODEL.routes.values()),
            "routes_total": len(MODEL.routes)}


@app.get("/api/stops", summary="Search stops by name")
def stops(q: str, limit: int = 8):
    return MODEL.search_stops(q, limit)


@app.get("/api/routes", summary="Search routes by name (e.g. 534UP)")
def routes(q: str = "", limit: int = 20):
    q = q.upper().strip()
    out = sorted({str(r["ticket_name"]) for r in MODEL.routes.values() if r["flow"] is not None and q in str(r["ticket_name"]).upper()},
                 key=lambda s: (not s.startswith(q), len(s), s))
    return out[:limit]


@app.get("/api/crowding/route/{route}", summary="Load per bus and level for every stretch and hour of a route")
def route_profile(route: str, daytype: str = "weekday"):
    rid = MODEL.find_route(route)
    if rid is None:
        raise HTTPException(404, f"Unknown route '{route}'")
    return MODEL.route_profile(rid, _dt(daytype))


@app.get("/api/crowding/network", summary="Busiest stretches citywide (for the map)")
def network(daytype: str = "weekday", hour: int = Query(9, ge=0, le=23), top: int = Query(3500, le=8000)):
    return MODEL.network(_dt(daytype), hour, top)


@app.get("/api/hotspots", summary="Where one extra bus removes the most crowding (passenger-minutes)")
def hotspots(daytype: str = "weekday", hour: int = Query(9, ge=0, le=23), top: int = 15):
    return MODEL.hotspots(_dt(daytype), hour, top)


@app.get("/api/ripple", summary="Ripple simulation: how a disruption spreads as crowding")
def ripple(event: str = Query(..., description="cancel | add | surge"), daytype: str = "weekday",
           hour: int = Query(18, ge=0, le=23), route: Optional[str] = None, lat: Optional[float] = None,
           lon: Optional[float] = None, riders: int = Query(2000, ge=0, le=50000), radius: int = Query(600, le=2000)):
    res = RIPPLE.simulate(event, _dt(daytype), hour, route=route, lat=lat, lon=lon, extra_riders=riders, radius_m=radius)
    if "error" in res:
        raise HTTPException(400, res["error"])
    return res


@app.get("/api/validation", summary="Checks against known patterns (from validation/validate.py)")
def validation():
    p = config.OUT_DIR / "validation.json"
    if not p.exists():
        return JSONResponse({"error": "run: python -m validation.validate"}, 404)
    return json.loads(p.read_text())


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)))
