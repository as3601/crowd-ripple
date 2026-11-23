"""Checks the crowding model against known patterns and against itself across weeks.

Run:  python -m validation.validate      (after python -m pipeline.build)
Writes data/processed/validation.json and docs/VALIDATION.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from crowd import flows  # noqa: E402
from crowd.store import Model  # noqa: E402

KEYS = [k for (k, *_r) in config.LEVELS]


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def level_idx(load, cap):
    c = config.CAPACITY[cap]
    th = np.array([c["seats"] * 0.85, c["nominal"] * 0.75, c["nominal"]])
    return np.searchsorted(th, load, side="left")


def main():
    m = Model()
    meta = m.meta
    dates = meta["dates"]
    dtype = meta["date_type"]
    t = dict(np.load(config.OUT_DIR / "tickets_mapped.npz"))
    E = meta["expansion"]
    out = {}

    # ---------------------------------------------------------------- 1. time-of-day peaks
    df = pd.DataFrame({"day": t["day"], "hour": t["hour"], "pax": t["pax"]})
    df["type"] = [dtype[dates[d]] for d in df["day"]]
    ndays = pd.Series(dtype).value_counts()
    hourly = (df.groupby(["type", "hour"])["pax"].sum().unstack(0).fillna(0) / ndays).fillna(0)
    wk = hourly["weekday"]
    am = int(wk.loc[5:11].idxmax())
    pm = int(wk.loc[15:22].idxmax())
    midday = float(wk.loc[12:15].mean())
    out["peaks"] = {"am_peak_hour": am, "pm_peak_hour": pm,
                    "am_peak_to_midday": round(float(wk[am]) / midday, 2),
                    "pm_peak_to_midday": round(float(wk[pm]) / midday, 2),
                    "hourly_ticketed_per_day": {c: [round(float(x), 1) for x in hourly[c].reindex(range(24), fill_value=0)]
                                                for c in hourly.columns}}

    # ---------------------------------------------------------------- 2. weekday vs weekend vs holidays
    daily = pd.Series(meta["daily_ticketed"])
    by_type = daily.groupby(pd.Series(dtype)).mean()
    out["daily"] = {"mean_by_type": {k: round(float(v)) for k, v in by_type.items()},
                    "saturday_vs_weekday": round(float(by_type["saturday"] / by_type["weekday"]), 2),
                    "sunday_vs_weekday": round(float(by_type["sunday"] / by_type["weekday"]), 2),
                    "holidays": {d: {"name": config.HOLIDAYS[d], "ticketed": int(daily[d]),
                                     "vs_weekday": round(float(daily[d] / by_type["weekday"]), 2)} for d in config.HOLIDAYS},
                    "series": {d: int(v) for d, v in daily.items()}}

    # ---------------------------------------------------------------- 3. crowding share by hour (rider-weighted)
    share = {}
    for dt_i, dt in enumerate(config.DAYTYPES):
        rows = []
        for h in range(24):
            tot = np.zeros(4)
            for r in m.routes.values():
                if r["flow"] is None:
                    continue
                f = r["flow"][dt_i, h]
                np.add.at(tot, r["lvl"][dt_i, h], f)
            rows.append([round(float(x / max(tot.sum(), 1)), 3) for x in tot])
        share[dt] = rows
    out["level_share_by_hour"] = share
    crowded_9 = sum(share["weekday"][9][2:])
    crowded_14 = sum(share["weekday"][14][2:])
    out["checks"] = {"crowded_or_packed_share_9am": round(crowded_9, 3),
                     "crowded_or_packed_share_2pm": round(crowded_14, 3),
                     "crowded_or_packed_share_6pm": round(sum(share["weekday"][18][2:]), 3),
                     "crowded_or_packed_share_sun_9am": round(sum(share["sunday"][9][2:]), 3)}

    # ---------------------------------------------------------------- 4. busiest corridors at the morning peak
    net = m.network("weekday", am, 40)
    seen, corr = set(), []
    for s in net:
        key = tuple(sorted((s["from"], s["to"])))
        if key in seen:
            continue
        seen.add(key)
        corr.append({"stretch": f"{s['from']} → {s['to']}", "riders_per_hour": s["riders"], "worst_route": s["route"],
                     "level": s["level"]})
    out["busiest_stretches_am"] = corr[:12]

    # ---------------------------------------------------------------- 5. week-to-week stability + hold-out test
    day_week = np.array([pd.Timestamp(d).isocalendar().week for d in dates])
    weekdays = np.array([dtype[d] == "weekday" for d in dates])
    weeks = sorted(int(x) for x in set(day_week[weekdays]))
    weeks = [w for w in weeks if ((day_week == w) & weekdays).sum() >= 3]
    train_days = np.flatnonzero(weekdays & np.isin(day_week, weeks[:-1]))
    test_days = np.flatnonzero(weekdays & (day_week == weeks[-1]))
    nd = len(dates)
    rt_week = {w: [] for w in weeks}
    pred, obs, wts, caps = [], [], [], []
    for rid, idx in flows.by_route(t):
        r = m.routes[rid]
        n = len(r["stops"])
        f = flows.route_flows(t["o"][idx], t["d"][idx], t["day"][idx], t["hour"][idx], t["pax"][idx], n, nd)
        bus = np.maximum(r["buses"][0], 1e-6)
        for w in weeks:
            dsel = np.flatnonzero(weekdays & (day_week == w))
            rt_week[w].append(f[dsel].sum(axis=2).mean(axis=0))      # route-hour rider-stretches
        tr = f[train_days].mean(axis=0) * E / bus
        te = f[test_days].mean(axis=0) * E / bus
        mask = (tr > 0) | (te > 0)
        pred.append(tr[mask]); obs.append(te[mask]); wts.append((tr + te)[mask]); caps.append(np.full(mask.sum(), r["cap"] == "9m"))
    pred, obs, wts, nine = map(np.concatenate, (pred, obs, wts, caps))
    pl = np.where(nine, level_idx(pred, "9m"), level_idx(pred, "12m"))
    ol = np.where(nine, level_idx(obs, "9m"), level_idx(obs, "12m"))
    w = wts / wts.sum()
    out["holdout"] = {
        "train": f"weekdays of ISO weeks {weeks[:-1]}", "test": f"weekdays of ISO week {weeks[-1]}",
        "stretch_hours": int(len(pred)),
        "load_spearman": round(spearman(pred, obs), 3),
        "load_mae_rider_weighted": round(float((np.abs(pred - obs) * w).sum()), 1),
        "mean_load_rider_weighted": round(float((obs * w).sum()), 1),
        "level_exact_rider_weighted": round(float(((pl == ol) * w).sum()), 3),
        "level_within_one_rider_weighted": round(float(((np.abs(pl - ol) <= 1) * w).sum()), 3),
        "level_exact_unweighted": round(float((pl == ol).mean()), 3),
    }
    W = {wk: np.concatenate([x.ravel() for x in v]) for wk, v in rt_week.items()}
    pairs = {}
    for i, a in enumerate(weeks):
        for b in weeks[i + 1:]:
            msk = (W[a] > 0) | (W[b] > 0)
            pairs[f"{a}-{b}"] = round(spearman(W[a][msk], W[b][msk]), 3)
    out["week_to_week_spearman_route_hour"] = pairs

    # ---------------------------------------------------------------- 6. confidence coverage
    conf = {"high": 0.0, "medium": 0.0, "low": 0.0}
    for r in m.routes.values():
        if r["flow"] is None:
            continue
        for h in range(24):
            tk = float(r["tickets"][0, h])
            fl = float(r["flow"][0, h].sum())
            c = "high" if tk >= 150 else "medium" if tk >= 40 else "low"
            conf[c] += fl
    tot = sum(conf.values())
    out["confidence_rider_weighted"] = {k: round(v / tot, 3) for k, v in conf.items()}

    # ---------------------------------------------------------------- 7. sensitivity to the expansion assumption
    sens = {}
    for system in (3_000_000, 4_000_000, 5_000_000):
        s = system / config.SYSTEM_DAILY_BOARDINGS
        tot = np.zeros(4)
        for r in m.routes.values():
            if r["flow"] is None:
                continue
            f = r["flow"][0, am] * s
            listed = np.maximum(r["buses"][0, am] - r["bus_shortfall"][0, am], 1e-6)   # schedule before the capacity fix
            bus = np.maximum(listed, r["flow"][0, am].max() * s / config.CAPACITY[r["cap"]]["crush"])
            lv = level_idx(f / bus, r["cap"])
            np.add.at(tot, lv, f)
        sens[f"{system / 1e5:.0f} lakh/day"] = {k: round(float(v / tot.sum()), 3) for k, v in zip(KEYS, tot)}
    out["sensitivity_am_peak_level_share"] = sens
    out["coverage"] = {k: round(v / meta["coverage"]["tickets_total_pax"], 3) for k, v in meta["coverage"].items()}
    out["stop_match_methods"] = meta.get("stop_match_methods", {})
    out["expansion"] = round(E, 1)

    (config.OUT_DIR / "validation.json").write_text(json.dumps(out, indent=1))
    write_md(out)
    print(json.dumps({k: out[k] for k in ("peaks", "checks", "holdout", "week_to_week_spearman_route_hour",
                                          "confidence_rider_weighted", "sensitivity_am_peak_level_share")}, indent=1)
          .replace('"hourly_ticketed_per_day"', '"hourly(omitted)"')[:4000])


def write_md(v):
    p, d, h, c = v["peaks"], v["daily"], v["holdout"], v["checks"]
    L = ["# Validation report", "",
         "Generated by `python -m validation.validate` from the October 2026 e-ticketing data. "
         "Every number below is recomputed on each run.", "",
         "## 1. Peaks are where Delhi expects them", "",
         f"- Morning peak: **{p['am_peak_hour']:02d}:00–{p['am_peak_hour'] + 1:02d}:00**, "
         f"{p['am_peak_to_midday']}× the midday (12–16h) rate.",
         f"- Evening peak: **{p['pm_peak_hour']:02d}:00–{p['pm_peak_hour'] + 1:02d}:00**, {p['pm_peak_to_midday']}× midday.",
         f"- Share of rider-stretches that are *crowded or packed* on weekdays: **{c['crowded_or_packed_share_9am']:.0%} at 9am**, "
         f"{c['crowded_or_packed_share_6pm']:.0%} at 6pm, {c['crowded_or_packed_share_2pm']:.0%} at 2pm; Sunday 9am: "
         f"{c['crowded_or_packed_share_sun_9am']:.0%}.", "",
         "## 2. Weekday vs weekend vs holidays", "",
         "| Day type | Ticketed riders / day |", "|---|---|"]
    L += [f"| {k} | {v:,} |" for k, v in d["mean_by_type"].items()]
    L += ["", f"Saturday is {d['saturday_vs_weekday']:.0%} and Sunday {d['sunday_vs_weekday']:.0%} of a weekday."]
    L += [f"- {x['name']} ({k}): {x['ticketed']:,} tickets, {x['vs_weekday']:.0%} of a normal weekday "
          "(excluded from the profiles)." for k, x in d["holidays"].items()]
    L += ["", f"## 3. Busiest stretches at the {p['am_peak_hour']:02d}:00 weekday peak", "",
          "| Stretch | Riders / hour (all routes) | Most crowded route | Level |", "|---|---|---|---|"]
    L += [f"| {s['stretch']} | {s['riders_per_hour']:,} | {s['worst_route']} | {s['level']} |" for s in v["busiest_stretches_am"]]
    L += ["", "## 4. Does it hold from one week to the next?", "",
          f"Hold-out test: build loads from {h['train']}, predict {h['test']}, on {h['stretch_hours']:,} route-stretch-hours.", "",
          "| Measure | Value |", "|---|---|",
          f"| Rank correlation of load per bus (Spearman) | {h['load_spearman']} |",
          f"| Mean abs. error, rider-weighted (people per bus) | {h['load_mae_rider_weighted']} (mean load {h['mean_load_rider_weighted']}) |",
          f"| Same crowding level, rider-weighted | {h['level_exact_rider_weighted']:.0%} |",
          f"| Within one level, rider-weighted | {h['level_within_one_rider_weighted']:.0%} |", "",
          "Week-to-week rank correlation of route-hour ridership (weekdays):", ""]
    L += [f"- weeks {k}: {val}" for k, val in v["week_to_week_spearman_route_hour"].items()]
    cf = v["confidence_rider_weighted"]
    L += ["", "## 5. How much to trust each estimate", "",
          f"Rider-weighted share of weekday estimates with high / medium / low confidence (≥150 / ≥40 / <40 real tickets "
          f"behind the route-hour in the month): **{cf['high']:.0%} / {cf['medium']:.0%} / {cf['low']:.0%}**.", "",
          "## 6. Sensitivity to the unticketed-rider assumption", "",
          f"Morning-peak level shares if Delhi's buses carry 30, 40 (default) or 50 lakh boardings a day "
          f"(expansion factor now {v['expansion']}):", "", "| Assumption | Seats | Standing | Crowded | Packed |", "|---|---|---|---|---|"]
    L += [f"| {k} | {x['seats']:.0%} | {x['standing']:.0%} | {x['crowded']:.0%} | {x['packed']:.0%} |"
          for k, x in v["sensitivity_am_peak_level_share"].items()]
    cv = v["coverage"]
    L += ["", "## 7. Data coverage", "",
          f"- Tickets on a route found in GTFS: {cv['route_matched_pax']:.0%}",
          f"- Tickets with both stops placed on the route: {cv['stop_matched_pax']:.0%} (the crowding model uses these)",
          f"- Stop-name matching method counts: {v['stop_match_methods']}", ""]
    (config.ROOT / "docs").mkdir(exist_ok=True)
    (config.ROOT / "docs" / "VALIDATION.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
