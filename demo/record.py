"""Records the demo: drives the running app in headless Edge (Chrome DevTools Protocol) with a visible cursor,
captures every scene as a screen recording, and logs caption timings.

Usage:  python demo/record.py [--base http://127.0.0.1:8010] [--out demo/build]
Output: <out>/<scene>/frames/*.jpg, <out>/scenes.json  (frame timestamps + captions per scene)
"""
import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import websockets

W, H = 1440, 810
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT = 9333
HERE = Path(__file__).resolve().parent

CURSOR_JS = r"""
(() => { if (window.__cur) return;
  const c = document.createElement('div'); c.id='__cur';
  c.style.cssText='position:fixed;left:0;top:0;width:26px;height:26px;z-index:2147483647;pointer-events:none;transform:translate(720px,700px);transition:transform .75s cubic-bezier(.45,.05,.25,1)';
  c.innerHTML='<svg width="26" height="26" viewBox="0 0 24 24"><path d="M4 2 L4 19 L8.5 14.8 L11.6 21.6 L14.4 20.4 L11.4 13.8 L17.6 13.6 Z" fill="#111" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>';
  document.documentElement.appendChild(c); window.__cur=c; window.__pos=[720,700];
  window.__click=(x,y)=>{const r=document.createElement('div');r.style.cssText=`position:fixed;left:${x-18}px;top:${y-18}px;width:36px;height:36px;border-radius:50%;background:rgba(11,110,105,.28);z-index:2147483646;pointer-events:none;transform:scale(.3);transition:transform .35s ease-out,opacity .45s ease-out`;document.documentElement.appendChild(r);requestAnimationFrame(()=>{r.style.transform='scale(1.4)';r.style.opacity='0'});setTimeout(()=>r.remove(),600)};
})();
"""


class CDP:
    def __init__(self, ws):
        self.ws, self.n, self.pending, self.handlers = ws, 0, {}, {}

    async def reader(self):
        async for raw in self.ws:
            m = json.loads(raw)
            if "id" in m and m["id"] in self.pending:
                self.pending.pop(m["id"]).set_result(m)
            elif "method" in m and m["method"] in self.handlers:
                asyncio.ensure_future(self.handlers[m["method"]](m["params"]))   # never block the reader

    async def send(self, method, params=None):
        self.n += 1
        fut = asyncio.get_event_loop().create_future()
        self.pending[self.n] = fut
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params or {}}))
        r = await fut
        if "error" in r:
            raise RuntimeError(f"{method}: {r['error']}")
        return r.get("result", {})


class Recorder:
    def __init__(self, cdp, out: Path):
        self.c, self.out = cdp, out
        self.scenes, self.cur = [], None
        self.pos = (720, 700)
        cdp.handlers["Page.screencastFrame"] = self.on_frame

    async def on_frame(self, p):
        await self.c.send("Page.screencastFrameAck", {"sessionId": p["sessionId"]})
        if self.cur is not None:
            i = len(self.cur["frames"])
            f = self.cur["dir"] / f"{i:05d}.jpg"
            f.write_bytes(base64.b64decode(p["data"]))
            self.cur["frames"].append([time.perf_counter() - self.cur["t0"], f.name])

    # ---------------------------------------------------------------- scene control
    async def start(self, name):
        d = self.out / name / "frames"
        shutil.rmtree(d.parent, ignore_errors=True)
        d.mkdir(parents=True)
        self.cur = {"name": name, "dir": d, "frames": [], "captions": [], "t0": time.perf_counter()}
        await self.c.send("Page.startScreencast", {"format": "jpeg", "quality": 88, "maxWidth": W, "maxHeight": H,
                                                   "everyNthFrame": 1})
        await self.js("document.body.style.transform='translateZ(0)'")   # force a first frame

    async def stop(self, hold=0.6):
        await asyncio.sleep(hold)
        await self.c.send("Page.stopScreencast")
        s = self.cur
        s["duration"] = time.perf_counter() - s["t0"]
        self.scenes.append({"name": s["name"], "duration": s["duration"], "frames": s["frames"], "captions": s["captions"]})
        print(f"  {s['name']}: {s['duration']:.1f}s, {len(s['frames'])} frames", flush=True)
        self.cur = None

    def cap(self, text):
        t = time.perf_counter() - self.cur["t0"]
        self.cur["captions"].append([round(t, 2), text])

    # ---------------------------------------------------------------- page helpers
    async def js(self, expr, await_promise=True):
        r = await self.c.send("Runtime.evaluate", {"expression": expr, "awaitPromise": await_promise, "returnByValue": True})
        if "exceptionDetails" in r:
            raise RuntimeError(f"JS error: {r['exceptionDetails'].get('exception', {}).get('description', r['exceptionDetails'])}\n{expr[:200]}")
        return r.get("result", {}).get("value")

    async def goto(self, url, wait=2.5):
        await self.c.send("Page.navigate", {"url": url})
        await asyncio.sleep(wait)
        await self.js(CURSOR_JS)
        await self.js(f"window.__cur.style.transition='none';window.__cur.style.transform='translate({self.pos[0]}px,{self.pos[1]}px)';"
                      "void window.__cur.offsetWidth; window.__cur.style.transition='transform .75s cubic-bezier(.45,.05,.25,1)'")

    async def box(self, sel):
        b = await self.js(f"(()=>{{const e=document.querySelector({json.dumps(sel)}); if(!e) return null; e.scrollIntoView({{block:'nearest'}}); const r=e.getBoundingClientRect(); return [r.left,r.top,r.width,r.height]}})()")
        if not b:
            raise RuntimeError(f"no element {sel}")
        return b

    async def move(self, x, y, dur=0.75):
        await self.js(f"window.__cur.style.transitionDuration='{dur}s';window.__cur.style.transform='translate({x}px,{y}px)'")
        await self.c.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})
        self.pos = (x, y)
        await asyncio.sleep(dur + 0.08)

    async def move_to(self, sel, fx=0.5, fy=0.5, dur=0.75):
        x, y, w, h = await self.box(sel)
        await self.move(x + w * fx, y + h * fy, dur)

    async def click_at(self, x, y):
        await self.move(x, y)
        await self.js(f"window.__click({x},{y})")
        for t in ("mousePressed", "mouseReleased"):
            await self.c.send("Input.dispatchMouseEvent", {"type": t, "x": x, "y": y, "button": "left", "clickCount": 1})
            await asyncio.sleep(0.06)

    async def click(self, sel, fx=0.5, fy=0.5, wait=0.4):
        x, y, w, h = await self.box(sel)
        await self.click_at(x + w * fx, y + h * fy)
        await asyncio.sleep(wait)

    async def type(self, text, delay=0.085):
        for ch in text:
            await self.c.send("Input.insertText", {"text": ch})
            await asyncio.sleep(delay)

    async def wheel(self, sel, dy, steps=6):
        x, y, w, h = await self.box(sel)
        await self.move(x + w * 0.5, y + min(h * 0.5, 400), 0.5)
        for _ in range(steps):
            await self.c.send("Input.dispatchMouseEvent", {"type": "mouseWheel", "x": self.pos[0], "y": self.pos[1],
                                                           "deltaX": 0, "deltaY": dy / steps})
            await asyncio.sleep(0.07)
        await asyncio.sleep(0.4)

    async def drag(self, x0, y0, x1, y1, steps=24):
        await self.move(x0, y0)
        await self.c.send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": x0, "y": y0, "button": "left", "clickCount": 1})
        await self.js("window.__cur.style.transitionDuration='0s'")
        for i in range(1, steps + 1):
            x, y = x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps
            await self.js(f"window.__cur.style.transform='translate({x}px,{y}px)'")
            await self.c.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y, "button": "left", "buttons": 1})
            await asyncio.sleep(0.035)
        await self.c.send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": x1, "y": y1, "button": "left", "clickCount": 1})
        self.pos = (x1, y1)
        await self.js("window.__cur.style.transitionDuration='.75s'")

    async def wait_for(self, expr, timeout=15):
        t = time.time()
        while time.time() - t < timeout:
            if await self.js(expr):
                return True
            await asyncio.sleep(0.2)
        raise RuntimeError(f"timeout waiting for {expr}")


# ==================================================================== the script
async def scenes(r: Recorder, base):
    slides = (HERE / "slides.html").as_uri()
    val = json.loads(urllib.request.urlopen(base + "/api/validation").read())
    ho = val["holdout"]

    for name, dur in (("title", 5.2), ("problem", 7.5), ("how", 10.5)):
        await r.goto(f"{slides}#{name}", wait=0.2)
        await r.js("window.__cur.style.display='none'")
        await r.c.send("Page.reload", {})
        await asyncio.sleep(0.15)
        await r.start(name)
        await asyncio.sleep(dur)
        await r.stop(0)

    # ---------------------------------------------------------------- commuter planner, typed search
    await r.goto(base + "/", wait=3)
    await r.js("document.querySelector('#depart').value='2026-10-27T09:00'")
    await r.start("planner")
    r.cap("This is the commuter planner. Let's go from Anand Vihar to Connaught Place.")
    await r.click("#from", 0.3)
    await r.type("Anand Vihar ISBT T")
    await r.wait_for("!document.querySelector('#from').nextElementSibling.hidden")
    await asyncio.sleep(0.5)
    await r.click("#from + .ac-list div")
    await r.click("#to", 0.3)
    await r.type("Super Bazar C")
    await r.wait_for("!document.querySelector('#to').nextElementSibling.hidden")
    await asyncio.sleep(0.5)
    await r.click("#to + .ac-list div")
    r.cap("Leaving at 6 pm on a weekday, the evening peak.")
    await r.move_to("#depart", 0.35)
    await asyncio.sleep(0.4)
    await r.js("document.querySelector('#depart').value='2026-10-27T18:00'")
    await asyncio.sleep(1.0)
    await r.click("#go", wait=0.2)
    await r.wait_for("document.querySelectorAll('.opt').length>0")
    await asyncio.sleep(0.6)
    await r.wheel(".side", 470)
    r.cap("Each option shows the travel time and how crowded the bus usually is.")
    await r.move_to(".opt.sel .strip", 0.5, 0.5)
    await asyncio.sleep(1.4)
    r.cap("The strip is the ride, part by part: green seats, yellow standing, orange crowded, red packed.")
    await r.move_to(".opt.sel .strip", 0.25, 0.5, 1.6)
    await r.move_to(".opt.sel .strip", 0.85, 0.5, 1.6)
    await asyncio.sleep(0.8)
    r.cap("Pick an option to see it on the map, coloured by crowding, plus each leg with stops and times.")
    await asyncio.sleep(2.4)
    info = await r.js("(()=>{const o=last.options;const f=o.find(x=>(x.tags||[]).includes('Fastest'))||o[0];const c=o.find(x=>(x.tags||[]).includes('Most comfortable'));return {f:f.total_min,fl:f.crowding.worst_label,c:c?c.total_min:null,cl:c?c.crowding.worst_label:null,ci:c?o.indexOf(c):-1}})()")
    if info and info["ci"] >= 0:
        await r.wheel(".side", 230)
        await r.click(f".opt[data-i='{info['ci']}'] .head", 0.2, 0.5, wait=0.8)
        extra = info["c"] - info["f"]
        r.cap(f"The fastest bus takes {info['f']} min ({info['fl'].lower()} at worst). This one takes "
              f"{info['c']} min, {abs(extra)} min {'more' if extra >= 0 else 'less'}, with "
              f"{info['cl'].lower()} all the way.")
        await asyncio.sleep(3.8)
    await r.wheel(".side", -2000, 8)
    r.cap("Slide towards 'Avoid crowds' and the options re-rank, so crowded minutes count for more.")
    x, y, w, h = await r.box("#comfort")
    await r.drag(x + w * 0.5, y + h / 2, x + w - 4, y + h / 2)
    await r.wait_for("last && last.comfort==1")
    await asyncio.sleep(0.8)
    await r.wheel(".side", 470)
    await asyncio.sleep(2.6)
    await r.wheel(".side", 2600, 10)
    r.cap("It also compares the same trip at a quieter hour, because timing often matters more than the route.")
    await r.move_to(".note", 0.5, 0.5)
    await asyncio.sleep(3.4)
    await r.stop()

    # ---------------------------------------------------------------- second example: morning peak, leg details
    await r.goto(base + "/", wait=3)
    await r.start("planner2")
    r.cap("Morning peak: Kashmere Gate to Nehru Place at 8:30, with one change.")
    await r.click("#examples .chip:nth-child(2)", wait=0.2)
    await r.wait_for("document.querySelectorAll('.opt').length>0")
    await asyncio.sleep(0.8)
    await r.wheel(".side", 480)
    r.cap("Each bus shows how often it comes, the expected wait and the chance it arrives full.")
    await r.move_to(".opt.sel .legs", 0.4, 0.45)
    await asyncio.sleep(3.2)
    r.cap("If buses are often full, the wait includes the buses that pass you by. Crowding costs time too.")
    await r.move_to(".opt.sel .facts", 0.3, 0.5)
    await asyncio.sleep(3.6)
    await r.stop()

    # ---------------------------------------------------------------- operator: network by hour
    await r.goto(base + "/operator", wait=4)
    await r.js("document.querySelector('#hour').value=6; document.querySelector('#hourl').textContent='06:00'; refresh()")
    await asyncio.sleep(2)
    await r.start("operator")
    r.cap("The operator console shows the whole network: the busiest stretches, coloured by crowding.")
    await r.move(1000, 420, 1.0)
    await asyncio.sleep(1.5)
    r.cap("Moving through the day: crowding builds to the morning peak, eases off midday, then returns in the evening.")
    await r.move_to("#hour", 0.3, 0.5)
    for h in range(7, 21):
        x, y, w, hh = await r.box("#hour")
        await r.move(x + w * (h / 23), y + hh / 2, 0.25)
        await r.js(f"document.querySelector('#hour').value={h}; document.querySelector('#hourl').textContent='{h:02d}:00'; refresh()")
        await asyncio.sleep(0.75 if h not in (9, 18) else 1.6)
    await asyncio.sleep(0.6)
    await r.stop()

    # ---------------------------------------------------------------- hotspots + route heatmap
    await r.js("document.querySelector('#hour').value=9; document.querySelector('#hourl').textContent='09:00'; refresh()")
    await asyncio.sleep(1.5)
    await r.start("hotspots")
    r.cap("Where would one more bus help most? Routes ranked by crowded passenger-minutes it would remove.")
    await r.wheel(".side", 420)
    await r.move_to("#hot tr.click:nth-child(2)", 0.5, 0.5)
    await asyncio.sleep(2.2)
    r.cap("Open a route to see every stretch at every hour. The red bands are its peaks.")
    await r.click("#hot tr.click:nth-child(2) td", 0.5, 0.5, wait=1.6)
    await r.wheel(".side", 300)
    await r.move_to("#heat", 0.55, 0.4)
    await asyncio.sleep(3.6)
    await r.stop()

    # ---------------------------------------------------------------- ripple: breakdown
    await r.wheel(".side", -3000, 6)
    await r.js("document.querySelector('#hour').value=18; document.querySelector('#hourl').textContent='18:00'; refresh()")
    await asyncio.sleep(1.5)
    await r.start("ripple")
    r.cap("The Ripple simulator: what happens next if a bus breaks down on route 525 at 6 pm?")
    await r.click(".tabs button[data-p='ripple']", wait=0.6)
    await r.move_to("#route", 0.3)
    await asyncio.sleep(0.6)
    await r.click("#run", wait=0.3)
    await r.wait_for("document.querySelectorAll('.impact').length>0")
    await asyncio.sleep(0.8)
    await r.wheel(".side", 520)
    r.cap("1st order: the next bus has to carry both loads, so riders are left behind and the queue takes time to clear.")
    await r.move_to(".impact.o1", 0.5, 0.5)
    await asyncio.sleep(3.8)
    await r.wheel(".side", 420)
    r.cap("2nd order: riders spill onto parallel routes. 3rd order: the same bus also misses its return trip.")
    await r.move_to(".impact.o2", 0.5, 0.5)
    await asyncio.sleep(3.6)
    r.cap("Each effect is mapped and comes with an action for the control room.")
    await r.move(1050, 430, 0.9)
    await asyncio.sleep(2.6)
    await r.stop()

    # ---------------------------------------------------------------- ripple: surge
    await r.wheel(".side", -3000, 6)
    await r.start("surge")
    r.cap("Or a surge: a metro line stops near Kashmere Gate and 3,000 extra people need a bus.")
    x, y, w, h = await r.box("#ev")
    await r.move(x + w * 0.5, y + h / 2)
    await r.js("document.querySelector('#ev').value='surge'; document.querySelector('#ev').onchange()")
    await asyncio.sleep(0.7)
    await r.click("[data-w='28.66750,77.22850']", wait=0.4)
    await r.click("#run", wait=0.3)
    await r.wait_for("document.querySelectorAll('.impact').length>0")
    await asyncio.sleep(0.6)
    await r.wheel(".side", 560)
    r.cap("It shows which routes can take the extra riders, which get packed, and how many can't board at all.")
    await r.move_to(".impact.o1", 0.5, 0.5)
    await asyncio.sleep(4.2)
    await r.stop()

    # ---------------------------------------------------------------- checks
    await r.wheel(".side", -3000, 6)
    await r.start("checks")
    r.cap("Checks against known patterns: peaks at 9 am and 6 pm, quieter weekends, dips on the two holidays.")
    await r.click(".tabs button[data-p='val']", wait=1.2)
    await r.wheel(".side", 330)
    await r.move_to("#val .card:nth-child(1)", 0.5, 0.6)
    await asyncio.sleep(3.0)
    await r.wheel(".side", 520)
    await r.move_to("#val .card:nth-child(3)", 0.5, 0.5)
    await asyncio.sleep(2.4)
    r.cap(f"Hold-out test: three weeks of data predict the fourth. Same crowding level for "
          f"{round(ho['level_exact_rider_weighted'] * 100)}% of riders, within one level for "
          f"{round(ho['level_within_one_rider_weighted'] * 100)}%.")
    await r.wheel(".side", 420)
    await r.move_to("#val .card:nth-child(4)", 0.5, 0.5)
    await asyncio.sleep(4.2)
    await r.stop()

    # ---------------------------------------------------------------- API
    await r.goto(base + "/docs", wait=3.5)
    await r.start("api")
    r.cap("Everything runs on an open, documented API that other apps can use.")
    await r.move(500, 380, 0.9)
    await asyncio.sleep(1.4)
    await r.click("#operations-default-journeys_journeys_get .opblock-summary", 0.3, 0.5, wait=1.0)
    await r.wheel("body", 260)
    await asyncio.sleep(1.6)
    await r.stop()
    await r.goto(base + "/journeys?from=Anand%20Vihar%20ISBT%20Terminal&to=Super%20Bazar%20Conn.%20Place&depart=2026-10-27T18:00", wait=3.5)
    await r.start("json")
    r.cap("GET /journeys returns each option as JSON, with crowding for every stretch.")
    await r.move(420, 300, 0.9)
    await asyncio.sleep(1.2)
    await r.wheel("body", 500, 8)
    await asyncio.sleep(1.6)
    await r.stop()

    await r.goto(f"{slides}?v1={ho['load_spearman']}&v2={round(ho['level_within_one_rider_weighted'] * 100)}%25#end", wait=0.2)
    await r.js("window.__cur.style.display='none'")
    await r.c.send("Page.reload", {})
    await asyncio.sleep(0.15)
    await r.start("end")
    await asyncio.sleep(6.5)
    await r.stop(0)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8010")
    ap.add_argument("--out", default=str(HERE / "build"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prof = out / "edge-profile"
    proc = subprocess.Popen([EDGE, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={prof}",
                             f"--window-size={W},{H}", "--hide-scrollbars", "--force-color-profile=srgb",
                             "--no-first-run", "--disable-extensions", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json").read())
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=50_000_000) as ws:
            c = CDP(ws)
            reader = asyncio.create_task(c.reader())
            for m in ("Page.enable", "Runtime.enable"):
                await c.send(m)
            await c.send("Emulation.setDeviceMetricsOverride", {"width": W, "height": H, "deviceScaleFactor": 1, "mobile": False})
            await c.send("Emulation.setEmulatedMedia", {"features": [{"name": "prefers-color-scheme", "value": "light"}]})
            r = Recorder(c, out)
            await scenes(r, a.base)
            (out / "scenes.json").write_text(json.dumps({"w": W, "h": H, "scenes": r.scenes}, indent=0))
            reader.cancel()
    finally:
        proc.terminate()


if __name__ == "__main__":
    asyncio.run(main())
