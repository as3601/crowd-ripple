"""Edits the recorded scenes into the final demo video (< 10 MB).

Usage: python demo/build_video.py [--build demo/build] [--out demo/CrowdRipple_demo.mp4] [--mb 9.3]
Needs ffmpeg/ffprobe on PATH (with libass).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FPS = 25
OW, OH = 1280, 720
XF = 0.45            # cross-fade between scenes (s)
NO_CAPTIONS = {"title", "problem", "how", "end"}


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-3000:])
        raise SystemExit(f"failed: {' '.join(map(str, cmd[:6]))} ...")
    return r


def scene_clip(sc, build: Path):
    d = build / sc["name"]
    frames = sc["frames"]
    lst = d / "frames.txt"
    lines = []
    for i, (t, f) in enumerate(frames):
        start = 0 if i == 0 else t
        end = frames[i + 1][0] if i + 1 < len(frames) else sc["duration"]
        lines.append(f"file 'frames/{f}'\nduration {max(end - start, 0.001):.4f}")
    lines.append(f"file 'frames/{frames[-1][1]}'")
    lst.write_text("\n".join(lines))
    out = d / "clip.mp4"
    if out.exists() and out.stat().st_mtime > lst.stat().st_mtime - 1 and "--reuse" in sys.argv:
        return out
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-vf", f"fps={FPS},scale={OW}:{OH}:flags=lanczos,format=yuv420p", "-t", f"{sc['duration']:.3f}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", str(out)])
    return out


def ts(t):
    h, r = divmod(max(t, 0), 3600)
    m, s = divmod(r, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def write_ass(scenes, offsets, path: Path):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OW}
PlayResY: {OH}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Segoe UI Semibold,29,&H00FFFFFF,&H00FFFFFF,&H28161616,&H28161616,0,0,0,0,100,100,0,0,3,9,0,2,150,150,34,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    for sc, off in zip(scenes, offsets):
        if sc["name"] in NO_CAPTIONS:
            continue
        caps = sc["captions"]
        for i, (t, text) in enumerate(caps):
            start = off + max(t, 0.15)
            end = off + (caps[i + 1][0] - 0.12 if i + 1 < len(caps) else sc["duration"] - XF - 0.1)
            if end - start < 1.2:
                end = start + 1.2
            ev.append(f"Dialogue: 0,{ts(start)},{ts(end)},Cap,,0,0,0,,{{\\fad(180,180)}}{text}")
    path.write_text(head + "\n".join(ev) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=str(HERE / "build"))
    ap.add_argument("--out", default=str(HERE / "CrowdRipple_demo.mp4"))
    ap.add_argument("--mb", type=float, default=9.4)
    ap.add_argument("--reuse", action="store_true", help="reuse scene clips already built")
    a = ap.parse_args()
    build = Path(a.build)
    meta = json.loads((build / "scenes.json").read_text())
    scenes = meta["scenes"]
    print("clips ...", flush=True)
    clips = [scene_clip(sc, build) for sc in scenes]
    durs = [sc["duration"] for sc in scenes]
    offsets, acc = [], 0.0
    for i, d in enumerate(durs):
        offsets.append(acc)
        acc += d - XF
    total = acc + XF
    print(f"total {total:.1f}s", flush=True)

    # cross-fade chain
    inputs, fc, prev = [], [], "0:v"
    for c in clips:
        inputs += ["-i", str(c)]
    for i in range(1, len(clips)):
        o = f"v{i}"
        fc.append(f"[{prev}][{i}:v]xfade=transition=fade:duration={XF}:offset={offsets[i]:.3f}[{o}]")
        prev = o
    ass = build / "captions.ass"
    write_ass(scenes, offsets, ass)
    ass_arg = str(ass).replace("\\", "/").replace(":", "\\:")
    fc.append(f"[{prev}]ass='{ass_arg}'[vout]")
    joined = build / "joined.mp4"
    print("compositing ...", flush=True)
    run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(fc), "-map", "[vout]",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p", str(joined)])

    music = build / "music.wav"
    run([sys.executable, str(HERE / "music.py"), f"{total:.2f}", str(music)])

    # two-pass encode to the size budget
    abr = 64
    vbr = int(a.mb * 8 * 1024 * 1024 / total / 1000 * 0.97 - abr)
    print(f"video {vbr} kb/s + audio {abr} kb/s", flush=True)
    # screen content: slow preset, adaptive quantisation and long GOPs keep text sharp at a low bitrate
    common = ["-c:v", "libx264", "-preset", "veryslow", "-b:v", f"{vbr}k", "-maxrate", f"{vbr * 3}k",
              "-bufsize", f"{vbr * 6}k", "-x264-params",
              "aq-mode=3:aq-strength=0.9:ref=6:bframes=5:keyint=500:min-keyint=25:deblock=-1,-1",
              "-pix_fmt", "yuv420p", "-r", "20"]
    log = str(build / "x264pass")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(joined), *common, "-pass", "1", "-passlogfile", log, "-an", "-f", "mp4", "NUL"])
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(joined), "-i", str(music), *common, "-pass", "2", "-passlogfile", log,
         "-map", "0:v", "-map", "1:a", "-af", "volume=0.42", "-c:a", "aac", "-b:a", f"{abr}k", "-shortest",
         "-movflags", "+faststart", a.out])
    size = Path(a.out).stat().st_size / 1024 / 1024
    print(f"done: {a.out}  {size:.2f} MB  {total:.1f}s")


if __name__ == "__main__":
    main()
