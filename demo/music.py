"""Generates a calm lo-fi background track (no samples, no licences): electric-piano chords, bass, soft drums.

Usage: python demo/music.py <seconds> <out.wav>
"""
import sys
import wave

import numpy as np

SR = 44100
BPM = 84
BEAT = 60 / BPM
rng = np.random.default_rng(7)


def note(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def epiano(freq, dur, vel=0.5):
    t = np.arange(int(dur * SR)) / SR
    env = np.exp(-t * 2.2) * (1 - np.exp(-t * 300))
    tine = np.exp(-t * 9)
    trem = 1 + 0.12 * np.sin(2 * np.pi * 4.8 * t)
    x = (np.sin(2 * np.pi * freq * t) + 0.32 * np.sin(2 * np.pi * 2 * freq * t) * tine
         + 0.12 * np.sin(2 * np.pi * 3.01 * freq * t) * tine * tine)
    return vel * x * env * trem


def pad(freqs, dur, vel=0.08):
    t = np.arange(int(dur * SR)) / SR
    env = np.minimum(t / 1.2, 1) * np.minimum((dur - t) / 1.0, 1).clip(0)
    x = sum(np.sin(2 * np.pi * f * t + d) + np.sin(2 * np.pi * f * 1.004 * t) for f, d in zip(freqs, (0, 1, 2, 3)))
    return vel * x * env / len(freqs)


def bass(freq, dur, vel=0.42):
    t = np.arange(int(dur * SR)) / SR
    env = (1 - np.exp(-t * 80)) * np.exp(-t * 1.3)
    return vel * (np.sin(2 * np.pi * freq * t) + 0.18 * np.sin(4 * np.pi * freq * t)) * env


def kick(vel=0.55):
    t = np.arange(int(0.35 * SR)) / SR
    f = 46 + 70 * np.exp(-t * 28)
    return vel * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)


def hat(vel=0.05):
    n = int(0.06 * SR)
    x = np.diff(rng.standard_normal(n + 1))
    return vel * x * np.exp(-np.arange(n) / SR * 70)


def snare(vel=0.12):
    n = int(0.18 * SR)
    t = np.arange(n) / SR
    return vel * (rng.standard_normal(n) * 0.6 + np.sin(2 * np.pi * 190 * t)) * np.exp(-t * 22)


def add(buf, x, at, pan=0.0):
    i = int(at * SR)
    if i >= buf.shape[0]:
        return
    x = x[: buf.shape[0] - i]
    buf[i:i + len(x), 0] += x * (1 - pan) ** 0.5
    buf[i:i + len(x), 1] += x * (1 + pan) ** 0.5


def main(seconds, out):
    buf = np.zeros((int((seconds + 3) * SR), 2))
    # Fmaj7 - Em7 - Dm7 - Cmaj7(9) ; one chord per bar (4 beats)
    chords = [[53, 57, 60, 64], [52, 55, 59, 62], [50, 53, 57, 60], [48, 52, 55, 59, 62]]
    roots = [41, 40, 38, 36]
    melody = [[72, None, 69, None, 67, None, None, 64], [71, None, 67, None, 64, None, 62, None],
              [69, None, 65, None, 64, None, None, None], [67, None, 64, None, 62, None, 64, None]]
    bar = 4 * BEAT
    nbars = int(seconds / bar) + 2
    for b in range(nbars):
        t0 = b * bar
        c = b % 4
        intro = b < 2
        add(buf, pad([note(m) for m in chords[c]], bar + 0.6), t0)
        for k, (hit, vel) in enumerate([(0, 0.34), (1.5, 0.22), (2.5, 0.26)]):
            sw = 0.03 if k else 0
            for j, m in enumerate(chords[c]):
                add(buf, epiano(note(m), bar, vel * 0.32), t0 + hit * BEAT + sw + j * 0.012, pan=-0.25 + 0.15 * j)
        add(buf, bass(note(roots[c]), 2.4 * BEAT), t0)
        add(buf, bass(note(roots[c] + 7), 1.4 * BEAT, 0.3), t0 + 2.5 * BEAT)
        if not intro:
            for beat in range(4):
                if beat in (0, 2):
                    add(buf, kick(), t0 + beat * BEAT)
                if beat in (1, 3):
                    add(buf, snare(), t0 + beat * BEAT + 0.01, pan=0.1)
                for e in (0, 0.5):
                    add(buf, hat(0.05 if e == 0 else 0.03), t0 + (beat + e) * BEAT + (0.06 if e else 0), pan=0.3)
        if b >= 4 and (b // 4) % 2 == 1:      # a gentle melody every other phrase
            for s, m in enumerate(melody[c]):
                if m:
                    add(buf, epiano(note(m), 1.6, 0.12), t0 + s * BEAT / 2 + 0.02, pan=0.2)
    # simple room: two short feedback-free delays
    wet = np.zeros_like(buf)
    for d, g in ((0.031, 0.25), (0.047, 0.2), (0.089, 0.14), (0.137, 0.09)):
        k = int(d * SR)
        wet[k:, 0] += buf[:-k, 1] * g
        wet[k:, 1] += buf[:-k, 0] * g
    buf = buf + wet
    # warm lowpass (one-pole) and soft clip
    a = 0.35
    for ch in range(2):
        y = buf[:, ch]
        out_ = np.empty_like(y)
        acc = 0.0
        # vectorised one-pole via lfilter-like recurrence (scipy optional)
        try:
            from scipy.signal import lfilter
            out_ = lfilter([a], [1, a - 1], y)
        except Exception:
            for i, v in enumerate(y):
                acc += a * (v - acc)
                out_[i] = acc
        buf[:, ch] = out_
    buf = buf[: int(seconds * SR)]
    fade_in, fade_out = int(1.5 * SR), int(3.5 * SR)
    buf[:fade_in] *= np.linspace(0, 1, fade_in)[:, None]
    buf[-fade_out:] *= np.linspace(1, 0, fade_out)[:, None] ** 1.5
    buf = np.tanh(buf / np.abs(buf).max() * 1.2) * 0.85
    pcm = (buf * 32767).astype(np.int16)
    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main(float(sys.argv[1]), sys.argv[2])
