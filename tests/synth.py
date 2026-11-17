"""Synthetische testgeluiden met bekende eigenschappen."""

from __future__ import annotations

import numpy as np

SR = 48_000
rng = np.random.default_rng(42)


def _tik(sr=SR, duur=0.012, helder=4000.0, amp=1.0):
    n = int(sr * duur)
    t = np.arange(n) / sr
    ruis = rng.standard_normal(n)
    # licht filteren richting gewenste helderheid
    from scipy.signal import butter, sosfilt

    sos = butter(2, [max(200, helder / 3), min(sr / 2 - 100, helder * 2)], btype="band", fs=sr, output="sos")
    return amp * sosfilt(sos, ruis) * np.exp(-t / (duur / 4))


def _plaats(totaal_s, events, sr=SR):
    y = np.zeros(int(totaal_s * sr))
    for t0, x in events:
        i = int(t0 * sr)
        y[i : i + len(x)] += x[: len(y) - i]
    return y


def stuiterbal():
    """Bal die stuitert: tussenpozen x0.75, kracht x0.8."""
    t, ioi, amp, ev = 0.1, 0.45, 1.0, []
    while ioi > 0.03 and t < 2.5:
        ev.append((t, _tik(helder=1500, duur=0.02, amp=amp)))
        t += ioi
        ioi *= 0.75
        amp *= 0.8
    return _plaats(2.6, ev)


def klok():
    return _plaats(4.2, [(0.1 + i * 0.5, _tik(helder=3000)) for i in range(8)])


def rits():
    """~35 tikjes per seconde gedurende 1,2 s."""
    ev, t = [], 0.05
    while t < 1.25:
        ev.append((t, _tik(helder=5000, duur=0.006, amp=0.6 + 0.4 * rng.random())))
        t += 1 / 35 * (0.85 + 0.3 * rng.random())
    return _plaats(1.35, ev)


def glas_ping():
    """Inharmonische, lang nagalmende partialen (glas/metaal)."""
    t = np.arange(int(SR * 2.0)) / SR
    y = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t / d)
            for f, a, d in [(2100, 1.0, 0.6), (5230, 0.5, 0.4), (8770, 0.3, 0.25)])
    y[: int(0.002 * SR)] *= np.linspace(0, 1, int(0.002 * SR))
    return y


def witte_ruis(duur=3.0):
    return 0.3 * rng.standard_normal(int(SR * duur))


def brom(duur=5.0):
    t = np.arange(int(SR * duur)) / SR
    return sum(0.5 / k * np.sin(2 * np.pi * 100 * k * t) for k in range(1, 6))


def fles_vullen(duur=3.0):
    """Ruis door een resonantie die van 300 naar 1200 Hz stijgt."""
    from scipy.signal import lfilter

    n = int(SR * duur)
    x = rng.standard_normal(n)
    y = np.zeros(n)
    blok = 1024
    zi = np.zeros(2)
    for i in range(0, n, blok):
        f = 300 * (1200 / 300) ** (i / n)
        r = 0.995
        w = 2 * np.pi * f / SR
        b, a = [1 - r], [1, -2 * r * np.cos(w), r * r]
        y[i : i + blok], zi = lfilter(b, a, x[i : i + blok], zi=zi)
    return y / np.max(np.abs(y)) * 0.8


def twee_delen():
    """Laag 'bonk' gevolgd door hoog gesis."""
    t = np.arange(int(SR * 0.4)) / SR
    bonk = np.sin(2 * np.pi * 90 * t) * np.exp(-t / 0.08)
    from scipy.signal import butter, sosfilt

    sis = sosfilt(butter(4, 6000, btype="high", fs=SR, output="sos"), rng.standard_normal(int(SR * 1.2))) * 0.5
    return _plaats(2.4, [(0.05, bonk), (0.9, sis)])
