"""Videotips uitpluizen: elk voorwerp uit een snelle carrousel als genummerd beeld op overzichtsvellen.

Qmusic toont soms een filmpje waarin honderden voorwerpen razendsnel voorbijflitsen (bijv. ~10 per
seconde). Dit module knipt per voorwerp het scherpste beeld eruit, snijdt bij op het oplichtende
voorwerp en zet alles genummerd (met tijdstip) op vellen, zodat je ze rustig kunt doorlopen.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _kern(beeld: np.ndarray) -> np.ndarray:
    """Verkleinde grijswaardenkern van het midden van het beeld (voor verschilmetingen)."""
    h, w = beeld.shape[:2]
    stuk = beeld[int(h * 0.2) : int(h * 0.85), int(w * 0.15) : int(w * 0.95)]
    ys = np.linspace(0, stuk.shape[0] - 1, 56).astype(int)
    xs = np.linspace(0, stuk.shape[1] - 1, 40).astype(int)
    return stuk[np.ix_(ys, xs)].astype(float)


def kies_voorwerpen(kernen: np.ndarray, sprong: float = 6.0, min_helder: float = 90.0,
                    zelfde: float = 6.0) -> list[int]:
    """Indexen van één (rustigst) beeld per voorwerp.

    kernen: array (n, h, w) met grijswaarden. Een 'sprong' in het gemiddelde verschil tussen
    opeenvolgende beelden markeert een nieuw voorwerp; donkere en dubbele beelden vallen weg.
    """
    d = np.r_[0.0, np.abs(np.diff(kernen, axis=0)).mean(axis=(1, 2))]
    grenzen = list(np.where(d > sprong)[0]) + [len(kernen)]
    gekozen: list[int] = []
    start, vorige = 0, None
    for g in grenzen:
        if g - start >= 1:
            i = start + int(np.argmin(d[start + 1 : g])) + 1 if g - start > 1 else start
            k = kernen[i]
            if k.max() >= min_helder and (vorige is None or np.abs(k - vorige).mean() >= zelfde):
                gekozen.append(i)
                vorige = k
        start = g
    return gekozen


def maak_overzicht(video: str | Path, uitmap: str | Path, draai: int = 0, kolommen: int = 8,
                   rijen: int = 6, tegel: int = 230) -> tuple[int, list[Path]]:
    """Maak genummerde overzichtsvellen van alle voorwerpen in een video. Geeft (aantal, vellen)."""
    from PIL import Image, ImageDraw, ImageOps

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is nodig om video's te lezen")
    uitmap = Path(uitmap)
    uitmap.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video), "-q:v", "3",
                        f"{tmp}/%06d.jpg"], check=True)
        paden = sorted(Path(tmp).glob("*.jpg"))
        if not paden:
            raise RuntimeError("Geen beelden gevonden in de video")
        duur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                     "default=nw=1:nk=1", str(video)], capture_output=True, text=True).stdout or 0)
        fps = len(paden) / duur if duur > 0 else 30.0
        kernen = np.stack([_kern(np.asarray(Image.open(p).convert("L"))) for p in paden])
        gekozen = kies_voorwerpen(kernen)

        tegels = []
        for n, i in enumerate(gekozen):
            im = Image.open(paden[i])
            a = np.asarray(im.convert("L"), dtype=float)
            ys, xs = np.where(a > np.percentile(a, 99.3))
            cy, cx = (int(np.median(ys)), int(np.median(xs))) if len(ys) else (im.height // 2, im.width // 2)
            half = int(min(im.width, im.height) * 0.3)
            t = im.crop((max(0, cx - half), max(0, cy - half), min(im.width, cx + half), min(im.height, cy + half)))
            if draai:
                t = t.rotate(draai, expand=True)
            t = ImageOps.pad(t, (tegel, tegel), color=(0, 0, 0))
            dr = ImageDraw.Draw(t)
            dr.rectangle((0, 0, 100, 15), fill=(0, 0, 0))
            dr.text((2, 2), f"#{n} {i / fps:.1f}s", fill=(255, 255, 0))
            tegels.append(t)

    vellen = []
    per = kolommen * rijen
    for b in range(0, len(tegels), per):
        sub = tegels[b : b + per]
        vel = Image.new("RGB", (kolommen * tegel, ((len(sub) + kolommen - 1) // kolommen) * tegel))
        for j, t in enumerate(sub):
            vel.paste(t, ((j % kolommen) * tegel, (j // kolommen) * tegel))
        pad = uitmap / f"voorwerpen_{b // per + 1:02d}.jpg"
        vel.save(pad, quality=82)
        vellen.append(pad)
    return len(tegels), vellen
