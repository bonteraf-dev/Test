"""Vergelijken met echte referentie-opnames uit online geluidsbibliotheken.

Bron: BBC Sound Effects (https://sound-effects.bbcrewind.co.uk), ruim 30.000 opnames met beschrijving.
Gebruik valt onder de RemArc-licentie van de BBC: persoonlijk, educatief of onderzoek. De bestanden
worden alleen lokaal gecachet (~/.cache/geluidspeurder/referenties) en niet verspreid.

Per referentie-opname worden de luidste gebeurtenissen gezocht en precies zo uitgeknipt als het
mysteriegeluid (zelfde lengte, klap op dezelfde plek). Elk venster wordt vergeleken op:
  - AI-gelijkenis (CLAP audio <-> audio), als het model beschikbaar is
  - tijdsverloop: de luidheidscurve (aanloop, klap, naklank)
  - spectrum: de klankkleur (mel-spectrum)
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio as audio_mod

BBC_ZOEK = "https://sound-effects-api.bbcrewind.co.uk/api/sfx/search"
BBC_MEDIA = "https://sound-effects-media.bbcrewind.co.uk/mp3/{id}.mp3"
CACHE = Path.home() / ".cache" / "geluidspeurder" / "referenties"


@dataclass
class Referentie:
    bron: str
    id: str
    beschrijving: str
    zoekterm: str
    pad: Path | None = None


@dataclass
class Match:
    referentie: Referentie
    tijd: float           # waar in de referentie het beste venster begint (s)
    totaal: float
    ai: float | None
    verloop: float
    spectrum: float
    naklank: int = 0          # aantal blijvende smalle metaaltonen na de klap in het referentie-venster
    naklank_db: float = 0.0   # sterkste prominentie daarvan


def zoek_bbc(zoekterm: str, aantal: int = 6, max_duur_s: float = 120) -> list[Referentie]:
    body = {"criteria": {"from": 0, "size": aantal * 2, "tags": None, "categories": None, "durations": None,
                         "continents": None, "sortBy": None, "source": None, "recordist": None, "habitat": None,
                         "query": zoekterm}}
    req = urllib.request.Request(BBC_ZOEK, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    uit = []
    for x in data.get("results", []):
        if x.get("duration", 0) / 1000 > max_duur_s:
            continue
        uit.append(Referentie("bbc", str(x["id"]), x.get("description", ""), zoekterm))
        if len(uit) >= aantal:
            break
    return uit


def download(ref: Referentie, cache: Path = CACHE) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    pad = cache / f"{ref.bron}_{ref.id}.mp3"
    if not pad.exists() or pad.stat().st_size == 0:
        urllib.request.urlretrieve(BBC_MEDIA.format(id=ref.id), pad)
    ref.pad = pad
    return pad


def _omhullende_db(y: np.ndarray, sr: int, ms: float = 5.0) -> np.ndarray:
    import librosa

    hop = int(sr * ms / 1000)
    env = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    db = 20 * np.log10(env + 1e-9)
    return np.maximum(db - db.max(), -60.0)


def _melspectrum(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=64, fmax=16000)
    v = 10 * np.log10(M.mean(axis=1) + 1e-12)
    return v - v.mean()


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def _gebeurtenissen(y: np.ndarray, sr: int, max_aantal: int = 6, afstand_s: float = 0.6) -> list[int]:
    """Indices (samples) van de luidste momenten in een opname, onderling minstens `afstand_s` uit elkaar."""
    env = _omhullende_db(y, sr, ms=5.0)
    volgorde = np.argsort(env)[::-1]
    gekozen: list[int] = []
    for i in volgorde:
        if env[i] < -30:
            break
        if all(abs(i - j) * 0.005 >= afstand_s for j in gekozen):
            gekozen.append(int(i))
        if len(gekozen) >= max_aantal:
            break
    return [int(i * sr * 0.005) for i in gekozen]


class Vergelijker:
    def __init__(self, mysterie: np.ndarray, sr: int, clap=None):
        self.sr = sr
        self.y = audio_mod.normaliseer(mysterie)
        self.clap = clap
        env = _omhullende_db(self.y, sr, ms=1.0)
        self.piek_s = float(np.argmax(env) * 0.001)
        self.duur_s = len(self.y) / sr
        self.env = _omhullende_db(self.y, sr)
        self.spec = _melspectrum(self.y, sr)
        self.emb = clap.audio_embedding(self.y, sr) if clap else None

    def venster(self, y: np.ndarray, piek: int) -> np.ndarray | None:
        a = piek - int(self.piek_s * self.sr)
        b = a + len(self.y)
        if a < 0:
            y, a, b = np.concatenate([np.zeros(-a, dtype=y.dtype), y]), 0, b - a
        if b > len(y):
            y = np.concatenate([y, np.zeros(b - len(y), dtype=y.dtype)])
        stuk = y[a:b]
        return audio_mod.normaliseer(stuk) if np.max(np.abs(stuk)) > 0 else None

    def vergelijk(self, ref: Referentie) -> Match | None:
        y = audio_mod.laad_audio(ref.pad, sr=self.sr)
        vensters = []
        for piek in _gebeurtenissen(y, self.sr):
            stuk = self.venster(y, piek)
            if stuk is not None:
                vensters.append((piek, stuk))
        if not vensters:
            return None
        ai_scores = [None] * len(vensters)
        if self.clap:  # alle vensters in één keer door het model (veel sneller)
            emb = self.clap._audio_batch([s.astype(np.float32) for _, s in vensters])
            ai_scores = [float(e @ self.emb) for e in emb]
        beste: Match | None = None
        for (piek, stuk), ai in zip(vensters, ai_scores):
            verloop = max(0.0, _corr(self.env, _omhullende_db(stuk, self.sr)))
            spectrum = (1 + _corr(self.spec, _melspectrum(stuk, self.sr))) / 2
            delen, gew = [verloop, spectrum], [1.0, 1.0]
            if ai is not None:
                delen.append(max(0.0, ai))
                gew.append(2.0)
            totaal = float(np.average(delen, weights=gew))
            start = max(0.0, piek / self.sr - self.piek_s)
            m = Match(ref, round(start, 2), totaal, ai, verloop, spectrum)
            if beste is None or m.totaal > beste.totaal:
                beste = m
        if beste is not None:  # vingerafdruk: blijft er na de klap een dunne metaaltoon hangen?
            from .features import _naklank

            stuk = self.venster(y, int((beste.tijd + self.piek_s) * self.sr))
            env = _omhullende_db(stuk, self.sr)
            lijnen = _naklank(stuk, self.sr, 10 ** (env / 20), 200.0)
            beste.naklank = len(lijnen)
            beste.naklank_db = max((r["prominentie_db"] for r in lijnen), default=0.0)
        return beste


def vergelijk_online(mysterie: np.ndarray, sr: int, zoektermen: list[str], per_term: int = 5, clap=None,
                     log=print) -> list[Match]:
    v = Vergelijker(mysterie, sr, clap)
    gezien: set[str] = set()
    resultaten: list[Match] = []
    for term in zoektermen:
        try:
            refs = zoek_bbc(term, per_term)
        except Exception as e:  # netwerk/API
            log(f"! Zoeken naar '{term}' mislukt: {e}")
            continue
        log(f"• '{term}': {len(refs)} opnames")
        for ref in refs:
            if ref.id in gezien:
                continue
            gezien.add(ref.id)
            try:
                download(ref)
                m = v.vergelijk(ref)
            except Exception as e:
                log(f"  ! {ref.id} overgeslagen: {e}")
                continue
            if m:
                resultaten.append(m)
    return sorted(resultaten, key=lambda m: m.totaal, reverse=True)


def maak_vergelijkingsfiguur(mysterie: np.ndarray, sr: int, matches: list[Match], pad: str | Path,
                             vergelijker: Vergelijker | None = None, aantal: int = 6) -> Path:
    """Spectrogrammen van het mysteriegeluid en de beste referentie-vensters onder elkaar."""
    import librosa
    import librosa.display
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = vergelijker or Vergelijker(mysterie, sr)
    rijen = [("MYSTERIEGELUID", v.y)]
    for m in matches[:aantal]:
        y = audio_mod.laad_audio(m.referentie.pad, sr=sr)
        piek = int((m.tijd + v.piek_s) * sr)
        stuk = v.venster(y, piek)
        if stuk is not None:
            ai = f", AI {m.ai:.2f}" if m.ai is not None else ""
            rijen.append((f"{m.totaal * 100:.0f}%{ai} - {m.referentie.beschrijving[:70]}", stuk))
    fig, assen = plt.subplots(len(rijen), 1, figsize=(10, 2.1 * len(rijen)), dpi=100)
    for ax, (titel, y) in zip(np.atleast_1d(assen), rijen):
        S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=256, n_mels=128,
                                                               fmax=16000), ref=np.max, top_db=80)
        librosa.display.specshow(S, sr=sr, hop_length=256, x_axis="time", y_axis="mel", fmax=16000, ax=ax,
                                 cmap="magma")
        ax.set_title(titel, fontsize=9, loc="left")
        ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(pad)
    plt.close(fig)
    return Path(pad)
