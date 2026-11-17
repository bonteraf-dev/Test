"""Audio laden, opschonen en opknippen in losse gebeurtenissen."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# CLAP verwacht 48 kHz; voor de signaalanalyse gebruiken we dezelfde rate.
STANDAARD_SR = 48_000


@dataclass
class Segment:
    """Een afgebakend stuk van de opname (in seconden)."""

    start: float
    eind: float

    @property
    def duur(self) -> float:
        return self.eind - self.start

    def knip(self, y: np.ndarray, sr: int) -> np.ndarray:
        return y[int(self.start * sr) : int(self.eind * sr)]


def laad_audio(
    pad: str | Path,
    sr: int = STANDAARD_SR,
    start: float | None = None,
    eind: float | None = None,
) -> np.ndarray:
    """Laad elk formaat dat ffmpeg kent (mp3, m4a, wav, mp4, ogg, ...) als mono float32."""
    pad = Path(pad)
    if not pad.exists():
        raise FileNotFoundError(f"Bestand niet gevonden: {pad}")

    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
        if start is not None:
            cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-i", str(pad)]
        if eind is not None:
            cmd += ["-t", f"{eind - (start or 0.0):.3f}"]
        cmd += ["-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
        res = subprocess.run(cmd, capture_output=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg kon {pad} niet lezen: {res.stderr.decode(errors='replace')}")
        y = np.frombuffer(res.stdout, dtype=np.float32).copy()
    else:  # zonder ffmpeg: alleen formaten die soundfile kent (wav, flac, ogg)
        import librosa

        offset = start or 0.0
        duur = (eind - offset) if eind is not None else None
        y, _ = librosa.load(str(pad), sr=sr, mono=True, offset=offset, duration=duur)
        y = y.astype(np.float32)

    if y.size == 0:
        raise ValueError(f"Geen audio gevonden in {pad} (controleer --start/--eind)")
    return y


def normaliseer(y: np.ndarray, piek: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(y))) if y.size else 0.0
    return y if m == 0 else (y / m * piek).astype(np.float32)


def stilte_drempel_db(y: np.ndarray, top_db: float = 35, frame: int = 512, hop: int = 128) -> float:
    """Hoeveel dB onder de piek telt als 'stil'?

    Houdt rekening met achtergrondruis van een radio-opname: als er een duidelijke ruisvloer is,
    ligt de grens een paar dB boven die vloer i.p.v. vast 35 dB onder de piek.
    """
    import librosa

    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-10)
    piek, vloer = float(db.max()), float(np.percentile(db, 10))
    if piek - vloer > 18:  # er is echt een stiller stuk (ruisvloer) aanwezig
        return float(min(top_db, max(12.0, piek - (vloer + 6.0))))
    return float(top_db)


def trim_stilte(y: np.ndarray, top_db: float = 35) -> np.ndarray:
    import librosa

    drempel = stilte_drempel_db(y, top_db)
    getrimd, (a, b) = librosa.effects.trim(y, top_db=drempel, frame_length=512, hop_length=128)
    if not getrimd.size:
        return y
    marge = 2048  # ~40 ms marge zodat de aanzet heel blijft
    return y[max(0, a - marge) : min(len(y), b + marge)]


def ontruis(y: np.ndarray, sr: int, sterkte: float = 1.5) -> np.ndarray:
    """Eenvoudige 'spectral gating': haalt constante achtergrondruis (radio-ruis, brom) weg.

    Het ruisprofiel wordt geschat uit de 10% stilste frames van de opname zelf.
    """
    import librosa

    n_fft, hop = 2048, 512
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, fase = np.abs(S), np.angle(S)
    energie = mag.sum(axis=0)
    stil = energie <= np.quantile(energie, 0.10)
    if stil.sum() < 3:
        return y
    ruis = np.median(mag[:, stil], axis=1, keepdims=True)
    drempel = ruis * sterkte
    masker = np.clip((mag - drempel) / np.maximum(mag, 1e-10), 0.0, 1.0)
    # masker iets uitsmeren in de tijd om 'muzikale ruis' te beperken
    from scipy.ndimage import uniform_filter1d

    masker = uniform_filter1d(masker, size=3, axis=1)
    schoon = librosa.istft(mag * masker * np.exp(1j * fase), hop_length=hop, length=len(y))
    return schoon.astype(np.float32)


def _nieuwheid(y: np.ndarray, sr: int, hop: int) -> np.ndarray:
    """Hoe sterk verandert de klankkleur rond elk frame? (voor knippen binnen een doorlopend geluid)"""
    import librosa

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)[1:]  # zonder luidheid
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)
    w = max(4, int(0.4 * sr / hop))
    n = mfcc.shape[1]
    nov = np.zeros(n)
    for t in range(w, n - w):
        voor = mfcc[:, t - w : t].mean(axis=1)
        na = mfcc[:, t : t + w].mean(axis=1)
        nov[t] = np.linalg.norm(voor - na) / np.sqrt(mfcc.shape[0])
    return nov


def segmenteer(
    y: np.ndarray,
    sr: int,
    top_db: float = 30,
    min_stilte: float = 0.20,
    min_duur: float = 0.015,
    splits_lange: float = 1.5,
) -> list[Segment]:
    """Knip de opname in gebeurtenissen.

    1. Gescheiden door stilte (bijv. 'klik ... klik ... bonk').
    2. Lange doorlopende stukken worden extra geknipt waar de klankkleur duidelijk
       verandert (bijv. 'water schenken' -> 'glas neerzetten').
    """
    import librosa

    hop = 256
    drempel = stilte_drempel_db(y, top_db, frame=1024, hop=hop)
    intervallen = librosa.effects.split(y, top_db=drempel, frame_length=1024, hop_length=hop)
    if len(intervallen) == 0:
        return [Segment(0.0, len(y) / sr)]

    # samenvoegen als de stilte ertussen te kort is
    samengevoegd: list[list[int]] = [list(intervallen[0])]
    for s, e in intervallen[1:]:
        if (s - samengevoegd[-1][1]) / sr < min_stilte:
            samengevoegd[-1][1] = e
        else:
            samengevoegd.append([s, e])

    segmenten: list[Segment] = []
    for s, e in samengevoegd:
        start, eind = s / sr, e / sr
        if eind - start < min_duur:
            continue
        if eind - start >= splits_lange:
            stuk = y[s:e]
            nov = _nieuwheid(stuk, sr, hop)
            if nov.max() > 0:
                from scipy.signal import find_peaks

                pieken, _ = find_peaks(nov, height=max(1.2, np.median(nov) * 3), distance=int(0.6 * sr / hop))
                grenzen = [start] + [start + p * hop / sr for p in pieken] + [eind]
                for a, b in zip(grenzen[:-1], grenzen[1:]):
                    if b - a >= min_duur:
                        segmenten.append(Segment(round(a, 3), round(b, 3)))
                continue
        segmenten.append(Segment(round(start, 3), round(eind, 3)))

    return segmenten or [Segment(0.0, len(y) / sr)]


def bereid_voor(
    pad: str | Path,
    start: float | None = None,
    eind: float | None = None,
    ontruisen: bool = False,
    sr: int = STANDAARD_SR,
) -> np.ndarray:
    """Laden + (optioneel) ontruisen + stilte wegknippen + normaliseren."""
    y = laad_audio(pad, sr=sr, start=start, eind=eind)
    if ontruisen:
        y = ontruis(y, sr)
    y = trim_stilte(y)
    return normaliseer(y)
