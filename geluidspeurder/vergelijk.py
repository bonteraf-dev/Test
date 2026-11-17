"""Vergelijk het mysteriegeluid met je eigen opnames ('klinkt mijn rits zo?')."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio as audio_mod
from .features import KENMERKEN, analyseer


@dataclass
class Vergelijking:
    bestand: str
    totaal: float
    ai: float | None
    klankkleur: float
    eigenschappen: float


def _klankkleur(y: np.ndarray, sr: int) -> np.ndarray:
    """Timbre-vingerafdruk: gemiddelde + spreiding van MFCC's en spectrale contrast."""
    import librosa

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[1:]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    # vaste schaal per groep (typische spreiding), zodat het ook met één eigen opname werkt
    return np.concatenate([mfcc.mean(1) / 15.0, mfcc.std(1) / 8.0, contrast.mean(1) / 5.0])


def _eig_vector(y: np.ndarray, sr: int) -> np.ndarray:
    segs = audio_mod.segmenteer(y, sr)
    e = analyseer(y, sr, segs).geheel.eigenschappen
    return np.array([e.get(k, 0.0) for k in KENMERKEN])


def vergelijk(mysterie: str | Path, opnames: list[str | Path], start=None, eind=None, ontruisen=False,
              gebruik_ai: bool = True, log=print) -> list[Vergelijking]:
    sr = audio_mod.STANDAARD_SR
    ref = audio_mod.bereid_voor(mysterie, start=start, eind=eind, ontruisen=ontruisen, sr=sr)
    eigen = [(Path(p).name, audio_mod.bereid_voor(p, ontruisen=ontruisen, sr=sr)) for p in opnames]

    clap = None
    if gebruik_ai:
        from . import clap as clap_mod

        ok, fout = clap_mod.beschikbaar()
        if ok:
            try:
                clap = clap_mod.ClapScorer()
            except Exception as e:
                log(f"! AI-model niet beschikbaar ({e}); vergelijk alleen op klank en eigenschappen.")
        else:
            log(f"! {fout}; vergelijk alleen op klank en eigenschappen.")

    k_ref, e_ref = _klankkleur(ref, sr), _eig_vector(ref, sr)
    a_ref = clap.audio_embedding(ref, sr) if clap else None

    uit = []
    for naam, y in eigen:
        k = _klankkleur(y, sr)
        afstand = np.linalg.norm(k - k_ref) / np.sqrt(len(k))
        klank = float(np.exp(-afstand))
        eig = float(1 - np.mean(np.abs(_eig_vector(y, sr) - e_ref)))
        ai = float(clap.audio_embedding(y, sr) @ a_ref) if clap else None
        delen = [klank, eig] + ([max(0.0, ai)] if ai is not None else [])
        gewichten = [1.0, 1.0] + ([2.0] if ai is not None else [])
        totaal = float(np.average(delen, weights=gewichten))
        uit.append(Vergelijking(naam, totaal, ai, klank, eig))
    return sorted(uit, key=lambda v: v.totaal, reverse=True)
