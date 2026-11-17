"""De volledige analyse van één opname, van audio tot advies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import audio as audio_mod
from .features import Analyse, analyseer, observaties
from .kandidaten import Kandidaat
from .rangschik import rangschik
from .rapport import maak_spectrogram, tekst_rapport
from .status import Status, normaliseer

MIN_SEGMENT_AI = 0.25  # kortere delen zijn te kort voor een zinnige AI-vergelijking


@dataclass
class Resultaat:
    bestand: str
    analyse: Analyse
    observaties: list[str]
    ranglijst: list[dict]
    per_segment: list[dict]
    clap_actief: bool
    spectrogram: Path | None
    brainstorm: str = ""
    oordeel: dict | None = None
    claude_kandidaten: list[dict] = field(default_factory=list)
    waarschuwingen: list[str] = field(default_factory=list)

    def tekst(self, top: int = 12) -> str:
        return tekst_rapport(self.bestand, self.analyse, self.observaties, self.ranglijst, self.per_segment,
                             self.clap_actief, self.oordeel, self.brainstorm, top, self.waarschuwingen)

    def als_dict(self) -> dict:
        return {
            "bestand": self.bestand,
            "analyse": self.analyse.als_dict(),
            "observaties": self.observaties,
            "ranglijst": self.ranglijst,
            "per_segment": self.per_segment,
            "clap_actief": self.clap_actief,
            "claude": {"brainstorm": self.brainstorm, "kandidaten": self.claude_kandidaten, "oordeel": self.oordeel},
            "waarschuwingen": self.waarschuwingen,
        }


def _per_segment(analyse: Analyse, y: np.ndarray, kandidaten: list[Kandidaat], status: Status,
                 clap, cand_emb: np.ndarray | None) -> list[dict]:
    uit = []
    for s, k in zip(analyse.segmenten, analyse.per_segment):
        sims = None
        if clap is not None and cand_emb is not None and s.duur >= MIN_SEGMENT_AI:
            sims = cand_emb @ clap.audio_embedding(s.knip(y, analyse.sr), analyse.sr)
        scores = rangschik(kandidaten, k.eigenschappen, status, sims)
        uit.append({"start": s.start, "eind": s.eind, "kenmerken": k.als_dict(),
                    "top": [x.als_dict() for x in scores if not x.uitgesloten][:5]})
    return uit


def analyseer_bestand(
    pad: str | Path,
    kandidaten: list[Kandidaat],
    status: Status,
    start: float | None = None,
    eind: float | None = None,
    ontruisen: bool = False,
    gebruik_ai: bool = True,
    gebruik_claude: bool = False,
    uitmap: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> Resultaat:
    pad = Path(pad)
    waarschuwingen: list[str] = []
    sr = audio_mod.STANDAARD_SR

    log("• Audio laden en opschonen ...")
    y = audio_mod.bereid_voor(pad, start=start, eind=eind, ontruisen=ontruisen, sr=sr)
    if len(y) / sr > 30:
        waarschuwingen.append(f"Opname is {len(y) / sr:.0f} s lang. Knip met --start/--eind alleen het geluid eruit "
                              "(zonder presentator/muziek) voor een veel betere analyse.")
    segmenten = audio_mod.segmenteer(y, sr)

    log(f"• Signaalanalyse ({len(segmenten)} deel/delen) ...")
    analyse = analyseer(y, sr, segmenten)
    obs = observaties(analyse)

    clap = cand_emb = sims = None
    if gebruik_ai:
        from . import clap as clap_mod

        ok, fout = clap_mod.beschikbaar()
        if ok:
            try:
                log("• AI-model (CLAP) laden en vergelijken ...")
                clap = clap_mod.ClapScorer()
                cand_emb = clap.kandidaat_embeddings(kandidaten)
                sims = cand_emb @ clap.audio_embedding(y, sr)
            except Exception as e:  # bv. geen internet bij eerste download
                waarschuwingen.append(f"AI-model kon niet geladen worden ({e}); alleen signaalanalyse gebruikt.")
                clap = cand_emb = sims = None
        else:
            waarschuwingen.append(f"Geen AI-vergelijking: {fout}. De ranglijst is daardoor grof.")

    scores = rangschik(kandidaten, analyse.geheel.eigenschappen, status, sims)
    ranglijst = [s.als_dict() for s in scores]
    per_segment = _per_segment(analyse, y, kandidaten, status, clap, cand_emb) if len(segmenten) > 1 else []

    uitmap = Path(uitmap) if uitmap else None
    spectrogram = None
    if uitmap:
        uitmap.mkdir(parents=True, exist_ok=True)
        spectrogram = maak_spectrogram(y, sr, analyse, uitmap / "spectrogram.png", titel=pad.name)

    res = Resultaat(bestand=pad.name, analyse=analyse, observaties=obs, ranglijst=ranglijst,
                    per_segment=per_segment, clap_actief=clap is not None, spectrogram=spectrogram,
                    waarschuwingen=waarschuwingen)

    if gebruik_claude:
        from .claude_brein import ClaudeBrein

        try:
            brein = ClaudeBrein()
            log("• Claude brainstormt over nieuwe kandidaten ...")
            res.brainstorm, nieuw = brein.brainstorm(analyse.als_dict(), obs, ranglijst, per_segment, status,
                                                     spectrogram)
            bekend = {normaliseer(k.naam) for k in kandidaten}
            nieuw = [k for k in nieuw if normaliseer(k.naam) not in bekend]
            res.claude_kandidaten = [k.als_dict() for k in nieuw]
            alle = kandidaten + nieuw
            sims2 = None
            if clap is not None:
                log(f"• {len(nieuw)} nieuwe kandidaten laten scoren door de AI ...")
                sims2 = clap.kandidaat_embeddings(alle) @ clap.audio_embedding(y, sr)
            res.ranglijst = [s.als_dict() for s in rangschik(alle, analyse.geheel.eigenschappen, status, sims2)]
            log("• Claude weegt alles af ...")
            res.oordeel = brein.eindoordeel(analyse.als_dict(), obs, res.ranglijst, per_segment, status,
                                            spectrogram, res.brainstorm)
        except Exception as e:
            res.waarschuwingen.append(f"Claude-stap mislukt: {e}")

    return res
