"""Alle bewijs combineren tot één gerangschikte lijst met kansen.

Bewijsbronnen per kandidaat:
  1. AI-gelijkenis (CLAP): klinkt de opname als de Engelse beschrijving?  (optioneel)
  2. Eigenschappen: passen de gemeten eigenschappen bij wat je van dit geluid verwacht?
  3. Spelstatus: al fout geraden (uitsluiten/straffen), 'goede richting' (belonen), categorie-hints.

Het resultaat is een kansverdeling *binnen de kandidatenlijst*: staat het juiste antwoord er
niet in, dan kan het ook niet winnen. Daarom bestaat de Claude-brainstorm (claude_brein.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .kandidaten import Kandidaat
from .status import Status, gelijkenis

TEGENGESTELD = {"kort": "lang", "lang": "kort", "impuls": "continu", "continu": "impuls",
                "laag": "hoog", "hoog": "laag", "stijgend": "dalend", "dalend": "stijgend"}

# Hoe verrassend is het als een kandidaat die een eigenschap NIET noemt, hem toch sterk laat horen?
# Onderscheidende eigenschappen (stuiteren, stijgen) wegen zwaar; algemene (hoog, kort) licht.
ONDERSCHEIDEND = {"toon": 0.6, "ruis": 0.5, "galm": 0.6, "impuls": 0.5, "continu": 0.6, "snelle_tikjes": 0.7,
                  "ritmisch": 0.7, "stuiteren": 0.85, "stijgend": 0.8, "dalend": 0.8, "laag": 0.35, "hoog": 0.35,
                  "kort": 0.3, "lang": 0.4, "meerdelig": 0.5}

# Gewichten in de log-score
W_CLAP = 1.6       # per standaarddeviatie CLAP-gelijkenis
W_KENMERK = 0.8    # op de log-waarschijnlijkheid van de eigenschappen
UITSLUIT_GRENS = 0.85


@dataclass
class Score:
    kandidaat: Kandidaat
    kans: float
    clap: float | None
    clap_z: float | None
    kenmerk: float
    fout_match: float
    fout_gok: str | None
    richting_match: float
    uitgesloten: bool
    reden: str

    def als_dict(self) -> dict:
        return {
            "naam": self.kandidaat.naam, "cat": self.kandidaat.cat, "bron": self.kandidaat.bron,
            "kans": round(self.kans, 4), "clap": None if self.clap is None else round(self.clap, 4),
            "clap_z": None if self.clap_z is None else round(self.clap_z, 2), "kenmerk": round(self.kenmerk, 3),
            "fout_match": self.fout_match, "fout_gok": self.fout_gok, "richting_match": self.richting_match,
            "uitgesloten": self.uitgesloten, "reden": self.reden,
        }


def kenmerk_logscore(verwacht: list[str], gemeten: dict[str, float]) -> float:
    """Log-waarschijnlijkheid dat de gemeten eigenschappen bij deze kandidaat horen.

    - verwachte eigenschap aanwezig  -> ~0, afwezig -> sterk negatief
    - niet-verwachte eigenschap sterk aanwezig -> negatief naar mate hij onderscheidend is
      (of het tegendeel is van iets wat de kandidaat wél verwacht)
    """
    if not verwacht:
        return math.log(0.3) * 2
    totaal = 0.0
    for t, waarde in gemeten.items():
        if t in verwacht:
            totaal += math.log(0.05 + waarde)
        else:
            w = 0.9 if TEGENGESTELD.get(t) in verwacht else ONDERSCHEIDEND.get(t, 0.5)
            totaal += math.log(1.05 - w * waarde)
    return totaal


def kenmerk_overeenkomst(verwacht: list[str], gemeten: dict[str, float]) -> float:
    """Leesbare versie (0..1): meetkundig gemiddelde van de waarschijnlijkheden."""
    if not gemeten:
        return 0.5
    return float(min(1.0, math.exp(kenmerk_logscore(verwacht, gemeten) / len(gemeten))))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def rangschik(
    kandidaten: list[Kandidaat],
    eigenschappen: dict[str, float],
    status: Status | None = None,
    clap_sims: np.ndarray | None = None,
) -> list[Score]:
    """Rangschik kandidaten. clap_sims: cosinus-gelijkenis per kandidaat (zelfde volgorde) of None."""
    status = status or Status()
    n = len(kandidaten)
    log = np.zeros(n)
    z = None
    if clap_sims is not None and n > 1:
        sims = np.asarray(clap_sims, dtype=float)
        z = (sims - sims.mean()) / (sims.std() + 1e-9)
        log += W_CLAP * z

    scores: list[Score] = []
    uitgesloten = np.zeros(n, dtype=bool)
    redenen: list[list[str]] = [[] for _ in range(n)]
    fout_matches, fout_gokken, richting = np.zeros(n), [None] * n, np.zeros(n)
    kenm = np.zeros(n)

    for i, k in enumerate(kandidaten):
        kenm[i] = kenmerk_overeenkomst(k.k, eigenschappen)
        log[i] += W_KENMERK * kenmerk_logscore(k.k, eigenschappen)

        namen = [k.naam, *k.alias]
        for gok in status.fout_geraden:
            m = gelijkenis(gok, namen)
            if m > fout_matches[i]:
                fout_matches[i], fout_gokken[i] = m, gok
        if fout_matches[i] >= UITSLUIT_GRENS:
            uitgesloten[i] = True
            redenen[i].append(f"al fout geraden ('{fout_gokken[i]}')")
        elif fout_matches[i] >= 0.5:
            log[i] += math.log(1 - 0.8 * fout_matches[i] ** 2)
            redenen[i].append(f"lijkt op foute gok '{fout_gokken[i]}'")

        for gok in status.goede_richting:
            richting[i] = max(richting[i], gelijkenis(gok, namen))
        if richting[i] >= 0.4:
            log[i] += math.log(1 + 2.0 * richting[i])
            redenen[i].append("in de goede richting")

        if k.cat in status.niet_categorie:
            uitgesloten[i] = True
            redenen[i].append(f"categorie '{k.cat}' uitgesloten")
        elif status.wel_categorie and k.cat not in status.wel_categorie:
            log[i] += math.log(0.1)
            redenen[i].append("buiten de gehinte categorie")

    kans = np.zeros(n)
    if (~uitgesloten).any():
        kans[~uitgesloten] = _softmax(log[~uitgesloten])

    for i, k in enumerate(kandidaten):
        scores.append(Score(
            kandidaat=k, kans=float(kans[i]),
            clap=None if clap_sims is None else float(clap_sims[i]),
            clap_z=None if z is None else float(z[i]),
            kenmerk=float(kenm[i]), fout_match=float(fout_matches[i]), fout_gok=fout_gokken[i],
            richting_match=float(richting[i]), uitgesloten=bool(uitgesloten[i]), reden="; ".join(redenen[i]),
        ))
    scores.sort(key=lambda s: (not s.uitgesloten, s.kans), reverse=True)
    return scores
