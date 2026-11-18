"""Spelstatus: wat is er al fout geraden, welke hints zijn er gegeven.

Dit is cruciaal bij Het Geluid: elke foute gok op de radio sluit een mogelijkheid uit.
Bewaard als leesbaar YAML-bestand dat je ook met de hand kunt bewerken.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import yaml

STOPWOORDEN = {
    "een", "de", "het", "van", "die", "dat", "op", "in", "met", "aan", "uit", "je", "wordt", "worden", "er", "iets",
    "geluid", "en", "of", "te", "om", "bij", "naar", "voor", "over", "door", "is", "zijn", "wat", "hij", "zij", "ze",
    "men", "iemand", "the", "a", "an", "of", "to",
    # vulwoorden uit lange, beschrijvende gokken ('... zodat je ... nadat je ...')
    "naar", "laten", "zodat", "wanneer", "nadat", "als", "hebt", "heeft", "waardoor", "daarvoor", "terug", "oude",
    "oud", "ouderwetse", "grote", "kleine", "klein", "groot", "blad", "dan", "tot", "zich", "jouw", "uw", "mijn",
}


@dataclass
class Status:
    naam: str = "Het Geluid"
    fout_geraden: list[str] = field(default_factory=list)
    goede_richting: list[str] = field(default_factory=list)  # gokken waarvan gezegd werd: 'warm' / 'goede richting'
    hints: list[str] = field(default_factory=list)
    niet_categorie: list[str] = field(default_factory=list)
    wel_categorie: list[str] = field(default_factory=list)
    notities: str = ""

    @classmethod
    def laad(cls, pad: str | Path | None) -> "Status":
        if not pad or not Path(pad).exists():
            return cls()
        data = yaml.safe_load(Path(pad).read_text(encoding="utf-8")) or {}
        bekend = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        for k in ("fout_geraden", "goede_richting", "hints", "niet_categorie", "wel_categorie"):
            bekend[k] = [str(x) for x in (bekend.get(k) or [])]
        return cls(**bekend)

    def bewaar(self, pad: str | Path) -> None:
        Path(pad).write_text(
            "# Status voor Het Geluid - mag je met de hand bewerken\n"
            + yaml.safe_dump(asdict(self), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )


def normaliseer(tekst: str) -> str:
    t = unicodedata.normalize("NFKD", tekst.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return " ".join(w for w in t.split() if w not in STOPWOORDEN)


def _stam(w: str) -> str:
    """Heel grove Nederlandse stam (meervoud/verkleinwoord/werkwoord eraf)."""
    for uitgang in ("tjes", "etjes", "jes", "tje", "je", "en", "s", "e"):
        if len(w) > len(uitgang) + 2 and w.endswith(uitgang):
            return w[: -len(uitgang)]
    return w


def _woord_match(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if min(len(a), len(b)) >= 4 and (a in b or b in a):  # samenstelling: 'rits' in 'tentrits'
        return 0.6
    if min(len(a), len(b)) >= 5 and SequenceMatcher(None, a, b).ratio() >= 0.85:  # tikfout
        return 0.9
    return 0.0


def gelijkenis(gok: str, namen: list[str]) -> float:
    """0..1: hoe sterk lijkt een gok op een kandidaat? `namen` = [hoofdnaam, *aliassen].

    Woordgewijs (met grove stam en samenstellingen), plus een vergelijking zonder spaties
    voor Nederlandse samenstellingen ('pingpong balletje' ~ 'pingpongbal').
    """
    g = normaliseer(gok)
    if not g:
        return 0.0
    g_woorden = [_stam(w) for w in g.split()]
    beste = 0.0
    for i_naam, naam in enumerate(namen):
        n = normaliseer(naam)
        if not n:
            continue
        if g == n:
            return 1.0
        n_woorden = [_stam(w) for w in n.split()]
        per_gok = float(np.mean([max(_woord_match(a, b) for b in n_woorden) for a in g_woorden]))
        per_naam = float(np.mean([max(_woord_match(b, a) for a in g_woorden) for b in n_woorden]))
        # hoeveel van de gok zit in de kandidaat (en omgekeerd: hoeveel van de kandidaat zit in de gok)
        score = max(per_gok * (0.6 + 0.4 * per_naam), per_naam * (0.6 + 0.4 * per_gok))
        # een specifieke gok (2+ woorden) die volledig in de kandidaat zit = dezelfde gok
        if len(g_woorden) >= 2 and per_gok >= 0.95:
            score = max(score, 0.9)
        # een korte kandidaat die volledig in een (lange, uitleggende) gok zit = ook dezelfde gok
        # ('Perforator' ~ 'Met een perforator een blad papier perforeren')
        # (alleen voor de hoofdnaam, of aliassen van 2+ woorden: 'klok' of 'slot' zit te vaak toevallig in een gok)
        if len(n_woorden) <= 3 and per_naam >= 0.85 and (i_naam == 0 or len(n_woorden) >= 2):
            score = max(score, 0.9)
        g2, n2 = g.replace(" ", ""), n.replace(" ", "")
        if min(len(g2), len(n2)) >= 10:
            zonder = SequenceMatcher(None, g2, n2).ratio()
            if zonder >= 0.8:
                score = max(score, zonder + 0.1)
        beste = max(beste, score)
    return round(min(beste, 1.0), 3)
