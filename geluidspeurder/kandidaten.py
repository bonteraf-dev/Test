"""Kandidaat-geluiden laden uit YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

from .features import KENMERKEN


@dataclass
class Kandidaat:
    naam: str
    cat: str
    en: list[str]
    k: list[str] = field(default_factory=list)
    alias: list[str] = field(default_factory=list)
    bron: str = "database"  # 'database', 'eigen', 'claude'

    def als_dict(self) -> dict:
        return {"naam": self.naam, "cat": self.cat, "en": self.en, "k": self.k, "alias": self.alias, "bron": self.bron}


def _van_dict(d: dict, bron: str) -> Kandidaat:
    if "naam" not in d or not d.get("en"):
        raise ValueError(f"Kandidaat mist 'naam' of 'en': {d}")
    onbekend = [k for k in d.get("k", []) if k not in KENMERKEN]
    if onbekend:
        raise ValueError(f"Onbekende eigenschap(pen) {onbekend} bij '{d['naam']}'. Kies uit: {', '.join(KENMERKEN)}")
    en = d["en"] if isinstance(d["en"], list) else [d["en"]]
    return Kandidaat(naam=str(d["naam"]), cat=str(d.get("cat", "overig")), en=[str(x) for x in en],
                     k=list(d.get("k", [])), alias=[str(a) for a in d.get("alias", [])], bron=bron)


def standaard_pad() -> Path:
    return Path(str(resources.files("geluidspeurder") / "data" / "kandidaten.yaml"))


def laad_kandidaten(extra: list[str | Path] | None = None, standaard: bool = True) -> list[Kandidaat]:
    """Laad de ingebouwde lijst en eventueel eigen YAML-bestanden (zelfde formaat)."""
    paden: list[tuple[Path, str]] = [(standaard_pad(), "database")] if standaard else []
    paden += [(Path(p), "eigen") for p in (extra or [])]
    lijst: list[Kandidaat] = []
    gezien: set[str] = set()
    for pad, bron in paden:
        data = yaml.safe_load(Path(pad).read_text(encoding="utf-8")) or []
        for d in data:
            kand = _van_dict(d, bron)
            sleutel = kand.naam.strip().lower()
            if sleutel in gezien:
                continue
            gezien.add(sleutel)
            lijst.append(kand)
    return lijst
