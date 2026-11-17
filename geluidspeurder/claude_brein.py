"""Claude als 'speurneus': redeneert over de metingen, het spectrogram en de spelstatus.

Twee stappen:
  1. brainstorm  -> nieuwe, specifieke kandidaten die nog niet in de lijst stonden
                    (die worden daarna door CLAP en de eigenschappen gescoord)
  2. eindoordeel -> weegt alles af en geeft een top 5 met eigen kansinschatting

Claude kan zelf geen audio horen; het krijgt de meetgegevens en het spectrogram (als afbeelding).
Vereist: pip install "geluidspeurder[claude]" en een API-sleutel (ANTHROPIC_API_KEY of `ant auth login`).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from .features import KENMERKEN
from .kandidaten import Kandidaat
from .status import Status

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

CONTEXT = """Je helpt bij 'Het Geluid', de radiowedstrijd van Qmusic. Er wordt steeds hetzelfde korte,
mysterieuze geluid afgespeeld; luisteraars bellen in om te raden wat het is. Elke foute gok wordt op de
radio genoemd, dus die mogelijkheden vallen af. Het antwoord is vrijwel altijd een alledaagse handeling
met een voorwerp (bijv. 'een kurk uit een fles trekken'), vaak in en om het huis, soms een combinatie van
twee handelingen. Het moet specifiek genoeg zijn: 'een rits' is iets anders dan 'de rits van een tent'.

Je krijgt objectieve metingen van een opname van het geluid (signaalanalyse), een spectrogram, de scores
van een audio-AI (CLAP, die de opname vergelijkt met Engelse beschrijvingen), en de spelstatus
(fout geraden antwoorden, hints). Redeneer als een akoestisch detective: welk materiaal, welke handeling,
welke volgorde, welke ruimte? Wees concreet en eerlijk over onzekerheid. Schrijf in het Nederlands."""

_KENMERK_ENUM = list(KENMERKEN)

SCHEMA_BRAINSTORM = {
    "type": "object",
    "properties": {
        "analyse": {"type": "string", "description": "Korte akoestische redenering (max ~150 woorden)"},
        "kandidaten": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "naam": {"type": "string", "description": "Specifieke Nederlandse omschrijving zoals je die zou inzenden"},
                    "cat": {"type": "string"},
                    "en": {"type": "array", "items": {"type": "string"},
                           "description": "2 korte Engelse geluidsbeschrijvingen voor CLAP"},
                    "k": {"type": "array", "items": {"type": "string", "enum": _KENMERK_ENUM}},
                },
                "required": ["naam", "cat", "en", "k"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["analyse", "kandidaten"],
    "additionalProperties": False,
}

SCHEMA_OORDEEL = {
    "type": "object",
    "properties": {
        "redenering": {"type": "string", "description": "Afweging van het bewijs (max ~250 woorden)"},
        "top": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "antwoord": {"type": "string"},
                    "kans": {"type": "number", "description": "Eigen inschatting 0..1"},
                    "waarom": {"type": "string"},
                },
                "required": ["antwoord", "kans", "waarom"],
                "additionalProperties": False,
            },
        },
        "al_geraden_equivalent": {"type": "array", "items": {"type": "string"},
                                  "description": "Kandidaten die in feite al fout geraden zijn"},
        "test_thuis": {"type": "array", "items": {"type": "string"},
                       "description": "Concrete geluiden om thuis op te nemen en te vergelijken (geluid vergelijk)"},
        "advies": {"type": "string", "description": "Wat zou jij inzenden en waarom (1-3 zinnen)"},
    },
    "required": ["redenering", "top", "al_geraden_equivalent", "test_thuis", "advies"],
    "additionalProperties": False,
}


def beschikbaar() -> tuple[bool, str]:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, "anthropic ontbreekt (installeer met: pip install \"geluidspeurder[claude]\")"
    return True, ""


def _status_tekst(status: Status) -> str:
    regels = [f"Fout geraden ({len(status.fout_geraden)}): " + ("; ".join(status.fout_geraden) or "nog niets")]
    if status.goede_richting:
        regels.append("Volgens de radio 'in de goede richting': " + "; ".join(status.goede_richting))
    if status.hints:
        regels.append("Hints: " + " | ".join(status.hints))
    if status.niet_categorie:
        regels.append("Uitgesloten categorieën: " + ", ".join(status.niet_categorie))
    if status.wel_categorie:
        regels.append("Waarschijnlijke categorieën: " + ", ".join(status.wel_categorie))
    if status.notities:
        regels.append("Notities: " + status.notities)
    return "\n".join(regels)


def _bewijs_tekst(analyse_dict: dict, observaties: list[str], ranglijst: list[dict], per_segment: list[dict]) -> str:
    geheel = analyse_dict["geheel"]
    meet = {k: geheel[k] for k in ("duur", "helderheid_hz", "vlakheid", "harmonisch_aandeel", "piekerigheid",
                                   "aanzet_ms", "uitsterf_s", "f0_mediaan_hz", "toon_verloop_st", "onset_tempo",
                                   "ioi_cv", "ioi_ratio", "ratel_hz", "ratel_sterkte", "partialen", "resonanties",
                                   "banden")}
    segs = [{"start": s["start"], "eind": s["eind"],
             "eigenschappen": {k: v for k, v in s["kenmerken"]["eigenschappen"].items() if v >= 0.4},
             "helderheid_hz": s["kenmerken"]["helderheid_hz"], "top_ai": s.get("top", [])[:5]}
            for s in per_segment]
    top = [{k: r[k] for k in ("naam", "kans", "clap_z", "kenmerk", "reden") if k in r} for r in ranglijst[:25]]
    return (
        "## Waarnemingen\n- " + "\n- ".join(observaties)
        + "\n\n## Eigenschappen geheel (0..1)\n" + json.dumps(geheel["eigenschappen"], ensure_ascii=False)
        + "\n\n## Ruwe metingen\n" + json.dumps(meet, ensure_ascii=False)
        + "\n\n## Delen (segmenten)\n" + json.dumps(segs, ensure_ascii=False)
        + "\n\n## Huidige ranglijst (kans = combinatie van CLAP-AI en eigenschappen; clap_z = hoeveel "
          "standaarddeviaties de AI-gelijkenis boven gemiddeld ligt)\n" + json.dumps(top, ensure_ascii=False)
    )


class ClaudeBrein:
    def __init__(self, model: str = MODEL, effort: str = "high"):
        ok, fout = beschikbaar()
        if not ok:
            raise RuntimeError(fout)
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.gebruikt_model: str | None = None

    def _vraag(self, tekst: str, schema: dict, spectrogram: Path | None) -> dict:
        inhoud: list[dict] = []
        if spectrogram and Path(spectrogram).exists():
            inhoud.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                       "data": base64.b64encode(Path(spectrogram).read_bytes()).decode()}})
        inhoud.append({"type": "text", "text": tekst})
        try:
            antwoord = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                betas=[FALLBACK_BETA],
                fallbacks="default",  # bij een (onterechte) weigering automatisch een ander model laten proberen
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                system=CONTEXT,
                messages=[{"role": "user", "content": inhoud}],
            )
        except self.anthropic.AuthenticationError as e:
            raise RuntimeError("Geen geldige API-sleutel. Zet ANTHROPIC_API_KEY of gebruik `ant auth login`.") from e
        except self.anthropic.RateLimitError as e:
            raise RuntimeError("Te veel verzoeken (rate limit). Probeer het zo opnieuw.") from e
        except self.anthropic.APIConnectionError as e:
            raise RuntimeError("Kan de Claude API niet bereiken (internetverbinding?).") from e
        except self.anthropic.APIStatusError as e:
            raise RuntimeError(f"Claude API-fout {e.status_code}: {e.message}") from e

        self.gebruikt_model = antwoord.model
        if antwoord.stop_reason == "refusal":
            raise RuntimeError("Claude weigerde dit verzoek.")
        if antwoord.stop_reason == "max_tokens":
            raise RuntimeError("Claude-antwoord werd afgekapt (max_tokens).")
        tekstblok = next((b.text for b in antwoord.content if b.type == "text"), None)
        if not tekstblok:
            raise RuntimeError("Claude gaf geen tekstantwoord.")
        return json.loads(tekstblok)

    def brainstorm(self, analyse_dict: dict, observaties: list[str], ranglijst: list[dict],
                   per_segment: list[dict], status: Status, spectrogram: Path | None,
                   aantal: int = 25) -> tuple[str, list[Kandidaat]]:
        vraag = (
            _bewijs_tekst(analyse_dict, observaties, ranglijst, per_segment)
            + "\n\n## Spelstatus\n" + _status_tekst(status)
            + f"\n\n## Opdracht\nBedenk {aantal} NIEUWE, specifieke kandidaten die goed passen bij het bewijs en "
            "die nog niet in de ranglijst staan en niet (in andere woorden) al fout geraden zijn. Denk ook aan "
            "combinaties van twee handelingen als er meerdere delen zijn. Varieer: niet allemaal varianten van "
            "één idee. Geef per kandidaat 2 korte Engelse beschrijvingen zoals een geluidseffect-bibliotheek "
            f"ze zou noemen, en de verwachte eigenschappen uit: {', '.join(_KENMERK_ENUM)}."
        )
        data = self._vraag(vraag, SCHEMA_BRAINSTORM, spectrogram)
        nieuw: list[Kandidaat] = []
        for d in data.get("kandidaten", []):
            en = [e for e in d.get("en", []) if e.strip()][:3]
            if not d.get("naam") or not en:
                continue
            nieuw.append(Kandidaat(naam=d["naam"].strip(), cat=d.get("cat", "overig") or "overig", en=en,
                                   k=[k for k in d.get("k", []) if k in KENMERKEN], alias=[], bron="claude"))
        return data.get("analyse", ""), nieuw

    def eindoordeel(self, analyse_dict: dict, observaties: list[str], ranglijst: list[dict],
                    per_segment: list[dict], status: Status, spectrogram: Path | None,
                    brainstorm_analyse: str = "") -> dict:
        vraag = (
            _bewijs_tekst(analyse_dict, observaties, ranglijst, per_segment)
            + "\n\n## Spelstatus\n" + _status_tekst(status)
            + (f"\n\n## Eerdere analyse\n{brainstorm_analyse}" if brainstorm_analyse else "")
            + "\n\n## Opdracht\nDe ranglijst bevat nu ook nieuwe kandidaten (bron 'claude'), gescoord door de "
            "audio-AI en de eigenschappen. Weeg al het bewijs af en geef je top 5 met je eigen, eerlijk "
            "gekalibreerde kans (die mag afwijken van de ranglijst; de kansen hoeven niet op te tellen tot 1 "
            "omdat het antwoord ook iets heel anders kan zijn). Noem kandidaten die in feite al fout geraden "
            "zijn, geef 3-6 concrete geluiden om thuis op te nemen ter vergelijking, en een kort advies."
        )
        return self._vraag(vraag, SCHEMA_OORDEEL, spectrogram)
