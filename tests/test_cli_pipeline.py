import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from geluidspeurder.cli import main
from geluidspeurder.kandidaten import laad_kandidaten
from geluidspeurder.pipeline import analyseer_bestand
from geluidspeurder.status import Status

from . import synth


@pytest.fixture
def wav(tmp_path):
    y = synth.stuiterbal()
    stil = np.zeros(synth.SR // 2)
    y = np.concatenate([stil, y / np.abs(y).max() * 0.8, stil])
    y = y + 0.001 * np.random.default_rng(3).standard_normal(len(y))
    pad = tmp_path / "geluid.wav"
    sf.write(pad, y.astype(np.float32), synth.SR)
    return pad


def test_cli_analyse_zonder_ai(wav, tmp_path, capsys):
    status = tmp_path / "status.yaml"
    assert main(["fout", "pingpongbal", "--status", str(status)]) == 0
    assert main(["hint", "het", "komt", "uit", "de", "woonkamer", "--status", str(status)]) == 0
    uit = tmp_path / "res"
    assert main(["analyse", str(wav), "--geen-ai", "--status", str(status), "--uit", str(uit), "--stil"]) == 0
    tekst = capsys.readouterr().out
    assert "RANGLIJST" in tekst and "stuitert" in tekst
    data = json.loads((uit / "analyse.json").read_text())
    assert data["analyse"]["geheel"]["eigenschappen"]["stuiteren"] > 0.6
    assert any(r["uitgesloten"] and r["naam"].startswith("Pingpongbal") for r in data["ranglijst"])
    assert (uit / "spectrogram.png").stat().st_size > 10_000
    assert Status.laad(status).hints == ["het komt uit de woonkamer"]


def test_cli_knippen_en_fouten(wav, tmp_path):
    uit = tmp_path / "r"
    assert main(["analyse", str(wav), "--geen-ai", "--start", "0:00.4", "--eind", "2.5", "--uit", str(uit),
                 "--stil", "--status", str(tmp_path / "s.yaml")]) == 0
    assert main(["analyse", str(tmp_path / "bestaat_niet.mp3"), "--geen-ai"]) == 1


def test_cli_vergelijk(wav, tmp_path, capsys):
    ander = tmp_path / "ruis.wav"
    sf.write(ander, synth.witte_ruis(1.5).astype(np.float32), synth.SR)
    zelfde = tmp_path / "bal2.wav"
    sf.write(zelfde, (synth.stuiterbal() * 0.5).astype(np.float32), synth.SR)
    assert main(["vergelijk", str(wav), str(ander), str(zelfde), "--geen-ai"]) == 0
    regels = [r for r in capsys.readouterr().out.splitlines() if r.strip().endswith(".wav")]
    assert regels[0].endswith("bal2.wav")


def test_cli_kandidaten_zoeken(capsys):
    assert main(["kandidaten", "--zoek", "rits"]) == 0
    assert "Rits van een jas" in capsys.readouterr().out


class _NepClient:
    """Doet alsof het de Claude API is (geen netwerk, geen kosten)."""

    def __init__(self):
        self.verzoeken = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.verzoeken.append(kw)
        schema = kw["output_config"]["format"]["schema"]
        if "kandidaten" in schema["properties"]:
            data = {"analyse": "Stuiterpatroon van een klein, hard voorwerp.",
                    "kandidaten": [{"naam": "Een flessendop die op de tegelvloer stuitert", "cat": "keuken",
                                    "en": ["a bottle cap bouncing on a tile floor"], "k": ["stuiteren", "impuls"]}]}
        else:
            data = {"redenering": "Past het best bij een stuiterende dop.",
                    "top": [{"antwoord": "Een flessendop die op de tegelvloer stuitert", "kans": 0.3,
                             "waarom": "stuiterpatroon"}],
                    "al_geraden_equivalent": [], "test_thuis": ["laat een dop vallen"], "advies": "Dop."}
        return SimpleNamespace(model=kw["model"], stop_reason="end_turn",
                               content=[SimpleNamespace(type="thinking", thinking=""),
                                        SimpleNamespace(type="text", text=json.dumps(data))])


def test_pipeline_met_nep_claude(wav, tmp_path, monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    nep = _NepClient()
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: nep)
    res = analyseer_bestand(wav, laad_kandidaten(), Status(fout_geraden=["pingpongbal"]), gebruik_ai=False,
                            gebruik_claude=True, uitmap=tmp_path / "uit", log=lambda m: None)
    assert not [w for w in res.waarschuwingen if "Claude" in w], res.waarschuwingen
    assert any(r["bron"] == "claude" for r in res.ranglijst)
    assert res.oordeel["top"][0]["antwoord"].startswith("Een flessendop")
    eerste = nep.verzoeken[0]
    assert eerste["model"] == "claude-opus-5-5"
    assert eerste["thinking"] == {"type": "adaptive"}
    assert eerste["fallbacks"] == "default"
    assert eerste["messages"][0]["content"][0]["type"] == "image"  # spectrogram meegestuurd
    assert "pingpongbal" in eerste["messages"][0]["content"][1]["text"]
    assert "CLAUDE - EINDOORDEEL" in res.tekst()


def test_fout_uit_geplakte_lijst(tmp_path):
    from geluidspeurder.cli import lees_gokken

    tekst = "1. Een rits\n2) kurk uit een fles\n- Fietspomp\n14:30 - Een blikje openen\nJan (Gent): de wc doortrekken\n\nKurk uit een fles\n"
    assert lees_gokken(tekst) == ["Een rits", "kurk uit een fles", "Fietspomp", "Een blikje openen",
                                  "de wc doortrekken"]
    lijst = tmp_path / "fout.txt"
    lijst.write_text(tekst, encoding="utf-8")
    status = tmp_path / "s.yaml"
    assert main(["fout", "--bestand", str(lijst), "--status", str(status)]) == 0
    assert len(Status.laad(status).fout_geraden) == 5


def _klap_met_zachte_metaalklank():
    t = np.arange(int(synth.SR * 1.1)) / synth.SR
    klap = synth._tik(helder=1500, duur=0.08, amp=1.0)
    ring = 0.01 * sum(np.sin(2 * np.pi * f * t) * np.exp(-t / 0.6) for f in (1260, 1822, 3469))  # -40 dB
    y = ring.copy()
    y[: len(klap)] += klap
    return y


def test_zachte_metaalnaklank_telt_als_galm():
    from geluidspeurder.audio import segmenteer
    from geluidspeurder.features import analyseer

    y = _klap_met_zachte_metaalklank().astype(np.float32)
    a = analyseer(y, synth.SR, segmenteer(y, synth.SR))
    assert a.geheel.eigenschappen["galm"] > 0.6
    assert {round(r["hz"], -2) for r in a.geheel.naklank} >= {1300.0, 1800.0, 3500.0}


def test_autoknip_haalt_geluid_tussen_jingle(tmp_path):
    stil = np.zeros(int(0.6 * synth.SR))
    jingle = 0.5 * synth.brom(3.0)
    geluid = _klap_met_zachte_metaalklank()
    y = np.concatenate([jingle, stil, geluid, stil, jingle]).astype(np.float32)
    pad = tmp_path / "met_jingle.wav"
    sf.write(pad, y, synth.SR)
    res = analyseer_bestand(pad, laad_kandidaten(), Status(), gebruik_ai=False, log=lambda m: None)
    assert any("Automatisch geknipt" in w for w in res.waarschuwingen)
    assert 0.9 < res.analyse.duur < 1.3
    zonder = analyseer_bestand(pad, laad_kandidaten(), Status(), gebruik_ai=False, autoknip=False, log=lambda m: None)
    assert zonder.analyse.duur > 5


def test_fout_uit_csv_export():
    from geluidspeurder.cli import lees_gokken

    csv_tekst = ("﻿Datum;Tijd;Antwoord;Naam;Woonplaats\n"
                 "2026-11-18;12:30;Het snijden van papier met een snijmachine;An;Gent\n"
                 "2026-11-18;11:30;Een autodeur toeslaan;Bo;Brugge\n")
    assert lees_gokken(csv_tekst) == ["Het snijden van papier met een snijmachine", "Een autodeur toeslaan"]
