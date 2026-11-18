import numpy as np
import pytest

from geluidspeurder.audio import normaliseer, segmenteer, trim_stilte
from geluidspeurder.features import KENMERKEN, analyseer, observaties

from . import synth


def _analyse(y):
    y = normaliseer(trim_stilte(y.astype(np.float32)))
    return analyseer(y, synth.SR, segmenteer(y, synth.SR))


@pytest.fixture(scope="module")
def resultaten():
    namen = ["stuiterbal", "klok", "rits", "glas_ping", "witte_ruis", "brom", "fles_vullen", "twee_delen"]
    return {n: _analyse(getattr(synth, n)()) for n in namen}


def test_alle_eigenschappen_aanwezig_en_begrensd(resultaten):
    for a in resultaten.values():
        e = a.geheel.eigenschappen
        assert set(e) == set(KENMERKEN)
        assert all(0.0 <= v <= 1.0 for v in e.values())


@pytest.mark.parametrize("naam,eigenschap", [
    ("stuiterbal", "stuiteren"),
    ("stuiterbal", "impuls"),
    ("klok", "ritmisch"),
    ("rits", "snelle_tikjes"),
    ("glas_ping", "galm"),
    ("witte_ruis", "ruis"),
    ("witte_ruis", "continu"),
    ("brom", "toon"),
    ("brom", "laag"),
    ("fles_vullen", "stijgend"),
    ("twee_delen", "meerdelig"),
])
def test_herkent_eigenschap(resultaten, naam, eigenschap):
    assert resultaten[naam].geheel.eigenschappen[eigenschap] > 0.6


@pytest.mark.parametrize("naam,eigenschap", [
    ("brom", "galm"),           # aanhoudend, sterft niet uit
    ("brom", "snelle_tikjes"),  # rimpel van een bromtoon is geen ratelen
    ("witte_ruis", "toon"),
    ("klok", "stuiteren"),
    ("stuiterbal", "ritmisch"),
    ("twee_delen", "stijgend"),  # van laag naar hoog deel is geen stijgende toon
    ("glas_ping", "ruis"),
])
def test_geen_valse_eigenschap(resultaten, naam, eigenschap):
    assert resultaten[naam].geheel.eigenschappen[eigenschap] < 0.4


def test_glas_heeft_inharmonische_partialen(resultaten):
    assert resultaten["glas_ping"].geheel.partialen == "inharmonisch"


def test_brom_heeft_harmonische_partialen_en_juiste_toon(resultaten):
    k = resultaten["brom"].geheel
    assert k.partialen == "harmonisch"
    assert 90 < k.f0_mediaan_hz < 110


def test_segmentatie_stilte(resultaten):
    assert len(resultaten["klok"].segmenten) >= 1
    assert len(resultaten["twee_delen"].segmenten) == 2


def test_observaties_zijn_tekst(resultaten):
    for a in resultaten.values():
        obs = observaties(a)
        assert obs and all(isinstance(o, str) and o for o in obs)


def test_trim_met_ruisvloer():
    y = np.concatenate([np.zeros(48000), synth.klok()[: 48000 // 2], np.zeros(48000)])
    y = y + 0.002 * np.random.default_rng(0).standard_normal(len(y))
    assert len(trim_stilte(y.astype(np.float32))) < 0.7 * len(y)


def test_vorm_uit_naklank():
    from geluidspeurder.features import vorm_uit_naklank

    staaf = vorm_uit_naklank([1257, 1297, 3468, 3553, 6571, 10769])
    vormen = [staaf["vorm"], *staaf["alternatieven"]]
    assert any(v.startswith("vrij staafje") for v in vormen) and any(v.startswith("ring") for v in vormen)
    assert staaf["grondtoon_hz"] == 1257
    assert staaf["dubbele_pieken"] >= 1 and len(staaf["lengte_cm"]) == 3
    assert vorm_uit_naklank([220, 440, 660, 880])["vorm"].startswith("snaar")
    assert vorm_uit_naklank([500, 3134, 8775])["vorm"].startswith("ingeklemde")
    assert vorm_uit_naklank([1000, 1500]) is None
