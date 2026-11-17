import pytest

from geluidspeurder.features import KENMERKEN
from geluidspeurder.kandidaten import Kandidaat, laad_kandidaten
from geluidspeurder.rangschik import rangschik
from geluidspeurder.status import Status, gelijkenis


@pytest.fixture(scope="module")
def kandidaten():
    return laad_kandidaten()


def test_database_geldig(kandidaten):
    assert len(kandidaten) > 150
    for k in kandidaten:
        assert k.naam and k.en and all(x in KENMERKEN for x in k.k)


@pytest.mark.parametrize("gok,naam,minimaal", [
    ("een rits", "Rits van een jas dichtritsen", 0.85),
    ("pingpong balletje", "Pingpongbal die stuitert op tafel", 0.85),
    ("een munt die tolt", "Munt die op tafel tolt en uitdraait", 0.85),
    ("Kurk", "Kurk uit een wijnfles trekken", 0.85),
])
def test_foute_gok_herkend(kandidaten, gok, naam, minimaal):
    k = next(k for k in kandidaten if k.naam == naam)
    assert gelijkenis(gok, [k.naam, *k.alias]) >= minimaal


@pytest.mark.parametrize("gok,naam", [
    ("een rits", "Rits van een tent openritsen"),      # specifieker -> niet uitsluiten
    ("fietsbel", "Freewheel van een fiets (achterwiel draait uit)"),
    ("een rits", "Zak rijst / macaroni schudden"),
    ("sleutels", "Sleutel in het slot omdraaien"),
])
def test_geen_onterechte_uitsluiting(kandidaten, gok, naam):
    k = next(k for k in kandidaten if k.naam == naam)
    assert gelijkenis(gok, [k.naam, *k.alias]) < 0.85


def _eig(**sterk):
    e = {k: 0.05 for k in KENMERKEN}
    e.update(sterk)
    return e


def test_rangschik_zonder_ai_volgt_eigenschappen(kandidaten):
    scores = rangschik(kandidaten, _eig(ritmisch=0.97, impuls=0.93, hoog=0.9, lang=0.4))
    top = [s.kandidaat.naam for s in scores[:5]]
    assert "Klok die tikt" in top
    assert abs(sum(s.kans for s in scores) - 1) < 1e-6


def test_fout_geraden_wordt_uitgesloten(kandidaten):
    e = _eig(stuiteren=0.98, impuls=0.9, hoog=0.6)
    zonder = rangschik(kandidaten, e)
    assert zonder[0].kandidaat.naam == "Pingpongbal die stuitert op tafel"
    met = rangschik(kandidaten, e, Status(fout_geraden=["pingpongbal"]))
    assert met[0].kandidaat.naam != "Pingpongbal die stuitert op tafel"
    pp = next(s for s in met if s.kandidaat.naam.startswith("Pingpongbal"))
    assert pp.uitgesloten and pp.kans == 0


def test_categorie_en_goede_richting():
    ks = [Kandidaat("A klik", "keuken", ["a click"], ["impuls"]),
          Kandidaat("B klik", "dier", ["a click"], ["impuls"]),
          Kandidaat("C fietsbel", "vervoer", ["a bell"], ["impuls"])]
    st = Status(niet_categorie=["dier"], goede_richting=["fietsbel"])
    scores = {s.kandidaat.naam: s for s in rangschik(ks, _eig(impuls=0.9), st)}
    assert scores["B klik"].uitgesloten
    assert scores["C fietsbel"].kans > scores["A klik"].kans


def test_clap_scores_wegen_mee():
    import numpy as np

    ks = [Kandidaat(f"K{i}", "x", ["a"], ["impuls"]) for i in range(10)]
    sims = np.full(10, 0.1)
    sims[7] = 0.4
    scores = rangschik(ks, _eig(impuls=0.9), clap_sims=sims)
    assert scores[0].kandidaat.naam == "K7"
    assert scores[0].clap_z > 2


def test_status_opslaan_en_laden(tmp_path):
    pad = tmp_path / "s.yaml"
    st = Status(fout_geraden=["een rits"], hints=["keuken"])
    st.bewaar(pad)
    terug = Status.laad(pad)
    assert terug.fout_geraden == ["een rits"] and terug.hints == ["keuken"]
    assert Status.laad(tmp_path / "bestaat_niet.yaml").fout_geraden == []
