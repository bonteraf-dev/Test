import numpy as np
import soundfile as sf

from geluidspeurder.online import Referentie, Vergelijker, maak_vergelijkingsfiguur

from . import synth


def test_vergelijker_vindt_zelfde_gebeurtenis_in_langere_opname(tmp_path):
    mysterie = synth.glas_ping()[: int(1.2 * synth.SR)].astype(np.float32)
    rng = np.random.default_rng(5)
    lang = np.concatenate([0.3 * synth.witte_ruis(1.0), np.zeros(int(2.0 * synth.SR)), mysterie * 0.7,
                           np.zeros(synth.SR), 0.5 * synth.brom(1.0)]) + 0.001 * rng.standard_normal(1)
    goed = tmp_path / "goed.wav"
    sf.write(goed, lang.astype(np.float32), synth.SR)
    fout = tmp_path / "fout.wav"
    sf.write(fout, synth.witte_ruis(4.0).astype(np.float32), synth.SR)

    v = Vergelijker(mysterie, synth.SR, clap=None)
    m_goed = v.vergelijk(Referentie("test", "1", "glas", "glass", goed))
    m_fout = v.vergelijk(Referentie("test", "2", "ruis", "noise", fout))
    assert abs(m_goed.tijd - 3.0) < 0.05
    assert m_goed.verloop > 0.9 and m_goed.spectrum > 0.9
    assert m_goed.totaal > m_fout.totaal + 0.2

    fig = maak_vergelijkingsfiguur(mysterie, synth.SR, [m_goed, m_fout], tmp_path / "v.png", v)
    assert fig.stat().st_size > 10_000
