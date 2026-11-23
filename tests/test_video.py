import shutil
import subprocess

import numpy as np
import pytest

from geluidspeurder.video import kies_voorwerpen, maak_overzicht


def test_kies_voorwerpen_een_per_voorwerp():
    rng = np.random.default_rng(0)
    voorwerpen = [rng.uniform(100, 255, (56, 40)) for _ in range(5)]
    kernen = []
    for v in voorwerpen:          # elk voorwerp 3 beelden, met een donker overgangsbeeld ertussen
        kernen += [v, v + 0.5, v]
        kernen.append(np.zeros((56, 40)))
    gekozen = kies_voorwerpen(np.stack(kernen))
    assert len(gekozen) == 5


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg ontbreekt")
def test_maak_overzicht_op_synthetische_video(tmp_path):
    video = tmp_path / "carrousel.mp4"
    # 4 heldere "voorwerpen" van elk 0,5 s op zwart (donkere beelden worden bewust overgeslagen)
    filters = ";".join(f"color=c={c}:s=320x240:d=0.5,drawbox=x=0:y=0:w=320:h=240:c=black:t=60[v{i}]"
                       for i, c in enumerate(["white", "yellow", "cyan", "pink"]))
    filters += ";" + "".join(f"[v{i}]" for i in range(4)) + "concat=n=4:v=1:a=0[out]"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-filter_complex", filters, "-map", "[out]",
                    "-r", "20", str(video)], check=True)
    n, vellen = maak_overzicht(video, tmp_path / "uit")
    assert 3 <= n <= 5 and vellen and vellen[0].stat().st_size > 1000
