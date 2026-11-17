"""Uitvoer: spectrogram (PNG), tekstrapport en JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import KENMERKEN, Analyse


def maak_spectrogram(y: np.ndarray, sr: int, analyse: Analyse, pad: str | Path, titel: str = "") -> Path:
    import librosa
    import librosa.display
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pad = Path(pad)
    fig = plt.figure(figsize=(12, 8.5), dpi=110)
    raster = fig.add_gridspec(3, 2, height_ratios=[1, 2.6, 1.3], width_ratios=[4, 1.4], hspace=0.5, wspace=0.25)
    t = np.arange(len(y)) / sr

    # golfvorm + segmenten + aanzetten
    ax0 = fig.add_subplot(raster[0, 0])
    ax0.plot(t, y, lw=0.4, color="#335")
    for i, s in enumerate(analyse.segmenten):
        ax0.axvspan(s.start, s.eind, color=f"C{i % 10}", alpha=0.15)
        ax0.text(s.start, 0.9, f"{i + 1}", fontsize=8, va="top")
    for o in analyse.geheel.onsets:
        ax0.axvline(o, color="red", lw=0.6, alpha=0.6)
    ax0.set_xlim(0, t[-1] if len(t) else 1)
    ax0.set_ylim(-1, 1)
    ax0.set_title(titel or "Golfvorm (gekleurd = delen, rood = aanzetten)", fontsize=10)
    ax0.set_yticks([])

    # mel-spectrogram
    ax1 = fig.add_subplot(raster[1, 0], sharex=ax0)
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=256, n_mels=160, fmax=sr // 2)
    librosa.display.specshow(librosa.power_to_db(M, ref=np.max, top_db=80), sr=sr, hop_length=256,
                             x_axis="time", y_axis="mel", fmax=sr // 2, ax=ax1, cmap="magma")
    for r in analyse.geheel.resonanties[:3]:
        ax1.axhline(r["hz"], color="cyan", lw=0.6, ls="--", alpha=0.7)
    ax1.set_title("Spectrogram (mel-schaal; stippellijnen = sterkste resonanties)", fontsize=10)
    ax1.set_xlabel("tijd (s)")

    # gemiddeld spectrum
    ax2 = fig.add_subplot(raster[1, 1])
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=4096)
    spec = 10 * np.log10(S.mean(axis=1) + 1e-12)
    ax2.semilogy(spec - spec.max(), np.maximum(f, 20), lw=0.8, color="#335")
    ax2.set_ylim(40, sr / 2)
    ax2.set_xlabel("dB")
    ax2.set_title("Gemiddeld spectrum", fontsize=10)

    # eigenschappen
    ax3 = fig.add_subplot(raster[2, :])
    e = analyse.geheel.eigenschappen
    namen = list(KENMERKEN)
    waarden = [e.get(n, 0) for n in namen]
    kleuren = ["#d62728" if w >= 0.6 else "#ff7f0e" if w >= 0.35 else "#bbbbbb" for w in waarden]
    ax3.bar(namen, waarden, color=kleuren)
    ax3.set_ylim(0, 1)
    ax3.set_title("Gemeten eigenschappen (0 = afwezig, 1 = sterk aanwezig)", fontsize=10)
    ax3.tick_params(axis="x", labelrotation=30, labelsize=8)

    fig.savefig(pad, bbox_inches="tight")
    plt.close(fig)
    return pad


def _balk(kans: float, breedte: int = 20) -> str:
    n = int(round(kans * breedte))
    return "█" * n + "·" * (breedte - n)


def tekst_rapport(
    bestand: str,
    analyse: Analyse,
    observaties: list[str],
    ranglijst: list[dict],
    per_segment: list[dict],
    clap_actief: bool,
    oordeel: dict | None = None,
    brainstorm: str = "",
    top: int = 12,
    waarschuwingen: list[str] | None = None,
) -> str:
    r: list[str] = []
    r.append("=" * 72)
    r.append(f" GELUIDSPEURDER  -  {bestand}")
    r.append("=" * 72)
    for w in waarschuwingen or []:
        r.append(f"! {w}")
    r.append("")
    r.append("WAT IK HOOR (signaalanalyse)")
    for o in observaties:
        r.append(f"  • {o}")
    sterk = [f"{k} {v:.2f}" for k, v in sorted(analyse.geheel.eigenschappen.items(), key=lambda kv: -kv[1]) if v >= 0.5]
    r.append(f"  Sterkste eigenschappen: {', '.join(sterk) or '-'}")

    if len(per_segment) > 1:
        r.append("")
        r.append("PER DEEL")
        for i, s in enumerate(per_segment, 1):
            top3 = ", ".join(f"{x['naam']} ({x['kans'] * 100:.0f}%)" for x in s.get("top", [])[:3])
            eig = ", ".join(k for k, v in s["kenmerken"]["eigenschappen"].items() if v >= 0.6)
            r.append(f"  {i}. {s['start']:.2f}-{s['eind']:.2f}s  [{eig or 'geen uitgesproken eigenschappen'}]")
            if top3:
                r.append(f"     -> {top3}")

    r.append("")
    bron = "AI (CLAP) + eigenschappen" if clap_actief else "alleen eigenschappen (geen AI-model actief: grove indicatie)"
    r.append(f"RANGLIJST  ({bron})")
    for i, x in enumerate([x for x in ranglijst if not x["uitgesloten"]][:top], 1):
        extra = []
        if x.get("clap_z") is not None:
            extra.append(f"AI {x['clap_z']:+.1f}σ")
        extra.append(f"eig {x['kenmerk']:.2f}")
        if x["bron"] != "database":
            extra.append(x["bron"])
        if x["reden"]:
            extra.append(x["reden"])
        r.append(f"  {i:2d}. {_balk(x['kans'])} {x['kans'] * 100:5.1f}%  {x['naam']}  ({'; '.join(extra)})")
    uit = [x for x in ranglijst if x["uitgesloten"]]
    if uit:
        r.append(f"  Uitgesloten: {', '.join(x['naam'] for x in uit[:10])}{' ...' if len(uit) > 10 else ''}")

    if brainstorm:
        r.append("")
        r.append("CLAUDE - AKOESTISCHE ANALYSE")
        r.append(f"  {brainstorm}")
    if oordeel:
        r.append("")
        r.append("CLAUDE - EINDOORDEEL")
        r.append(f"  {oordeel.get('redenering', '')}")
        r.append("")
        for i, t in enumerate(oordeel.get("top", [])[:5], 1):
            r.append(f"  {i}. {t['antwoord']}  (~{float(t['kans']) * 100:.0f}%) - {t['waarom']}")
        if oordeel.get("al_geraden_equivalent"):
            r.append(f"  Al fout geraden (in andere woorden): {', '.join(oordeel['al_geraden_equivalent'])}")
        if oordeel.get("test_thuis"):
            r.append("  Neem thuis op en vergelijk met `geluid vergelijk`:")
            for t in oordeel["test_thuis"]:
                r.append(f"    - {t}")
        r.append("")
        r.append(f"  ADVIES: {oordeel.get('advies', '')}")
    else:
        beste = next((x for x in ranglijst if not x["uitgesloten"]), None)
        if beste:
            r.append("")
            r.append(f"BESTE GOK: {beste['naam']}  ({beste['kans'] * 100:.0f}% binnen de kandidatenlijst)")
    r.append("")
    return "\n".join(r)


def bewaar_json(pad: str | Path, data: dict) -> Path:
    pad = Path(pad)
    pad.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return pad
