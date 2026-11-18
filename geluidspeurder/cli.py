"""Opdrachtregel: `geluid <commando>`."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__
from .status import Status

STANDAARD_STATUS = "status.yaml"


def _tijd(waarde: str) -> float:
    """Accepteert '12.5' of '0:12.5' of '1:02:03'."""
    delen = [float(x) for x in waarde.replace(",", ".").split(":")]
    sec = 0.0
    for d in delen:
        sec = sec * 60 + d
    return sec


def cmd_analyse(a: argparse.Namespace) -> int:
    from .kandidaten import laad_kandidaten
    from .pipeline import analyseer_bestand
    from .rapport import bewaar_json

    status = Status.laad(a.status)
    kandidaten = laad_kandidaten(a.kandidaten)
    uitmap = Path(a.uit) if a.uit else Path("resultaten") / Path(a.bestand).stem
    res = analyseer_bestand(a.bestand, kandidaten, status, start=a.start, eind=a.eind, ontruisen=a.ontruis,
                            gebruik_ai=not a.geen_ai, gebruik_claude=a.claude, uitmap=uitmap,
                            autoknip=not a.geen_autoknip,
                            log=(lambda _m: None) if a.stil else (lambda m: print(m, file=sys.stderr)))
    tekst = res.tekst(top=a.top)
    print(tekst)
    (uitmap / "rapport.txt").write_text(tekst, encoding="utf-8")
    bewaar_json(uitmap / "analyse.json", res.als_dict())
    print(f"Opgeslagen in {uitmap}/ (rapport.txt, analyse.json, spectrogram.png)", file=sys.stderr)
    return 0


def _status_wijzig(a: argparse.Namespace, veld: str, waarden: list[str]) -> int:
    st = Status.laad(a.status)
    lijst = getattr(st, veld)
    for w in waarden:
        if w not in lijst:
            lijst.append(w)
    st.bewaar(a.status)
    print(f"{veld.replace('_', ' ')}: {', '.join(lijst)}   (bewaard in {a.status})")
    return 0


def lees_gokken(tekst: str) -> list[str]:
    """Haal gokken uit geplakte tekst: één per regel (of gescheiden door ';').

    Nummering, opsommingstekens, tijdstippen en namen vooraan ('12. ', '- ', '14:30 ', 'Jan: ') worden
    weggehaald, zodat je de lijst van de website of app zo kunt plakken.
    """
    import csv
    import io
    import re

    tekst = tekst.lstrip("\ufeff")
    eerste = tekst.splitlines()[0] if tekst.strip() else ""
    if "antwoord" in eerste.lower():  # CSV-export met een kolom 'Antwoord' (namen e.d. worden genegeerd)
        scheiding = max(";,\t", key=eerste.count)
        lezer = csv.DictReader(io.StringIO(tekst), delimiter=scheiding)
        kolom = next(k for k in lezer.fieldnames or [] if k and "antwoord" in k.lower())
        tekst = "\n".join((rij.get(kolom) or "").replace(";", ",") for rij in lezer)

    gokken: list[str] = []
    for regel in re.split(r"[\n;]+", tekst):
        r = regel.strip()
        r = re.sub(r"^\s*(\d+[.)]|[-*\u2022\u2013])\s*", "", r)        # 12.  /  -  /  •
        r = re.sub(r"^\s*\d{1,2}[:.h]\d{2}\s*[-\u2013:]?\s*", "", r)  # 14:30 -
        r = re.sub(r"^[A-Z][\w\- ]{0,25}(\([^)]*\))?\s*:\s+", "", r)  # Naam (Gent):
        r = r.strip(" \"'\u201c\u201d.")
        if len(r) >= 3 and r.lower() not in {g.lower() for g in gokken}:
            gokken.append(r)
    return gokken


def cmd_fout(a):
    gokken = list(a.gok or [])
    if a.bestand:
        gokken += lees_gokken(Path(a.bestand).read_text(encoding="utf-8"))
    if not gokken:
        print("Geef gokken op, of --bestand met één gok per regel.", file=sys.stderr)
        return 1
    return _status_wijzig(a, "fout_geraden", gokken)


def cmd_richting(a):
    return _status_wijzig(a, "goede_richting", a.gok)


def cmd_hint(a):
    return _status_wijzig(a, "hints", [" ".join(a.tekst)])


def cmd_categorie(a):
    st = Status.laad(a.status)
    for c in a.niet or []:
        if c not in st.niet_categorie:
            st.niet_categorie.append(c)
    for c in a.wel or []:
        if c not in st.wel_categorie:
            st.wel_categorie.append(c)
    st.bewaar(a.status)
    print(f"niet: {st.niet_categorie}  wel: {st.wel_categorie}")
    return 0


def cmd_status(a):
    st = Status.laad(a.status)
    print(f"Status ({a.status}):")
    print(f"  Fout geraden ({len(st.fout_geraden)}): {'; '.join(st.fout_geraden) or '-'}")
    print(f"  Goede richting: {'; '.join(st.goede_richting) or '-'}")
    print(f"  Hints: {' | '.join(st.hints) or '-'}")
    print(f"  Categorie niet: {st.niet_categorie or '-'}  wel: {st.wel_categorie or '-'}")
    return 0


def cmd_kandidaten(a):
    from .kandidaten import laad_kandidaten
    from .status import gelijkenis

    ks = laad_kandidaten(a.kandidaten)
    if a.cat:
        ks = [k for k in ks if k.cat == a.cat]
    if a.zoek:
        ks = [k for k in ks if gelijkenis(a.zoek, [k.naam, *k.alias, *k.en]) >= 0.4 or a.zoek.lower() in k.naam.lower()]
    for k in ks:
        print(f"[{k.cat:12s}] {k.naam}   ({', '.join(k.k)})")
    print(f"\n{len(ks)} kandidaten. Categorieën: {', '.join(sorted({k.cat for k in laad_kandidaten(a.kandidaten)}))}")
    return 0


def cmd_vergelijk(a):
    from .vergelijk import vergelijk

    res = vergelijk(a.mysterie, a.opnames, start=a.start, eind=a.eind, ontruisen=a.ontruis,
                    gebruik_ai=not a.geen_ai, log=lambda m: print(m, file=sys.stderr))
    print(f"Welke eigen opname lijkt het meest op {Path(a.mysterie).name}?\n")
    print(f"  {'totaal':>6}  {'AI':>5}  {'klank':>5}  {'eig.':>5}  bestand")
    for v in res:
        ai = f"{v.ai:5.2f}" if v.ai is not None else "    -"
        print(f"  {v.totaal * 100:5.0f}%  {ai}  {v.klankkleur:5.2f}  {v.eigenschappen:5.2f}  {v.bestand}")
    return 0


def cmd_idee(a):
    from . import audio as audio_mod
    from . import clap as clap_mod

    ok, fout = clap_mod.beschikbaar()
    if not ok:
        print(f"Kan niet: {fout}", file=sys.stderr)
        return 1
    sr = audio_mod.STANDAARD_SR
    y = audio_mod.bereid_voor(a.bestand, start=a.start, eind=a.eind, sr=sr)
    clap = clap_mod.ClapScorer()
    emb = clap.audio_embedding(y, sr)
    sims = clap.vrije_tekst(emb, a.beschrijving)
    print("AI-gelijkenis per (Engelse) beschrijving - hoger = beter; vergelijk vooral onderling:\n")
    for s, t in sorted(zip(sims, a.beschrijving), reverse=True):
        print(f"  {s:6.3f}  {t}")
    return 0


def cmd_online(a):
    from . import audio as audio_mod
    from .online import vergelijk_online

    sr = audio_mod.STANDAARD_SR
    y = audio_mod.bereid_voor(a.mysterie, start=a.start, eind=a.eind, sr=sr)
    if a.start is None and a.eind is None:
        from .pipeline import _knip_jingle_weg

        y = _knip_jingle_weg(y, sr, [], lambda m: print(m, file=sys.stderr))
    clap = None
    if not a.geen_ai:
        from . import clap as clap_mod

        ok, fout = clap_mod.beschikbaar()
        if ok:
            try:
                clap = clap_mod.ClapScorer()
            except Exception as e:
                print(f"! AI-model niet beschikbaar ({e})", file=sys.stderr)
        else:
            print(f"! {fout}", file=sys.stderr)
    termen = list(a.zoek)
    if a.termen:
        termen += [t.strip() for t in Path(a.termen).read_text(encoding="utf-8").splitlines()
                   if t.strip() and not t.startswith("#")]
    res = vergelijk_online(y, sr, termen, per_term=a.per_term, clap=clap, log=lambda m: print(m, file=sys.stderr))
    print(f"\nBeste overeenkomsten met echte opnames (BBC Sound Effects), van {len(res)} vergeleken:\n")
    print(f"  {'totaal':>6}  {'AI':>5}  {'verloop':>7}  {'spectr.':>7}  zoekterm / beschrijving")
    for m in res[: a.top]:
        ai = f"{m.ai:5.2f}" if m.ai is not None else "    -"
        print(f"  {m.totaal * 100:5.1f}%  {ai}  {m.verloop:7.2f}  {m.spectrum:7.2f}  [{m.referentie.zoekterm}] "
              f"{m.referentie.beschrijving[:70]}  (bbc {m.referentie.id} @ {m.tijd:.1f}s)")
    if a.json:
        rijen = [{"totaal": m.totaal, "ai": m.ai, "verloop": m.verloop, "spectrum": m.spectrum, "tijd": m.tijd,
                  "id": m.referentie.id, "zoekterm": m.referentie.zoekterm, "beschrijving": m.referentie.beschrijving,
                  "pad": str(m.referentie.pad)} for m in res]
        Path(a.json).write_text(json.dumps(rijen, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def cmd_opnemen(a):
    if not shutil.which("ffmpeg"):
        print("ffmpeg is nodig om op te nemen.", file=sys.stderr)
        return 1
    uit = a.uit or "opname.mp3"
    print(f"Neem {a.duur:.0f} s op van {a.url} -> {uit}  (Ctrl+C om eerder te stoppen)", file=sys.stderr)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-i", a.url, "-t", str(a.duur), "-vn", uit]
    try:
        return subprocess.run(cmd, check=False).returncode
    except KeyboardInterrupt:
        return 0


def maak_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="geluid", description="Geluidspeurder: analyseer 'Het Geluid' en doe een "
                                "onderbouwde gok.")
    p.add_argument("--versie", action="version", version=__version__)
    sub = p.add_subparsers(dest="commando", required=True)

    def status_arg(sp):
        sp.add_argument("--status", default=STANDAARD_STATUS, help=f"statusbestand (standaard {STANDAARD_STATUS})")

    def knip_args(sp):
        sp.add_argument("--start", type=_tijd, help="begin van het geluid in de opname (s of m:ss)")
        sp.add_argument("--eind", type=_tijd, help="einde van het geluid in de opname (s of m:ss)")
        sp.add_argument("--ontruis", action="store_true", help="achtergrondruis/brom eerst wegfilteren")

    s = sub.add_parser("analyse", help="analyseer een opname en geef de beste gokken")
    s.add_argument("bestand")
    knip_args(s)
    status_arg(s)
    s.add_argument("--kandidaten", nargs="*", default=[], help="extra YAML-bestand(en) met eigen kandidaten")
    s.add_argument("--geen-ai", action="store_true", help="geen CLAP-model gebruiken (alleen signaalanalyse)")
    s.add_argument("--claude", action="store_true", help="laat Claude meedenken (API-sleutel nodig)")
    s.add_argument("--geen-autoknip", action="store_true",
                   help="niet automatisch het fragment tussen de jingle eruit halen")
    s.add_argument("--uit", help="map voor resultaten (standaard resultaten/<bestandsnaam>)")
    s.add_argument("--top", type=int, default=12, help="aantal kandidaten in het rapport")
    s.add_argument("--stil", action="store_true", help="geen voortgangsmeldingen")
    s.set_defaults(func=cmd_analyse)

    s = sub.add_parser("fout", help="voeg op de radio fout geraden antwoord(en) toe")
    s.add_argument("gok", nargs="*")
    s.add_argument("--bestand", help="tekstbestand met geplakte lijst (één gok per regel)")
    status_arg(s)
    s.set_defaults(func=cmd_fout)

    s = sub.add_parser("richting", help="gok die volgens de radio 'in de goede richting' zat")
    s.add_argument("gok", nargs="+")
    status_arg(s)
    s.set_defaults(func=cmd_richting)

    s = sub.add_parser("hint", help="voeg een hint toe (vrije tekst, gebruikt door Claude)")
    s.add_argument("tekst", nargs="+")
    status_arg(s)
    s.set_defaults(func=cmd_hint)

    s = sub.add_parser("categorie", help="sluit categorieën uit of geef aan waar het geluid vandaan komt")
    s.add_argument("--niet", nargs="*", help="bijv. dier vervoer")
    s.add_argument("--wel", nargs="*", help="bijv. keuken badkamer")
    status_arg(s)
    s.set_defaults(func=cmd_categorie)

    s = sub.add_parser("status", help="toon fout geraden antwoorden en hints")
    status_arg(s)
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("kandidaten", help="doorzoek de kandidatenlijst")
    s.add_argument("--zoek")
    s.add_argument("--cat")
    s.add_argument("--kandidaten", nargs="*", default=[])
    s.set_defaults(func=cmd_kandidaten)

    s = sub.add_parser("vergelijk", help="vergelijk het mysteriegeluid met je eigen opnames")
    s.add_argument("mysterie")
    s.add_argument("opnames", nargs="+")
    knip_args(s)
    s.add_argument("--geen-ai", action="store_true")
    s.set_defaults(func=cmd_vergelijk)

    s = sub.add_parser("idee", help="test eigen ideeën (Engelse beschrijvingen) tegen de opname met de AI")
    s.add_argument("bestand")
    s.add_argument("beschrijving", nargs="+", help='bijv. "a tape measure snapping shut"')
    s.add_argument("--start", type=_tijd)
    s.add_argument("--eind", type=_tijd)
    s.set_defaults(func=cmd_idee)

    s = sub.add_parser("online", help="vergelijk met echte opnames uit de BBC-geluidsbibliotheek")
    s.add_argument("mysterie")
    s.add_argument("zoek", nargs="*", help='Engelse zoektermen, bijv. "mousetrap" "letterbox"')
    s.add_argument("--termen", help="tekstbestand met één zoekterm per regel")
    s.add_argument("--per-term", type=int, default=5, help="aantal opnames per zoekterm (standaard 5)")
    s.add_argument("--top", type=int, default=25)
    s.add_argument("--json", help="alle resultaten opslaan als JSON")
    s.add_argument("--start", type=_tijd)
    s.add_argument("--eind", type=_tijd)
    s.add_argument("--geen-ai", action="store_true")
    s.set_defaults(func=cmd_online)

    s = sub.add_parser("opnemen", help="neem een (radio)stream op met ffmpeg")
    s.add_argument("url", help="stream-URL van de radiozender")
    s.add_argument("--duur", type=float, default=60.0, help="seconden (standaard 60)")
    s.add_argument("--uit", help="uitvoerbestand (standaard opname.mp3)")
    s.set_defaults(func=cmd_opnemen)
    return p


def main(argv: list[str] | None = None) -> int:
    args = maak_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Fout: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
