# Geluidspeurder 🔊🕵️

Een hulpmiddel voor **Het Geluid** van Qmusic. Het analyseert een opname van het geluid en doet een onderbouwde gok: een ranglijst met kansen, een uitleg van wat er te horen is, en (optioneel) een eindoordeel van Claude.

## Hoe het werkt

```
opname ──► 1. opschonen ──► 2. signaalanalyse ──► 3. AI-vergelijking ──► 4. combineren ──► 5. Claude (optioneel)
           knippen,          wat hoor je precies?    klinkt het als ...?     + foute gokken     brainstormt nieuwe
           ruis eruit        (altijd, offline)       (CLAP-model)            + hints            kandidaten en weegt af
```

1. **Opschonen**: stilte eraf, optioneel achtergrondruis wegfilteren, en de opname opknippen in losse delen (bijv. *klik … klik … bonk*).
2. **Signaalanalyse** (pure natuurkunde, geen AI). Gemeten worden onder meer:
   - **klankkleur**: laag/dof of hoog/scherp, ruisachtig (lucht, water, papier) of tonaal (piep, fluit, motor);
   - **materiaal**: klinkt het na (metaal, glas, keramiek) of sterft het direct uit (hout, karton, stof)? Zijn de boventonen harmonisch (snaar, buis) of inharmonisch (metaal, bel)?
   - **ritme**: regelmatig (klok, stappen), snel ratelen (rits, freewheel) of **stuiteren**, dus tikken die steeds sneller en zachter komen (bal, dop, munt);
   - **verloop**: stijgt de toon (fles vullen, opwinden) of daalt hij (leeglopen, afremmen)?
3. **AI-vergelijking** met [CLAP](https://huggingface.co/laion/larger_clap_general): een model dat audio en tekst in dezelfde "betekenisruimte" plaatst. Daarmee wordt de opname vergeleken met ruim 220 alledaagse geluiden uit `geluidspeurder/data/kandidaten.yaml`, zowel voor de hele opname als per deel.
4. **Combineren**: de AI-score en de gemeten eigenschappen samen geven een kans per kandidaat. Antwoorden die op de radio **al fout geraden** zijn, worden uitgesloten of gestraft, ook als ze anders geschreven zijn (*"pingpong balletje"* ≈ *"Pingpongbal die stuitert"*). Hints over de categorie tellen ook mee.
5. **Claude** (optioneel, `--claude`): krijgt de metingen, het spectrogram en de spelstatus, bedenkt nieuwe specifieke kandidaten die niet in de lijst staan (die laat het programma daarna weer door de AI scoren), en geeft een top 5 met redenering, een advies en suggesties voor geluiden om thuis na te bootsen.

## Installatie

Je hebt Python 3.10+ en [ffmpeg](https://ffmpeg.org/) nodig (voor mp3/m4a/mp4).

```bash
pip install -e .              # basis: signaalanalyse + ranglijst
pip install -e ".[ai]"        # + AI-herkenning (torch + transformers, ~600 MB model bij eerste gebruik)
pip install -e ".[claude]"    # + Claude als meedenker (API-sleutel: ANTHROPIC_API_KEY of `ant auth login`)
```

## Gebruik

### 1. Opname maken
Neem het geluid op van de radio, de app of de website. Of gebruik de ingebouwde recorder met de stream-URL van de zender:

```bash
geluid opnemen <stream-url> --duur 120 --uit radio.mp3
```

### 2. Analyseren
Knip met `--start`/`--eind` **alleen het geluid** eruit, zonder presentator of muziek. Dat maakt een groot verschil.

```bash
geluid analyse radio.mp3 --start 1:02.4 --eind 1:05.1
geluid analyse radio.mp3 --start 1:02.4 --eind 1:05.1 --ontruis --claude
```

Je krijgt een rapport in de terminal, en in `resultaten/<naam>/` staan `rapport.txt`, `analyse.json` en `spectrogram.png`.

Voorbeeld (synthetisch stuitergeluid, zonder AI-model):

```
WAT IK HOOR (signaalanalyse)
  • Impulsief: scherpe aanzet (5 ms) -> botsing, tik, klik, klap.
  • Tikken komen steeds sneller en zachter -> iets valt en stuitert of tolt uit (bal, dop, munt, knikker).
RANGLIJST
   1. ██··················  11.0%  Dobbelstenen gooien
   2. ██··················  11.0%  Jenga-toren die omvalt
  Uitgesloten: Pingpongbal die stuitert op tafel
```

### 3. Bijhouden wat er al fout geraden is
Elke foute gok op de radio sluit iets uit. Houd het bij in `status.yaml` (je kunt dat bestand ook met de hand bewerken):

```bash
geluid fout "een rits" "fietspomp" "kurk uit een fles"
geluid richting "iets met metaal"           # "je zit in de goede richting"
geluid hint "het komt uit de keuken"
geluid categorie --niet dier vervoer --wel keuken
geluid status
```

Draai daarna `geluid analyse` opnieuw: de ranglijst en Claude houden er rekening mee.

### 4. Hypotheses testen
- **Eigen ideeën tegen de opname houden** (AI, Engelse beschrijvingen):
  ```bash
  geluid idee radio.mp3 --start 1:02.4 --eind 1:05.1 "a tape measure snapping shut" "a seatbelt retracting"
  ```
- **Zelf opnemen en vergelijken**: neem thuis kandidaat-geluiden op met je telefoon en kijk welke het meest lijkt:
  ```bash
  geluid vergelijk radio.mp3 --start 1:02.4 --eind 1:05.1 rolmaat.m4a gordel.m4a rits.m4a
  ```

### 5. Kandidatenlijst uitbreiden
```bash
geluid kandidaten --zoek rits
geluid analyse radio.mp3 --kandidaten mijn_ideeen.yaml
```
`mijn_ideeen.yaml` heeft hetzelfde formaat als `geluidspeurder/data/kandidaten.yaml`:
```yaml
- {naam: "Rolmaat die terugschiet", cat: gereedschap, en: ["a tape measure retracting"], k: [snelle_tikjes, impuls, kort]}
```

## Eerlijk over de grenzen
- De kans in de ranglijst is relatief **binnen de kandidatenlijst**. Staat het juiste antwoord er niet in, dan kan het ook niet winnen. Daarvoor zijn `--claude`, `idee` en eigen kandidaten bedoeld.
- Het Geluid is bewust lastig gekozen. Zie dit als een slimme assistent die de mogelijkheden flink kan verkleinen, niet als een orakel.
- Radio-opnames zijn gecomprimeerd en hebben vaak muziek of stemmen eronder. Knip strak en probeer `--ontruis`.
- Zonder AI-model (`[ai]`) is de ranglijst grof; de signaalanalyse zelf blijft wel bruikbaar.

## Ontwikkelen
```bash
pip install -e ".[dev]"
pytest
```
De tests gebruiken synthetische geluiden met bekende eigenschappen (stuiterbal, klok, rits, glas, fles vullen …) in `tests/synth.py`.

| Bestand | Wat |
|---|---|
| `geluidspeurder/audio.py` | laden (ffmpeg), knippen, ontruisen, segmenteren |
| `geluidspeurder/features.py` | akoestische metingen → eigenschappen → Nederlandse waarnemingen |
| `geluidspeurder/clap.py` | AI-vergelijking audio ↔ tekst en audio ↔ audio |
| `geluidspeurder/rangschik.py` | bewijs combineren tot kansen |
| `geluidspeurder/status.py` | foute gokken, hints, slimme naamvergelijking |
| `geluidspeurder/claude_brein.py` | Claude-brainstorm en eindoordeel |
| `geluidspeurder/pipeline.py` | alles aan elkaar geknoopt |
| `geluidspeurder/data/kandidaten.yaml` | ruim 220 alledaagse geluiden |
