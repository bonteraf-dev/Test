"""Akoestische kenmerken meten en vertalen naar begrijpelijke eigenschappen.

Alles hier is pure signaalanalyse (geen AI): het werkt dus altijd, ook offline.
De 'eigenschappen' (KENMERKEN) zijn scores tussen 0 en 1 die later met de verwachte
eigenschappen van elke kandidaat worden vergeleken.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .audio import Segment

# Eigenschappen die een kandidaat kan hebben, met uitleg (gebruikt in rapport en database).
KENMERKEN: dict[str, str] = {
    "toon": "duidelijke toonhoogte (fluit, piep, snaar, motorbrom)",
    "ruis": "ruisachtig zonder toon (lucht, water, schuren, papier)",
    "galm": "klinkt na / resoneert (metaal, glas, keramiek, bel)",
    "impuls": "harde, korte tik of klap",
    "continu": "aanhoudend, gelijkmatig geluid",
    "snelle_tikjes": "snelle reeks klikjes of ratelen (rits, ratel, freewheel)",
    "ritmisch": "regelmatige herhaling (klok, stappen, kloppen, pomp)",
    "stuiteren": "steeds snellere, zachtere tikken (stuiteren, uittollen)",
    "stijgend": "toonhoogte of resonantie stijgt (fles vullen, opwinden)",
    "dalend": "toonhoogte of resonantie daalt (leeglopen, uitdraaien)",
    "laag": "veel lage tonen (groot, zwaar, dof)",
    "hoog": "veel hoge tonen (klein, licht, scherp, sissen)",
    "kort": "korter dan ongeveer 1,5 seconde",
    "lang": "langer dan ongeveer 4 seconden",
    "meerdelig": "bestaat uit duidelijk verschillende delen",
}

BANDEN = [("sub/laag (<250 Hz)", 0, 250), ("laag-midden (250-1000 Hz)", 250, 1000),
          ("midden (1-4 kHz)", 1000, 4000), ("hoog (4-10 kHz)", 4000, 10000), ("zeer hoog (>10 kHz)", 10000, 1e9)]


def _sig(x: float, midden: float, breedte: float) -> float:
    return float(1.0 / (1.0 + math.exp(-(x - midden) / breedte)))


def _rond(x, n=3):
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return round(float(x), n)


@dataclass
class Kenmerken:
    duur: float
    luidheid_db: float
    piekerigheid: float          # aandeel energie in de 5% luidste momenten (impulsief -> hoog)
    omhullende_std_db: float     # hoe sterk de luidheid schommelt
    aanzet_ms: float             # hoe snel het geluid opkomt (ms)
    uitsterf_s: float            # tijd van piek tot -20 dB
    helderheid_hz: float         # spectraal zwaartepunt
    helderheid_helling: float    # verandering zwaartepunt (halve tonen/s)
    bandbreedte_hz: float
    rolloff_hz: float
    vlakheid: float              # 0 = zuivere toon, ~0.5 = witte ruis
    harmonisch_aandeel: float    # energie in tonale component (HPSS)
    banden: dict[str, float]
    f0_mediaan_hz: float | None
    stemhebbend: float           # aandeel frames met een toonhoogte
    toon_verloop_st: float       # totale stijging/daling (halve tonen)
    toon_verloop_r: float        # hoe consistent die stijging/daling is (|r|)
    resonanties: list[dict]      # opvallende pieken in het spectrum
    partialen: str | None        # 'harmonisch', 'inharmonisch' of None
    onsets: list[float]
    onset_tempo: float           # aanzetten per seconde
    ioi_cv: float | None         # regelmaat (0 = metronoom)
    ioi_ratio: float | None      # mediane verhouding opeenvolgende tussenpozen (<1 = versnelt)
    ioi_dalend: float | None     # aandeel tussenpozen dat korter wordt
    kracht_trend: float | None   # helling van de onset-sterkte (neg = wordt zachter)
    ratel_hz: float | None       # dominante 'ratelfrequentie' van de omhullende (8-80 Hz)
    ratel_sterkte: float         # hoe uitgesproken dat ratelen is (periodiciteit 0..1)
    sterft_uit: bool = True      # zakt het geluid binnen de opname >20 dB weg na de piek?
    eigenschappen: dict[str, float] = field(default_factory=dict)

    def als_dict(self) -> dict:
        d = asdict(self)
        return {k: (_rond(v) if isinstance(v, float) else v) for k, v in d.items()}


def _theil_sen(t: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """Robuuste helling + correlatie."""
    from scipy.stats import theilslopes

    if len(t) < 5 or np.ptp(t) == 0:
        return 0.0, 0.0
    helling = float(theilslopes(v, t)[0])
    r = float(np.corrcoef(t, v)[0, 1]) if np.std(v) > 0 else 0.0
    return helling, (0.0 if math.isnan(r) else r)


def _partialen(fs: list[float], tolerantie_hz: float = 10.0) -> str | None:
    """Zijn de resonanties (bijna) veelvouden van één grondtoon? Metaal/glas: nee; snaar/buis: ja."""
    fs = sorted(f for f in fs if f)
    if len(fs) < 3:
        return None
    kandidaten = {fs[0], fs[0] / 2, fs[0] / 3, *(b - a for a, b in zip(fs, fs[1:]) if b - a > 40)}
    beste = 0.0
    for f0 in kandidaten:
        if f0 < 40 or max(fs) / f0 > 24:
            continue
        fout = np.array([abs(f - max(1, round(f / f0)) * f0) for f in fs])
        grens = np.maximum(tolerantie_hz, 0.025 * np.array(fs))
        beste = max(beste, float(np.mean(fout <= grens)))
    return "harmonisch" if beste >= 0.8 else "inharmonisch"


def _ratel(env: np.ndarray, env_sr: float) -> tuple[float | None, float]:
    """Periodiciteit van de omhullende tussen 8 en 80 Hz (rits, ratel, freewheel)."""
    if len(env) < env_sr * 0.25:
        return None, 0.0
    from scipy.ndimage import uniform_filter1d

    vloer = env.max() * 10 ** (-40 / 20)
    actief = np.where(env > vloer)[0]
    if len(actief) == 0 or actief[-1] - actief[0] < env_sr * 0.2:
        return None, 0.0
    e = np.log(np.maximum(env[actief[0] : actief[-1] + 1], vloer))
    e = e - uniform_filter1d(e, size=int(env_sr * 0.15))  # trage verloop eruit
    e = e - e.mean()
    if np.std(e) < 1e-6:
        return None, 0.0
    ac = np.correlate(e, e, mode="full")[len(e) - 1 :]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(env_sr / 80), int(env_sr / 8)
    if hi >= len(ac):
        return None, 0.0
    i = lo + int(np.argmax(ac[lo:hi]))
    # alleen diepe modulatie telt (tikjes met stiltes ertussen), geen rimpel van een bromtoon
    diepte = min(1.0, float(np.std(e)) / 0.6)
    return float(env_sr / i), float(max(0.0, ac[i]) * diepte)


def _omhullende(y: np.ndarray, sr: int, ms: float = 1.0) -> tuple[np.ndarray, float]:
    import librosa

    hop = max(1, int(sr * ms / 1000))
    env = librosa.feature.rms(y=y, frame_length=hop * 4, hop_length=hop, center=True)[0]
    return env, sr / hop


def meet(y: np.ndarray, sr: int, segmenten: list[Segment] | None = None) -> Kenmerken:
    """Meet alle kenmerken van een (stuk) audio."""
    import librosa

    duur = len(y) / sr
    y = y.astype(np.float32)
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))

    n_fft, hop = 2048, 512
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    S = np.abs(D)
    P = S**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop)[0]
    rms_db = 20 * np.log10(rms + 1e-10)
    actief = rms_db > rms_db.max() - 35
    if actief.sum() < 2:
        actief = np.ones_like(actief, dtype=bool)
    gewicht = rms[actief] / (rms[actief].sum() + 1e-12)

    # --- klankkleur ---
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandbreedte = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    vlak = librosa.feature.spectral_flatness(S=S)[0]
    helder = float(np.sum(centroid[actief] * gewicht))
    t_frames = librosa.frames_to_time(np.arange(len(centroid)), sr=sr, hop_length=hop)
    st_centroid = 12 * np.log2(np.maximum(centroid, 20) / 440.0)
    helder_helling, _ = _theil_sen(t_frames[actief], st_centroid[actief])

    totaal = P[:, actief].sum() + 1e-12
    banden = {naam: _rond(P[(freqs >= lo) & (freqs < hi)][:, actief].sum() / totaal) for naam, lo, hi in BANDEN}

    # --- tonaal vs. percussief ---
    Hc, Pcc = librosa.decompose.hpss(D, margin=1.0)
    H, Pc = np.abs(Hc), np.abs(Pcc)
    harm = float((H**2).sum() / ((H**2).sum() + (Pc**2).sum() + 1e-12))

    # --- omhullende: piekerigheid, aanzet, uitsterven ---
    env, env_sr = _omhullende(y, sr, ms=5.0)
    e2 = env**2
    k = max(1, int(0.05 * len(e2)))
    piekerig = float(np.sort(e2)[-k:].sum() / (e2.sum() + 1e-12))
    env_db = 20 * np.log10(env + 1e-10)
    env_actief = env_db > env_db.max() - 35
    env_std = float(np.std(env_db[env_actief])) if env_actief.sum() > 2 else 0.0

    env1, env1_sr = _omhullende(y, sr, ms=1.0)
    i_piek = int(np.argmax(env1))
    piek = env1[i_piek]
    voor = env1[: i_piek + 1]
    onder10 = np.where(voor < 0.1 * piek)[0]
    i10 = onder10[-1] if len(onder10) else 0
    aanzet_ms = (i_piek - i10) / env1_sr * 1000
    j_piek = int(np.argmax(env))
    na = env_db[j_piek:]
    onder = np.where(na < env_db[j_piek] - 20)[0]
    # sterft het geluid binnen de opname echt uit? (anders is het aanhoudend, geen galm)
    sterft_uit = bool(len(onder))
    uitsterf = (onder[0] if len(onder) else len(na)) / env_sr

    # --- resonanties in het gemiddelde spectrum ---
    from scipy.signal import find_peaks

    from scipy.signal import welch

    a_frames = np.where(actief)[0]
    stuk = y[a_frames[0] * hop : min(len(y), (a_frames[-1] + 1) * hop + n_fft)]
    nper = int(min(8192, max(1024, 2 ** int(np.log2(max(len(stuk), 1024))))))
    wf, wp = welch(stuk, fs=sr, nperseg=nper, noverlap=nper // 2)
    gem = 10 * np.log10(wp + 1e-20)
    bereik = (wf > 60) & (wf < min(16000, sr / 2 - 100))
    pk, eig = find_peaks(gem[bereik], prominence=10, distance=max(2, int(15 / (wf[1] - wf[0]))))
    res_f, res_p = wf[bereik][pk], eig["prominences"]
    volgorde = np.argsort(res_p)[::-1][:6]
    resonanties = [{"hz": _rond(res_f[i], 0), "prominentie_db": _rond(res_p[i], 1)} for i in volgorde]
    partialen = _partialen([r["hz"] for r in resonanties[:5]], tolerantie_hz=1.5 * float(wf[1] - wf[0]))

    # --- toonhoogte (pyin) ---
    y16 = librosa.resample(librosa.istft(Hc, hop_length=hop, length=len(y)), orig_sr=sr, target_sr=16000)
    f0_med, stemhebbend, verloop_st, verloop_r = None, 0.0, 0.0, 0.0
    if len(y16) >= 4096:
        f0, voiced, _ = librosa.pyin(y16, fmin=60, fmax=2000, sr=16000, frame_length=2048, hop_length=256)
        voiced = voiced & np.isfinite(f0)
        stemhebbend = float(voiced.mean()) if len(voiced) else 0.0
        if voiced.sum() >= 5:
            f0_med = float(np.median(f0[voiced]))
            tv = librosa.frames_to_time(np.where(voiced)[0], sr=16000, hop_length=256)
            st = 12 * np.log2(f0[voiced] / 440.0)
            helling, verloop_r = _theil_sen(tv, st)
            verloop_st = helling * float(np.ptp(tv))

    # resonantie die stijgt/daalt (bijv. fles vullen) - ook bij ruisachtige geluiden
    band = (freqs > 150) & (freqs < 8000)
    from scipy.ndimage import median_filter

    piekfreq = freqs[band][np.argmax(median_filter(S[band][:, :], size=(5, 1)), axis=0)]
    if actief.sum() >= 8:
        st_piek = 12 * np.log2(piekfreq[actief] / 440.0)
        h2, r2 = _theil_sen(t_frames[actief], st_piek)
        tot2 = h2 * float(np.ptp(t_frames[actief]))
        if abs(r2) * abs(tot2) > abs(verloop_r) * abs(verloop_st):
            verloop_st, verloop_r = tot2, r2

    # --- aanzetten / ritme ---
    o_hop = 256
    o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=o_hop)
    # robuust normaliseren: één uitschieter (bijv. de allereerste klik na digitale stilte) mag de rest
    # niet wegdrukken
    o_ref = float(np.percentile(o_env, 99)) if o_env.size else 1.0
    o_env = np.clip(o_env / (o_ref + 1e-9), 0, 1.5)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=o_hop, units="frames",
                                              normalize=False, pre_max=2, post_max=2, pre_avg=6, post_avg=6,
                                              wait=3, delta=0.07)
    onsets = librosa.frames_to_time(onset_frames, sr=sr, hop_length=o_hop)
    # zwakke 'onsets' (ruis) negeren
    if len(onset_frames):
        piekwaarden = o_env[onset_frames]
        drempel = max(0.2 * float(np.percentile(piekwaarden, 90)), 3.0 * float(np.median(o_env)) + 1e-6)
        sterk = piekwaarden >= drempel
        # een echte aanzet is ook een luidheidssprong (>= 4 dB), geen rimpel in een aanhoudend geluid
        for j, t0 in enumerate(onsets):
            i = int(t0 * env_sr)
            if i <= int(0.01 * env_sr):  # aanzet aan het begin van de (getrimde) opname
                continue
            na_max = env_db[i : i + max(2, int(0.03 * env_sr))].max(initial=-200)
            voor_min = env_db[max(0, i - int(0.04 * env_sr)) : i + 1].min(initial=200)
            if na_max - voor_min < 4.0:
                sterk[j] = False
        onset_frames, onsets = onset_frames[sterk], onsets[sterk]
    tempo = len(onsets) / max(duur, 0.25)
    ioi_cv = ioi_ratio = ioi_dalend = kracht_trend = None
    if len(onsets) >= 3:
        ioi = np.diff(onsets)
        ioi_cv = float(np.std(ioi) / (np.mean(ioi) + 1e-9))
        r = ioi[1:] / (ioi[:-1] + 1e-9)
        ioi_ratio = float(np.median(r))
        ioi_dalend = float(np.mean(r < 0.97))
        kracht = o_env[onset_frames]
        kracht_trend, _ = _theil_sen(np.arange(len(kracht), dtype=float), kracht / (kracht.max() + 1e-9))

    # --- ratelen: modulatiespectrum van de omhullende ---
    ratel_hz, ratel_sterkte = _ratel(env1, env1_sr)

    km = Kenmerken(
        duur=duur,
        luidheid_db=float(np.mean(rms_db[actief])),
        piekerigheid=piekerig,
        omhullende_std_db=env_std,
        aanzet_ms=float(aanzet_ms),
        uitsterf_s=float(uitsterf),
        helderheid_hz=helder,
        helderheid_helling=float(helder_helling),
        bandbreedte_hz=float(np.sum(bandbreedte[actief] * gewicht)),
        rolloff_hz=float(np.sum(rolloff[actief] * gewicht)),
        vlakheid=float(np.sum(vlak[actief] * gewicht)),
        harmonisch_aandeel=harm,
        banden=banden,
        f0_mediaan_hz=f0_med,
        stemhebbend=stemhebbend,
        toon_verloop_st=float(verloop_st),
        toon_verloop_r=float(abs(verloop_r)),
        resonanties=resonanties,
        partialen=partialen,
        onsets=[_rond(t) for t in onsets],
        onset_tempo=float(tempo),
        ioi_cv=ioi_cv,
        ioi_ratio=ioi_ratio,
        ioi_dalend=ioi_dalend,
        kracht_trend=kracht_trend,
        ratel_hz=ratel_hz,
        ratel_sterkte=float(ratel_sterkte),
        sterft_uit=sterft_uit,
    )
    km.eigenschappen = eigenschappen(km, segmenten)
    return km


def eigenschappen(k: Kenmerken, segmenten: list[Segment] | None = None,
                  segment_helderheid: list[float] | None = None) -> dict[str, float]:
    """Vertaal metingen naar scores 0..1 per eigenschap (zie KENMERKEN)."""
    e: dict[str, float] = {}
    max_prom = max([r["prominentie_db"] for r in k.resonanties], default=0.0)

    e["ruis"] = _sig(k.vlakheid, 0.10, 0.035) * (1 - 0.6 * _sig(k.harmonisch_aandeel, 0.8, 0.05))
    e["toon"] = max(
        _sig(k.harmonisch_aandeel, 0.7, 0.06) * (1 - _sig(k.vlakheid, 0.12, 0.04)),
        _sig(k.stemhebbend, 0.45, 0.1) * (1 - _sig(k.vlakheid, 0.15, 0.04)),
    )
    e["galm"] = (_sig(k.uitsterf_s, 0.25, 0.08) * _sig(max_prom, 14, 3) * (1 - 0.5 * e["ruis"])
                 * (1.0 if k.sterft_uit else 0.25))
    e["impuls"] = _sig(k.piekerigheid, 0.30, 0.07) * _sig(-k.aanzet_ms, -25, 8)
    e["continu"] = _sig(-k.omhullende_std_db, -6, 1.5) * _sig(k.duur, 1.0, 0.3) * (1 - _sig(k.piekerigheid, 0.25, 0.05))
    e["snelle_tikjes"] = max(
        _sig(k.onset_tempo, 9, 2) * _sig(k.piekerigheid, 0.10, 0.04),
        _sig(k.ratel_sterkte, 0.35, 0.07) * _sig(k.onset_tempo, 4, 1.5),
    )
    if k.ioi_cv is not None and len(k.onsets) >= 3:
        e["ritmisch"] = _sig(len(k.onsets), 3.5, 0.7) * _sig(-k.ioi_cv, -0.22, 0.06)
    else:
        e["ritmisch"] = 0.0
    if k.ioi_ratio is not None and len(k.onsets) >= 4:
        zachter = 1.0 if (k.kracht_trend or 0) < -0.02 else 0.5
        e["stuiteren"] = (_sig(len(k.onsets), 4.0, 0.7) * _sig(k.ioi_dalend or 0, 0.65, 0.08)
                          * _sig(-k.ioi_ratio, -0.93, 0.03) * zachter)
    else:
        e["stuiteren"] = 0.0
    consistent = _sig(k.toon_verloop_r, 0.5, 0.1)
    e["stijgend"] = _sig(k.toon_verloop_st, 3.0, 1.0) * consistent
    e["dalend"] = _sig(-k.toon_verloop_st, 3.0, 1.0) * consistent
    e["laag"] = _sig(math.log2(700 / max(k.helderheid_hz, 20)), 0, 0.35)
    e["hoog"] = _sig(math.log2(max(k.helderheid_hz, 20) / 3500), 0, 0.35)
    e["kort"] = _sig(-k.duur, -1.5, 0.3)
    e["lang"] = _sig(k.duur, 4.0, 0.7)
    e["meerdelig"] = 0.0
    if segmenten and len(segmenten) >= 2 and segment_helderheid:
        spreiding = float(np.ptp(np.log2(np.maximum(segment_helderheid, 20))))
        e["meerdelig"] = _sig(len(segmenten), 1.5, 0.2) * _sig(spreiding, 0.6, 0.2)
    return {naam: round(v, 3) for naam, v in e.items()}


@dataclass
class Analyse:
    sr: int
    duur: float
    geheel: Kenmerken
    segmenten: list[Segment]
    per_segment: list[Kenmerken]

    def als_dict(self) -> dict:
        return {
            "duur": _rond(self.duur),
            "geheel": self.geheel.als_dict(),
            "segmenten": [
                {"start": s.start, "eind": s.eind, "kenmerken": k.als_dict()}
                for s, k in zip(self.segmenten, self.per_segment)
            ],
        }


def analyseer(y: np.ndarray, sr: int, segmenten: list[Segment]) -> Analyse:
    per_segment = [meet(s.knip(y, sr), sr) for s in segmenten]
    geheel = meet(y, sr, segmenten)
    if len(segmenten) >= 2:
        geheel.eigenschappen = eigenschappen(geheel, segmenten, [k.helderheid_hz for k in per_segment])
        # stijgen/dalen alleen binnen een doorlopend deel meten, niet van deel naar deel
        for naam in ("stijgend", "dalend"):
            geheel.eigenschappen[naam] = max(
                [k.eigenschappen[naam] for s, k in zip(segmenten, per_segment) if s.duur >= 0.5], default=0.0)
    return Analyse(sr=sr, duur=len(y) / sr, geheel=geheel, segmenten=segmenten, per_segment=per_segment)


def _hz(f: float) -> str:
    return f"{f / 1000:.1f} kHz" if f >= 1000 else f"{f:.0f} Hz"


def observaties(a: Analyse) -> list[str]:
    """Menselijke, Nederlandstalige waarnemingen + wat ze kunnen betekenen."""
    k, e = a.geheel, a.geheel.eigenschappen
    obs: list[str] = []

    n = len(a.segmenten)
    delen = ", ".join(f"{s.start:.2f}-{s.eind:.2f}s" for s in a.segmenten[:8])
    obs.append(f"Duur {k.duur:.2f} s, {n} afzonderlijk(e) deel/delen ({delen}{', ...' if n > 8 else ''}).")

    sterkst = max(k.banden, key=k.banden.get)
    obs.append(f"Klankkleur: zwaartepunt rond {_hz(k.helderheid_hz)}, meeste energie in {sterkst} "
               f"({k.banden[sterkst] * 100:.0f}%).")
    if e["laag"] > 0.6:
        obs.append("Veel lage tonen -> groot, zwaar of dof object (bonk, dichtslaande klep/deur, rommelen).")
    if e["hoog"] > 0.6:
        obs.append("Veel hoge tonen -> klein, licht of scherp (sleutels, glas, sissen, spuitbus, papier).")

    if e["ruis"] > 0.6:
        obs.append(f"Ruisachtig (vlakheid {k.vlakheid:.2f}) -> lucht, water, schuren, borstelen, papier of plastic.")
    if e["toon"] > 0.6:
        f0 = f" rond {_hz(k.f0_mediaan_hz)}" if k.f0_mediaan_hz else ""
        obs.append(f"Duidelijk tonaal ({k.harmonisch_aandeel * 100:.0f}% tonale energie){f0} -> "
                   "piep, fluit, snaar, motor/apparaat of resonerend voorwerp.")
    if k.resonanties:
        top = ", ".join(f"{_hz(r['hz'])} (+{r['prominentie_db']:.0f} dB)" for r in k.resonanties[:3])
        extra = ""
        if k.partialen == "inharmonisch":
            extra = " Boventonen zijn géén veelvouden -> typisch voor metaal, glas, bellen, potten en pannen."
        elif k.partialen == "harmonisch":
            extra = " Boventonen zijn veelvouden -> snaar, blaasgeluid, stem, motor of buis."
        obs.append(f"Opvallende resonanties: {top}.{extra}")
    if e["galm"] > 0.5:
        obs.append(f"Klinkt na ({k.uitsterf_s:.2f} s tot -20 dB) -> hard, resonerend materiaal (metaal, glas, keramiek).")
    elif k.uitsterf_s < 0.08 and e["impuls"] > 0.5:
        obs.append("Sterft direct uit -> dof/gedempt materiaal (hout, karton, plastic, stof) of zachte ondergrond.")

    if e["impuls"] > 0.6:
        obs.append(f"Impulsief: scherpe aanzet ({k.aanzet_ms:.0f} ms) -> botsing, tik, klik, klap.")
    if e["continu"] > 0.6:
        obs.append("Aanhoudend en gelijkmatig -> apparaat, motor, stromend water/lucht, wrijven.")
    if e["snelle_tikjes"] > 0.5:
        r = f" (~{k.ratel_hz:.0f} tikjes/s)" if k.ratel_hz and k.ratel_sterkte > 0.2 else ""
        obs.append(f"Snelle reeks tikjes{r} -> rits, ratel, freewheel, kraaltjes, kaarten schudden, tandwiel.")
    if e["ritmisch"] > 0.5:
        obs.append(f"Regelmatig ritme ({k.onset_tempo:.1f} per seconde, regelmaat {k.ioi_cv:.2f}) -> klok, stappen, "
                   "kloppen, pompen, knippen.")
    if e["stuiteren"] > 0.5:
        obs.append("Tikken komen steeds sneller en zachter -> iets valt en stuitert of tolt uit "
                   "(bal, dop, munt, knikker).")
    if e["stijgend"] > 0.5:
        obs.append(f"Toon/resonantie stijgt ({k.toon_verloop_st:+.1f} halve tonen) -> fles/glas vullen, opwinden, "
                   "iets dat sneller gaat draaien.")
    if e["dalend"] > 0.5:
        obs.append(f"Toon/resonantie daalt ({k.toon_verloop_st:+.1f} halve tonen) -> leeglopen, afremmen, uitdraaien.")
    if e["meerdelig"] > 0.5:
        obs.append("De delen klinken duidelijk verschillend -> waarschijnlijk een handeling in stappen "
                   "(bijv. openen + neerzetten).")
    if len(obs) <= 3:
        obs.append("Geen uitgesproken kenmerken gevonden; vertrouw vooral op de AI-vergelijking.")
    return obs
