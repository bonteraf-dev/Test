"""AI-herkenning met CLAP (Contrastive Language-Audio Pretraining).

CLAP zet audio en tekst in dezelfde 'betekenisruimte'. Daardoor kunnen we de opname
vergelijken met honderden beschrijvingen ('a zipper being zipped up', ...) zonder het
model daarvoor te trainen (zero-shot). Ook kan het twee opnames met elkaar vergelijken.

Vereist: pip install "geluidspeurder[ai]"  (torch + transformers). Bij het eerste gebruik
wordt het model (~600 MB) van huggingface.co gedownload en daarna lokaal bewaard.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from .kandidaten import Kandidaat

MODEL_STANDAARD = os.environ.get("GELUID_CLAP_MODEL", "laion/larger_clap_general")
SR = 48_000
VENSTER_S = 10.0  # CLAP is getraind op fragmenten van ~10 s
SJABLONEN = ["{}", "the sound of {}", "a recording of {}"]


def beschikbaar() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        return False, f"{e.name} ontbreekt (installeer met: pip install \"geluidspeurder[ai]\")"
    return True, ""


def _als_tensor(uit):
    """transformers-versies verschillen in wat get_*_features teruggeeft."""
    import torch

    if isinstance(uit, torch.Tensor):
        return uit
    for naam in ("text_embeds", "audio_embeds", "pooler_output"):
        v = getattr(uit, naam, None)
        if isinstance(v, torch.Tensor):
            return v
    return uit[0]


def _normeer(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


class ClapScorer:
    def __init__(self, model: str = MODEL_STANDAARD, cache_map: str | Path | None = None):
        ok, fout = beschikbaar()
        if not ok:
            raise RuntimeError(fout)
        import torch
        from transformers import ClapModel, ClapProcessor

        self.torch = torch
        self.model_naam = model
        self.processor = ClapProcessor.from_pretrained(model)
        self.model = ClapModel.from_pretrained(model).eval()
        self.apparaat = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.apparaat)
        cache_map = Path(cache_map or Path.home() / ".cache" / "geluidspeurder")
        cache_map.mkdir(parents=True, exist_ok=True)
        tag = hashlib.sha1(model.encode()).hexdigest()[:10]
        self._cache_pad = cache_map / f"clap_tekst_{tag}.npz"
        self._cache: dict[str, np.ndarray] = {}
        if self._cache_pad.exists():
            data = np.load(self._cache_pad, allow_pickle=False)
            self._cache = dict(zip(data["sleutels"].tolist(), data["emb"]))

    # ------------------------------------------------------------------ tekst
    def tekst_embeddings(self, teksten: list[str]) -> np.ndarray:
        sleutels = [hashlib.sha1(t.encode()).hexdigest() for t in teksten]
        nieuw = [(s, t) for s, t in dict(zip(sleutels, teksten)).items() if s not in self._cache]
        for i in range(0, len(nieuw), 64):
            blok = nieuw[i : i + 64]
            inv = self.processor(text=[t for _, t in blok], return_tensors="pt", padding=True)
            inv = {k: v.to(self.apparaat) for k, v in inv.items()}
            with self.torch.no_grad():
                emb = _als_tensor(self.model.get_text_features(**inv)).cpu().numpy()
            for (s, _), e in zip(blok, _normeer(emb)):
                self._cache[s] = e.astype(np.float32)
        if nieuw:
            np.savez(self._cache_pad, sleutels=np.array(list(self._cache)), emb=np.stack(list(self._cache.values())))
        return np.stack([self._cache[s] for s in sleutels])

    def kandidaat_embeddings(self, kandidaten: list[Kandidaat]) -> np.ndarray:
        """Eén embedding per kandidaat: gemiddelde over alle beschrijvingen x sjablonen."""
        teksten, eigenaar = [], []
        for i, k in enumerate(kandidaten):
            for zin in k.en:
                for sj in SJABLONEN:
                    teksten.append(sj.format(zin))
                    eigenaar.append(i)
        emb = self.tekst_embeddings(teksten)
        eigenaar = np.array(eigenaar)
        return _normeer(np.stack([emb[eigenaar == i].mean(axis=0) for i in range(len(kandidaten))]))

    # ------------------------------------------------------------------ audio
    def _audio_batch(self, stukken: list[np.ndarray]) -> np.ndarray:
        try:
            inv = self.processor(audio=stukken, sampling_rate=SR, return_tensors="pt")
        except (TypeError, ValueError):
            inv = self.processor(audios=stukken, sampling_rate=SR, return_tensors="pt")
        inv = {k: v.to(self.apparaat) for k, v in inv.items() if hasattr(v, "to")}
        with self.torch.no_grad():
            return _normeer(_als_tensor(self.model.get_audio_features(**inv)).cpu().numpy())

    def audio_embedding(self, y: np.ndarray, sr: int = SR) -> np.ndarray:
        """Embedding van een opname; langer dan 10 s -> overlappende vensters, gemiddeld."""
        if sr != SR:
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        n = int(VENSTER_S * SR)
        if len(y) <= n:
            stukken = [y.astype(np.float32)]
        else:
            stap = n // 2
            stukken = [y[i : i + n].astype(np.float32) for i in range(0, len(y) - n // 4, stap)]
        emb = self._audio_batch(stukken)
        return _normeer(emb.mean(axis=0))

    # ------------------------------------------------------------------ scores
    def gelijkenis(self, audio_emb: np.ndarray, kandidaten: list[Kandidaat]) -> np.ndarray:
        return self.kandidaat_embeddings(kandidaten) @ audio_emb

    def vrije_tekst(self, audio_emb: np.ndarray, beschrijvingen: list[str]) -> np.ndarray:
        """Score losse Engelse zinnen (bijv. ideeën van Claude of van jezelf)."""
        return self.tekst_embeddings(beschrijvingen) @ audio_emb
