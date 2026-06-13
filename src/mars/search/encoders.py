from __future__ import annotations

import base64
import hashlib
import io
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageOps

from mars.retrieval.vector_index import l2_normalize


class SearchEncoder(Protocol):
    dim: int
    name: str

    def encode_texts(self, texts: list[str]) -> np.ndarray: ...

    def encode_images(self, images: list[str | bytes | Image.Image]) -> np.ndarray: ...


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _hash_int(value: str | bytes, *, digest_size: int = 8) -> int:
    data = value if isinstance(value, bytes) else value.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=digest_size).digest(), "little")


def load_image(image: str | bytes | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    if isinstance(image, bytes):
        return ImageOps.exif_transpose(Image.open(io.BytesIO(image))).convert("RGB")
    candidate = str(image)
    if candidate.startswith("data:image") and "," in candidate:
        candidate = candidate.split(",", 1)[1]
    if len(candidate) > 200 and not Path(candidate).exists():
        decoded = base64.b64decode(candidate)
        return ImageOps.exif_transpose(Image.open(io.BytesIO(decoded))).convert("RGB")
    return ImageOps.exif_transpose(Image.open(candidate)).convert("RGB")


@dataclass
class FallbackEncoder:
    """Deterministic encoder used when CLIP is unavailable or disabled."""

    dim: int = 128
    seed: int = 42
    name: str = "fallback"

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            tokens = _TOKEN_RE.findall((text or "").lower()) or ["empty"]
            for position, token in enumerate(tokens):
                bucket = _hash_int(f"{self.seed}:text:{token}") % self.dim
                sign = 1.0 if (_hash_int(f"sign:{token}") % 2 == 0) else -1.0
                vec[bucket] += sign / np.sqrt(position + 1.0)
            rows.append(vec)
        return l2_normalize(np.vstack(rows))

    def encode_images(self, images: list[str | bytes | Image.Image]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for image in images:
            try:
                img = load_image(image).resize((64, 64))
                arr = np.asarray(img, dtype=np.float32) / 255.0
                features: list[float] = []
                for channel in range(3):
                    hist, _ = np.histogram(arr[:, :, channel], bins=16, range=(0.0, 1.0))
                    features.extend(hist.astype(np.float32).tolist())
                features.extend(arr.mean(axis=(0, 1)).tolist())
                features.extend(arr.std(axis=(0, 1)).tolist())
                gray = arr.mean(axis=2)
                edge = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
                features.append(float(edge))
            except Exception:
                features = [float(_hash_int(str(image)) % 997)]

            vec = np.zeros(self.dim, dtype=np.float32)
            for idx, value in enumerate(features):
                bucket = _hash_int(f"{self.seed}:image:{idx}") % self.dim
                vec[bucket] += float(value)
            rows.append(vec)
        return l2_normalize(np.vstack(rows))


class ClipEncoder:
    """Optional CLIP wrapper. Imports are lazy so fallback mode stays lightweight."""

    def __init__(self, model_name: str, *, device: str = "cpu") -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        local_files_only = _clip_local_files_only()
        self.processor = CLIPProcessor.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=local_files_only,
        )
        self.model = CLIPModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).to(device)
        self.model.eval()
        self.device = device
        self.dim = int(self.model.config.projection_dim)
        self.name = f"clip:{model_name}"
        self._text_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._text_cache_size = 4096

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if len(texts) == 1:
            cached = self._text_cache_get(texts[0])
            if cached is not None:
                return cached.copy()
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(
            self.device
        )
        with self.torch.inference_mode():
            embeddings = self.model.get_text_features(**inputs)
        embeddings = _tensor_from_clip_output(embeddings)
        encoded = l2_normalize(embeddings.detach().cpu().numpy().astype(np.float32))
        if len(texts) == 1:
            self._text_cache_set(texts[0], encoded)
        return encoded

    def encode_images(self, images: list[str | bytes | Image.Image]) -> np.ndarray:
        loaded = [load_image(image) for image in images]
        inputs = self.processor(images=loaded, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            embeddings = self.model.get_image_features(**inputs)
        embeddings = _tensor_from_clip_output(embeddings)
        return l2_normalize(embeddings.detach().cpu().numpy().astype(np.float32))

    def _text_cache_get(self, text: str) -> np.ndarray | None:
        key = str(text)
        value = self._text_cache.get(key)
        if value is None:
            return None
        self._text_cache.move_to_end(key)
        return value

    def _text_cache_set(self, text: str, value: np.ndarray) -> None:
        key = str(text)
        self._text_cache[key] = value.copy()
        self._text_cache.move_to_end(key)
        while len(self._text_cache) > self._text_cache_size:
            self._text_cache.popitem(last=False)


def create_encoder(
    *,
    encoder_type: str,
    dim: int,
    seed: int,
    clip_model: str,
    allow_fallback: bool,
) -> SearchEncoder:
    if encoder_type.lower() == "clip":
        try:
            return ClipEncoder(clip_model)
        except Exception:
            if not allow_fallback:
                raise
    return FallbackEncoder(dim=dim, seed=seed)


def _clip_local_files_only() -> bool:
    raw = os.environ.get("MARS_CLIP_LOCAL_FILES_ONLY", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _tensor_from_clip_output(output):
    if hasattr(output, "detach"):
        return output
    if hasattr(output, "image_embeds"):
        return output.image_embeds
    if hasattr(output, "text_embeds"):
        return output.text_embeds
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"Unsupported CLIP output type: {type(output)!r}")
