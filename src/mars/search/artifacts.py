from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mars.retrieval import VectorIndex
from mars.retrieval.vector_index import l2_normalize
from mars.search.encoders import SearchEncoder

TEXT_COLUMNS = (
    "name",
    "description",
    "category_l3",
    "category_l2",
    "category_l1",
    "category",
    "leaf_category",
    "mid_category",
    "top_category",
    "color",
    "price_tier",
)


def product_search_text(row: pd.Series) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for column in TEXT_COLUMNS:
        if column in row:
            for value in _field_values(row[column]):
                _append_unique(parts, seen, value)
    for value in _field_values(row.get("style_tags")):
        _append_unique(parts, seen, value)
    for value in _category_aliases(parts):
        _append_unique(parts, seen, value)
    return " ".join(part for part in parts if part)


def _append_unique(parts: list[str], seen: set[str], value: Any) -> None:
    text = _clean_text(value)
    key = text.lower()
    if text and key not in seen:
        seen.add(key)
        parts.append(text)


def _field_values(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [_clean_text(item) for item in value if _clean_text(item)]
    text = _clean_text(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        tokens = re.findall(r"'([^']+)'|\"([^\"]+)\"", text)
        parsed = [_clean_text(left or right) for left, right in tokens]
        if parsed:
            return [item for item in parsed if item]
    return [text]


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace("/", " ")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _category_aliases(parts: list[str]) -> list[str]:
    text = " ".join(parts).lower()
    aliases: list[str] = []
    alias_map = {
        "vest top": ["tank top", "sleeveless top"],
        "swimwear bottom": ["swim bottom", "bikini bottom", "bikini bottoms"],
        "swimwear": ["swimsuit", "bathing suit"],
        "trousers": ["pants"],
        "leggings": ["tights"],
        "hoodie": ["hooded sweatshirt"],
        "sweater": ["knitwear", "jumper"],
        "t-shirt": ["tee shirt", "tee"],
        "t shirt": ["tee shirt", "tee"],
        "shirt": ["button shirt"],
        "bra": ["bralette", "underwear"],
        "lingeries tights": ["lingerie", "tights", "underwear"],
        "dress": ["dresses"],
        "skirt": ["skirts"],
        "jacket": ["outerwear"],
        "denim": ["jeans"],
        "high-waisted": ["high waist"],
        "midrise": ["mid rise", "mid waist"],
        "mid waist": ["midrise", "mid rise"],
    }
    for key, values in alias_map.items():
        if key in text:
            aliases.extend(values)
    if re.search(r"\bhw\b", text):
        aliases.extend(["high waist", "high-waisted"])
    if re.search(r"\btrs\b", text):
        aliases.append("trousers")
    return aliases


@dataclass
class SearchArtifacts:
    metadata: pd.DataFrame
    text_index: VectorIndex
    image_index: VectorIndex
    joint_index: VectorIndex
    manifest: dict[str, Any]

    @classmethod
    def load(cls, artifact_dir: str | Path) -> SearchArtifacts:
        base = Path(artifact_dir)
        return cls(
            metadata=pd.read_parquet(base / "product_meta.parquet"),
            text_index=VectorIndex.load(base, "text"),
            image_index=VectorIndex.load(base, "image"),
            joint_index=VectorIndex.load(base, "joint"),
            manifest=json.loads((base / "index_manifest.json").read_text(encoding="utf-8")),
        )


def build_search_artifacts(
    products_path: str | Path,
    artifact_dir: str | Path,
    *,
    encoder: SearchEncoder,
    index_type: str = "hnsw",
    batch_size: int = 128,
    image_text_fallback: bool = True,
) -> SearchArtifacts:
    products = pd.read_parquet(products_path)
    if "product_id" not in products or "name" not in products or "price" not in products:
        raise ValueError("products parquet must include product_id, name, and price")

    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    products = products.reset_index(drop=True).copy()
    products["search_text"] = [product_search_text(row) for _, row in products.iterrows()]

    text_embeddings = _encode_batches(
        encoder.encode_texts,
        products["search_text"].tolist(),
        batch_size,
    )
    try:
        if image_text_fallback and encoder.name == "fallback":
            image_embeddings = text_embeddings.copy()
        else:
            image_embeddings = _encode_image_batches_with_fallback(
                encoder=encoder,
                images=_image_inputs(products),
                text_embeddings=text_embeddings,
                batch_size=batch_size,
            )
    except Exception:
        if not image_text_fallback:
            raise
        image_embeddings = text_embeddings.copy()

    joint_embeddings = l2_normalize((0.45 * text_embeddings) + (0.55 * image_embeddings))
    np.save(output_dir / "text_embeddings.npy", text_embeddings)
    np.save(output_dir / "image_embeddings.npy", image_embeddings)
    np.save(output_dir / "joint_embeddings.npy", joint_embeddings)
    products.to_parquet(output_dir / "product_meta.parquet", index=False)

    text_index = VectorIndex.build(text_embeddings, index_type=index_type)
    image_index = VectorIndex.build(image_embeddings, index_type=index_type)
    joint_index = VectorIndex.build(joint_embeddings, index_type=index_type)
    text_index.save(output_dir, "text")
    image_index.save(output_dir, "image")
    joint_index.save(output_dir, "joint")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "encoder_type": encoder.name,
        "embedding_dim": int(text_embeddings.shape[1]),
        "product_count": int(len(products)),
        "index_type": index_type,
        "indexes": {
            "text": text_index.backend,
            "image": image_index.backend,
            "joint": joint_index.backend,
        },
    }
    (output_dir / "index_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return SearchArtifacts(products, text_index, image_index, joint_index, manifest)


def _image_inputs(products: pd.DataFrame) -> list[str]:
    if "image_path" not in products:
        return products["search_text"].tolist()
    return [str(value) if pd.notna(value) else "" for value in products["image_path"]]


def _encode_batches(
    func: Callable[[list[Any]], np.ndarray],
    values: list[Any],
    batch_size: int,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    progress_every = int(os.environ.get("MARS_EMBED_PROGRESS_EVERY", "0") or 0)
    started_at = time.perf_counter()
    total_batches = max(1, (len(values) + batch_size - 1) // batch_size)
    for batch_idx, start in enumerate(range(0, len(values), batch_size), start=1):
        batches.append(func(values[start : start + batch_size]))
        if progress_every > 0 and (
            batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == total_batches
        ):
            elapsed = time.perf_counter() - started_at
            print(
                "[search-artifacts] encoded "
                f"{min(start + batch_size, len(values))}/{len(values)} "
                f"rows ({batch_idx}/{total_batches} batches, {elapsed:.1f}s)",
                flush=True,
            )
    return np.vstack(batches).astype(np.float32)


def _encode_image_batches_with_fallback(
    *,
    encoder: SearchEncoder,
    images: list[Any],
    text_embeddings: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    progress_every = int(os.environ.get("MARS_EMBED_PROGRESS_EVERY", "0") or 0)
    started_at = time.perf_counter()
    total_batches = max(1, (len(images) + batch_size - 1) // batch_size)
    fallback_count = 0
    for batch_idx, start in enumerate(range(0, len(images), batch_size), start=1):
        end = min(start + batch_size, len(images))
        batch_images = images[start:end]
        try:
            encoded = encoder.encode_images(batch_images)
        except Exception:
            rows: list[np.ndarray] = []
            for offset, image in enumerate(batch_images):
                try:
                    rows.append(encoder.encode_images([image])[0])
                except Exception:
                    rows.append(text_embeddings[start + offset])
                    fallback_count += 1
            encoded = np.vstack(rows).astype(np.float32)
        batches.append(encoded)
        if progress_every > 0 and (
            batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == total_batches
        ):
            elapsed = time.perf_counter() - started_at
            print(
                "[search-artifacts] encoded images "
                f"{end}/{len(images)} rows ({batch_idx}/{total_batches} batches, "
                f"{elapsed:.1f}s, fallback={fallback_count})",
                flush=True,
            )
    return np.vstack(batches).astype(np.float32)
