from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


@dataclass
class VectorIndex:
    """Small FAISS-first vector index with a deterministic NumPy fallback."""

    vectors: np.ndarray
    backend: str = "numpy"
    index_type: str = "flat"
    faiss_index: Any | None = None

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        *,
        index_type: str = "hnsw",
        prefer_faiss: bool = True,
        hnsw_m: int = 32,
    ) -> VectorIndex:
        normalized = l2_normalize(vectors)
        if prefer_faiss:
            try:
                import faiss  # type: ignore

                dim = int(normalized.shape[1])
                if index_type.lower() == "hnsw":
                    index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
                    index.hnsw.efConstruction = 200
                    index.hnsw.efSearch = 64
                else:
                    index = faiss.IndexFlatIP(dim)
                index.add(normalized)
                return cls(normalized, "faiss", index_type.lower(), index)
            except Exception:
                pass
        return cls(normalized, "numpy", "flat", None)

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vectors = l2_normalize(query)
        top_k = min(top_k, len(self.vectors))
        if self.backend == "faiss" and self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query_vectors.astype(np.float32), top_k)
            return indices[0].astype(np.int64), scores[0].astype(np.float32)

        scores = self.vectors @ query_vectors[0]
        if top_k >= len(scores):
            order = np.argsort(-scores)
        else:
            unsorted = np.argpartition(-scores, top_k - 1)[:top_k]
            order = unsorted[np.argsort(-scores[unsorted])]
        return order.astype(np.int64), scores[order].astype(np.float32)

    def save(self, directory: str | Path, name: str) -> Path:
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "backend": self.backend,
            "index_type": self.index_type,
            "dim": int(self.vectors.shape[1]),
            "count": int(self.vectors.shape[0]),
            "name": name,
        }
        manifest_path = output_dir / f"{name}_index.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        np.save(output_dir / f"{name}_vectors.npy", self.vectors.astype(np.float32))
        if self.backend == "faiss" and self.faiss_index is not None:
            try:
                import faiss  # type: ignore

                faiss.write_index(self.faiss_index, str(output_dir / f"{name}.faiss"))
            except Exception:
                pass
        return manifest_path

    @classmethod
    def load(cls, directory: str | Path, name: str) -> VectorIndex:
        input_dir = Path(directory)
        manifest = json.loads((input_dir / f"{name}_index.json").read_text(encoding="utf-8"))
        vectors = np.load(input_dir / f"{name}_vectors.npy").astype(np.float32)
        if manifest.get("backend") == "faiss" and (input_dir / f"{name}.faiss").exists():
            try:
                import faiss  # type: ignore

                index = faiss.read_index(str(input_dir / f"{name}.faiss"))
                return cls(vectors, "faiss", str(manifest.get("index_type", "hnsw")), index)
            except Exception:
                pass
        return cls(vectors, "numpy", "flat", None)
