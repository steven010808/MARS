from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mars.config import load_config
from mars.retrieval import VectorIndex


def _load_embeddings(artifact_dir: Path, name: str) -> np.ndarray:
    path = artifact_dir / f"{name}_embeddings.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing embedding file: {path}")
    return np.load(path).astype(np.float32)


def _faiss_index_type(path: Path) -> str:
    import faiss  # type: ignore

    index = faiss.read_index(str(path))
    return type(index).__name__


def _write_manifest(
    manifest_path: Path,
    *,
    index_type: str,
    built_indexes: dict[str, VectorIndex],
    faiss_types: dict[str, str],
) -> None:
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "index_type": index_type,
            "indexes": {name: index.backend for name, index in built_indexes.items()},
            "faiss_index_classes": faiss_types,
            "faiss_rebuilt_at": datetime.now(UTC).isoformat(),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild search FAISS indexes from existing embedding .npy files. "
            "This does not re-encode CLIP embeddings."
        )
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", default="full")
    parser.add_argument("--index-type", default=None, help="Defaults to search.index_type.")
    parser.add_argument("--hnsw-m", type=int, default=32)
    args = parser.parse_args()

    config = load_config(args.config, mode=args.mode)
    artifact_dir = config.paths.artifacts_dir / "search"
    index_type = str(args.index_type or config.search.index_type).lower()
    if index_type != "hnsw":
        raise ValueError(
            "submission spec requires an ANN index such as HNSW. "
            "Use --index-type hnsw unless intentionally testing another mode."
        )

    built_indexes: dict[str, VectorIndex] = {}
    faiss_types: dict[str, str] = {}
    for name in ("text", "image", "joint"):
        embeddings = _load_embeddings(artifact_dir, name)
        index = VectorIndex.build(embeddings, index_type=index_type, hnsw_m=args.hnsw_m)
        if index.backend != "faiss":
            raise RuntimeError("faiss is required to rebuild submission HNSW indexes")
        index.save(artifact_dir, name)
        built_indexes[name] = index
        faiss_types[name] = _faiss_index_type(artifact_dir / f"{name}.faiss")

    _write_manifest(
        artifact_dir / "index_manifest.json",
        index_type=index_type,
        built_indexes=built_indexes,
        faiss_types=faiss_types,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "artifact_dir": str(artifact_dir),
                "index_type": index_type,
                "faiss_index_classes": faiss_types,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
