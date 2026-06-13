from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mars.config import load_config
from mars.retrieval import VectorIndex
from mars.retrieval.vector_index import l2_normalize
from mars.search.artifacts import _encode_batches, product_search_text
from mars.search.encoders import create_encoder


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild only the text/joint search artifacts after changing H&M "
            "product metadata text. Existing image embeddings are reused."
        )
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", default="full")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow deterministic fallback text embeddings. By default this script requires CLIP.",
    )
    args = parser.parse_args()

    os.environ["MARS_EMBED_PROGRESS_EVERY"] = str(max(args.progress_every, 0))
    config = load_config(args.config, mode=args.mode)
    artifact_dir = config.paths.artifacts_dir / "search"
    products_path = config.paths.processed_dir / "products.parquet"
    metadata_path = artifact_dir / "product_meta.parquet"
    manifest_path = artifact_dir / "index_manifest.json"
    image_embeddings_path = artifact_dir / "image_embeddings.npy"

    metadata = pd.read_parquet(metadata_path)
    products = pd.read_parquet(products_path)
    products["product_id"] = products["product_id"].astype(str)
    metadata["product_id"] = metadata["product_id"].astype(str)

    ordered_ids = metadata["product_id"].tolist()
    product_by_id = products.drop_duplicates("product_id").set_index("product_id")
    missing = [product_id for product_id in ordered_ids if product_id not in product_by_id.index]
    if missing:
        raise ValueError(f"processed products missing {len(missing)} artifact product IDs")
    ordered_products = product_by_id.loc[ordered_ids].reset_index()
    ordered_products["search_text"] = [
        product_search_text(row) for _, row in ordered_products.iterrows()
    ]

    encoder = create_encoder(
        encoder_type=config.search.encoder_type,
        dim=config.search.embedding_dim,
        seed=config.seed,
        clip_model=config.search.clip_model,
        allow_fallback=args.allow_fallback,
    )
    if not args.allow_fallback and not encoder.name.startswith("clip:"):
        raise RuntimeError(
            f"Expected CLIP encoder for submission artifacts, got {encoder.name!r}. "
            "Install/load transformers+torch CLIP or rerun explicitly with --allow-fallback."
        )
    text_embeddings = _encode_batches(
        encoder.encode_texts,
        ordered_products["search_text"].tolist(),
        args.batch_size,
    )

    if image_embeddings_path.exists():
        image_embeddings = np.load(image_embeddings_path).astype(np.float32)
        if image_embeddings.shape != text_embeddings.shape:
            raise ValueError(
                "image/text embedding shape mismatch: "
                f"{image_embeddings.shape} vs {text_embeddings.shape}"
            )
    else:
        image_embeddings = text_embeddings.copy()

    joint_embeddings = l2_normalize((0.45 * text_embeddings) + (0.55 * image_embeddings))

    np.save(artifact_dir / "text_embeddings.npy", text_embeddings)
    np.save(artifact_dir / "joint_embeddings.npy", joint_embeddings)
    ordered_products.to_parquet(metadata_path, index=False)

    text_index = VectorIndex.build(text_embeddings, index_type=config.search.index_type)
    joint_index = VectorIndex.build(joint_embeddings, index_type=config.search.index_type)
    text_index.save(artifact_dir, "text")
    joint_index.save(artifact_dir, "joint")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "encoder_type": encoder.name,
            "embedding_dim": int(text_embeddings.shape[1]),
            "product_count": int(len(ordered_products)),
            "index_type": config.search.index_type,
            "search_text_version": "enriched_hm_metadata_v2",
            "text_rebuilt_at": datetime.now(UTC).isoformat(),
            "text_source_columns": [
                "name",
                "description",
                "category_l1",
                "category_l2",
                "category_l3",
                "category",
                "top_category",
                "mid_category",
                "leaf_category",
                "color",
                "price_tier",
                "style_tags",
                "metadata_aliases",
            ],
        }
    )
    indexes = manifest.setdefault("indexes", {})
    indexes["text"] = text_index.backend
    indexes["joint"] = joint_index.backend
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "products": int(len(ordered_products)),
                "encoder": encoder.name,
                "text_index": text_index.backend,
                "joint_index": joint_index.backend,
                "artifact_dir": str(artifact_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
