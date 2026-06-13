from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mars.config import load_config
from mars.config.settings import ensure_runtime_dirs
from mars.data.hm_pipeline import prepare_runtime_dataset
from mars.recommendation.artifacts import build_recommendation_artifacts
from mars.search.artifacts import build_search_artifacts
from mars.search.encoders import create_encoder


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MARS search and recommendation artifacts.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--encoder", default="fallback")
    parser.add_argument("--rebuild-raw", action="store_true")
    parser.add_argument("--clean-processed", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, mode=args.mode)
    ensure_runtime_dirs(config)
    manifest = prepare_runtime_dataset(
        config,
        rebuild_raw=args.rebuild_raw,
        clean_processed=args.clean_processed,
    )
    encoder = create_encoder(
        encoder_type=args.encoder,
        dim=config.search.embedding_dim,
        seed=config.seed,
        clip_model=config.search.clip_model,
        allow_fallback=config.search.allow_fallback_encoder,
    )
    search = build_search_artifacts(
        products_path=config.paths.processed_dir / "products.parquet",
        artifact_dir=config.paths.artifacts_dir / "search",
        encoder=encoder,
        index_type=config.search.index_type,
    )
    recsys = build_recommendation_artifacts(config=config)
    print(
        {
            "manifest": str(manifest.path),
            "mode": config.active_mode,
            "search_products": search.manifest["product_count"],
            "search_encoder": search.manifest["encoder_type"],
            "recsys_version": recsys.version,
            "recsys_products": len(recsys.products),
            "recsys_users": len(recsys.users),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
