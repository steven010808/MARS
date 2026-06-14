# Prepare processed data, reusable artifacts, reports, and the active registry
# before the API/dashboard/worker services start.

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mars.config import load_config
from mars.config.settings import ensure_runtime_dirs
from mars.ct import ModelRegistry
from mars.data.hm_pipeline import RuntimeManifest, prepare_runtime_dataset
from mars.evaluation.runner import run_evaluation
from mars.recommendation.artifacts import artifact_path as recommendation_artifact_path
from mars.recommendation.artifacts import (
    build_recommendation_artifacts,
    load_recommendation_artifacts,
)
from mars.recommendation.artifacts import item_index_dir as recommendation_item_index_dir
from mars.search.artifacts import build_search_artifacts
from mars.search.behavior_model import (
    build_query_behavior_model_payload,
    write_query_behavior_model,
)
from mars.search.encoders import create_encoder
from mars.search.service import _clip_query_text


def _log(message: str) -> None:
    print(f"[bootstrap] {message}", flush=True)


# Check whether the existing processed Parquet dataset can be reused safely.
def _can_reuse_processed(config) -> RuntimeManifest | None:
    manifest_path = config.paths.processed_dir / "manifest.json"
    required = [
        config.paths.processed_dir / "products.parquet",
        config.paths.processed_dir / "users.parquet",
        config.paths.processed_dir / "events.parquet",
        config.paths.processed_dir / "sessions.parquet",
        config.paths.processed_dir / "search_queries.parquet",
        config.paths.processed_dir / "reco_interactions.parquet",
        config.paths.processed_dir / "train_events.parquet",
        config.paths.processed_dir / "test_events.parquet",
    ]
    if not manifest_path.exists() or not all(path.exists() for path in required):
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("mode") != config.active_mode:
        return None
    row_counts = payload.get("row_counts", {})
    if int(row_counts.get("products", 0)) <= 0 or int(row_counts.get("users", 0)) <= 0:
        return None
    if int(row_counts.get("events", 0)) <= 0:
        return None
    return RuntimeManifest(path=manifest_path, payload=payload)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare dataset, artifacts, reports, and registry for MARS."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--encoder", default=None, choices=["fallback", "clip"])
    parser.add_argument("--rebuild-raw", action="store_true")
    parser.add_argument("--clean-processed", action="store_true")
    parser.add_argument(
        "--clean-artifacts",
        action="store_true",
        help="Delete search/recommendation/report/registry runtime outputs before rebuilding.",
    )
    parser.add_argument(
        "--reuse-existing-artifacts",
        action="store_true",
        help=(
            "Reuse existing search/recommendation artifacts when present. "
            "Useful for fast Docker demos."
        ),
    )
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, mode=args.mode)
    ensure_runtime_dirs(config)
    if args.clean_artifacts:
        _clean_artifacts(config)
        ensure_runtime_dirs(config)
    manifest = None
    if args.reuse_existing_artifacts and not args.rebuild_raw and not args.clean_processed:
        manifest = _can_reuse_processed(config)
    if manifest is None:
        _ensure_external_hm_catalog_files(config)

    if manifest is None:
        _log(f"prepare dataset mode={config.active_mode}")
        manifest = prepare_runtime_dataset(
            config,
            rebuild_raw=args.rebuild_raw,
            clean_processed=args.clean_processed,
        )
    else:
        _log(f"reuse existing processed dataset mode={config.active_mode}")
    _log(f"dataset ready rows={manifest.payload.get('row_counts', {})}")

    search_dir = config.paths.artifacts_dir / "search"
    search_manifest_path = search_dir / "index_manifest.json"
    expected_encoder = args.encoder or config.search.encoder_type
    search_encoder = None
    if args.reuse_existing_artifacts and _can_reuse_search_artifacts(
        config,
        search_manifest_path,
        expected_encoder=expected_encoder,
    ):
        _log("reuse existing search artifacts")
        search_manifest = json.loads(search_manifest_path.read_text(encoding="utf-8"))
        encoder_name = str(search_manifest.get("encoder_type", config.search.encoder_type))
    else:
        _log("build search artifacts")
        encoder = create_encoder(
            encoder_type=expected_encoder,
            dim=config.search.embedding_dim,
            seed=config.seed,
            clip_model=config.search.clip_model,
            allow_fallback=config.search.allow_fallback_encoder,
        )
        search = build_search_artifacts(
            products_path=config.paths.processed_dir / "products.parquet",
            artifact_dir=search_dir,
            encoder=encoder,
            index_type=config.search.index_type,
        )
        search_manifest = search.manifest
        encoder_name = encoder.name
        search_encoder = encoder
    _ensure_search_behavior_model(config)
    _ensure_query_embedding_cache(config, search_encoder)

    recsys_path = recommendation_artifact_path(config)
    if args.reuse_existing_artifacts and _can_reuse_recommendation_artifacts(config, recsys_path):
        _log("reuse existing recommendation artifacts")
        recsys = load_recommendation_artifacts(recsys_path, config)
    else:
        _log("build recommendation artifacts")
        recsys = build_recommendation_artifacts(config=config)

    if args.clean_processed or not args.reuse_existing_artifacts:
        _clear_search_prediction_cache(config)

    metrics_path = config.paths.artifacts_dir / "reports" / "metrics.json"
    if args.reuse_existing_artifacts and _can_reuse_metrics(config, metrics_path):
        _log("reuse existing evaluation report")
        json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        _log("run evaluation")
        metrics = run_evaluation(config)
        metrics.to_dict()
    registered_version = None
    if args.register:
        registered_version = _ensure_registered_artifact_version(config, encoder_name)

    print(
        json.dumps(
            {
                "manifest": str(manifest.path),
                "mode": config.active_mode,
                "search_products": search_manifest["product_count"],
                "search_encoder": search_manifest["encoder_type"],
                "recsys_version": recsys.version,
                "metrics_path": str(config.paths.artifacts_dir / "reports" / "metrics.json"),
                "ct_reasons": [],
                "registered_version": registered_version,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _ensure_external_hm_catalog_files(config) -> None:
    raw_cfg = config.raw if isinstance(config.raw, dict) else {}
    simulator_cfg = raw_cfg.get("simulator", {})
    catalog_cfg = simulator_cfg.get("catalog", {})
    if str(catalog_cfg.get("source", "")) != "external_hm":
        return

    hm_cfg = catalog_cfg.get("external_hm", {})
    clean_path = Path(str(hm_cfg.get("products_master_path", "")))
    if not clean_path:
        return
    if clean_path.exists():
        return

    processed_dir = clean_path.parent
    master_path = processed_dir / "hm_products_master.csv"
    master_manifest = processed_dir / "hm_products_master_manifest.json"
    clean_manifest = processed_dir / "hm_products_master_clean_50k_manifest.json"

    raw_root = Path("data/external/hm/raw")
    articles_path = raw_root / "articles.csv"
    transactions_path = raw_root / "transactions_train.csv"
    images_root = raw_root / "images"
    label_cfg = raw_cfg.get("search_labels", {})
    qrels_path = Path(str(label_cfg.get("qrels_path", "data/external/hnm_search/raw/qrels.csv")))

    if not master_path.exists():
        missing = [
            str(path)
            for path in (articles_path, transactions_path, images_root)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "H&M clean 50K catalog is missing and raw Kaggle inputs are incomplete. "
                f"Missing: {missing}. Expected raw files under data/external/hm/raw/."
            )
        _log("build H&M product master from Kaggle raw")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.artifacts.build_hm_products_master",
                "--articles",
                str(articles_path),
                "--transactions",
                str(transactions_path),
                "--images-root",
                str(images_root),
                "--output",
                str(master_path),
                "--manifest",
                str(master_manifest),
            ],
            check=True,
        )

    if not qrels_path.exists():
        raise FileNotFoundError(
            "Microsoft H&M search qrels are required to reproduce the submitted 50K catalog. "
            f"Missing: {qrels_path}. Place qrels.csv and queries.csv under "
            "data/external/hnm_search/raw/."
        )

    _log("build clean balanced H&M 50K catalog")
    include_negatives = bool(label_cfg.get("include_negatives", True))
    clean_command = [
        sys.executable,
        "-m",
        "scripts.artifacts.build_clean_hm_catalog_50k",
        "--input",
        str(master_path),
        "--output",
        str(clean_path),
        "--manifest",
        str(clean_manifest),
        "--image-root",
        str(images_root),
        "--target",
        "50000",
        "--new-item-days-threshold",
        str(int(hm_cfg.get("new_item_days_threshold", 7))),
        "--hnm-search-qrels",
        str(qrels_path),
    ]
    if include_negatives:
        clean_command.append("--hnm-search-include-negatives")
    else:
        clean_command.append("--no-hnm-search-include-negatives")
    allowed = list(hm_cfg.get("allowed_top_categories", []))
    if allowed:
        clean_command.extend(["--allowed-top-categories", *[str(value) for value in allowed]])
    subprocess.run(clean_command, check=True)


def _clear_search_prediction_cache(config) -> None:
    reports_dir = config.paths.artifacts_dir / "reports"
    for name in ("search_predictions.json", "search_prediction_latency.json"):
        path = reports_dir / name
        if path.exists():
            path.unlink()


def _ensure_registered_artifact_version(config, encoder_name: str) -> str:
    """Register the baseline artifact once and reuse it on later bootstraps."""
    registry = ModelRegistry(config.paths.artifacts_dir / "registry" / "models.json")
    artifact_path = str(config.paths.artifacts_dir)
    metrics_path = str(config.paths.artifacts_dir / "reports" / "metrics.json")
    metadata = {"mode": config.active_mode, "encoder": encoder_name}
    payload = registry.load()
    active_version = payload.get("active_version")

    for entry in payload.get("versions", []):
        if (
            entry.get("version") == active_version
            and entry.get("status") == "active"
            and entry.get("artifact_path") == artifact_path
            and entry.get("metrics_path") == metrics_path
            and entry.get("metadata", {}) == metadata
        ):
            _log(f"reuse registered artifact version {active_version}")
            return str(active_version)

    _log("register artifact version")
    entry = registry.register(
        artifact_path=config.paths.artifacts_dir,
        metrics_path=config.paths.artifacts_dir / "reports" / "metrics.json",
        metadata=metadata,
        activate=True,
    )
    return entry.version


def _clean_artifacts(config) -> None:
    root = config.paths.artifacts_dir.resolve()
    for name in ("search", "recsys", "reports", "registry"):
        target = (config.paths.artifacts_dir / name).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError(f"Refusing to clean artifact path outside {root}: {target}")
        if target.exists():
            shutil.rmtree(target)


def _ensure_search_behavior_model(config) -> None:
    raw_search = config.raw.get("search", {}) if isinstance(config.raw, dict) else {}
    if not bool(raw_search.get("query_behavior_model_required", False)):
        return
    configured_path = raw_search.get("query_behavior_model_path")
    output_path = (
        Path(str(configured_path))
        if configured_path
        else config.paths.artifacts_dir / "search" / "query_behavior_model.json.gz"
    )
    if _can_reuse_search_behavior_model(config, output_path):
        _log("reuse existing search behavior model")
        return
    _log("build search behavior model")
    payload = build_query_behavior_model_payload(config)
    write_query_behavior_model(payload, output_path)


def _can_reuse_search_behavior_model(config, output_path: Path) -> bool:
    if not output_path.exists():
        return False
    try:
        import gzip

        with gzip.open(output_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        _log("search behavior model reuse skipped: model is unreadable")
        return False
    if payload.get("schema_version") != "search-query-behavior.v1":
        _log(
            "search behavior model reuse skipped: schema mismatch "
            f"actual={payload.get('schema_version')}"
        )
        return False
    if str(payload.get("split", "")) != "train":
        _log(f"search behavior model reuse skipped: split mismatch actual={payload.get('split')}")
        return False
    if int(payload.get("seed", -1)) != int(config.seed):
        _log(
            "search behavior model reuse skipped: seed mismatch "
            f"actual={payload.get('seed')} expected={config.seed}"
        )
        return False
    expected_count = _processed_product_count(config)
    if expected_count and int(payload.get("catalog_products", -1)) != expected_count:
        _log(
            "search behavior model reuse skipped: product_count mismatch "
            f"actual={payload.get('catalog_products')} expected={expected_count}"
        )
        return False
    if not payload.get("query_prior"):
        _log("search behavior model reuse skipped: query prior is empty")
        return False
    return True


def _ensure_query_embedding_cache(config, encoder=None) -> None:
    raw_search = config.raw.get("search", {}) if isinstance(config.raw, dict) else {}
    if not bool(raw_search.get("query_behavior_model_required", False)):
        return
    configured_model_path = raw_search.get("query_behavior_model_path")
    model_path = (
        Path(str(configured_model_path))
        if configured_model_path
        else config.paths.artifacts_dir / "search" / "query_behavior_model.json.gz"
    )
    if not model_path.exists():
        return
    configured_cache_path = raw_search.get("query_embedding_cache_path")
    cache_path = (
        Path(str(configured_cache_path))
        if configured_cache_path
        else config.paths.artifacts_dir / "search" / "query_embedding_cache.npz"
    )
    import gzip
    import json

    import numpy as np

    with gzip.open(model_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    query_keys = sorted(str(key) for key in (payload.get("query_prior", {}) or {}).keys())
    if not query_keys:
        return
    if _can_reuse_query_embedding_cache(config, cache_path, query_keys):
        _log(f"reuse existing query embedding cache rows={len(query_keys)}")
        return
    if encoder is None:
        encoder = create_encoder(
            encoder_type=config.search.encoder_type,
            dim=config.search.embedding_dim,
            seed=config.seed,
            clip_model=config.search.clip_model,
            allow_fallback=config.search.allow_fallback_encoder,
        )
    clip_texts = [_clip_query_text(query) for query in query_keys]
    _log(f"build query embedding cache rows={len(clip_texts)}")
    embeddings = encoder.encode_texts(clip_texts).astype("float32")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        query_keys=np.asarray(query_keys, dtype=object),
        clip_texts=np.asarray(clip_texts, dtype=object),
        embeddings=embeddings,
    )


def _can_reuse_query_embedding_cache(config, cache_path: Path, query_keys: list[str]) -> bool:
    if not cache_path.exists():
        return False
    try:
        import numpy as np

        with np.load(cache_path, allow_pickle=True) as data:
            cached_keys = [str(key) for key in data["query_keys"].tolist()]
            embeddings = data["embeddings"]
    except Exception:
        _log("query embedding cache reuse skipped: cache is unreadable")
        return False
    if len(cached_keys) != len(query_keys):
        _log(
            "query embedding cache reuse skipped: row_count mismatch "
            f"actual={len(cached_keys)} expected={len(query_keys)}"
        )
        return False
    if cached_keys != query_keys:
        _log("query embedding cache reuse skipped: query keys mismatch")
        return False
    if len(getattr(embeddings, "shape", ())) != 2:
        _log("query embedding cache reuse skipped: embeddings are not a 2D matrix")
        return False
    if int(embeddings.shape[0]) != len(query_keys):
        _log(
            "query embedding cache reuse skipped: embedding row_count mismatch "
            f"actual={embeddings.shape[0]} expected={len(query_keys)}"
        )
        return False
    if int(embeddings.shape[1]) != int(config.search.embedding_dim):
        _log(
            "query embedding cache reuse skipped: embedding_dim mismatch "
            f"actual={embeddings.shape[1]} expected={config.search.embedding_dim}"
        )
        return False
    return True


def _can_reuse_search_artifacts(config, manifest_path: Path, *, expected_encoder: str) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        _log("search artifact reuse skipped: manifest is unreadable")
        return False

    actual_encoder = str(manifest.get("encoder_type", ""))
    expected = expected_encoder.lower()
    if expected == "clip" and not actual_encoder.startswith("clip:"):
        _log(f"search artifact reuse skipped: encoder mismatch actual={actual_encoder}")
        return False
    if expected == "fallback" and actual_encoder != "fallback":
        _log(f"search artifact reuse skipped: encoder mismatch actual={actual_encoder}")
        return False

    expected_count = _processed_product_count(config)
    if expected_count and int(manifest.get("product_count", -1)) != expected_count:
        _log(
            "search artifact reuse skipped: product_count mismatch "
            f"actual={manifest.get('product_count')} expected={expected_count}"
        )
        return False

    indexes = manifest.get("indexes", {})
    if _faiss_available() and any(str(backend) != "faiss" for backend in indexes.values()):
        _log(f"search artifact reuse skipped: FAISS available but indexes={indexes}")
        return False
    return True


def _processed_product_count(config) -> int:
    products_path = config.paths.processed_dir / "products.parquet"
    if not products_path.exists():
        return 0
    try:
        import pandas as pd

        return int(len(pd.read_parquet(products_path, columns=["product_id"])))
    except Exception:
        return 0


def _faiss_available() -> bool:
    try:
        import faiss  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _torch_available() -> bool:
    try:
        import torch  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _can_reuse_recommendation_artifacts(config, recsys_path: Path) -> bool:
    if not recsys_path.exists():
        return False
    try:
        recsys = load_recommendation_artifacts(recsys_path, config)
    except Exception:
        _log("recommendation artifact reuse skipped: artifact is unreadable")
        return False
    if int(recsys.embedding_dim) != int(config.recommendation.embedding_dim):
        _log(
            "recommendation artifact reuse skipped: embedding_dim mismatch "
            f"actual={recsys.embedding_dim} expected={config.recommendation.embedding_dim}"
        )
        return False
    training_source = str(getattr(recsys, "training_events_source", ""))
    if not (
        training_source == "train_events.parquet"
        or training_source.startswith("live_feedback_refresh:")
    ):
        _log(
            "recommendation artifact reuse skipped: unsupported training source "
            f"actual={training_source!r}"
        )
        return False
    if not getattr(recsys, "category_transitions", {}):
        _log("recommendation artifact reuse skipped: category transition graph is missing")
        return False
    expected_count = _processed_product_count(config)
    if expected_count and len(recsys.products) != expected_count:
        _log(
            "recommendation artifact reuse skipped: product_count mismatch "
            f"actual={len(recsys.products)} expected={expected_count}"
        )
        return False
    if _torch_available() and (
        not recsys.ranking_model or recsys.ranking_model.get("model_type") != "torch_wide_deep"
    ):
        _log("recommendation artifact reuse skipped: trained Wide&Deep ranker is missing")
        return False
    if _torch_available() and (
        not recsys.two_tower_model or recsys.two_tower_model.get("model_type") != "torch_two_tower"
    ):
        _log("recommendation artifact reuse skipped: trained Two-Tower model is missing")
        return False
    index_manifest = recommendation_item_index_dir(config) / "items_index.json"
    if not index_manifest.exists():
        _log("recommendation artifact reuse skipped: item FAISS index is missing")
        return False
    try:
        manifest = json.loads(index_manifest.read_text(encoding="utf-8"))
    except Exception:
        _log("recommendation artifact reuse skipped: item index manifest is unreadable")
        return False
    if int(manifest.get("count", -1)) != len(recsys.products):
        _log(
            "recommendation artifact reuse skipped: item index count mismatch "
            f"actual={manifest.get('count')} expected={len(recsys.products)}"
        )
        return False
    if _faiss_available() and str(manifest.get("backend")) != "faiss":
        _log(
            "recommendation artifact reuse skipped: FAISS available but "
            f"item backend={manifest.get('backend')}"
        )
        return False
    return True


def _can_reuse_metrics(config, metrics_path: Path) -> bool:
    if not metrics_path.exists():
        return False
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        _log("evaluation report reuse skipped: metrics.json is unreadable")
        return False
    system = payload.get("system", {}) if isinstance(payload, dict) else {}
    if int(system.get("products", -1)) != int(config.mode.products):
        _log(
            "evaluation report reuse skipped: product_count mismatch "
            f"actual={system.get('products')} expected={config.mode.products}"
        )
        return False
    if int(system.get("users", -1)) != int(config.mode.users):
        _log(
            "evaluation report reuse skipped: user_count mismatch "
            f"actual={system.get('users')} expected={config.mode.users}"
        )
        return False
    if int(system.get("events", -1)) != int(config.mode.events):
        _log(
            "evaluation report reuse skipped: event_count mismatch "
            f"actual={system.get('events')} expected={config.mode.events}"
        )
        return False
    if not payload.get("search") or not payload.get("recommendation"):
        _log("evaluation report reuse skipped: search/recommendation metrics missing")
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
