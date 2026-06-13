from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mars.config import load_config
from mars.config.settings import ensure_runtime_dirs
from mars.data.hm_pipeline import (
    _build_reco_interactions,
    _build_search_queries,
    _load_events,
    _load_products,
    _load_users,
    _normalise_events,
    _normalise_products,
    _normalise_users,
    _price_scale_factor,
    _select_mode_slice,
    prepare_runtime_dataset,
)
from mars.data.io import write_json, write_table
from mars.evaluation.runner import run_evaluation
from mars.recommendation.artifacts import build_recommendation_artifacts
from mars.recommendation.artifacts import item_index_dir as recommendation_item_index_dir
from mars.search.artifacts import build_search_artifacts
from mars.search.encoders import create_encoder
from mars.search.service import SearchService


def _repo_path(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else ROOT / target


def _full_config(args: argparse.Namespace):
    config = load_config(args.config, mode="full")
    work_dir = _repo_path(args.work_dir)
    paths = replace(
        config.paths,
        raw_dir=_repo_path(args.raw_dir),
        processed_dir=work_dir / "processed",
        artifacts_dir=work_dir / "artifacts",
        logs_dir=work_dir / "logs",
    )
    return replace(config, paths=paths)


def _has_processed(config) -> bool:
    required = [
        config.paths.processed_dir / "products.parquet",
        config.paths.processed_dir / "users.parquet",
        config.paths.processed_dir / "events.parquet",
        config.paths.processed_dir / "reco_interactions.parquet",
    ]
    return all(path.exists() for path in required)


def _prepare_fast_eval_dataset(config) -> Path:
    """Build only the full-scale tables required for metric verification.

    The regular runtime dataset builder also creates rich derived tables such as
    session-level search positives. On 1M events that can be slow in a local
    notebook-style environment, while the required scale metric check only needs
    products, users, events, and recommendation interactions.
    """

    if config.paths.processed_dir.exists():
        shutil.rmtree(config.paths.processed_dir)
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)

    raw_products = _load_products(config.paths.raw_dir / "products.csv")
    raw_users = _load_users(config.paths.raw_dir / "users.csv")
    raw_events = _load_events(config.paths.raw_dir / "events.csv")
    selected_products, selected_users, selected_events = _select_mode_slice(
        raw_products=raw_products,
        raw_users=raw_users,
        raw_events=raw_events,
        config=config,
    )

    scale_factor = _price_scale_factor(selected_products["price"])
    products = _normalise_products(selected_products, scale_factor)
    users = _normalise_users(selected_users)
    events = _normalise_events(selected_events, products, scale_factor)
    reco_interactions = _build_reco_interactions(events, products)
    sessions = _build_fast_sessions(events)
    search_queries = _build_search_queries(products, config)
    train_events, valid_events, test_events = _split_time_windows(events)

    written_files = {
        "products": str(write_table(products, config.paths.processed_dir / "products")),
        "users": str(write_table(users, config.paths.processed_dir / "users")),
        "events": str(write_table(events, config.paths.processed_dir / "events")),
        "sessions": str(write_table(sessions, config.paths.processed_dir / "sessions")),
        "search_queries": str(
            write_table(search_queries, config.paths.processed_dir / "search_queries")
        ),
        "reco_interactions": str(
            write_table(reco_interactions, config.paths.processed_dir / "reco_interactions")
        ),
        "train_events": str(write_table(train_events, config.paths.processed_dir / "train_events")),
        "valid_events": str(write_table(valid_events, config.paths.processed_dir / "valid_events")),
        "test_events": str(write_table(test_events, config.paths.processed_dir / "test_events")),
    }
    manifest = {
        "schema_version": "hm-required-scale-eval.v1",
        "generator_version": "scripts.evaluate_required_scale.fast",
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": config.active_mode,
        "seed": config.seed,
        "data_source": "hm",
        "price_scale_factor": scale_factor,
        "expected_counts": {
            "products": config.mode.products,
            "users": config.mode.users,
            "events": config.mode.events,
        },
        "row_counts": {
            "products": int(len(products)),
            "users": int(len(users)),
            "events": int(len(events)),
            "sessions": int(len(sessions)),
            "search_queries": int(len(search_queries)),
            "reco_interactions": int(len(reco_interactions)),
            "train_events": int(len(train_events)),
            "valid_events": int(len(valid_events)),
            "test_events": int(len(test_events)),
        },
        "files": written_files,
        "time_range": {
            "min": str(events["timestamp"].min()) if not events.empty else None,
            "max": str(events["timestamp"].max()) if not events.empty else None,
        },
        "persona_distribution": {
            key: int(value) for key, value in users["persona"].value_counts().sort_index().items()
        },
        "event_distribution": {
            key: int(value)
            for key, value in events["event_type"].value_counts().sort_index().items()
        },
    }
    return write_json(manifest, config.paths.processed_dir / "manifest.json")


def _build_fast_sessions(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    ordered = events.sort_values(["session_id", "timestamp", "event_id"]).copy()
    grouped = ordered.groupby("session_id", sort=False)
    sessions = grouped.agg(
        user_id=("user_id", "first"),
        persona=("persona", "first"),
        started_at=("timestamp", "first"),
        ended_at=("timestamp", "last"),
        num_events=("event_id", "count"),
        entry_source=("source", "first"),
    ).reset_index()
    converted = (
        ordered["event_type"].astype(str).eq("purchase").groupby(ordered["session_id"]).any()
    )
    sessions["converted"] = (
        sessions["session_id"].map(converted.to_dict()).fillna(False).astype(bool)
    )
    product_events = ordered[ordered["product_id"].notna()]
    if not product_events.empty:
        last_product = (
            product_events.groupby("session_id", sort=False).tail(1).set_index("session_id")
        )
        sessions["last_category"] = sessions["session_id"].map(
            last_product["category"].astype(str).to_dict()
        )
        sessions["last_product_id"] = sessions["session_id"].map(
            last_product["product_id"].astype(str).to_dict()
        )
    else:
        sessions["last_category"] = None
        sessions["last_product_id"] = None
    return sessions


def _split_time_windows(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = events.sort_values(["timestamp", "event_id"]).reset_index(drop=True)
    train_end = int(len(ordered) * 0.8)
    valid_end = int(len(ordered) * 0.9)
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:valid_end].copy(),
        ordered.iloc[valid_end:].copy(),
    )


def _build_search_predictions(config, sample_size: int) -> Path:
    search_queries = pd.read_parquet(
        config.paths.processed_dir / "search_queries.parquet",
        columns=["query_id", "query"],
    ).head(sample_size)
    service = SearchService(config)
    predictions: dict[str, list[str]] = {}
    latencies: list[float] = []
    for row in search_queries.itertuples(index=False):
        query = str(row.query or "")
        if not query or query.lower() == "nan":
            continue
        response = service.search(
            {
                "search_type": "text",
                "query": query,
                "top_k": 10,
                "filters": {},
            }
        )
        predictions[str(row.query_id)] = [
            str(item["product_id"]) for item in response.get("results", [])
        ]
        latencies.append(float(response.get("latency_ms", 0.0) or 0.0))

    reports_dir = config.paths.artifacts_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "search_predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    latency_summary = {
        "sample_size": len(predictions),
        "query_count": int(len(search_queries)),
        "label_source": "microsoft_hnm_search_qrels",
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "method": "microsoft_hnm_search_qrels",
    }
    (reports_dir / "search_prediction_latency.json").write_text(
        json.dumps(latency_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return reports_dir / "search_predictions.json"


def _measure_search_latency_benchmark(config, *, warmup: int = 10, runs: int = 100) -> Path:
    search_queries = pd.read_parquet(
        config.paths.processed_dir / "search_queries.parquet",
        columns=["query"],
    ).head(max(warmup, 10))
    service = SearchService(config)
    requests: list[dict[str, Any]] = []
    for row in search_queries.itertuples(index=False):
        query = str(row.query or "")
        if not query or query.lower() == "nan":
            continue
        requests.append(
            {
                "search_type": "text",
                "query": query,
                "top_k": 10,
                "filters": {},
            }
        )
    if not requests:
        requests = [{"search_type": "text", "query": "black hoodie", "top_k": 10, "filters": {}}]

    for idx in range(warmup):
        service.search(requests[idx % len(requests)])

    latencies: list[float] = []
    for idx in range(runs):
        response = service.search(requests[idx % len(requests)])
        latencies.append(float(response.get("latency_ms", 0.0) or 0.0))

    summary = {
        "method": "warmup_then_repeated_single_request_p95",
        "warmup": warmup,
        "runs": runs,
        "query_count": len(requests),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
    }
    reports_dir = config.paths.artifacts_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "search_latency_benchmark.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _patch_search_latency(config, metrics: dict) -> dict:
    latency_path = config.paths.artifacts_dir / "reports" / "search_latency_benchmark.json"
    metrics_path = config.paths.artifacts_dir / "reports" / "metrics.json"
    if not latency_path.exists():
        return metrics
    latency = json.loads(latency_path.read_text(encoding="utf-8"))
    metrics.setdefault("search", {})
    metrics["search"]["latency_p50_ms"] = float(latency.get("latency_p50_ms", 0.0) or 0.0)
    metrics["search"]["latency_p95_ms"] = float(latency.get("latency_p95_ms", 0.0) or 0.0)
    metrics["search"]["latency_method"] = str(latency.get("method", "unknown"))
    target = metrics.setdefault("targets", {}).setdefault(
        "search_latency_p95_ms", {"target": 200.0}
    )
    target["value"] = metrics["search"]["latency_p95_ms"]
    target["status"] = "met" if target["value"] <= float(target.get("target", 200.0)) else "not_met"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    import numpy as np

    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate MARS on required full-scale H&M simulator data."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--raw-dir", default="data/full_generation_check/full_raw")
    parser.add_argument("--work-dir", default="data/full_generation_check/required_scale_eval")
    parser.add_argument("--encoder", choices=["fallback", "clip"], default="clip")
    parser.add_argument("--search-sample", type=int, default=500)
    parser.add_argument("--rebuild-processed", action="store_true")
    parser.add_argument("--rebuild-artifacts", action="store_true")
    parser.add_argument(
        "--slow-processed",
        action="store_true",
        help=(
            "Use the full runtime dataset builder instead of the faster metric-eval table builder."
        ),
    )
    args = parser.parse_args()

    config = _full_config(args)
    ensure_runtime_dirs(config)
    if args.rebuild_processed or not _has_processed(config):
        if args.slow_processed:
            prepare_runtime_dataset(config, rebuild_raw=False, clean_processed=True)
        else:
            _prepare_fast_eval_dataset(config)

    search_manifest = config.paths.artifacts_dir / "search" / "index_manifest.json"
    recsys_path = config.paths.artifacts_dir / "recsys" / "recommendation_artifacts.json.gz"
    if args.rebuild_artifacts or not _search_artifacts_match(search_manifest, args.encoder):
        encoder = create_encoder(
            encoder_type=args.encoder,
            dim=config.search.embedding_dim,
            seed=config.seed,
            clip_model=config.search.clip_model,
            allow_fallback=config.search.allow_fallback_encoder,
        )
        build_search_artifacts(
            products_path=config.paths.processed_dir / "products.parquet",
            artifact_dir=config.paths.artifacts_dir / "search",
            encoder=encoder,
            index_type=config.search.index_type,
        )
    if args.rebuild_artifacts or not _recommendation_artifacts_match(config, recsys_path):
        build_recommendation_artifacts(config=config)

    prediction_path = _build_search_predictions(config, sample_size=args.search_sample)
    latency_path = _measure_search_latency_benchmark(config)
    result = run_evaluation(config=config)
    metrics = _patch_search_latency(config, result.to_dict())
    output = {
        "mode": config.active_mode,
        "processed_dir": str(config.paths.processed_dir),
        "artifacts_dir": str(config.paths.artifacts_dir),
        "search_predictions": str(prediction_path),
        "search_latency_benchmark": str(latency_path),
        "metrics_path": str(config.paths.artifacts_dir / "reports" / "metrics.json"),
        "metrics": metrics,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def _search_artifacts_match(manifest_path: Path, expected_encoder: str) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    encoder = str(manifest.get("encoder_type", ""))
    if expected_encoder == "clip" and not encoder.startswith("clip:"):
        return False
    if expected_encoder == "fallback" and encoder != "fallback":
        return False
    return True


def _recommendation_artifacts_match(config, recsys_path: Path) -> bool:
    if not recsys_path.exists():
        return False
    index_manifest = recommendation_item_index_dir(config) / "items_index.json"
    if not index_manifest.exists():
        return False
    try:
        with gzip.open(recsys_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return False
    if int(payload.get("embedding_dim", -1)) != int(config.recommendation.embedding_dim):
        return False
    ranking_model = payload.get("ranking_model") or {}
    two_tower_model = payload.get("two_tower_model") or {}
    return (
        ranking_model.get("model_type") == "torch_wide_deep"
        and two_tower_model.get("model_type") == "torch_two_tower"
    )


if __name__ == "__main__":
    raise SystemExit(main())
