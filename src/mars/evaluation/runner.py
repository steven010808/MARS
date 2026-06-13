from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mars.config import MarsConfig, load_config
from mars.config.settings import ensure_runtime_dirs
from mars.evaluation.ab import assign_bucket, build_ab_report
from mars.evaluation.metrics import (
    auc_score,
    coverage_at_k,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from mars.search.qrels import evaluation_qrels_split, qrels_split_settings, select_qrels_split


@dataclass(frozen=True)
class EvaluationResult:
    generated_at: str
    mode: str
    search: dict[str, Any]
    recommendation: dict[str, Any]
    ab_test: dict[str, Any]
    baselines: dict[str, Any]
    system: dict[str, Any]
    targets: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_evaluation(
    config: MarsConfig | None = None,
    config_path: str | Path = "configs/config.yaml",
    mode: str | None = None,
    output_path: str | Path | None = None,
) -> EvaluationResult:
    cfg = config or load_config(config_path, mode=mode)
    ensure_runtime_dirs(cfg)

    processed = cfg.paths.processed_dir
    reports_dir = cfg.paths.artifacts_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    products = _read_table(processed / "products.parquet")
    users = _read_table(processed / "users.parquet")
    events = _read_table(processed / "events.parquet")
    train_events = _read_table(processed / "train_events.parquet")
    test_events = _read_table(processed / "test_events.parquet")
    interactions = _read_table(processed / "reco_interactions.parquet")
    search_queries = _read_table(processed / "search_queries.parquet")
    search_eval_queries = select_qrels_split(search_queries, cfg, evaluation_qrels_split(cfg))
    search_predictions, search_latency = _ensure_search_predictions(
        cfg, search_eval_queries, reports_dir
    )
    recommendation_predictions = _read_json_mapping(reports_dir / "recommendation_predictions.json")

    search_metrics = _search_metrics(
        cfg, search_eval_queries, products, search_predictions, search_latency
    )
    search_metrics["split_diagnostics"] = _search_split_diagnostics(search_queries, cfg)
    recommendation_metrics = _recommendation_metrics(
        config=cfg,
        training_events=train_events if not train_events.empty else events,
        evaluation_events=test_events if not test_events.empty else events,
        products=products,
        predictions=recommendation_predictions,
        candidate_k=cfg.recommendation.candidate_k,
    )
    ab_report = build_ab_report(_with_ab_group(events), experiment_key="mars_default").to_dict()
    baselines = _baseline_metrics(
        config=cfg,
        search_queries=search_eval_queries,
        products=products,
        search_metrics=search_metrics,
        recommendation_metrics=recommendation_metrics,
        training_events=train_events if not train_events.empty else events,
        evaluation_events=test_events if not test_events.empty else events,
        candidate_k=cfg.recommendation.candidate_k,
    )
    system_metrics = {
        "products": int(len(products)),
        "users": int(len(users)),
        "events": int(len(events)),
        "search_queries": int(len(search_queries)),
        "search_evaluation_queries": int(len(search_eval_queries)),
        "interactions": int(len(interactions)),
        "artifact_readiness": {
            "processed_dir": processed.exists(),
            "events": not events.empty,
            "products": not products.empty,
            "users": not users.empty,
            "search_predictions": bool(search_predictions),
            "recommendation_predictions": bool(recommendation_predictions),
        },
        "split_counts": {
            "train_events": int(len(train_events)),
            "test_events": int(len(test_events)),
        },
    }
    targets = _target_status(search_metrics, recommendation_metrics, cfg.monitoring.ctr_threshold)

    result = EvaluationResult(
        generated_at=datetime.now(UTC).isoformat(),
        mode=cfg.active_mode,
        search=search_metrics,
        recommendation=recommendation_metrics,
        ab_test=ab_report,
        baselines=baselines,
        system=system_metrics,
        targets=targets,
    )
    destination = Path(output_path) if output_path else reports_dir / "metrics.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.DataFrame()


def _read_json_mapping(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): [str(item) for item in value] for key, value in raw.items()}


def _ensure_search_predictions(
    config: MarsConfig,
    search_queries: pd.DataFrame,
    reports_dir: Path,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    prediction_path = reports_dir / "search_predictions.json"
    latency_path = reports_dir / "search_prediction_latency.json"
    predictions = _read_json_mapping(prediction_path)
    latency = _read_json_dict(latency_path)
    sample_size = _search_sample_size(config, len(search_queries))
    expected_query_ids = _expected_search_query_ids(search_queries, sample_size)
    service_signature = _search_prediction_signature(config)
    cache_is_current = (
        bool(predictions)
        and bool(latency)
        and str(latency.get("label_source", "")) == "microsoft_hnm_search_qrels"
        and str(latency.get("search_service_signature", "")) == service_signature
        and expected_query_ids.issubset(set(predictions))
    )
    if cache_is_current:
        latency = _refresh_multimodal_latency(config, latency, latency_path)
        return predictions, latency
    predictions = {}
    latency = {}
    if search_queries.empty or "query" not in search_queries.columns:
        return predictions, latency

    try:
        from mars.search.service import SearchService

        service = SearchService(config)
    except Exception:
        return predictions, latency

    output: dict[str, list[str]] = {}
    prediction_latencies: list[float] = []
    query_result_cache: dict[str, list[str]] = {}
    for row in search_queries.head(sample_size).itertuples(index=False):
        query = str(getattr(row, "query", "") or "")
        if not query or query.lower() == "nan":
            continue
        query_cache_key = query.strip().lower()
        try:
            if query_cache_key in query_result_cache:
                ranked_products = query_result_cache[query_cache_key]
            else:
                response = service.search(
                    {
                        "search_type": "text",
                        "query": query,
                        "top_k": 10,
                        "filters": {},
                    }
                )
                ranked_products = [str(item["product_id"]) for item in response.get("results", [])]
                query_result_cache[query_cache_key] = ranked_products
                prediction_latencies.append(float(response.get("latency_ms", 0.0) or 0.0))
        except Exception:
            continue
        query_id = str(getattr(row, "query_id", len(output)))
        output[query_id] = list(ranked_products)

    if output:
        benchmark_latency = _measure_search_latency_benchmark(service, search_queries)
        multimodal_latency = _measure_multimodal_search_latency(service)
        text_p50 = _percentile_values(prediction_latencies, 50)
        text_p95 = _percentile_values(prediction_latencies, 95)
        mode_p95_values = [
            value
            for value in (
                text_p95,
                multimodal_latency.get("image_latency_p95_ms", 0.0),
                multimodal_latency.get("hybrid_latency_p95_ms", 0.0),
            )
            if float(value or 0.0) > 0
        ]
        prediction_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        latency = {
            "sample_size": len(output),
            "query_count": int(len(search_queries)),
            "label_source": "microsoft_hnm_search_qrels",
            "search_service_signature": service_signature,
            "image_artifact_signature": _image_artifact_signature(config),
            "latency_method": "unique_request_p95_by_search_mode",
            "unique_text_request_count": len(query_result_cache),
            "latency_p50_ms": text_p50,
            "latency_p95_ms": max(mode_p95_values, default=text_p95),
            "text_latency_p50_ms": text_p50,
            "text_latency_p95_ms": text_p95,
            **multimodal_latency,
            "latency_warmup": benchmark_latency["warmup"],
            "latency_runs": benchmark_latency["runs"],
            "latency_query_count": benchmark_latency["query_count"],
            "repeated_query_latency_p50_ms": benchmark_latency["latency_p50_ms"],
            "repeated_query_latency_p95_ms": benchmark_latency["latency_p95_ms"],
            "prediction_generation_p50_ms": _percentile_values(prediction_latencies, 50),
            "prediction_generation_p95_ms": _percentile_values(prediction_latencies, 95),
        }
        latency_path.write_text(
            json.dumps(latency, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        predictions = output
    return predictions, latency


def _measure_search_latency_benchmark(
    service: Any,
    search_queries: pd.DataFrame,
    *,
    warmup: int = 10,
    runs: int = 100,
) -> dict[str, Any]:
    """Measure serving latency using the protocol in the project spec.

    Prediction generation intentionally runs over many unique queries to build
    the offline ranking file. The latency target, however, is defined as
    warm-up requests followed by single-request p95, so we keep it separate.
    """

    requests: list[dict[str, Any]] = []
    if not search_queries.empty and "query" in search_queries.columns:
        for row in search_queries.head(max(warmup, 10)).itertuples(index=False):
            query = str(getattr(row, "query", "") or "")
            if query and query.lower() != "nan":
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

    return {
        "warmup": warmup,
        "runs": runs,
        "query_count": len(requests),
        "latency_p50_ms": _percentile_values(latencies, 50),
        "latency_p95_ms": _percentile_values(latencies, 95),
    }


def _measure_multimodal_search_latency(
    service: Any,
    *,
    sample_size: int = 50,
    warmup: int = 10,
) -> dict[str, Any]:
    records = getattr(service, "_metadata_records", [])
    image_records = [
        row for row in records if row.get("image_path") and Path(str(row["image_path"])).exists()
    ]
    warmup_records = image_records[: max(0, warmup)]
    for row in warmup_records:
        image_path = str(row["image_path"])
        try:
            service.search(
                {
                    "search_type": "image",
                    "image_path": image_path,
                    "top_k": 10,
                    "filters": {},
                }
            )
            service.search(
                {
                    "search_type": "hybrid",
                    "query": str(row.get("name", "") or ""),
                    "image_path": image_path,
                    "top_k": 10,
                    "filters": {},
                }
            )
        except Exception:
            continue
    measurement_records = image_records[len(warmup_records) : len(warmup_records) + sample_size]
    image_latencies: list[float] = []
    hybrid_latencies: list[float] = []
    for row in measurement_records:
        image_path = str(row["image_path"])
        try:
            image_response = service.search(
                {
                    "search_type": "image",
                    "image_path": image_path,
                    "top_k": 10,
                    "filters": {},
                }
            )
            image_latencies.append(float(image_response.get("latency_ms", 0.0) or 0.0))
            hybrid_response = service.search(
                {
                    "search_type": "hybrid",
                    "query": str(row.get("name", "") or ""),
                    "image_path": image_path,
                    "top_k": 10,
                    "filters": {},
                }
            )
            hybrid_latencies.append(float(hybrid_response.get("latency_ms", 0.0) or 0.0))
        except Exception:
            continue
    return {
        "multimodal_latency_sample_size": len(image_latencies),
        "multimodal_latency_warmup": len(warmup_records),
        "image_latency_p50_ms": _percentile_values(image_latencies, 50),
        "image_latency_p95_ms": _percentile_values(image_latencies, 95),
        "hybrid_latency_p50_ms": _percentile_values(hybrid_latencies, 50),
        "hybrid_latency_p95_ms": _percentile_values(hybrid_latencies, 95),
    }


def _refresh_multimodal_latency(
    config: MarsConfig,
    latency: dict[str, Any],
    latency_path: Path,
) -> dict[str, Any]:
    image_signature = _image_artifact_signature(config)
    if (
        latency.get("image_artifact_signature") == image_signature
        and str(latency.get("latency_method", "")) == "unique_request_p95_by_search_mode"
    ):
        return latency
    try:
        from mars.search.service import SearchService

        service = SearchService(config)
        multimodal_latency = _measure_multimodal_search_latency(service)
    except Exception:
        return latency
    text_p50 = float(
        latency.get("prediction_generation_p50_ms", latency.get("latency_p50_ms", 0.0)) or 0.0
    )
    text_p95 = float(
        latency.get("prediction_generation_p95_ms", latency.get("latency_p95_ms", 0.0)) or 0.0
    )
    mode_p95_values = [
        value
        for value in (
            text_p95,
            multimodal_latency.get("image_latency_p95_ms", 0.0),
            multimodal_latency.get("hybrid_latency_p95_ms", 0.0),
        )
        if float(value or 0.0) > 0
    ]
    updated = {
        **latency,
        "image_artifact_signature": image_signature,
        "latency_method": "unique_request_p95_by_search_mode",
        "latency_p50_ms": text_p50,
        "latency_p95_ms": max(mode_p95_values, default=text_p95),
        "text_latency_p50_ms": text_p50,
        "text_latency_p95_ms": text_p95,
        **multimodal_latency,
    }
    latency_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    return updated


def _expected_search_query_ids(search_queries: pd.DataFrame, sample_size: int) -> set[str]:
    if search_queries.empty or "query_id" not in search_queries.columns:
        return set()
    return set(search_queries["query_id"].astype(str).head(sample_size).tolist())


def _search_prediction_signature(config: MarsConfig) -> str:
    raw_search = config.raw.get("search", {}) if isinstance(config.raw, dict) else {}
    top_k = raw_search.get("query_prior_top_k", 0)
    boost = raw_search.get("query_prior_boost", 0.0)
    holdout_count = raw_search.get("query_prior_holdout_count", 0)
    token_top_k = raw_search.get("query_token_prior_top_k", 0)
    token_boost = raw_search.get("query_token_prior_boost", 0.0)
    token_holdout_count = raw_search.get("query_token_prior_holdout_count", 0)
    token_exclude_holdout = raw_search.get("query_token_prior_exclude_holdout_query_keys", True)
    train_only = raw_search.get("qrels_prior_train_only", True)
    split_seed = raw_search.get("qrels_split_seed", config.seed)
    train_ratio = raw_search.get("qrels_train_ratio", 0.8)
    valid_ratio = raw_search.get("qrels_valid_ratio", 0.1)
    eval_split = evaluation_qrels_split(config)
    artifact_manifest = _read_json_dict(
        config.paths.artifacts_dir / "search" / "index_manifest.json"
    )
    allow_fallback_encoder = raw_search.get(
        "allow_fallback_encoder",
        config.search.allow_fallback_encoder,
    )
    search_text_version = artifact_manifest.get("search_text_version", "unknown")
    text_rebuilt_at = artifact_manifest.get("text_rebuilt_at", "unknown")
    encoder_type = artifact_manifest.get("encoder_type", "unknown")
    qrels_path = config.paths.processed_dir / "search_queries.parquet"
    qrels_signature = (
        f"{qrels_path.stat().st_size}:{qrels_path.stat().st_mtime_ns}"
        if qrels_path.exists()
        else "missing"
    )
    behavior_model_path = Path(
        str(
            raw_search.get(
                "query_behavior_model_path",
                config.paths.artifacts_dir / "search" / "query_behavior_model.json.gz",
            )
        )
    )
    behavior_model_signature = (
        f"{behavior_model_path.stat().st_size}:{behavior_model_path.stat().st_mtime_ns}"
        if behavior_model_path.exists()
        else "missing"
    )
    query_embedding_cache_path = Path(
        str(
            raw_search.get(
                "query_embedding_cache_path",
                config.paths.artifacts_dir / "search" / "query_embedding_cache.npz",
            )
        )
    )
    query_embedding_cache_signature = (
        f"{query_embedding_cache_path.stat().st_size}:{query_embedding_cache_path.stat().st_mtime_ns}"
        if query_embedding_cache_path.exists()
        else "missing"
    )
    return (
        "search_service_schema=5;"
        f"query_prior_top_k={top_k};query_prior_boost={boost};"
        f"query_prior_holdout_count={holdout_count};"
        f"query_token_prior_top_k={token_top_k};query_token_prior_boost={token_boost};"
        f"query_token_prior_holdout_count={token_holdout_count};"
        f"query_token_prior_exclude_holdout_query_keys={token_exclude_holdout};"
        f"qrels_prior_train_only={train_only};qrels_split_seed={split_seed};"
        f"qrels_train_ratio={train_ratio};qrels_valid_ratio={valid_ratio};"
        f"evaluation_split={eval_split};"
        f"encoder={encoder_type};allow_fallback_encoder={allow_fallback_encoder};"
        f"search_text_version={search_text_version};"
        f"text_rebuilt_at={text_rebuilt_at};qrels={qrels_signature};"
        f"query_behavior_model={behavior_model_signature};"
        f"query_embedding_cache={query_embedding_cache_signature}"
    )


def _image_artifact_signature(config: MarsConfig) -> str:
    manifest = _read_json_dict(config.paths.artifacts_dir / "search" / "index_manifest.json")
    return ";".join(
        [
            str(manifest.get("image_embedding_source", "unknown")),
            str(manifest.get("image_rebuilt_at", "unknown")),
            str(manifest.get("image_fallback_count", "unknown")),
        ]
    )


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _search_sample_size(config: MarsConfig, available: int | None = None) -> int:
    raw_eval = config.raw.get("evaluation", {}) if isinstance(config.raw, dict) else {}
    configured = raw_eval.get("search_sample") if isinstance(raw_eval, dict) else None
    if str(configured).lower() == "all":
        return max(0, int(available or 0))
    if configured:
        return (
            min(max(1, int(configured)), available)
            if available is not None
            else max(1, int(configured))
        )
    return 200 if config.active_mode == "dev" else 500


def _search_metrics(
    config: MarsConfig,
    search_queries: pd.DataFrame,
    products: pd.DataFrame,
    predictions: dict[str, list[str]],
    latency: dict[str, Any] | None = None,
    *,
    select_primary: bool = True,
) -> dict[str, Any]:
    if search_queries.empty or "positive_product_ids" not in search_queries.columns:
        return _empty_search("no_search_queries")
    if not predictions:
        return _empty_search("missing_predictions")

    relevant: dict[str, set[str]] = {}
    ranked: dict[str, list[str]] = {}
    product_categories = _product_categories(products)
    category_hits = 0
    category_evaluated = 0
    for _, row in search_queries.head(max(len(predictions), 500)).iterrows():
        query_id = str(row.get("query_id", row.name))
        if query_id not in predictions:
            continue
        positives = _as_product_list(row.get("positive_product_ids"))
        if not positives:
            continue
        relevant[query_id] = set(positives)
        ranked[query_id] = predictions[query_id]
        query_category = str(row.get("category_intent", "") or "")
        if query_category:
            category_evaluated += 1
            if any(
                product_categories.get(product_id) == query_category
                for product_id in ranked[query_id][:10]
            ):
                category_hits += 1

    latency = latency or {}
    production = {
        "mrr_at_10": mrr_at_k(ranked, relevant, k=10),
        "ndcg_at_10": ndcg_at_k(ranked, relevant, k=10),
        "recall_at_10": recall_at_k(ranked, relevant, k=10),
        "category_hit_at_10": _safe_div(category_hits, category_evaluated),
        "latency_p50_ms": float(latency.get("latency_p50_ms", 0.0) or 0.0),
        "latency_p95_ms": float(latency.get("latency_p95_ms", 0.0) or 0.0),
        "text_latency_p95_ms": float(latency.get("text_latency_p95_ms", 0.0) or 0.0),
        "image_latency_p95_ms": float(latency.get("image_latency_p95_ms", 0.0) or 0.0),
        "hybrid_latency_p95_ms": float(latency.get("hybrid_latency_p95_ms", 0.0) or 0.0),
        "multimodal_latency_sample_size": int(
            latency.get("multimodal_latency_sample_size", 0) or 0
        ),
        "unique_text_request_count": int(latency.get("unique_text_request_count", 0) or 0),
        "latency_method": str(latency.get("latency_method", "unknown")),
        "evaluated_queries": len(ranked),
        "prediction_sample_size": int(
            latency.get("sample_size", len(predictions)) or len(predictions)
        ),
        "label_source": "microsoft_hnm_search_qrels",
        "source": "supervised_qrels_test_split",
        "evaluation_split": evaluation_qrels_split(config),
        "metric_scope": "full_test" if len(ranked) == len(search_queries) else "test_sample",
    }
    return _select_search_primary(config, production) if select_primary else production


def _select_search_primary(config: MarsConfig, production: dict[str, Any]) -> dict[str, Any]:
    raw_eval = config.raw.get("evaluation", {}) if isinstance(config.raw, dict) else {}
    primary = str(raw_eval.get("search_primary", "supervised_qrels_test_split"))
    mode_latencies = [
        float(production.get(key, 0.0) or 0.0)
        for key in (
            "text_latency_p95_ms",
            "image_latency_p95_ms",
            "hybrid_latency_p95_ms",
        )
    ]
    passed = (
        float(production.get("mrr_at_10", 0.0) or 0.0) >= 0.55
        and float(production.get("ndcg_at_10", 0.0) or 0.0) >= 0.50
        and 0.0 < float(production.get("latency_p95_ms", 0.0) or 0.0) <= 200.0
        and int(production.get("multimodal_latency_sample_size", 0) or 0) > 0
        and all(0.0 < latency <= 200.0 for latency in mode_latencies)
    )
    return {
        **production,
        "primary_evaluation": primary,
        "quality_status": "pass" if passed else "fail",
        "production_with_qrels_prior": production,
        "supervised_qrels_test_split": production,
        "definition": (
            "CLIP + FAISS + lexical retrieval with supervised historical qrels priors "
            "built from the deterministic train split and evaluated only on the "
            "held-out test split."
        ),
    }


def _search_split_diagnostics(search_queries: pd.DataFrame, config: MarsConfig) -> dict[str, Any]:
    if search_queries.empty:
        return {}
    train = select_qrels_split(search_queries, config, "train")
    valid = select_qrels_split(search_queries, config, "valid")
    test = select_qrels_split(search_queries, config, "test")
    split_seed, _train_ratio, _valid_ratio = qrels_split_settings(config)
    train_query_keys = set(train["query"].map(_normalise_search_query))
    test_query_keys = set(test["query"].map(_normalise_search_query))
    train_positives = _flatten_positive_ids(train)
    test_positives = _flatten_positive_ids(test)
    return {
        "strategy": "query_id_hash",
        "seed": split_seed,
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "query_id_overlap": 0,
        "test_rows_with_seen_query_pct": 100.0
        * float(test["query"].map(_normalise_search_query).isin(train_query_keys).mean()),
        "test_unique_query_overlap_pct": 100.0
        * _safe_div(len(train_query_keys & test_query_keys), len(test_query_keys)),
        "test_positive_article_overlap_pct": 100.0
        * _safe_div(len(train_positives & test_positives), len(test_positives)),
        "interpretation": (
            "Each qrels query_id/transaction is isolated, while repeated query text and "
            "previously purchased articles may appear across train and test."
        ),
    }


def _normalise_search_query(value: Any) -> str:
    return " ".join(
        token
        for token in "".join(
            character.lower() if character.isalnum() else " " for character in str(value or "")
        ).split()
        if token
    )


def _flatten_positive_ids(search_queries: pd.DataFrame) -> set[str]:
    positives: set[str] = set()
    for value in search_queries.get("positive_product_ids", pd.Series(dtype=object)):
        positives.update(_as_product_list(value))
    return positives


def _empty_search(source: str) -> dict[str, Any]:
    return {
        "mrr_at_10": 0.0,
        "ndcg_at_10": 0.0,
        "recall_at_10": 0.0,
        "category_hit_at_10": 0.0,
        "latency_p50_ms": 0.0,
        "latency_p95_ms": 0.0,
        "text_latency_p95_ms": 0.0,
        "image_latency_p95_ms": 0.0,
        "hybrid_latency_p95_ms": 0.0,
        "multimodal_latency_sample_size": 0,
        "unique_text_request_count": 0,
        "evaluated_queries": 0,
        "prediction_sample_size": 0,
        "source": source,
    }


def _as_product_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = [token for token in text.replace(",", " ").split() if token]
        return _as_product_list(parsed)
    if pd.isna(value):
        return []
    return [str(value)]


def _product_categories(products: pd.DataFrame) -> dict[str, str]:
    if products.empty or "product_id" not in products.columns:
        return {}
    category_column = next(
        (
            column
            for column in ("category_l1", "category", "top_category")
            if column in products.columns
        ),
        None,
    )
    if not category_column:
        return {}
    return {
        str(row.product_id): str(getattr(row, category_column))
        for row in products[["product_id", category_column]].itertuples(index=False)
    }


def _recommendation_metrics(
    config: MarsConfig,
    training_events: pd.DataFrame,
    evaluation_events: pd.DataFrame,
    products: pd.DataFrame,
    predictions: dict[str, list[str]],
    candidate_k: int,
) -> dict[str, Any]:
    catalog = _catalog_items(products, pd.DataFrame(), training_events)
    continuation_eval = _session_recommendation_instances(
        evaluation_events,
        products,
        max_recent=config.recommendation.session_recent_n,
    )
    click_eval = _session_click_prediction_instances(
        evaluation_events,
        products,
        max_recent=config.recommendation.session_recent_n,
    )
    strict_eval = _strict_discovery_instances(
        evaluation_events,
        products,
        max_recent=config.recommendation.session_recent_n,
    )
    raw_eval = config.raw.get("evaluation", {}) if isinstance(config.raw, dict) else {}
    task_limit = int(raw_eval.get("recommendation_sample", 1500) or 1500)
    tasks = {
        "session_click_prediction": _recommendation_task_metrics(
            config=config,
            task_name="session_click_prediction",
            task=click_eval,
            catalog=catalog,
            candidate_k=candidate_k,
            limit=task_limit,
        ),
        "session_continuation": _recommendation_task_metrics(
            config=config,
            task_name="session_continuation",
            task=continuation_eval,
            catalog=catalog,
            candidate_k=candidate_k,
            limit=task_limit,
        ),
        "strict_discovery": _recommendation_task_metrics(
            config=config,
            task_name="strict_discovery",
            task=strict_eval,
            catalog=catalog,
            candidate_k=candidate_k,
            limit=task_limit,
        ),
    }
    primary_task = "session_click_prediction"
    if not tasks[primary_task]["evaluated_instances"]:
        primary_task = (
            "strict_discovery"
            if tasks["strict_discovery"]["evaluated_instances"]
            else "session_continuation"
        )
    primary = tasks[primary_task]

    return {
        "recall_at_300": primary["recall_at_300"],
        "hit_rate_at_50": primary["hit_rate_at_50"],
        "ndcg_at_50": primary["ndcg_at_50"],
        "coverage_at_50": primary["coverage_at_50"],
        "auc": primary["auc"],
        "candidate_latency_p50_ms": primary["candidate_latency_p50_ms"],
        "candidate_latency_p95_ms": primary["candidate_latency_p95_ms"],
        "ranking_latency_p50_ms": primary["ranking_latency_p50_ms"],
        "ranking_latency_p95_ms": primary["ranking_latency_p95_ms"],
        "reranking_latency_p50_ms": primary["reranking_latency_p50_ms"],
        "reranking_latency_p95_ms": primary["reranking_latency_p95_ms"],
        "total_latency_p50_ms": primary["total_latency_p50_ms"],
        "total_latency_p95_ms": primary["total_latency_p95_ms"],
        "evaluated_users": primary["evaluated_users"],
        "evaluated_instances": primary["evaluated_instances"],
        "primary_task": primary_task,
        "tasks": tasks,
        "source": f"service_{primary_task}",
        "evaluation_split": "test_events",
        "training_events": int(len(training_events)),
        "evaluation_events": int(len(evaluation_events)),
        "prediction_override_used": bool(predictions),
    }


def _recommendation_task_metrics(
    *,
    config: MarsConfig,
    task_name: str,
    task: dict[str, Any],
    catalog: list[str],
    candidate_k: int,
    limit: int,
) -> dict[str, Any]:
    relevant = dict(task.get("relevant", {}))
    contexts = dict(task.get("contexts", {}))
    user_ids = dict(task.get("user_ids", {}))
    if limit > 0 and len(relevant) > limit:
        keys = sorted(relevant)[:limit]
        relevant = {key: relevant[key] for key in keys}
        contexts = {key: contexts.get(key, {}) for key in keys}
        user_ids = {key: user_ids.get(key, key) for key in keys}
    if not relevant:
        return _empty_recommendation_task(task_name, "no_relevant_instances")

    live = _service_ranked_by_user(
        config,
        relevant,
        candidate_k=candidate_k,
        contexts=contexts,
        user_ids=user_ids,
    )
    ranked = live["ranked"]
    relevant = {key: relevant[key] for key in ranked if key in relevant}
    y_true = live.get("auc_labels", [])
    y_score = live.get("auc_scores", [])
    latency_summary = live.get("latency", {})
    return {
        "recall_at_300": recall_at_k(ranked, relevant, k=300),
        "hit_rate_at_50": hit_rate_at_k(ranked, relevant, k=50),
        "ndcg_at_50": ndcg_at_k(ranked, relevant, k=50),
        "coverage_at_50": coverage_at_k(ranked, catalog, k=50),
        "auc": auc_score(y_true, y_score),
        "candidate_latency_p50_ms": latency_summary.get("candidate_p50_ms", 0.0),
        "candidate_latency_p95_ms": latency_summary.get("candidate_p95_ms", 0.0),
        "ranking_latency_p50_ms": latency_summary.get("ranking_p50_ms", 0.0),
        "ranking_latency_p95_ms": latency_summary.get("ranking_p95_ms", 0.0),
        "reranking_latency_p50_ms": latency_summary.get("reranking_p50_ms", 0.0),
        "reranking_latency_p95_ms": latency_summary.get("reranking_p95_ms", 0.0),
        "total_latency_p50_ms": latency_summary.get("total_p50_ms", 0.0),
        "total_latency_p95_ms": latency_summary.get("total_p95_ms", 0.0),
        "evaluated_users": int(live.get("evaluated_users", 0)),
        "evaluated_instances": int(live.get("evaluated_instances", len(ranked))),
        "source": f"service_{task_name}",
        "definition": task.get("definition", ""),
    }


def _empty_recommendation_task(task_name: str, source: str) -> dict[str, Any]:
    return {
        "recall_at_300": 0.0,
        "hit_rate_at_50": 0.0,
        "ndcg_at_50": 0.0,
        "coverage_at_50": 0.0,
        "auc": 0.5,
        "candidate_latency_p50_ms": 0.0,
        "candidate_latency_p95_ms": 0.0,
        "ranking_latency_p50_ms": 0.0,
        "ranking_latency_p95_ms": 0.0,
        "reranking_latency_p50_ms": 0.0,
        "reranking_latency_p95_ms": 0.0,
        "total_latency_p50_ms": 0.0,
        "total_latency_p95_ms": 0.0,
        "evaluated_users": 0,
        "evaluated_instances": 0,
        "source": source,
        "definition": task_name,
    }


def _baseline_metrics(
    *,
    config: MarsConfig,
    search_queries: pd.DataFrame,
    products: pd.DataFrame,
    search_metrics: dict[str, Any],
    recommendation_metrics: dict[str, Any],
    training_events: pd.DataFrame,
    evaluation_events: pd.DataFrame,
    candidate_k: int,
) -> dict[str, Any]:
    search_baseline = _search_bm25_baseline(config, search_queries, products)
    recommendation_baselines = _recommendation_baselines(
        config=config,
        products=products,
        training_events=training_events,
        evaluation_events=evaluation_events,
        candidate_k=candidate_k,
    )
    return {
        "search": {
            "bm25_text_only": search_baseline,
            "improvement_vs_bm25_text_only": _improvement_block(
                search_metrics,
                search_baseline,
                ["mrr_at_10", "ndcg_at_10", "recall_at_10"],
            ),
        },
        "recommendation": {
            **recommendation_baselines,
            "improvement_vs_popularity_only": _improvement_block(
                recommendation_metrics,
                recommendation_baselines.get("popularity_only", {}),
                ["recall_at_300", "hit_rate_at_50", "ndcg_at_50", "coverage_at_50"],
            ),
            "improvement_vs_two_tower_only": _improvement_block(
                recommendation_metrics,
                recommendation_baselines.get("two_tower_only", {}),
                ["recall_at_300", "hit_rate_at_50", "ndcg_at_50", "coverage_at_50"],
            ),
        },
    }


def _search_bm25_baseline(
    config: MarsConfig,
    search_queries: pd.DataFrame,
    products: pd.DataFrame,
) -> dict[str, Any]:
    if search_queries.empty or "query" not in search_queries.columns:
        return _empty_search("bm25_text_only_missing_queries")
    try:
        from mars.search.service import SearchService, _query_tokens

        service = SearchService(config)
    except Exception:
        return _empty_search("bm25_text_only_unavailable")

    sample_size = _search_sample_size(config, len(search_queries))
    ranked: dict[str, list[str]] = {}
    for row in search_queries.head(sample_size).itertuples(index=False):
        query = str(getattr(row, "query", "") or "")
        if not query or query.lower() == "nan":
            continue
        query_id = str(getattr(row, "query_id", len(ranked)))
        tokens = _query_tokens(query)
        lexical_candidates = service._lexical_candidate_scores(tokens, top_k=10)  # noqa: SLF001
        ranked[query_id] = [
            str(service._metadata_records[row_id].get("product_id"))  # noqa: SLF001
            for row_id, _score in lexical_candidates
            if 0 <= row_id < len(service._metadata_records)  # noqa: SLF001
        ]
    return _search_metrics(
        config,
        search_queries,
        products,
        ranked,
        {"sample_size": len(ranked)},
        select_primary=False,
    )


def _recommendation_baselines(
    *,
    config: MarsConfig,
    products: pd.DataFrame,
    training_events: pd.DataFrame,
    evaluation_events: pd.DataFrame,
    candidate_k: int,
) -> dict[str, Any]:
    catalog = _catalog_items(products, pd.DataFrame(), training_events)
    session_eval = _session_recommendation_instances(
        evaluation_events,
        products,
        max_recent=config.recommendation.session_recent_n,
    )
    relevant = session_eval.get("relevant", {})
    if not relevant:
        relevant = _relevant_by_user(pd.DataFrame(), evaluation_events, min_label=0.7)
    if not relevant:
        return {
            "popularity_only": _empty_recommendation_baseline("popularity_only_no_relevant"),
            "two_tower_only": _empty_recommendation_baseline("two_tower_only_no_relevant"),
        }
    raw_eval = config.raw.get("evaluation", {}) if isinstance(config.raw, dict) else {}
    baseline_limit = int(raw_eval.get("recommendation_sample", 400) or 400)
    if baseline_limit > 0 and len(relevant) > baseline_limit:
        keys = sorted(relevant)[:baseline_limit]
        relevant = {key: relevant[key] for key in keys}
        session_eval["contexts"] = {
            key: session_eval.get("contexts", {}).get(key, {}) for key in keys
        }
        session_eval["user_ids"] = {
            key: session_eval.get("user_ids", {}).get(key, key) for key in keys
        }

    popularity_order = _popularity_order(training_events, products)
    popularity_ranked = {
        evaluation_key: popularity_order[: max(candidate_k, 50)] for evaluation_key in relevant
    }
    two_tower_ranked = _two_tower_only_ranked(
        config,
        relevant,
        candidate_k=candidate_k,
        contexts=session_eval.get("contexts", {}),
        user_ids=session_eval.get("user_ids", {}),
    )
    return {
        "popularity_only": _ranked_metric_block(
            popularity_ranked,
            relevant,
            catalog,
            source="popularity_only",
        ),
        "two_tower_only": _ranked_metric_block(
            two_tower_ranked,
            relevant,
            catalog,
            source="two_tower_candidate_generation_only",
        ),
    }


def _two_tower_only_ranked(
    config: MarsConfig,
    relevant: dict[str, set[str]],
    *,
    candidate_k: int,
    contexts: dict[str, dict[str, Any]] | None = None,
    user_ids: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    if not relevant:
        return {}
    try:
        from mars.recommendation.service import RecommendationService

        service = RecommendationService(config)
    except Exception:
        return {}

    contexts = contexts or {}
    user_ids = user_ids or {}
    ranked: dict[str, list[str]] = {}
    for evaluation_key in sorted(relevant):
        user_id = str(user_ids.get(evaluation_key, evaluation_key))
        session_context = dict(contexts.get(evaluation_key, {}))
        candidates = service.generate_candidates(user_id, session_context, candidate_k=candidate_k)
        ranked[evaluation_key] = [
            str(item["product"].get("product_id")) for item in candidates[: max(candidate_k, 50)]
        ]
    return ranked


def _popularity_order(training_events: pd.DataFrame, products: pd.DataFrame) -> list[str]:
    if not training_events.empty and {"product_id", "event_type"}.issubset(training_events.columns):
        weights = {"search": 0.05, "view": 0.2, "cart": 0.7, "purchase": 1.0}
        weighted = training_events.dropna(subset=["product_id"]).copy()
        weighted["_weight"] = weighted["event_type"].astype(str).map(weights).fillna(0.0)
        counts = weighted.groupby("product_id")["_weight"].sum().sort_values(ascending=False)
        order = [str(product_id) for product_id in counts.index.tolist()]
        if order:
            seen = set(order)
            tail = [
                str(product_id)
                for product_id in products.get("product_id", pd.Series(dtype=str))
                .astype(str)
                .tolist()
                if str(product_id) not in seen
            ]
            return order + tail
    if "popularity_prior" in products.columns:
        return (
            products.sort_values("popularity_prior", ascending=False)["product_id"]
            .astype(str)
            .tolist()
        )
    return products.get("product_id", pd.Series(dtype=str)).astype(str).tolist()


def _ranked_metric_block(
    ranked: dict[str, list[str]],
    relevant: dict[str, set[str]],
    catalog: list[str],
    *,
    source: str,
) -> dict[str, Any]:
    if not ranked or not relevant:
        return _empty_recommendation_baseline(source)
    return {
        "recall_at_300": recall_at_k(ranked, relevant, k=300),
        "hit_rate_at_50": hit_rate_at_k(ranked, relevant, k=50),
        "ndcg_at_50": ndcg_at_k(ranked, relevant, k=50),
        "coverage_at_50": coverage_at_k(ranked, catalog, k=50),
        "evaluated_instances": len(ranked),
        "source": source,
    }


def _empty_recommendation_baseline(source: str) -> dict[str, Any]:
    return {
        "recall_at_300": 0.0,
        "hit_rate_at_50": 0.0,
        "ndcg_at_50": 0.0,
        "coverage_at_50": 0.0,
        "evaluated_instances": 0,
        "source": source,
    }


def _improvement_block(
    final_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    keys: list[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in keys:
        final_value = float(final_metrics.get(key, 0.0) or 0.0)
        baseline_value = float(baseline_metrics.get(key, 0.0) or 0.0)
        output[key] = {
            "final": final_value,
            "baseline": baseline_value,
            "absolute_delta": final_value - baseline_value,
            "relative_improvement": (
                None if baseline_value == 0.0 else (final_value - baseline_value) / baseline_value
            ),
        }
    return output


def _catalog_items(
    products: pd.DataFrame, interactions: pd.DataFrame, events: pd.DataFrame
) -> list[str]:
    for frame in [products, interactions, events]:
        if not frame.empty and "product_id" in frame.columns:
            values = frame["product_id"].dropna().astype(str).unique().tolist()
            if values:
                return values
    return []


def _relevant_by_user(
    interactions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    min_label: float = 0.0,
) -> dict[str, set[str]]:
    source = interactions if not interactions.empty else events
    if source.empty or not {"user_id", "product_id"}.issubset(source.columns):
        return {}
    if "label" in source.columns:
        source = source[source["label"].astype(float) >= min_label]
    elif "event_type" in source.columns:
        if min_label >= 0.7:
            source = source[source["event_type"].astype(str).isin(["cart", "purchase"])]
        else:
            source = source[source["event_type"].astype(str).isin(["view", "cart", "purchase"])]

    relevant: dict[str, set[str]] = {}
    for user_id, rows in source.dropna(subset=["user_id", "product_id"]).groupby("user_id"):
        relevant[str(user_id)] = set(rows["product_id"].astype(str).tolist())
    return relevant


def _session_recommendation_instances(
    events: pd.DataFrame,
    products: pd.DataFrame,
    *,
    max_recent: int,
) -> dict[str, Any]:
    if events.empty or not {"user_id", "session_id", "event_type", "product_id"}.issubset(
        events.columns
    ):
        return {"relevant": {}, "contexts": {}, "user_ids": {}, "evaluated_users": 0}
    category_by_product = _product_categories(products)
    relevant: dict[str, set[str]] = {}
    contexts: dict[str, dict[str, Any]] = {}
    user_ids: dict[str, str] = {}
    unique_users: set[str] = set()

    sort_columns = [
        column for column in ("session_id", "timestamp", "event_id") if column in events.columns
    ]
    ordered = events.sort_values(sort_columns) if sort_columns else events
    for session_id, rows in ordered.groupby("session_id", sort=False):
        recent_products: list[str] = []
        recent_categories: list[str] = []
        event_counts: dict[str, int] = {}
        previous_events = 0
        for row in rows.itertuples(index=False):
            event_type = str(getattr(row, "event_type", "") or "")
            product_id = str(getattr(row, "product_id", "") or "")
            if product_id.lower() == "nan":
                product_id = ""
            user_id = str(getattr(row, "user_id", "") or "")
            event_id = str(getattr(row, "event_id", previous_events) or previous_events)
            if event_type in {"cart", "purchase"} and product_id and recent_products:
                key = f"{user_id}::{session_id}::{event_id}"
                relevant[key] = {product_id}
                contexts[key] = {
                    "recent_products": recent_products[-max_recent:],
                    "recent_categories": recent_categories[-max_recent:],
                    "event_counts": dict(event_counts),
                    "num_recent_events": previous_events,
                    "session_id": str(session_id),
                    "evaluation_context": "test_session_prefix_before_target",
                }
                user_ids[key] = user_id
                unique_users.add(user_id)

            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            previous_events += 1
            if event_type in {"view", "cart", "purchase"} and product_id:
                recent_products.append(product_id)
                category = category_by_product.get(product_id)
                if category:
                    recent_categories.append(category)

    return {
        "relevant": relevant,
        "contexts": contexts,
        "user_ids": user_ids,
        "evaluated_users": len(unique_users),
    }


def _session_click_prediction_instances(
    events: pd.DataFrame,
    products: pd.DataFrame,
    *,
    max_recent: int,
) -> dict[str, Any]:
    if events.empty or not {"user_id", "session_id", "event_type", "product_id"}.issubset(
        events.columns
    ):
        return {"relevant": {}, "contexts": {}, "user_ids": {}, "evaluated_users": 0}
    relevant: dict[str, set[str]] = {}
    contexts: dict[str, dict[str, Any]] = {}
    user_ids: dict[str, str] = {}
    unique_users: set[str] = set()

    sort_columns = [
        column for column in ("session_id", "timestamp", "event_id") if column in events.columns
    ]
    ordered = events.sort_values(sort_columns) if sort_columns else events
    for session_id, rows in ordered.groupby("session_id", sort=False):
        recent_products: list[str] = []
        recent_categories: list[str] = []
        event_counts: dict[str, int] = {}
        previous_events = 0
        pending_search_context: dict[str, Any] | None = None
        for row in rows.itertuples(index=False):
            event_type = str(getattr(row, "event_type", "") or "")
            product_id = str(getattr(row, "product_id", "") or "")
            if product_id.lower() == "nan":
                product_id = ""
            user_id = str(getattr(row, "user_id", "") or "")
            event_id = str(getattr(row, "event_id", previous_events) or previous_events)
            query = str(getattr(row, "query", "") or "")
            category = _row_category(row)

            if event_type == "search":
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                previous_events += 1
                query_categories = recent_categories + ([category] if category else [])
                pending_search_context = {
                    "recent_products": recent_products[-max_recent:],
                    "recent_categories": query_categories[-max_recent:],
                    "event_counts": dict(event_counts),
                    "num_recent_events": previous_events,
                    "session_id": str(session_id),
                    "query": query,
                    "query_intent_category": category,
                    "evaluation_context": "search_prefix_before_first_view",
                }
                if category:
                    recent_categories.append(category)
                continue

            if event_type == "view" and product_id and pending_search_context:
                key = f"{user_id}::{session_id}::{event_id}"
                relevant[key] = {product_id}
                contexts[key] = dict(pending_search_context)
                user_ids[key] = user_id
                unique_users.add(user_id)
                pending_search_context = None

            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            previous_events += 1
            if event_type in {"view", "cart", "purchase"} and product_id:
                recent_products.append(product_id)
                if category:
                    recent_categories.append(category)

    return {
        "relevant": relevant,
        "contexts": contexts,
        "user_ids": user_ids,
        "evaluated_users": len(unique_users),
        "definition": (
            "first view after a search event, before the viewed product enters recent_products"
        ),
    }


def _strict_discovery_instances(
    events: pd.DataFrame,
    products: pd.DataFrame,
    *,
    max_recent: int,
) -> dict[str, Any]:
    if events.empty or not {"user_id", "session_id", "event_type", "product_id"}.issubset(
        events.columns
    ):
        return {"relevant": {}, "contexts": {}, "user_ids": {}, "evaluated_users": 0}
    relevant: dict[str, set[str]] = {}
    contexts: dict[str, dict[str, Any]] = {}
    user_ids: dict[str, str] = {}
    unique_users: set[str] = set()

    sort_columns = [
        column for column in ("session_id", "timestamp", "event_id") if column in events.columns
    ]
    ordered = events.sort_values(sort_columns) if sort_columns else events
    for session_id, rows in ordered.groupby("session_id", sort=False):
        recent_products: list[str] = []
        recent_categories: list[str] = []
        event_counts: dict[str, int] = {}
        previous_events = 0
        for row in rows.itertuples(index=False):
            event_type = str(getattr(row, "event_type", "") or "")
            product_id = str(getattr(row, "product_id", "") or "")
            if product_id.lower() == "nan":
                product_id = ""
            user_id = str(getattr(row, "user_id", "") or "")
            event_id = str(getattr(row, "event_id", previous_events) or previous_events)
            query = str(getattr(row, "query", "") or "")
            category = _row_category(row)

            if event_type == "search":
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                previous_events += 1
                if category:
                    recent_categories.append(category)
                continue

            if (
                event_type in {"view", "cart", "purchase"}
                and product_id
                and recent_products
                and product_id not in set(recent_products)
            ):
                key = f"{user_id}::{session_id}::{event_id}"
                relevant[key] = {product_id}
                contexts[key] = {
                    "recent_products": recent_products[-max_recent:],
                    "recent_categories": recent_categories[-max_recent:],
                    "event_counts": dict(event_counts),
                    "num_recent_events": previous_events,
                    "session_id": str(session_id),
                    "query": query,
                    "query_intent_category": category,
                    "evaluation_context": "next_distinct_product_before_target",
                }
                user_ids[key] = user_id
                unique_users.add(user_id)

            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            previous_events += 1
            if event_type in {"view", "cart", "purchase"} and product_id:
                recent_products.append(product_id)
                if category:
                    recent_categories.append(category)

    return {
        "relevant": relevant,
        "contexts": contexts,
        "user_ids": user_ids,
        "evaluated_users": len(unique_users),
        "definition": "next clicked product that is not already in recent_products",
    }


def _row_category(row: Any) -> str:
    for column in ("category", "query_intent_category", "category_l1", "top_category"):
        value = getattr(row, column, "")
        if value is None:
            continue
        text = str(value)
        if text and text.lower() != "nan":
            return text
    return ""


def _baseline_ranked_by_user(
    relevant: dict[str, set[str]], catalog: list[str], candidate_k: int
) -> dict[str, list[str]]:
    ranked: dict[str, list[str]] = {}
    for user_id, positives in relevant.items():
        remaining = [item for item in catalog if item not in positives]
        ranked[user_id] = list(positives) + remaining[: max(candidate_k, 50)]
    return ranked


def _service_ranked_by_user(
    config: MarsConfig,
    relevant: dict[str, set[str]],
    *,
    candidate_k: int,
    contexts: dict[str, dict[str, Any]] | None = None,
    user_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not relevant:
        return {
            "ranked": {},
            "latency": {},
            "evaluated_users": 0,
            "auc_labels": [],
            "auc_scores": [],
        }

    from mars.recommendation.service import RecommendationService

    service = RecommendationService(config)
    max_users = len(relevant)
    ranked: dict[str, list[str]] = {}
    candidate_ms: list[float] = []
    ranking_ms: list[float] = []
    reranking_ms: list[float] = []
    total_ms: list[float] = []
    auc_labels: list[float] = []
    auc_scores: list[float] = []
    contexts = contexts or {}
    user_ids = user_ids or {}

    for idx, evaluation_key in enumerate(sorted(relevant)[:max_users]):
        user_id = str(user_ids.get(evaluation_key, evaluation_key))
        started = time.perf_counter()
        session_context: dict[str, Any] = dict(contexts.get(evaluation_key, {}))

        stage_started = time.perf_counter()
        candidates = service.generate_candidates(user_id, session_context, candidate_k=candidate_k)
        candidate_ms.append(_elapsed_ms(stage_started))

        user = service.artifacts.users.get(user_id) if service.artifacts else None
        stage_started = time.perf_counter()
        ranked_items = service.rank_candidates(user, candidates, session_context)
        ranking_ms.append(_elapsed_ms(stage_started))

        stage_started = time.perf_counter()
        reranked_items = service.rerank(
            user_id,
            ranked_items,
            top_n=min(50, len(ranked_items)),
            request_id=f"eval-{idx}",
        )
        reranking_ms.append(_elapsed_ms(stage_started))

        reranked_ids = [str(item["product"].get("product_id")) for item in reranked_items]
        seen = set(reranked_ids)
        remaining_ids = [
            str(item["product"].get("product_id"))
            for item in ranked_items
            if str(item["product"].get("product_id")) not in seen
        ]
        ranked[evaluation_key] = (
            reranked_ids + remaining_ids[: max(candidate_k - len(reranked_ids), 0)]
        )

        positives = relevant.get(evaluation_key, set())
        for item in ranked_items[: min(candidate_k, 300)]:
            product_id = str(item["product"].get("product_id"))
            auc_labels.append(1.0 if product_id in positives else 0.0)
            auc_scores.append(
                float(item.get("ranking_score", item.get("candidate_score", 0.0)) or 0.0)
            )

        total_ms.append(_elapsed_ms(started))

    return {
        "ranked": ranked,
        "latency": {
            "candidate_p50_ms": _percentile_values(candidate_ms, 50),
            "candidate_p95_ms": _percentile_values(candidate_ms, 95),
            "ranking_p50_ms": _percentile_values(ranking_ms, 50),
            "ranking_p95_ms": _percentile_values(ranking_ms, 95),
            "reranking_p50_ms": _percentile_values(reranking_ms, 50),
            "reranking_p95_ms": _percentile_values(reranking_ms, 95),
            "total_p50_ms": _percentile_values(total_ms, 50),
            "total_p95_ms": _percentile_values(total_ms, 95),
        },
        "evaluated_users": len({str(user_ids.get(key, key)) for key in ranked}),
        "evaluated_instances": len(ranked),
        "auc_labels": auc_labels,
        "auc_scores": auc_scores,
    }


def _ranking_labels(
    interactions: pd.DataFrame, events: pd.DataFrame
) -> tuple[list[float], list[float]]:
    source = interactions if not interactions.empty else events
    if source.empty:
        return [], []
    if "label" in source.columns:
        labels = source["label"].astype(float).clip(0, 1).tolist()
        scores = source.get("score", source["label"]).astype(float).tolist()
        return labels, scores
    if "event_type" in source.columns:
        weights = {"search": 0.05, "view": 0.2, "cart": 0.7, "purchase": 1.0}
        scores = source["event_type"].astype(str).map(weights).fillna(0.0).astype(float).tolist()
        labels = source["event_type"].astype(str).isin(["cart", "purchase"]).astype(float).tolist()
        return labels, scores
    return [], []


def _with_ab_group(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty or "ab_group" in events.columns:
        return events
    output = events.copy()
    output["ab_group"] = (
        output["user_id"].astype(str).map(lambda user: assign_bucket(user).bucket)
        if "user_id" in output.columns
        else "control"
    )
    return output


def _percentile(frame: pd.DataFrame, column: str, percentile: float) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return 0.0 if values.empty else float(np.percentile(values.to_numpy(dtype=float), percentile))


def _percentile_values(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _target_status(
    search: dict[str, Any], recommendation: dict[str, Any], ctr_threshold: float
) -> dict[str, Any]:
    return {
        "search_mrr_at_10": _status(search.get("mrr_at_10", 0.0), 0.55, True),
        "search_ndcg_at_10": _status(search.get("ndcg_at_10", 0.0), 0.50, True),
        "search_latency_p95_ms": _status(search.get("latency_p95_ms", 0.0), 200.0, False),
        "candidate_recall_at_300": _status(recommendation.get("recall_at_300", 0.0), 0.30, True),
        "ranking_auc": _status(recommendation.get("auc", 0.5), 0.70, True),
        "recommendation_hit_rate_at_50": _status(
            recommendation.get("hit_rate_at_50", 0.0), 0.20, True
        ),
        "recommendation_ndcg_at_50": _status(recommendation.get("ndcg_at_50", 0.0), 0.08, True),
        "recommendation_coverage_at_50": _status(
            recommendation.get("coverage_at_50", 0.0), 0.20, True
        ),
        "monitoring_ctr_threshold": ctr_threshold,
    }


def _status(value: float, target: float, higher_is_better: bool) -> dict[str, float | str]:
    met = value >= target if higher_is_better else value <= target
    return {"value": float(value), "target": float(target), "status": "met" if met else "not_met"}
