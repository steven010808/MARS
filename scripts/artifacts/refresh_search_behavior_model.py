from __future__ import annotations

import argparse
import copy
import gzip
import json
import shutil
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mars.config import MarsConfig, load_config  # noqa: E402
from mars.evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k  # noqa: E402
from mars.search.behavior_model import (  # noqa: E402
    behavior_model_summary,
    build_query_behavior_model_payload,
    write_query_behavior_model,
)
from mars.search.feedback import (  # noqa: E402
    build_search_feedback_frame,
    feedback_summary,
    read_event_log,
)
from mars.search.qrels import select_qrels_split  # noqa: E402
from mars.search.service import SearchService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build, validate, and optionally promote a live-updated search behavior model."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", default="full")
    parser.add_argument("--logs", default="")
    parser.add_argument("--feedback", default="")
    parser.add_argument("--candidate", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--validation-ratio", type=float, default=None)
    parser.add_argument("--max-eval-queries", type=int, default=5000)
    parser.add_argument("--max-mrr-drop", type=float, default=0.002)
    parser.add_argument("--max-ndcg-drop", type=float, default=0.002)
    parser.add_argument("--max-latency-p95-ms", type=float, default=200.0)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, mode=args.mode)
    raw_search = config.raw.get("search", {}) if isinstance(config.raw, dict) else {}
    online = raw_search.get("online_learning", {})
    online = online if isinstance(online, dict) else {}
    if not bool(online.get("enabled", False)):
        raise SystemExit("search.online_learning.enabled must be true to refresh live behavior")

    generated_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    logs_path = Path(args.logs) if args.logs else config.paths.logs_dir / "api_events.jsonl"
    default_feedback_path = online.get("feedback_path") or (
        config.paths.processed_dir / "search_feedback.parquet"
    )
    feedback_path = Path(args.feedback) if args.feedback else Path(str(default_feedback_path))
    candidate_path = (
        Path(args.candidate)
        if args.candidate
        else config.paths.artifacts_dir
        / "search"
        / "candidates"
        / f"query_behavior_model_{generated_at}.json.gz"
    )
    report_path = (
        Path(args.report)
        if args.report
        else config.paths.artifacts_dir / "reports" / "search_behavior_refresh.json"
    )
    validation_ratio = (
        float(args.validation_ratio)
        if args.validation_ratio is not None
        else float(online.get("validation_ratio", 0.2) or 0.2)
    )

    feedback = _build_feedback(config, logs_path, feedback_path, validation_ratio)
    payload = build_query_behavior_model_payload(config, feedback_path=feedback_path)
    write_query_behavior_model(payload, candidate_path)

    baseline_metrics = evaluate_search_behavior_model(
        config,
        split="valid",
        max_queries=args.max_eval_queries,
    )
    candidate_config = _config_with_behavior_model_path(config, candidate_path)
    candidate_metrics = evaluate_search_behavior_model(
        candidate_config,
        split="valid",
        max_queries=args.max_eval_queries,
    )
    live_baseline = evaluate_live_feedback(config, feedback_path, max_queries=args.max_eval_queries)
    live_candidate = evaluate_live_feedback(
        candidate_config,
        feedback_path,
        max_queries=args.max_eval_queries,
    )
    decision = _promotion_decision(
        baseline_metrics,
        candidate_metrics,
        live_baseline,
        live_candidate,
        max_mrr_drop=args.max_mrr_drop,
        max_ndcg_drop=args.max_ndcg_drop,
        max_latency_p95_ms=args.max_latency_p95_ms,
    )
    promoted = False
    if args.promote and decision["promote"]:
        promoted = _promote_candidate(config, candidate_path, decision, payload)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "logs_path": str(logs_path),
        "feedback": feedback,
        "candidate": behavior_model_summary(payload, candidate_path),
        "baseline_valid": baseline_metrics,
        "candidate_valid": candidate_metrics,
        "baseline_live_valid": live_baseline,
        "candidate_live_valid": live_candidate,
        "decision": decision,
        "promote_requested": bool(args.promote),
        "promoted": promoted,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def evaluate_search_behavior_model(
    config: MarsConfig,
    *,
    split: str = "valid",
    max_queries: int = 5000,
) -> dict[str, Any]:
    queries_path = config.paths.processed_dir / "search_queries.parquet"
    if not queries_path.exists():
        return _empty_metrics("missing_qrels")
    queries = pd.read_parquet(queries_path)
    evaluation = select_qrels_split(queries, config, split)
    return _evaluate_query_frame(
        config,
        evaluation,
        max_queries=max_queries,
        source=f"qrels_{split}",
    )


def evaluate_live_feedback(
    config: MarsConfig,
    feedback_path: str | Path,
    *,
    max_queries: int = 5000,
) -> dict[str, Any]:
    path = Path(feedback_path)
    if not path.exists():
        return _empty_metrics("missing_live_feedback")
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if frame.empty or "split" not in frame:
        return _empty_metrics("empty_live_feedback")
    positives = frame[(frame["split"].astype(str) == "valid") & (frame["label"].astype(float) > 0)]
    if positives.empty:
        return _empty_metrics("no_live_valid_positives")
    grouped = (
        positives.groupby(["query", "query_key"], as_index=False)["product_id"]
        .agg(lambda values: list(dict.fromkeys(str(value) for value in values)))
        .rename(columns={"product_id": "positive_product_ids"})
    )
    grouped["query_id"] = [f"live-valid-{idx}" for idx in range(len(grouped))]
    return _evaluate_query_frame(
        config,
        grouped,
        max_queries=max_queries,
        source="live_feedback_valid",
    )


def _evaluate_query_frame(
    config: MarsConfig,
    queries: pd.DataFrame,
    *,
    max_queries: int,
    source: str,
) -> dict[str, Any]:
    if queries.empty:
        return _empty_metrics(f"empty_{source}")
    sample = queries.head(max(1, int(max_queries)))
    try:
        service = SearchService(config)
    except Exception as exc:
        return {**_empty_metrics("service_unavailable"), "error": exc.__class__.__name__}

    predictions: dict[str, list[str]] = {}
    relevant: dict[str, set[str]] = {}
    latencies: list[float] = []
    query_cache: dict[str, tuple[list[str], float]] = {}
    for row in sample.itertuples(index=False):
        query = str(getattr(row, "query", "") or "")
        if not query:
            continue
        query_id = str(getattr(row, "query_id", len(predictions)))
        positives = {
            str(product_id)
            for product_id in _as_product_list(getattr(row, "positive_product_ids", []))
        }
        if not positives:
            continue
        cache_key = query.strip().lower()
        if cache_key in query_cache:
            ranked, latency_ms = query_cache[cache_key]
        else:
            started = time.perf_counter()
            response = service.search(
                {
                    "search_type": "text",
                    "query": query,
                    "top_k": 10,
                    "filters": {},
                }
            )
            latency_ms = float(response.get("latency_ms", 0.0) or 0.0)
            if latency_ms <= 0:
                latency_ms = (time.perf_counter() - started) * 1000.0
            ranked = [str(item["product_id"]) for item in response.get("results", [])]
            query_cache[cache_key] = (ranked, latency_ms)
            latencies.append(latency_ms)
        predictions[query_id] = list(ranked)
        relevant[query_id] = positives
    return {
        "source": source,
        "evaluated_queries": int(len(relevant)),
        "unique_queries": int(len(query_cache)),
        "mrr_at_10": mrr_at_k(predictions, relevant, k=10),
        "ndcg_at_10": ndcg_at_k(predictions, relevant, k=10),
        "recall_at_10": recall_at_k(predictions, relevant, k=10),
        "latency_p95_ms": _percentile(latencies, 95),
    }


def _promotion_decision(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    live_baseline: dict[str, Any],
    live_candidate: dict[str, Any],
    *,
    max_mrr_drop: float,
    max_ndcg_drop: float,
    max_latency_p95_ms: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    if int(candidate.get("evaluated_queries", 0) or 0) <= 0:
        reasons.append("candidate_valid_empty")
    if float(candidate.get("mrr_at_10", 0.0) or 0.0) < max(
        0.55,
        float(baseline.get("mrr_at_10", 0.0) or 0.0) - max_mrr_drop,
    ):
        reasons.append("candidate_mrr_regressed")
    if float(candidate.get("ndcg_at_10", 0.0) or 0.0) < max(
        0.50,
        float(baseline.get("ndcg_at_10", 0.0) or 0.0) - max_ndcg_drop,
    ):
        reasons.append("candidate_ndcg_regressed")
    if float(candidate.get("latency_p95_ms", 0.0) or 0.0) > max_latency_p95_ms:
        reasons.append("candidate_latency_over_budget")
    if int(live_candidate.get("evaluated_queries", 0) or 0) > 0:
        live_mrr_floor = float(live_baseline.get("mrr_at_10", 0.0) or 0.0) - max_mrr_drop
        live_ndcg_floor = float(live_baseline.get("ndcg_at_10", 0.0) or 0.0) - max_ndcg_drop
        if float(live_candidate.get("mrr_at_10", 0.0) or 0.0) < live_mrr_floor:
            reasons.append("candidate_live_mrr_regressed")
        if float(live_candidate.get("ndcg_at_10", 0.0) or 0.0) < live_ndcg_floor:
            reasons.append("candidate_live_ndcg_regressed")
    return {
        "promote": not reasons,
        "reasons": reasons,
        "criteria": {
            "qrels_valid_mrr_min": 0.55,
            "qrels_valid_ndcg_min": 0.50,
            "max_mrr_drop": max_mrr_drop,
            "max_ndcg_drop": max_ndcg_drop,
            "max_latency_p95_ms": max_latency_p95_ms,
        },
    }


def _build_feedback(
    config: MarsConfig,
    logs_path: Path,
    feedback_path: Path,
    validation_ratio: float,
) -> dict[str, Any]:
    events = read_event_log(logs_path)
    frame = build_search_feedback_frame(
        events,
        catalog_products=_catalog_products(config),
        validation_ratio=validation_ratio,
    )
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(feedback_path, index=False)
    return {
        **feedback_summary(frame),
        "path": str(feedback_path),
        "validation_ratio": validation_ratio,
    }


def _promote_candidate(
    config: MarsConfig,
    candidate_path: Path,
    decision: dict[str, Any],
    payload: dict[str, Any],
) -> bool:
    raw_search = config.raw.get("search", {}) if isinstance(config.raw, dict) else {}
    destination = Path(
        str(
            raw_search.get(
                "query_behavior_model_path",
                config.paths.artifacts_dir / "search" / "query_behavior_model.json.gz",
            )
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_path, destination)
    _register_search_model(config, destination, candidate_path, decision, payload)
    return True


def _register_search_model(
    config: MarsConfig,
    active_path: Path,
    candidate_path: Path,
    decision: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    registry_path = config.paths.artifacts_dir / "registry" / "search_models.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {"active_version": None, "versions": []}
    versions = registry.setdefault("versions", [])
    version = f"sv{len(versions) + 1:04d}"
    for entry in versions:
        if entry.get("status") == "active":
            entry["status"] = "archived"
    entry = {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "active",
        "active_artifact_path": str(active_path),
        "candidate_artifact_path": str(candidate_path),
        "decision": decision,
        "metadata": {
            "schema_version": payload.get("schema_version"),
            "generated_at": payload.get("generated_at"),
            "live_feedback": payload.get("live_feedback", {}),
        },
    }
    versions.append(entry)
    registry["active_version"] = version
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")


def _config_with_behavior_model_path(config: MarsConfig, behavior_path: Path) -> MarsConfig:
    raw = copy.deepcopy(config.raw)
    raw_search = raw.setdefault("search", {})
    raw_search["query_behavior_model_path"] = str(behavior_path)
    raw_search["query_behavior_model_required"] = True
    return replace(config, raw=raw)


def _catalog_products(config: MarsConfig) -> set[str]:
    for path in (
        config.paths.artifacts_dir / "search" / "product_meta.parquet",
        config.paths.processed_dir / "products.parquet",
    ):
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["product_id"])
        return set(frame["product_id"].astype(str))
    return set()


def _as_product_list(value: Any) -> list[str]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [item.strip() for item in value.split(",") if item.strip()]
        return _as_product_list(parsed)
    return []


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _empty_metrics(reason: str) -> dict[str, Any]:
    return {
        "source": reason,
        "evaluated_queries": 0,
        "unique_queries": 0,
        "mrr_at_10": 0.0,
        "ndcg_at_10": 0.0,
        "recall_at_10": 0.0,
        "latency_p95_ms": 0.0,
    }


def _read_behavior_model(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
