from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from collections import deque
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mars.config import load_config  # noqa: E402
from mars.config.settings import ensure_runtime_dirs  # noqa: E402
from mars.ct import CTMonitor, ModelRegistry  # noqa: E402
from mars.data.hm_pipeline import prepare_runtime_dataset  # noqa: E402
from mars.evaluation.runner import run_evaluation  # noqa: E402
from mars.recommendation.artifacts import (  # noqa: E402
    build_recommendation_artifacts,
    load_recommendation_artifacts,
    save_recommendation_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run periodic evaluation and CT monitoring.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument(
        "--logs",
        default=None,
        help="Live event log path. Defaults to logs/api_events.jsonl.",
    )
    parser.add_argument(
        "--source-key",
        default="api_events_jsonl",
        help="State key used to track new live logs separately from offline parquet counts.",
    )
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="Only report CT decisions without rebuilding recommendation artifacts.",
    )
    parser.add_argument(
        "--no-search-refresh",
        action="store_true",
        help="Disable automatic search behavior-model refresh on live-log CT triggers.",
    )
    parser.add_argument(
        "--no-search-promote",
        action="store_true",
        help="Build and validate search behavior-model candidates without promoting them.",
    )
    parser.add_argument(
        "--refresh-metrics-each-check",
        action="store_true",
        help=(
            "Run full offline evaluation before every CT check. "
            "By default the worker reads metrics.json."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config, mode=args.mode)
    ensure_runtime_dirs(config)
    monitor = CTMonitor(config)
    logs_path = Path(args.logs) if args.logs else config.paths.logs_dir / "api_events.jsonl"
    metrics_path = config.paths.artifacts_dir / "reports" / "metrics.json"
    log_count_cache: dict[str, Any] = {}

    while True:
        metrics = (
            run_evaluation(config).to_dict()
            if args.refresh_metrics_each_check or not metrics_path.exists()
            else _read_json(metrics_path)
        )
        metrics = _with_live_monitoring_metrics(config, metrics)
        current_log_count = _count_logs(logs_path, cache=log_count_cache)
        decision = monitor.evaluate(
            metrics,
            current_log_count=current_log_count,
            source_key=args.source_key,
            advance_log_count=False,
        )
        payload = {
            "checked_at": decision.snapshot.checked_at,
            "log_source": str(logs_path),
            "current_log_count": current_log_count,
            "new_logs": decision.snapshot.new_logs,
            "threshold": decision.snapshot.thresholds["new_logs_threshold"],
            "should_retrain": decision.should_retrain,
            "reasons": decision.reasons,
        }
        if _should_retrain(decision.reasons) and not args.no_retrain:
            payload["retrain_status"] = "started"
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            payload["retrain"] = _run_retrain(
                config,
                decision.reasons,
                current_log_count,
                logs_path=logs_path,
                source_key=args.source_key,
                config_path=args.config,
                search_refresh_enabled=not args.no_search_refresh,
                search_promote_enabled=not args.no_search_promote,
            )
            payload["retrain_status"] = "completed"
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        time.sleep(max(args.interval, 1))


def _should_retrain(reasons: list[str]) -> bool:
    return "new_logs_threshold_reached" in reasons


def _run_retrain(
    config,
    reasons: list[str],
    current_log_count: int,
    *,
    logs_path: Path,
    source_key: str,
    config_path: str,
    search_refresh_enabled: bool,
    search_promote_enabled: bool,
) -> dict[str, Any]:
    manifest_status = _ensure_processed_dataset(config)
    if _ct_lightweight_refresh_enabled(config):
        artifacts, artifact_refresh = _refresh_recommendation_artifacts_from_live_logs(
            config,
            logs_path=logs_path,
        )
    else:
        training_config = _ct_training_config(config)
        artifacts = build_recommendation_artifacts(config=training_config)
        artifact_refresh = {
            "mode": "full_rebuild",
            "live_events_used": 0,
            "updated_products": 0,
        }
    search_refresh = _run_search_refresh(
        config,
        config_path=config_path,
        logs_path=logs_path,
        enabled=search_refresh_enabled,
        promote_enabled=search_promote_enabled,
    )
    _clear_prediction_cache(config)
    metrics_path = config.paths.artifacts_dir / "reports" / "metrics.json"
    if _ct_skip_full_evaluation(config):
        metrics_payload = _read_json(metrics_path)
        if not metrics_payload:
            metrics_payload = run_evaluation(config).to_dict()
    else:
        metrics_payload = run_evaluation(config).to_dict()
    entry = ModelRegistry(config.paths.artifacts_dir / "registry" / "models.json").register(
        artifact_path=config.paths.artifacts_dir / "recsys",
        metrics_path=metrics_path,
        metadata={
            "mode": config.active_mode,
            "job": "ct_retrain",
            "reasons": reasons,
            "live_log_count": current_log_count,
            "recsys_version": artifacts.version,
            "artifact_refresh": artifact_refresh,
            "processed_manifest": manifest_status,
            "search_refresh": search_refresh,
            "metrics": metrics_payload,
            "ct_evaluation": (
                "reused_latest_metrics_snapshot"
                if _ct_skip_full_evaluation(config)
                else "full_run_evaluation"
            ),
        },
        activate=True,
    )
    consumed_log_count = _count_logs(logs_path)
    log_count_advanced = search_refresh.get("status") != "failed"
    if log_count_advanced:
        _mark_retrain_complete(
            config=config,
            metrics=metrics_payload,
            current_log_count=consumed_log_count,
            source_key=source_key,
            registered_version=entry.version,
        )
    return {
        "status": "completed",
        "registered_version": entry.version,
        "recsys_version": artifacts.version,
        "artifact_refresh": artifact_refresh,
        "processed_manifest": manifest_status,
        "search_refresh": search_refresh,
        "trigger_log_count": current_log_count,
        "consumed_log_count": consumed_log_count,
        "log_count_advanced": log_count_advanced,
        "artifact_path": str(config.paths.artifacts_dir / "recsys"),
        "metrics_path": str(metrics_path),
    }


def _run_search_refresh(
    config,
    *,
    config_path: str,
    logs_path: Path,
    enabled: bool,
    promote_enabled: bool,
) -> dict[str, Any]:
    raw_search = config.raw.get("search", {}) if isinstance(config.raw, dict) else {}
    online = raw_search.get("online_learning", {}) if isinstance(raw_search, dict) else {}
    online = online if isinstance(online, dict) else {}
    if not enabled:
        return {"status": "skipped", "reason": "disabled_by_worker_flag"}
    if not bool(online.get("enabled", False)):
        return {"status": "skipped", "reason": "search_online_learning_disabled"}
    if not bool(online.get("auto_refresh", True)):
        return {"status": "skipped", "reason": "search_auto_refresh_disabled"}

    command = [
        sys.executable,
        str(ROOT / "scripts" / "artifacts" / "refresh_search_behavior_model.py"),
        "--config",
        str(config_path),
        "--mode",
        str(config.active_mode),
        "--logs",
        str(logs_path),
        "--max-eval-queries",
        str(int(online.get("max_eval_queries", 5000) or 5000)),
        "--max-mrr-drop",
        str(float(online.get("max_mrr_drop", 0.002) or 0.002)),
        "--max-ndcg-drop",
        str(float(online.get("max_ndcg_drop", 0.002) or 0.002)),
        "--max-latency-p95-ms",
        str(float(online.get("max_latency_p95_ms", 200.0) or 200.0)),
    ]
    if promote_enabled and bool(online.get("promote_on_pass", True)):
        command.append("--promote")
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    payload: dict[str, Any] = {
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": int(result.returncode),
        "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "promote_requested": "--promote" in command,
        "report_path": str(config.paths.artifacts_dir / "reports" / "search_behavior_refresh.json"),
    }
    if result.stdout:
        payload["stdout_tail"] = result.stdout[-2000:]
    if result.stderr:
        payload["stderr_tail"] = result.stderr[-2000:]
    return payload


def _refresh_recommendation_artifacts_from_live_logs(
    config,
    *,
    logs_path: Path,
) -> tuple[Any, dict[str, Any]]:
    artifacts = load_recommendation_artifacts(config=config)
    recommendation_raw = (
        config.raw.get("recommendation", {}) if isinstance(config.raw, dict) else {}
    )
    ct_training = recommendation_raw.get("ct_training", {})
    ct_training = ct_training if isinstance(ct_training, dict) else {}
    max_events = int(ct_training.get("live_refresh_max_events", 50_000) or 50_000)
    product_scores = _live_product_scores(logs_path, max_events=max_events)
    product_set = set(artifacts.item_index)
    updates = 0
    for product in artifacts.products:
        product_id = str(product.get("product_id", ""))
        if product_id not in product_scores:
            continue
        base = float(product.get("popularity_prior", 0.0) or 0.0)
        live_score = float(product_scores[product_id])
        product["live_popularity_prior"] = live_score
        product["popularity_prior"] = base + live_score
        updates += 1

    artifacts.version = datetime.now(UTC).strftime("recsys-live-%Y%m%d%H%M%S")
    artifacts.popularity_order = _rank_product_ids(
        artifacts.products,
        primary="popularity_prior",
        secondary="live_popularity_prior",
    )
    artifacts.trending_order = _rank_product_ids(
        artifacts.products,
        primary="is_new",
        secondary="popularity_prior",
    )
    artifacts.training_events_source = f"live_feedback_refresh:{logs_path}"
    artifacts.training_event_count = int(artifacts.training_event_count or 0) + sum(
        1 for product_id in product_scores if product_id in product_set
    )
    save_recommendation_artifacts(artifacts, config=config)
    return artifacts, {
        "mode": "lightweight_live_feedback_refresh",
        "logs_path": str(logs_path),
        "max_events": max_events,
        "live_events_used": len(product_scores),
        "updated_products": updates,
    }


def _live_product_scores(logs_path: Path, *, max_events: int) -> dict[str, float]:
    if not logs_path.exists():
        return {}
    lines: deque[str] = deque(maxlen=max(max_events, 1))
    with logs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                lines.append(line)
    weights = {
        "view": 1.0,
        "cart": 3.0,
        "purchase": 5.0,
    }
    scores: dict[str, float] = {}
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        product_id = str(event.get("product_id") or "")
        if not product_id:
            continue
        event_type = str(event.get("event_type") or "").lower()
        weight = weights.get(event_type)
        if weight is None:
            metadata = event.get("metadata", {})
            if isinstance(metadata, dict) and metadata.get("event_role") == "exposure":
                weight = 0.05
            else:
                continue
        scores[product_id] = scores.get(product_id, 0.0) + weight
    return scores


def _rank_product_ids(
    products: list[dict[str, Any]],
    *,
    primary: str,
    secondary: str,
) -> list[str]:
    return [
        str(product.get("product_id"))
        for product in sorted(
            products,
            key=lambda product: (
                float(product.get(primary, 0.0) or 0.0),
                float(product.get(secondary, 0.0) or 0.0),
                str(product.get("product_id", "")),
            ),
            reverse=True,
        )
    ]


def _ct_lightweight_refresh_enabled(config) -> bool:
    recommendation_raw = (
        config.raw.get("recommendation", {}) if isinstance(config.raw, dict) else {}
    )
    ct_training = recommendation_raw.get("ct_training", {})
    if not isinstance(ct_training, dict):
        return False
    return bool(ct_training.get("lightweight_live_refresh", False))


def _ct_training_config(config):
    raw = copy.deepcopy(config.raw)
    recommendation_raw = raw.setdefault("recommendation", {})
    ct_training = recommendation_raw.get("ct_training", {})
    if isinstance(ct_training, dict):
        recommendation_raw["two_tower_max_positive_samples"] = int(
            ct_training.get(
                "two_tower_max_positive_samples",
                recommendation_raw.get("two_tower_max_positive_samples", 4_000),
            )
            or 4_000
        )
        recommendation_raw["ranker_max_samples"] = int(
            ct_training.get(
                "ranker_max_samples",
                recommendation_raw.get("ranker_max_samples", 20_000),
            )
            or 20_000
        )
    return replace(config, raw=raw)


def _ct_skip_full_evaluation(config) -> bool:
    recommendation_raw = (
        config.raw.get("recommendation", {}) if isinstance(config.raw, dict) else {}
    )
    ct_training = recommendation_raw.get("ct_training", {})
    if not isinstance(ct_training, dict):
        return False
    return bool(ct_training.get("skip_full_evaluation", False))


def _mark_retrain_complete(
    *,
    config,
    metrics: dict[str, Any],
    current_log_count: int,
    source_key: str,
    registered_version: str,
) -> None:
    state_path = config.paths.artifacts_dir / "registry" / "ct_state.json"
    state = _read_json(state_path)
    checked_at = datetime.now(UTC).isoformat()
    ctr_value = _extract(metrics, ("ab_test", "buckets", "treatment", "ctr"), 0.0)
    cvr_value = _extract(metrics, ("ab_test", "buckets", "treatment", "cvr"), 0.0)
    hit_rate = _extract(metrics, ("recommendation", "hit_rate_at_50"), 0.0)
    snapshot = {
        "checked_at": checked_at,
        "ctr": float(ctr_value),
        "cvr": float(cvr_value),
        "hit_rate": float(hit_rate),
        "new_logs": 0,
        "thresholds": {
            "ctr_threshold": config.monitoring.ctr_threshold,
            "hitrate_threshold": config.monitoring.hitrate_threshold,
            "new_logs_threshold": config.monitoring.new_logs_threshold,
            "ctr_min_logs": config.monitoring.ctr_min_logs,
        },
    }
    log_sources = state.get("log_sources", {})
    if not isinstance(log_sources, dict):
        log_sources = {}
    log_sources[source_key] = {
        "last_checked_at": checked_at,
        "last_log_count": int(current_log_count),
        "current_log_count": int(current_log_count),
        "pending_new_logs": 0,
        "last_retrain_version": registered_version,
    }
    state.update(
        {
            "last_checked_at": checked_at,
            "last_log_count": int(current_log_count),
            "last_log_source": source_key,
            "last_retrain_version": registered_version,
            "log_sources": log_sources,
            "last_decision": {
                "should_retrain": False,
                "reasons": [],
                "snapshot": snapshot,
            },
        }
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _ensure_processed_dataset(config) -> dict[str, str | int | bool]:
    manifest_path = config.paths.processed_dir / "manifest.json"
    status: dict[str, str | int | bool] = {
        "path": str(manifest_path),
        "rebuilt": False,
    }
    manifest = _read_json(manifest_path)
    rows = manifest.get("row_counts", {}) if isinstance(manifest, dict) else {}
    mode_matches = manifest.get("mode") == config.active_mode
    products_ok = int(rows.get("products", 0) or 0) >= int(config.mode.products)
    users_ok = int(rows.get("users", 0) or 0) >= int(config.mode.users)
    events_ok = int(rows.get("events", 0) or 0) >= int(config.mode.events)
    if not (mode_matches and products_ok and users_ok and events_ok):
        manifest = prepare_runtime_dataset(config, clean_processed=True).payload
        rows = manifest.get("row_counts", {})
        status["rebuilt"] = True
    status.update(
        {
            "mode": str(manifest.get("mode")),
            "products": int(rows.get("products", 0) or 0),
            "users": int(rows.get("users", 0) or 0),
            "events": int(rows.get("events", 0) or 0),
        }
    )
    return status


def _clear_prediction_cache(config) -> None:
    reports_dir = config.paths.artifacts_dir / "reports"
    for name in ("recommendation_predictions.json",):
        path = reports_dir / name
        if path.exists():
            path.unlink()


def _count_logs(path: Path, cache: dict[str, Any] | None = None) -> int:
    if not path.exists():
        if cache is not None:
            cache.clear()
        return 0
    if path.suffix == ".jsonl":
        return _count_jsonl_logs(path, cache)
    if path.suffix == ".parquet":
        import pandas as pd

        return int(len(pd.read_parquet(path)))
    if path.suffix == ".csv":
        import pandas as pd

        return int(len(pd.read_csv(path)))
    return 0


def _count_jsonl_logs(path: Path, cache: dict[str, Any] | None = None) -> int:
    stat = path.stat()
    current_size = int(stat.st_size)
    cache = cache if cache is not None else {}
    cached_path = cache.get("path")
    cached_size = int(cache.get("size", 0) or 0)
    cached_count = int(cache.get("count", 0) or 0)
    try:
        if cached_path == str(path) and 0 <= cached_size <= current_size:
            if cached_size == current_size:
                return cached_count
            with path.open("rb") as handle:
                handle.seek(cached_size)
                new_count = _count_newlines_from_stream(handle)
            total = cached_count + new_count
        else:
            with path.open("rb") as handle:
                total = _count_newlines_from_stream(handle)
    except OSError:
        if cached_path == str(path) and cached_count > 0:
            return cached_count
        raise
    cache.update({"path": str(path), "size": current_size, "count": int(total)})
    return int(total)


def _count_newlines_from_stream(handle: Any, chunk_size: int = 1024 * 1024) -> int:
    total = 0
    while True:
        chunk = handle.read(chunk_size)
        if not chunk:
            break
        total += chunk.count(b"\n")
    return total


def _with_live_monitoring_metrics(config, metrics: dict[str, Any]) -> dict[str, Any]:
    live = _read_live_surface_stats(config)
    if not live:
        return metrics
    enriched = copy.deepcopy(metrics)
    recommendation = live.get("recommendation", {})
    search = live.get("search", {})
    chosen = recommendation if int(recommendation.get("impressions", 0) or 0) > 0 else search
    if int(chosen.get("impressions", 0) or 0) <= 0:
        return enriched
    monitoring = enriched.setdefault("monitoring", {})
    monitoring.update(
        {
            "ctr": float(chosen.get("ctr", 0.0) or 0.0),
            "cvr": float(chosen.get("cvr", 0.0) or 0.0),
            "ctr_source": str(chosen.get("source", "live_surface")),
            "impressions": int(chosen.get("impressions", 0) or 0),
            "clicks": int(chosen.get("clicks", 0) or 0),
            "conversions": int(chosen.get("conversions", 0) or 0),
        }
    )
    return enriched


def _read_live_surface_stats(config) -> dict[str, dict[str, float | int | str]]:
    try:
        import redis

        client = redis.from_url(config.redis_url, socket_connect_timeout=0.2)
        result: dict[str, dict[str, float | int | str]] = {}
        for surface in ("recommendation", "search"):
            raw = client.hgetall(f"live:surface:{surface}")
            if not raw:
                continue
            decoded = {
                (key.decode() if isinstance(key, bytes) else str(key)): int(value)
                for key, value in raw.items()
            }
            impressions = int(decoded.get("impressions", 0))
            clicks = int(decoded.get("clicks", 0))
            conversions = int(decoded.get("conversions", 0))
            result[surface] = {
                "source": f"live:surface:{surface}",
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "ctr": clicks / impressions if impressions else 0.0,
                "cvr": conversions / impressions if impressions else 0.0,
            }
        return result
    except Exception:
        return {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _extract(payload: dict[str, Any], path: tuple[str, ...], default: Any) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


if __name__ == "__main__":
    raise SystemExit(main())
