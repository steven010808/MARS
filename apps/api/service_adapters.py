from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from apps.api.schemas import (
    ABAssignRequest,
    ABReportResponse,
    EventRequest,
    EventResponse,
    PipelineLatency,
    RecommendationItem,
    RecommendationResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from mars.config.settings import MarsConfig


LIVE_RECENT_EVENT_LIMIT = 5000


def _load_class(module_path: str, class_name: str) -> type[Any] | None:
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception:
        return None


def _instantiate(service_class: type[Any] | None, config: MarsConfig) -> Any | None:
    if service_class is None:
        return None
    for args in ((config,), ()):
        try:
            return service_class(*args)
        except Exception:
            continue
    return None


def _call_first(service: Any, method_names: tuple[str, ...], *args: Any, **kwargs: Any) -> Any:
    if service is None:
        raise RuntimeError("service unavailable")
    for name in method_names:
        method = getattr(service, name, None)
        if method is None:
            continue
        if kwargs:
            compatible_kwargs = _compatible_kwargs(method, kwargs)
            if compatible_kwargs != kwargs:
                for call_args in (args, ()):
                    try:
                        return method(*call_args, **compatible_kwargs)
                    except TypeError:
                        continue
        for call_args, call_kwargs in ((args, kwargs), (args, {}), ((), kwargs), ((), {})):
            try:
                return method(*call_args, **call_kwargs)
            except TypeError:
                continue
    raise RuntimeError("compatible service method unavailable")


def _compatible_kwargs(method: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return kwargs
    parameters = signature.parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return kwargs
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    return {key: value for key, value in kwargs.items() if key in accepted}


class ApiRuntime:
    def __init__(self, config: MarsConfig) -> None:
        self.config = config
        self.redis_client = self._connect_redis()
        self.search_service = _instantiate(
            _load_class("mars.search.service", "SearchService"),
            config,
        )
        self.recommendation_service = _instantiate(
            _load_class("mars.recommendation.service", "RecommendationService"),
            config,
        )
        self._search_reload_lock = RLock()
        self._search_reload_error: str | None = None
        self._served_search_version = self._active_search_version()
        self._recommendation_reload_lock = RLock()
        self._recommendation_reload_error: str | None = None
        self._served_recommendation_version = self._active_recommendation_version()
        self._event_log_count_cache: tuple[int, int, int] | None = None
        self._event_log_lock = RLock()
        self._attach_recommendation_session_store()

    def _attach_recommendation_session_store(self) -> None:
        self._attach_session_store_to(self.recommendation_service)

    def _attach_session_store_to(self, service: Any | None) -> None:
        if service is None or self.redis_client is None:
            return
        try:
            from mars.recommendation.session import RedisSessionStore

            service.session_store = RedisSessionStore(
                self.redis_client,
                recent_n=self.config.recommendation.session_recent_n,
            )
        except Exception:
            return

    def _connect_redis(self) -> Any | None:
        try:
            import redis

            client = redis.from_url(self.config.redis_url, socket_connect_timeout=0.2)
            client.ping()
            return client
        except Exception:
            return None

    def services_status(self) -> dict[str, str]:
        return {
            "search": "ready" if self.search_service else "fallback",
            "search_model": self._served_search_version,
            "recommendation": "ready" if self.recommendation_service else "fallback",
            "recommendation_model": self._served_recommendation_version,
            "redis": "ready" if self.redis_client else "unavailable",
        }

    def artifacts_status(self) -> dict[str, bool]:
        paths = self.config.paths
        manifest = self._read_json(paths.processed_dir / "manifest.json")
        row_counts = manifest.get("row_counts", {}) if isinstance(manifest, dict) else {}
        expected_products = int(row_counts.get("products", 0) or self.config.mode.products)
        expected_users = int(row_counts.get("users", 0) or self.config.mode.users)
        expected_events = int(row_counts.get("events", 0) or self.config.mode.events)
        processed_ready = (
            (paths.processed_dir / "manifest.json").exists()
            and (paths.processed_dir / "products.parquet").exists()
            and (paths.processed_dir / "users.parquet").exists()
            and (paths.processed_dir / "events.parquet").exists()
            and int(row_counts.get("products", 0) or 0) >= self.config.mode.products
            and int(row_counts.get("users", 0) or 0) >= self.config.mode.users
            and int(row_counts.get("events", 0) or 0) >= self.config.mode.events
        )
        search_manifest = self._read_json(paths.artifacts_dir / "search" / "index_manifest.json")
        recsys_index = self._read_json(paths.artifacts_dir / "recsys" / "items_index.json")
        metrics = self._read_json(paths.artifacts_dir / "reports" / "metrics.json")
        metrics_system = metrics.get("system", {}) if isinstance(metrics, dict) else {}
        return {
            "processed_data": processed_ready,
            "manifest": (paths.processed_dir / "manifest.json").exists(),
            "search_index": (
                (paths.artifacts_dir / "search" / "index_manifest.json").exists()
                and int(search_manifest.get("product_count", -1)) == expected_products
            ),
            "recsys_models": (
                (paths.artifacts_dir / "recsys" / "recommendation_artifacts.json.gz").exists()
                and int(recsys_index.get("count", -1)) == expected_products
            ),
            "reports": (
                (paths.artifacts_dir / "reports" / "metrics.json").exists()
                and int(metrics_system.get("products", -1)) == expected_products
                and int(metrics_system.get("users", -1)) == expected_users
                and int(metrics_system.get("events", -1)) == expected_events
            ),
            "registry": (paths.artifacts_dir / "registry").exists(),
        }

    def search(self, request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        search_id = f"Q{uuid4().hex[:16]}"
        session_id = request.session_id or f"S-search-{uuid4().hex[:8]}"
        image_reference = request.image_url or request.image_path
        try:
            self._maybe_reload_search_service()
            raw = _call_first(
                self.search_service,
                ("search", "run", "__call__"),
                request,
                query=request.query,
                image_base64=request.image_base64,
                image_url=image_reference,
                image_path=image_reference,
                search_type=request.search_type.value,
                top_k=request.top_k,
                filters=request.filters.model_dump(),
                hybrid_weights=request.hybrid_weights.model_dump(),
            )
            response = SearchResponse.model_validate(raw)
            response.debug = {
                **response.debug,
                "search_id": search_id,
                "session_id": session_id,
            }
            self._record_search_exposure(request, response, search_id, session_id)
            return response
        except Exception as exc:
            results = self._fallback_search_results(request, self._fallback_products())
            response = SearchResponse(
                search_type=request.search_type,
                results=results,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                total_count=len(results),
                debug={
                    "mode": "fallback",
                    "reason": exc.__class__.__name__,
                    "encoder_type": self.config.search.encoder_type,
                    "search_id": search_id,
                    "session_id": session_id,
                },
            )
            self._record_search_exposure(request, response, search_id, session_id)
            return response

    def _record_search_exposure(
        self,
        request: SearchRequest,
        response: SearchResponse,
        search_id: str,
        session_id: str,
    ) -> None:
        raw_search = self.config.raw.get("search", {}) if isinstance(self.config.raw, dict) else {}
        if not bool(raw_search.get("log_search_exposures", True)):
            return
        result_ids = [str(item.product_id) for item in response.results]
        rank_map = {product_id: rank for rank, product_id in enumerate(result_ids, start=1)}
        timestamp = datetime.now(UTC).isoformat()
        payload = {
            "event_id": f"E{uuid4().hex[:16]}",
            "user_id": request.user_id or "anonymous",
            "session_id": session_id,
            "event_type": "search",
            "product_id": None,
            "query": request.query,
            "timestamp": timestamp,
            "source": "api_search",
            "category": request.filters.category,
            "metadata": {
                "source_surface": "search",
                "surface": "search",
                "event_role": "exposure",
                "search_id": search_id,
                "search_type": str(request.search_type),
                "top_k": int(request.top_k),
                "result_product_ids": result_ids,
                "rank_map": rank_map,
                "scores": {str(item.product_id): float(item.score) for item in response.results},
                "filters": request.filters.model_dump(exclude_none=True),
                "model_version": self._active_search_version(),
            },
        }
        with self._event_log_lock:
            self._rotate_event_log_if_needed()
            self._update_redis_features(payload)
            self._append_event_log(payload)

    def recommend(
        self,
        user_id: str,
        top_n: int,
        session_id: str | None,
        experiment_key: str | None = None,
    ) -> RecommendationResponse:
        started = time.perf_counter()
        ab_bucket = _deterministic_bucket(experiment_key, user_id) if experiment_key else None
        strategy = "control" if ab_bucket == "control" else "treatment"
        request_id = (
            f"{experiment_key}:{ab_bucket}:{session_id or 'no-session'}"
            if experiment_key
            else f"direct:{session_id or 'no-session'}"
        )
        try:
            self._maybe_reload_recommendation_service()
            raw = _call_first(
                self.recommendation_service,
                ("recommend", "run", "__call__"),
                user_id=user_id,
                top_n=top_n,
                session_id=session_id,
                request_id=request_id,
                strategy=strategy,
            )
            response = RecommendationResponse.model_validate(raw)
            response.session_context.setdefault("session_id", session_id)
            if experiment_key:
                response.session_context["ab_experiment_key"] = experiment_key
                response.session_context["ab_bucket"] = ab_bucket
                response.session_context["recommendation_strategy"] = strategy
            return response
        except Exception:
            candidate_start = time.perf_counter()
            recent_products = self._redis_lrange(f"user:{user_id}:recent_products", 0, 9)
            recent_categories = self._redis_lrange(f"user:{user_id}:recent_categories", 0, 9)
            products = self._rank_fallback_products(
                self._fallback_products(),
                user_id,
                recent_categories,
            )
            candidate_ms = round((time.perf_counter() - candidate_start) * 1000, 3)
            ranking_start = time.perf_counter()
            ranked = products[: max(top_n * 3, top_n)]
            ranking_ms = round((time.perf_counter() - ranking_start) * 1000, 3)
            rerank_start = time.perf_counter()
            recommendations = self._fallback_recommendation_items(ranked, top_n)
            reranking_ms = round((time.perf_counter() - rerank_start) * 1000, 3)
            return RecommendationResponse(
                user_id=user_id,
                recommendations=recommendations,
                pipeline_latency=PipelineLatency(
                    candidate_ms=candidate_ms,
                    ranking_ms=ranking_ms,
                    reranking_ms=reranking_ms,
                    total_ms=round((time.perf_counter() - started) * 1000, 3),
                ),
                session_context={
                    "session_id": session_id,
                    "recent_products": recent_products,
                    "recent_categories": recent_categories,
                    "recent_clicks": recent_products,
                    "session_interest": recent_categories[0] if recent_categories else "",
                    "source": "redis" if self.redis_client else "fallback_empty",
                    "ab_experiment_key": experiment_key,
                    "ab_bucket": ab_bucket,
                    "recommendation_strategy": strategy,
                },
            )

    def record_event(self, request: EventRequest) -> EventResponse:
        event_id = f"E{uuid4().hex[:16]}"
        session_id = request.session_id or (
            f"S{datetime.now(UTC).strftime('%Y%m%d')}-{uuid4().hex[:8]}"
        )
        timestamp = request.timestamp or datetime.now(UTC).isoformat()
        payload = request.model_dump(mode="json")
        payload.update({"event_id": event_id, "session_id": session_id, "timestamp": timestamp})
        metadata = payload.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
            payload["metadata"] = metadata
        if not payload.get("arm"):
            arm = metadata.get("arm")
            if arm:
                payload["arm"] = arm
        if not payload.get("category"):
            category = metadata.get("category") or payload.get("category")
            if category:
                payload["category"] = category

        with self._event_log_lock:
            self._rotate_event_log_if_needed()
            redis_updated = self._update_redis_features(payload)
            if self.recommendation_service is not None:
                try:
                    self.recommendation_service.update_event(payload)
                    redis_updated = redis_updated or bool(self.redis_client)
                except Exception:
                    pass
            log_path = self._append_event_log(payload)

        return EventResponse(
            accepted=True,
            event_id=event_id,
            session_id=session_id,
            event_type=request.event_type,
            redis_updated=redis_updated,
            durable_log=str(log_path),
        )

    def reset_live_run(self, reason: str = "manual_dashboard_reset") -> dict[str, Any]:
        return self._rotate_event_log(reason=reason, reset_live_state=True, force=True)

    def prepare_retrain_state(self) -> dict[str, Any]:
        """Seed live events near the CT retrain trigger without crossing it."""

        log_path = self.config.paths.logs_dir / "api_events.jsonl"
        state_path = self.config.paths.artifacts_dir / "registry" / "ct_state.json"
        source_key = "api_events_jsonl"
        threshold = max(int(self.config.monitoring.new_logs_threshold or 1), 1)
        now = datetime.now(UTC)

        with self._event_log_lock:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            state = self._read_json(state_path)
            log_sources = state.get("log_sources", {})
            if not isinstance(log_sources, dict):
                log_sources = {}
            source_state = log_sources.get(source_key, {})
            if not isinstance(source_state, dict):
                source_state = {}
            baseline = int(source_state.get("last_log_count", state.get("last_log_count", 0)) or 0)
            before_count = self._count_jsonl_lines_uncached(log_path)
            pending_before = max(before_count - baseline, 0)
            holdback = 0 if threshold <= 1 else min(max(threshold // 100, 1), threshold - 1)
            target_pending = max(threshold - holdback, 1)
            append_count = max(target_pending - pending_before, 0)
            generated_events = self._retrain_seed_events(
                append_count,
                start_index=before_count,
                started_at=now,
            )
            if generated_events:
                with log_path.open("a", encoding="utf-8") as handle:
                    for event in generated_events:
                        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            after_count = before_count + append_count
            pending_after = max(after_count - baseline, 0)
            self._event_log_count_cache = None
            self._sync_retrain_seed_redis(generated_events, after_count=after_count)
            self._write_retrain_seed_state(
                state_path=state_path,
                state=state,
                source_key=source_key,
                baseline=baseline,
                current_count=after_count,
                pending_count=pending_after,
                threshold=threshold,
                added_count=append_count,
                now=now,
            )

        return {
            "prepared": pending_after >= target_pending,
            "ready_to_retrain": pending_after >= threshold,
            "reason": (
                "new_logs_threshold_reached"
                if pending_after >= threshold
                else "near_new_logs_threshold"
            ),
            "added_events": append_count,
            "previous_log_count": before_count,
            "current_log_count": after_count,
            "last_log_count": baseline,
            "pending_new_logs": pending_after,
            "threshold": threshold,
            "target_pending_new_logs": target_pending,
            "remaining_to_threshold": max(threshold - pending_after, 0),
            "log_path": str(log_path),
            "worker_note": (
                "Worker will run CT retraining on its next check."
                if pending_after >= threshold
                else "A few more live events are still required."
            ),
        }

    def _retrain_seed_events(
        self,
        count: int,
        *,
        start_index: int,
        started_at: datetime,
    ) -> list[dict[str, Any]]:
        if count <= 0:
            return []
        events: list[dict[str, Any]] = []
        patterns = (
            ("search", "search", "user_action", None, "black socks"),
            ("view", "search", "user_action", "P0000148", "black socks"),
            ("cart", "search", "user_action", "P0000148", "black socks"),
            ("purchase", "search", "user_action", "P0000148", "black socks"),
            ("view", "recommendation", "user_action", "P0000215", None),
            ("cart", "recommendation", "user_action", "P0000215", None),
            ("purchase", "recommendation", "user_action", "P0000215", None),
            ("search", "search", "user_action", None, "red trousers"),
            ("view", "search", "user_action", "P0000320", "red trousers"),
            ("view", "recommendation", "exposure", "P0000450", None),
        )
        for offset in range(count):
            event_type, surface, role, product_id, query = patterns[offset % len(patterns)]
            event_index = start_index + offset + 1
            event_id = f"ECTSEED{event_index:010d}"
            session_id = f"S-retrain-seed-{event_index // len(patterns):06d}"
            timestamp = started_at.isoformat()
            metadata = {
                "source_surface": surface,
                "surface": surface,
                "event_role": role,
                "seed_source": "dashboard_retrain_state",
                "experiment_key": "mars_default",
                "ab_bucket": "treatment" if event_index % 2 else "control",
                "rank": (offset % 10) + 1,
            }
            payload: dict[str, Any] = {
                "event_id": event_id,
                "user_id": f"U-seed-{event_index % 50:03d}",
                "event_type": event_type,
                "product_id": product_id,
                "session_id": session_id,
                "query": query,
                "arm": metadata["ab_bucket"],
                "timestamp": timestamp,
                "source": "dashboard_retrain_seed",
                "category": "operations",
                "metadata": metadata,
            }
            events.append(payload)
        return events

    def _sync_retrain_seed_redis(
        self,
        events: list[dict[str, Any]],
        *,
        after_count: int,
    ) -> None:
        if self.redis_client is None:
            return
        try:
            pipe = self.redis_client.pipeline()
            pipe.set("system:logged_events", int(after_count))
            surface_counts: dict[str, dict[str, int]] = {}
            for event in events:
                metadata = event.get("metadata", {})
                metadata = metadata if isinstance(metadata, dict) else {}
                surface = str(metadata.get("source_surface") or metadata.get("surface") or "")
                if surface not in {"search", "recommendation"}:
                    continue
                counts = surface_counts.setdefault(
                    surface,
                    {"impressions": 0, "clicks": 0, "carts": 0, "conversions": 0},
                )
                event_type = str(event.get("event_type") or "")
                event_role = str(metadata.get("event_role") or "")
                if event_role == "exposure" or event_type == "search":
                    counts["impressions"] += 1
                if event_type == "view":
                    counts["clicks"] += 1
                elif event_type == "cart":
                    counts["carts"] += 1
                elif event_type == "purchase":
                    counts["conversions"] += 1
            for surface, counts in surface_counts.items():
                key = f"live:surface:{surface}"
                for field, value in counts.items():
                    if value:
                        pipe.hincrby(key, field, value)
            for event in events[-80:]:
                pipe.lpush("live:recent_events", json.dumps(event, ensure_ascii=False))
            pipe.ltrim("live:recent_events", 0, LIVE_RECENT_EVENT_LIMIT - 1)
            pipe.execute()
        except Exception:
            return

    def _write_retrain_seed_state(
        self,
        *,
        state_path: Path,
        state: dict[str, Any],
        source_key: str,
        baseline: int,
        current_count: int,
        pending_count: int,
        threshold: int,
        added_count: int,
        now: datetime,
    ) -> None:
        checked_at = now.isoformat()
        log_sources = state.get("log_sources", {})
        if not isinstance(log_sources, dict):
            log_sources = {}
        log_sources[source_key] = {
            "last_checked_at": checked_at,
            "last_log_count": baseline,
            "current_log_count": current_count,
            "pending_new_logs": pending_count,
            "seeded_at": checked_at,
            "seeded_events": added_count,
        }
        reasons = ["new_logs_threshold_reached"] if pending_count >= threshold else []
        snapshot = {
            "checked_at": checked_at,
            "ctr": 0.10,
            "cvr": 0.02,
            "hit_rate": 0.46,
            "new_logs": pending_count,
            "thresholds": {
                "ctr_threshold": self.config.monitoring.ctr_threshold,
                "hitrate_threshold": self.config.monitoring.hitrate_threshold,
                "new_logs_threshold": threshold,
                "ctr_min_logs": self.config.monitoring.ctr_min_logs,
            },
        }
        seed_runs = state.get("retrain_seed_runs", [])
        if not isinstance(seed_runs, list):
            seed_runs = []
        seed_runs.append(
            {
                "seeded_at": checked_at,
                "added_events": added_count,
                "baseline": baseline,
                "current_log_count": current_count,
                "pending_new_logs": pending_count,
                "threshold": threshold,
            }
        )
        state.update(
            {
                "last_checked_at": checked_at,
                "last_log_count": baseline,
                "last_log_source": source_key,
                "log_sources": log_sources,
                "last_decision": {
                    "should_retrain": bool(reasons),
                    "reasons": reasons,
                    "snapshot": snapshot,
                },
                "retrain_seed_runs": seed_runs[-20:],
            }
        )
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    def _append_event_log(self, payload: dict[str, Any]) -> Path:
        log_path = self.config.paths.logs_dir / "api_events.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._event_log_lock:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._event_log_count_cache = None
        return log_path

    def _rotate_event_log_if_needed(self) -> dict[str, Any] | None:
        log_path = self.config.paths.logs_dir / "api_events.jsonl"
        try:
            if not log_path.exists():
                return None
            stat = log_path.stat()
        except OSError:
            return None
        if stat.st_size <= 0:
            return None
        logging_raw = (
            self.config.raw.get("logging", {}) if isinstance(self.config.raw, dict) else {}
        )
        live_raw = logging_raw.get("live_events", {}) if isinstance(logging_raw, dict) else {}
        max_mb = float(live_raw.get("max_file_mb", logging_raw.get("live_event_log_max_mb", 100)))
        rotate_daily = bool(
            live_raw.get("rotate_daily", logging_raw.get("live_event_log_rotate_daily", True))
        )
        max_bytes = int(max(max_mb, 1.0) * 1024 * 1024)
        reasons: list[str] = []
        if stat.st_size >= max_bytes:
            reasons.append("size_limit")
        if rotate_daily:
            modified_date = datetime.fromtimestamp(stat.st_mtime, UTC).date()
            if modified_date != datetime.now(UTC).date():
                reasons.append("daily_boundary")
        if not reasons:
            return None
        return self._rotate_event_log(reason="+".join(reasons), reset_live_state=True, force=False)

    def _rotate_event_log(
        self,
        *,
        reason: str,
        reset_live_state: bool,
        force: bool,
    ) -> dict[str, Any]:
        log_path = self.config.paths.logs_dir / "api_events.jsonl"
        archive_dir = self.config.paths.logs_dir / "archive"
        archive_path: Path | None = None
        before_bytes = 0
        before_lines = 0
        rotated = False
        now = datetime.now(UTC)
        with self._event_log_lock:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            archive_dir.mkdir(parents=True, exist_ok=True)
            if log_path.exists():
                before_bytes = int(log_path.stat().st_size)
            if log_path.exists() and (force or before_bytes > 0):
                before_lines = self._count_jsonl_lines_uncached(log_path)
                if before_bytes > 0:
                    archive_name = (
                        f"api_events_{now.strftime('%Y%m%d_%H%M%S')}_{_safe_slug(reason)}.jsonl"
                    )
                    archive_path = archive_dir / archive_name
                    suffix = 1
                    while archive_path.exists():
                        archive_path = archive_dir / (
                            f"api_events_{now.strftime('%Y%m%d_%H%M%S')}_{_safe_slug(reason)}_{suffix}.jsonl"
                        )
                        suffix += 1
                    log_path.replace(archive_path)
                    rotated = True
            log_path.touch(exist_ok=True)
            self._event_log_count_cache = None
            reset_keys = self._reset_live_redis_state() if reset_live_state else []
            self._realign_ct_state_after_log_reset(
                reason=reason,
                archive_path=archive_path,
                previous_bytes=before_bytes,
                previous_lines=before_lines,
                reset_keys=reset_keys,
            )
        return {
            "rotated": rotated,
            "reason": reason,
            "archive_path": str(archive_path) if archive_path else None,
            "new_log_path": str(log_path),
            "previous_bytes": before_bytes,
            "previous_lines": before_lines,
            "reset_redis_keys": reset_keys,
            "reset_redis_key_count": len(reset_keys),
        }

    @staticmethod
    def _count_jsonl_lines_uncached(path: Path) -> int:
        if not path.exists():
            return 0
        count = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                count += chunk.count(b"\n")
        return count

    def _reset_live_redis_state(self) -> list[str]:
        if self.redis_client is None:
            return []
        keys = [
            "live:recent_events",
            "live:surface:search",
            "live:surface:recommendation",
            "system:logged_events",
        ]
        try:
            keys.extend(
                key.decode("utf-8") if isinstance(key, bytes) else str(key)
                for key in self.redis_client.scan_iter("ab:*")
            )
            unique_keys = sorted(set(keys))
            if unique_keys:
                self.redis_client.delete(*unique_keys)
            return unique_keys
        except Exception:
            return []

    def _realign_ct_state_after_log_reset(
        self,
        *,
        reason: str,
        archive_path: Path | None,
        previous_bytes: int,
        previous_lines: int,
        reset_keys: list[str],
    ) -> None:
        state_path = self.config.paths.artifacts_dir / "registry" / "ct_state.json"
        state = self._read_json(state_path)
        now = datetime.now(UTC).isoformat()
        source_key = "api_events_jsonl"
        log_sources = state.get("log_sources", {})
        if not isinstance(log_sources, dict):
            log_sources = {}
        log_sources[source_key] = {
            "last_checked_at": now,
            "last_log_count": 0,
            "current_log_count": 0,
            "pending_new_logs": 0,
            "reset_reason": reason,
            "archive_path": str(archive_path) if archive_path else None,
        }
        history = state.get("log_rotations", [])
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "rotated_at": now,
                "reason": reason,
                "archive_path": str(archive_path) if archive_path else None,
                "previous_bytes": previous_bytes,
                "previous_lines": previous_lines,
                "reset_redis_key_count": len(reset_keys),
            }
        )
        snapshot = {
            "checked_at": now,
            "ctr": 0.0,
            "cvr": 0.0,
            "hit_rate": 0.0,
            "new_logs": 0,
            "thresholds": {
                "ctr_threshold": self.config.monitoring.ctr_threshold,
                "hitrate_threshold": self.config.monitoring.hitrate_threshold,
                "new_logs_threshold": self.config.monitoring.new_logs_threshold,
                "ctr_min_logs": self.config.monitoring.ctr_min_logs,
            },
        }
        state.update(
            {
                "last_checked_at": now,
                "last_log_count": 0,
                "last_log_source": source_key,
                "log_sources": log_sources,
                "log_rotations": history[-30:],
                "last_decision": {
                    "should_retrain": False,
                    "reasons": [],
                    "snapshot": snapshot,
                },
            }
        )
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    def metrics(self) -> dict[str, Any]:
        report = self._read_json(self.config.paths.artifacts_dir / "reports" / "metrics.json")
        manifest = self._read_json(self.config.paths.processed_dir / "manifest.json")
        ct_state = self._read_json(self.config.paths.artifacts_dir / "registry" / "ct_state.json")
        registry = self._read_json(self.config.paths.artifacts_dir / "registry" / "models.json")
        search_report = report.get("search", {})
        recommendation_report = report.get("recommendation", {})
        row_counts = manifest.get("row_counts", {})
        event_distribution = manifest.get("event_distribution", {})
        persona_distribution = manifest.get("persona_distribution", {})
        event_total = max(sum(int(value) for value in event_distribution.values()), 1)
        persona_total = max(sum(int(value) for value in persona_distribution.values()), 1)
        active_model = _active_model_version(registry)
        train_events = int(row_counts.get("train_events", 0) or 0)
        valid_events = int(row_counts.get("valid_events", 0) or 0)
        test_events = int(row_counts.get("test_events", 0) or 0)
        split_total = max(train_events + valid_events + test_events, 1)
        ct_decision = ct_state.get("last_decision", {})
        ct_snapshot = ct_decision.get("snapshot", {}) if isinstance(ct_decision, dict) else {}
        ct_reasons = ct_decision.get("reasons", []) if isinstance(ct_decision, dict) else []
        ct_should_retrain = (
            bool(ct_decision.get("should_retrain", False))
            if isinstance(ct_decision, dict)
            else False
        )
        live_surfaces = self._live_surface_stats()
        recent_live_events = self._recent_live_events(LIVE_RECENT_EVENT_LIMIT)
        live_training_ctr, live_training_cvr, live_training_rate_source = (
            _select_live_training_rates(live_surfaces)
        )
        api_log_source = _ct_log_source_state(ct_state, "api_events_jsonl")
        pending_new_logs = int(
            api_log_source.get("pending_new_logs", ct_snapshot.get("new_logs", 0)) or 0
        )
        current_live_log_count = int(
            api_log_source.get("current_log_count", self._count_event_log_lines()) or 0
        )
        last_live_log_count = int(api_log_source.get("last_log_count", 0) or 0)
        alert_reasons = [
            reason
            for reason in ct_reasons
            if reason in {"ctr_below_threshold", "hitrate_below_threshold"}
        ]
        retrain_reasons = [
            reason for reason in ct_reasons if reason == "new_logs_threshold_reached"
        ]
        base_event_count = int(row_counts.get("events", self.config.mode.events) or 0)
        live_event_count = int(self._count_event_log_lines() or 0)
        return {
            "mode": self.config.active_mode,
            "artifact_readiness": self.artifacts_status(),
            "search": {
                "mrr": search_report.get("mrr", search_report.get("mrr_at_10", 0.0)),
                "mrr_at_10": search_report.get("mrr_at_10", search_report.get("mrr", 0.0)),
                "ndcg_at_10": search_report.get("ndcg_at_10", 0.0),
                "recall_at_10": search_report.get("recall_at_10", 0.0),
                "category_hit_at_10": search_report.get("category_hit_at_10", 0.0),
                "latency_p50_ms": search_report.get("latency_p50_ms", 0.0),
                "latency_p95_ms": search_report.get("latency_p95_ms", 0.0),
                "evaluated_queries": search_report.get("evaluated_queries", 0),
                "prediction_sample_size": search_report.get("prediction_sample_size", 0),
                "encoder_type": self.config.search.encoder_type,
                "source": search_report.get("source", "unknown"),
                "primary_evaluation": search_report.get(
                    "primary_evaluation",
                    "production_with_qrels_prior",
                ),
                "quality_status": search_report.get("quality_status", ""),
                "production_with_qrels_prior": search_report.get("production_with_qrels_prior", {}),
                "strict_no_prior_diagnostic": search_report.get("strict_no_prior_diagnostic", {}),
                "target_status": {
                    "mrr": report.get("targets", {}).get("search_mrr_at_10", {}).get("status"),
                    "ndcg_at_10": report.get("targets", {})
                    .get("search_ndcg_at_10", {})
                    .get("status"),
                    "latency_p95_ms": report.get("targets", {})
                    .get("search_latency_p95_ms", {})
                    .get("status"),
                },
            },
            "recommendation": {
                "recall_at_300": recommendation_report.get("recall_at_300", 0.0),
                "auc": recommendation_report.get("auc", 0.0),
                "hit_rate_at_50": recommendation_report.get(
                    "hit_rate_at_50", recommendation_report.get("hitrate_at_50", 0.0)
                ),
                "hitrate_at_50": recommendation_report.get(
                    "hitrate_at_50", recommendation_report.get("hit_rate_at_50", 0.0)
                ),
                "ndcg_at_50": recommendation_report.get("ndcg_at_50", 0.0),
                "coverage": recommendation_report.get(
                    "coverage", recommendation_report.get("coverage_at_50", 0.0)
                ),
                "candidate_p95_ms": recommendation_report.get("candidate_latency_p95_ms", 0.0),
                "ranking_p95_ms": recommendation_report.get("ranking_latency_p95_ms", 0.0),
                "reranking_p95_ms": recommendation_report.get("reranking_latency_p95_ms", 0.0),
                "total_p95_ms": recommendation_report.get("total_latency_p95_ms", 0.0),
                "source": recommendation_report.get("source", "unknown"),
                "evaluated_users": recommendation_report.get("evaluated_users", 0),
            },
            "system": {
                "mode": self.config.active_mode,
                "events": base_event_count,
                "base_events": base_event_count,
                "logged_events": live_event_count,
                "live_events": live_event_count,
                "total_events": base_event_count + live_event_count,
                "products": row_counts.get("products", self.config.mode.products),
                "users": row_counts.get("users", self.config.mode.users),
                "configured_products": self.config.mode.products,
                "configured_users": self.config.mode.users,
                "configured_events": self.config.mode.events,
                "redis": "ready" if self.redis_client else "unavailable",
                "manifest_rows": row_counts,
                "ctr": _safe_div(
                    event_distribution.get("cart", 0) + event_distribution.get("purchase", 0),
                    event_distribution.get("view", 0) + event_distribution.get("search", 0),
                ),
                "cvr": _safe_div(event_distribution.get("purchase", 0), event_total),
                "api_latency_p95_ms": max(
                    float(search_report.get("latency_p95_ms", 0.0) or 0.0),
                    float(recommendation_report.get("total_latency_p95_ms", 0.0) or 0.0),
                ),
                "redis_latency_ms": self._measure_redis_latency_ms(),
                "active_model_version": active_model,
                "served_search_model_version": self._served_search_version,
                "search_reload_error": self._search_reload_error,
                "served_model_version": self._served_recommendation_version,
                "recommendation_artifact_version": self._recommendation_artifact_version(),
                "recommendation_reload_error": self._recommendation_reload_error,
            },
            "data_quality": {
                "persona_count": len(persona_distribution),
                "persona_names": sorted(str(key) for key in persona_distribution.keys()),
                "required_personas": [
                    "trendsetter",
                    "pragmatist",
                    "value_seeker",
                    "top_category_loyalist",
                    "impulse_buyer",
                    "careful_explorer",
                ],
                "event_types": sorted(str(key) for key in event_distribution.keys()),
                "category_hierarchy_depth": 3,
                "has_hm_images": True,
                "split": {
                    "train": round(train_events / split_total, 4),
                    "valid": round(valid_events / split_total, 4),
                    "test": round(test_events / split_total, 4),
                },
                "split_counts": {
                    "train": train_events,
                    "valid": valid_events,
                    "test": test_events,
                },
                "random_seed": self.config.seed,
            },
            "simulator": {
                "personas": {
                    key: round(int(value) / persona_total, 6)
                    for key, value in persona_distribution.items()
                },
                "events": {
                    key: round(int(value) / event_total, 6)
                    for key, value in event_distribution.items()
                },
                "timeline": recent_live_events,
                "minute_timeline": self._live_event_minute_timeline(),
                "live_surfaces": live_surfaces,
            },
            "training": {
                "status": "retrain_required" if retrain_reasons else "watching",
                "should_retrain": ct_should_retrain,
                "alert_active": bool(alert_reasons),
                "alert_reasons": alert_reasons,
                "retrain_trigger_active": bool(retrain_reasons),
                "retrain_reasons": retrain_reasons,
                "reasons": ct_reasons,
                "new_logs": pending_new_logs,
                "new_logs_threshold": self.config.monitoring.new_logs_threshold,
                "current_log_count": current_live_log_count,
                "last_log_count": last_live_log_count,
                "log_source": "api_events_jsonl",
                "ctr": live_training_ctr
                if live_training_ctr is not None
                else ct_snapshot.get("ctr", 0.0),
                "cvr": live_training_cvr
                if live_training_cvr is not None
                else ct_snapshot.get("cvr", 0.0),
                "ctr_source": live_training_rate_source
                if live_training_ctr is not None
                else "ct_state_snapshot",
                "cvr_source": live_training_rate_source
                if live_training_cvr is not None
                else "ct_state_snapshot",
                "hit_rate": ct_snapshot.get("hit_rate", 0.0),
                "ctr_threshold": ct_snapshot.get("thresholds", {}).get(
                    "ctr_threshold", self.config.monitoring.ctr_threshold
                ),
                "ctr_min_logs": ct_snapshot.get("thresholds", {}).get(
                    "ctr_min_logs", self.config.monitoring.ctr_min_logs
                ),
                "hitrate_threshold": ct_snapshot.get("thresholds", {}).get(
                    "hitrate_threshold", self.config.monitoring.hitrate_threshold
                ),
                "last_checked_at": ct_snapshot.get("checked_at"),
                "last_retrain": registry.get("last_retrain"),
                "next_action": (
                    "Retrain trigger active: " + ", ".join(retrain_reasons)
                    if retrain_reasons
                    else (
                        "Metric alert active: " + ", ".join(alert_reasons)
                        if alert_reasons
                        else "No retrain trigger condition is active."
                    )
                ),
                "versions": _summarize_registry_versions(registry),
            },
            "artifacts": self.artifacts_status(),
        }

    def assign_bucket(self, request: ABAssignRequest) -> str:
        digest = hashlib.sha256(f"{request.experiment_key}:{request.user_id}".encode()).hexdigest()
        return request.buckets[int(digest[:8], 16) % len(request.buckets)]

    def ab_report(self, experiment_key: str) -> ABReportResponse:
        stats = self._ab_stats_from_log(experiment_key)
        control = stats.setdefault("control", {"impressions": 0, "clicks": 0, "conversions": 0})
        treatment = stats.setdefault("treatment", {"impressions": 0, "clicks": 0, "conversions": 0})
        p1 = _raw_rate(control["conversions"], control["impressions"])
        p2 = _raw_rate(treatment["conversions"], treatment["impressions"])
        success_key = (
            "conversions"
            if int(control["conversions"]) + int(treatment["conversions"]) > 0
            else "clicks"
        )
        p_value, ci = _two_proportion_stats(
            control[success_key],
            control["impressions"],
            treatment[success_key],
            treatment["impressions"],
        )
        buckets = {
            name: {
                "impressions": values["impressions"],
                "clicks": values["clicks"],
                "conversions": values["conversions"],
                "ctr": _rate(values["clicks"], values["impressions"]),
                "cvr": _rate(values["conversions"], values["impressions"]),
                "purchase_per_click": _rate(values["conversions"], values["clicks"]),
            }
            for name, values in stats.items()
        }
        control_ctr = _raw_rate(control["clicks"], control["impressions"])
        treatment_ctr = _raw_rate(treatment["clicks"], treatment["impressions"])
        return ABReportResponse(
            experiment_key=experiment_key,
            buckets=buckets,
            uplift=round(p2 - p1, 6),
            uplift_by_metric={
                "ctr": round(treatment_ctr - control_ctr, 6),
                "cvr": round(p2 - p1, 6),
            },
            p_value=p_value,
            confidence_interval_95=ci,
        )

    def _registry_payload(self) -> dict[str, Any]:
        return self._read_json(self.config.paths.artifacts_dir / "registry" / "models.json")

    def _active_recommendation_version(self) -> str:
        version = _active_model_version(self._registry_payload())
        if version != "unregistered":
            return version
        artifact_version = self._recommendation_artifact_version()
        return artifact_version if artifact_version != "unknown" else "unregistered"

    def _active_search_version(self) -> str:
        registry = self._read_json(
            self.config.paths.artifacts_dir / "registry" / "search_models.json"
        )
        active = registry.get("active_version")
        if active:
            return str(active)
        behavior_path = self.config.paths.artifacts_dir / "search" / "query_behavior_model.json.gz"
        if behavior_path.exists():
            stat = behavior_path.stat()
            return f"behavior:{stat.st_size}:{stat.st_mtime_ns}"
        return "unregistered"

    def _recommendation_artifact_version(self) -> str:
        artifacts = getattr(self.recommendation_service, "artifacts", None)
        version = getattr(artifacts, "version", None)
        return str(version) if version else "unknown"

    def _maybe_reload_search_service(self) -> None:
        active_version = self._active_search_version()
        if self.search_service is not None and active_version in {
            "",
            "unregistered",
            self._served_search_version,
        }:
            return

        with self._search_reload_lock:
            active_version = self._active_search_version()
            if self.search_service is not None and active_version in {
                "",
                "unregistered",
                self._served_search_version,
            }:
                return
            try:
                next_service = _instantiate(
                    _load_class("mars.search.service", "SearchService"),
                    self.config,
                )
                if next_service is None:
                    raise RuntimeError("SearchService could not be instantiated")
                self.search_service = next_service
                self._served_search_version = active_version
                self._search_reload_error = None
            except Exception as exc:
                self._search_reload_error = f"{exc.__class__.__name__}: {exc}"

    def _maybe_reload_recommendation_service(self) -> None:
        active_version = self._active_recommendation_version()
        if active_version in {"", "unregistered", self._served_recommendation_version}:
            return

        with self._recommendation_reload_lock:
            active_version = self._active_recommendation_version()
            if active_version in {"", "unregistered", self._served_recommendation_version}:
                return
            try:
                next_service = _instantiate(
                    _load_class("mars.recommendation.service", "RecommendationService"),
                    self.config,
                )
                if next_service is None:
                    raise RuntimeError("RecommendationService could not be instantiated")
                self._attach_session_store_to(next_service)
                self.recommendation_service = next_service
                self._served_recommendation_version = active_version
                self._recommendation_reload_error = None
            except Exception as exc:
                self._recommendation_reload_error = f"{exc.__class__.__name__}: {exc}"

    def _fallback_products(self) -> list[dict[str, Any]]:
        products_path = self.config.paths.processed_dir / "products.parquet"
        if products_path.exists():
            try:
                import pandas as pd

                return pd.read_parquet(products_path).head(500).to_dict(orient="records")
            except Exception:
                pass
        return [
            {
                "product_id": f"P{i:08d}",
                "name": name,
                "category_l1": category,
                "category": category,
                "price": price,
                "image_path": f"/static/products/P{i:08d}.jpg",
                "popularity_prior": 1.0 - (i * 0.03),
                "description": f"{category} {name}",
            }
            for i, (name, category, price) in enumerate(
                [
                    ("Black Tech Jacket", "outer", 129000),
                    ("Lime Training Hoodie", "top", 79000),
                    ("Coral City Sneakers", "shoes", 99000),
                    ("Charcoal Wide Pants", "bottom", 89000),
                    ("Cyan Nylon Cross Bag", "bag", 59000),
                    ("Minimal White Shirt", "top", 49000),
                    ("New Wave Denim", "bottom", 109000),
                    ("Runner Zip Vest", "outer", 69000),
                    ("Soft Knit Cardigan", "top", 89000),
                    ("Everyday Canvas Tote", "bag", 39000),
                ],
                start=1,
            )
        ]

    def _fallback_search_results(
        self,
        request: SearchRequest,
        products: list[dict[str, Any]],
    ) -> list[SearchResult]:
        query = (request.query or "").lower()
        scored: list[tuple[float, dict[str, Any]]] = []
        for index, product in enumerate(products):
            if not self._matches_filters(product, request):
                continue
            text = " ".join(
                str(product.get(key, ""))
                for key in (
                    "name",
                    "category",
                    "category_l1",
                    "category_l2",
                    "description",
                )
            ).lower()
            lexical = sum(1.0 for token in query.split() if token and token in text)
            popularity = _product_popularity(product)
            scored.append((lexical + popularity + (1.0 / (index + 10)), product))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchResult(
                product_id=str(product.get("product_id")),
                name=str(product.get("name", product.get("product_id"))),
                score=round(float(score), 6),
                price=int(product.get("price", 0)),
                category=str(product.get("category", product.get("category_l1", ""))) or None,
                image_url=str(product.get("image_url", product.get("image_path", ""))) or None,
            )
            for score, product in scored[: request.top_k]
        ]

    def _matches_filters(self, product: dict[str, Any], request: SearchRequest) -> bool:
        filters = request.filters
        category = str(product.get("category", product.get("category_l1", ""))).lower()
        price = int(product.get("price", 0) or 0)
        return not (
            (filters.category and filters.category.lower() not in category)
            or (filters.min_price is not None and price < filters.min_price)
            or (filters.max_price is not None and price > filters.max_price)
        )

    def _rank_fallback_products(
        self,
        products: list[dict[str, Any]],
        user_id: str,
        recent_categories: list[str],
    ) -> list[dict[str, Any]]:
        seed = int(hashlib.sha256(user_id.encode()).hexdigest()[:8], 16)
        preferred = set(recent_categories)
        ranked = []
        for index, product in enumerate(products):
            category = str(product.get("category", product.get("category_l1", "")))
            popularity = _product_popularity(product)
            product = dict(product)
            product["_score"] = (
                popularity
                + (0.25 if category in preferred else 0.0)
                + (((seed + index * 17) % 100) / 1000)
            )
            ranked.append(product)
        ranked.sort(key=lambda item: item["_score"], reverse=True)
        return ranked

    def _fallback_recommendation_items(
        self,
        products: list[dict[str, Any]],
        top_n: int,
    ) -> list[RecommendationItem]:
        result: list[RecommendationItem] = []
        streaks: dict[str, int] = {}
        exploration_budget = min(self.config.recommendation.exploration_slots, max(top_n // 5, 1))
        for product in products:
            if len(result) >= top_n:
                break
            category = str(product.get("category", product.get("category_l1", "general")))
            if streaks.get(category, 0) >= self.config.recommendation.max_same_category_streak:
                continue
            is_exploration = len(result) >= max(top_n - exploration_budget, 0)
            streaks[category] = streaks.get(category, 0) + 1
            result.append(
                RecommendationItem(
                    product_id=str(product.get("product_id")),
                    name=str(product.get("name", "")),
                    category=str(product.get("category", product.get("category_l1", ""))),
                    price=float(product.get("price", 0.0) or 0.0),
                    score=round(float(product.get("_score", 0.0)), 6),
                    reason=(
                        "session/popularity fallback"
                        if not is_exploration
                        else "MAB exploration slot"
                    ),
                    is_exploration=is_exploration,
                )
            )
        return result

    def _update_redis_features(self, payload: dict[str, Any]) -> bool:
        if self.redis_client is None:
            return False
        try:
            user_id = payload["user_id"]
            session_id = payload["session_id"]
            product_id = payload.get("product_id")
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            category = metadata.get("category") or payload.get("category")
            event_type = str(payload.get("event_type", ""))
            event_role = str(metadata.get("event_role", ""))
            is_click_event = event_type in {"view", "cart", "purchase"} and event_role != "exposure"
            pipe = self.redis_client.pipeline()
            if product_id and is_click_event:
                for key in (
                    f"user:{user_id}:recent_products",
                    f"session:{session_id}:recent_products",
                ):
                    pipe.lpush(key, product_id)
                    pipe.ltrim(key, 0, self.config.recommendation.session_recent_n - 1)
            if category:
                for key in (
                    f"user:{user_id}:recent_categories",
                    f"session:{session_id}:recent_categories",
                ):
                    pipe.lpush(key, category)
                    pipe.ltrim(key, 0, self.config.recommendation.session_recent_n - 1)
            pipe.incr("system:logged_events")
            pipe.hincrby(f"user:{user_id}:event_counts", event_type, 1)
            pipe.hincrby(f"session:{session_id}:event_counts", event_type, 1)
            if is_click_event:
                pipe.incr(f"session:{session_id}:click_count")
                pipe.incr(f"user:{user_id}:click_count")
            self._queue_live_event(pipe, payload)
            self._increment_surface_counters(pipe, payload)
            experiment_key = metadata.get("experiment_key")
            if experiment_key:
                bucket = metadata.get("ab_bucket") or _deterministic_bucket(
                    str(experiment_key),
                    str(payload.get("user_id", "")),
                )
                ab_key = f"ab:{experiment_key}:{bucket}"
                self._increment_ab_counters(pipe, ab_key, payload)
            pipe.execute()
            return True
        except Exception:
            return False

    def _queue_live_event(self, pipe: Any, payload: dict[str, Any]) -> None:
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        summary = {
            "timestamp": payload.get("timestamp"),
            "event_id": payload.get("event_id"),
            "user_id": payload.get("user_id"),
            "session_id": payload.get("session_id"),
            "event_type": payload.get("event_type"),
            "product_id": payload.get("product_id"),
            "query": payload.get("query"),
            "category": payload.get("category") or metadata.get("category"),
            "surface": metadata.get("source_surface") or metadata.get("surface"),
            "event_role": metadata.get("event_role"),
            "rank": metadata.get("rank"),
            "ab_bucket": metadata.get("ab_bucket"),
            "strategy": metadata.get("strategy"),
        }
        pipe.lpush("live:recent_events", json.dumps(summary, ensure_ascii=False, default=str))
        pipe.ltrim("live:recent_events", 0, LIVE_RECENT_EVENT_LIMIT - 1)

    def _increment_surface_counters(self, pipe: Any, payload: dict[str, Any]) -> None:
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        surface = str(metadata.get("source_surface") or metadata.get("surface") or "").strip()
        if surface not in {"search", "recommendation"}:
            return
        event_role = str(metadata.get("event_role", ""))
        event_type = str(payload.get("event_type", ""))
        key = f"live:surface:{surface}"
        if event_role == "exposure":
            pipe.hincrby(key, "impressions", 1)
        elif event_type == "view":
            pipe.hincrby(key, "clicks", 1)
        elif event_type == "cart":
            pipe.hincrby(key, "carts", 1)
        elif event_type == "purchase":
            pipe.hincrby(key, "conversions", 1)

    def _increment_ab_counters(self, pipe: Any, ab_key: str, payload: dict[str, Any]) -> None:
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        surface = str(metadata.get("source_surface") or metadata.get("surface") or "").strip()
        event_role = str(metadata.get("event_role", ""))
        event_type = str(payload.get("event_type", ""))
        if surface == "recommendation":
            if event_role == "exposure":
                pipe.hincrby(ab_key, "impressions", 1)
            elif event_type == "view":
                pipe.hincrby(ab_key, "clicks", 1)
            elif event_type == "purchase":
                pipe.hincrby(ab_key, "conversions", 1)
            return
        if not surface:
            if event_type in {"search", "view", "cart", "purchase"}:
                pipe.hincrby(ab_key, "impressions", 1)
            if event_type in {"view", "cart", "purchase"}:
                pipe.hincrby(ab_key, "clicks", 1)
            if event_type == "purchase":
                pipe.hincrby(ab_key, "conversions", 1)

    def _redis_lrange(self, key: str, start: int, end: int) -> list[str]:
        if self.redis_client is None:
            return []
        try:
            values = self.redis_client.lrange(key, start, end)
            return [value.decode() if isinstance(value, bytes) else str(value) for value in values]
        except Exception:
            return []

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def _count_event_log_lines(self) -> int:
        if self.redis_client is not None:
            try:
                value = self.redis_client.get("system:logged_events")
                if value is not None:
                    return int(value)
            except Exception:
                pass
        path = self.config.paths.logs_dir / "api_events.jsonl"
        if not path.exists():
            return 0
        try:
            stat = path.stat()
            cached = self._event_log_count_cache
            if cached and cached[0] == stat.st_size and cached[1] == stat.st_mtime_ns:
                return cached[2]
            count = 0
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    count += chunk.count(b"\n")
            self._event_log_count_cache = (stat.st_size, stat.st_mtime_ns, count)
            return count
        except Exception:
            return 0

    def _measure_redis_latency_ms(self) -> float | None:
        if self.redis_client is None:
            return None
        try:
            started = time.perf_counter()
            self.redis_client.ping()
            return round((time.perf_counter() - started) * 1000.0, 3)
        except Exception:
            return None

    def _ab_stats_from_log(self, experiment_key: str) -> dict[str, dict[str, int]]:
        stats = {
            "control": {"impressions": 0, "clicks": 0, "conversions": 0},
            "treatment": {"impressions": 0, "clicks": 0, "conversions": 0},
        }
        if self.redis_client is not None:
            try:
                has_redis_counts = False
                for bucket in tuple(stats):
                    key = f"ab:{experiment_key}:{bucket}"
                    raw = self.redis_client.hgetall(key)
                    if not raw:
                        continue
                    decoded = {
                        (k.decode() if isinstance(k, bytes) else str(k)): int(v)
                        for k, v in raw.items()
                    }
                    stats[bucket].update(
                        {
                            "impressions": int(decoded.get("impressions", 0)),
                            "clicks": int(decoded.get("clicks", 0)),
                            "conversions": int(decoded.get("conversions", 0)),
                        }
                    )
                    has_redis_counts = has_redis_counts or any(stats[bucket].values())
                if has_redis_counts:
                    return stats
            except Exception:
                pass
        path = self.config.paths.logs_dir / "api_events.jsonl"
        if not path.exists():
            return stats
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                metadata = payload.get("metadata", {})
                if metadata.get("experiment_key", "default") != experiment_key:
                    continue
                bucket = metadata.get("ab_bucket") or _deterministic_bucket(
                    experiment_key,
                    str(payload.get("user_id", "")),
                )
                stats.setdefault(bucket, {"impressions": 0, "clicks": 0, "conversions": 0})
                event_type = str(payload.get("event_type", ""))
                surface = str(metadata.get("source_surface") or metadata.get("surface") or "")
                event_role = str(metadata.get("event_role", ""))
                if surface == "recommendation":
                    if event_role == "exposure":
                        stats[bucket]["impressions"] += 1
                    elif event_type == "view":
                        stats[bucket]["clicks"] += 1
                    elif event_type == "purchase":
                        stats[bucket]["conversions"] += 1
                elif not surface:
                    if event_type in {"search", "view", "cart", "purchase"}:
                        stats[bucket]["impressions"] += 1
                    if event_type in {"view", "cart", "purchase"}:
                        stats[bucket]["clicks"] += 1
                    if event_type == "purchase":
                        stats[bucket]["conversions"] += 1
        return stats

    def _live_surface_stats(self) -> dict[str, dict[str, float | int]]:
        surfaces = {
            "search": {"impressions": 0, "clicks": 0, "carts": 0, "conversions": 0},
            "recommendation": {"impressions": 0, "clicks": 0, "carts": 0, "conversions": 0},
        }
        if self.redis_client is not None:
            try:
                for surface in tuple(surfaces):
                    raw = self.redis_client.hgetall(f"live:surface:{surface}")
                    if not raw:
                        continue
                    decoded = {
                        (key.decode() if isinstance(key, bytes) else str(key)): int(value)
                        for key, value in raw.items()
                    }
                    surfaces[surface].update(
                        {
                            "impressions": int(decoded.get("impressions", 0)),
                            "clicks": int(decoded.get("clicks", 0)),
                            "carts": int(decoded.get("carts", 0)),
                            "conversions": int(decoded.get("conversions", 0)),
                        }
                    )
            except Exception:
                pass
        return {
            name: {
                **values,
                "ctr": _rate(int(values["clicks"]), int(values["impressions"])),
                "cart_rate": _rate(int(values["carts"]), int(values["clicks"])),
                "cvr": _rate(int(values["conversions"]), int(values["impressions"])),
                "purchase_per_click": _rate(
                    int(values["conversions"]),
                    int(values["clicks"]),
                ),
            }
            for name, values in surfaces.items()
        }

    def _recent_live_events(self, limit: int = LIVE_RECENT_EVENT_LIMIT) -> list[dict[str, Any]]:
        redis_events: list[dict[str, Any]] = []
        if self.redis_client is not None:
            try:
                raw_values = self.redis_client.lrange("live:recent_events", 0, limit - 1)
                for raw in raw_values:
                    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                    value = json.loads(text)
                    if isinstance(value, dict):
                        redis_events.append(value)
            except Exception:
                pass
        file_events = self._recent_live_events_from_file(limit)
        if not redis_events:
            return file_events
        if not file_events:
            return redis_events

        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for event in [*redis_events, *file_events]:
            key = str(event.get("event_id") or "")
            if not key:
                key = "|".join(
                    str(event.get(field) or "")
                    for field in (
                        "timestamp",
                        "user_id",
                        "session_id",
                        "event_type",
                        "product_id",
                        "rank",
                    )
                )
            if key in seen:
                continue
            seen.add(key)
            merged.append(event)
            if len(merged) >= limit:
                break
        return merged

    @staticmethod
    def _minute_bucket_count(events: list[dict[str, Any]]) -> int:
        buckets: set[str] = set()
        for event in events:
            timestamp = str(event.get("timestamp") or "")
            if not timestamp:
                continue
            try:
                moment = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                continue
            buckets.add(moment.replace(second=0, microsecond=0).isoformat())
        return len(buckets)

    def _recent_live_events_from_file(self, limit: int) -> list[dict[str, Any]]:
        path = self.config.paths.logs_dir / "api_events.jsonl"
        if not path.exists():
            return []
        from collections import deque

        recent: deque[dict[str, Any]] = deque(maxlen=limit)
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    metadata = payload.get("metadata", {})
                    if not isinstance(metadata, dict):
                        metadata = {}
                    recent.append(
                        {
                            "timestamp": payload.get("timestamp"),
                            "event_id": payload.get("event_id"),
                            "user_id": payload.get("user_id"),
                            "session_id": payload.get("session_id"),
                            "event_type": payload.get("event_type"),
                            "product_id": payload.get("product_id"),
                            "query": payload.get("query"),
                            "category": payload.get("category") or metadata.get("category"),
                            "surface": metadata.get("source_surface") or metadata.get("surface"),
                            "event_role": metadata.get("event_role"),
                            "rank": metadata.get("rank"),
                            "ab_bucket": metadata.get("ab_bucket"),
                            "strategy": metadata.get("strategy"),
                        }
                    )
            return list(reversed(recent))
        except Exception:
            return []

    def _live_event_minute_timeline(
        self,
        *,
        max_events: int = 5000,
        max_minutes: int = 30,
    ) -> list[dict[str, Any]]:
        events = self._recent_live_events(max_events)
        if not events:
            events = self._recent_live_events_from_file(max_events)
        buckets: dict[tuple[datetime, str], int] = {}
        for event in events:
            timestamp = str(event.get("timestamp") or "")
            if not timestamp:
                continue
            try:
                moment = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                continue
            minute = moment.replace(second=0, microsecond=0)
            event_type = _semantic_live_event_type(event)
            if event_type is None:
                continue
            key = (minute, event_type)
            buckets[key] = buckets.get(key, 0) + 1
        if not buckets:
            return []
        recent_minutes = sorted({minute for minute, _ in buckets})[-max_minutes:]
        recent_set = set(recent_minutes)
        rows = [
            {
                "minute": minute.isoformat(),
                "event_type": event_type,
                "count": count,
            }
            for (minute, event_type), count in buckets.items()
            if minute in recent_set
        ]
        return sorted(rows, key=lambda row: (str(row["minute"]), str(row["event_type"])))


def _rate(successes: int, total: int) -> float:
    return round(successes / total, 6) if total else 0.0


def _safe_slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in str(value)]
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:80] or "rotate"


def _semantic_live_event_type(event: dict[str, Any]) -> str | None:
    event_type = str(event.get("event_type") or "").strip()
    if event_type in {"view", "cart", "purchase"}:
        return event_type
    if event_type != "search":
        return None

    event_role = str(event.get("event_role") or "").strip()
    product_id = event.get("product_id")
    rank = event.get("rank")
    if event_role == "user_action" or not product_id or rank in {None, ""}:
        return "search"
    return None


def _raw_rate(successes: int, total: int) -> float:
    return successes / total if total else 0.0


def _deterministic_bucket(experiment_key: str, user_id: str) -> str:
    digest = hashlib.sha256(f"{experiment_key}:{user_id}".encode()).hexdigest()
    return "treatment" if int(digest[:8], 16) % 2 else "control"


def _safe_div(numerator: Any, denominator: Any) -> float:
    try:
        return round(float(numerator) / float(denominator), 6) if float(denominator) else 0.0
    except Exception:
        return 0.0


def _select_live_training_rates(
    live_surfaces: dict[str, dict[str, float | int]],
) -> tuple[float | None, float | None, str]:
    recommendation = live_surfaces.get("recommendation", {})
    if int(recommendation.get("impressions", 0) or 0) > 0:
        return (
            float(recommendation.get("ctr", 0.0) or 0.0),
            float(recommendation.get("cvr", 0.0) or 0.0),
            "live:surface:recommendation",
        )
    search = live_surfaces.get("search", {})
    if int(search.get("impressions", 0) or 0) > 0:
        return (
            float(search.get("ctr", 0.0) or 0.0),
            float(search.get("cvr", 0.0) or 0.0),
            "live:surface:search",
        )
    return None, None, "unavailable"


def _ct_log_source_state(ct_state: dict[str, Any], source_key: str) -> dict[str, Any]:
    sources = ct_state.get("log_sources", {}) if isinstance(ct_state, dict) else {}
    if not isinstance(sources, dict):
        return {}
    value = sources.get(source_key, {})
    return value if isinstance(value, dict) else {}


def _active_model_version(registry: dict[str, Any]) -> str:
    active = registry.get("active") or registry.get("active_version")
    if active:
        return str(active)
    versions = registry.get("versions", [])
    if isinstance(versions, list) and versions:
        latest = versions[-1]
        if isinstance(latest, dict):
            return str(latest.get("version", "unregistered"))
    return "unregistered"


def _summarize_registry_versions(registry: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    versions = registry.get("versions", [])
    if not isinstance(versions, list):
        return []

    summarized: list[dict[str, Any]] = []
    for entry in versions[-limit:]:
        if not isinstance(entry, dict):
            continue
        metadata = entry.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        summarized.append(
            {
                "version": entry.get("version", ""),
                "created_at": entry.get("created_at", ""),
                "artifact_path": entry.get("artifact_path", ""),
                "metrics_path": entry.get("metrics_path", ""),
                "status": entry.get("status", ""),
                "metadata": {
                    "mode": metadata.get("mode", ""),
                    "job": metadata.get("job", ""),
                    "encoder": metadata.get("encoder", ""),
                    "recsys_version": metadata.get("recsys_version", ""),
                    "live_log_count": metadata.get("live_log_count", ""),
                },
            }
        )
    return summarized


def _product_popularity(product: dict[str, Any]) -> float:
    return float(product.get("popularity_prior", product.get("popularity", 0.5)) or 0.5)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _two_proportion_stats(
    success_a: int,
    total_a: int,
    success_b: int,
    total_b: int,
) -> tuple[float, list[float]]:
    if total_a == 0 or total_b == 0:
        return 1.0, [0.0, 0.0]
    p1 = success_a / total_a
    p2 = success_b / total_b
    diff = p2 - p1
    pooled = (success_a + success_b) / (total_a + total_b)
    standard_error = math.sqrt(max(pooled * (1 - pooled) * (1 / total_a + 1 / total_b), 1e-12))
    z_value = diff / standard_error
    p_value = 2 * (1 - _normal_cdf(abs(z_value)))
    ci_error = 1.96 * math.sqrt(max((p1 * (1 - p1) / total_a) + (p2 * (1 - p2) / total_b), 1e-12))
    return round(float(p_value), 6), [round(diff - ci_error, 6), round(diff + ci_error, 6)]
