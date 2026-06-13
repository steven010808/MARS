from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mars.search.service import _normalize_query_key

ENGAGEMENT_WEIGHTS: dict[str, int] = {
    "view": 1,
    "cart": 3,
    "purchase": 5,
}


def read_event_log(path: str | Path) -> list[dict[str, Any]]:
    log_path = Path(path)
    if not log_path.exists():
        return []
    events: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
    return events


def build_search_feedback_frame(
    events: Iterable[dict[str, Any]],
    *,
    catalog_products: set[str] | None = None,
    validation_ratio: float = 0.2,
    click_window_seconds: int = 60 * 60 * 24,
    include_unclicked_negatives: bool = True,
    max_negative_rank: int = 10,
    require_search_surface: bool = True,
) -> pd.DataFrame:
    catalog = {str(product_id) for product_id in catalog_products or set()}
    require_catalog = bool(catalog)
    exposures: dict[str, dict[str, Any]] = {}
    session_exposures: dict[str, list[dict[str, Any]]] = defaultdict(list)
    positives_by_search: dict[str, set[str]] = defaultdict(set)
    rows: list[dict[str, Any]] = []

    for order, payload in enumerate(events):
        if not isinstance(payload, dict):
            continue
        metadata = _metadata(payload)
        event_type = str(payload.get("event_type", "") or "")
        timestamp = str(payload.get("timestamp", "") or "")
        timestamp_seconds = _timestamp_seconds(timestamp, fallback=float(order))

        if _is_search_exposure(payload, metadata):
            exposure = _exposure_record(payload, metadata, order, timestamp_seconds)
            if not exposure["query_key"] or not exposure["result_product_ids"]:
                continue
            search_id = str(exposure["search_id"])
            exposures[search_id] = exposure
            session_id = str(exposure.get("session_id", "") or "")
            if session_id:
                session_exposures[session_id].append(exposure)
            continue

        if event_type not in ENGAGEMENT_WEIGHTS:
            continue
        product_id = str(payload.get("product_id", "") or "")
        if not product_id or (require_catalog and product_id not in catalog):
            continue
        exposure = _matching_exposure(
            payload,
            metadata,
            exposures,
            session_exposures,
            timestamp_seconds,
            click_window_seconds=click_window_seconds,
        )
        if require_search_surface and not _has_search_feedback_marker(metadata, exposure):
            continue
        query = (
            str(payload.get("query", "") or "")
            or str(metadata.get("query", "") or "")
            or (str(exposure.get("query", "") or "") if exposure else "")
        )
        query_key = _normalize_query_key(query)
        if not query_key:
            continue
        search_id = str(metadata.get("search_id", "") or (exposure or {}).get("search_id", ""))
        rank = _coerce_int(metadata.get("rank"), default=None)
        if rank is None and exposure:
            rank = exposure.get("rank_by_product", {}).get(product_id)
        row = {
            "query": query,
            "query_key": query_key,
            "product_id": product_id,
            "label": 1,
            "weight": int(ENGAGEMENT_WEIGHTS[event_type]),
            "event_type": event_type,
            "search_id": search_id,
            "session_id": str(payload.get("session_id", "") or ""),
            "user_id": str(payload.get("user_id", "") or ""),
            "rank": int(rank) if rank is not None else None,
            "timestamp": timestamp,
            "timestamp_seconds": float(timestamp_seconds),
            "source_model_version": str(metadata.get("model_version", "") or ""),
        }
        rows.append(row)
        if search_id:
            positives_by_search[search_id].add(product_id)

    if include_unclicked_negatives:
        rows.extend(
            _negative_rows(
                exposures.values(),
                positives_by_search,
                catalog,
                require_catalog=require_catalog,
                max_negative_rank=max_negative_rank,
            )
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return _empty_feedback_frame()
    frame = frame.sort_values(["timestamp_seconds", "search_id", "product_id"]).reset_index(
        drop=True
    )
    frame["split"] = _time_ordered_split(frame, validation_ratio=validation_ratio)
    return frame[
        [
            "query",
            "query_key",
            "product_id",
            "label",
            "weight",
            "event_type",
            "split",
            "search_id",
            "session_id",
            "user_id",
            "rank",
            "timestamp",
            "timestamp_seconds",
            "source_model_version",
        ]
    ]


def feedback_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "positive_rows": 0,
            "negative_rows": 0,
            "queries": 0,
            "products": 0,
            "searches": 0,
            "splits": {},
        }
    positives = frame[frame["label"].astype(float) > 0]
    negatives = frame[frame["label"].astype(float) <= 0]
    return {
        "rows": int(len(frame)),
        "positive_rows": int(len(positives)),
        "negative_rows": int(len(negatives)),
        "queries": int(frame["query_key"].nunique()),
        "products": int(frame["product_id"].nunique()),
        "searches": int(frame["search_id"].replace("", pd.NA).dropna().nunique()),
        "splits": {
            str(key): int(value)
            for key, value in frame["split"].value_counts(dropna=False).sort_index().items()
        },
        "positive_weight": (
            float(positives["weight"].astype(float).sum()) if not positives.empty else 0.0
        ),
    }


def _is_search_exposure(payload: dict[str, Any], metadata: dict[str, Any]) -> bool:
    surface = str(metadata.get("source_surface") or metadata.get("surface") or "")
    return (
        str(payload.get("event_type", "") or "") == "search"
        and surface == "search"
        and str(metadata.get("event_role", "") or "") == "exposure"
    )


def _has_search_feedback_marker(
    metadata: dict[str, Any],
    exposure: dict[str, Any] | None,
) -> bool:
    surface = str(metadata.get("source_surface") or metadata.get("surface") or "")
    return surface == "search" or bool(metadata.get("search_id")) or exposure is not None


def _exposure_record(
    payload: dict[str, Any],
    metadata: dict[str, Any],
    order: int,
    timestamp_seconds: float,
) -> dict[str, Any]:
    search_id = str(metadata.get("search_id") or payload.get("event_id") or f"search-{order}")
    result_product_ids = _as_string_list(
        metadata.get("result_product_ids")
        or metadata.get("ranked_product_ids")
        or metadata.get("product_ids")
    )
    rank_by_product = {
        product_id: rank for rank, product_id in enumerate(result_product_ids, start=1)
    }
    rank_map = metadata.get("rank_map")
    if isinstance(rank_map, dict):
        for product_id, rank in rank_map.items():
            coerced = _coerce_int(rank, default=None)
            if coerced is not None:
                rank_by_product[str(product_id)] = int(coerced)
    query = str(payload.get("query", "") or metadata.get("query", "") or "")
    return {
        "search_id": search_id,
        "query": query,
        "query_key": _normalize_query_key(query),
        "result_product_ids": result_product_ids,
        "rank_by_product": rank_by_product,
        "session_id": str(payload.get("session_id", "") or ""),
        "user_id": str(payload.get("user_id", "") or ""),
        "timestamp": str(payload.get("timestamp", "") or ""),
        "timestamp_seconds": float(timestamp_seconds),
        "source_model_version": str(metadata.get("model_version", "") or ""),
    }


def _matching_exposure(
    payload: dict[str, Any],
    metadata: dict[str, Any],
    exposures: dict[str, dict[str, Any]],
    session_exposures: dict[str, list[dict[str, Any]]],
    timestamp_seconds: float,
    *,
    click_window_seconds: int,
) -> dict[str, Any] | None:
    search_id = str(metadata.get("search_id", "") or "")
    if search_id and search_id in exposures:
        return exposures[search_id]
    session_id = str(payload.get("session_id", "") or "")
    if not session_id:
        return None
    query_key = _normalize_query_key(
        str(payload.get("query", "") or metadata.get("query", "") or "")
    )
    candidates = session_exposures.get(session_id, [])
    for exposure in reversed(candidates):
        age = timestamp_seconds - float(exposure.get("timestamp_seconds", 0.0) or 0.0)
        if age < 0 or age > click_window_seconds:
            continue
        if query_key and query_key != exposure.get("query_key"):
            continue
        return exposure
    return None


def _negative_rows(
    exposures: Iterable[dict[str, Any]],
    positives_by_search: dict[str, set[str]],
    catalog: set[str],
    *,
    require_catalog: bool,
    max_negative_rank: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exposure in exposures:
        search_id = str(exposure.get("search_id", "") or "")
        positives = positives_by_search.get(search_id, set())
        for rank, product_id in enumerate(
            exposure.get("result_product_ids", [])[:max_negative_rank],
            start=1,
        ):
            product_id = str(product_id)
            if product_id in positives or (require_catalog and product_id not in catalog):
                continue
            rows.append(
                {
                    "query": str(exposure.get("query", "") or ""),
                    "query_key": str(exposure.get("query_key", "") or ""),
                    "product_id": product_id,
                    "label": 0,
                    "weight": 1,
                    "event_type": "impression",
                    "search_id": search_id,
                    "session_id": str(exposure.get("session_id", "") or ""),
                    "user_id": str(exposure.get("user_id", "") or ""),
                    "rank": rank,
                    "timestamp": str(exposure.get("timestamp", "") or ""),
                    "timestamp_seconds": float(exposure.get("timestamp_seconds", 0.0) or 0.0),
                    "source_model_version": str(exposure.get("source_model_version", "") or ""),
                }
            )
    return rows


def _time_ordered_split(frame: pd.DataFrame, *, validation_ratio: float) -> list[str]:
    validation_ratio = max(0.0, min(float(validation_ratio), 0.5))
    if validation_ratio <= 0.0 or frame.empty:
        return ["train"] * len(frame)
    search_ids = (
        frame[["search_id", "timestamp_seconds"]]
        .assign(search_id=lambda value: value["search_id"].replace("", pd.NA))
        .dropna(subset=["search_id"])
        .groupby("search_id", as_index=False)["timestamp_seconds"]
        .max()
        .sort_values("timestamp_seconds")
    )
    if len(search_ids) >= 5:
        valid_count = max(1, int(round(len(search_ids) * validation_ratio)))
        valid_searches = set(search_ids.tail(valid_count)["search_id"].astype(str))
        return [
            "valid" if str(search_id) in valid_searches else "train"
            for search_id in frame["search_id"].astype(str)
        ]
    if len(frame) < 20:
        return ["train"] * len(frame)
    valid_count = max(1, int(round(len(frame) * validation_ratio)))
    valid_start = len(frame) - valid_count
    return ["valid" if row_index >= valid_start else "train" for row_index in range(len(frame))]


def _metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [item.strip() for item in value.split(",")]
        value = parsed
    if not isinstance(value, Iterable):
        return []
    return [str(item) for item in value if str(item)]


def _coerce_int(value: Any, default: int | None = 0) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _timestamp_seconds(value: str, *, fallback: float) -> float:
    if not value:
        return fallback
    try:
        normalized = value.replace("Z", "+00:00")
        return float(datetime.fromisoformat(normalized).timestamp())
    except ValueError:
        return fallback


def _empty_feedback_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "query",
            "query_key",
            "product_id",
            "label",
            "weight",
            "event_type",
            "split",
            "search_id",
            "session_id",
            "user_id",
            "rank",
            "timestamp",
            "timestamp_seconds",
            "source_model_version",
        ]
    )
