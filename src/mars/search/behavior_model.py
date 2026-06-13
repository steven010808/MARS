from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mars.config import MarsConfig
from mars.search.qrels import qrels_split_settings, select_qrels_split
from mars.search.service import _as_product_ids, _normalize_query_key, _query_tokens


def build_query_behavior_model_payload(
    config: MarsConfig,
    *,
    feedback_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    raw_search = config.raw.get("search", {}) if isinstance(config.raw, dict) else {}
    query_top_k = int(raw_search.get("query_prior_top_k", 10) or 10)
    token_top_k = int(raw_search.get("query_token_prior_top_k", 1200) or 1200)
    qrels_path = config.paths.processed_dir / "search_queries.parquet"
    product_meta_path = (
        Path(metadata_path)
        if metadata_path
        else config.paths.artifacts_dir / "search" / "product_meta.parquet"
    )

    queries = pd.read_parquet(
        qrels_path,
        columns=["query_id", "query", "positive_product_ids"],
    )
    train = select_qrels_split(queries, config, "train")
    metadata = pd.read_parquet(product_meta_path, columns=["product_id"])
    catalog = set(metadata["product_id"].astype(str))

    query_counts: dict[str, Counter[str]] = defaultdict(Counter)
    token_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in train.itertuples(index=False):
        product_ids = [
            product_id
            for product_id in _as_product_ids(row.positive_product_ids)
            if product_id in catalog
        ]
        _add_query_product_counts(
            query_counts,
            token_counts,
            query=str(row.query),
            product_ids=product_ids,
            weight=1,
        )

    feedback_stats = _add_feedback_counts(
        config,
        query_counts,
        token_counts,
        catalog,
        feedback_path=feedback_path,
    )

    query_prior = {
        key: [product_id for product_id, _count in counts.most_common(query_top_k)]
        for key, counts in query_counts.items()
    }
    query_token_prior = {
        token: [[product_id, int(count)] for product_id, count in counts.most_common(token_top_k)]
        for token, counts in token_counts.items()
    }
    seed, train_ratio, valid_ratio = qrels_split_settings(config)
    query_ids = sorted(train["query_id"].astype(str).tolist())
    return {
        "schema_version": "search-query-behavior.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "split": "train",
        "split_strategy": "query_id_hash",
        "seed": seed,
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "train_rows": int(len(train)),
        "train_query_ids_sha256": hashlib.sha256("\n".join(query_ids).encode()).hexdigest(),
        "catalog_products": int(len(catalog)),
        "query_prior_top_k": query_top_k,
        "query_token_prior_top_k": token_top_k,
        "query_prior": query_prior,
        "query_token_prior": query_token_prior,
        "live_feedback": feedback_stats,
    }


def write_query_behavior_model(payload: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    return path


def behavior_model_summary(
    payload: dict[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    summary = {
        "output": str(output_path) if output_path else "",
        "train_rows": int(payload.get("train_rows", 0) or 0),
        "query_keys": len(payload.get("query_prior", {}) or {}),
        "token_keys": len(payload.get("query_token_prior", {}) or {}),
        "live_feedback": payload.get("live_feedback", {}),
    }
    if output_path:
        path = Path(output_path)
        if path.exists():
            summary["size_bytes"] = path.stat().st_size
    return summary


def _add_feedback_counts(
    config: MarsConfig,
    query_counts: dict[str, Counter[str]],
    token_counts: dict[str, Counter[str]],
    catalog: set[str],
    *,
    feedback_path: str | Path | None,
) -> dict[str, Any]:
    raw_search = config.raw.get("search", {}) if isinstance(config.raw, dict) else {}
    online = raw_search.get("online_learning", {})
    online = online if isinstance(online, dict) else {}
    if not bool(online.get("enabled", False)):
        return {"enabled": False, "rows": 0, "train_positive_rows": 0, "valid_rows": 0}

    configured_path = feedback_path or online.get("feedback_path")
    if not configured_path:
        return {"enabled": True, "path": "", "rows": 0, "train_positive_rows": 0, "valid_rows": 0}
    path = Path(str(configured_path))
    if not path.exists():
        return {
            "enabled": True,
            "path": str(path),
            "rows": 0,
            "train_positive_rows": 0,
            "valid_rows": 0,
            "missing": True,
        }

    frame = _read_feedback_frame(path)
    if frame.empty:
        return {
            "enabled": True,
            "path": str(path),
            "rows": 0,
            "train_positive_rows": 0,
            "valid_rows": 0,
            "sha256": _file_sha256(path),
        }

    split = _series_or_default(frame, "split", "train").astype(str)
    labels = _series_or_default(frame, "label", 0).astype(float)
    train = frame[(split != "valid") & (labels > 0)]
    weight_multiplier = float(online.get("weight_multiplier", 1.0) or 1.0)
    min_weight = float(online.get("min_positive_weight", 1.0) or 1.0)
    max_weight = int(online.get("max_query_product_weight", 25) or 25)
    for row in train.itertuples(index=False):
        query = str(getattr(row, "query", "") or "")
        product_id = str(getattr(row, "product_id", "") or "")
        if not query or product_id not in catalog:
            continue
        raw_weight = float(getattr(row, "weight", 1.0) or 1.0) * weight_multiplier
        if raw_weight < min_weight:
            continue
        weight = max(1, min(int(round(raw_weight)), max_weight))
        _add_query_product_counts(
            query_counts,
            token_counts,
            query=query,
            product_ids=[product_id],
            weight=weight,
        )

    valid_rows = int((split == "valid").sum())
    return {
        "enabled": True,
        "path": str(path),
        "rows": int(len(frame)),
        "train_positive_rows": int(len(train)),
        "valid_rows": valid_rows,
        "sha256": _file_sha256(path),
        "weight_multiplier": weight_multiplier,
        "min_positive_weight": min_weight,
        "max_query_product_weight": max_weight,
    }


def _add_query_product_counts(
    query_counts: dict[str, Counter[str]],
    token_counts: dict[str, Counter[str]],
    *,
    query: str,
    product_ids: list[str],
    weight: int,
) -> None:
    key = _normalize_query_key(query)
    if not key:
        return
    for product_id in product_ids:
        query_counts[key][product_id] += int(weight)
        for token in _query_tokens(query):
            token_counts[token][product_id] += int(weight)


def _read_feedback_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.DataFrame()


def _series_or_default(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in frame:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
