from __future__ import annotations

import csv
import gzip
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mars.config.settings import MarsConfig, load_config
from mars.recommendation.models import (
    TwoTowerModel,
    fit_torch_two_tower_model,
    fit_torch_wide_deep_ranker,
    project_item_vectors_with_two_tower,
)
from mars.recommendation.session_encoder import fit_gru_session_encoder
from mars.retrieval import VectorIndex

ARTIFACT_FILE = "recommendation_artifacts.json.gz"
ITEM_INDEX_NAME = "items"


@dataclass(slots=True)
class RecommendationArtifacts:
    version: str
    embedding_dim: int
    products: list[dict[str, Any]]
    users: dict[str, dict[str, Any]]
    item_embeddings: list[list[float]]
    popularity_order: list[str]
    trending_order: list[str]
    item_index: dict[str, int]
    user_histories: dict[str, list[str]] = field(default_factory=dict)
    category_transitions: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    two_tower_model: dict[str, Any] | None = None
    ranking_model: dict[str, Any] | None = None
    session_encoder_model: dict[str, Any] | None = None
    training_events_source: str = ""
    training_event_count: int = 0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RecommendationArtifacts:
        products = [dict(product) for product in payload.get("products", [])]
        users_raw = payload.get("users", {})
        if isinstance(users_raw, list):
            users = {
                str(user.get("user_id")): dict(user) for user in users_raw if user.get("user_id")
            }
        else:
            users = {str(key): dict(value) for key, value in users_raw.items()}
        item_index = {str(product.get("product_id")): idx for idx, product in enumerate(products)}
        user_histories = {
            str(user_id): [str(product_id) for product_id in product_ids]
            for user_id, product_ids in payload.get("user_histories", {}).items()
        }
        return cls(
            version=str(payload.get("version", "unknown")),
            embedding_dim=int(payload.get("embedding_dim", 64)),
            products=products,
            users=users,
            item_embeddings=[list(map(float, row)) for row in payload.get("item_embeddings", [])],
            popularity_order=[str(value) for value in payload.get("popularity_order", [])],
            trending_order=[str(value) for value in payload.get("trending_order", [])],
            item_index=item_index,
            user_histories=user_histories,
            category_transitions={
                str(source): [dict(item) for item in targets]
                for source, targets in payload.get("category_transitions", {}).items()
            },
            two_tower_model=payload.get("two_tower_model"),
            ranking_model=payload.get("ranking_model"),
            session_encoder_model=payload.get("session_encoder_model"),
            training_events_source=str(payload.get("training_events_source", "")),
            training_event_count=int(payload.get("training_event_count", 0) or 0),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "embedding_dim": self.embedding_dim,
            "products": _json_safe(self.products),
            "users": _json_safe(self.users),
            "item_embeddings": _json_safe(self.item_embeddings),
            "popularity_order": self.popularity_order,
            "trending_order": self.trending_order,
            "user_histories": self.user_histories,
            "category_transitions": _json_safe(self.category_transitions),
            "two_tower_model": _json_safe(self.two_tower_model),
            "ranking_model": _json_safe(self.ranking_model),
            "session_encoder_model": _json_safe(self.session_encoder_model),
            "training_events_source": self.training_events_source,
            "training_event_count": self.training_event_count,
        }

    def product_by_id(self, product_id: str) -> dict[str, Any] | None:
        idx = self.item_index.get(product_id)
        if idx is None:
            return None
        return self.products[idx]


def artifact_path(config: MarsConfig) -> Path:
    return config.paths.artifacts_dir / "recsys" / ARTIFACT_FILE


def item_index_dir(config: MarsConfig, output_path: str | Path | None = None) -> Path:
    if output_path:
        return Path(output_path).parent
    return config.paths.artifacts_dir / "recsys"


def load_recommendation_artifacts(
    path: str | Path | None = None,
    config: MarsConfig | None = None,
) -> RecommendationArtifacts:
    config = config or load_config()
    target = Path(path) if path else artifact_path(config)
    with gzip.open(target, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return RecommendationArtifacts.from_payload(payload)


def save_recommendation_artifacts(
    artifacts: RecommendationArtifacts,
    path: str | Path | None = None,
    config: MarsConfig | None = None,
) -> Path:
    config = config or load_config()
    target = Path(path) if path else artifact_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with gzip.open(tmp_target, "wt", encoding="utf-8") as handle:
            json.dump(artifacts.to_payload(), handle, ensure_ascii=False)
        tmp_target.replace(target)
    finally:
        if tmp_target.exists():
            tmp_target.unlink()
    return target


def build_recommendation_artifacts(
    config: MarsConfig | None = None,
    processed_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    live_events_path: str | Path | None = None,
) -> RecommendationArtifacts:
    config = config or load_config()
    source_dir = Path(processed_dir) if processed_dir else config.paths.processed_dir
    products = _normalise_products(_load_table(source_dir / "products"))
    users = _normalise_users(_load_table(source_dir / "users"))
    events, events_source = _load_training_events(source_dir, live_events_path=live_events_path)

    popularity = _product_counts(events)
    user_histories = _user_histories(events)
    for product in products:
        product_id = str(product.get("product_id"))
        event_popularity = popularity.get(product_id)
        if event_popularity is not None:
            product["popularity_prior"] = event_popularity

    model = TwoTowerModel(
        embedding_dim=max(16, int(config.recommendation.embedding_dim)), seed=config.seed
    )
    recommendation_raw = (
        config.raw.get("recommendation", {}) if isinstance(config.raw, dict) else {}
    )
    two_tower_max_positive_samples = int(
        recommendation_raw.get("two_tower_max_positive_samples", 4_000) or 4_000
    )
    ranker_max_samples = int(recommendation_raw.get("ranker_max_samples", 20_000) or 20_000)
    session_encoder_max_samples = int(
        recommendation_raw.get("session_encoder_max_samples", 4_000) or 4_000
    )
    users_by_id = {str(user.get("user_id")): user for user in users}
    products_by_id = {str(product.get("product_id")): product for product in products}
    popularity_order = _rank_products(products, "popularity_prior")
    trending_order = _rank_products(products, "is_new", secondary="popularity_prior")
    category_transitions = _category_transitions(events, products_by_id)
    version = datetime.now(UTC).strftime("recsys-%Y%m%d%H%M%S")
    two_tower_model = fit_torch_two_tower_model(
        users=users_by_id,
        products_by_id=products_by_id,
        events=events,
        base_model=model,
        seed=config.seed,
        max_positive_samples=two_tower_max_positive_samples,
    )
    base_item_embeddings = [model.encode_item(product) for product in products]
    item_embeddings = project_item_vectors_with_two_tower(two_tower_model, base_item_embeddings)
    item_embeddings_by_product_id = {
        str(product.get("product_id")): vector
        for product, vector in zip(products, item_embeddings, strict=False)
        if product.get("product_id")
    }
    ranking_model = fit_torch_wide_deep_ranker(
        users=users_by_id,
        products_by_id=products_by_id,
        events=events,
        two_tower=model,
        seed=config.seed,
        max_samples=ranker_max_samples,
    )
    session_encoder_model = fit_gru_session_encoder(
        events=events,
        item_embeddings=item_embeddings_by_product_id,
        embedding_dim=model.embedding_dim,
        seed=config.seed,
        max_sequence_length=config.recommendation.session_recent_n,
        max_samples=session_encoder_max_samples,
    )
    artifacts = RecommendationArtifacts(
        version=version,
        embedding_dim=model.embedding_dim,
        products=products,
        users=users_by_id,
        item_embeddings=item_embeddings,
        popularity_order=popularity_order,
        trending_order=trending_order,
        item_index={str(product.get("product_id")): idx for idx, product in enumerate(products)},
        user_histories=user_histories,
        category_transitions=category_transitions,
        two_tower_model=two_tower_model,
        ranking_model=ranking_model,
        session_encoder_model=session_encoder_model,
        training_events_source=events_source,
        training_event_count=len(events),
    )
    save_recommendation_artifacts(artifacts, output_path, config)
    _save_item_vector_index(
        artifacts,
        item_index_dir(config, output_path),
        index_type=config.search.index_type,
    )
    return artifacts


def _load_training_events(
    source_dir: Path,
    live_events_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    train_events = _load_table(source_dir / "train_events")
    if train_events:
        events = train_events
        source = "train_events.parquet"
    else:
        events = _load_table(source_dir / "events")
        source = "events.parquet"
    if not live_events_path:
        return events, source
    live_path = Path(live_events_path)
    live_events = _load_table(live_path)
    if not live_events:
        return events, source
    return _dedupe_events([*events, *live_events]), f"{source}+{live_path.name}"


def _dedupe_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for event in events:
        event_id = str(event.get("event_id", "")).strip()
        identity = event_id or json.dumps(event, ensure_ascii=False, sort_keys=True, default=str)
        if identity in seen:
            continue
        seen.add(identity)
        output.append(event)
    return output


def load_item_vector_index(
    config: MarsConfig,
    output_path: str | Path | None = None,
) -> VectorIndex | None:
    directory = item_index_dir(config, output_path)
    manifest_path = directory / f"{ITEM_INDEX_NAME}_index.json"
    if not manifest_path.exists():
        return None
    return VectorIndex.load(directory, ITEM_INDEX_NAME)


def _save_item_vector_index(
    artifacts: RecommendationArtifacts,
    directory: str | Path,
    *,
    index_type: str,
) -> Path | None:
    if not artifacts.item_embeddings:
        return None
    vectors = np.asarray(artifacts.item_embeddings, dtype=np.float32)
    index = VectorIndex.build(vectors, index_type=index_type, prefer_faiss=True)
    return index.save(directory, ITEM_INDEX_NAME)


def _load_table(base_path: Path) -> list[dict[str, Any]]:
    candidates = [
        base_path.with_suffix(".parquet"),
        base_path.with_suffix(".jsonl"),
        base_path.with_suffix(".json"),
        base_path.with_suffix(".csv"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            try:
                import pandas as pd  # type: ignore

                return pd.read_parquet(path).to_dict(orient="records")
            except Exception:
                continue
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        if path.suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                return [dict(row) for row in payload]
            if isinstance(payload, dict) and "rows" in payload:
                return [dict(row) for row in payload["rows"]]
        if path.suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
    return []


def _normalise_products(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        product = dict(row)
        product.setdefault("product_id", f"P{idx + 1:08d}")
        product.setdefault("name", f"Product {idx + 1}")
        product.setdefault(
            "category", product.get("top_category", product.get("category_l1", "unknown"))
        )
        product.setdefault(
            "category_l1", product.get("top_category", product.get("category", "unknown"))
        )
        product.setdefault(
            "category_l2", product.get("mid_category", product.get("category_l1", "unknown"))
        )
        product.setdefault(
            "category_l3", product.get("leaf_category", product.get("category_l2", "unknown"))
        )
        product["price"] = float(product.get("price", 0) or 0)
        product["popularity_prior"] = float(
            product.get("popularity_prior", product.get("popularity", 0.0)) or 0.0
        )
        product["margin_score"] = float(product.get("margin_score", 0.0) or 0.0)
        product["is_new"] = _to_bool(product.get("is_new", False))
        products.append(product)
    return products


def _normalise_users(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    users: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        user = dict(row)
        user.setdefault("user_id", f"U{idx + 1:08d}")
        user.setdefault("persona", "cold")
        user.setdefault(
            "preferred_categories",
            _list_like(user.get("preferred_categories", user.get("preferred_top_categories", []))),
        )
        user["price_sensitivity"] = float(user.get("price_sensitivity", 0.5) or 0.5)
        users.append(user)
    return users


def _product_counts(events: list[dict[str, Any]]) -> dict[str, float]:
    weights = {"view": 0.2, "cart": 0.7, "purchase": 1.0, "search": 0.05}
    counts: dict[str, float] = {}
    for event in events:
        product_id = event.get("product_id")
        if not product_id:
            continue
        counts[str(product_id)] = counts.get(str(product_id), 0.0) + weights.get(
            str(event.get("event_type")), 0.1
        )
    if not counts:
        return {}
    max_count = max(counts.values()) or 1.0
    return {product_id: value / max_count for product_id, value in counts.items()}


def _user_histories(
    events: list[dict[str, Any]], limit_per_user: int = 500
) -> dict[str, list[str]]:
    weights = {"view": 0.2, "cart": 0.7, "purchase": 1.0, "search": 0.0}
    scores: dict[str, dict[str, float]] = {}
    order_bonus = 0.0
    for event in events:
        user_id = event.get("user_id")
        product_id = event.get("product_id")
        if not user_id or not product_id:
            continue
        event_weight = weights.get(str(event.get("event_type")), 0.0)
        if event_weight <= 0:
            continue
        bucket = scores.setdefault(str(user_id), {})
        order_bonus += 1e-9
        bucket[str(product_id)] = bucket.get(str(product_id), 0.0) + event_weight + order_bonus

    histories: dict[str, list[str]] = {}
    for user_id, product_scores in scores.items():
        ranked = sorted(product_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
        histories[user_id] = [product_id for product_id, _ in ranked[:limit_per_user]]
    return histories


def _category_transitions(
    events: list[dict[str, Any]],
    products_by_id: dict[str, dict[str, Any]],
    *,
    max_targets: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    session_events: dict[str, list[dict[str, Any]]] = {}
    for idx, event in enumerate(events):
        if str(event.get("event_type")) not in {"view", "cart", "purchase"}:
            continue
        product_id = str(event.get("product_id") or "")
        product = products_by_id.get(product_id)
        if not product:
            continue
        category = _category_key(product)
        if not category:
            continue
        session_id = str(event.get("session_id") or event.get("user_id") or "global")
        row = dict(event)
        row["_category_key"] = category
        row["_order"] = idx
        session_events.setdefault(session_id, []).append(row)

    counts: dict[str, dict[str, int]] = {}
    for rows in session_events.values():
        ordered = sorted(
            rows,
            key=lambda row: (
                str(row.get("timestamp", "")),
                str(row.get("event_id", "")),
                int(row.get("_order", 0)),
            ),
        )
        previous = ""
        for row in ordered:
            current = str(row.get("_category_key", ""))
            if previous and current:
                bucket = counts.setdefault(previous, {})
                bucket[current] = bucket.get(current, 0) + 1
            previous = current

    transitions: dict[str, list[dict[str, Any]]] = {}
    for source, targets in counts.items():
        total = sum(targets.values()) or 1
        ranked = sorted(targets.items(), key=lambda item: (item[1], item[0]), reverse=True)
        transitions[source] = [
            {
                "category": target,
                "probability": count / total,
                "count": count,
            }
            for target, count in ranked[:max_targets]
        ]
    return transitions


def _category_key(product: dict[str, Any]) -> str:
    for key in (
        "category_l3",
        "leaf_category",
        "category_l2",
        "mid_category",
        "category_l1",
        "category",
    ):
        value = product.get(key)
        if value:
            return str(value)
    return ""


def _rank_products(
    products: list[dict[str, Any]], primary: str, secondary: str = "popularity_prior"
) -> list[str]:
    ranked = sorted(
        products,
        key=lambda product: (
            float(product.get(primary, 0.0) or 0.0),
            float(product.get(secondary, 0.0) or 0.0),
            str(product.get("product_id", "")),
        ),
        reverse=True,
    )
    return [str(product.get("product_id")) for product in ranked]


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _list_like(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        return [part.strip(" '\"") for part in text.strip("[]").split(",") if part.strip()]
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]
