from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - optional dependency surface for final Docker runtime
    import torch
    from torch import nn
except Exception:  # pragma: no cover - local Python may not have torch installed
    torch = None
    nn = None


def _stable_float(token: str) -> float:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big") / float(2**64 - 1)
    return (value * 2.0) - 1.0


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        return vector
    return [value / norm for value in vector]


def _hash_embedding(tokens: Iterable[str], dim: int, prefix: str) -> list[float]:
    vector = [0.0] * dim
    count = 0
    for raw_token in tokens:
        token = str(raw_token).strip().lower()
        if not token:
            continue
        count += 1
        for idx in range(dim):
            vector[idx] += _stable_float(f"{prefix}:{token}:{idx}")
    if count == 0:
        return [0.0] * dim
    return _normalise([value / count for value in vector])


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


RANKER_FEATURE_NAMES = [
    "candidate_score",
    "category_match",
    "recent_category_match",
    "recent_product_match",
    "popularity",
    "margin",
    "is_new",
    "price_fit",
    "price_sensitivity",
    "persona_trendsetter",
    "persona_value_seeker",
    "persona_top_category_loyalist",
    "persona_impulse_buyer",
    "persona_careful_explorer",
    "persona_pragmatist",
]


@dataclass(slots=True)
class TwoTowerModel:
    """Feature-hash base tower used as deterministic input features for trained towers."""

    embedding_dim: int = 64
    seed: int = 42

    def encode_user(
        self,
        user: dict[str, Any] | None,
        session_context: dict[str, Any] | None = None,
    ) -> list[float]:
        user = user or {}
        session_context = session_context or {}
        tokens: list[str] = [
            f"user:{user.get('user_id', 'unknown')}",
            f"persona:{user.get('persona', 'cold')}",
            f"age:{user.get('age_bucket', '')}",
            f"gender:{user.get('gender', '')}",
        ]
        tokens.extend(f"category:{value}" for value in _as_list(user.get("preferred_categories")))
        tokens.extend(
            f"recent_category:{value}" for value in session_context.get("recent_categories", [])
        )
        tokens.extend(
            f"recent_product:{value}" for value in session_context.get("recent_products", [])
        )
        tokens.extend(
            f"recent_category_pos:{position}:{value}"
            for position, value in enumerate(session_context.get("recent_categories", []))
        )
        tokens.extend(
            f"recent_product_pos:{position}:{value}"
            for position, value in enumerate(session_context.get("recent_products", []))
        )
        return _hash_embedding(tokens, self.embedding_dim, f"user:{self.seed}")

    def encode_item(self, product: dict[str, Any]) -> list[float]:
        tokens: list[str] = [
            f"product:{product.get('product_id', '')}",
            f"name:{product.get('name', '')}",
            f"category:{product.get('category_l1', product.get('category', ''))}",
            f"category2:{product.get('category_l2', '')}",
            f"color:{product.get('color', '')}",
            f"new:{product.get('is_new', False)}",
            f"price_bucket:{_price_bucket(product.get('price', 0.0))}",
        ]
        tokens.extend(f"tag:{value}" for value in _as_list(product.get("style_tags")))
        return _hash_embedding(tokens, self.embedding_dim, f"item:{self.seed}")

    def score(self, user_vector: list[float], item_vector: list[float]) -> float:
        return dot(user_vector, item_vector)


class TorchTwoTower(nn.Module if nn else object):  # pragma: no cover - exercised in Docker runtime
    def __init__(self, embedding_dim: int = 64) -> None:
        if nn:
            super().__init__()
            self.user_projection = nn.Linear(embedding_dim, embedding_dim)
            self.item_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, user_features: Any, item_features: Any) -> Any:
        if not nn:
            raise RuntimeError("PyTorch is not installed")
        user = self.project_user(user_features)
        item = self.project_item(item_features)
        return (user * item).sum(dim=-1) * 8.0

    def project_user(self, user_features: Any) -> Any:
        if not nn:
            raise RuntimeError("PyTorch is not installed")
        return torch.nn.functional.normalize(self.user_projection(user_features), dim=-1)

    def project_item(self, item_features: Any) -> Any:
        if not nn:
            raise RuntimeError("PyTorch is not installed")
        return torch.nn.functional.normalize(self.item_projection(item_features), dim=-1)


@dataclass(slots=True)
class TrainedTwoTowerModel:
    """Serving wrapper for a trained PyTorch Two-Tower projection checkpoint."""

    embedding_dim: int = 64
    seed: int = 42
    model_payload: dict[str, Any] | None = None
    _base_model: TwoTowerModel = field(init=False, repr=False)
    _torch_model: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._base_model = TwoTowerModel(embedding_dim=self.embedding_dim, seed=self.seed)
        self._torch_model = _load_torch_two_tower(self.model_payload)

    def encode_user(
        self,
        user: dict[str, Any] | None,
        session_context: dict[str, Any] | None = None,
    ) -> list[float]:
        base_vector = self._base_model.encode_user(user, session_context)
        return _project_single_vector(self._torch_model, base_vector, tower="user")

    def encode_item(self, product: dict[str, Any]) -> list[float]:
        base_vector = self._base_model.encode_item(product)
        return _project_single_vector(self._torch_model, base_vector, tower="item")

    def score(self, user_vector: list[float], item_vector: list[float]) -> float:
        return dot(user_vector, item_vector)


@dataclass(slots=True)
class WideDeepRanker:
    """Wide&Deep ranker with a PyTorch checkpoint when one is available."""

    seed: int = 42
    model_payload: dict[str, Any] | None = None
    model_blend_weight: float = 0.65
    _torch_model: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._torch_model = _load_torch_wide_deep(self.model_payload)
        if self.model_payload and "model_blend_weight" in self.model_payload:
            self.model_blend_weight = float(self.model_payload["model_blend_weight"])

    def score(
        self,
        user: dict[str, Any] | None,
        product: dict[str, Any],
        candidate_score: float,
        session_context: dict[str, Any] | None = None,
    ) -> float:
        return self.score_many(user, [(product, candidate_score)], session_context)[0]

    def score_many(
        self,
        user: dict[str, Any] | None,
        product_scores: list[tuple[dict[str, Any], float]],
        session_context: dict[str, Any] | None = None,
    ) -> list[float]:
        features = [
            ranker_feature_vector(user, product, candidate_score, session_context or {})
            for product, candidate_score in product_scores
        ]
        heuristic_scores = [
            _heuristic_score(user, product, candidate_score, session_context or {})
            for product, candidate_score in product_scores
        ]
        if not features or self._torch_model is None or torch is None:
            return heuristic_scores
        try:
            with torch.inference_mode():
                tensor = torch.tensor(features, dtype=torch.float32)
                learned_scores = self._torch_model(tensor).detach().cpu().tolist()
        except Exception:
            return heuristic_scores
        blend = min(max(float(self.model_blend_weight), 0.0), 1.0)
        return [
            float((blend * float(learned)) + ((1.0 - blend) * float(heuristic)))
            for learned, heuristic in zip(learned_scores, heuristic_scores, strict=False)
        ]


def ranker_feature_vector(
    user: dict[str, Any] | None,
    product: dict[str, Any],
    candidate_score: float,
    session_context: dict[str, Any] | None = None,
) -> list[float]:
    user = user or {}
    session_context = session_context or {}
    category = str(product.get("category_l1", product.get("category", "")))
    preferred_categories = set(map(str, _as_list(user.get("preferred_categories"))))
    recent_categories = set(map(str, session_context.get("recent_categories", [])))
    recent_products = set(map(str, session_context.get("recent_products", [])))
    popularity = float(product.get("popularity_prior", product.get("popularity", 0.0)) or 0.0)
    margin = float(product.get("margin_score", 0.0) or 0.0)
    is_new = 1.0 if product.get("is_new", False) else 0.0
    price = float(product.get("price", 0) or 0.0)
    price_sensitivity = float(user.get("price_sensitivity", 0.5) or 0.5)
    price_fit = max(0.0, 1.0 - min(price, 300_000.0) / 300_000.0) * price_sensitivity
    persona = str(user.get("persona", "cold"))
    return [
        float(candidate_score),
        1.0 if category in preferred_categories else 0.0,
        1.0 if category in recent_categories else 0.0,
        1.0 if product.get("product_id") in recent_products else 0.0,
        popularity,
        margin,
        is_new,
        price_fit,
        price_sensitivity,
        1.0 if persona == "trendsetter" else 0.0,
        1.0 if persona == "value_seeker" else 0.0,
        1.0 if persona == "top_category_loyalist" else 0.0,
        1.0 if persona == "impulse_buyer" else 0.0,
        1.0 if persona in {"careful_explorer", "careful_researcher"} else 0.0,
        1.0 if persona == "pragmatist" else 0.0,
    ]


def _heuristic_score(
    user: dict[str, Any] | None,
    product: dict[str, Any],
    candidate_score: float,
    session_context: dict[str, Any] | None = None,
) -> float:
    user = user or {}
    session_context = session_context or {}
    category = str(product.get("category_l1", product.get("category", "")))
    preferred_categories = set(map(str, _as_list(user.get("preferred_categories"))))
    recent_categories = set(map(str, session_context.get("recent_categories", [])))
    recent_products = set(map(str, session_context.get("recent_products", [])))

    popularity = float(product.get("popularity_prior", product.get("popularity", 0.0)) or 0.0)
    margin = float(product.get("margin_score", 0.0) or 0.0)
    is_new = 1.0 if product.get("is_new", False) else 0.0
    price = float(product.get("price", 0) or 0.0)
    price_sensitivity = float(user.get("price_sensitivity", 0.5) or 0.5)
    price_fit = max(0.0, 1.0 - min(price, 300_000.0) / 300_000.0) * price_sensitivity

    wide = 0.0
    wide += 0.35 if category in preferred_categories else 0.0
    wide += 0.20 if category in recent_categories else 0.0
    wide += 0.15 if product.get("product_id") in recent_products else 0.0

    persona = str(user.get("persona", "cold"))
    persona_bonus = {
        "trendsetter": 0.18 * is_new + 0.10 * popularity,
        "bargain_hunter": 0.22 * price_fit,
        "value_seeker": 0.22 * price_fit,
        "top_category_loyalist": 0.18 if category in preferred_categories else 0.0,
        "impulse_buyer": 0.12 * popularity + 0.08 * margin,
        "careful_researcher": 0.12 if category in preferred_categories else 0.0,
        "careful_explorer": 0.12 if category in preferred_categories else 0.0,
        "pragmatist": 0.10 * price_fit + 0.10 * popularity,
    }.get(persona, 0.08 * popularity)

    deep = 1.4 * candidate_score + 0.35 * popularity + 0.15 * margin + 0.18 * price_fit
    return sigmoid(deep + wide + persona_bonus)


class TorchWideDeep(nn.Module if nn else object):  # pragma: no cover - optional placeholder
    def __init__(self, feature_dim: int = 32) -> None:
        if nn:
            super().__init__()
            self.wide = nn.Linear(feature_dim, 1)
            self.deep = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, features: Any) -> Any:
        if not nn:
            raise RuntimeError("PyTorch is not installed")
        return torch.sigmoid(self.wide(features) + self.deep(features)).squeeze(-1)


def fit_torch_wide_deep_ranker(
    *,
    users: dict[str, dict[str, Any]],
    products_by_id: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
    two_tower: TwoTowerModel,
    seed: int,
    max_samples: int = 20_000,
) -> dict[str, Any] | None:
    if torch is None or nn is None:
        return None
    _stabilise_windows_torch_training()
    rng = random.Random(seed)
    positives: list[tuple[list[float], float]] = []
    negatives: list[tuple[list[float], float]] = []
    positive_contexts: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    item_vectors: dict[str, list[float]] = {}
    product_ids = list(products_by_id)
    clicked_exposure_ids = _clicked_exposure_ids(events)

    for event, session_context in _sampled_events_with_context(
        events,
        rng,
        max_scan=max(max_samples * 20, max_samples),
    ):
        event_type = str(event.get("event_type", ""))
        metadata = _event_metadata(event)
        event_role = str(metadata.get("event_role", ""))
        exposure_id = str(metadata.get("exposure_id", ""))
        user_id = str(event.get("user_id", ""))
        product_id = str(event.get("product_id", ""))
        user = users.get(user_id)
        product = products_by_id.get(product_id)
        if not user or not product:
            continue
        if event_role == "exposure":
            label = 1.0 if exposure_id and exposure_id in clicked_exposure_ids else 0.0
        elif event_role == "response" and exposure_id:
            continue
        elif event_type in {"view", "cart", "purchase"}:
            label = 1.0
        else:
            continue
        user_vector = two_tower.encode_user(user, session_context)
        item_vector = item_vectors.setdefault(product_id, two_tower.encode_item(product))
        candidate_score = sigmoid(two_tower.score(user_vector, item_vector))
        features = ranker_feature_vector(user, product, candidate_score, session_context)
        if label > 0:
            positives.append((features, label))
            positive_contexts.append((user, session_context, product_id))
        else:
            negatives.append((features, label))
        if len(positives) >= max_samples // 5 and len(negatives) >= (max_samples // 5) * 4:
            break

    negative_target = min(max_samples, max(1, len(positives) * 4))
    for user, session_context, positive_product_id in positive_contexts:
        attempts = 0
        while len(negatives) < negative_target and attempts < 20:
            attempts += 1
            negative_product_id = rng.choice(product_ids)
            if negative_product_id == positive_product_id:
                continue
            product = products_by_id.get(negative_product_id)
            if not product:
                continue
            user_vector = two_tower.encode_user(user, session_context)
            item_vector = item_vectors.setdefault(
                negative_product_id,
                two_tower.encode_item(product),
            )
            candidate_score = sigmoid(two_tower.score(user_vector, item_vector))
            negatives.append(
                (
                    ranker_feature_vector(user, product, candidate_score, session_context),
                    0.0,
                )
            )
        if len(negatives) >= negative_target:
            break

    if not positives or not negatives:
        return None
    rng.shuffle(positives)
    rng.shuffle(negatives)
    positive_limit = min(len(positives), max(1, max_samples // 5))
    negative_limit = min(len(negatives), positive_limit * 4)
    samples = positives[:positive_limit] + negatives[:negative_limit]
    rng.shuffle(samples)

    x = torch.tensor([features for features, _ in samples], dtype=torch.float32)
    y = torch.tensor([label for _, label in samples], dtype=torch.float32)
    torch.manual_seed(seed)
    model = TorchWideDeep(feature_dim=len(RANKER_FEATURE_NAMES))
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.BCELoss()
    generator = torch.Generator().manual_seed(seed)
    batch_size = min(512, max(64, len(samples)))
    for _epoch in range(5):
        order = torch.randperm(len(samples), generator=generator)
        for start in range(0, len(samples), batch_size):
            idx = order[start : start + batch_size]
            prediction = model(x[idx])
            loss = loss_fn(prediction, y[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    state = {key: value.detach().cpu().tolist() for key, value in model.state_dict().items()}
    return {
        "model_type": "torch_wide_deep",
        "feature_names": RANKER_FEATURE_NAMES,
        "feature_dim": len(RANKER_FEATURE_NAMES),
        "state_dict": state,
        "trained_samples": len(samples),
        "positive_samples": positive_limit,
        "negative_samples": negative_limit,
        "negative_sampling": (
            "explicit exposure non-click negatives with random unclicked fallback, capped near 1:4"
        ),
        "label_definition": "view/cart/purchase response or clicked exposure is positive",
        "model_blend_weight": 0.65,
        "seed": seed,
    }


def fit_torch_two_tower_model(
    *,
    users: dict[str, dict[str, Any]],
    products_by_id: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
    base_model: TwoTowerModel,
    seed: int,
    max_positive_samples: int = 4_000,
    negatives_per_positive: int = 4,
) -> dict[str, Any] | None:
    if torch is None or nn is None:
        return None
    _stabilise_windows_torch_training()
    product_ids = [str(product_id) for product_id in products_by_id]
    if not users or not product_ids:
        return None

    rng = random.Random(seed)
    positives: list[tuple[str, str, dict[str, Any]]] = []
    for event, session_context in _sampled_events_with_context(
        events,
        rng,
        max_scan=max(max_positive_samples * 80, max_positive_samples),
    ):
        event_type = str(event.get("event_type", ""))
        if event_type not in {"cart", "purchase"}:
            continue
        user_id = str(event.get("user_id", ""))
        product_id = str(event.get("product_id", ""))
        if user_id not in users or product_id not in products_by_id:
            continue
        positives.append((user_id, product_id, session_context))
        if len(positives) >= max_positive_samples:
            break
    if not positives:
        return None

    item_vectors: dict[str, list[float]] = {}
    samples: list[tuple[list[float], list[float], float]] = []
    for user_id, positive_product_id, session_context in positives:
        user = users[user_id]
        positive_product = products_by_id[positive_product_id]
        user_vector = base_model.encode_user(user, session_context)
        positive_item_vector = item_vectors.setdefault(
            positive_product_id, base_model.encode_item(positive_product)
        )
        samples.append((user_vector, positive_item_vector, 1.0))

        negative_count = 0
        attempts = 0
        while negative_count < negatives_per_positive and attempts < negatives_per_positive * 20:
            attempts += 1
            negative_product_id = rng.choice(product_ids)
            if negative_product_id == positive_product_id:
                continue
            negative_product = products_by_id.get(negative_product_id)
            if not negative_product:
                continue
            negative_item_vector = item_vectors.setdefault(
                negative_product_id, base_model.encode_item(negative_product)
            )
            samples.append((user_vector, negative_item_vector, 0.0))
            negative_count += 1

    if len(samples) <= len(positives):
        return None

    rng.shuffle(samples)
    x_user = torch.tensor([user_vec for user_vec, _, _ in samples], dtype=torch.float32)
    x_item = torch.tensor([item_vec for _, item_vec, _ in samples], dtype=torch.float32)
    y = torch.tensor([label for _, _, label in samples], dtype=torch.float32)

    torch.manual_seed(seed)
    model = TorchTwoTower(embedding_dim=base_model.embedding_dim)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    batch_size = min(512, max(64, len(samples)))
    last_loss = 0.0
    for _epoch in range(4):
        order = torch.randperm(len(samples), generator=generator)
        for start in range(0, len(samples), batch_size):
            idx = order[start : start + batch_size]
            logits = model(x_user[idx], x_item[idx])
            loss = loss_fn(logits, y[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            last_loss = float(loss.detach().cpu().item())

    state = {key: value.detach().cpu().tolist() for key, value in model.state_dict().items()}
    return {
        "model_type": "torch_two_tower",
        "base_model": "feature_hash_two_tower",
        "embedding_dim": base_model.embedding_dim,
        "state_dict": state,
        "trained_samples": len(samples),
        "positive_samples": len(positives),
        "negative_samples": len(samples) - len(positives),
        "negative_sampling": f"random negatives 1:{negatives_per_positive}",
        "objective": (
            "binary cross entropy over contextual cart/purchase pairs and random negatives"
        ),
        "epochs": 4,
        "final_loss": last_loss,
        "seed": seed,
    }


def project_item_vectors_with_two_tower(
    payload: dict[str, Any] | None,
    item_vectors: list[list[float]],
    *,
    batch_size: int = 2048,
) -> list[list[float]]:
    return _project_vectors(payload, item_vectors, tower="item", batch_size=batch_size)


def _sampled_events_with_context(
    events: list[dict[str, Any]],
    rng: random.Random,
    *,
    max_scan: int,
    max_recent: int = 20,
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    selected_indices = (
        set(range(len(events)))
        if len(events) <= max_scan
        else set(rng.sample(range(len(events)), max_scan))
    )
    states: dict[str, dict[str, Any]] = {}
    for index, event in enumerate(events):
        session_id = str(event.get("session_id") or event.get("user_id") or "global")
        state = states.setdefault(
            session_id,
            {
                "products": [],
                "categories": [],
                "event_counts": {},
                "num_recent_events": 0,
            },
        )
        if index in selected_indices:
            context = {
                "recent_products": _recent_unique(state["products"], max_recent),
                "recent_categories": _recent_unique(state["categories"], max_recent),
                "event_counts": dict(state["event_counts"]),
                "num_recent_events": int(state["num_recent_events"]),
                "query": str(event.get("query", "") or ""),
                "query_intent_category": str(
                    event.get("query_intent_category", event.get("category", "")) or ""
                ),
            }
            yield event, context
        _update_training_state(state, event, max_recent=max_recent)


def _update_training_state(
    state: dict[str, Any],
    event: dict[str, Any],
    *,
    max_recent: int,
) -> None:
    metadata = _event_metadata(event)
    if str(metadata.get("event_role", "")) == "exposure":
        return
    event_type = str(event.get("event_type", ""))
    counts = state["event_counts"]
    counts[event_type] = counts.get(event_type, 0) + 1
    state["num_recent_events"] = int(state["num_recent_events"]) + 1
    product_id = str(event.get("product_id", ""))
    if event_type in {"view", "cart", "purchase"} and product_id and product_id.lower() != "nan":
        state["products"].append(product_id)
    category = str(event.get("category", event.get("query_intent_category", "")) or "")
    if category and category.lower() != "nan":
        state["categories"].append(category)
    state["products"] = state["products"][-max_recent * 3 :]
    state["categories"] = state["categories"][-max_recent * 3 :]


def _recent_unique(values: list[Any], limit: int) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in reversed(values):
        item = str(value)
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
        if len(output) >= limit:
            break
    return output


def _event_metadata(event: dict[str, Any]) -> dict[str, Any]:
    metadata = event.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _clicked_exposure_ids(events: list[dict[str, Any]]) -> set[str]:
    clicked: set[str] = set()
    for event in events:
        metadata = _event_metadata(event)
        exposure_id = str(metadata.get("exposure_id", ""))
        if (
            exposure_id
            and str(metadata.get("event_role", "")) == "response"
            and str(event.get("event_type", "")) in {"view", "cart", "purchase"}
        ):
            clicked.add(exposure_id)
    return clicked


def _load_torch_wide_deep(payload: dict[str, Any] | None) -> Any:
    if torch is None or nn is None or not payload:
        return None
    if payload.get("model_type") != "torch_wide_deep":
        return None
    try:
        model = TorchWideDeep(
            feature_dim=int(payload.get("feature_dim", len(RANKER_FEATURE_NAMES)))
        )
        state_dict = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in dict(payload.get("state_dict", {})).items()
        }
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception:
        return None


def _load_torch_two_tower(payload: dict[str, Any] | None) -> Any:
    if torch is None or nn is None or not payload:
        return None
    if payload.get("model_type") != "torch_two_tower":
        return None
    try:
        model = TorchTwoTower(embedding_dim=int(payload.get("embedding_dim", 64)))
        state_dict = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in dict(payload.get("state_dict", {})).items()
        }
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception:
        return None


def _stabilise_windows_torch_training() -> None:
    if torch is None or not sys.platform.startswith("win"):
        return
    try:
        if torch.get_num_threads() > 1:
            torch.set_num_threads(1)
    except Exception:
        pass
    try:
        if torch.get_num_interop_threads() > 1:
            torch.set_num_interop_threads(1)
    except Exception:
        pass


def _project_single_vector(model: Any, vector: list[float], *, tower: str) -> list[float]:
    if model is None or torch is None:
        return _normalise(vector)
    try:
        with torch.inference_mode():
            tensor = torch.tensor([vector], dtype=torch.float32)
            projected = (
                model.project_user(tensor) if tower == "user" else model.project_item(tensor)
            )
            return projected[0].detach().cpu().tolist()
    except Exception:
        return _normalise(vector)


def _project_vectors(
    payload: dict[str, Any] | None,
    vectors: list[list[float]],
    *,
    tower: str,
    batch_size: int,
) -> list[list[float]]:
    model = _load_torch_two_tower(payload)
    if model is None or torch is None or not vectors:
        return [_normalise(list(map(float, vector))) for vector in vectors]
    projected_vectors: list[list[float]] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(vectors), max(int(batch_size), 1)):
                batch = torch.tensor(vectors[start : start + batch_size], dtype=torch.float32)
                projected = (
                    model.project_user(batch) if tower == "user" else model.project_item(batch)
                )
                projected_vectors.extend(projected.detach().cpu().tolist())
        return projected_vectors
    except Exception:
        return [_normalise(list(map(float, vector))) for vector in vectors]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            return [part.strip(" '\"") for part in stripped.strip("[]").split(",") if part.strip()]
        if "|" in stripped:
            return [part.strip() for part in stripped.split("|") if part.strip()]
        if "," in stripped:
            return [part.strip() for part in stripped.split(",") if part.strip()]
    return [value]


def _price_bucket(value: Any) -> str:
    try:
        price = max(float(value or 0.0), 0.0)
    except (TypeError, ValueError):
        price = 0.0
    boundaries = (10_000, 20_000, 30_000, 50_000, 80_000, 120_000, 200_000)
    for boundary in boundaries:
        if price < boundary:
            return f"under_{boundary}"
    return "200000_plus"
