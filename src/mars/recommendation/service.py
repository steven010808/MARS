from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mars.config.settings import MarsConfig, RecommendationStrategyConfig, load_config
from mars.recommendation.artifacts import (
    RecommendationArtifacts,
    artifact_path,
    build_recommendation_artifacts,
    load_item_vector_index,
    load_recommendation_artifacts,
)
from mars.recommendation.models import TrainedTwoTowerModel, TwoTowerModel, WideDeepRanker
from mars.recommendation.rerank import EpsilonGreedyMAB, rerank_candidates
from mars.recommendation.session import InMemorySessionStore, SessionStore
from mars.recommendation.session_encoder import GRUSessionEncoder, combine_user_and_session_vectors
from mars.retrieval import VectorIndex


@dataclass
class RecommendationService:
    config: MarsConfig = field(default_factory=load_config)
    artifacts: RecommendationArtifacts | None = None
    session_store: SessionStore | None = None
    two_tower: TwoTowerModel | TrainedTwoTowerModel | None = None
    ranker: WideDeepRanker | None = None
    item_vector_index: VectorIndex | None = None
    session_encoder: GRUSessionEncoder | None = None
    mab: EpsilonGreedyMAB = field(default_factory=EpsilonGreedyMAB)

    def __post_init__(self) -> None:
        if self.artifacts is None:
            self.artifacts = self._load_or_build_artifacts()
        dim = (
            self.artifacts.embedding_dim
            if self.artifacts
            else max(16, self.config.recommendation.embedding_dim)
        )
        self.two_tower = self.two_tower or TrainedTwoTowerModel(
            embedding_dim=dim,
            seed=self.config.seed,
            model_payload=self.artifacts.two_tower_model if self.artifacts else None,
        )
        self.ranker = self.ranker or WideDeepRanker(
            seed=self.config.seed,
            model_payload=self.artifacts.ranking_model if self.artifacts else None,
        )
        self.item_vector_index = self.item_vector_index or self._load_or_build_item_vector_index()
        self.session_encoder = self.session_encoder or GRUSessionEncoder(
            embedding_dim=dim,
            seed=self.config.seed,
            max_sequence_length=self.config.recommendation.session_recent_n,
            model_payload=self.artifacts.session_encoder_model if self.artifacts else None,
        )
        self.session_store = self.session_store or InMemorySessionStore(
            recent_n=self.config.recommendation.session_recent_n
        )
        self._item_embeddings_by_product_id = self._build_item_embedding_lookup()
        self._products_by_category = self._build_products_by_category()
        self._trending_products = self._build_trending_products(limit=5_000)
        self._catalog_products = sorted(
            self.artifacts.products if self.artifacts else [],
            key=lambda product: str(product.get("product_id", "")),
        )
        self._product_token_weights, self._products_by_token = self._build_product_token_index()

    def recommend(
        self,
        user_id: str,
        top_n: int | None = None,
        session_id: str | None = None,
        request_id: str = "",
        strategy: str = "treatment",
    ) -> dict[str, Any]:
        top_n = max(1, min(int(top_n or self.config.recommendation.final_top_n), 100))
        strategy = _normalise_strategy(strategy)
        strategy_config = _strategy_config(self.config, strategy)
        started = time.perf_counter()

        stage_started = time.perf_counter()
        user = self.artifacts.users.get(user_id) if self.artifacts else None
        session_context = (
            self.session_store.get_context(user_id, session_id) if self.session_store else {}
        )
        session_context = _with_session_aliases(session_context)
        history_count = self._history_count(user_id, session_context)
        session_context["history_count"] = history_count
        session_context["cold_start"] = bool(user is None or history_count < 5)
        scoring_context = _strategy_scoring_context(session_context, strategy, strategy_config)
        transition_categories = (
            self._transition_categories(scoring_context)
            if _strategy_uses_transitions(strategy_config)
            else []
        )
        scoring_context["transition_categories"] = transition_categories
        candidates = self.generate_candidates(user_id, scoring_context)
        candidate_ms = _elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        ranked = self.rank_candidates(user, candidates, scoring_context)
        ranking_ms = _elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        reranked = self.rerank(
            user_id,
            ranked,
            top_n,
            request_id,
            strategy=strategy,
            user=user,
            session_context=scoring_context,
        )
        reranking_ms = _elapsed_ms(stage_started)
        session_context["recommendation_strategy"] = strategy
        session_context["recommendation_strategy_label"] = strategy_config.label
        session_context["strategy_mix"] = _strategy_mix(strategy_config, top_n)
        session_context["session_features_enabled"] = strategy != "control"
        for key in ("session_encoder", "transition_categories"):
            if key in scoring_context and strategy != "control":
                session_context[key] = scoring_context[key]
        for key in ("repeat_explore_gate", "strategy_mix_effective"):
            if key in scoring_context:
                session_context[key] = scoring_context[key]

        recommendations = [
            {
                "product_id": str(item["product"].get("product_id")),
                "name": str(item["product"].get("name", "")),
                "category": str(
                    item["product"].get("category_l1", item["product"].get("category", ""))
                ),
                "price": float(item["product"].get("price", 0.0) or 0.0),
                "score": round(float(item.get("ranking_score", 0.0)), 6),
                "reason": str(item.get("reason") or _reason_for(user, item)),
                "is_exploration": bool(item.get("is_exploration", False)),
                "arm": str(item.get("arm")) if item.get("arm") else None,
            }
            for item in reranked
        ]
        total_ms = _elapsed_ms(started)
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "pipeline_latency": {
                "candidate_ms": round(candidate_ms, 3),
                "ranking_ms": round(ranking_ms, 3),
                "reranking_ms": round(reranking_ms, 3),
                "total_ms": round(total_ms, 3),
            },
            "session_context": session_context,
        }

    def generate_candidates(
        self,
        user_id: str,
        session_context: dict[str, Any] | None = None,
        candidate_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.artifacts or not self.artifacts.products:
            return []
        candidate_k = int(candidate_k or self.config.recommendation.candidate_k)
        session_context = session_context or {}
        user = self.artifacts.users.get(user_id)
        history_ids = self.artifacts.user_histories.get(user_id, []) if self.artifacts else []
        history_count = self._history_count(user_id, session_context)
        if user is None or history_count < 5:
            return self._cold_start_candidates(
                candidate_k,
                user=user,
                session_context=session_context,
                history_count=history_count,
            )

        recent_product_ids = list(map(str, session_context.get("recent_products", [])))
        recent_products = set(recent_product_ids)
        seed_products: list[dict[str, Any]] = []
        ann_scores: dict[str, float] = {}
        query_scores: dict[str, float] = {}
        query_tokens = _query_tokens_from_context(session_context)
        ann_seed_limit = min(
            len(self.artifacts.products),
            max(candidate_k, int(candidate_k * 1.25)),
        )
        for product, ann_score in self._ann_seed_products(user, session_context, ann_seed_limit):
            product_id = str(product.get("product_id"))
            ann_scores[product_id] = max(ann_scores.get(product_id, -1.0), ann_score)
            seed_products.append(product)
        for product_id in reversed(
            recent_product_ids[-self.config.recommendation.session_recent_n :]
        ):
            product = self.artifacts.product_by_id(product_id) if self.artifacts else None
            if product:
                seed_products.append(product)
        for product, query_score in self._query_seed_products(
            query_tokens,
            limit=max(candidate_k * 2, 128),
        ):
            product_id = str(product.get("product_id"))
            query_scores[product_id] = max(query_scores.get(product_id, 0.0), query_score)
            seed_products.append(product)
        target_categories = _ordered_unique(
            [
                *map(str, _as_list(user.get("preferred_categories"))),
                *map(str, session_context.get("recent_categories", [])),
                *map(str, session_context.get("transition_categories", [])),
            ]
        )
        history_id_set = set(history_ids)
        history_rank = {product_id: idx for idx, product_id in enumerate(history_ids)}
        for product_id in history_ids[:candidate_k]:
            product = self.artifacts.product_by_id(product_id) if self.artifacts else None
            if product:
                seed_products.append(product)
        category_seed_limit = max(candidate_k, 64)
        for category in target_categories:
            seed_products.extend(self._products_by_category.get(category, [])[:category_seed_limit])
        popularity_seed_limit = max(candidate_k * 2, 64)
        for product_id in (self.artifacts.popularity_order if self.artifacts else [])[
            :popularity_seed_limit
        ]:
            product = self.artifacts.product_by_id(product_id) if self.artifacts else None
            if product:
                seed_products.append(product)
        trending_seed_limit = max(candidate_k // 2, 32)
        for product_id in (self.artifacts.trending_order if self.artifacts else [])[
            :trending_seed_limit
        ]:
            product = self.artifacts.product_by_id(product_id) if self.artifacts else None
            if product:
                seed_products.append(product)

        if not seed_products:
            seed_products = list(self.artifacts.products)

        scored: list[dict[str, Any]] = []
        seen_product_ids: set[str] = set()
        for product in seed_products:
            product_id = str(product.get("product_id"))
            if product_id in seen_product_ids:
                continue
            seen_product_ids.add(product_id)
            ann_score = ann_scores.get(product_id)
            score = _candidate_score(user, product, session_context)
            if ann_score is not None:
                score += 0.85 * ann_score
            query_score = query_scores.get(product_id)
            if query_score is not None:
                score += 2.40 * query_score
            if product_id in history_id_set:
                rank = history_rank.get(product_id, candidate_k)
                score += 0.3 + 1.2 * (1.0 - min(rank, candidate_k) / max(candidate_k, 1))
            if str(product.get("product_id")) in recent_products:
                score += 1.45 + self.config.recommendation.session_weight * 0.4
                reason = "session_recent_item"
            elif query_score is not None:
                reason = "query_intent_match"
            else:
                reason = _candidate_reason(self.item_vector_index, ann_score)
            scored.append(
                {
                    "product": product,
                    "candidate_score": score,
                    "query_score": float(query_score or 0.0),
                    "reason": reason,
                    "is_exploration": False,
                }
            )
        scored.sort(key=lambda item: item["candidate_score"], reverse=True)
        target = min(candidate_k, len(scored))
        return scored[:target]

    def rank_candidates(
        self,
        user: dict[str, Any] | None,
        candidates: list[dict[str, Any]],
        session_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        session_context = session_context or {}
        recent_products = set(map(str, session_context.get("recent_products", [])))
        recent_categories = set(map(str, session_context.get("recent_categories", [])))
        ranked: list[dict[str, Any]] = []
        product_scores = [
            (item["product"], float(item.get("candidate_score", 0.0))) for item in candidates
        ]
        scores = (
            self.ranker.score_many(user, product_scores, session_context) if self.ranker else []
        )
        denominator = max(len(candidates) - 1, 1)
        for candidate_rank, (item, ranking_score) in enumerate(
            zip(candidates, scores, strict=False)
        ):
            product = item["product"]
            product_id = str(product.get("product_id"))
            category = str(product.get("category_l1", product.get("category", "")))
            candidate_rank_prior = 1.0 - (candidate_rank / denominator)
            ranking_score = (0.20 * float(ranking_score)) + (0.80 * candidate_rank_prior)
            if product_id in recent_products or item.get("reason") == "session_recent_item":
                ranking_score += 2.25
            elif float(item.get("query_score", 0.0) or 0.0) > 0:
                ranking_score += 0.90 * float(item.get("query_score", 0.0) or 0.0)
            elif category in recent_categories:
                ranking_score += 0.35
            ranked.append(
                {
                    **item,
                    "ranking_score": ranking_score,
                    "candidate_rank_prior": candidate_rank_prior,
                }
            )
        ranked.sort(key=lambda item: item["ranking_score"], reverse=True)
        return ranked

    def rerank(
        self,
        user_id: str,
        ranked: list[dict[str, Any]],
        top_n: int,
        request_id: str = "",
        strategy: str = "treatment",
        user: dict[str, Any] | None = None,
        session_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        strategy_key = _normalise_strategy(strategy)
        strategy_config = _strategy_config(self.config, strategy_key)
        if strategy_key == "baseline_vanilla":
            return rerank_candidates(
                ranked,
                top_n=top_n,
                exploration_slots=min(self.config.recommendation.exploration_slots, top_n),
                max_same_category_streak=self.config.recommendation.max_same_category_streak,
                trending_products=self._trending_products,
                mab=self.mab,
                user_id=user_id,
                request_id=request_id,
            )
        return _strategy_slot_rerank(
            ranked,
            top_n=top_n,
            strategy_key=strategy_key,
            strategy_config=strategy_config,
            max_same_category_streak=self.config.recommendation.max_same_category_streak,
            trending_products=self._trending_products,
            catalog_products=self._catalog_products,
            mab=self.mab,
            user_id=user_id,
            request_id=request_id,
            user=user,
            session_context=session_context or {},
            artifacts=self.artifacts,
        )

    def update_event(self, event: dict[str, Any]) -> dict[str, Any]:
        arm = event.get("arm") or event.get("metadata", {}).get("arm")
        if arm and event.get("event_type") in {"cart", "purchase"}:
            self.mab.update(str(arm), 1.0)
        elif arm and event.get("event_type") == "view":
            self.mab.update(str(arm), 0.2)
        if self.session_store:
            return _with_session_aliases(self.session_store.update_event(event))
        return {}

    def _fallback_candidates(self, candidate_k: int) -> list[dict[str, Any]]:
        return self._cold_start_candidates(
            candidate_k, user=None, session_context={}, history_count=0
        )

    def _cold_start_candidates(
        self,
        candidate_k: int,
        *,
        user: dict[str, Any] | None,
        session_context: dict[str, Any] | None,
        history_count: int,
    ) -> list[dict[str, Any]]:
        if not self.artifacts:
            return []
        session_context = session_context or {}
        preferred_categories = set(map(str, _as_list((user or {}).get("preferred_categories"))))
        recent_categories = set(map(str, session_context.get("recent_categories", [])))
        recent_product_ids = list(map(str, session_context.get("recent_products", [])))
        target_categories = preferred_categories | recent_categories
        if user and recent_product_ids:
            self._combined_user_vector(user, session_context)

        score_by_product: dict[str, float] = {}
        reason_by_product: dict[str, str] = {}

        def add_order(product_ids: list[str], reason: str, base_weight: float) -> None:
            limit = max(candidate_k * 3, 96)
            for rank, product_id in enumerate(product_ids[:limit]):
                product_id = str(product_id)
                product = self.artifacts.product_by_id(product_id)
                if not product:
                    continue
                popularity = float(
                    product.get("popularity_prior", product.get("popularity", 0.0)) or 0.0
                )
                category = str(product.get("category_l1", product.get("category", "")))
                category_boost = 0.12 if category in target_categories else 0.0
                new_boost = 0.10 if product.get("is_new", False) else 0.0
                score = base_weight
                score += max(0.0, 1.0 - rank / max(limit, 1)) * 0.45
                score += popularity * 0.35
                score += category_boost + new_boost
                if user:
                    price_sensitivity = float(user.get("price_sensitivity", 0.5) or 0.5)
                    budget_max = float(user.get("budget_max", 0.0) or 0.0)
                    price = float(product.get("price", 0.0) or 0.0)
                    if budget_max > 0:
                        price_fit = max(
                            0.0, 1.0 - max(price - budget_max, 0.0) / max(budget_max, 1.0)
                        )
                    else:
                        price_fit = max(0.0, 1.0 - min(price, 300_000.0) / 300_000.0)
                    score += 0.10 * price_fit * price_sensitivity
                if score > score_by_product.get(product_id, -1.0):
                    score_by_product[product_id] = score
                    reason_by_product[product_id] = reason

        pure_new_user = user is None and not recent_product_ids and not recent_categories
        add_order(
            list(self.artifacts.popularity_order),
            "cold_start_popularity",
            0.90 if pure_new_user else 0.66,
        )
        add_order(
            list(self.artifacts.trending_order),
            "cold_start_trending",
            0.55 if pure_new_user else 0.62,
        )
        add_order(recent_product_ids, "session_recent_item", 0.78)
        new_product_ids = [
            str(product.get("product_id"))
            for product in self.artifacts.products
            if product.get("is_new", False) and product.get("product_id")
        ]
        add_order(new_product_ids, "cold_start_new_item", 0.50)
        for category in sorted(target_categories):
            products = self._products_by_category.get(category, [])
            add_order(
                [
                    str(product.get("product_id"))
                    for product in products
                    if product.get("product_id")
                ],
                "cold_start_category_preference",
                0.48,
            )

        if not score_by_product:
            add_order(
                [str(product.get("product_id")) for product in self.artifacts.products],
                "cold_start_popularity",
                0.25,
            )

        product_ids = [
            product_id
            for product_id, _score in sorted(
                score_by_product.items(),
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )
        ][:candidate_k]
        candidates = []
        for rank, product_id in enumerate(product_ids):
            product = self.artifacts.product_by_id(product_id) if self.artifacts else None
            if not product:
                continue
            base_score = score_by_product.get(product_id, 0.0)
            reason = reason_by_product.get(product_id, "cold_start_popularity")
            candidates.append(
                {
                    "product": product,
                    "candidate_score": base_score
                    + max(0.0, 1.0 - (rank / max(candidate_k, 1))) * 0.05,
                    "reason": reason,
                    "is_exploration": False,
                    "history_count": history_count,
                }
            )
        return candidates

    def _history_count(self, user_id: str, session_context: dict[str, Any] | None = None) -> int:
        history_ids = self.artifacts.user_histories.get(user_id, []) if self.artifacts else []
        recent_products = list(map(str, (session_context or {}).get("recent_products", [])))
        return len({*map(str, history_ids), *recent_products})

    def _build_products_by_category(self) -> dict[str, list[dict[str, Any]]]:
        if not self.artifacts:
            return {}
        buckets: dict[str, list[dict[str, Any]]] = {}
        for product in self.artifacts.products:
            for category in _product_category_keys(product):
                buckets.setdefault(category, []).append(product)
        for products in buckets.values():
            products.sort(
                key=lambda product: (
                    float(product.get("popularity_prior", 0.0) or 0.0),
                    float(product.get("margin_score", 0.0) or 0.0),
                    str(product.get("product_id", "")),
                ),
                reverse=True,
            )
        return buckets

    def _transition_categories(self, session_context: dict[str, Any]) -> list[str]:
        if not self.artifacts or not self.artifacts.category_transitions:
            return []
        source_categories = _current_category_keys(session_context, self.artifacts)
        scores: dict[str, float] = {}
        for source in source_categories:
            for item in self.artifacts.category_transitions.get(source, []):
                category = str(item.get("category", ""))
                if not category or category in source_categories:
                    continue
                probability = float(item.get("probability", 0.0) or 0.0)
                scores[category] = max(scores.get(category, 0.0), probability)
        return [
            category
            for category, _score in sorted(
                scores.items(), key=lambda item: (item[1], item[0]), reverse=True
            )[:6]
        ]

    def _build_trending_products(self, *, limit: int) -> list[dict[str, Any]]:
        if not self.artifacts:
            return []
        products: list[dict[str, Any]] = []
        for product_id in self.artifacts.trending_order[:limit]:
            product = self.artifacts.product_by_id(product_id)
            if product:
                products.append(product)
        return products

    def _build_product_token_index(
        self,
    ) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
        token_weights: dict[str, dict[str, float]] = {}
        token_index: dict[str, list[str]] = {}
        if not self.artifacts:
            return token_weights, token_index
        for product in self.artifacts.products:
            product_id = str(product.get("product_id", ""))
            if not product_id:
                continue
            weights = _product_token_weights(product)
            token_weights[product_id] = weights
            for token, weight in weights.items():
                if weight <= 0:
                    continue
                token_index.setdefault(token, []).append(product_id)
        # Keep each posting list bounded and stable. Popularity prior is already folded into
        # downstream scoring, so deterministic product id order is enough here.
        for token, product_ids in token_index.items():
            token_index[token] = product_ids[:2500]
        return token_weights, token_index

    def _query_seed_products(
        self,
        query_tokens: list[str],
        *,
        limit: int,
    ) -> list[tuple[dict[str, Any], float]]:
        if not self.artifacts or not query_tokens:
            return []
        score_by_product: dict[str, float] = {}
        for token in query_tokens:
            for product_id in self._products_by_token.get(token, [])[:1200]:
                product_weights = self._product_token_weights.get(product_id, {})
                score_by_product[product_id] = score_by_product.get(
                    product_id, 0.0
                ) + product_weights.get(token, 0.0)
        if not score_by_product:
            return []
        adjusted_scores: dict[str, float] = {}
        for product_id, score in score_by_product.items():
            product = self.artifacts.product_by_id(product_id)
            popularity = float((product or {}).get("popularity_prior", 0.0) or 0.0)
            adjusted_scores[product_id] = score + (0.65 * popularity)
        max_score = max(adjusted_scores.values()) or 1.0
        ranked = sorted(
            adjusted_scores.items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )[:limit]
        output: list[tuple[dict[str, Any], float]] = []
        for product_id, score in ranked:
            product = self.artifacts.product_by_id(product_id)
            if product:
                output.append((product, min(1.0, float(score) / max_score)))
        return output

    def _build_item_embedding_lookup(self) -> dict[str, list[float]]:
        if not self.artifacts:
            return {}
        lookup: dict[str, list[float]] = {}
        for product, vector in zip(
            self.artifacts.products,
            self.artifacts.item_embeddings,
            strict=False,
        ):
            product_id = str(product.get("product_id", ""))
            if product_id and vector:
                lookup[product_id] = list(map(float, vector))
        return lookup

    def _load_or_build_artifacts(self) -> RecommendationArtifacts:
        path = artifact_path(self.config)
        if path.exists():
            return load_recommendation_artifacts(path, self.config)
        if _has_simulator_outputs(self.config.paths.processed_dir):
            return build_recommendation_artifacts(self.config)
        return RecommendationArtifacts(
            version="empty",
            embedding_dim=max(16, self.config.recommendation.embedding_dim),
            products=[],
            users={},
            item_embeddings=[],
            popularity_order=[],
            trending_order=[],
            item_index={},
            user_histories={},
            two_tower_model=None,
            ranking_model=None,
            session_encoder_model=None,
        )

    def _load_or_build_item_vector_index(self) -> VectorIndex | None:
        if not self.artifacts or not self.artifacts.item_embeddings:
            return None
        try:
            loaded = load_item_vector_index(self.config)
            if loaded and len(loaded.vectors) == len(self.artifacts.products):
                return loaded
        except Exception:
            pass
        vectors = np.asarray(self.artifacts.item_embeddings, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(self.artifacts.products):
            return None
        return VectorIndex.build(
            vectors,
            index_type=self.config.search.index_type,
            prefer_faiss=True,
        )

    def _ann_seed_products(
        self,
        user: dict[str, Any],
        session_context: dict[str, Any],
        top_k: int,
    ) -> list[tuple[dict[str, Any], float]]:
        if not self.artifacts or not self.item_vector_index:
            return []
        if not self.artifacts.products:
            return []
        user_vector = np.asarray(
            self._combined_user_vector(user, session_context), dtype=np.float32
        )
        if user_vector.size == 0:
            return []
        indices, scores = self.item_vector_index.search(
            user_vector, min(top_k, len(self.artifacts.products))
        )
        output: list[tuple[dict[str, Any], float]] = []
        for idx, raw_score in zip(indices.tolist(), scores.tolist(), strict=False):
            if idx < 0 or idx >= len(self.artifacts.products):
                continue
            product = self.artifacts.products[int(idx)]
            # Cosine/IP scores may be negative; map to a stable 0..1 boost for ranking.
            ann_score = max(0.0, min(1.0, (float(raw_score) + 1.0) / 2.0))
            output.append((product, ann_score))
        return output

    def _combined_user_vector(
        self,
        user: dict[str, Any],
        session_context: dict[str, Any],
    ) -> list[float]:
        if not self.two_tower:
            return []
        long_term_vector = self.two_tower.encode_user(user, {})
        recent_products = list(map(str, session_context.get("recent_products", [])))
        session_result = (
            self.session_encoder.encode(recent_products, self._item_embeddings_by_product_id)
            if self.session_encoder
            else None
        )
        if not session_result:
            if recent_products or session_context.get("recent_categories"):
                session_context["session_encoder"] = {
                    "type": "gru",
                    "sequence_length": 0,
                    "status": "no_matching_item_embeddings",
                    "long_term_weight": _strategy_long_term_weight(session_context, self.config),
                    "session_weight": _strategy_session_weight(session_context, self.config),
                }
                return self.two_tower.encode_user(user, session_context)
            session_context["session_encoder"] = {
                "type": "gru",
                "sequence_length": 0,
                "status": "empty_session",
                "long_term_weight": _strategy_long_term_weight(session_context, self.config),
                "session_weight": _strategy_session_weight(session_context, self.config),
            }
            return long_term_vector

        session_context["session_encoder"] = {
            "type": session_result.encoder_type,
            "sequence_length": session_result.sequence_length,
            "source_products": session_result.source_products[-5:],
            "long_term_weight": _strategy_long_term_weight(session_context, self.config),
            "session_weight": _strategy_session_weight(session_context, self.config),
        }
        return combine_user_and_session_vectors(
            long_term_vector,
            session_result.vector,
            long_term_weight=_strategy_long_term_weight(session_context, self.config),
            session_weight=_strategy_session_weight(session_context, self.config),
        )


def _has_simulator_outputs(processed_dir: Path) -> bool:
    return any(
        (processed_dir / f"products{suffix}").exists()
        for suffix in (".parquet", ".jsonl", ".json", ".csv")
    )


def _with_session_aliases(session_context: dict[str, Any]) -> dict[str, Any]:
    context = dict(session_context or {})
    recent_products = list(map(str, context.get("recent_products", [])))
    recent_categories = list(map(str, context.get("recent_categories", [])))
    context.setdefault("recent_clicks", recent_products)
    context.setdefault("session_interest", recent_categories[0] if recent_categories else "")
    return context


def _normalise_strategy(value: str | None) -> str:
    normalised = str(value or "").strip().lower().replace("-", "_")
    if normalised == "control":
        return "control"
    if normalised in {"baseline", "baseline_vanilla", "legacy", "legacy_vanilla", "vanilla"}:
        return "baseline_vanilla"
    return "treatment"


def _strategy_config(config: MarsConfig, strategy: str) -> RecommendationStrategyConfig:
    configured = config.recommendation.strategies.get(strategy)
    if configured:
        return configured
    if strategy == "baseline_vanilla":
        return RecommendationStrategyConfig(
            label="BaselineVanilla",
            current_category_slots=0,
            transition_slots=0,
            long_term_slots=0,
            exploration_slots=config.recommendation.exploration_slots,
            long_term_weight=config.recommendation.long_term_weight,
            session_weight=config.recommendation.session_weight,
            transition_boost=0.0,
            adaptive_gate=False,
            adaptive_margin=0.15,
        )
    if strategy == "control":
        return RecommendationStrategyConfig(
            label="RankOnlyControl",
            current_category_slots=0,
            transition_slots=0,
            long_term_slots=10,
            exploration_slots=0,
            long_term_weight=1.0,
            session_weight=0.0,
            transition_boost=0.0,
        )
    return RecommendationStrategyConfig(
        label="ComplementGraphExplore",
        current_category_slots=5,
        transition_slots=2,
        long_term_slots=1,
        exploration_slots=2,
        long_term_weight=0.55,
        session_weight=0.45,
        transition_boost=0.55,
        adaptive_gate=False,
        adaptive_margin=0.15,
    )


def _strategy_uses_transitions(strategy_config: RecommendationStrategyConfig) -> bool:
    return bool(
        int(strategy_config.transition_slots) > 0
        or float(strategy_config.transition_boost) > 0.0
        or bool(strategy_config.adaptive_gate)
    )


def _strategy_scoring_context(
    session_context: dict[str, Any],
    strategy: str,
    strategy_config: RecommendationStrategyConfig,
) -> dict[str, Any]:
    context = dict(session_context)
    if strategy == "control":
        context["recent_products"] = []
        context["recent_categories"] = []
        context["recent_clicks"] = []
        context["session_interest"] = ""
        context["event_counts"] = {}
        context["num_recent_events"] = 0
    context["_recommendation_strategy_key"] = strategy
    context["_strategy_label"] = strategy_config.label
    context["_strategy_long_term_weight"] = strategy_config.long_term_weight
    context["_strategy_session_weight"] = strategy_config.session_weight
    return context


def _strategy_long_term_weight(session_context: dict[str, Any], config: MarsConfig) -> float:
    return float(
        session_context.get("_strategy_long_term_weight", config.recommendation.long_term_weight)
    )


def _strategy_session_weight(session_context: dict[str, Any], config: MarsConfig) -> float:
    return float(
        session_context.get("_strategy_session_weight", config.recommendation.session_weight)
    )


def _strategy_mix(strategy_config: RecommendationStrategyConfig, top_n: int) -> dict[str, Any]:
    slots = _scaled_slots(strategy_config, top_n)
    return {
        "label": strategy_config.label,
        "slots": slots,
        "long_term_weight": strategy_config.long_term_weight,
        "session_weight": strategy_config.session_weight,
        "transition_boost": strategy_config.transition_boost,
        "adaptive_gate": strategy_config.adaptive_gate,
        "adaptive_margin": strategy_config.adaptive_margin,
    }


def _strategy_slot_rerank(
    ranked: list[dict[str, Any]],
    *,
    top_n: int,
    strategy_key: str,
    strategy_config: RecommendationStrategyConfig,
    max_same_category_streak: int,
    trending_products: list[dict[str, Any]],
    catalog_products: list[dict[str, Any]],
    mab: EpsilonGreedyMAB,
    user_id: str,
    request_id: str,
    user: dict[str, Any] | None,
    session_context: dict[str, Any],
    artifacts: RecommendationArtifacts | None,
) -> list[dict[str, Any]]:
    transition_scores = _transition_scores(session_context, artifacts)
    current_categories = set(_current_category_keys(session_context, artifacts))
    long_term_categories = set(map(str, _as_list((user or {}).get("preferred_categories"))))
    base_slots = _scaled_slots(strategy_config, top_n)
    gate = _repeat_explore_gate_decision(
        session_context,
        user=user,
        transition_scores=transition_scores,
        artifacts=artifacts,
        adaptive_margin=strategy_config.adaptive_margin,
    )
    slots = _adaptive_slots(base_slots, strategy_config, gate, top_n)
    if strategy_config.adaptive_gate:
        session_context["repeat_explore_gate"] = gate
        session_context["strategy_mix_effective"] = {
            **_strategy_mix(strategy_config, top_n),
            "slots": slots,
            "base_slots": base_slots,
            "adaptive_mode": gate["mode"],
        }

    pending = [_with_transition_score(item, transition_scores, strategy_config) for item in ranked]
    pending.sort(key=lambda item: float(item.get("ranking_score", 0.0) or 0.0), reverse=True)
    selected: list[dict[str, Any]] = []
    exploration_slots = min(top_n, slots["exploration_slots"])
    relevance_slots = max(0, top_n - exploration_slots)

    if strategy_key == "treatment":
        _append_ranked_prefix(
            selected,
            pending,
            limit=min(relevance_slots, max(1, int(round(top_n * 0.20)))),
            reason=f"{strategy_config.label}:protected_relevance",
        )

    _pick_slot_items(
        selected,
        pending,
        limit=min(slots["current_category_slots"], max(0, relevance_slots - len(selected))),
        max_same_category_streak=max_same_category_streak,
        predicate=lambda item: bool(
            current_categories & set(_product_category_keys(item["product"]))
        ),
        reason=f"{strategy_config.label}:current_intent",
    )
    _pick_slot_items(
        selected,
        pending,
        limit=min(slots["transition_slots"], max(0, relevance_slots - len(selected))),
        max_same_category_streak=max_same_category_streak,
        predicate=lambda item: float(item.get("transition_score", 0.0) or 0.0) > 0,
        reason=f"{strategy_config.label}:category_transition",
    )
    _pick_slot_items(
        selected,
        pending,
        limit=min(slots["long_term_slots"], max(0, relevance_slots - len(selected))),
        max_same_category_streak=max_same_category_streak,
        predicate=lambda item: (
            str(item["product"].get("category_l1", item["product"].get("category", "")))
            in long_term_categories
        ),
        reason=f"{strategy_config.label}:long_term_preference",
    )
    _pick_slot_items(
        selected,
        pending,
        limit=max(0, relevance_slots - len(selected)),
        max_same_category_streak=max_same_category_streak,
        predicate=lambda _item: True,
        reason=f"{strategy_config.label}:ranked_fill",
    )
    if strategy_key == "treatment" and relevance_slots > 0:
        _ensure_new_item_exposure(
            selected,
            trending_products,
            top_n=relevance_slots,
            max_same_category_streak=max_same_category_streak,
            strategy_label=strategy_config.label,
        )
    catalog_pool = _catalog_exploration_pool(
        catalog_products,
        rotation_token=f"{strategy_key}:{user_id}:{request_id}",
        limit=max(128, exploration_slots * 24),
    )
    _append_exploration_items(
        selected,
        exploration_slots,
        top_n=top_n,
        max_same_category_streak=max_same_category_streak,
        exploration_products=[
            *catalog_pool,
            *[item["product"] for item in pending],
            *trending_products[:128],
        ],
        mab=mab,
        user_id=user_id,
        request_id=f"{strategy_key}:{request_id}",
        strategy_label=strategy_config.label,
        user=user,
    )
    _pick_slot_items(
        selected,
        pending,
        limit=max(0, top_n - len(selected)),
        max_same_category_streak=max_same_category_streak,
        predicate=lambda _item: True,
        reason=f"{strategy_config.label}:ranked_fill",
    )
    for item in selected:
        item.setdefault("is_exploration", False)
        item.setdefault("arm", strategy_config.label)
    return selected[:top_n]


def _scaled_slots(strategy_config: RecommendationStrategyConfig, top_n: int) -> dict[str, int]:
    raw_slots = {
        "current_category_slots": max(0, int(strategy_config.current_category_slots)),
        "transition_slots": max(0, int(strategy_config.transition_slots)),
        "long_term_slots": max(0, int(strategy_config.long_term_slots)),
        "exploration_slots": max(0, int(strategy_config.exploration_slots)),
    }
    total = sum(raw_slots.values()) or 1
    if total == top_n:
        return raw_slots
    exact = {key: (value / total) * top_n for key, value in raw_slots.items()}
    scaled = {key: int(value) for key, value in exact.items()}
    remaining = max(0, top_n - sum(scaled.values()))
    for key, _value in sorted(
        exact.items(), key=lambda item: (item[1] - int(item[1]), item[0]), reverse=True
    ):
        if remaining <= 0:
            break
        if raw_slots[key] <= 0:
            continue
        scaled[key] += 1
        remaining -= 1
    if raw_slots["exploration_slots"] > 0 and scaled["exploration_slots"] == 0 and top_n >= 3:
        donor = max(
            (key for key in scaled if key != "exploration_slots"),
            key=lambda key: scaled[key],
        )
        if scaled[donor] > 0:
            scaled[donor] -= 1
            scaled["exploration_slots"] = 1
    return scaled


def _adaptive_slots(
    base_slots: dict[str, int],
    strategy_config: RecommendationStrategyConfig,
    gate: dict[str, Any],
    top_n: int,
) -> dict[str, int]:
    slots = dict(base_slots)
    if not strategy_config.adaptive_gate or top_n <= 1:
        return slots
    mode = str(gate.get("mode", "neutral"))
    shift = max(1, int(round(top_n / 10)))
    if mode == "explore":
        _move_slot(slots, "current_category_slots", "transition_slots", shift)
        _move_slot(slots, "current_category_slots", "exploration_slots", shift)
    elif mode == "repeat":
        _move_slot(slots, "transition_slots", "current_category_slots", shift)
    return slots


def _move_slot(slots: dict[str, int], source: str, target: str, count: int) -> None:
    amount = min(max(0, int(count)), max(0, int(slots.get(source, 0))))
    if amount <= 0:
        return
    slots[source] = int(slots.get(source, 0)) - amount
    slots[target] = int(slots.get(target, 0)) + amount


def _repeat_explore_gate_decision(
    session_context: dict[str, Any],
    *,
    user: dict[str, Any] | None,
    transition_scores: dict[str, float],
    artifacts: RecommendationArtifacts | None,
    adaptive_margin: float,
) -> dict[str, Any]:
    categories = _session_category_sequence(session_context, artifacts)
    event_counts = session_context.get("event_counts", {})
    if not isinstance(event_counts, dict):
        event_counts = {}
    views = int(event_counts.get("view", 0) or 0)
    carts = int(event_counts.get("cart", 0) or 0)
    purchases = int(event_counts.get("purchase", 0) or 0)
    recent_products = list(map(str, session_context.get("recent_products", [])))
    switch_ratio = _category_switch_ratio(categories)
    same_streak = max(_head_streak(categories), _tail_streak(categories))
    max_transition = max(transition_scores.values(), default=0.0)
    persona = str((user or {}).get("persona", ""))
    num_recent_events = int(session_context.get("num_recent_events", 0) or 0)

    has_commit_signal = bool(carts or purchases)
    browsing_without_commit = views >= 2 and not has_commit_signal
    repeat_score = 0.0
    repeat_score += 0.16 if recent_products else 0.0
    if len(categories) >= 2 and not browsing_without_commit:
        repeat_score += min(same_streak, 3) * 0.05
        repeat_score += (1.0 - switch_ratio) * 0.10
    repeat_score += 0.30 if has_commit_signal else 0.0
    repeat_score += 0.06 if views == 1 and switch_ratio < 0.20 else 0.0

    explore_score = 0.0
    explore_score += min(max(views - (2 * carts) - (3 * purchases), 0), 6) * 0.08
    explore_score += switch_ratio * 0.22 if len(categories) >= 2 else 0.0
    explore_score += min(max_transition, 0.55) * 0.60
    explore_score += (
        0.12 if persona in {"trendsetter", "careful_explorer", "careful_researcher"} else 0.0
    )
    explore_score += 0.08 if num_recent_events >= 4 and not (carts or purchases) else 0.0

    margin = max(0.0, float(adaptive_margin))
    if explore_score > repeat_score + margin:
        mode = "explore"
    elif repeat_score > explore_score + margin:
        mode = "repeat"
    else:
        mode = "neutral"
    return {
        "enabled": True,
        "mode": mode,
        "repeat_score": round(repeat_score, 6),
        "explore_score": round(explore_score, 6),
        "margin": round(margin, 6),
        "same_category_streak": same_streak,
        "category_switch_ratio": round(switch_ratio, 6),
        "max_transition_score": round(max_transition, 6),
        "recent_product_count": len(recent_products),
        "view_count": views,
        "cart_count": carts,
        "purchase_count": purchases,
    }


def _session_category_sequence(
    session_context: dict[str, Any],
    artifacts: RecommendationArtifacts | None,
) -> list[str]:
    categories: list[str] = []
    if artifacts:
        for product_id in map(str, session_context.get("recent_products", [])):
            product = artifacts.product_by_id(product_id)
            if product:
                category = product.get("category_l1", product.get("category"))
                if category:
                    categories.append(str(category))
    categories.extend(str(value) for value in session_context.get("recent_categories", []) if value)
    query_category = session_context.get("query_intent_category")
    if query_category and str(query_category) not in categories:
        categories.append(str(query_category))
    return [category for category in categories if category]


def _category_switch_ratio(categories: list[str]) -> float:
    if len(categories) < 2:
        return 0.0
    switches = sum(
        1 for left, right in zip(categories, categories[1:], strict=False) if left != right
    )
    return switches / max(len(categories) - 1, 1)


def _head_streak(values: list[str]) -> int:
    if not values:
        return 0
    target = values[0]
    count = 0
    for value in values:
        if value != target:
            break
        count += 1
    return count


def _tail_streak(values: list[str]) -> int:
    if not values:
        return 0
    target = values[-1]
    count = 0
    for value in reversed(values):
        if value != target:
            break
        count += 1
    return count


def _pick_slot_items(
    selected: list[dict[str, Any]],
    pending: list[dict[str, Any]],
    *,
    limit: int,
    max_same_category_streak: int,
    predicate: Any,
    reason: str,
) -> None:
    if limit <= 0:
        return
    while pending and sum(1 for item in selected if item.get("slot_reason") == reason) < limit:
        selected_ids = {str(item["product"].get("product_id")) for item in selected}
        chosen_index = -1
        for idx, candidate in enumerate(pending):
            if str(candidate["product"].get("product_id")) in selected_ids:
                continue
            if not predicate(candidate):
                continue
            if _would_break_category_streak(selected, candidate, max_same_category_streak):
                continue
            chosen_index = idx
            break
        if chosen_index < 0:
            break
        item = pending.pop(chosen_index)
        item["slot_reason"] = reason
        item["reason"] = reason
        item["is_exploration"] = False
        selected.append(item)


def _append_ranked_prefix(
    selected: list[dict[str, Any]],
    pending: list[dict[str, Any]],
    *,
    limit: int,
    reason: str,
) -> None:
    if limit <= 0 or not pending:
        return
    prefix = pending[:limit]
    del pending[:limit]
    for item in prefix:
        item["slot_reason"] = reason
        item.setdefault("is_exploration", False)
        selected.append(item)


def _append_exploration_items(
    selected: list[dict[str, Any]],
    exploration_slots: int,
    *,
    top_n: int,
    max_same_category_streak: int,
    exploration_products: list[dict[str, Any]],
    mab: EpsilonGreedyMAB,
    user_id: str,
    request_id: str,
    strategy_label: str,
    user: dict[str, Any] | None,
) -> None:
    if exploration_slots <= 0 or not exploration_products or len(selected) >= top_n:
        return
    selected_ids = {str(item["product"].get("product_id")) for item in selected}
    arm = mab.choose_arm(user_id, request_id)
    for product in _rank_exploration_products(
        exploration_products,
        arm,
        selected,
        user,
        rotation_token=f"{user_id}:{request_id}:{arm}",
    ):
        if len(selected) >= top_n or exploration_slots <= 0:
            break
        product_id = str(product.get("product_id"))
        if product_id in selected_ids:
            continue
        candidate = {
            "product": product,
            "candidate_score": 0.0,
            "ranking_score": 0.35 + float(product.get("popularity_prior", 0.0) or 0.0) * 0.2,
            "reason": f"{strategy_label}:exploration:{arm}",
            "slot_reason": f"{strategy_label}:exploration",
            "is_exploration": True,
            "arm": arm,
        }
        if _would_break_category_streak(selected, candidate, max_same_category_streak):
            continue
        selected.append(candidate)
        selected_ids.add(product_id)
        exploration_slots -= 1


def _rank_exploration_products(
    products: list[dict[str, Any]],
    arm: str,
    selected: list[dict[str, Any]],
    user: dict[str, Any] | None,
    rotation_token: str,
) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for product in products:
        product_id = str(product.get("product_id", ""))
        if product_id:
            unique.setdefault(product_id, product)
    selected_categories = {
        str(item["product"].get("category_l1", item["product"].get("category", "")))
        for item in selected
    }
    preferred_categories = set(map(str, _as_list((user or {}).get("preferred_categories"))))
    budget_max = float((user or {}).get("budget_max", 0.0) or 0.0)
    order_prior = {
        product_id: 1.0 - (rank / max(len(unique) - 1, 1)) for rank, product_id in enumerate(unique)
    }

    def score(product: dict[str, Any]) -> tuple[float, float, float, str]:
        popularity = float(product.get("popularity_prior", product.get("popularity", 0.0)) or 0.0)
        is_new = 1.0 if product.get("is_new", False) else 0.0
        price = float(product.get("price", 0.0) or 0.0)
        category = str(product.get("category_l1", product.get("category", "")))
        category_affinity = 1.0 if category in preferred_categories | selected_categories else 0.0
        personalized_prior = order_prior.get(str(product.get("product_id", "")), 0.0)
        if budget_max > 0:
            price_fit = max(0.0, 1.0 - abs(price - budget_max) / max(budget_max, 1.0))
        else:
            price_fit = max(0.0, 1.0 - min(price, 300_000.0) / 300_000.0)
        if arm == "novelty":
            return (is_new, personalized_prior, popularity, str(product.get("product_id", "")))
        if arm == "diversity":
            return (
                1.0 if category not in selected_categories else 0.0,
                personalized_prior,
                popularity,
                str(product.get("product_id", "")),
            )
        if arm == "price_match":
            return (price_fit, personalized_prior, popularity, str(product.get("product_id", "")))
        return (
            category_affinity,
            personalized_prior,
            popularity,
            str(product.get("product_id", "")),
        )

    ranked = sorted(unique.values(), key=score, reverse=True)
    pool_size = min(len(ranked), max(64, min(512, (len(selected) + 1) * 8)))
    if pool_size <= 1:
        return ranked
    digest = hashlib.blake2b(rotation_token.encode("utf-8"), digest_size=8).digest()
    offset = int.from_bytes(digest, "big") % pool_size
    return ranked[offset:pool_size] + ranked[:offset] + ranked[pool_size:]


def _catalog_exploration_pool(
    products: list[dict[str, Any]],
    *,
    rotation_token: str,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or not products:
        return []
    if len(products) <= limit:
        return list(products)

    digest = hashlib.blake2b(rotation_token.encode("utf-8"), digest_size=16).digest()
    size = len(products)
    index = int.from_bytes(digest[:8], "big") % size
    stride = (int.from_bytes(digest[8:], "big") % (size - 1)) + 1
    while math.gcd(stride, size) != 1:
        stride = (stride + 1) % size or 1

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for _ in range(size):
        product = products[index]
        product_id = str(product.get("product_id", ""))
        if product_id and product_id not in seen_ids:
            selected.append(product)
            seen_ids.add(product_id)
            if len(selected) >= limit:
                break
        index = (index + stride) % size
    return selected


def _ensure_new_item_exposure(
    selected: list[dict[str, Any]],
    products: list[dict[str, Any]],
    *,
    top_n: int,
    max_same_category_streak: int,
    strategy_label: str,
) -> None:
    if any(item["product"].get("is_new", False) for item in selected):
        return
    selected_ids = {str(item["product"].get("product_id")) for item in selected}
    base = selected[: max(0, top_n - 1)] if len(selected) >= top_n else selected
    for product in products:
        product_id = str(product.get("product_id", ""))
        if not product_id or product_id in selected_ids or not product.get("is_new", False):
            continue
        candidate = {
            "product": product,
            "candidate_score": 0.0,
            "ranking_score": 0.0,
            "reason": f"{strategy_label}:new_item_guarantee",
            "slot_reason": f"{strategy_label}:new_item_guarantee",
            "is_exploration": False,
            "arm": None,
        }
        if _would_break_category_streak(base, candidate, max_same_category_streak):
            continue
        if len(selected) >= top_n:
            selected[-1] = candidate
        else:
            selected.append(candidate)
        return


def _with_transition_score(
    item: dict[str, Any],
    transition_scores: dict[str, float],
    strategy_config: RecommendationStrategyConfig,
) -> dict[str, Any]:
    output = dict(item)
    product = output.get("product", {})
    product_transition_score = max(
        (transition_scores.get(category, 0.0) for category in _product_category_keys(product)),
        default=0.0,
    )
    output["transition_score"] = product_transition_score
    if product_transition_score > 0 and strategy_config.transition_boost > 0:
        output["ranking_score"] = float(output.get("ranking_score", 0.0) or 0.0)
        output["ranking_score"] += strategy_config.transition_boost * product_transition_score
    return output


def _transition_scores(
    session_context: dict[str, Any],
    artifacts: RecommendationArtifacts | None,
) -> dict[str, float]:
    if not artifacts or not artifacts.category_transitions:
        return {}
    source_categories = set(_current_category_keys(session_context, artifacts))
    scores: dict[str, float] = {}
    for source in source_categories:
        for item in artifacts.category_transitions.get(source, []):
            target = str(item.get("category", "")).strip()
            if not target or target in source_categories:
                continue
            probability = float(item.get("probability", 0.0) or 0.0)
            scores[target] = max(scores.get(target, 0.0), probability)
    return scores


def _current_category_keys(
    session_context: dict[str, Any],
    artifacts: RecommendationArtifacts | None,
) -> list[str]:
    categories: list[str] = []
    if artifacts:
        for product_id in map(str, session_context.get("recent_products", [])):
            product = artifacts.product_by_id(product_id)
            if product:
                categories.extend(_product_category_keys(product))
    categories.extend(map(str, session_context.get("recent_categories", [])))
    for key in ("query_intent_category", "session_interest"):
        value = session_context.get(key)
        if value:
            categories.append(str(value))
    return _ordered_unique(categories)


def _product_category_keys(product: dict[str, Any]) -> list[str]:
    return _ordered_unique(
        [
            str(product.get(key, ""))
            for key in (
                "category_l3",
                "leaf_category",
                "category_l2",
                "mid_category",
                "category_l1",
                "top_category",
                "category",
            )
            if product.get(key)
        ]
    )


def _would_break_category_streak(
    selected: list[dict[str, Any]],
    candidate: dict[str, Any],
    max_same_category_streak: int,
) -> bool:
    if max_same_category_streak <= 0 or len(selected) < max_same_category_streak:
        return False
    category = candidate["product"].get("category_l1", candidate["product"].get("category"))
    return all(
        item["product"].get("category_l1", item["product"].get("category")) == category
        for item in selected[-max_same_category_streak:]
    )


def _candidate_reason(item_vector_index: VectorIndex | None, ann_score: float | None) -> str:
    if ann_score is not None:
        return "two_tower_faiss"
    if item_vector_index is not None:
        return "two_tower_faiss_augmented"
    return "two_tower_fallback"


def _reason_for(user: dict[str, Any] | None, item: dict[str, Any]) -> str:
    if item.get("is_exploration"):
        return str(item.get("reason", "exploration"))
    if user is None:
        return str(item.get("reason", "cold_start_popularity"))
    product = item["product"]
    category = product.get("category_l1", product.get("category"))
    if category in set(map(str, _as_list(user.get("preferred_categories")))):
        return "matches_preferred_category"
    return str(item.get("reason", "ranked_by_wide_deep"))


def _candidate_score(
    user: dict[str, Any],
    product: dict[str, Any],
    session_context: dict[str, Any],
) -> float:
    category = str(product.get("category_l1", product.get("category", "")))
    preferred_categories = set(map(str, _as_list(user.get("preferred_categories"))))
    recent_categories = set(map(str, session_context.get("recent_categories", [])))
    popularity = float(product.get("popularity_prior", product.get("popularity", 0.0)) or 0.0)
    margin = float(product.get("margin_score", 0.0) or 0.0)
    is_new = 1.0 if product.get("is_new", False) else 0.0
    price = float(product.get("price", 0.0) or 0.0)
    budget_max = float(user.get("budget_max", 0.0) or 0.0)
    price_sensitivity = float(user.get("price_sensitivity", 0.5) or 0.5)
    if budget_max > 0:
        price_fit = max(0.0, 1.0 - max(price - budget_max, 0.0) / max(budget_max, 1.0))
    else:
        price_fit = max(0.0, 1.0 - min(price, 300_000.0) / 300_000.0)
    score = 0.0
    score += 0.58 if category in preferred_categories else 0.0
    score += 0.22 if category in recent_categories else 0.0
    score += 0.34 * popularity
    score += 0.10 * margin
    score += 0.22 * price_fit * price_sensitivity
    persona = str(user.get("persona", ""))
    if persona == "trendsetter":
        score += 0.12 * is_new + 0.10 * popularity
    elif persona == "impulse_buyer":
        score += 0.16 * popularity + 0.08 * is_new
    elif persona == "top_category_loyalist" and category in preferred_categories:
        score += 0.18
    elif persona in {"value_seeker", "pragmatist", "careful_explorer"}:
        score += 0.12 * price_fit * price_sensitivity
    return score


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _query_tokens_from_context(session_context: dict[str, Any]) -> list[str]:
    query = str(session_context.get("query") or "").strip()
    if not query:
        return []
    parts = [
        query,
        session_context.get("query_intent_category"),
        session_context.get("session_interest"),
    ]
    return _tokenize(" ".join(str(part or "") for part in parts), include_ngrams=True)[:18]


def _product_token_weights(product: dict[str, Any]) -> dict[str, float]:
    fields = {
        "name": 4.0,
        "category_l3": 3.5,
        "category_l2": 2.4,
        "category_l1": 2.0,
        "category": 2.0,
        "leaf_category": 3.2,
        "mid_category": 2.2,
        "top_category": 1.8,
        "color": 3.0,
        "style_tags": 2.3,
        "description": 1.2,
        "price_tier": 0.4,
    }
    weights: dict[str, float] = {}
    for feature_name, weight in fields.items():
        value = product.get(feature_name)
        values = _as_list(value)
        for item in values:
            include_ngrams = feature_name in {
                "name",
                "category_l3",
                "leaf_category",
                "color",
                "style_tags",
            }
            for token in _tokenize(str(item or ""), include_ngrams=include_ngrams):
                weights[token] = weights.get(token, 0.0) + weight
    for token in _product_alias_tokens(weights):
        weights[token] = max(weights.get(token, 0.0), 1.6)
    return weights


def _tokenize(text: str, *, include_ngrams: bool = False) -> list[str]:
    normalized = str(text or "").lower().replace("/", " ").replace("-", " ")
    tokens: list[str] = []
    for raw in normalized.split():
        token = "".join(ch for ch in raw if ch.isalnum())
        if len(token) >= 2 and token not in tokens:
            tokens.append(token)
    for alias in _phrase_aliases(normalized):
        if alias not in tokens:
            tokens.append(alias)
    if include_ngrams:
        base = [token for token in tokens if "__" not in token]
        for size in (2, 3):
            for idx in range(0, max(0, len(base) - size + 1)):
                ngram = "__".join(base[idx : idx + size])
                if ngram not in tokens:
                    tokens.append(ngram)
    return tokens


def _phrase_aliases(text: str) -> list[str]:
    alias_map = {
        "tank top": ["vest", "sleeveless"],
        "vest top": ["tank", "sleeveless"],
        "swim bottom": ["swimwear", "bikini"],
        "bikini bottom": ["swimwear", "swim"],
        "high waist": ["highwaist", "hw"],
        "high waisted": ["highwaist", "hw"],
        "mid waist": ["midrise"],
        "mid rise": ["midrise"],
        "t shirt": ["tee"],
        "t-shirt": ["tee"],
        "hooded sweatshirt": ["hoodie"],
    }
    aliases: list[str] = []
    for phrase, values in alias_map.items():
        if phrase in text:
            aliases.extend(values)
    if re.search(r"\btrs\b", text):
        aliases.append("trousers")
    return aliases


def _product_alias_tokens(weights: dict[str, float]) -> list[str]:
    tokens = set(weights)
    aliases: list[str] = []
    reverse_aliases = {
        "trousers": ["pants"],
        "pants": ["trousers"],
        "jumper": ["sweater", "knitwear"],
        "sweater": ["jumper", "knitwear"],
        "dress": ["dresses"],
        "skirt": ["skirts"],
        "hoodie": ["hooded", "sweatshirt"],
        "tee": ["tshirt", "shirt"],
        "black": ["dark"],
        "blue": ["denim"],
    }
    for token, values in reverse_aliases.items():
        if token in tokens:
            aliases.extend(values)
    return aliases


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
