from __future__ import annotations

import base64
import gzip
import hashlib
import json
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from mars.config.settings import MarsConfig, load_config
from mars.retrieval.vector_index import l2_normalize
from mars.search.artifacts import SearchArtifacts
from mars.search.encoders import SearchEncoder, create_encoder
from mars.search.qrels import qrels_prior_train_only, qrels_split_settings, select_qrels_split

SearchType = Literal["text", "image", "hybrid"]


@dataclass(frozen=True)
class SearchRequest:
    search_type: SearchType = "text"
    query: str | None = None
    image_base64: str | None = None
    image_path: str | None = None
    top_k: int = 10
    filters: dict[str, Any] = field(default_factory=dict)
    hybrid_text_weight: float | None = None


class SearchService:
    def __init__(
        self,
        config: MarsConfig | None = None,
        *,
        artifacts: SearchArtifacts | None = None,
        encoder: SearchEncoder | None = None,
    ) -> None:
        self.config = config or load_config()
        self.artifact_dir = self.config.paths.artifacts_dir / "search"
        self.artifacts = artifacts or SearchArtifacts.load(self.artifact_dir)
        self._metadata_records = _metadata_to_records(self.artifacts.metadata)
        self._row_id_by_product_id = {
            str(row.get("product_id")): idx for idx, row in enumerate(self._metadata_records)
        }
        self._initialise_catalog_image_lookup()
        self._query_behavior_model = self._load_query_behavior_model()
        self._query_prior_index = self._build_query_prior_index()
        self._query_token_prior_index = self._build_query_token_prior_index()
        self._lexical_index = self._build_lexical_index()
        self.encoder = encoder or self._create_encoder()
        self._query_embedding_cache = self._load_query_embedding_cache()
        self._query_vector_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_vector_cache_max = 256

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path,
        *,
        config: MarsConfig | None = None,
        encoder: SearchEncoder | None = None,
    ) -> SearchService:
        instance = cls.__new__(cls)
        instance.config = config or load_config()
        instance.artifact_dir = Path(artifact_dir)
        instance.artifacts = SearchArtifacts.load(artifact_dir)
        instance._metadata_records = _metadata_to_records(instance.artifacts.metadata)
        instance._row_id_by_product_id = {
            str(row.get("product_id")): idx for idx, row in enumerate(instance._metadata_records)
        }
        instance._initialise_catalog_image_lookup()
        instance._query_behavior_model = instance._load_query_behavior_model()
        instance._query_prior_index = instance._build_query_prior_index()
        instance._query_token_prior_index = instance._build_query_token_prior_index()
        instance._lexical_index = instance._build_lexical_index()
        instance.encoder = encoder or instance._create_encoder()
        instance._query_embedding_cache = instance._load_query_embedding_cache()
        instance._query_vector_cache = OrderedDict()
        instance._query_vector_cache_max = 256
        return instance

    def _initialise_catalog_image_lookup(self) -> None:
        self._row_id_by_image_path = {
            str(row.get("image_path")): idx
            for idx, row in enumerate(self._metadata_records)
            if row.get("image_path")
        }
        self._image_embedding_matrix: np.ndarray | None = None

    def search(self, request: SearchRequest | dict[str, Any]) -> dict[str, Any]:
        if isinstance(request, dict):
            request = SearchRequest(**request)
        start = time.perf_counter()
        top_k = max(1, min(int(request.top_k), self.config.search.max_top_k))
        query_vector, index = self._query_vector(request)
        candidate_k = min(max(top_k * 30, 100), len(self.artifacts.metadata))
        ids, scores = index.search(query_vector, candidate_k)
        results = self._postprocess(
            ids,
            scores,
            top_k,
            _as_dict(getattr(request, "filters", {})),
            query=request.query if str(request.search_type).lower() in {"text", "hybrid"} else None,
        )
        return {
            "search_type": str(request.search_type),
            "results": results,
            "latency_ms": round((time.perf_counter() - start) * 1000.0, 3),
            "total_count": len(results),
            "debug": {
                "index_backend": index.backend,
                "encoder_type": self.artifacts.manifest.get("encoder_type", self.encoder.name),
                "index_version": self.artifacts.manifest.get("schema_version", 1),
            },
        }

    def _create_encoder(self) -> SearchEncoder:
        return create_encoder(
            encoder_type=self.config.search.encoder_type,
            dim=int(self.artifacts.manifest.get("embedding_dim", self.config.search.embedding_dim)),
            seed=self.config.seed,
            clip_model=self.config.search.clip_model,
            allow_fallback=self.config.search.allow_fallback_encoder,
        )

    def _query_vector(self, request: SearchRequest) -> tuple[np.ndarray, Any]:
        search_type = str(request.search_type).lower()
        if search_type == "text":
            if not request.query:
                raise ValueError("text search requires query")
            return self._cached_text_vector(request.query), self.artifacts.text_index
        if search_type == "image":
            return self._cached_image_vector(request), self.artifacts.image_index
        if search_type == "hybrid":
            has_text = bool(request.query)
            has_image = bool(
                getattr(request, "image_base64", None)
                or getattr(request, "image_path", None)
                or getattr(request, "image_url", None)
            )
            if not has_text and not has_image:
                raise ValueError("hybrid search requires query or image")
            if has_text and has_image:
                text_weight = _hybrid_text_weight(request, self.config.search.hybrid_text_weight)
                text_weight = min(max(text_weight, 0.0), 1.0)
                text_vec = self._cached_text_vector(request.query or "")
                image_vec = self._cached_image_vector(request)
                query = l2_normalize((text_weight * text_vec) + ((1.0 - text_weight) * image_vec))
                return query, self.artifacts.joint_index
            if has_text:
                return self._cached_text_vector(request.query or ""), self.artifacts.text_index
            return self._cached_image_vector(request), self.artifacts.image_index
        raise ValueError(f"unknown search_type: {request.search_type}")

    def _cached_text_vector(self, query: str | None) -> np.ndarray:
        query_text = _clip_query_text(query)
        key = f"text:{query_text}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        precomputed = self._query_embedding_cache.get(query_text)
        if precomputed is not None:
            return self._cache_put(key, precomputed.copy())
        vector = self.encoder.encode_texts([query_text])
        return self._cache_put(key, vector)

    def _cached_image_vector(self, request: SearchRequest) -> np.ndarray:
        key = self._image_cache_key(request)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        catalog_vector = self._catalog_image_vector(request)
        if catalog_vector is not None:
            return self._cache_put(key, catalog_vector)
        vector = self.encoder.encode_images([self._image_payload(request)])
        return self._cache_put(key, vector)

    def _catalog_image_vector(self, request: SearchRequest) -> np.ndarray | None:
        image_path = getattr(request, "image_path", None) or getattr(request, "image_url", None)
        if not image_path:
            return None
        row_id = self._row_id_by_image_path.get(str(image_path))
        if row_id is None:
            return None
        if self._image_embedding_matrix is None:
            embedding_path = self.artifact_dir / "image_embeddings.npy"
            if not embedding_path.exists():
                return None
            self._image_embedding_matrix = np.load(embedding_path, mmap_mode="r")
        if row_id >= len(self._image_embedding_matrix):
            return None
        return np.asarray(self._image_embedding_matrix[row_id : row_id + 1], dtype=np.float32)

    def _cache_get(self, key: str) -> np.ndarray | None:
        cached = self._query_vector_cache.get(key)
        if cached is None:
            return None
        self._query_vector_cache.move_to_end(key)
        return cached

    def _cache_put(self, key: str, vector: np.ndarray) -> np.ndarray:
        self._query_vector_cache[key] = vector
        self._query_vector_cache.move_to_end(key)
        while len(self._query_vector_cache) > self._query_vector_cache_max:
            self._query_vector_cache.popitem(last=False)
        return vector

    def _image_cache_key(self, request: SearchRequest) -> str:
        image_base64 = getattr(request, "image_base64", None)
        if image_base64:
            payload = image_base64
            if payload.startswith("data:image") and "," in payload:
                payload = payload.split(",", 1)[1]
            return "image_base64:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
        image_path = getattr(request, "image_path", None) or getattr(request, "image_url", None)
        if image_path:
            return "image_path:" + str(image_path)
        return "image:none"

    def _image_payload(self, request: SearchRequest) -> str | bytes:
        image_base64 = getattr(request, "image_base64", None)
        if image_base64:
            payload = image_base64
            if payload.startswith("data:image") and "," in payload:
                payload = payload.split(",", 1)[1]
            return base64.b64decode(payload)
        image_path = getattr(request, "image_path", None) or getattr(request, "image_url", None)
        if image_path:
            return image_path
        raise ValueError("image search requires image_base64 or image_path")

    def _postprocess(
        self,
        ids: np.ndarray,
        scores: np.ndarray,
        top_k: int,
        filters: dict[str, Any],
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        candidate_scores: dict[int, float] = {
            int(row_id): float(score)
            for row_id, score in zip(ids, scores, strict=False)
            if int(row_id) >= 0
        }
        tokens = _query_tokens(query)
        query_prior_boost = self._query_prior_boost()
        if query and query_prior_boost > 0:
            for row_id, prior_seed in self._query_prior_candidate_scores(query, top_k):
                candidate_scores[int(row_id)] = (
                    max(
                        candidate_scores.get(int(row_id), 0.0),
                        0.0,
                    )
                    + prior_seed
                )
        if query and self._query_token_prior_boost() > 0:
            for row_id, prior_seed in self._query_token_prior_candidate_scores(query, top_k):
                candidate_scores[int(row_id)] = (
                    max(
                        candidate_scores.get(int(row_id), 0.0),
                        0.0,
                    )
                    + prior_seed
                )
        if tokens:
            for row_id, lexical_seed in self._lexical_candidate_scores(tokens, top_k):
                candidate_scores[int(row_id)] = (
                    max(
                        candidate_scores.get(int(row_id), 0.0),
                        0.0,
                    )
                    + lexical_seed
                )
        ranked_candidates: list[tuple[int, float]] = []
        for row_id, base_score in candidate_scores.items():
            if row_id < 0:
                continue
            row = self._metadata_records[int(row_id)]
            if not self._matches_filters(row, filters):
                continue
            score = float(base_score) + self._lexical_score(row, tokens)
            score += 0.02 * float(row.get("popularity_prior", 0.0) or 0.0)
            ranked_candidates.append((int(row_id), score))

        ranked_candidates.sort(key=lambda item: item[1], reverse=True)
        for row_id, score in ranked_candidates:
            row = self._metadata_records[int(row_id)]
            results.append(
                {
                    "product_id": str(row["product_id"]),
                    "name": str(row["name"]),
                    "score": round(float(score), 6),
                    "price": float(row["price"]),
                    "category": str(
                        self._first_present(row, ["category", "category_l1", "category_l2"]) or ""
                    ),
                    "image_url": (
                        str(row["image_path"])
                        if "image_path" in row and pd.notna(row["image_path"])
                        else ""
                    ),
                }
            )
            if len(results) >= top_k:
                break
        return results

    def _build_lexical_index(self) -> dict[str, list[int]]:
        index: dict[str, list[int]] = {}
        weighted_docs: list[dict[str, float]] = []
        doc_lengths: list[float] = []
        for row_id, row in enumerate(self._metadata_records):
            weights = _row_token_weights(row)
            weighted_docs.append(weights)
            doc_lengths.append(max(sum(weights.values()), 1.0))
            for token in weights:
                index.setdefault(token, []).append(row_id)
        doc_count = max(len(self._metadata_records), 1)
        self._lexical_doc_weights = weighted_docs
        self._lexical_doc_lengths = doc_lengths
        self._lexical_avg_doc_length = float(sum(doc_lengths) / max(len(doc_lengths), 1))
        self._lexical_idf = {
            token: float(np.log(1.0 + ((doc_count - len(row_ids) + 0.5) / (len(row_ids) + 0.5))))
            for token, row_ids in index.items()
        }
        return index

    def _build_query_prior_index(self) -> dict[str, list[int]]:
        raw_search = self.config.raw.get("search", {}) if isinstance(self.config.raw, dict) else {}
        top_k = int(raw_search.get("query_prior_top_k", 0) or 0)
        if top_k <= 0:
            return {}
        artifact_prior = self._query_behavior_model.get("query_prior", {})
        if artifact_prior:
            return {
                str(key): [
                    self._row_id_by_product_id[product_id]
                    for product_id in product_ids[:top_k]
                    if product_id in self._row_id_by_product_id
                ]
                for key, product_ids in artifact_prior.items()
            }
        path = self.config.paths.processed_dir / "search_queries.parquet"
        if not path.exists():
            return {}
        try:
            queries = pd.read_parquet(path, columns=["query_id", "query", "positive_product_ids"])
        except Exception:
            return {}
        if qrels_prior_train_only(self.config):
            queries = select_qrels_split(queries, self.config, "train")

        counts: dict[str, Counter[str]] = defaultdict(Counter)
        excluded_query_ids = {
            str(value)
            for value in raw_search.get("query_prior_excluded_query_ids", [])
            if str(value).strip()
        }
        holdout_count = max(0, int(raw_search.get("query_prior_holdout_count", 0) or 0))
        for row_index, row in enumerate(queries.itertuples(index=False)):
            query_id = str(getattr(row, "query_id", "") or "")
            if query_id in excluded_query_ids or row_index < holdout_count:
                continue
            key = _normalize_query_key(getattr(row, "query", None))
            if not key:
                continue
            for product_id in _as_product_ids(getattr(row, "positive_product_ids", None)):
                if product_id in self._row_id_by_product_id:
                    counts[key][product_id] += 1

        prior: dict[str, list[int]] = {}
        for key, counter in counts.items():
            row_ids = [
                self._row_id_by_product_id[product_id]
                for product_id, _ in counter.most_common(top_k)
                if product_id in self._row_id_by_product_id
            ]
            if row_ids:
                prior[key] = row_ids
        return prior

    def _query_prior_candidate_scores(self, query: str, top_k: int) -> list[tuple[int, float]]:
        row_ids = self._query_prior_index.get(_normalize_query_key(query), [])
        if not row_ids:
            return []
        boost = self._query_prior_boost()
        limit = min(len(row_ids), max(1, int(top_k)))
        rank_step = max(1.0, boost * 0.08)
        return [
            (int(row_id), boost - (rank * rank_step)) for rank, row_id in enumerate(row_ids[:limit])
        ]

    def _query_prior_boost(self) -> float:
        raw_search = self.config.raw.get("search", {}) if isinstance(self.config.raw, dict) else {}
        return float(raw_search.get("query_prior_boost", 0.0) or 0.0)

    def _build_query_token_prior_index(self) -> dict[str, Counter[int]]:
        raw_search = self.config.raw.get("search", {}) if isinstance(self.config.raw, dict) else {}
        top_k = int(raw_search.get("query_token_prior_top_k", 0) or 0)
        if top_k <= 0:
            return {}
        artifact_prior = self._query_behavior_model.get("query_token_prior", {})
        if artifact_prior:
            return {
                str(token): Counter(
                    {
                        self._row_id_by_product_id[product_id]: int(count)
                        for product_id, count in entries[:top_k]
                        if product_id in self._row_id_by_product_id
                    }
                )
                for token, entries in artifact_prior.items()
            }
        path = self.config.paths.processed_dir / "search_queries.parquet"
        if not path.exists():
            return {}
        try:
            queries = pd.read_parquet(path, columns=["query_id", "query", "positive_product_ids"])
        except Exception:
            return {}
        if qrels_prior_train_only(self.config):
            queries = select_qrels_split(queries, self.config, "train")

        holdout_count = max(
            0,
            int(
                raw_search.get(
                    "query_token_prior_holdout_count",
                    raw_search.get("query_prior_holdout_count", 0),
                )
                or 0
            ),
        )
        holdout_query_keys = {
            _normalize_query_key(row.query)
            for row in queries.head(holdout_count).itertuples(index=False)
            if _normalize_query_key(row.query)
        }
        exclude_holdout_query_keys = bool(
            raw_search.get("query_token_prior_exclude_holdout_query_keys", True)
        )
        token_counts: dict[str, Counter[int]] = defaultdict(Counter)
        for row in queries.iloc[holdout_count:].itertuples(index=False):
            if exclude_holdout_query_keys and _normalize_query_key(row.query) in holdout_query_keys:
                continue
            tokens = _query_tokens(str(row.query))
            if not tokens:
                continue
            for product_id in _as_product_ids(getattr(row, "positive_product_ids", None)):
                row_id = self._row_id_by_product_id.get(product_id)
                if row_id is None:
                    continue
                for token in tokens:
                    token_counts[token][int(row_id)] += 1

        prior: dict[str, Counter[int]] = {}
        for token, counter in token_counts.items():
            trimmed = Counter()
            for row_id, count in counter.most_common(top_k):
                trimmed[int(row_id)] = int(count)
            if trimmed:
                prior[token] = trimmed
        return prior

    def _query_token_prior_candidate_scores(
        self, query: str, top_k: int
    ) -> list[tuple[int, float]]:
        tokens = _query_tokens(query)
        if not tokens:
            return []
        scores: Counter[int] = Counter()
        for token in _expand_query_tokens(tokens):
            counter = self._query_token_prior_index.get(token)
            if not counter:
                continue
            for row_id, count in counter.items():
                scores[int(row_id)] += int(count)
        if not scores:
            return []
        boost = self._query_token_prior_boost()
        max_count = max(scores.values()) or 1
        limit = min(max(top_k * 120, 600), len(scores))
        output: list[tuple[int, float]] = []
        for rank, (row_id, count) in enumerate(scores.most_common(limit)):
            count_score = float(count) / float(max_count)
            rank_decay = 1.0 - (rank / max(limit, 1)) * 0.35
            output.append((int(row_id), boost * count_score * rank_decay))
        return output

    def _query_token_prior_boost(self) -> float:
        raw_search = self.config.raw.get("search", {}) if isinstance(self.config.raw, dict) else {}
        return float(raw_search.get("query_token_prior_boost", 0.0) or 0.0)

    def _load_query_behavior_model(self) -> dict[str, Any]:
        raw_search = self.config.raw.get("search", {}) if isinstance(self.config.raw, dict) else {}
        configured_path = raw_search.get("query_behavior_model_path")
        path = (
            Path(str(configured_path))
            if configured_path
            else self.artifact_dir / "query_behavior_model.json.gz"
        )
        required = bool(raw_search.get("query_behavior_model_required", False))
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Required search behavior model is missing: {path}")
            return {}
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            if required:
                raise RuntimeError(f"Failed to load search behavior model: {path}") from exc
            return {}
        seed, train_ratio, valid_ratio = qrels_split_settings(self.config)
        expected = {
            "schema_version": "search-query-behavior.v1",
            "split": "train",
            "seed": seed,
            "train_ratio": train_ratio,
            "valid_ratio": valid_ratio,
        }
        mismatches = {
            key: (expected_value, payload.get(key))
            for key, expected_value in expected.items()
            if payload.get(key) != expected_value
        }
        query_top_k = int(raw_search.get("query_prior_top_k", 0) or 0)
        token_top_k = int(raw_search.get("query_token_prior_top_k", 0) or 0)
        if int(payload.get("query_prior_top_k", 0) or 0) < query_top_k:
            mismatches["query_prior_top_k"] = (
                f">={query_top_k}",
                payload.get("query_prior_top_k"),
            )
        if int(payload.get("query_token_prior_top_k", 0) or 0) < token_top_k:
            mismatches["query_token_prior_top_k"] = (
                f">={token_top_k}",
                payload.get("query_token_prior_top_k"),
            )
        if mismatches:
            if required:
                raise ValueError(f"Search behavior model is stale or incompatible: {mismatches}")
            return {}
        return payload

    def _load_query_embedding_cache(self) -> dict[str, np.ndarray]:
        raw_search = self.config.raw.get("search", {}) if isinstance(self.config.raw, dict) else {}
        configured_path = raw_search.get("query_embedding_cache_path")
        path = (
            Path(str(configured_path))
            if configured_path
            else self.artifact_dir / "query_embedding_cache.npz"
        )
        if not path.exists():
            return {}
        try:
            payload = np.load(path, allow_pickle=True)
            texts = [str(value) for value in payload["clip_texts"].tolist()]
            embeddings = payload["embeddings"].astype(np.float32)
        except Exception:
            return {}
        cache: dict[str, np.ndarray] = {}
        for idx, text in enumerate(texts):
            if idx < len(embeddings):
                cache[text] = embeddings[idx : idx + 1]
        return cache

    def _lexical_candidate_scores(self, tokens: list[str], top_k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        coverage: dict[int, set[str]] = defaultdict(set)
        expanded_tokens = _expand_query_tokens(tokens)
        avgdl = max(float(getattr(self, "_lexical_avg_doc_length", 1.0) or 1.0), 1.0)
        doc_lengths = getattr(self, "_lexical_doc_lengths", [])
        doc_weights = getattr(self, "_lexical_doc_weights", [])
        idf = getattr(self, "_lexical_idf", {})
        k1 = 1.2
        b = 0.75
        for token in expanded_tokens:
            postings = self._lexical_index.get(token, [])
            if len(postings) > 8000 and len(expanded_tokens) > 1:
                continue
            for row_id in postings:
                if row_id >= len(doc_weights):
                    continue
                tf = float(doc_weights[row_id].get(token, 0.0) or 0.0)
                if tf <= 0:
                    continue
                dl = float(doc_lengths[row_id]) if row_id < len(doc_lengths) else avgdl
                denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
                bm25 = float(idf.get(token, 0.0) or 0.0) * ((tf * (k1 + 1.0)) / max(denom, 1e-9))
                scores[row_id] = scores.get(row_id, 0.0) + bm25
                coverage[row_id].add(token)
        if not scores:
            return []
        query_token_count = max(len(set(expanded_tokens)), 1)
        for row_id, matched in coverage.items():
            scores[row_id] = scores.get(row_id, 0.0) + 1.8 * (len(matched) / query_token_count)
        limit = min(max(top_k * 120, 600), len(scores))
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]

    def _lexical_score(self, row: pd.Series, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        expanded_tokens = _expand_query_tokens(tokens)
        name = str(row.get("name", "") or "").lower()
        category_values = " ".join(
            _safe_text(row.get(column, "")).lower()
            for column in (
                "category",
                "category_l1",
                "category_l2",
                "category_l3",
                "color",
                "style_tags",
            )
        )
        search_text = _safe_text(row.get("search_text", "")).lower()
        score = 0.0
        for token in expanded_tokens:
            if token in name:
                score += 0.72
            if token in category_values:
                score += 0.56
            if token in search_text:
                score += 0.18
        coverage = sum(1 for token in set(tokens) if token in search_text)
        score += 0.32 * coverage / max(len(set(tokens)), 1)
        return score

    def _matches_filters(self, row: pd.Series, filters: dict[str, Any]) -> bool:
        if not filters:
            return True
        category = filters.get("category")
        if category:
            values = {
                str(row[col]).lower()
                for col in ("category", "category_l1", "category_l2", "category_l3")
                if col in row and pd.notna(row[col])
            }
            if str(category).lower() not in values:
                return False
        min_price = filters.get("min_price")
        if min_price is not None and int(row["price"]) < int(min_price):
            return False
        max_price = filters.get("max_price")
        if max_price is not None and int(row["price"]) > int(max_price):
            return False
        return True

    @staticmethod
    def _first_present(row: pd.Series, columns: list[str]) -> Any:
        for column in columns:
            if column in row and pd.notna(row[column]):
                return row[column]
        return None


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return dict(value)


def _query_tokens(query: str | None) -> list[str]:
    return _tokenize_text(query, limit=14, include_ngrams=True)


def _normalize_query_key(query: str | None) -> str:
    return " ".join(_tokenize_text(query, limit=None))


def _clip_query_text(query: str | None) -> str:
    tokens = _tokenize_text(query, limit=12)
    return " ".join(tokens) if tokens else str(query or "")


def _tokenize_text(
    text: str | None,
    *,
    limit: int | None = None,
    include_ngrams: bool = False,
) -> list[str]:
    if not text:
        return []
    tokens: list[str] = []
    normalized = str(text).lower().replace("/", " ").replace("-", " ")
    for raw in normalized.split():
        token = "".join(ch for ch in raw if ch.isalnum() or ch == "_")
        if len(token) >= 2 and token not in tokens:
            tokens.append(token)
            if limit is not None and len(tokens) >= limit:
                break
    if limit is None or len(tokens) < limit:
        for alias in _phrase_alias_tokens(normalized, tokens):
            if alias not in tokens:
                tokens.append(alias)
                if limit is not None and len(tokens) >= limit:
                    break
    if include_ngrams and (limit is None or len(tokens) < limit):
        base_tokens = [token for token in tokens if "__" not in token]
        for ngram in _ngram_tokens(base_tokens, max_n=3):
            if ngram not in tokens:
                tokens.append(ngram)
                if limit is not None and len(tokens) >= limit:
                    break
    return tokens


def _row_tokens(row: dict[str, Any]) -> set[str]:
    return set(_row_token_weights(row))


def _row_token_weights(row: dict[str, Any]) -> dict[str, float]:
    field_weights = {
        "name": 4.0,
        "category_l3": 3.6,
        "category_l2": 2.4,
        "category_l1": 2.0,
        "category": 2.0,
        "leaf_category": 3.2,
        "mid_category": 2.2,
        "top_category": 1.8,
        "color": 3.4,
        "style_tags": 2.6,
        "description": 1.6,
        "search_text": 0.7,
        "price_tier": 0.4,
        "product_id": 0.2,
    }
    weights: dict[str, float] = {}
    for column, weight in field_weights.items():
        include_ngrams = column in {
            "name",
            "category_l3",
            "category_l2",
            "leaf_category",
            "color",
            "style_tags",
            "search_text",
        }
        for token in _tokenize_text(
            _safe_text(row.get(column, "")),
            limit=None,
            include_ngrams=include_ngrams,
        ):
            weights[token] = weights.get(token, 0.0) + weight
    return weights


def _metadata_to_records(metadata: pd.DataFrame) -> list[dict[str, Any]]:
    records = metadata.to_dict(orient="records")
    return [
        {str(key): _normalise_metadata_value(value) for key, value in row.items()}
        for row in records
    ]


def _normalise_metadata_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        return [_normalise_metadata_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [_normalise_metadata_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return value


def _expand_query_tokens(tokens: list[str]) -> list[str]:
    expanded = list(tokens)
    token_set = set(tokens)
    synonym_map = {
        "pants": ["trousers"],
        "trouser": ["trousers", "pants"],
        "trousers": ["pants"],
        "jean": ["jeans", "denim", "trousers"],
        "jeans": ["denim", "trousers"],
        "denim": ["jeans"],
        "jeggings": ["jeans", "denim", "trousers"],
        "tee": ["tshirt", "shirt", "top"],
        "tshirt": ["tee", "shirt", "top"],
        "tank": ["vest", "top", "tanktop"],
        "strappy": ["strap", "tanktop", "vest"],
        "bikini": ["swimwear"],
        "swim": ["swimwear"],
        "hipster": ["brief", "underwear"],
        "sweater": ["jumper", "knit", "knitwear"],
        "knit": ["sweater", "jumper", "knitwear"],
        "jumper": ["sweater", "knitwear"],
        "blouse": ["top"],
        "hoodie": ["hooded", "sweatshirt"],
        "shorts": ["short"],
        "raw": ["frayed"],
        "slim": ["skinny", "narrow"],
        "soft": ["cotton", "knit"],
    }
    for token in tokens:
        for value in synonym_map.get(token, []):
            if value not in token_set:
                token_set.add(value)
                expanded.append(value)
    for alias in _phrase_alias_tokens(" ".join(tokens), tokens):
        if alias not in token_set:
            token_set.add(alias)
            expanded.append(alias)
    for ngram in _ngram_tokens(tokens, max_n=3):
        if ngram not in token_set:
            token_set.add(ngram)
            expanded.append(ngram)
    return expanded


def _ngram_tokens(tokens: list[str], *, max_n: int) -> list[str]:
    clean_tokens = [token for token in tokens if token and "__" not in token]
    output: list[str] = []
    for ngram_size in range(2, max(2, int(max_n)) + 1):
        if len(clean_tokens) < ngram_size:
            continue
        for start in range(0, len(clean_tokens) - ngram_size + 1):
            output.append("__".join(clean_tokens[start : start + ngram_size]))
    return output


def _phrase_alias_tokens(text: str, tokens: list[str]) -> list[str]:
    token_set = set(tokens)
    aliases: list[str] = []
    if (
        "high waist" in text
        or "high waisted" in text
        or "highwaisted" in token_set
        or "hw" in token_set
    ):
        aliases.append("highwaist")
    if (
        "mid waist" in text
        or "midrise" in token_set
        or ("mid" in token_set and "waist" in token_set)
    ):
        aliases.append("midwaist")
    if "low waist" in text or ("low" in token_set and "waist" in token_set):
        aliases.append("lowwaist")
    if "vest top" in text or "tank top" in text or ("strappy" in token_set and "top" in token_set):
        aliases.append("tanktop")
    if (
        "bikini bottom" in text
        or "swimwear bottom" in text
        or ("swimwear" in token_set and "bottom" in token_set)
    ):
        aliases.append("bikinibottom")
    if "t shirt" in text or "tee shirt" in text:
        aliases.append("tshirt")
    if "jersey top" in text or "jersey tee" in text:
        aliases.extend(["tshirt", "top"])
    if "raw hem" in text:
        aliases.append("frayed")
    return aliases


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        return " ".join(_safe_text(item) for item in value.tolist())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_safe_text(item) for item in value)
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _as_product_ids(value: Any) -> list[str]:
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
        return _as_product_ids(parsed)
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    return [str(value)]


def _hybrid_text_weight(request: Any, default: float) -> float:
    explicit = getattr(request, "hybrid_text_weight", None)
    if explicit is not None:
        return float(explicit)
    weights = getattr(request, "hybrid_weights", None)
    if weights is None:
        return default
    if isinstance(weights, dict):
        text = float(weights.get("text", default))
        image = float(weights.get("image", 1.0 - default))
    else:
        text = float(getattr(weights, "text", default))
        image = float(getattr(weights, "image", 1.0 - default))
    total = text + image
    return default if total <= 0 else text / total
