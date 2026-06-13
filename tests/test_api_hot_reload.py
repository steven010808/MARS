from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from apps.api import service_adapters
from apps.api.schemas import SearchRequest
from apps.api.service_adapters import ApiRuntime
from mars.config.settings import MarsConfig, ModeConfig, PathsConfig


class FakeRecommendationService:
    created = 0

    def __init__(self, config: MarsConfig) -> None:
        FakeRecommendationService.created += 1
        self.name = f"fake-rec-{FakeRecommendationService.created}"
        self.artifacts = SimpleNamespace(version=f"artifact-{FakeRecommendationService.created}")
        self.session_store = None

    def recommend(self, user_id: str, top_n: int, session_id: str | None = None) -> dict:
        return {
            "user_id": user_id,
            "recommendations": [
                {
                    "product_id": "P00000001",
                    "score": 0.9,
                    "reason": self.name,
                    "is_exploration": False,
                }
            ][:top_n],
            "pipeline_latency": {
                "candidate_ms": 1.0,
                "ranking_ms": 1.0,
                "reranking_ms": 1.0,
                "total_ms": 3.0,
            },
            "session_context": {"session_id": session_id, "recent_clicks": []},
        }


class FakeSearchService:
    created = 0

    def __init__(self, config: MarsConfig) -> None:
        FakeSearchService.created += 1
        self.name = f"fake-search-{FakeSearchService.created}"

    def search(self, request) -> dict:
        return {
            "search_type": "text",
            "results": [
                {
                    "product_id": "P00000001",
                    "name": self.name,
                    "score": 1.0,
                    "price": 1000.0,
                }
            ],
            "latency_ms": 1.0,
            "total_count": 1,
            "debug": {"service": self.name},
        }


def _config(tmp_path: Path) -> MarsConfig:
    return MarsConfig(
        active_mode="dev",
        modes={
            "dev": ModeConfig(products=10, users=10, events=10),
            "full": ModeConfig(products=50_000, users=10_000, events=1_000_000),
        },
        paths=PathsConfig(
            data_dir=tmp_path / "data",
            processed_dir=tmp_path / "data" / "processed",
            raw_dir=tmp_path / "data" / "raw",
            artifacts_dir=tmp_path / "artifacts",
            logs_dir=tmp_path / "logs",
        ),
    )


def _write_registry(config: MarsConfig, version: str) -> None:
    path = config.paths.artifacts_dir / "registry" / "models.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "active_version": version,
                "versions": [{"version": version, "status": "active"}],
            }
        ),
        encoding="utf-8",
    )


def _write_search_registry(config: MarsConfig, version: str) -> None:
    path = config.paths.artifacts_dir / "registry" / "search_models.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "active_version": version,
                "versions": [{"version": version, "status": "active"}],
            }
        ),
        encoding="utf-8",
    )


def test_recommendation_service_hot_reloads_when_active_registry_version_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    FakeRecommendationService.created = 0
    config = _config(tmp_path)
    _write_registry(config, "v0001")

    monkeypatch.setattr(ApiRuntime, "_connect_redis", lambda self: None)
    monkeypatch.setattr(
        service_adapters,
        "_load_class",
        lambda module_path, class_name: (
            FakeRecommendationService if class_name == "RecommendationService" else None
        ),
    )

    runtime = ApiRuntime(config)
    first_service = runtime.recommendation_service
    assert runtime._served_recommendation_version == "v0001"

    _write_registry(config, "v0002")
    response = runtime.recommend("U00000001", 1, "S-hot")

    assert response.recommendations[0].reason == "fake-rec-2"
    assert runtime.recommendation_service is not first_service
    assert runtime._served_recommendation_version == "v0002"
    assert runtime._recommendation_reload_error is None


def test_search_service_hot_reloads_when_active_registry_version_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    FakeSearchService.created = 0
    config = _config(tmp_path)
    _write_search_registry(config, "sv0001")

    monkeypatch.setattr(ApiRuntime, "_connect_redis", lambda self: None)
    monkeypatch.setattr(
        service_adapters,
        "_load_class",
        lambda module_path, class_name: (
            FakeSearchService if class_name == "SearchService" else None
        ),
    )

    runtime = ApiRuntime(config)
    first_service = runtime.search_service
    assert runtime._served_search_version == "sv0001"

    _write_search_registry(config, "sv0002")
    response = runtime.search(SearchRequest(search_type="text", query="red sneaker", top_k=1))

    assert response.results[0].name == "fake-search-2"
    assert runtime.search_service is not first_service
    assert runtime._served_search_version == "sv0002"
    assert runtime._search_reload_error is None
