from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)


def test_healthz_contract() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["app"] == "mars-api"
    assert payload["version"]
    assert payload["config_profile"]
    assert payload["status"] in {"ok", "degraded"}
    assert "services" in payload
    assert "artifacts" in payload


def test_search_text_contract() -> None:
    response = client.post(
        "/api/search",
        json={"search_type": "text", "query": "black jacket", "top_k": 5},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["search_type"] == "text"
    assert "results" in payload
    assert "latency_ms" in payload
    assert "total_count" in payload
    if payload["results"]:
        assert {"product_id", "name", "score", "price"}.issubset(payload["results"][0])


def test_search_accepts_query_text_alias_from_spec_example() -> None:
    response = client.post(
        "/api/search",
        json={"search_type": "text", "query_text": "black jacket", "top_k": 3},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["search_type"] == "text"
    assert "latency_ms" in payload
    assert "total_count" in payload


def test_search_validation_error_for_missing_query() -> None:
    response = client.post("/api/search", json={"search_type": "text", "top_k": 5})
    assert response.status_code == 422


def test_recommend_contract() -> None:
    response = client.get("/api/recommend", params={"user_id": "U00000001", "top_n": 10})
    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "U00000001"
    assert "recommendations" in payload
    assert {"candidate_ms", "ranking_ms", "reranking_ms", "total_ms"}.issubset(
        payload["pipeline_latency"]
    )
    assert "session_context" in payload
    if payload["recommendations"]:
        assert {"product_id", "score", "reason", "is_exploration"}.issubset(
            payload["recommendations"][0]
        )


def test_recommend_ab_strategy_context() -> None:
    response = client.get(
        "/api/recommend",
        params={"user_id": "U00000001", "top_n": 10, "experiment_key": "mars_default"},
    )

    assert response.status_code == 200
    context = response.json()["session_context"]
    assert context["ab_experiment_key"] == "mars_default"
    assert context["ab_bucket"] in {"control", "treatment"}
    assert context["recommendation_strategy"] in {"control", "treatment"}
    assert context["recommendation_strategy_label"] in {
        "RankOnlyControl",
        "ComplementGraphExplore",
        "RepeatExploreGate",
    }
    assert "strategy_mix" in context


def test_event_contract_and_recommend_session_context() -> None:
    event = client.post(
        "/api/events",
        json={
            "user_id": "U00000001",
            "session_id": "S-test",
            "event_type": "view",
            "product_id": "P00000001",
            "metadata": {"category": "outer"},
        },
    )
    assert event.status_code == 200
    event_payload = event.json()
    assert event_payload["accepted"] is True
    assert event_payload["event_id"]
    assert event_payload["session_id"] == "S-test"
    assert event_payload["event_type"] == "view"
    assert "redis_updated" in event_payload
    assert "durable_log" in event_payload

    recommend = client.get(
        "/api/recommend",
        params={"user_id": "U00000001", "top_n": 3, "session_id": "S-test"},
    )
    assert recommend.status_code == 200
    assert recommend.json()["session_context"]["session_id"] == "S-test"


def test_event_top_level_category_updates_session_context() -> None:
    event = client.post(
        "/api/events",
        json={
            "user_id": "U00000002",
            "session_id": "S-category-alias",
            "event_type": "view",
            "product_id": "P00000002",
            "category": "Dress",
        },
    )
    assert event.status_code == 200

    recommend = client.get(
        "/api/recommend",
        params={"user_id": "U00000002", "top_n": 3, "session_id": "S-category-alias"},
    )
    assert recommend.status_code == 200
    context = recommend.json()["session_context"]
    assert "Dress" in context.get("recent_categories", [])
    assert context.get("session_interest") == "Dress"


def test_metrics_contract() -> None:
    response = client.get("/api/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert {"search", "recommendation", "system", "artifacts"}.issubset(payload)


def test_ab_assign_is_deterministic() -> None:
    request = {"user_id": "U00000001", "experiment_key": "homepage"}
    first = client.post("/api/ab/assign", json=request)
    second = client.post("/api/ab/assign", json=request)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["bucket"] == second.json()["bucket"]
    assert first.json()["assignment_method"] == "deterministic_hash"


def test_ab_report_contract() -> None:
    response = client.get("/api/ab/report", params={"experiment_key": "homepage"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["experiment_key"] == "homepage"
    assert "buckets" in payload
    assert "p_value" in payload
    assert "confidence_interval_95" in payload
    assert len(payload["confidence_interval_95"]) == 2
