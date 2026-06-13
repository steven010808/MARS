from __future__ import annotations

import httpx

from apps.dashboard.api_client import MarsApiClient


def test_metrics_falls_back_to_demo_data() -> None:
    client = MarsApiClient(base_url="http://127.0.0.1:9", timeout=0.01)
    response = client.metrics()

    assert response.is_demo
    assert "system" in response.data
    assert "search" in response.data
    assert "recommendation" in response.data


def test_search_fallback_preserves_required_contract() -> None:
    client = MarsApiClient(base_url="http://127.0.0.1:9", timeout=0.01)
    response = client.search("black minimal jacket")

    assert response.is_demo
    assert {"search_type", "results", "latency_ms", "total_count"}.issubset(response.data)
    assert {"product_id", "name", "score", "price"}.issubset(response.data["results"][0])


def test_recommend_fallback_preserves_required_contract() -> None:
    client = MarsApiClient(base_url="http://127.0.0.1:9", timeout=0.01)
    response = client.recommend("U000001")

    assert response.is_demo
    assert {"user_id", "recommendations", "pipeline_latency", "session_context"}.issubset(
        response.data
    )
    assert {"product_id", "score", "reason", "is_exploration"}.issubset(
        response.data["recommendations"][0]
    )


def test_live_metrics_response_uses_api_source() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
    original_client = httpx.Client

    class MockedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    httpx.Client = MockedClient
    try:
        response = MarsApiClient(base_url="http://api.test").metrics()
    finally:
        httpx.Client = original_client

    assert not response.is_demo
    assert response.data == {"ok": True}


def test_record_event_sends_search_feedback_payload() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["payload"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "accepted": True,
                "event_id": "E1",
                "session_id": "S1",
                "event_type": "view",
                "redis_updated": True,
                "durable_log": "logs/api_events.jsonl",
            },
        )

    transport = httpx.MockTransport(handler)
    original_client = httpx.Client

    class MockedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    httpx.Client = MockedClient
    try:
        response = MarsApiClient(base_url="http://api.test").record_event(
            user_id="dashboard-search-user",
            event_type="view",
            product_id="P1",
            session_id="S1",
            query="white shirts",
            metadata={"source_surface": "search", "search_id": "Q1", "rank": 1},
        )
    finally:
        httpx.Client = original_client

    assert not response.is_demo
    assert seen["method"] == "POST"
    assert str(seen["url"]).endswith("/api/events")
    assert '"search_id":"Q1"' in str(seen["payload"])
    assert '"rank":1' in str(seen["payload"])
