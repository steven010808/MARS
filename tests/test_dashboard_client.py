from __future__ import annotations

import json

import httpx

from apps.dashboard.api_client import MarsApiClient
from apps.dashboard.app import (
    event_plot_frame,
    live_event_series_frame,
    live_event_trend_figure,
    model_versions_display_frame,
)


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


def test_image_search_sends_image_payload_without_query() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.read().decode())
        return httpx.Response(
            200,
            json={
                "search_type": "image",
                "results": [],
                "latency_ms": 1.0,
                "total_count": 0,
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
        response = MarsApiClient(base_url="http://api.test").search(
            query="ignored text",
            search_type="image",
            image_base64="data:image/png;base64,abc",
            top_k=5,
        )
    finally:
        httpx.Client = original_client

    assert not response.is_demo
    assert seen["payload"] == {
        "search_type": "image",
        "top_k": 5,
        "image_base64": "data:image/png;base64,abc",
    }


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


def test_live_event_series_prefers_recent_five_minute_raw_timeline() -> None:
    metrics = {
        "simulator": {
            "minute_timeline": [
                {
                    "minute": "2026-06-14T15:00:00+00:00",
                    "event_type": "search",
                    "count": 999,
                }
            ],
            "timeline": [
                {
                    "timestamp": "2026-06-14T14:54:55.100000+00:00",
                    "event_type": "purchase",
                    "product_id": "OLD",
                },
                {
                    "timestamp": "2026-06-14T15:00:01.100000+00:00",
                    "event_type": "search",
                    "event_role": "user_action",
                },
                {
                    "timestamp": "2026-06-14T15:00:06.100000+00:00",
                    "event_type": "view",
                    "product_id": "P1",
                },
                {
                    "timestamp": "2026-06-14T15:00:11.100000+00:00",
                    "event_type": "purchase",
                    "product_id": "P1",
                },
                {
                    "timestamp": "2026-06-14T15:00:11.200000+00:00",
                    "event_type": "search",
                    "event_role": "user_action",
                },
            ],
        }
    }

    series = live_event_series_frame(metrics)

    assert not series.empty
    assert series["minute"].nunique() == 3
    assert series["count"].max() == 1
    assert series["count"].sum() == 4
    diffs = (
        series[["minute"]]
        .drop_duplicates()
        .sort_values("minute")["minute"]
        .diff()
        .dropna()
        .dt.total_seconds()
        .tolist()
    )
    assert diffs == [5.0, 5.0]

    plot_frame = event_plot_frame(series, "ko")
    plot_diffs = (
        plot_frame[["minute"]]
        .drop_duplicates()
        .sort_values("minute")["minute"]
        .diff()
        .dropna()
        .dt.total_seconds()
        .tolist()
    )
    assert plot_diffs == [5.0, 5.0]

    figure = live_event_trend_figure(series, "ko")
    assert {trace.mode for trace in figure.data} == {"lines"}
    assert {trace.stackgroup for trace in figure.data} == {"1"}
    assert {trace.yaxis for trace in figure.data} == {"y"}
    assert {trace.line.shape for trace in figure.data} == {"linear"}
    for trace in figure.data:
        values = list(trace.y)
        assert values == sorted(values)
    assert max(max(trace.y) for trace in figure.data) > 1
    assert figure.layout.yaxis.title.text == "누적 이벤트 수"


def test_model_registry_timestamps_are_displayed_in_kst() -> None:
    rows = [
        {
            "version": "v0001",
            "created_at": "2026-06-09T23:43:32.162168+00:00",
            "status": "active",
            "artifact_path": "artifacts/recsys",
            "metrics_path": "artifacts/reports/metrics.json",
            "metadata": {"mode": "full"},
        }
    ]

    display = model_versions_display_frame(rows, "ko")

    assert "생성(KST)" in display.columns
    assert display.loc[0, "생성(KST)"] == "06-10 08:43"


def test_prepare_retrain_state_posts_admin_endpoint() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "prepared": True,
                "ready_to_retrain": False,
                "added_events": 9,
                "pending_new_logs": 9,
                "threshold": 10,
                "remaining_to_threshold": 1,
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
        response = MarsApiClient(base_url="http://api.test").prepare_retrain_state()
    finally:
        httpx.Client = original_client

    assert not response.is_demo
    assert response.data["prepared"] is True
    assert seen["method"] == "POST"
    assert str(seen["url"]).endswith("/api/admin/live/prepare-retrain")


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
