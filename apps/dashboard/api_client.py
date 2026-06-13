from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

from apps.dashboard import demo_data


@dataclass(frozen=True)
class DashboardResponse:
    data: dict[str, Any]
    source: str
    error: str | None = None

    @property
    def is_demo(self) -> bool:
        return self.source == "demo"


class MarsApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 12.0) -> None:
        default_url = (
            os.getenv("MARS_API_BASE_URL") or os.getenv("API_BASE_URL") or "http://api:8000"
        )
        self.base_url = (base_url or default_url).rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        fallback: dict[str, Any],
        **kwargs: Any,
    ) -> DashboardResponse:
        try:
            with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
                response = client.request(method, path, **kwargs)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("API returned a non-object JSON payload")
                return DashboardResponse(data=payload, source="api")
        except Exception as exc:  # Streamlit must stay usable during backend work.
            return DashboardResponse(data=fallback, source="demo", error=str(exc))

    def health(self) -> DashboardResponse:
        return self._request("GET", "/healthz", fallback={"status": "demo", "ready": False})

    def metrics(self) -> DashboardResponse:
        return self._request("GET", "/api/metrics", fallback=demo_data.metrics_payload())

    def search(
        self,
        query: str = "",
        search_type: str = "text",
        top_k: int = 10,
        image_url: str | None = None,
    ) -> DashboardResponse:
        payload: dict[str, Any] = {"search_type": search_type, "top_k": top_k}
        if query and search_type in {"text", "hybrid"}:
            payload["query"] = query
        if image_url and search_type in {"image", "hybrid"}:
            payload["image_url"] = image_url
        return self._request(
            "POST",
            "/api/search",
            json=payload,
            fallback=demo_data.search_payload(query),
        )

    def record_event(
        self,
        *,
        user_id: str,
        event_type: str,
        product_id: str,
        session_id: str,
        query: str,
        metadata: dict[str, Any],
    ) -> DashboardResponse:
        payload = {
            "user_id": user_id,
            "event_type": event_type,
            "product_id": product_id,
            "session_id": session_id,
            "query": query,
            "metadata": metadata,
        }
        return self._request(
            "POST",
            "/api/events",
            json=payload,
            fallback={"accepted": False, "event_type": event_type, "product_id": product_id},
        )

    def recommend(
        self,
        user_id: str,
        top_n: int = 10,
        session_id: str | None = None,
        experiment_key: str | None = None,
    ) -> DashboardResponse:
        params: dict[str, Any] = {"user_id": user_id, "top_n": top_n}
        if session_id:
            params["session_id"] = session_id
        if experiment_key:
            params["experiment_key"] = experiment_key
        return self._request(
            "GET",
            "/api/recommend",
            params=params,
            fallback=demo_data.recommend_payload(user_id),
        )

    def ab_report(self, experiment_key: str = "mars_default") -> DashboardResponse:
        return self._request(
            "GET",
            "/api/ab/report",
            params={"experiment_key": experiment_key},
            fallback=demo_data.ab_report_payload(),
        )
