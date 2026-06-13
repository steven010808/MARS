from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import httpx


def wait_for_health(base_url: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}/healthz", timeout=3.0)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1.0)
    raise RuntimeError(f"API did not become healthy within {timeout_s:.0f}s: {last_error}")


def assert_keys(payload: dict[str, Any], keys: set[str], label: str) -> None:
    missing = keys.difference(payload)
    if missing:
        raise AssertionError(f"{label} missing keys: {sorted(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="MARS API smoke test")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=10.0) as client:
        health = wait_for_health(args.base_url, args.timeout)
        print(f"healthz: {health}")

        search = client.post(
            "/api/search",
            json={"query": "black minimal jacket", "search_type": "text", "top_k": 5},
        )
        search.raise_for_status()
        search_payload = search.json()
        assert_keys(
            search_payload, {"search_type", "results", "latency_ms", "total_count"}, "search"
        )
        if search_payload["results"]:
            assert_keys(
                search_payload["results"][0],
                {"product_id", "name", "score", "price"},
                "search result",
            )
        print(f"search: {search_payload.get('total_count')} results")

        rec = client.get(
            "/api/recommend", params={"user_id": "U000001", "top_n": 10, "session_id": "S-smoke"}
        )
        rec.raise_for_status()
        rec_payload = rec.json()
        assert_keys(
            rec_payload,
            {"user_id", "recommendations", "pipeline_latency", "session_context"},
            "recommend",
        )
        if rec_payload["recommendations"]:
            assert_keys(
                rec_payload["recommendations"][0],
                {"product_id", "score", "reason", "is_exploration"},
                "recommendation",
            )
        print(f"recommend: {len(rec_payload.get('recommendations', []))} items")

        event_payload = {
            "user_id": "U000001",
            "session_id": "S-smoke",
            "event_type": "view",
            "product_id": rec_payload["recommendations"][0]["product_id"]
            if rec_payload["recommendations"]
            else "P00000001",
            "source": "recommend",
        }
        event = client.post("/api/events", json=event_payload)
        event.raise_for_status()
        print("event: accepted")

        metrics = client.get("/api/metrics")
        metrics.raise_for_status()
        print("metrics: accepted")

        ab_report = client.get("/api/ab/report", params={"experiment_key": "mars_default"})
        ab_report.raise_for_status()
        assert_keys(ab_report.json(), {"p_value", "confidence_interval_95"}, "ab report")
        print("ab report: accepted")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
