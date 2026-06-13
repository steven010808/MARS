from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((p / 100) * (len(ordered) - 1))))
    return ordered[index]


def timed_request(client: httpx.Client, method: str, path: str, **kwargs: Any) -> float:
    start = time.perf_counter()
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    return (time.perf_counter() - start) * 1000


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "mean_ms": round(statistics.fmean(values), 3) if values else 0.0,
        "p50_ms": round(percentile(values, 50), 3),
        "p95_ms": round(percentile(values, 95), 3),
        "max_ms": round(max(values), 3) if values else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MARS lightweight API performance benchmark")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--search-requests", type=int, default=100)
    parser.add_argument("--recommend-requests", type=int, default=200)
    parser.add_argument("--output-dir", default="artifacts/reports")
    args = parser.parse_args()

    search_latencies: list[float] = []
    reco_latencies: list[float] = []
    queries = [
        "black minimal jacket",
        "lime knit vest",
        "water resistant sneakers",
        "quiet luxury wool coat",
        "cyan shoulder bag",
    ]

    with httpx.Client(base_url=args.base_url, timeout=15.0) as client:
        for _ in range(10):
            client.post(
                "/api/search", json={"query": queries[0], "search_type": "text", "top_k": 10}
            )
            client.get("/api/recommend", params={"user_id": "U000001", "top_n": 10})

        for i in range(args.search_requests):
            search_latencies.append(
                timed_request(
                    client,
                    "POST",
                    "/api/search",
                    json={"query": queries[i % len(queries)], "search_type": "text", "top_k": 10},
                )
            )

        for i in range(args.recommend_requests):
            reco_latencies.append(
                timed_request(
                    client,
                    "GET",
                    "/api/recommend",
                    params={
                        "user_id": f"U{i % 1000:06d}",
                        "top_n": 10,
                        "session_id": f"S-perf-{i}",
                    },
                )
            )

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "search": summarize(search_latencies),
        "recommendation": summarize(reco_latencies),
        "targets": {
            "search_p95_ms": 200,
            "feature_store_ms": 10,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"performance_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
