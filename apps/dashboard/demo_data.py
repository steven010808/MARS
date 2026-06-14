from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any


def _series(base: float, step: float, count: int = 14) -> list[dict[str, Any]]:
    start = datetime.now(UTC) - timedelta(days=count - 1)
    return [
        {
            "date": (start + timedelta(days=i)).date().isoformat(),
            "value": round(base + step * i + ((i % 3) - 1) * step * 0.6, 4),
        }
        for i in range(count)
    ]


def metrics_payload() -> dict[str, Any]:
    return {
        "mode": "demo",
        "artifact_readiness": {
            "data": True,
            "search_index": True,
            "recommender": True,
            "reports": True,
        },
        "system": {
            "events": 50000,
            "base_events": 50000,
            "logged_events": 7420,
            "live_events": 7420,
            "total_events": 57420,
            "products": 5000,
            "users": 1000,
            "ctr": 0.048,
            "cvr": 0.011,
            "api_latency_p95_ms": 142.6,
            "redis_latency_ms": 3.8,
            "active_model_version": "demo-2026-04-15-001",
        },
        "search": {
            "mrr": 0.61,
            "ndcg_at_10": 0.54,
            "recall_at_10": 0.68,
            "latency_p50_ms": 51.2,
            "latency_p95_ms": 138.4,
            "encoder_type": "fallback-demo",
            "target_status": {
                "mrr": "met",
                "ndcg_at_10": "met",
                "latency_p95_ms": "met",
            },
        },
        "recommendation": {
            "recall_at_300": 0.34,
            "auc": 0.73,
            "hitrate_at_50": 0.23,
            "ndcg_at_50": 0.094,
            "coverage": 0.31,
            "candidate_p95_ms": 37.4,
            "ranking_p95_ms": 63.7,
            "reranking_p95_ms": 12.9,
            "total_p95_ms": 121.8,
        },
        "simulator": {
            "personas": {
                "trendsetter": 0.18,
                "pragmatist": 0.19,
                "bargain_hunter": 0.17,
                "top_category_loyalist": 0.16,
                "impulse_buyer": 0.15,
                "careful_researcher": 0.15,
            },
            "events": {
                "search": 0.24,
                "view": 0.53,
                "cart": 0.15,
                "purchase": 0.08,
            },
            "timeline": _series(2600, 185),
        },
        "training": {
            "status": "watching",
            "new_logs": 7420,
            "new_logs_threshold": 10000,
            "ctr": 0.048,
            "cvr": 0.011,
            "ctr_source": "demo:surface:recommendation",
            "cvr_source": "demo:surface:recommendation",
            "ctr_threshold": 0.03,
            "hitrate_threshold": 0.20,
            "last_retrain": "2026-04-15T04:30:00Z",
            "next_action": "No retrain yet. Waiting for 2580 more events.",
            "versions": [
                {"version": "demo-2026-04-15-001", "status": "active", "hitrate": 0.23},
                {"version": "demo-2026-04-14-002", "status": "archived", "hitrate": 0.21},
            ],
        },
    }


def search_payload(query: str = "black minimal jacket") -> dict[str, Any]:
    products = [
        ("P00001042", "Black Minimal City Jacket", 0.873, 129000, "outer"),
        ("P00003210", "Charcoal Utility Windbreaker", 0.842, 99000, "outer"),
        ("P00000498", "Graphite Commuter Blazer", 0.817, 159000, "outer"),
        ("P00002941", "Ink Water-Resistant Shell", 0.798, 119000, "outer"),
        ("P00001882", "Matte Black Cropped Jacket", 0.781, 89000, "outer"),
    ]
    return {
        "search_type": "text",
        "query": query,
        "latency_ms": 64.7,
        "total_count": len(products),
        "results": [
            {
                "product_id": product_id,
                "name": name,
                "score": score,
                "price": price,
                "category": category,
                "image_url": "",
            }
            for product_id, name, score, price, category in products
        ],
    }


def recommend_payload(user_id: str = "U000001") -> dict[str, Any]:
    rows = [
        ("P00004311", "Soft Tech Cargo Pants", 0.931, "recent views lean utility", False),
        ("P00001270", "Lime Accent Knit Vest", 0.884, "persona trend affinity", True),
        ("P00003880", "Daily Water-Repel Sneaker", 0.862, "high category match", False),
        ("P00000937", "Cyan Stitch Shoulder Bag", 0.831, "diversity slot", True),
        ("P00002725", "Quiet Luxury Wool Coat", 0.807, "category preference", False),
    ]
    return {
        "user_id": user_id,
        "recommendations": [
            {
                "product_id": product_id,
                "name": name,
                "score": score,
                "reason": reason,
                "is_exploration": explore,
            }
            for product_id, name, score, reason, explore in rows
        ],
        "pipeline_latency": {
            "candidate_ms": 25.2,
            "ranking_ms": 48.1,
            "reranking_ms": 9.4,
            "total_ms": 82.7,
        },
        "session_context": {
            "recent_products": ["P00000212", "P00001042", "P00004311"],
            "recent_categories": ["outer", "pants", "sneakers"],
            "redis_status": "demo",
        },
    }


def ab_report_payload() -> dict[str, Any]:
    return {
        "experiment_key": "mars_default",
        "buckets": {
            "control": {
                "impressions": 24120,
                "clicks": 1041,
                "conversions": 224,
                "ctr": 0.0432,
                "cvr": 0.0093,
                "purchase_per_click": 0.2152,
            },
            "treatment": {
                "impressions": 23944,
                "clicks": 1236,
                "conversions": 281,
                "ctr": 0.0516,
                "cvr": 0.0117,
                "purchase_per_click": 0.2273,
            },
        },
        "uplift": 0.0024,
        "uplift_by_metric": {"ctr": 0.0084, "cvr": 0.0024},
        "p_value": 0.0037,
        "confidence_interval_95": [0.0031, 0.0138],
        "significant": True,
        "method": "two_proportion_z_test",
    }
