from __future__ import annotations

from mars.config.settings import MarsConfig
from mars.evaluation.metrics import (
    auc_score,
    coverage_at_k,
    ctr,
    cvr,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from mars.evaluation.runner import _select_search_primary


def test_ranking_metrics_on_toy_fixture() -> None:
    ranked = {
        "q1": ["p3", "p1", "p2"],
        "q2": ["p4", "p5"],
    }
    relevant = {
        "q1": {"p1": 2.0, "p2": 1.0},
        "q2": {"p5": 1.0},
    }

    assert mrr_at_k(ranked, relevant, k=3) == 0.5
    assert round(ndcg_at_k(ranked, relevant, k=3), 6) == 0.650301
    assert recall_at_k(ranked, relevant, k=2) == 0.75
    assert hit_rate_at_k(ranked, relevant, k=1) == 0.0
    assert hit_rate_at_k(ranked, relevant, k=2) == 1.0


def test_coverage_auc_ctr_cvr() -> None:
    ranked = {"u1": ["p1", "p2"], "u2": ["p2", "p3"]}
    assert coverage_at_k(ranked, ["p1", "p2", "p3", "p4"], k=2) == 0.75
    assert auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75
    assert ctr(3, 10) == 0.3
    assert cvr(2, 4) == 0.5


def test_search_quality_status_requires_all_search_targets() -> None:
    config = MarsConfig(raw={"evaluation": {"search_primary": "supervised_qrels_test_split"}})
    passing = _select_search_primary(
        config,
        {
            "mrr_at_10": 0.60,
            "ndcg_at_10": 0.55,
            "latency_p95_ms": 150.0,
            "text_latency_p95_ms": 140.0,
            "image_latency_p95_ms": 150.0,
            "hybrid_latency_p95_ms": 145.0,
            "multimodal_latency_sample_size": 50,
        },
    )
    failing_quality = _select_search_primary(
        config,
        {
            "mrr_at_10": 0.54,
            "ndcg_at_10": 0.55,
            "latency_p95_ms": 150.0,
            "text_latency_p95_ms": 140.0,
            "image_latency_p95_ms": 150.0,
            "hybrid_latency_p95_ms": 145.0,
            "multimodal_latency_sample_size": 50,
        },
    )
    failing_latency = _select_search_primary(
        config,
        {
            "mrr_at_10": 0.60,
            "ndcg_at_10": 0.55,
            "latency_p95_ms": 201.0,
            "text_latency_p95_ms": 140.0,
            "image_latency_p95_ms": 201.0,
            "hybrid_latency_p95_ms": 145.0,
            "multimodal_latency_sample_size": 50,
        },
    )
    missing_multimodal = _select_search_primary(
        config,
        {"mrr_at_10": 0.60, "ndcg_at_10": 0.55, "latency_p95_ms": 150.0},
    )

    assert passing["quality_status"] == "pass"
    assert failing_quality["quality_status"] == "fail"
    assert failing_latency["quality_status"] == "fail"
    assert missing_multimodal["quality_status"] == "fail"
