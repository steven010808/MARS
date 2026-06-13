"""Compatibility imports for the assignment's expected evaluation path."""

from mars.evaluation.metrics import (
    auc_score,
    coverage_at_k,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)

__all__ = [
    "auc_score",
    "coverage_at_k",
    "hit_rate_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
]
