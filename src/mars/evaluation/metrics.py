from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import log2

import numpy as np

RelevantItems = Iterable[str] | Mapping[str, float]


def _relevance_lookup(relevant: RelevantItems) -> dict[str, float]:
    if isinstance(relevant, Mapping):
        return {str(key): float(value) for key, value in relevant.items() if float(value) > 0}
    return {str(item): 1.0 for item in relevant}


def _dedupe_ranked(ranked_items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in ranked_items:
        item_id = str(item)
        if item_id in seen:
            continue
        seen.add(item_id)
        output.append(item_id)
    return output


def mrr_at_k(
    ranked_by_query: Mapping[str, Sequence[str]],
    relevant_by_query: Mapping[str, RelevantItems],
    k: int = 10,
) -> float:
    if k <= 0 or not relevant_by_query:
        return 0.0

    reciprocal_ranks: list[float] = []
    for query_id, relevant in relevant_by_query.items():
        relevant_lookup = _relevance_lookup(relevant)
        if not relevant_lookup:
            reciprocal_ranks.append(0.0)
            continue
        ranked = _dedupe_ranked(ranked_by_query.get(query_id, ()))[:k]
        rr = 0.0
        for index, item_id in enumerate(ranked, start=1):
            if item_id in relevant_lookup:
                rr = 1.0 / index
                break
        reciprocal_ranks.append(rr)
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def ndcg_at_k(
    ranked_by_query: Mapping[str, Sequence[str]],
    relevant_by_query: Mapping[str, RelevantItems],
    k: int = 10,
) -> float:
    if k <= 0 or not relevant_by_query:
        return 0.0

    scores: list[float] = []
    for query_id, relevant in relevant_by_query.items():
        relevant_lookup = _relevance_lookup(relevant)
        if not relevant_lookup:
            scores.append(0.0)
            continue
        ranked = _dedupe_ranked(ranked_by_query.get(query_id, ()))[:k]
        dcg = 0.0
        for index, item_id in enumerate(ranked, start=1):
            gain = relevant_lookup.get(item_id, 0.0)
            if gain > 0:
                dcg += gain / log2(index + 1)
        ideal_gains = sorted(relevant_lookup.values(), reverse=True)[:k]
        idcg = sum(gain / log2(index + 1) for index, gain in enumerate(ideal_gains, start=1))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k(
    ranked_by_user: Mapping[str, Sequence[str]],
    relevant_by_user: Mapping[str, RelevantItems],
    k: int = 300,
) -> float:
    if k <= 0 or not relevant_by_user:
        return 0.0

    recalls: list[float] = []
    for user_id, relevant in relevant_by_user.items():
        relevant_set = set(_relevance_lookup(relevant))
        if not relevant_set:
            recalls.append(0.0)
            continue
        ranked_set = set(_dedupe_ranked(ranked_by_user.get(user_id, ()))[:k])
        recalls.append(len(ranked_set & relevant_set) / len(relevant_set))
    return float(np.mean(recalls)) if recalls else 0.0


def hit_rate_at_k(
    ranked_by_user: Mapping[str, Sequence[str]],
    relevant_by_user: Mapping[str, RelevantItems],
    k: int = 50,
) -> float:
    if k <= 0 or not relevant_by_user:
        return 0.0

    hits: list[float] = []
    for user_id, relevant in relevant_by_user.items():
        relevant_set = set(_relevance_lookup(relevant))
        if not relevant_set:
            hits.append(0.0)
            continue
        ranked_set = set(_dedupe_ranked(ranked_by_user.get(user_id, ()))[:k])
        hits.append(float(bool(ranked_set & relevant_set)))
    return float(np.mean(hits)) if hits else 0.0


def coverage_at_k(
    ranked_by_user: Mapping[str, Sequence[str]], catalog_items: Iterable[str], k: int = 50
) -> float:
    catalog_set = {str(item) for item in catalog_items}
    if k <= 0 or not catalog_set:
        return 0.0

    recommended: set[str] = set()
    for ranked in ranked_by_user.values():
        recommended.update(_dedupe_ranked(ranked)[:k])
    return len(recommended & catalog_set) / len(catalog_set)


def auc_score(y_true: Sequence[int | float | bool], y_score: Sequence[int | float]) -> float:
    labels = np.asarray(y_true, dtype=float)
    scores = np.asarray(y_score, dtype=float)
    if labels.size == 0 or labels.size != scores.size:
        return 0.5
    positives = labels > 0
    pos_count = int(positives.sum())
    neg_count = int(labels.size - pos_count)
    if pos_count == 0 or neg_count == 0:
        return 0.5

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(labels.size, dtype=float)
    start = 0
    while start < labels.size:
        end = start + 1
        while end < labels.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    rank_sum_pos = float(ranks[positives].sum())
    auc = (rank_sum_pos - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)
    return float(auc)


def ctr(clicks: int | float, impressions: int | float) -> float:
    impressions = float(impressions)
    return 0.0 if impressions <= 0 else float(clicks) / impressions


def cvr(conversions: int | float, visits_or_clicks: int | float) -> float:
    denominator = float(visits_or_clicks)
    return 0.0 if denominator <= 0 else float(conversions) / denominator
