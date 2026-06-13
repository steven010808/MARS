from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EpsilonGreedyMAB:
    epsilon: float = 0.10
    arms: tuple[str, ...] = ("relevance", "diversity", "novelty", "price_match")
    stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for arm in self.arms:
            self.stats.setdefault(arm, {"successes": 1.0, "trials": 2.0})

    def choose_arm(self, user_id: str, request_id: str = "") -> str:
        draw = _stable_unit(f"{user_id}:{request_id}:explore")
        if draw < self.epsilon:
            index = int(_stable_unit(f"{user_id}:{request_id}:arm") * len(self.arms)) % len(
                self.arms
            )
            return self.arms[index]
        return max(self.arms, key=self._mean_reward)

    def update(self, arm: str, reward: float) -> None:
        if arm not in self.stats:
            self.stats[arm] = {"successes": 1.0, "trials": 2.0}
        self.stats[arm]["successes"] += max(0.0, float(reward))
        self.stats[arm]["trials"] += 1.0

    def _mean_reward(self, arm: str) -> float:
        stat = self.stats[arm]
        return stat["successes"] / max(stat["trials"], 1.0)


def rerank_candidates(
    candidates: list[dict[str, Any]],
    *,
    top_n: int,
    exploration_slots: int,
    max_same_category_streak: int,
    trending_products: list[dict[str, Any]],
    mab: EpsilonGreedyMAB,
    user_id: str,
    request_id: str = "",
) -> list[dict[str, Any]]:
    if top_n <= 0:
        return []
    arm = mab.choose_arm(user_id, request_id)
    ranked = _apply_arm(candidates, arm)
    selected = _select_with_category_guard(ranked, top_n, max_same_category_streak)

    if exploration_slots > 0 and trending_products:
        selected_ids = {str(item["product"]["product_id"]) for item in selected}
        exploration = []
        keep = max(0, top_n - exploration_slots)
        base_selection = selected[:keep]
        for product in trending_products:
            product_id = str(product.get("product_id"))
            if product_id in selected_ids:
                continue
            candidate = {
                "product": product,
                "candidate_score": 0.0,
                "ranking_score": 0.35 + float(product.get("popularity_prior", 0.0) or 0.0) * 0.2,
                "reason": f"exploration:{arm}",
                "is_exploration": True,
                "arm": arm,
            }
            if _would_break_streak(
                base_selection + exploration, candidate, max_same_category_streak
            ):
                continue
            exploration.append(candidate)
            selected_ids.add(product_id)
            if len(exploration) >= exploration_slots:
                break
        if exploration:
            selected = selected[:keep] + exploration

    for item in selected:
        item.setdefault("arm", arm)
        item.setdefault("is_exploration", False)
    return selected[:top_n]


def _apply_arm(candidates: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    def score(item: dict[str, Any]) -> float:
        product = item["product"]
        base = float(item.get("ranking_score", 0.0))
        if arm == "diversity":
            return base + _stable_unit(str(product.get("category_l1", ""))) * 0.08
        if arm == "novelty":
            return base + (0.15 if product.get("is_new") else 0.0)
        if arm == "price_match":
            price = float(product.get("price", 0) or 0)
            return base + max(0.0, 1.0 - min(price, 300_000.0) / 300_000.0) * 0.08
        return base

    return sorted(candidates, key=score, reverse=True)


def _select_with_category_guard(
    candidates: list[dict[str, Any]],
    top_n: int,
    max_same_category_streak: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    pending = list(candidates)
    while pending and len(selected) < top_n:
        chosen_index = 0
        for idx, candidate in enumerate(pending):
            if not _would_break_streak(selected, candidate, max_same_category_streak):
                chosen_index = idx
                break
        selected.append(pending.pop(chosen_index))
    return selected


def _would_break_streak(
    selected: list[dict[str, Any]],
    candidate: dict[str, Any],
    max_same_category_streak: int,
) -> bool:
    if max_same_category_streak <= 0 or len(selected) < max_same_category_streak:
        return False
    category = candidate["product"].get("category_l1", candidate["product"].get("category"))
    streak = selected[-max_same_category_streak:]
    return all(
        item["product"].get("category_l1", item["product"].get("category")) == category
        for item in streak
    )


def _stable_unit(token: str) -> float:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)
