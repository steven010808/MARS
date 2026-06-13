from __future__ import annotations

from dataclasses import dataclass

PERSONAS = (
    "trendsetter",
    "pragmatist",
    "value_seeker",
    "top_category_loyalist",
    "impulse_buyer",
    "careful_explorer",
)

EVENT_TYPES = ("search", "view", "cart", "purchase")


@dataclass(frozen=True)
class ProductRecord:
    product_id: str
    name: str
    category_l1: str
    category_l2: str
    category_l3: str
    price: int
    color: str
    style_tags: list[str]
    description: str
    image_path: str
    created_at: str
    popularity_prior: float
    margin_score: float
    is_new: bool


@dataclass(frozen=True)
class UserRecord:
    user_id: str
    persona: str
    age_bucket: str
    gender: str
    preferred_categories: list[str]
    price_sensitivity: float
    trend_affinity: float
    category_loyalty: float
    exploration_rate: float
    session_frequency: float
    created_at: str


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    user_id: str
    session_id: str
    event_type: str
    product_id: str | None
    query: str | None
    query_intent_category: str | None
    timestamp: str
    rank_position: int | None
    source: str
    device: str
    persona: str
    price_at_event: int | None
    dwell_time_sec: float | None
    is_positive: bool
    event_weight: float
