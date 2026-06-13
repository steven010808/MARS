from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from mars.schemas import EVENT_TYPES


class SearchType(StrEnum):
    text = "text"
    image = "image"
    hybrid = "hybrid"


class EventType(StrEnum):
    search = "search"
    view = "view"
    cart = "cart"
    purchase = "purchase"


class SearchFilters(BaseModel):
    category: str | None = None
    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_price_range(self) -> SearchFilters:
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError("min_price must be less than or equal to max_price")
        return self


class HybridWeights(BaseModel):
    text: float = Field(default=0.5, ge=0.0, le=1.0)
    image: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_non_zero(self) -> HybridWeights:
        if self.text + self.image <= 0:
            raise ValueError("At least one hybrid weight must be positive")
        return self


class SearchRequest(BaseModel):
    user_id: str | None = Field(default=None, min_length=1)
    session_id: str | None = Field(default=None, min_length=1)
    query: str | None = Field(default=None, min_length=1)
    image_base64: str | None = Field(default=None, min_length=1)
    image_url: str | None = Field(default=None, min_length=1)
    image_path: str | None = Field(default=None, min_length=1)
    search_type: SearchType = SearchType.text
    top_k: int = Field(default=10, ge=1, le=100)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    hybrid_weights: HybridWeights = Field(default_factory=HybridWeights)

    @model_validator(mode="before")
    @classmethod
    def normalize_query_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if not data.get("query") and data.get("query_text"):
                data["query"] = data["query_text"]
            if not data.get("image_url") and data.get("image_path"):
                data["image_url"] = data["image_path"]
        return data

    @model_validator(mode="after")
    def validate_modality(self) -> SearchRequest:
        has_text = bool(self.query)
        has_image = bool(self.image_base64 or self.image_url or self.image_path)
        if self.search_type == SearchType.text and not has_text:
            raise ValueError("query is required for text search")
        if self.search_type == SearchType.image and not has_image:
            raise ValueError("image_base64, image_url, or image_path is required for image search")
        if self.search_type == SearchType.hybrid and not (has_text or has_image):
            raise ValueError("query or image is required for hybrid search")
        return self


class SearchResult(BaseModel):
    product_id: str
    name: str
    score: float
    price: float
    category: str | None = None
    image_url: str | None = None


class SearchResponse(BaseModel):
    search_type: SearchType
    results: list[SearchResult]
    latency_ms: float
    total_count: int
    debug: dict[str, Any] = Field(default_factory=dict)


class PipelineLatency(BaseModel):
    candidate_ms: float
    ranking_ms: float
    reranking_ms: float
    total_ms: float


class RecommendationItem(BaseModel):
    product_id: str
    score: float
    reason: str
    is_exploration: bool
    arm: str | None = None
    name: str | None = None
    category: str | None = None
    price: float | None = None


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: list[RecommendationItem]
    pipeline_latency: PipelineLatency
    session_context: dict[str, Any]


class EventRequest(BaseModel):
    user_id: str = Field(min_length=1)
    event_type: EventType
    product_id: str | None = None
    session_id: str | None = None
    query: str | None = None
    arm: str | None = None
    timestamp: str | None = None
    source: str | None = None
    category: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_type")
    @classmethod
    def validate_known_event_type(cls, value: EventType) -> EventType:
        if value.value not in EVENT_TYPES:
            raise ValueError(f"Unknown event_type: {value}")
        return value

    @model_validator(mode="after")
    def normalize_category_metadata(self) -> EventRequest:
        if self.category and not self.metadata.get("category"):
            self.metadata = {**self.metadata, "category": self.category}
        return self


class EventResponse(BaseModel):
    accepted: bool
    event_id: str
    session_id: str
    event_type: EventType
    redis_updated: bool
    durable_log: str


class MetricsResponse(BaseModel):
    mode: str | None = None
    artifact_readiness: dict[str, Any] = Field(default_factory=dict)
    search: dict[str, Any]
    recommendation: dict[str, Any]
    system: dict[str, Any]
    data_quality: dict[str, Any] = Field(default_factory=dict)
    simulator: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any]


class ABAssignRequest(BaseModel):
    user_id: str = Field(min_length=1)
    experiment_key: str = Field(default="default", min_length=1)
    buckets: list[str] = Field(default_factory=lambda: ["control", "treatment"])

    @field_validator("buckets")
    @classmethod
    def validate_buckets(cls, value: list[str]) -> list[str]:
        if len(value) < 2:
            raise ValueError("At least two buckets are required")
        if len(set(value)) != len(value):
            raise ValueError("Buckets must be unique")
        return value


class ABAssignResponse(BaseModel):
    user_id: str
    experiment_key: str
    bucket: str
    assignment_method: Literal["deterministic_hash"] = "deterministic_hash"


class ABReportResponse(BaseModel):
    experiment_key: str
    buckets: dict[str, dict[str, float | int]]
    uplift: float
    uplift_by_metric: dict[str, float] = Field(default_factory=dict)
    p_value: float
    confidence_interval_95: list[float]
    method: Literal["two_proportion_z_test"] = "two_proportion_z_test"


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app: str
    version: str
    config_profile: str
    services: dict[str, str]
    artifacts: dict[str, bool]
