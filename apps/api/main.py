from __future__ import annotations

import os

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from apps.api.schemas import (
    ABAssignRequest,
    ABAssignResponse,
    ABReportResponse,
    EventRequest,
    EventResponse,
    HealthResponse,
    MetricsResponse,
    RecommendationResponse,
    SearchRequest,
    SearchResponse,
)
from apps.api.service_adapters import ApiRuntime
from mars.config.settings import ensure_runtime_dirs, load_config

APP_VERSION = "0.1.0"


def create_app() -> FastAPI:
    config = load_config(
        os.getenv("MARS_CONFIG", "configs/config.yaml"),
        mode=os.getenv("MARS_MODE"),
    )
    ensure_runtime_dirs(config)
    runtime = ApiRuntime(config)

    app = FastAPI(
        title="MARS API",
        version=APP_VERSION,
        description="Multimodal Adaptive Retrieval & Suggestion API",
    )
    app.state.config = config
    app.state.runtime = runtime
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        services = runtime.services_status()
        artifacts = runtime.artifacts_status()
        status = (
            "ok"
            if (
                services["search"] == "ready"
                and services["recommendation"] == "ready"
                and services["redis"] == "ready"
                and artifacts.get("processed_data")
                and artifacts.get("search_index")
                and artifacts.get("recsys_models")
            )
            else "degraded"
        )
        return HealthResponse(
            status=status,
            app="mars-api",
            version=APP_VERSION,
            config_profile=config.active_mode,
            services=services,
            artifacts=artifacts,
        )

    @app.post("/api/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        return runtime.search(request)

    @app.get("/api/recommend", response_model=RecommendationResponse)
    def recommend(
        user_id: str = Query(min_length=1),
        top_n: int = Query(default=10, ge=1, le=100),
        session_id: str | None = None,
        experiment_key: str | None = Query(default=None, min_length=1),
    ) -> RecommendationResponse:
        return runtime.recommend(
            user_id=user_id,
            top_n=top_n,
            session_id=session_id,
            experiment_key=experiment_key,
        )

    @app.post("/api/events", response_model=EventResponse)
    def record_event(request: EventRequest) -> EventResponse:
        return runtime.record_event(request)

    @app.get("/api/metrics", response_model=MetricsResponse)
    def metrics() -> MetricsResponse:
        return MetricsResponse.model_validate(runtime.metrics())

    @app.post("/api/ab/assign", response_model=ABAssignResponse)
    def assign_ab(request: ABAssignRequest) -> ABAssignResponse:
        return ABAssignResponse(
            user_id=request.user_id,
            experiment_key=request.experiment_key,
            bucket=runtime.assign_bucket(request),
        )

    @app.get("/api/ab/report", response_model=ABReportResponse)
    def ab_report(experiment_key: str = "mars_default") -> ABReportResponse:
        return runtime.ab_report(experiment_key)

    return app


app = create_app()
