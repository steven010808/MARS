# Code Traceability

## Scope

This matrix traces the final capstone requirements to implementation files, runtime artifacts, and verification coverage. It uses the active MARS runtime as the primary source of truth. The only remaining `src/` path is a compatibility wrapper for the spec wording around `src/evaluation/metrics.py`.

## Requirement-to-Code Matrix

| Requirement / capability | Main implementation | Supporting artifacts / outputs | Verification evidence |
|---|---|---|---|
| H&M-backed product foundation | `src/mars/data/hm_pipeline.py`, `configs/config.yaml` | `data/external/hm/processed/hm_products_master_clean_50k.csv`, `data/processed/products.parquet` | Bootstrap flow in `scripts/runtime/bootstrap_runtime.py` |
| User/persona generation | `src/mars/data/raw_simulator/service.py`, `src/mars/data/raw_simulator/personas.py` | `data/raw/users.csv`, `data/processed/users.parquet` | Processed user schema consumed by `src/mars/recommendation/artifacts.py` |
| Event/session generation | `src/mars/data/raw_simulator/service.py`, `src/mars/data/raw_simulator/events.py`, `src/mars/data/hm_pipeline.py` | `data/raw/events.csv`, `data/processed/events.parquet`, `sessions.parquet` | `tests/test_simulator.py` |
| Train/valid/test event split | `src/mars/data/hm_pipeline.py` | `train_events.parquet`, `valid_events.parquet`, `test_events.parquet` | Manifest row counts in `data/processed/manifest.json` |
| Text/image/hybrid search indexing | `src/mars/search/artifacts.py`, `src/mars/search/encoders.py`, `src/mars/retrieval/vector_index.py` | `artifacts/search/index_manifest.json`, embedding `.npy` files | `tests/test_search_agent3.py` |
| Search serving contract | `src/mars/search/service.py`, `apps/api/schemas.py`, `apps/api/main.py` | `/api/search` response payload | `tests/test_api_contract.py`, `tests/test_search_agent3.py` |
| Recommendation artifact build | `src/mars/recommendation/artifacts.py` | `artifacts/recsys/recommendation_artifacts.json.gz` | Bootstrap/build scripts |
| Candidate generation | `src/mars/recommendation/service.py`, `src/mars/recommendation/models.py` | In-memory candidate lists during runtime | `tests/test_recommendation.py` |
| Ranking and reranking | `src/mars/recommendation/service.py`, `src/mars/recommendation/rerank.py` | Ordered recommendation list with exploration flags | `tests/test_recommendation.py` |
| Session-aware recommendation context | `src/mars/recommendation/session.py`, `apps/api/service_adapters.py` | Session context in `/api/recommend` | `tests/test_recommendation.py`, `tests/test_api_contract.py` |
| Event ingestion and durable logging | `apps/api/service_adapters.py` | `logs/api_events.jsonl` | `tests/test_api_contract.py` |
| Redis-assisted live context | `apps/api/service_adapters.py`, `docker-compose.yml` | Redis lists/hashes for recent products/categories | Health and runtime fallback behavior in `tests/test_api_contract.py` |
| Metrics aggregation | `src/mars/evaluation/runner.py` | `artifacts/reports/metrics.json` | `tests/evaluation/test_metrics.py` |
| A/B assignment and reporting | `src/mars/evaluation/ab.py`, `apps/api/service_adapters.py`, `apps/api/main.py` | `/api/ab/assign`, `/api/ab/report` | `tests/evaluation/test_ab_ct_registry.py`, `tests/test_api_contract.py` |
| Continuous training threshold checks | `src/mars/ct/monitor.py`, `apps/worker/main.py`, `scripts/runtime/worker_loop.py` | `artifacts/registry/ct_state.json` | `tests/evaluation/test_ab_ct_registry.py` |
| Model version registry | `src/mars/ct/registry.py`, `scripts/runtime/bootstrap_runtime.py` | `artifacts/registry/models.json` | `tests/evaluation/test_ab_ct_registry.py` |
| Dashboard fallback/live operator UX | `apps/dashboard/api_client.py`, `apps/dashboard/app.py`, `apps/dashboard/demo_data.py` | Streamlit pages for control room, search, recommendation, experiments, model ops, live log, QA, guide | `tests/test_dashboard_client.py` |
| Containerized demo flow | `Dockerfile`, `docker-compose.yml`, `apps/simulator/main.py`, `apps/worker/main.py` | Bootstrap + API + dashboard + simulator + worker + Redis services | Manual runtime flow documented in `README.md` |

## Module Ownership Map

| Area | Current primary package | Why it matters |
|---|---|---|
| Final serving/runtime | `apps/`, `src/mars/` | `apps/` exposes Docker-facing entrypoints; `src/mars/` keeps feature code split by data/search/recommendation/evaluation/CT. |
| Raw simulator fallback | `src/mars/data/raw_simulator/` | Preserves raw generator functions used by the H&M runtime prep path without keeping a second top-level simulator tree. |
| Evaluation compatibility path | `src/evaluation/` | Preserves the spec-requested `src/evaluation/metrics.py` import path while delegating to `src/mars/evaluation/metrics.py`. |
| Submission documentation | `docs/`, `README.md` | Reviewer-facing project narrative and operational guidance. |

## Test Coverage Map

| Test file | Covered behavior |
|---|---|
| `tests/test_api_contract.py` | API health, search, recommendation, events, metrics, A/B endpoint contracts |
| `tests/test_search_agent3.py` | Search artifact build, deterministic encoding, text/image/hybrid retrieval |
| `tests/test_recommendation.py` | Candidate generation, fallback logic, reranking, session updates |
| `tests/test_dashboard_client.py` | Dashboard API client live/fallback behavior |
| `tests/evaluation/test_metrics.py` | Ranking metrics and CTR/CVR helpers |
| `tests/evaluation/test_ab_ct_registry.py` | A/B significance logic, CT decisions, model registry |

## Traceability Notes

- The final runtime intentionally keeps graceful fallback behavior when Redis or fully built artifacts are unavailable. That fallback path is part of the delivered system, not an accident.
- The strongest integration checkpoints are `scripts/runtime/bootstrap_runtime.py` and `docker-compose.yml`, because they connect data preparation, artifact build, evaluation, registry, serving, and monitoring into one reproducible flow.


