# 요구사항 구현 대응표

명세서 요구사항을 실제 코드와 검증 항목에 연결한 표이다.

## 1. 기능별 구현 위치

| 요구사항 | 구현 위치 | 산출물/확인 경로 |
| --- | --- | --- |
| 50K H&M 상품 catalog | `src/mars/data/hm_pipeline.py`, `scripts/artifacts/build_clean_hm_catalog_50k.py` | `data/processed/products.parquet` |
| 사용자/persona 생성 | `src/mars/data/raw_simulator/service.py`, `src/mars/data/raw_simulator/personas.py` | `data/processed/users.parquet` |
| search/view/cart/purchase 이벤트 | `src/mars/data/raw_simulator/events.py`, `src/mars/data/hm_pipeline.py` | `data/processed/events.parquet` |
| train/valid/test split | `src/mars/data/hm_pipeline.py` | `train_events.parquet`, `valid_events.parquet`, `test_events.parquet` |
| text/image/hybrid 검색 | `src/mars/search/service.py`, `src/mars/search/encoders.py` | `/api/search` |
| FAISS ANN index | `src/mars/search/artifacts.py`, `src/mars/retrieval/vector_index.py` | `artifacts/search/*.faiss` |
| 검색 qrels 평가 | `src/mars/search/qrels.py`, `src/mars/evaluation/runner.py` | `artifacts/reports/metrics.json` |
| 추천 후보 생성 | `src/mars/recommendation/service.py`, `src/mars/recommendation/models.py` | Recall@300 |
| ranking / re-ranking | `src/mars/recommendation/service.py`, `src/mars/recommendation/rerank.py` | AUC, HitRate@50, NDCG@50 |
| session context | `src/mars/recommendation/session.py`, `apps/api/service_adapters.py` | `/api/recommend`의 `session_context` |
| Redis feature store | `src/mars/recommendation/session.py`, `docker-compose.yml` | Redis recent events / counters |
| event ingestion | `apps/api/main.py`, `apps/api/service_adapters.py` | `/api/events`, `logs/api_events.jsonl` |
| A/B test | `src/mars/evaluation/ab.py`, `apps/api/main.py` | `/api/ab/assign`, `/api/ab/report` |
| Continuous Training | `src/mars/ct/monitor.py`, `scripts/runtime/worker_loop.py` | `artifacts/registry/ct_state.json` |
| model registry | `src/mars/ct/registry.py` | `artifacts/registry/models.json` |
| dashboard | `apps/dashboard/app.py`, `apps/dashboard/api_client.py` | `http://localhost:8501` |
| Docker 실행 | `Dockerfile`, `docker-compose.yml` | `docker compose up --build` |

## 2. 테스트 대응

| 테스트 파일 | 확인 항목 |
| --- | --- |
| `tests/test_api_contract.py` | health, search, recommendation, events, metrics, A/B API 계약 |
| `tests/test_api_search_schema.py` | search request alias와 image path schema |
| `tests/test_search_agent3.py` | 검색 artifact, vector index, qrels split |
| `tests/test_search_continuous_learning.py` | 검색 feedback와 behavior model 갱신 |
| `tests/test_recommendation.py` | 후보 생성, ranking, re-ranking, session update |
| `tests/evaluation/test_metrics.py` | MRR, NDCG, Recall, AUC 등 metric |
| `tests/evaluation/test_ab_ct_registry.py` | A/B 통계, CT trigger, registry |
| `tests/test_dashboard_client.py` | dashboard API client live/error 응답 처리 |
| `tests/test_simulator.py` | simulator output과 manifest 검증 |

## 3. 제출 기준 확인 명령

```powershell
python -m ruff check apps src scripts tests
python -m ruff format --check apps src scripts tests
python -m pytest -q
```

현재 검증 결과는 `58 passed`이다.
