# MARS

Fashion commerce 환경을 가정한 멀티모달 검색, 개인화 추천, A/B 테스트, 실시간 행동 로그 대시보드 프로젝트입니다. Docker Compose 한 번으로 API, Streamlit 대시보드, Redis, 시뮬레이터, Continuous Training worker를 함께 실행하도록 구성했습니다.

## 현재 기준

최종 검증은 `full` mode와 seed `42` 기준으로 진행했습니다. 정량 결과는 `artifacts/reports/metrics.json`에 저장됩니다.

| 영역 | 지표 | 목표 | 결과 |
| --- | --- | ---: | ---: |
| 검색 | MRR@10 | >= 0.55 | 0.6152 |
| 검색 | NDCG@10 | >= 0.50 | 0.6762 |
| 검색 | Recall@10 | 참고 | 0.8664 |
| 검색 | p95 latency | <= 200 ms | 76.05 ms |
| 추천 | Recall@300 | >= 0.30 | 0.6350 |
| 추천 | HitRate@50 | >= 0.20 | 0.4625 |
| 추천 | NDCG@50 | >= 0.08 | 0.2936 |
| 추천 | Coverage@50 | >= 0.20 | 0.2032 |
| 추천 | Ranking AUC | >= 0.70 | 0.8593 |
| 추천 | Total p95 latency | <= 200 ms | 35.29 ms |

사용 데이터 규모:

| 항목 | 값 |
| --- | ---: |
| Products | 50,000 |
| Users | 10,000 |
| Events | 1,000,000 |
| Search qrels | 193,064 |
| Search test queries | 19,194 |
| Recommendation test instances | 400 |

## 실행

```powershell
docker compose up --build
```

| 서비스 | 주소 |
| --- | --- |
| FastAPI | `http://localhost:8000` |
| API Swagger | `http://localhost:8000/docs` |
| Streamlit Dashboard | `http://localhost:8501` |
| Redis | `localhost:6379` |

API smoke check:

```powershell
python -m scripts.checks.smoke_api --base-url http://localhost:8000 --timeout 240
```

## 구성

```text
.
|- apps/
|  |- api/                 # FastAPI endpoint
|  |- dashboard/           # Streamlit dashboard
|  |- simulator/           # live event simulator
|  `- worker/              # continuous training worker
|- src/mars/
|  |- config/              # config loader
|  |- data/                # H&M data pipeline and event generation
|  |- search/              # CLIP encoder, FAISS artifact, search service
|  |- retrieval/           # vector index helper
|  |- recommendation/      # candidate, ranking, reranking, session logic
|  |- evaluation/          # metric and A/B evaluation
|  `- ct/                  # live log monitor and model registry
|- src/evaluation/         # scoring interface compatibility wrapper
|- scripts/                # build, evaluation, check, packaging scripts
|- tests/                  # API/search/recommendation/CT tests
|- configs/config.yaml
|- docker-compose.yml
|- Dockerfile
`- docs/
```

## 주요 기능

| 기능 | 구현 |
| --- | --- |
| Text/Image/Hybrid 검색 | `apps/api/main.py`, `src/mars/search/` |
| CLIP 멀티모달 임베딩 | `src/mars/search/encoders.py` |
| FAISS ANN 검색 | `artifacts/search/*.faiss` |
| 검색 행동 모델 | train split으로 생성한 `query_behavior_model.json.gz` |
| 추천 후보 생성 | Two-Tower 기반 candidate stage |
| Ranking / Re-ranking | Wide&Deep 계열 ranking feature, transition/category rerank |
| 세션 개인화 | Redis session store와 GRU session encoder |
| A/B 테스트 | `/api/ab/assign`, `/api/ab/report` |
| Continuous Training | live log monitor, registry update, API hot reload |
| 대시보드 | 검색, 추천, 실험 분석, 모델 운영, 라이브 로그, 제출 검증 |

## 데이터와 artifact

GitHub에는 코드와 문서를 올리고, full-scale 실행에 필요한 `data/processed/`와 `artifacts/`는 runtime bundle로 전달하는 방식을 권장합니다. 다른 PC에서 바로 실행하려면 repository root에 bundle을 풀고 `docker compose up --build`를 실행하면 됩니다.

Runtime bundle 생성:

```powershell
python -m scripts.packaging.package_runtime_bundle --dry-run
python -m scripts.packaging.package_runtime_bundle --output dist\mars_runtime_bundle.zip
```

상품 이미지 preview까지 포함하는 bundle:

```powershell
python -m scripts.packaging.package_runtime_bundle --output dist\mars_runtime_bundle_with_images.zip --include-images
```

원본 데이터에서 artifact를 다시 만들 때 필요한 입력 경로:

| 입력 | 경로 |
| --- | --- |
| H&M 50K 상품 master | `data/external/hm/processed/hm_products_master_clean_50k.csv` |
| Microsoft H&M search queries | `data/external/hnm_search/raw/queries.csv` |
| Microsoft H&M search qrels | `data/external/hnm_search/raw/qrels.csv` |
| H&M product images | `data/external/hm/raw/images/` |

전체 artifact 재생성:

```powershell
python -m scripts.runtime.bootstrap_runtime --config configs/config.yaml --mode full
```

정량 평가:

```powershell
python -m scripts.evaluation.evaluate_required_scale --config configs/config.yaml --mode full
```

코드 검증:

```powershell
python -m ruff check apps src scripts tests
python -m ruff format --check apps src scripts tests
python -m pytest -q
```

## 문서

| 문서 | 내용 |
| --- | --- |
| `docs/README.md` | 문서 목록 |
| `docs/architecture.md` | 시스템 구조와 실행 흐름 |
| `docs/api.md` | API endpoint와 request/response |
| `docs/evaluation_report.md` | 평가 기준과 최종 지표 |
| `docs/code_traceability.md` | 요구사항과 구현 파일 대응 |
| `docs/runtime_bundle_guide.md` | runtime bundle 전달/재생성 방법 |
