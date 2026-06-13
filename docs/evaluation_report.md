# MARS 평가 결과 리포트

작성일: 2026-06-13  
실행 mode: `full`  
기준 파일: `data/processed/manifest.json`, `artifacts/reports/metrics.json`  
비고: 검색 지표는 seed 42 query_id hash split의 held-out test 전체에서 산출한다.

---

## 1. 평가 프로토콜

| 항목 | 기준 |
| --- | --- |
| 데이터 규모 | 상품 50,000건, 사용자 10,000명, 행동 로그 1,000,000건 |
| 데이터 분할 | train / valid / test = 8 / 1 / 1 |
| 분할 방식 | 시간 기반 분할 |
| seed | 42 |
| 검색 평가 k | 10 |
| 추천 평가 k | 50 |
| 추천 primary task | `session_click_prediction` |
| 추천 평가 split | `test_events.parquet` |

추천 평가는 test event log를 시간순으로 replay하여 target event 직전의 session context를 복원한 뒤, 동일 context에 대해 추천 결과가 target 상품을 포함하는지 측정한다.

---

## 2. 데이터 구성

| 항목 | 값 |
| --- | ---: |
| products | 50,000 |
| users | 10,000 |
| events | 1,000,000 |
| sessions | 240,836 |
| search query labels | 193,064 |
| recommendation interactions | 1,092,497 |
| train events | 799,998 |
| valid events | 99,997 |
| test events | 100,005 |

Event 분포:

| event_type | count |
| --- | ---: |
| search | 240,836 |
| view | 735,726 |
| cart | 19,718 |
| purchase | 3,720 |

Persona 분포:

| persona | users |
| --- | ---: |
| careful_explorer | 1,558 |
| impulse_buyer | 1,660 |
| pragmatist | 1,927 |
| top_category_loyalist | 1,395 |
| trendsetter | 1,185 |
| value_seeker | 2,275 |

---

## 3. 검색 평가

검색 API와 CLIP/FAISS artifact는 구현되어 있다. 검색 qrels는 seed 42의
`train / valid / test = 8 / 1 / 1` query_id hash split으로 나누고, train qrels로만 supervised
historical prior를 구성한 뒤 held-out test qrels에서 평가한다.

검색 지표 산출 기준:

- `MRR@10`: Top-10 안에서 첫 positive product의 reciprocal rank 평균
- `NDCG@10`: Top-10의 discounted gain
- `Recall@10`: positive product가 Top-10 안에 포함되는 비율
- Latency: text, image, hybrid 요청 p95 중 제출 기준에 사용하는 검색 p95

| 지표 | 명세서 목표 | 최종 측정값 | 판정 |
| --- | ---: | ---: | --- |
| MRR@10 | >= 0.55 | `0.6152` | PASS |
| NDCG@10 | >= 0.50 | `0.6762` | PASS |
| Search p95 latency | <= 200ms | `76.05ms` | PASS |

검색 구현 근거:

| 항목 | 구현 |
| --- | --- |
| Text encoder | CLIP text encoder |
| Image encoder | CLIP image encoder |
| Index | FAISS HNSW |
| API | `POST /api/search` |
| Artifact | `artifacts/search/` |
| Train-only behavior model | `artifacts/search/query_behavior_model.json.gz` |

실제 검색 서비스는 위 behavior model만 로드하며, 평가 label 파일인
`data/processed/search_queries.parquet`를 서빙 중 읽지 않는 것을 별도 guard test로 확인했다.

검색 baseline과 최종 결과:

| 조건 | MRR@10 | NDCG@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| BM25 text-only | `0.03366` | `0.04642` | `0.08883` |
| 최종 supervised search | `0.6152` | `0.6762` | `0.8664` |

검색 passing score는 train/test를 격리한 qrels train-only supervised system score다. 동일 검색어가 train/test에 반복될 수 있으므로 query-disjoint 또는 pure CLIP-only 점수로 해석하지 않는다.

---

## 4. 추천 평가

추천 지표는 `artifacts/reports/metrics.json`의 `recommendation` section을 기준으로 정리했다.

추천 지표 산출 기준:

- `Recall@300`: 후보 300개 안에 target item이 포함되는 비율
- `HitRate@50`: Top-50 안에 target item이 포함되는 비율
- `NDCG@50`: target item의 Top-50 순위 할인 점수
- `Coverage@50`: 추천된 고유 상품 수 / 전체 상품 수
- `AUC`: ranker score와 target label의 ROC-AUC

| 지표 | 명세서 목표 | 현재값 | 판정 |
| --- | ---: | ---: | --- |
| Recall@300 | >= 0.30 | 0.6350 | PASS |
| Candidate p95 latency | <= 100ms | 27.71ms | PASS |
| Ranking AUC | >= 0.70 | 0.8593 | PASS |
| HitRate@50 | >= 0.20 | 0.4625 | PASS |
| NDCG@50 | >= 0.08 | 0.2936 | PASS |
| Coverage@50 | >= 0.20 | 0.2032 | PASS |
| Total p95 latency | <= 200ms | 35.29ms | PASS |

Task별 참고 지표:

| Task | Recall@300 | HitRate@50 | NDCG@50 | Coverage@50 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `session_click_prediction` | 0.6350 | 0.4625 | 0.2936 | 0.2032 | 0.8593 |
| `session_continuation` | 1.0000 | 1.0000 | 0.8382 | 0.2443 | 0.9984 |
| `strict_discovery` | 0.0475 | 0.0075 | 0.0015 | 0.1652 | 0.5794 |

해석:

- `session_click_prediction`은 search 직후 첫 view를 맞추는 primary task이다.
- `session_continuation`은 이미 같은 상품 흐름이 이어지는 cart/purchase 성격이 강하므로 보조 지표로만 사용한다.
- `strict_discovery`는 이미 본 상품을 제외하고 새로운 상품 발견을 맞추는 보수적 진단 지표이다.

---

## 5. Baseline 비교

추천 baseline:

| Baseline | Recall@300 | HitRate@50 | NDCG@50 | Coverage@50 |
| --- | ---: | ---: | ---: | ---: |
| popularity-only | 0.0271 | 0.0051 | 0.0013 | 0.0010 |
| final recommendation | 0.6350 | 0.4625 | 0.2936 | 0.2032 |

Two-Tower-only는 candidate generation 품질 확인용 baseline으로 별도 기록되어 있다.

검색 최종 MRR은 BM25 대비 `+1727.79%`, NDCG는 `+1356.72%`, Recall은
`+875.39%` 개선됐다. 단, exact-query historical prior 의존도가 높아 query-disjoint
성능으로 해석하지 않는다.

---

## 6. A/B 테스트

| 항목 | 구현 |
| --- | --- |
| experiment key | `mars_default` 또는 사용자 지정 key |
| bucket assignment | `experiment_key + user_id` deterministic hash |
| control | `BaselineVanilla` |
| treatment | `ComplementGraphExplore` |
| 검정 방법 | two-proportion z-test |
| 출력 | CTR, CVR, uplift, p-value, 95% confidence interval |

현재 누적 리포트 기준:

| Bucket | Impressions | Clicks | Conversions | CTR | CVR |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 494,218 | 375,472 | 1,836 | 0.7597 | 0.0037 |
| treatment | 505,782 | 383,692 | 1,884 | 0.7586 | 0.0037 |

| 항목 | 값 |
| --- | ---: |
| CTR uplift | -0.0011 |
| CVR uplift | 0.0000 |
| p-value | 0.9348 |
| 95% CI | [-0.0002, 0.0002] |
| significant | false |

해석:

- live A/B는 운영 모니터링 성격으로 제공된다.
- 현재 제출 번들에는 별도 paired replay 리포트를 포함하지 않고,
  `artifacts/reports/metrics.json`의 A/B 통계와 `/api/ab/report` 계약을 기준으로 확인한다.
- 추천 전략의 오프라인 품질 판단은 추천 평가 지표와 live A/B 모니터링을 분리해 해석한다.

---

## 7. Continuous Training 평가

| 항목 | 구현 상태 |
| --- | --- |
| 모니터링 지표 | HitRate, CTR, 신규 live log count |
| 신규 로그 임계값 | 10,000 |
| trigger reason | `new_logs_threshold_reached`, metric threshold breach |
| retrain 처리 | recommendation artifact live-feedback refresh, optional full rebuild |
| version 관리 | `artifacts/registry/models.json` |
| serving update | API lazy hot reload |

현재 registry version은 live simulator와 worker 실행 여부에 따라 계속 증가할 수 있다. 최신 값은 `/healthz`와 `artifacts/registry/models.json`을 기준으로 확인한다.

제출 기본값에서는 CT worker가 live feedback을 반영해 추천 artifact의 popularity/trending prior를 빠르게 갱신하고 registry를 갱신한다. 필요 시 config를 통해 full rebuild 경로로 전환할 수 있다. 검색 behavior-model refresh는 코드와 수동 스크립트를 유지하되, 기본 Docker 데모의 CT 완료 시간을 안정화하기 위해 `search.online_learning.auto_refresh=false`로 분리한다.

---

## 8. 재현 명령

```powershell
cd <repository-root>
docker compose up --build
```

별도 smoke test:

```powershell
python -m scripts.checks.smoke_api --base-url http://localhost:8000 --timeout 240
```

Full artifact 재생성:

```powershell
python -m scripts.runtime.bootstrap_runtime --config configs\config.yaml --mode full --rebuild-raw --clean-processed --clean-artifacts --register
```




