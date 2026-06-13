# 평가 결과

기준 파일은 `artifacts/reports/metrics.json`이고, 실행 조건은 `full` mode와 seed `42`이다.

## 1. 데이터 규모

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

이벤트 분포:

| event_type | count |
| --- | ---: |
| search | 240,836 |
| view | 735,726 |
| cart | 19,718 |
| purchase | 3,720 |

## 2. 검색 평가

검색 평가는 Microsoft H&M synthetic qrels를 사용했다. `query_id` 기준 hash split으로 train, valid, test를 나누고 held-out test split에서 MRR@10, NDCG@10, Recall@10을 계산했다.

| 항목 | 값 |
| --- | --- |
| label source | `microsoft_hnm_search_qrels` |
| split method | `query_id_hash` |
| split seed | `42` |
| train qrels | 154,652 |
| valid qrels | 19,218 |
| test qrels | 19,194 |
| query_id overlap | 0 |

검색 서비스는 train split으로 만든 behavior model과 FAISS artifact를 로드한다. Test qrels는 평가 runner에서만 사용한다.

| 지표 | 목표 | 측정값 | 판정 |
| --- | ---: | ---: | --- |
| MRR@10 | >= 0.55 | 0.6152 | PASS |
| NDCG@10 | >= 0.50 | 0.6762 | PASS |
| Recall@10 | 참고 | 0.8664 | PASS |
| Category Hit@10 | 참고 | 0.9860 | PASS |
| p95 latency | <= 200 ms | 76.05 ms | PASS |

BM25 text-only baseline 대비 MRR@10은 0.0337에서 0.6152로 개선됐다.

## 3. 추천 평가

Primary task는 `session_click_prediction`이다. Search event 이후 처음 발생한 view 상품을 target으로 두고, target이 `recent_products`에 들어가기 전 context에서 추천 결과를 평가했다.

| 지표 | 목표 | 측정값 | 판정 |
| --- | ---: | ---: | --- |
| Recall@300 | >= 0.30 | 0.6350 | PASS |
| Ranking AUC | >= 0.70 | 0.8593 | PASS |
| HitRate@50 | >= 0.20 | 0.4625 | PASS |
| NDCG@50 | >= 0.08 | 0.2936 | PASS |
| Coverage@50 | >= 0.20 | 0.2032 | PASS |
| Total p95 latency | <= 200 ms | 35.29 ms | PASS |

보조 task 결과:

| Task | Recall@300 | HitRate@50 | NDCG@50 | Coverage@50 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `session_click_prediction` | 0.6350 | 0.4625 | 0.2936 | 0.2032 | 0.8593 |
| `session_continuation` | 1.0000 | 1.0000 | 0.8382 | 0.2443 | 0.9984 |
| `strict_discovery` | 0.0475 | 0.0075 | 0.0015 | 0.1652 | 0.5794 |

## 4. A/B 테스트

| 항목 | 값 |
| --- | --- |
| experiment key | `mars_default` |
| assignment | `experiment_key + user_id` deterministic hash |
| control | `RankOnlyControl` |
| treatment | `ComplementGraphExplore` |
| test method | two-proportion z-test |

현재 full 데이터 기준으로 treatment의 통계적 우위는 확인되지 않았다. 대시보드는 bucket별 impressions, clicks, conversions, CTR, CVR, uplift, p-value, 95% CI를 표시한다.

| 지표 | 값 |
| --- | ---: |
| CTR uplift | -0.0011 |
| CVR uplift | 0.00001 |
| p-value | 0.9348 |
| 95% CI | [-0.000229, 0.000249] |

## 5. Continuous Training

| 항목 | 구현 |
| --- | --- |
| live log | `logs/api_events.jsonl` |
| trigger 기준 | 신규 로그 수, CTR, HitRate |
| registry | `artifacts/registry/models.json` |
| 반영 방식 | API lazy hot reload |

API와 simulator가 남긴 live event를 worker가 주기적으로 확인한다. 조건을 만족하면 registry version을 갱신하고, API는 active version 변경을 감지해 다음 요청부터 새 artifact를 로드한다.

## 6. 검증 명령

```powershell
python -m ruff check apps src scripts tests
python -m ruff format --check apps src scripts tests
python -m pytest -q
```

최종 테스트 결과:

```text
58 passed
```
