# API 문서

FastAPI entrypoint는 `apps/api/main.py`이고 기본 주소는 `http://localhost:8000`이다.

## Endpoint 요약

| Method | Path | 용도 |
| --- | --- | --- |
| `GET` | `/healthz` | 서비스와 artifact 준비 상태 확인 |
| `POST` | `/api/search` | text/image/hybrid 상품 검색 |
| `GET` | `/api/recommend` | 사용자/세션 기반 추천 |
| `POST` | `/api/events` | search/view/cart/purchase 이벤트 적재 |
| `GET` | `/api/metrics` | 대시보드용 집계 metric 조회 |
| `POST` | `/api/ab/assign` | deterministic A/B bucket 배정 |
| `GET` | `/api/ab/report` | A/B uplift, p-value, confidence interval 조회 |

## 1. Health Check

```http
GET /healthz
```

대표 응답:

```json
{
  "status": "ok",
  "app": "mars-api",
  "version": "0.1.0",
  "config_profile": "full",
  "services": {
    "search": "ready",
    "recommendation": "ready",
    "redis": "ready"
  },
  "artifacts": {
    "processed_data": true,
    "manifest": true,
    "search_index": true,
    "recsys_models": true,
    "reports": true,
    "registry": true
  }
}
```

## 2. Search

```http
POST /api/search
Content-Type: application/json
```

요청 예시:

```json
{
  "query": "black socks",
  "search_type": "text",
  "top_k": 10,
  "filters": {
    "category": "Menswear",
    "min_price": 1000,
    "max_price": 50000
  },
  "hybrid_weights": {
    "text": 0.6,
    "image": 0.4
  }
}
```

요청 규칙:

| 항목 | 설명 |
| --- | --- |
| `search_type` | `text`, `image`, `hybrid` |
| `query` | text 검색에 사용한다. 명세서 alias인 `query_text`도 허용한다. |
| `image_base64`, `image_url` | image 검색 입력이다. 둘 중 하나를 사용한다. |
| `top_k` | `1..100` 범위 |
| `filters` | category, price 범위 필터 |

대표 응답:

```json
{
  "search_type": "text",
  "results": [
    {
      "product_id": "P00000001",
      "name": "Black Socks",
      "score": 0.932145,
      "price": 12900,
      "category": "Menswear",
      "image_url": "data/external/hm/raw/images/..."
    }
  ],
  "latency_ms": 18.4,
  "total_count": 10,
  "debug": {
    "index_backend": "faiss",
    "encoder_type": "clip:openai/clip-vit-base-patch32",
    "index_version": 1
  }
}
```

## 3. Recommendation

```http
GET /api/recommend?user_id=U000001&top_n=10&session_id=S000001
```

대표 응답:

```json
{
  "user_id": "U000001",
  "recommendations": [
    {
      "product_id": "P00001023",
      "score": 0.812233,
      "reason": "matches_preferred_category",
      "is_exploration": false
    }
  ],
  "pipeline_latency": {
    "candidate_ms": 12.5,
    "ranking_ms": 3.8,
    "reranking_ms": 1.1,
    "total_ms": 17.7
  },
  "session_context": {
    "session_id": "S000001",
    "recent_products": ["P00000011", "P00000021"],
    "recent_categories": ["Menswear"],
    "event_counts": {
      "view": 3,
      "cart": 1,
      "purchase": 0
    }
  }
}
```

추천 pipeline은 candidate generation, ranking, reranking 순서로 실행된다. Redis가 연결된 경우 최근 행동과 세션 context가 응답에 반영된다.

## 4. Event Logging

```http
POST /api/events
Content-Type: application/json
```

요청 예시:

```json
{
  "user_id": "U000001",
  "session_id": "S000001",
  "event_type": "cart",
  "product_id": "P00001023",
  "screen": "recommendation",
  "metadata": {
    "rank": 3
  }
}
```

지원 event type:

| event_type | 의미 |
| --- | --- |
| `search` | 검색 요청 |
| `view` | 상품 조회 |
| `cart` | 장바구니 |
| `purchase` | 구매 |

이벤트는 `logs/api_events.jsonl`에 적재되고, Redis session context와 Continuous Training worker 입력으로 사용된다.

## 5. Metrics

```http
GET /api/metrics
```

응답에는 다음 항목이 포함된다.

| 영역 | 주요 항목 |
| --- | --- |
| System | products, users, events, artifact readiness |
| Search | MRR@10, NDCG@10, Recall@10, latency |
| Recommendation | Recall@300, HitRate@50, NDCG@50, Coverage@50, AUC, latency |
| Live log | 누적 이벤트 수, 화면별 노출/클릭/전환, 분 단위 이벤트 흐름 |
| Continuous Training | trigger 상태, registry version |

## 6. A/B Test

Bucket 배정:

```http
POST /api/ab/assign
Content-Type: application/json
```

```json
{
  "user_id": "U000001",
  "experiment_key": "mars_default"
}
```

대표 응답:

```json
{
  "user_id": "U000001",
  "experiment_key": "mars_default",
  "bucket": "control",
  "strategy": "RankOnlyControl"
}
```

Report 조회:

```http
GET /api/ab/report?experiment_key=mars_default
```

대표 응답:

```json
{
  "experiment_key": "mars_default",
  "buckets": {
    "control": {
      "impressions": 494218,
      "clicks": 375472,
      "conversions": 1836,
      "ctr": 0.7597,
      "cvr": 0.0037
    },
    "treatment": {
      "impressions": 505782,
      "clicks": 383692,
      "conversions": 1884,
      "ctr": 0.7586,
      "cvr": 0.0037
    }
  },
  "uplift": {
    "ctr": -0.0011,
    "cvr": 0.00001
  },
  "p_value": 0.9348,
  "confidence_interval_95": [-0.000229, 0.000249],
  "method": "two_proportion_z_test"
}
```

CTR은 `clicks / impressions`, CVR은 `conversions / impressions`, purchase-per-click은 `conversions / clicks`로 계산한다.
