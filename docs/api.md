# API Documentation

## Base Runtime

- Local API base URL: `http://localhost:8000`
- Framework: FastAPI
- Main entrypoint: `apps/api/main.py`

## Endpoint Summary

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Service, artifact, and configuration readiness |
| `POST` | `/api/search` | Text, image, or hybrid product search |
| `GET` | `/api/recommend` | Multi-stage recommendations for a user/session |
| `POST` | `/api/events` | Record view/cart/purchase/search events |
| `GET` | `/api/metrics` | Aggregated dashboard and training metrics |
| `POST` | `/api/ab/assign` | Deterministic A/B bucket assignment |
| `GET` | `/api/ab/report` | A/B report with uplift and significance |

## 1. Health Check

### Request

```http
GET /healthz
```

### Response shape

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

### Request

```http
POST /api/search
Content-Type: application/json
```

### Body

```json
{
  "query": "black minimal jacket",
  "search_type": "text",
  "top_k": 10,
  "filters": {
    "category": "outer",
    "min_price": 30000,
    "max_price": 200000
  },
  "hybrid_weights": {
    "text": 0.6,
    "image": 0.4
  }
}
```

### Notes

- `search_type` supports `text`, `image`, and `hybrid`.
- Text search requires `query`; the spec example alias `query_text` is accepted and normalized to `query`.
- Image search requires `image_base64` or `image_url`.
- Hybrid search accepts text only, image only, or both.

### Response shape

```json
{
  "search_type": "text",
  "results": [
    {
      "product_id": "P00000001",
      "name": "Black Leather Jacket",
      "score": 0.932145,
      "price": 129000,
      "category": "outer",
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

### Request

```http
GET /api/recommend?user_id=U000001&top_n=10&session_id=S-demo
```

### Response shape

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
    "session_id": "S-demo",
    "recent_products": ["P00000091"],
    "recent_clicks": ["P00000091"],
    "recent_categories": ["Ladieswear"],
    "session_interest": "Ladieswear",
    "source": "redis"
  }
}
```

## 4. Event Ingestion

### Request

```http
POST /api/events
Content-Type: application/json
```

```json
{
  "user_id": "U000001",
  "session_id": "S-demo",
  "event_type": "view",
  "product_id": "P00001023",
  "source": "dashboard",
  "category": "Ladieswear",
  "metadata": {
    "category": "Ladieswear"
  }
}
```

`category` can be sent either as a top-level field or as `metadata.category`. The API mirrors it into the event payload used by Redis/session recommendation.

### Response shape

```json
{
  "accepted": true,
  "event_id": "E9f0b3d5a8e421ab",
  "session_id": "S-demo",
  "event_type": "view",
  "redis_updated": true,
  "durable_log": "logs/api_events.jsonl"
}
```

## 5. Metrics

### Request

```http
GET /api/metrics
```

### Response contents

The response aggregates:

- search quality (`mrr`, `ndcg_at_10`, `recall_at_10`, latency)
- recommendation quality (`recall_at_300`, `auc`, `hitrate_at_50`, `ndcg_at_50`, coverage, latency)
- system counts and artifact readiness
- simulator persona/event mix
- continuous-training status and registered versions

## 6. A/B Assignment

### Request

```http
POST /api/ab/assign
Content-Type: application/json
```

```json
{
  "user_id": "U000001",
  "experiment_key": "homepage_rerank_v1",
  "buckets": ["control", "treatment"]
}
```

### Response shape

```json
{
  "user_id": "U000001",
  "experiment_key": "homepage_rerank_v1",
  "bucket": "control",
  "assignment_method": "deterministic_hash"
}
```

## 7. A/B Report

### Request

```http
GET /api/ab/report?experiment_key=homepage_rerank_v1
```

`experiment_key`를 생략하면 live simulator가 사용하는 `mars_default` 실험을 기본으로 조회한다.

### Response shape

```json
{
  "experiment_key": "homepage_rerank_v1",
  "buckets": {
    "control": {
      "impressions": 1000,
      "clicks": 80,
      "conversions": 41,
      "ctr": 0.08,
      "cvr": 0.041
    },
    "treatment": {
      "impressions": 1000,
      "clicks": 95,
      "conversions": 49,
      "ctr": 0.095,
      "cvr": 0.049
    }
  },
  "uplift": 0.008,
  "p_value": 0.312,
  "confidence_interval_95": [-0.007, 0.023],
  "method": "two_proportion_z_test"
}
```

CTR은 `clicks / impressions`, CVR은 `conversions / impressions`, `purchase_per_click`은 `conversions / clicks` 기준으로 계산한다.

## Validation Rules

- `top_k`: `1..100`
- `top_n`: `1..100`
- `min_price <= max_price`
- A/B `buckets` must contain at least two unique values
- `event_type` is constrained to the known event vocabulary

## Failure and Fallback Behavior

- If Redis is unavailable, the API still returns recommendation responses with fallback session context.
- If final artifacts are unavailable, search and recommendation fall back to deterministic lightweight logic so the endpoint contracts remain valid.
- The dashboard client treats API failures as a demo-data scenario rather than a hard stop.


