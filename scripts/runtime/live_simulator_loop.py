from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mars.config import load_config

SURFACE_SEARCH = "search"
SURFACE_RECOMMENDATION = "recommendation"


def _clean_value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _wait_for_api(client: httpx.Client, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            response = client.get("/healthz", timeout=3.0)
            response.raise_for_status()
            return
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            last_error = str(exc)
            time.sleep(1.0)
    raise RuntimeError(f"API did not become healthy within {timeout_s:.0f}s: {last_error}")


def _assign_bucket(experiment_key: str, user_id: str) -> str:
    digest = hashlib.sha256(f"{experiment_key}:{user_id}".encode()).hexdigest()
    return "treatment" if int(digest[:8], 16) % 2 else "control"


def _strategy_for_bucket(bucket: str) -> str:
    return "control" if bucket == "control" else "treatment"


def _load_events(events_path: Path, limit: int) -> list[dict[str, Any]]:
    if not events_path.exists():
        raise FileNotFoundError(f"events parquet not found: {events_path}")
    columns = pd.read_parquet(events_path).columns
    wanted = [
        column
        for column in [
            "user_id",
            "session_id",
            "event_type",
            "product_id",
            "query",
            "timestamp",
            "persona",
            "category",
        ]
        if column in columns
    ]
    frame = pd.read_parquet(events_path, columns=wanted).head(limit)
    events: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        event_type = _clean_value(row.get("event_type")) or "view"
        product_id = _clean_value(row.get("product_id"))
        query = _clean_value(row.get("query"))
        if event_type == "search" and not query:
            query = _clean_value(row.get("category")) or "fashion item"
        if event_type != "search" and not product_id:
            continue
        user_id = _clean_value(row.get("user_id")) or "U-live"
        events.append(
            {
                "user_id": user_id,
                "session_id": _clean_value(row.get("session_id")) or "S-live",
                "event_type": event_type,
                "product_id": product_id,
                "query": query,
                "timestamp": _clean_value(row.get("timestamp")),
                "source": "live-simulator-replay",
                "metadata": {
                    "experiment_key": "mars_default",
                    "ab_bucket": _assign_bucket("mars_default", user_id),
                    "persona": _clean_value(row.get("persona")),
                    "category": _clean_value(row.get("category")),
                    "event_role": "replay",
                },
            }
        )
    if not events:
        raise RuntimeError(f"no replayable events found in {events_path}")
    return events


def _load_users_products(config: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    users_path = config.paths.processed_dir / "users.parquet"
    products_path = config.paths.processed_dir / "products.parquet"
    if not users_path.exists() or not products_path.exists():
        raise FileNotFoundError(
            "processed users/products parquet files are required for generated live simulation"
        )
    users = pd.read_parquet(users_path).to_dict(orient="records")
    products = pd.read_parquet(products_path).to_dict(orient="records")
    if not users or not products:
        raise RuntimeError("users/products parquet files are empty")
    return users, products


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    if hasattr(value, "tolist"):
        return [str(item) for item in value.tolist() if str(item).strip()]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def _products_by_category(products: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for product in products:
        category = _category(product)
        grouped.setdefault(category, []).append(product)
    return grouped


def _category(product: dict[str, Any]) -> str:
    return (
        _clean_value(product.get("category_l1"))
        or _clean_value(product.get("category"))
        or _clean_value(product.get("top_category"))
        or "Fashion"
    )


def _leaf(product: dict[str, Any]) -> str:
    return (
        _clean_value(product.get("category_l3"))
        or _clean_value(product.get("leaf_category"))
        or _clean_value(product.get("name"))
        or "item"
    )


def _build_query(product: dict[str, Any], rng: random.Random) -> str:
    color = _clean_value(product.get("color"))
    category = _category(product)
    leaf = _leaf(product)
    name = _clean_value(product.get("name"))
    variants = [
        " ".join(part for part in [color, leaf] if part),
        " ".join(part for part in [color, category, leaf] if part),
        " ".join(part for part in [category, leaf] if part),
        name or leaf,
    ]
    query = rng.choice([value for value in variants if value])
    return query.strip() or "fashion item"


def _transition_config(config: Any, persona: str) -> dict[str, float]:
    transition = (
        config.raw.get("simulator", {}).get("transition_prob", {})
        if isinstance(config.raw, dict)
        else {}
    )
    value = transition.get(persona, {}) if isinstance(transition, dict) else {}
    return {
        "view_to_cart": float(value.get("view_to_cart", 0.04)),
        "cart_to_purchase": float(value.get("cart_to_purchase", 0.25)),
        "direct_purchase_from_view": float(value.get("direct_purchase_from_view", 0.002)),
    }


def _preferred_categories(user: dict[str, Any]) -> list[str]:
    categories = _as_list(user.get("preferred_categories"))
    preferred_top = _clean_value(user.get("preferred_top_category"))
    if preferred_top and preferred_top not in categories:
        categories.append(preferred_top)
    return categories


def _choose_product(
    *,
    user: dict[str, Any],
    products: list[dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]],
    rng: random.Random,
) -> dict[str, Any]:
    preferred = [category for category in _preferred_categories(user) if category in grouped]
    if preferred and rng.random() < 0.82:
        bucket = grouped[rng.choice(preferred)]
    else:
        bucket = grouped.get(rng.choice(list(grouped))) if grouped else products
    return dict(rng.choice(bucket or products))


def _rank_choice(items: list[dict[str, Any]], rng: random.Random) -> dict[str, Any] | None:
    if not items:
        return None
    weights = [1.0 / (idx + 1) for idx in range(len(items))]
    return dict(rng.choices(items, weights=weights, k=1)[0])


def _extract_results(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    raw = payload.get(key, [])
    return [dict(item) for item in raw if isinstance(item, dict)]


def _call_search(
    client: httpx.Client,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    response = client.post(
        "/api/search",
        json={"search_type": "text", "query": query, "top_k": top_k},
    )
    response.raise_for_status()
    return _extract_results(response.json(), "results")


def _call_recommend(
    client: httpx.Client,
    user_id: str,
    session_id: str,
    top_n: int,
    experiment_key: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    response = client.get(
        "/api/recommend",
        params={
            "user_id": user_id,
            "session_id": session_id,
            "top_n": top_n,
            "experiment_key": experiment_key,
        },
    )
    response.raise_for_status()
    payload = response.json()
    return _extract_results(payload, "recommendations"), dict(payload.get("session_context", {}))


def _fallback_results(product: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "product_id": str(product.get("product_id")),
            "name": str(product.get("name", product.get("product_id"))),
            "category": _category(product),
            "price": float(product.get("price", 0.0) or 0.0),
            "score": 1.0,
        }
    ]


def _event_payload(
    *,
    user_id: str,
    session_id: str,
    event_type: str,
    product: dict[str, Any] | None,
    query: str | None,
    source: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    product_id = _clean_value(product.get("product_id")) if product else None
    category = _category(product) if product else metadata.get("category")
    return {
        "user_id": user_id,
        "session_id": session_id,
        "event_type": event_type,
        "product_id": product_id,
        "query": query,
        "timestamp": datetime.now(UTC).isoformat(),
        "source": source,
        "category": category,
        "metadata": {**metadata, "category": category},
    }


def _exposure_events(
    *,
    user_id: str,
    session_id: str,
    query: str,
    surface: str,
    results: list[dict[str, Any]],
    experiment_key: str | None,
    bucket: str | None,
    strategy: str | None,
    max_items: int,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    request_id = f"{surface}:{session_id}:{uuid4().hex[:8]}"
    for rank, result in enumerate(results[:max_items], start=1):
        product = {
            "product_id": result.get("product_id"),
            "name": result.get("name"),
            "category_l1": result.get("category"),
            "price": result.get("price", 0.0),
        }
        metadata = {
            "event_role": "exposure",
            "surface": surface,
            "source_surface": surface,
            "request_id": request_id,
            "exposure_id": f"{request_id}:{rank}",
            "rank": rank,
            "strategy": strategy,
        }
        if experiment_key and bucket:
            metadata.update({"experiment_key": experiment_key, "ab_bucket": bucket})
        events.append(
            _event_payload(
                user_id=user_id,
                session_id=session_id,
                event_type="search",
                product=product,
                query=query,
                source="live-simulator-generate",
                metadata=metadata,
            )
        )
    return events


def _followup_events(
    *,
    user_id: str,
    session_id: str,
    product: dict[str, Any],
    query: str,
    source_surface: str,
    exposure: dict[str, Any],
    transition: dict[str, float],
    rng: random.Random,
) -> list[dict[str, Any]]:
    metadata = dict(exposure.get("metadata", {}))
    metadata.update(
        {
            "event_role": "response",
            "source_surface": source_surface,
            "surface": source_surface,
        }
    )
    events = [
        _event_payload(
            user_id=user_id,
            session_id=session_id,
            event_type="view",
            product=product,
            query=query,
            source="live-simulator-generate",
            metadata=metadata,
        )
    ]
    if rng.random() < transition["view_to_cart"]:
        events.append(
            _event_payload(
                user_id=user_id,
                session_id=session_id,
                event_type="cart",
                product=product,
                query=query,
                source="live-simulator-generate",
                metadata=metadata,
            )
        )
        if rng.random() < transition["cart_to_purchase"]:
            events.append(
                _event_payload(
                    user_id=user_id,
                    session_id=session_id,
                    event_type="purchase",
                    product=product,
                    query=query,
                    source="live-simulator-generate",
                    metadata=metadata,
                )
            )
    elif rng.random() < transition["direct_purchase_from_view"]:
        events.append(
            _event_payload(
                user_id=user_id,
                session_id=session_id,
                event_type="purchase",
                product=product,
                query=query,
                source="live-simulator-generate",
                metadata=metadata,
            )
        )
    return events


def _generate_session_events(
    *,
    client: httpx.Client,
    config: Any,
    users: list[dict[str, Any]],
    products: list[dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    top_k: int,
    top_n: int,
    experiment_key: str,
) -> list[dict[str, Any]]:
    user = dict(rng.choice(users))
    user_id = str(user.get("user_id", "U-live"))
    persona = str(user.get("persona", "unknown"))
    session_id = f"S-live-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:6]}"
    seed_product = _choose_product(user=user, products=products, grouped=grouped, rng=rng)
    query = _build_query(seed_product, rng)
    bucket = _assign_bucket(experiment_key, user_id)
    strategy = _strategy_for_bucket(bucket)

    try:
        search_results = _call_search(client, query, top_k)
    except Exception:
        search_results = _fallback_results(seed_product)
    try:
        recommendation_results, session_context = _call_recommend(
            client, user_id, session_id, top_n, experiment_key
        )
        bucket = str(session_context.get("ab_bucket") or bucket)
        strategy = str(session_context.get("recommendation_strategy") or strategy)
    except Exception:
        recommendation_results = _fallback_results(seed_product)

    events = [
        _event_payload(
            user_id=user_id,
            session_id=session_id,
            event_type="search",
            product=None,
            query=query,
            source="live-simulator-generate",
            metadata={
                "event_role": "user_action",
                "surface": SURFACE_SEARCH,
                "persona": persona,
                "query_seed_product_id": seed_product.get("product_id"),
                "category": _category(seed_product),
            },
        )
    ]
    search_exposures = _exposure_events(
        user_id=user_id,
        session_id=session_id,
        query=query,
        surface=SURFACE_SEARCH,
        results=search_results,
        experiment_key=None,
        bucket=None,
        strategy=None,
        max_items=top_k,
    )
    recommendation_exposures = _exposure_events(
        user_id=user_id,
        session_id=session_id,
        query=query,
        surface=SURFACE_RECOMMENDATION,
        results=recommendation_results,
        experiment_key=experiment_key,
        bucket=bucket,
        strategy=strategy,
        max_items=top_n,
    )
    events.extend(search_exposures)
    events.extend(recommendation_exposures)

    choose_recommendation = bool(recommendation_exposures) and rng.random() < 0.55
    selected_pool = recommendation_exposures if choose_recommendation else search_exposures
    selected_surface = SURFACE_RECOMMENDATION if choose_recommendation else SURFACE_SEARCH
    selected_exposure = _rank_choice(selected_pool, rng)
    if selected_exposure:
        selected_product = {
            "product_id": selected_exposure.get("product_id"),
            "category_l1": selected_exposure.get("category"),
            "name": selected_exposure.get("metadata", {}).get("name"),
        }
        transition = _transition_config(config, persona)
        events.extend(
            _followup_events(
                user_id=user_id,
                session_id=session_id,
                product=selected_product,
                query=query,
                source_surface=selected_surface,
                exposure=selected_exposure,
                transition=transition,
                rng=rng,
            )
        )
    return events


def _post_event_with_retries(
    client: httpx.Client,
    payload: dict[str, Any],
    *,
    max_retries: int,
) -> tuple[bool, str | None]:
    attempts = max(1, max_retries + 1)
    last_error: str | None = None
    for attempt in range(attempts):
        try:
            response = client.post("/api/events", json=payload)
            response.raise_for_status()
            accepted = bool(response.json().get("accepted", False))
            return accepted, None
        except Exception as exc:  # pragma: no cover - runtime resilience
            last_error = f"{exc.__class__.__name__}: {exc}"
            if attempt < attempts - 1:
                time.sleep(min(2.0, 0.25 * (attempt + 1)))
    return False, last_error


def _jittered_batch_size(base_size: int, rng: random.Random, jitter: float) -> int:
    base_size = max(int(base_size), 0)
    if base_size == 0:
        return 0
    if jitter <= 0:
        return base_size
    if base_size == 1:
        # Keep roughly one session per tick on average, but avoid a flat metronome.
        return rng.choices((0, 1, 2, 3), weights=(0.18, 0.58, 0.20, 0.04), k=1)[0]
    spread = max(1, round(base_size * min(jitter, 1.0) * 0.5))
    return rng.randint(max(0, base_size - spread), base_size + spread)


def main() -> int:
    parser = argparse.ArgumentParser(description="Send live simulator events into the MARS API.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", default="dev")
    parser.add_argument("--api-base-url", default="http://api:8000")
    parser.add_argument("--source", choices=["generate", "replay"], default="generate")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--traffic-jitter", type=float, default=1.0)
    parser.add_argument("--event-limit", type=int, default=50000)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--experiment-key", default="mars_default")
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument("--request-timeout", type=float, default=20.0)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config, mode=args.mode)
    rng = random.Random(config.seed + 20260516)
    replay_cycle: Any = None
    users: list[dict[str, Any]] = []
    products: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    if args.source == "replay":
        replay_cycle = itertools.cycle(
            _load_events(config.paths.processed_dir / "events.parquet", args.event_limit)
        )
    else:
        users, products = _load_users_products(config)
        grouped = _products_by_category(products)

    total_sent = 0
    with httpx.Client(base_url=args.api_base_url, timeout=args.request_timeout) as client:
        _wait_for_api(client, args.startup_timeout)
        batch_index = 0
        while True:
            accepted = 0
            failed = 0
            generated = 0
            last_error = None
            payloads: list[dict[str, Any]] = []
            session_count = max(args.batch_size, 1)
            if args.source == "replay":
                for _ in range(session_count):
                    payloads.append(next(replay_cycle))
            else:
                session_count = _jittered_batch_size(args.batch_size, rng, args.traffic_jitter)
                for _ in range(session_count):
                    payloads.extend(
                        _generate_session_events(
                            client=client,
                            config=config,
                            users=users,
                            products=products,
                            grouped=grouped,
                            rng=rng,
                            top_k=max(args.top_k, 1),
                            top_n=max(args.top_n, 1),
                            experiment_key=args.experiment_key,
                        )
                    )
            for payload in payloads:
                event_accepted, error = _post_event_with_retries(
                    client,
                    payload,
                    max_retries=args.max_retries,
                )
                accepted += int(event_accepted)
                generated += 1
                total_sent += int(event_accepted)
                if error:
                    failed += 1
                    last_error = error
                if args.max_events and total_sent >= args.max_events:
                    break
            batch_index += 1
            print(
                json.dumps(
                    {
                        "simulator": "live",
                        "source": args.source,
                        "batch": batch_index,
                        "generated": generated,
                        "accepted": accepted,
                        "failed": failed,
                        "total_sent": total_sent,
                        "session_batch_size": session_count,
                        "configured_batch_size": args.batch_size,
                        "traffic_jitter": args.traffic_jitter,
                        "sleep_s": args.interval,
                        "last_error": last_error,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if args.max_events and total_sent >= args.max_events:
                return 0
            time.sleep(max(args.interval, 0.01))


if __name__ == "__main__":
    raise SystemExit(main())
