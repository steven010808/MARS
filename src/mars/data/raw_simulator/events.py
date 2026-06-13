from __future__ import annotations

import ast
import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_TRANSITION_PROB = {
    "trendsetter": {
        "view_to_cart": 0.24,
        "cart_to_purchase": 0.34,
        "direct_purchase_from_view": 0.06,
    },
    "pragmatist": {
        "view_to_cart": 0.16,
        "cart_to_purchase": 0.28,
        "direct_purchase_from_view": 0.03,
    },
    "value_seeker": {
        "view_to_cart": 0.18,
        "cart_to_purchase": 0.24,
        "direct_purchase_from_view": 0.02,
    },
    "top_category_loyalist": {
        "view_to_cart": 0.26,
        "cart_to_purchase": 0.38,
        "direct_purchase_from_view": 0.05,
    },
    "impulse_buyer": {
        "view_to_cart": 0.31,
        "cart_to_purchase": 0.42,
        "direct_purchase_from_view": 0.09,
    },
    "careful_explorer": {
        "view_to_cart": 0.12,
        "cart_to_purchase": 0.20,
        "direct_purchase_from_view": 0.02,
    },
}

DEFAULT_VIEW_BOUNDS = {
    "trendsetter": (3, 6),
    "pragmatist": (2, 4),
    "value_seeker": (2, 5),
    "top_category_loyalist": (2, 5),
    "impulse_buyer": (2, 4),
    "careful_explorer": (3, 6),
}

REQUIRED_EVENT_COLUMNS = [
    "event_id",
    "session_id",
    "user_id",
    "persona",
    "event_type",
    "timestamp",
    "query_text",
    "product_id",
    "top_category",
    "price_tier",
    "price",
    "position",
    "source_reason",
]

ALLOWED_EVENT_TYPES = {"search", "view", "cart", "purchase"}

DEFAULT_HM_TOP_CATEGORIES = [
    "Ladieswear",
    "Baby/Children",
    "Divided",
    "Menswear",
    "Sport",
]


def _get_simulator_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("simulator") or {}


def _get_events_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return _get_simulator_cfg(config).get("events") or {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def _parse_list_like(value: Any) -> list[str]:
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass

    return [part.strip() for part in text.replace("|", ",").split(",") if part.strip()]


def _extract_budget(user_row: pd.Series) -> int | None:
    for key in ("budget", "budget_limit", "budget_max", "budget_cap"):
        if key in user_row.index:
            parsed = _safe_int(user_row[key])
            if parsed is not None:
                return parsed
    return None


def _extract_preference_columns(users_df: pd.DataFrame) -> list[str]:
    return [col for col in users_df.columns if col.startswith("category_pref_")]


def _extract_fallback_categories(products_df: pd.DataFrame, pref_cols: list[str]) -> list[str]:
    if pref_cols:
        return [col.replace("category_pref_", "", 1) for col in pref_cols]

    if "top_category" in products_df.columns:
        categories = sorted(products_df["top_category"].dropna().astype(str).unique().tolist())
        if categories:
            return categories

    return DEFAULT_HM_TOP_CATEGORIES


def _sample_focus_category(
    user_row: pd.Series,
    pref_cols: list[str],
    fallback_categories: list[str],
    rng: random.Random,
) -> str:
    if not pref_cols:
        return rng.choice(fallback_categories)

    categories: list[str] = []
    weights: list[float] = []

    for col in pref_cols:
        category = col.replace("category_pref_", "", 1)
        weight = _safe_float(user_row.get(col, 0.0), 0.0)
        categories.append(category)
        weights.append(max(weight, 0.0))

    if sum(weights) <= 0:
        preferred = str(user_row.get("preferred_top_category", "") or "").strip()
        if preferred and preferred in fallback_categories:
            return preferred
        return rng.choice(fallback_categories)

    return rng.choices(categories, weights=weights, k=1)[0]


def _get_transition_prob(config: dict[str, Any], persona: str) -> dict[str, float]:
    events_cfg = _get_events_cfg(config)
    transition_cfg = events_cfg.get("transition_prob") or {}
    configured = transition_cfg.get(persona)

    if configured:
        return {
            "view_to_cart": float(configured.get("view_to_cart", 0.15)),
            "cart_to_purchase": float(configured.get("cart_to_purchase", 0.25)),
            "direct_purchase_from_view": float(configured.get("direct_purchase_from_view", 0.03)),
        }

    return DEFAULT_TRANSITION_PROB.get(persona, DEFAULT_TRANSITION_PROB["pragmatist"])


def _get_view_bounds(config: dict[str, Any], persona: str) -> tuple[int, int]:
    events_cfg = _get_events_cfg(config)
    view_cfg = events_cfg.get("views_per_session_by_persona") or {}
    configured = view_cfg.get(persona)

    if isinstance(configured, list) and len(configured) == 2:
        lo = int(configured[0])
        hi = int(configured[1])
        return min(lo, hi), max(lo, hi)

    return DEFAULT_VIEW_BOUNDS.get(persona, DEFAULT_VIEW_BOUNDS["pragmatist"])


def _sample_target_session_length(
    persona: str,
    config: dict[str, Any],
    remaining_slots: int,
    rng: random.Random,
) -> int:
    transition = _get_transition_prob(config, persona)
    min_views, max_views = _get_view_bounds(config, persona)

    sampled_views = rng.randint(min_views, max_views)
    sampled_length = 1 + sampled_views

    if rng.random() < transition["view_to_cart"]:
        sampled_length += 1
        if rng.random() < transition["cart_to_purchase"]:
            sampled_length += 1
    elif rng.random() < transition["direct_purchase_from_view"]:
        sampled_length += 1

    desired_length = min(sampled_length, remaining_slots)
    desired_length = max(2, desired_length)

    if remaining_slots - desired_length == 1:
        if desired_length < remaining_slots:
            desired_length += 1
        elif desired_length > 2:
            desired_length -= 1

    return min(desired_length, remaining_slots)


def _build_product_store(products_df: pd.DataFrame) -> dict[str, Any]:
    df = products_df.reset_index(drop=True).copy()
    n_rows = len(df)

    def _series_or_default(column: str, default: Any) -> pd.Series:
        if column in df.columns:
            return df[column]
        return pd.Series([default] * n_rows)

    price = pd.to_numeric(_series_or_default("price", 0.0), errors="coerce").fillna(0.0)
    popularity_seed = pd.to_numeric(
        _series_or_default("popularity_seed", 0.5),
        errors="coerce",
    ).fillna(0.5)

    top_category = _series_or_default("top_category", "").fillna("").astype(str).to_numpy()
    price_tier = _series_or_default("price_tier", "").fillna("").astype(str).to_numpy()
    color = _series_or_default("color", "").fillna("").astype(str).to_numpy()
    leaf_category = _series_or_default("leaf_category", "").fillna("").astype(str).to_numpy()
    name = _series_or_default("name", "").fillna("").astype(str).to_numpy()

    product_id_series = _series_or_default("product_id", "").fillna("").astype(str)
    product_id_series = product_id_series.str.replace(r"\.0$", "", regex=True).str.zfill(10)
    product_id = product_id_series.to_numpy()

    is_new = _series_or_default("is_new", False).fillna(False).astype(bool).to_numpy()

    all_indices = np.arange(n_rows, dtype=np.int32)

    category_to_indices: dict[str, np.ndarray] = {}
    for category in sorted(pd.Series(top_category).dropna().unique().tolist()):
        category_to_indices[str(category)] = np.flatnonzero(top_category == category).astype(
            np.int32
        )

    return {
        "all_indices": all_indices,
        "top_category": top_category,
        "price_tier": price_tier,
        "price": price.to_numpy(dtype=np.float64),
        "popularity_seed": popularity_seed.to_numpy(dtype=np.float64),
        "is_new": is_new,
        "color": color,
        "leaf_category": leaf_category,
        "name": name,
        "product_id": product_id,
        "category_to_indices": category_to_indices,
    }


def _select_candidate_indices(
    store: dict[str, Any],
    focus_category: str,
    user_row: pd.Series,
    config: dict[str, Any],
) -> np.ndarray:
    indices = store["category_to_indices"].get(focus_category, store["all_indices"])

    budget = _extract_budget(user_row)
    events_cfg = _get_events_cfg(config)
    tolerance = float(events_cfg.get("budget_tolerance_multiplier", 1.25))

    if budget is not None and indices.size > 0:
        filtered = indices[store["price"][indices] <= budget * tolerance]
        if filtered.size >= 3:
            indices = filtered

    if indices.size == 0:
        indices = store["all_indices"]

    return indices


def _build_query_text(
    focus_category: str,
    candidate_indices: np.ndarray,
    store: dict[str, Any],
    config: dict[str, Any],
    rng: random.Random,
) -> str:
    events_cfg = _get_events_cfg(config)
    query_pool_cfg = events_cfg.get("query_pool") or {}
    query_pool = query_pool_cfg.get(focus_category, [])

    if query_pool:
        base_token = rng.choice(list(query_pool))
    elif candidate_indices.size > 0:
        leaf_values = store["leaf_category"][candidate_indices]
        leaf_values = [value for value in leaf_values.tolist() if value]
        base_token = rng.choice(leaf_values) if leaf_values else focus_category
    else:
        base_token = focus_category

    if candidate_indices.size > 0 and rng.random() < 0.35:
        colors = np.unique(store["color"][candidate_indices])
        colors = [value for value in colors.tolist() if value]
        if colors:
            return f"{rng.choice(colors)} {base_token}"

    return base_token


def _build_query_text_for_product(
    focus_category: str,
    product_idx: int,
    store: dict[str, Any],
    config: dict[str, Any],
    rng: random.Random,
) -> str:
    events_cfg = _get_events_cfg(config)
    exact_prob = float(events_cfg.get("query_anchor_exact_leaf_prob", 0.72))
    color_prob = float(events_cfg.get("query_anchor_color_prob", 0.45))
    name_prob = float(events_cfg.get("query_anchor_name_prob", 0.35))
    leaf = str(store["leaf_category"][product_idx] or focus_category).strip() or focus_category
    color = str(store["color"][product_idx] or "").strip()
    name = str(store["name"][product_idx] or "").strip()
    if name and rng.random() < name_prob:
        token = f"{name} {leaf}"
    else:
        token = leaf if rng.random() < exact_prob else focus_category
    if color and rng.random() < color_prob:
        return f"{color} {token}"
    return token


def _score_candidate_indices(
    candidate_indices: np.ndarray,
    store: dict[str, Any],
    user_row: pd.Series,
    config: dict[str, Any],
) -> np.ndarray:
    preferred_top_category = str(user_row.get("preferred_top_category", "") or "").strip()
    preferred_tiers = _parse_list_like(user_row.get("preferred_price_tiers"))
    budget = _extract_budget(user_row)

    events_cfg = _get_events_cfg(config)
    tolerance = float(events_cfg.get("budget_tolerance_multiplier", 1.25))

    weights = np.ones(candidate_indices.shape[0], dtype=np.float64)

    candidate_categories = store["top_category"][candidate_indices]
    candidate_tiers = store["price_tier"][candidate_indices]
    candidate_prices = store["price"][candidate_indices]
    candidate_popularity = store["popularity_seed"][candidate_indices]
    candidate_is_new = store["is_new"][candidate_indices]
    popularity_strength = float(events_cfg.get("search_result_popularity_strength", 2.50))

    if preferred_top_category:
        weights *= np.where(candidate_categories == preferred_top_category, 1.8, 1.0)

    if preferred_tiers:
        weights *= np.where(np.isin(candidate_tiers, preferred_tiers), 1.6, 1.0)

    if budget is not None:
        price_multiplier = np.where(
            candidate_prices <= budget,
            1.45,
            np.where(candidate_prices <= budget * tolerance, 1.12, 0.65),
        )
        weights *= price_multiplier

    weights *= np.where(candidate_is_new, 1.05, 1.0)
    weights *= 0.25 + popularity_strength * candidate_popularity

    return np.maximum(weights, 0.001)


def _sample_product_indices(
    candidate_indices: np.ndarray,
    weights: np.ndarray,
    sample_size: int,
    np_rng: np.random.Generator,
) -> np.ndarray:
    if candidate_indices.size == 0 or sample_size <= 0:
        return np.array([], dtype=np.int32)

    probabilities = weights / weights.sum()
    replace = candidate_indices.size < sample_size

    chosen = np_rng.choice(
        candidate_indices,
        size=sample_size,
        replace=replace,
        p=probabilities,
    )

    return np.asarray(chosen, dtype=np.int32)


def _make_product_event(
    event_idx: int,
    session_id: str,
    user_id: str,
    persona: str,
    event_type: str,
    timestamp: datetime,
    query_text: str,
    product_idx: int,
    position: int | None,
    source_reason: str,
    store: dict[str, Any],
) -> dict[str, Any]:
    return {
        "event_id": f"E{event_idx:010d}",
        "session_id": session_id,
        "user_id": user_id,
        "persona": persona,
        "event_type": event_type,
        "timestamp": timestamp.isoformat(),
        "query_text": query_text,
        "product_id": store["product_id"][product_idx],
        "top_category": store["top_category"][product_idx],
        "price_tier": store["price_tier"][product_idx],
        "price": float(store["price"][product_idx]),
        "position": position,
        "source_reason": source_reason,
    }


def _build_session_events(
    session_id: str,
    user_row: pd.Series,
    candidate_indices: np.ndarray,
    store: dict[str, Any],
    config: dict[str, Any],
    rng: random.Random,
    np_rng: np.random.Generator,
    start_time: datetime,
    event_idx_start: int,
    desired_length: int,
    focus_category: str,
    query_text: str,
) -> tuple[list[dict[str, Any]], int]:
    user_id = str(user_row.get("user_id"))
    persona = str(user_row.get("persona", "pragmatist"))
    cursor = start_time
    event_idx = event_idx_start

    session_events: list[dict[str, Any]] = []

    session_events.append(
        {
            "event_id": f"E{event_idx:010d}",
            "session_id": session_id,
            "user_id": user_id,
            "persona": persona,
            "event_type": "search",
            "timestamp": cursor.isoformat(),
            "query_text": query_text,
            "product_id": None,
            "top_category": focus_category,
            "price_tier": None,
            "price": None,
            "position": None,
            "source_reason": "session_start",
        }
    )
    event_idx += 1
    cursor += timedelta(seconds=rng.randint(3, 15))

    remaining_after_search = desired_length - 1
    transition = _get_transition_prob(config, persona)

    has_cart = False
    has_purchase = False
    purchase_reason = ""

    if remaining_after_search >= 3 and rng.random() < transition["view_to_cart"]:
        has_cart = True
        if rng.random() < transition["cart_to_purchase"]:
            has_purchase = True
            purchase_reason = "cart_to_purchase"
    elif remaining_after_search >= 2 and rng.random() < transition["direct_purchase_from_view"]:
        has_purchase = True
        purchase_reason = "view_to_purchase"
    elif remaining_after_search >= 2 and rng.random() < transition["view_to_cart"]:
        has_cart = True

    view_count = desired_length - 1 - int(has_cart) - int(has_purchase)

    while view_count < 1:
        if has_purchase and has_cart:
            has_purchase = False
            purchase_reason = ""
        elif has_cart:
            has_cart = False
        elif has_purchase:
            has_purchase = False
            purchase_reason = ""
        view_count = desired_length - 1 - int(has_cart) - int(has_purchase)

    weights = _score_candidate_indices(
        candidate_indices=candidate_indices,
        store=store,
        user_row=user_row,
        config=config,
    )

    viewed_indices = _sample_product_indices(
        candidate_indices=candidate_indices,
        weights=weights,
        sample_size=view_count,
        np_rng=np_rng,
    )
    if viewed_indices.size > 0:
        query_text = _build_query_text_for_product(
            focus_category=focus_category,
            product_idx=int(viewed_indices[0]),
            store=store,
            config=config,
            rng=rng,
        )
        session_events[0]["query_text"] = query_text

    for position, product_idx in enumerate(viewed_indices.tolist(), start=1):
        session_events.append(
            _make_product_event(
                event_idx=event_idx,
                session_id=session_id,
                user_id=user_id,
                persona=persona,
                event_type="view",
                timestamp=cursor,
                query_text=query_text,
                product_idx=int(product_idx),
                position=position,
                source_reason="search_result",
                store=store,
            )
        )
        event_idx += 1
        cursor += timedelta(seconds=rng.randint(5, 20))

    cart_product_idx: int | None = None
    if has_cart:
        cart_product_idx = int(viewed_indices[0])
        session_events.append(
            _make_product_event(
                event_idx=event_idx,
                session_id=session_id,
                user_id=user_id,
                persona=persona,
                event_type="cart",
                timestamp=cursor,
                query_text=query_text,
                product_idx=cart_product_idx,
                position=1,
                source_reason="view_to_cart",
                store=store,
            )
        )
        event_idx += 1
        cursor += timedelta(seconds=rng.randint(5, 25))

    if has_purchase:
        if purchase_reason == "cart_to_purchase" and cart_product_idx is not None:
            purchase_product_idx = cart_product_idx
        else:
            purchase_product_idx = int(viewed_indices[0])

        session_events.append(
            _make_product_event(
                event_idx=event_idx,
                session_id=session_id,
                user_id=user_id,
                persona=persona,
                event_type="purchase",
                timestamp=cursor,
                query_text=query_text,
                product_idx=purchase_product_idx,
                position=1,
                source_reason=purchase_reason,
                store=store,
            )
        )
        event_idx += 1

    return session_events, event_idx


def validate_events(events_df: pd.DataFrame) -> list[str]:
    issues: list[str] = []

    missing_columns = [
        column for column in REQUIRED_EVENT_COLUMNS if column not in events_df.columns
    ]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")
        return issues

    if events_df.empty:
        issues.append("events_df is empty.")
        return issues

    invalid_types = sorted(set(events_df["event_type"]) - ALLOWED_EVENT_TYPES)
    if invalid_types:
        issues.append(f"Invalid event types found: {invalid_types}")

    required_non_null_columns = [
        "event_id",
        "session_id",
        "user_id",
        "persona",
        "event_type",
        "timestamp",
    ]
    for column in required_non_null_columns:
        null_count = int(events_df[column].isna().sum())
        if null_count > 0:
            issues.append(f"Column '{column}' has {null_count} null values.")

    non_search_missing_product = events_df[
        (events_df["event_type"] != "search") & (events_df["product_id"].isna())
    ]
    if not non_search_missing_product.empty:
        issues.append(f"Non-search events with null product_id: {len(non_search_missing_product)}")

    sorted_df = events_df.sort_values(["session_id", "timestamp", "event_id"]).reset_index(
        drop=True
    )
    session_ids = sorted_df["session_id"].astype(str)
    event_types = sorted_df["event_type"].astype(str)

    search_counts = event_types.eq("search").groupby(session_ids).sum()
    invalid_search_counts = search_counts[search_counts.ne(1)]
    for session_id, search_count in invalid_search_counts.head(20).items():
        issues.append(f"Session {session_id} has {int(search_count)} search events.")
    if len(invalid_search_counts) > 20:
        issues.append(
            f"{len(invalid_search_counts) - 20} more sessions have invalid search counts."
        )

    first_events = event_types.groupby(session_ids).first()
    invalid_first_events = first_events[first_events.ne("search")]
    for session_id, first_event_type in invalid_first_events.head(20).items():
        issues.append(f"Session {session_id} does not start with search: {first_event_type}.")
    if len(invalid_first_events) > 20:
        issues.append(f"{len(invalid_first_events) - 20} more sessions do not start with search.")

    product_df = sorted_df[sorted_df["product_id"].notna()].copy()
    if not product_df.empty:
        key = product_df["session_id"].astype(str) + "\x1f" + product_df["product_id"].astype(str)
        is_view = product_df["event_type"].eq("view").astype(int)
        is_cart = product_df["event_type"].eq("cart").astype(int)
        seen_view_before = (is_view.groupby(key).cumsum() - is_view).gt(0)
        seen_cart_before = (is_cart.groupby(key).cumsum() - is_cart).gt(0)

        invalid_carts = product_df[product_df["event_type"].eq("cart") & ~seen_view_before]
        for row in invalid_carts.head(20).itertuples(index=False):
            issues.append(
                f"Session {row.session_id} has cart product without prior view: {row.product_id}"
            )
        if len(invalid_carts) > 20:
            issues.append(f"{len(invalid_carts) - 20} more cart events have no prior view.")

        invalid_purchases = product_df[
            product_df["event_type"].eq("purchase") & ~(seen_view_before | seen_cart_before)
        ]
        for row in invalid_purchases.head(20).itertuples(index=False):
            issues.append(
                "Session "
                f"{row.session_id} has purchase product without prior view/cart: "
                f"{row.product_id}"
            )
        if len(invalid_purchases) > 20:
            issues.append(
                f"{len(invalid_purchases) - 20} more purchase events have no prior view/cart."
            )

    return issues


def generate_events(
    users_df: pd.DataFrame,
    products_df: pd.DataFrame,
    config: dict[str, Any],
    rng: random.Random,
) -> pd.DataFrame:
    pref_cols = _extract_preference_columns(users_df)
    fallback_categories = _extract_fallback_categories(products_df, pref_cols)

    simulator_cfg = _get_simulator_cfg(config)
    scale_cfg = simulator_cfg.get("scale") or {}
    events_cfg = _get_events_cfg(config)

    target_event_count = int(scale_cfg.get("events", 5000))
    lookback_days = int(events_cfg.get("lookback_days", 30))

    now = datetime.utcnow()
    users_records = users_df.to_dict("records")
    if not users_records:
        return pd.DataFrame(columns=REQUIRED_EVENT_COLUMNS)

    rng.shuffle(users_records)

    store = _build_product_store(products_df)
    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))

    events: list[dict[str, Any]] = []
    event_idx = 1
    session_idx = 1
    user_pointer = 0
    progress_every = int(events_cfg.get("progress_every_events", 100_000))
    last_progress_bucket = 0

    while len(events) < target_event_count:
        remaining_slots = target_event_count - len(events)
        if remaining_slots < 2:
            break

        user_record = users_records[user_pointer % len(users_records)]
        user_pointer += 1

        user_row = pd.Series(user_record)
        persona = str(user_row.get("persona", "pragmatist"))

        session_id = f"S{session_idx:09d}"
        session_idx += 1

        base_time = now - timedelta(
            days=rng.randint(0, max(lookback_days - 1, 0)),
            hours=rng.randint(0, 23),
            minutes=rng.randint(0, 59),
            seconds=rng.randint(0, 59),
        )

        focus_category = _sample_focus_category(
            user_row=user_row,
            pref_cols=pref_cols,
            fallback_categories=fallback_categories,
            rng=rng,
        )

        candidate_indices = _select_candidate_indices(
            store=store,
            focus_category=focus_category,
            user_row=user_row,
            config=config,
        )

        query_text = _build_query_text(
            focus_category=focus_category,
            candidate_indices=candidate_indices,
            store=store,
            config=config,
            rng=rng,
        )

        desired_length = _sample_target_session_length(
            persona=persona,
            config=config,
            remaining_slots=remaining_slots,
            rng=rng,
        )

        session_events, event_idx = _build_session_events(
            session_id=session_id,
            user_row=user_row,
            candidate_indices=candidate_indices,
            store=store,
            config=config,
            rng=rng,
            np_rng=np_rng,
            start_time=base_time,
            event_idx_start=event_idx,
            desired_length=desired_length,
            focus_category=focus_category,
            query_text=query_text,
        )

        events.extend(session_events)
        if progress_every > 0:
            progress_bucket = len(events) // progress_every
            if progress_bucket > last_progress_bucket:
                last_progress_bucket = progress_bucket
                print(
                    "[simulator] generated_events="
                    f"{min(len(events), target_event_count):,}/{target_event_count:,} "
                    f"sessions={session_idx - 1:,}",
                    flush=True,
                )

    events_df = pd.DataFrame(events, columns=REQUIRED_EVENT_COLUMNS)
    events_df = events_df.sort_values(["timestamp", "event_id"]).reset_index(drop=True)

    return events_df
