from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mars.config.settings import MarsConfig, ensure_runtime_dirs
from mars.data.io import write_json, write_table

EVENT_WEIGHTS = {"search": 0.05, "view": 0.20, "cart": 0.70, "purchase": 1.00}
DEVICE_POOL = ("mobile", "desktop", "tablet")
PERSONA_SIGNALS: dict[str, dict[str, float]] = {
    "trendsetter": {
        "trend_affinity": 0.92,
        "category_loyalty": 0.30,
        "exploration_rate": 0.28,
        "session_frequency": 0.82,
    },
    "pragmatist": {
        "trend_affinity": 0.38,
        "category_loyalty": 0.42,
        "exploration_rate": 0.12,
        "session_frequency": 0.62,
    },
    "value_seeker": {
        "trend_affinity": 0.26,
        "category_loyalty": 0.20,
        "exploration_rate": 0.16,
        "session_frequency": 0.70,
    },
    "top_category_loyalist": {
        "trend_affinity": 0.48,
        "category_loyalty": 0.88,
        "exploration_rate": 0.08,
        "session_frequency": 0.58,
    },
    "impulse_buyer": {
        "trend_affinity": 0.72,
        "category_loyalty": 0.28,
        "exploration_rate": 0.24,
        "session_frequency": 0.88,
    },
    "careful_explorer": {
        "trend_affinity": 0.30,
        "category_loyalty": 0.44,
        "exploration_rate": 0.10,
        "session_frequency": 0.54,
    },
}


@dataclass(frozen=True)
class RuntimeManifest:
    path: Path
    payload: dict[str, Any]


def prepare_runtime_dataset(
    config: MarsConfig,
    *,
    rebuild_raw: bool = False,
    clean_processed: bool = False,
) -> RuntimeManifest:
    """Build the processed runtime tables used by serving and evaluation."""
    ensure_runtime_dirs(config)
    if clean_processed:
        _clean_processed_outputs(config.paths.processed_dir)
    raw_paths = _ensure_raw_simulator_outputs(config, rebuild_raw=rebuild_raw)
    raw_products = _load_products(raw_paths["products"])
    raw_users = _load_users(raw_paths["users"])
    raw_events = _load_events(raw_paths["events"])
    selected_products, selected_users, selected_events = _select_mode_slice(
        raw_products=raw_products,
        raw_users=raw_users,
        raw_events=raw_events,
        config=config,
    )

    scale_factor = _price_scale_factor(selected_products["price"])
    products = _normalise_products(selected_products, scale_factor)
    users = _normalise_users(selected_users)
    events = _normalise_events(selected_events, products, scale_factor)
    sessions = _build_sessions(events)
    search_queries = _build_search_queries(products, config)
    reco_interactions = _build_reco_interactions(events, products)
    train_events, valid_events, test_events = _split_events(events, config)

    written_files = {
        "products": str(write_table(products, config.paths.processed_dir / "products")),
        "users": str(write_table(users, config.paths.processed_dir / "users")),
        "events": str(write_table(events, config.paths.processed_dir / "events")),
        "sessions": str(write_table(sessions, config.paths.processed_dir / "sessions")),
        "search_queries": str(
            write_table(search_queries, config.paths.processed_dir / "search_queries")
        ),
        "reco_interactions": str(
            write_table(reco_interactions, config.paths.processed_dir / "reco_interactions")
        ),
        "train_events": str(write_table(train_events, config.paths.processed_dir / "train_events")),
        "valid_events": str(write_table(valid_events, config.paths.processed_dir / "valid_events")),
        "test_events": str(write_table(test_events, config.paths.processed_dir / "test_events")),
    }

    manifest = {
        "schema_version": "hm-runtime.v1",
        "generator_version": "mars.data.hm_pipeline.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": config.active_mode,
        "seed": config.seed,
        "config_hash": _config_hash(config),
        "data_source": "hm",
        "price_scale_factor": scale_factor,
        "expected_counts": {
            "products": config.mode.products,
            "users": config.mode.users,
            "events": config.mode.events,
        },
        "row_counts": {
            "products": int(len(products)),
            "users": int(len(users)),
            "events": int(len(events)),
            "sessions": int(len(sessions)),
            "search_queries": int(len(search_queries)),
            "reco_interactions": int(len(reco_interactions)),
            "train_events": int(len(train_events)),
            "valid_events": int(len(valid_events)),
            "test_events": int(len(test_events)),
        },
        "files": written_files,
        "time_range": {
            "min": str(events["timestamp"].min()) if not events.empty else None,
            "max": str(events["timestamp"].max()) if not events.empty else None,
        },
        "persona_distribution": {
            key: int(value) for key, value in users["persona"].value_counts().sort_index().items()
        },
        "event_distribution": {
            key: int(value)
            for key, value in events["event_type"].value_counts().sort_index().items()
        },
        "category_distribution": {
            key: int(value)
            for key, value in products["category_l1"].value_counts().sort_index().items()
        },
        "search_label_source": _search_label_config(config).get("source", "microsoft_hnm_search"),
        "raw_inputs": {key: str(path) for key, path in raw_paths.items()},
    }
    manifest_path = write_json(manifest, config.paths.processed_dir / "manifest.json")
    return RuntimeManifest(path=manifest_path, payload=manifest)


def _clean_processed_outputs(processed_dir: Path) -> None:
    stems = (
        "products",
        "users",
        "events",
        "sessions",
        "search_queries",
        "reco_interactions",
        "train_events",
        "valid_events",
        "test_events",
        "manifest",
    )
    for stem in stems:
        for suffix in (".parquet", ".csv", ".json"):
            target = processed_dir / f"{stem}{suffix}"
            if target.exists():
                target.unlink()


def _ensure_raw_simulator_outputs(config: MarsConfig, *, rebuild_raw: bool) -> dict[str, Path]:
    raw_dir = config.paths.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    products = raw_dir / "products.csv"
    users = raw_dir / "users.csv"
    events = raw_dir / "events.csv"

    if rebuild_raw or not products.exists() or not users.exists():
        from mars.data.raw_simulator.service import run_day2_sample_generation

        run_day2_sample_generation()
    if rebuild_raw or not events.exists():
        from mars.data.raw_simulator.service import run_day3_event_generation

        run_day3_event_generation()
    return {"products": products, "users": users, "events": events}


def _load_products(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"product_id": str, "product_code": str})
    frame["product_id"] = _normalize_id_series(frame["product_id"])
    for column in (
        "name",
        "description",
        "top_category",
        "mid_category",
        "leaf_category",
        "color",
        "style_tags",
        "image_path",
        "source",
        "price_tier",
    ):
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str)
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce").fillna(0.0)
    frame["popularity_seed"] = pd.to_numeric(
        frame.get("popularity_seed", 0.0), errors="coerce"
    ).fillna(0.0)
    frame["has_image"] = (
        pd.to_numeric(frame.get("has_image", 0), errors="coerce").fillna(0).astype(int)
    )
    frame["is_new"] = frame.get("is_new", False).map(_to_bool)
    frame = frame.sort_values(
        ["popularity_seed", "product_id"], ascending=[False, True]
    ).reset_index(drop=True)
    return frame


def _load_users(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"user_id": str})
    frame["user_id"] = frame["user_id"].astype(str).str.strip()
    for column in (
        "persona",
        "preferred_top_categories",
        "preferred_top_category",
        "preferred_price_tiers",
    ):
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str)
    for column in (
        "price_sensitivity",
        "base_conversion_rate",
        "budget_min",
        "budget_max",
        "signup_days_ago",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _load_events(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={
            "event_id": str,
            "session_id": str,
            "user_id": str,
            "persona": str,
            "event_type": str,
            "query_text": str,
            "product_id": str,
            "top_category": str,
            "price_tier": str,
        },
    )
    frame["event_id"] = frame["event_id"].astype(str).str.strip()
    frame["session_id"] = frame["session_id"].astype(str).str.strip()
    frame["user_id"] = frame["user_id"].astype(str).str.strip()
    frame["product_id"] = _normalize_nullable_id_series(frame["product_id"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame["position"] = pd.to_numeric(_series_or_default(frame, "position"), errors="coerce")
    frame["price"] = pd.to_numeric(_series_or_default(frame, "price"), errors="coerce")
    frame["query_text"] = _series_or_default(frame, "query_text", "").fillna("").astype(str)
    frame["source_reason"] = _series_or_default(frame, "source_reason", "").fillna("").astype(str)
    frame = (
        frame.dropna(subset=["timestamp"])
        .sort_values(["timestamp", "event_id"])
        .reset_index(drop=True)
    )
    return frame


def _select_mode_slice(
    *,
    raw_products: pd.DataFrame,
    raw_users: pd.DataFrame,
    raw_events: pd.DataFrame,
    config: MarsConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config.active_mode == "full":
        return raw_products.copy(), raw_users.copy(), raw_events.copy()

    target_users = min(config.mode.users, len(raw_users))
    target_products = min(config.mode.products, len(raw_products))
    target_events = config.mode.events

    selected_user_ids = _select_persona_balanced_user_ids(
        raw_users=raw_users,
        raw_events=raw_events,
        target_users=target_users,
        config=config,
    )
    users = raw_users[raw_users["user_id"].isin(selected_user_ids)].copy()

    filtered_events = raw_events[raw_events["user_id"].isin(selected_user_ids)].copy()
    product_activity = (
        filtered_events[filtered_events["product_id"].notna()]
        .groupby("product_id", as_index=False)
        .size()
        .sort_values(["size", "product_id"], ascending=[False, True])
    )
    selected_product_ids = set(product_activity["product_id"].head(target_products).tolist())
    products = raw_products[raw_products["product_id"].isin(selected_product_ids)].copy()

    keep_event = filtered_events["event_type"].eq("search") | filtered_events["product_id"].isin(
        selected_product_ids
    )
    filtered_events = filtered_events[keep_event].copy()
    events = _trim_sessions_to_event_budget(filtered_events, target_events)

    used_products = set(events["product_id"].dropna().astype(str))
    if len(used_products) < target_products:
        extra = raw_products[~raw_products["product_id"].isin(used_products)].head(
            target_products - len(used_products)
        )
        products = pd.concat([products, extra], ignore_index=True)
    products = products.drop_duplicates(subset=["product_id"]).reset_index(drop=True)

    used_users = set(events["user_id"].astype(str))
    users = users[users["user_id"].isin(used_users)].copy().reset_index(drop=True)
    return products.reset_index(drop=True), users, events.reset_index(drop=True)


def _select_persona_balanced_user_ids(
    *,
    raw_users: pd.DataFrame,
    raw_events: pd.DataFrame,
    target_users: int,
    config: MarsConfig,
) -> list[str]:
    user_activity = raw_events.groupby("user_id", as_index=False).size()
    pool = (
        raw_users[["user_id", "persona"]]
        .drop_duplicates(subset=["user_id"])
        .merge(
            user_activity,
            on="user_id",
            how="left",
        )
    )
    pool["size"] = pd.to_numeric(pool["size"], errors="coerce").fillna(0).astype(int)
    pool["persona"] = pool["persona"].fillna("unknown").astype(str)

    ratios = _configured_persona_ratios(config)
    available_counts = pool["persona"].value_counts().to_dict()
    quotas = _persona_user_quotas(ratios, available_counts, target_users)
    if not quotas:
        return (
            pool.sort_values(["size", "user_id"], ascending=[False, True])["user_id"]
            .head(target_users)
            .astype(str)
            .tolist()
        )

    selected: list[pd.DataFrame] = []
    for persona, quota in quotas.items():
        if quota <= 0:
            continue
        group = pool[pool["persona"].eq(persona)].sort_values(
            ["size", "user_id"], ascending=[False, True]
        )
        selected.append(group.head(quota))

    selected_frame = pd.concat(selected, ignore_index=True) if selected else pool.iloc[:0].copy()
    selected_ids = set(selected_frame["user_id"].astype(str))
    if len(selected_ids) < target_users:
        remaining = pool[~pool["user_id"].astype(str).isin(selected_ids)]
        fill = remaining.sort_values(["size", "user_id"], ascending=[False, True]).head(
            target_users - len(selected_ids)
        )
        selected_frame = pd.concat([selected_frame, fill], ignore_index=True)

    return (
        selected_frame.drop_duplicates(subset=["user_id"])
        .sort_values(["persona", "size", "user_id"], ascending=[True, False, True])["user_id"]
        .head(target_users)
        .astype(str)
        .tolist()
    )


def _configured_persona_ratios(config: MarsConfig) -> dict[str, float]:
    simulator = config.raw.get("simulator", {}) if isinstance(config.raw, dict) else {}
    personas = simulator.get("personas", {}) if isinstance(simulator, dict) else {}
    ratios: dict[str, float] = {}
    for persona, payload in personas.items():
        if not isinstance(payload, dict):
            continue
        ratio = float(payload.get("ratio", 0.0) or 0.0)
        if ratio > 0:
            ratios[str(persona)] = ratio

    total = sum(ratios.values())
    if total <= 0:
        return {}
    return {persona: ratio / total for persona, ratio in ratios.items()}


def _persona_user_quotas(
    ratios: dict[str, float],
    available_counts: dict[str, int],
    target_users: int,
) -> dict[str, int]:
    eligible = {
        persona: ratio
        for persona, ratio in ratios.items()
        if int(available_counts.get(persona, 0)) > 0
    }
    if not eligible:
        return {}

    total_ratio = sum(eligible.values())
    raw_quotas = {
        persona: (ratio / total_ratio) * target_users for persona, ratio in eligible.items()
    }
    quotas = {
        persona: min(int(available_counts.get(persona, 0)), int(np.floor(quota)))
        for persona, quota in raw_quotas.items()
    }

    if target_users >= len(eligible):
        for persona in eligible:
            if quotas[persona] == 0:
                quotas[persona] = 1

    while sum(quotas.values()) > target_users:
        persona = max(quotas, key=lambda key: (quotas[key] - raw_quotas[key], quotas[key]))
        quotas[persona] -= 1

    while sum(quotas.values()) < target_users:
        candidates = [
            persona
            for persona in eligible
            if quotas[persona] < int(available_counts.get(persona, 0))
        ]
        if not candidates:
            break
        persona = max(candidates, key=lambda key: (raw_quotas[key] - quotas[key], eligible[key]))
        quotas[persona] += 1

    return quotas


def _trim_sessions_to_event_budget(events: pd.DataFrame, target_events: int) -> pd.DataFrame:
    if len(events) <= target_events:
        return events.sort_values(["timestamp", "event_id"]).reset_index(drop=True)

    session_sizes = (
        events.groupby("session_id", as_index=False)
        .agg(session_start=("timestamp", "min"), size=("event_id", "count"))
        .sort_values(["session_start", "session_id"])
    )

    selected_sessions: list[str] = []
    running = 0
    for row in session_sizes.itertuples(index=False):
        if selected_sessions and running >= target_events:
            break
        selected_sessions.append(str(row.session_id))
        running += int(row.size)

    return (
        events[events["session_id"].isin(selected_sessions)]
        .sort_values(["timestamp", "event_id"])
        .reset_index(drop=True)
    )


def _price_scale_factor(price_series: pd.Series) -> int:
    median = float(pd.to_numeric(price_series, errors="coerce").dropna().median() or 0.0)
    return 1_000_000 if median < 10 else 1


def _normalise_products(products: pd.DataFrame, scale_factor: int) -> pd.DataFrame:
    out = products.copy()
    out["price"] = (pd.to_numeric(out["price"], errors="coerce").fillna(0.0) * scale_factor).round(
        2
    )
    out["category"] = out["top_category"]
    out["category_l1"] = out["top_category"]
    out["category_l2"] = out["mid_category"]
    out["category_l3"] = out["leaf_category"]
    out["style_tags"] = out["style_tags"].map(_split_pipe)
    out["margin_score"] = out["price"].rank(pct=True, method="average").fillna(0.0).round(6)
    out["created_at"] = (
        _series_or_default(
            out,
            "last_purchase_date",
            _series_or_default(out, "created_at", ""),
        )
        .fillna("")
        .astype(str)
    )
    out["image_path"] = out["image_path"].map(_hm_image_repo_path)
    out["popularity_prior"] = (
        pd.to_numeric(out["popularity_seed"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    )
    out["has_image"] = pd.to_numeric(out["has_image"], errors="coerce").fillna(0).astype(int)
    columns = [
        "product_id",
        "product_code",
        "name",
        "description",
        "category",
        "category_l1",
        "category_l2",
        "category_l3",
        "price",
        "price_tier",
        "color",
        "style_tags",
        "image_path",
        "has_image",
        "created_at",
        "popularity_prior",
        "margin_score",
        "is_new",
        "source",
        "top_category",
        "mid_category",
        "leaf_category",
    ]
    return out[columns].copy()


def _normalise_users(users: pd.DataFrame) -> pd.DataFrame:
    out = users.copy()
    out["preferred_categories"] = _series_or_default(out, "preferred_top_categories", "").map(
        _split_pipe
    )
    out["age_bucket"] = out["user_id"].map(
        lambda value: _stable_choice(value, ("18-24", "25-34", "35-44", "45-54", "55+"))
    )
    out["gender"] = out["user_id"].map(
        lambda value: _stable_choice(f"{value}:gender", ("female", "male", "unknown"))
    )
    out["trend_affinity"] = out["persona"].map(
        lambda value: PERSONA_SIGNALS.get(str(value), {}).get("trend_affinity", 0.45)
    )
    out["category_loyalty"] = out["persona"].map(
        lambda value: PERSONA_SIGNALS.get(str(value), {}).get("category_loyalty", 0.35)
    )
    out["exploration_rate"] = out["persona"].map(
        lambda value: PERSONA_SIGNALS.get(str(value), {}).get("exploration_rate", 0.15)
    )
    out["session_frequency"] = out["persona"].map(
        lambda value: PERSONA_SIGNALS.get(str(value), {}).get("session_frequency", 0.60)
    )
    out["created_at"] = (
        _series_or_default(out, "signup_days_ago", 0).fillna(0).map(_created_at_from_signup_days)
    )
    columns = [
        "user_id",
        "persona",
        "age_bucket",
        "gender",
        "preferred_categories",
        "price_sensitivity",
        "trend_affinity",
        "category_loyalty",
        "exploration_rate",
        "session_frequency",
        "created_at",
        "base_conversion_rate",
        "preferred_top_category",
        "preferred_top_category_tier",
        "preferred_price_tiers",
        "budget_min",
        "budget_max",
    ]
    existing = [column for column in columns if column in out.columns]
    return out[existing].copy()


def _normalise_events(
    events: pd.DataFrame, products: pd.DataFrame, scale_factor: int
) -> pd.DataFrame:
    out = events.copy()
    out["query"] = out["query_text"].replace("", pd.NA)
    out["query_intent_category"] = out["top_category"].replace("", pd.NA)
    out["rank_position"] = out["position"].astype("Int64")
    out["source"] = out["source_reason"].replace("", "simulator")
    out["device"] = out["session_id"].map(lambda value: _stable_choice(value, DEVICE_POOL))
    out["price_at_event"] = (
        pd.to_numeric(out["price"], errors="coerce").fillna(0.0) * scale_factor
    ).round(2)
    out.loc[out["event_type"].eq("search"), "price_at_event"] = np.nan
    out["dwell_time_sec"] = out.apply(_dwell_time, axis=1)
    out["is_positive"] = out["event_type"].isin(["cart", "purchase"])
    out["event_weight"] = out["event_type"].map(EVENT_WEIGHTS).fillna(0.0)
    out["category"] = out["top_category"]
    columns = [
        "event_id",
        "user_id",
        "session_id",
        "event_type",
        "product_id",
        "query",
        "query_intent_category",
        "timestamp",
        "rank_position",
        "source",
        "device",
        "persona",
        "category",
        "price_at_event",
        "dwell_time_sec",
        "is_positive",
        "event_weight",
    ]
    out = out[columns].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    product_categories = products.set_index("product_id")["category_l1"].to_dict()
    mask = out["product_id"].notna()
    out.loc[mask, "category"] = (
        out.loc[mask, "product_id"].map(product_categories).fillna(out.loc[mask, "category"])
    )
    return out.sort_values(["timestamp", "event_id"]).reset_index(drop=True)


def _build_sessions(events: pd.DataFrame) -> pd.DataFrame:
    grouped = events.groupby("session_id", sort=False)
    rows: list[dict[str, Any]] = []
    for session_id, frame in grouped:
        frame = frame.sort_values(["timestamp", "event_id"])
        rows.append(
            {
                "session_id": str(session_id),
                "user_id": str(frame["user_id"].iloc[0]),
                "persona": str(frame["persona"].iloc[0]),
                "started_at": frame["timestamp"].iloc[0],
                "ended_at": frame["timestamp"].iloc[-1],
                "num_events": int(len(frame)),
                "entry_source": str(frame["source"].iloc[0]),
                "converted": bool(frame["event_type"].isin(["purchase"]).any()),
                "last_category": next(
                    (
                        str(value)
                        for value in reversed(frame["category"].fillna("").tolist())
                        if value
                    ),
                    None,
                ),
                "last_product_id": next(
                    (
                        str(value)
                        for value in reversed(frame["product_id"].dropna().astype(str).tolist())
                    ),
                    None,
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_search_queries(products: pd.DataFrame, config: MarsConfig) -> pd.DataFrame:
    """Build search labels from Microsoft H&M synthetic query/qrels files.

    The previous implementation inferred query positives from simulator sessions.
    The project now materializes an external-style processed table once during
    dataset preparation, then evaluation consumes only this parquet output.
    """

    label_cfg = _search_label_config(config)
    source = str(label_cfg.get("source", "microsoft_hnm_search"))
    if source != "microsoft_hnm_search":
        raise ValueError(f"Unsupported search label source: {source}")

    queries_path, qrels_path = _search_label_paths(config)
    if not queries_path.exists() or not qrels_path.exists():
        raise FileNotFoundError(
            "Microsoft H&M search label files are required. "
            f"Missing queries={queries_path.exists()} qrels={qrels_path.exists()}"
        )

    queries = pd.read_csv(
        queries_path,
        dtype={"query_id": str, "transaction_id": str, "query_text": str},
    )
    qrels = pd.read_csv(
        qrels_path,
        dtype={"query_id": str, "positive_ids": str, "negative_ids": str},
    )
    catalog = products.copy()
    catalog["product_id"] = _normalize_id_series(catalog["product_id"])
    product_ids = set(catalog["product_id"].astype(str).tolist())
    category_by_product = catalog.set_index("product_id")["category_l1"].to_dict()

    qrels = qrels.merge(
        queries[["query_id", "transaction_id", "query_text"]], on="query_id", how="inner"
    )
    rows: list[dict[str, Any]] = []
    include_negatives = bool(label_cfg.get("include_negatives", True))
    for row in qrels.itertuples(index=False):
        query = str(getattr(row, "query_text", "") or "").strip()
        if not query or query.lower() in {"nan", "none", "null"}:
            continue
        positives = [
            product_id
            for product_id in _parse_qrel_ids(getattr(row, "positive_ids", ""))
            if product_id in product_ids
        ]
        if not positives:
            continue
        negatives = []
        if include_negatives:
            negatives = [
                product_id
                for product_id in _parse_qrel_ids(getattr(row, "negative_ids", ""))
                if product_id in product_ids and product_id not in positives
            ]
        category_intent = category_by_product.get(positives[0])
        raw_query_id = str(getattr(row, "query_id", "") or "")
        rows.append(
            {
                "query_id": f"HNM{raw_query_id.zfill(9)}",
                "external_query_id": raw_query_id,
                "transaction_id": str(getattr(row, "transaction_id", "") or ""),
                "query": query,
                "user_id": None,
                "timestamp": None,
                "positive_product_ids": positives,
                "negative_product_ids": negatives,
                "category_intent": category_intent,
                "persona": None,
                "source": source,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "query_id",
            "external_query_id",
            "transaction_id",
            "query",
            "user_id",
            "timestamp",
            "positive_product_ids",
            "negative_product_ids",
            "category_intent",
            "persona",
            "source",
        ],
    )


def _search_label_config(config: MarsConfig) -> dict[str, Any]:
    raw = config.raw.get("search_labels", {}) if isinstance(config.raw, dict) else {}
    if not isinstance(raw, dict):
        return {"source": "microsoft_hnm_search"}
    return raw


def _search_label_paths(config: MarsConfig) -> tuple[Path, Path]:
    label_cfg = _search_label_config(config)
    queries = Path(label_cfg.get("queries_path", "data/external/hnm_search/raw/queries.csv"))
    qrels = Path(label_cfg.get("qrels_path", "data/external/hnm_search/raw/qrels.csv"))
    return queries, qrels


def _parse_qrel_ids(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    ids: list[str] = []
    seen: set[str] = set()
    for token in str(value).replace(",", " ").split():
        product_id = _normalize_product_id_token(token)
        if product_id and product_id not in seen:
            ids.append(product_id)
            seen.add(product_id)
    return ids


def _normalize_product_id_token(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na"}:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(10)


def _build_reco_interactions(events: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    positives = events[events["product_id"].notna()].copy()
    positives["label"] = (
        positives["event_type"].map({"view": 0.2, "cart": 0.7, "purchase": 1.0}).fillna(0.0)
    )
    rows = [
        {
            "user_id": row.user_id,
            "product_id": row.product_id,
            "label": float(row.label),
            "event_weight": float(row.event_weight),
            "timestamp": row.timestamp,
            "source_event_id": row.event_id,
        }
        for row in positives.itertuples(index=False)
    ]

    product_ids = products["product_id"].astype(str).tolist()
    if not product_ids:
        return pd.DataFrame(rows)
    positives_only = positives[positives["event_type"].isin(["cart", "purchase"])]
    positive_pairs = set(zip(positives_only["user_id"], positives_only["product_id"], strict=False))
    user_ids = positives["user_id"].astype(str).drop_duplicates().tolist()
    rng = np.random.default_rng(42)
    negative_count = min(len(rows), max(1, len(events) // 3))
    for idx in range(negative_count):
        user_id = user_ids[int(rng.integers(0, len(user_ids)))]
        product_id = product_ids[int(rng.integers(0, len(product_ids)))]
        attempts = 0
        while (user_id, product_id) in positive_pairs and attempts < 12:
            product_id = product_ids[int(rng.integers(0, len(product_ids)))]
            attempts += 1
        rows.append(
            {
                "user_id": user_id,
                "product_id": product_id,
                "label": 0.0,
                "event_weight": 0.0,
                "timestamp": events["timestamp"].iloc[int(rng.integers(0, len(events)))],
                "source_event_id": f"N{idx + 1:010d}",
            }
        )
    return pd.DataFrame(rows)


def _split_events(
    events: pd.DataFrame, config: MarsConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_cfg = (
        config.raw.get("simulator", {}).get("split", {}) if isinstance(config.raw, dict) else {}
    )
    train_ratio = float(split_cfg.get("train_ratio", 0.8))
    valid_ratio = float(split_cfg.get("valid_ratio", 0.1))
    test_ratio = float(split_cfg.get("test_ratio", 0.1))
    from mars.data.raw_simulator.split import time_based_session_split

    train, valid, test = time_based_session_split(
        events_df=events.copy(),
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )
    return train, valid, test


def _normalize_id_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(10)


def _normalize_nullable_id_series(series: pd.Series) -> pd.Series:
    def normalize(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        if not text or text.lower() in {"nan", "none", "null", "na"}:
            return None
        return text.zfill(10)

    return series.map(normalize)


def _split_pipe(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def _hm_image_repo_path(relative_path: Any) -> str:
    rel = str(relative_path or "").strip().replace("\\", "/")
    if not rel:
        return ""
    return f"data/external/hm/raw/images/{rel}"


def _created_at_from_signup_days(days: Any) -> str:
    try:
        delta = int(float(days))
    except Exception:
        delta = 0
    return (datetime.now(UTC) - timedelta(days=max(delta, 0))).isoformat()


def _stable_choice(token: str, values: Iterable[str]) -> str:
    values = tuple(values)
    digest = hashlib.blake2b(str(token).encode("utf-8"), digest_size=8).digest()
    index = int.from_bytes(digest, "big") % len(values)
    return str(values[index])


def _dwell_time(row: pd.Series) -> float | None:
    event_type = str(row.get("event_type", ""))
    if event_type == "search":
        return None
    digest = hashlib.blake2b(f"{row.get('event_id')}:{event_type}".encode(), digest_size=8).digest()
    unit = int.from_bytes(digest, "big") / float(2**64 - 1)
    mean = {"view": 42.0, "cart": 78.0, "purchase": 115.0}.get(event_type, 30.0)
    return round(mean * (0.45 + unit), 3)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _config_hash(config: MarsConfig) -> str:
    payload = (
        f"{config.active_mode}:{config.mode.products}:{config.mode.users}:"
        f"{config.mode.events}:{config.seed}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _series_or_default(frame: pd.DataFrame, column: str, default: Any = None) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)
