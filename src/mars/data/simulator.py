from __future__ import annotations

import gc
import hashlib
import math
import random
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mars.config.settings import MarsConfig, ensure_runtime_dirs
from mars.data.io import read_json, read_table, write_json, write_table
from mars.schemas.core import EVENT_TYPES, PERSONAS

CATEGORY_TREE: dict[str, dict[str, list[str]]] = {
    "apparel": {
        "outerwear": ["jacket", "coat", "cardigan", "padding"],
        "tops": ["shirt", "hoodie", "knit", "blouse"],
        "bottoms": ["denim", "slacks", "skirt", "shorts"],
    },
    "shoes": {
        "sneakers": ["running", "court", "platform", "retro"],
        "formal": ["loafer", "derby", "heel", "flat"],
        "boots": ["ankle", "chelsea", "walker", "western"],
    },
    "bags": {
        "daily": ["tote", "crossbody", "backpack", "shoulder"],
        "travel": ["duffle", "carrier", "weekender", "pouch"],
        "mini": ["clutch", "phonebag", "wallet", "microbag"],
    },
    "accessories": {
        "jewelry": ["necklace", "ring", "bracelet", "earring"],
        "headwear": ["cap", "beanie", "bucket_hat", "hairband"],
        "eyewear": ["sunglasses", "glasses", "goggles", "clipon"],
    },
}

COLORS = ("black", "white", "gray", "navy", "green", "red", "pink", "silver", "cream", "cyan")
STYLE_TAGS = (
    "minimal",
    "street",
    "office",
    "athleisure",
    "luxury",
    "vintage",
    "cute",
    "genderless",
    "outdoor",
    "party",
)
AGE_BUCKETS = ("18-24", "25-34", "35-44", "45-54", "55+")
GENDERS = ("female", "male", "non_binary", "unknown")
DEVICES = ("mobile", "desktop", "tablet")
SOURCES = ("search", "recommend", "organic", "exploration")

PERSONA_PROFILE: dict[str, dict[str, float]] = {
    "trendsetter": {
        "price_sensitivity": 0.25,
        "trend_affinity": 0.95,
        "category_loyalty": 0.35,
        "exploration_rate": 0.28,
        "session_frequency": 0.82,
        "purchase_bias": 0.90,
    },
    "pragmatist": {
        "price_sensitivity": 0.55,
        "trend_affinity": 0.35,
        "category_loyalty": 0.45,
        "exploration_rate": 0.12,
        "session_frequency": 0.62,
        "purchase_bias": 1.05,
    },
    "value_seeker": {
        "price_sensitivity": 0.93,
        "trend_affinity": 0.25,
        "category_loyalty": 0.22,
        "exploration_rate": 0.18,
        "session_frequency": 0.70,
        "purchase_bias": 0.80,
    },
    "top_category_loyalist": {
        "price_sensitivity": 0.40,
        "trend_affinity": 0.50,
        "category_loyalty": 0.92,
        "exploration_rate": 0.08,
        "session_frequency": 0.58,
        "purchase_bias": 1.00,
    },
    "impulse_buyer": {
        "price_sensitivity": 0.38,
        "trend_affinity": 0.72,
        "category_loyalty": 0.30,
        "exploration_rate": 0.24,
        "session_frequency": 0.88,
        "purchase_bias": 1.45,
    },
    "careful_explorer": {
        "price_sensitivity": 0.68,
        "trend_affinity": 0.30,
        "category_loyalty": 0.48,
        "exploration_rate": 0.10,
        "session_frequency": 0.52,
        "purchase_bias": 0.62,
    },
}

EVENT_WEIGHT = {"search": 0.05, "view": 0.20, "cart": 0.70, "purchase": 1.00}


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    checks: dict[str, bool]
    details: dict[str, Any]

    def to_pretty_text(self) -> str:
        lines = ["MARS simulator validation"]
        for name, passed in self.checks.items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} {name}")
        lines.append(f"details: {self.details}")
        return "\n".join(lines)


def generate_dataset(config: MarsConfig, seed: int | None = None, clean: bool = False) -> Path:
    run_seed = config.seed if seed is None else seed
    random.seed(run_seed)
    np.random.seed(run_seed)
    rng = np.random.default_rng(run_seed)
    ensure_runtime_dirs(config)

    processed_dir = config.paths.processed_dir
    raw_dir = config.paths.raw_dir
    if clean:
        _clean_simulator_outputs(processed_dir, raw_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    products = _generate_products(config.mode.products, rng, raw_dir)
    users = _generate_users(config.mode.users, products, rng)
    events, sessions = _generate_events(config.mode.events, users, products, rng)
    search_queries = _build_search_queries(events)
    reco_interactions = _build_reco_interactions(events, products, rng)

    paths = {
        "products": str(write_table(products, processed_dir / "products")),
        "users": str(write_table(users, processed_dir / "users")),
        "events": str(write_table(events, processed_dir / "events")),
        "sessions": str(write_table(sessions, processed_dir / "sessions")),
        "search_queries": str(write_table(search_queries, processed_dir / "search_queries")),
        "reco_interactions": str(
            write_table(reco_interactions, processed_dir / "reco_interactions")
        ),
    }
    manifest = _build_manifest(
        config=config,
        seed=run_seed,
        paths=paths,
        products=products,
        users=users,
        events=events,
        sessions=sessions,
        search_queries=search_queries,
        reco_interactions=reco_interactions,
    )
    return write_json(manifest, processed_dir / "manifest.json")


def validate_manifest(manifest_path: str | Path) -> ValidationReport:
    manifest = read_json(manifest_path)
    files = manifest.get("files", {})
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {
        "mode": manifest.get("mode"),
        "row_counts": manifest.get("row_counts", {}),
    }

    checks["required_files_exist"] = all(Path(path).exists() for path in files.values())
    checks["six_personas"] = set(manifest.get("persona_distribution", {})) == set(PERSONAS)
    checks["four_event_types"] = set(manifest.get("event_distribution", {})) == set(EVENT_TYPES)

    expected = manifest.get("expected_counts", {})
    rows = manifest.get("row_counts", {})
    checks["product_count"] = rows.get("products") == expected.get("products")
    checks["user_count"] = rows.get("users") == expected.get("users")
    checks["event_count"] = rows.get("events", 0) >= expected.get("events", math.inf)
    checks["has_search_queries"] = rows.get("search_queries", 0) > 0
    checks["has_reco_interactions"] = rows.get("reco_interactions", 0) > 0

    if checks["required_files_exist"]:
        events = read_table(files["events"])
        sessions = read_table(files["sessions"])
        checks["event_ids_unique"] = events["event_id"].is_unique
        checks["session_ids_present"] = events["session_id"].notna().all()
        checks["sessions_have_events"] = (
            sessions["num_events"].min() > 0 if len(sessions) else False
        )
        details["time_range"] = {
            "min": str(events["timestamp"].min()),
            "max": str(events["timestamp"].max()),
        }
    else:
        checks["event_ids_unique"] = False
        checks["session_ids_present"] = False
        checks["sessions_have_events"] = False

    return ValidationReport(ok=all(checks.values()), checks=checks, details=details)


def summarize_manifest(manifest_path: str | Path) -> str:
    manifest = read_json(manifest_path)
    lines = [
        "MARS simulator summary",
        f"- mode: {manifest.get('mode')}",
        f"- seed: {manifest.get('seed')}",
        f"- schema_version: {manifest.get('schema_version')}",
        f"- row_counts: {manifest.get('row_counts')}",
        f"- personas: {manifest.get('persona_distribution')}",
        f"- events: {manifest.get('event_distribution')}",
        f"- generated_at: {manifest.get('generated_at')}",
    ]
    return "\n".join(lines)


def _clean_simulator_outputs(processed_dir: Path, raw_dir: Path) -> None:
    for name in (
        "products",
        "users",
        "events",
        "sessions",
        "search_queries",
        "reco_interactions",
        "manifest",
    ):
        for suffix in (".parquet", ".csv", ".json"):
            path = processed_dir / f"{name}{suffix}"
            if path.exists():
                _unlink_generated_file(path)
    image_dir = raw_dir / "images"
    if image_dir.exists():
        try:
            shutil.rmtree(image_dir)
        except PermissionError:
            # Some restricted Windows runners allow overwrites but deny deletes.
            # Keeping generated placeholders is safe because they are deterministic.
            pass


def _unlink_generated_file(path: Path) -> None:
    for attempt in range(5):
        try:
            path.unlink()
            return
        except PermissionError:
            if attempt < 4:
                gc.collect()
                time.sleep(0.05)
                continue
            if not _can_overwrite_generated_file(path):
                raise
            return


def _can_overwrite_generated_file(path: Path) -> bool:
    try:
        with path.open("ab"):
            return True
    except OSError:
        return False


def _generate_products(count: int, rng: np.random.Generator, raw_dir: Path) -> pd.DataFrame:
    image_dir = raw_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    category_rows = [
        (l1, l2, l3)
        for l1, children in CATEGORY_TREE.items()
        for l2, leaves in children.items()
        for l3 in leaves
    ]
    category_weights = rng.dirichlet(np.ones(len(category_rows)) * 1.8)
    base_date = datetime(2025, 1, 1, tzinfo=UTC)
    asset_cache: dict[tuple[str, str], str] = {}

    rows: list[dict[str, Any]] = []
    for idx in range(count):
        l1, l2, l3 = category_rows[int(rng.choice(len(category_rows), p=category_weights))]
        color = str(rng.choice(COLORS))
        tags = list(rng.choice(STYLE_TAGS, size=3, replace=False))
        tier = int(rng.integers(0, 4))
        category_price = {
            "apparel": 65_000,
            "shoes": 92_000,
            "bags": 118_000,
            "accessories": 38_000,
        }[l1]
        price_noise = float(rng.lognormal(mean=0.0, sigma=0.42))
        price = int(round((category_price * (1 + tier * 0.18) * price_noise) / 1000) * 1000)
        price = max(9_000, min(price, 690_000))
        style_phrase = " ".join(tags[:2])
        name = f"{color.title()} {style_phrase.title()} {l3.replace('_', ' ').title()}"
        description = (
            f"{color} {l3.replace('_', ' ')} for {tags[0]} {tags[1]} styling. "
            f"Designed in the {l2} line with {tags[2]} details."
        )
        rows.append(
            {
                "product_id": f"P{idx + 1:08d}",
                "name": name,
                "category_l1": l1,
                "category_l2": l2,
                "category_l3": l3,
                "price": price,
                "color": color,
                "style_tags": tags,
                "description": description,
                "image_path": _ensure_placeholder_asset(image_dir, asset_cache, l1, color),
                "created_at": (base_date + timedelta(days=int(rng.integers(0, 455)))).isoformat(),
                "popularity_prior": round(float(rng.beta(2.2, 7.0)), 6),
                "margin_score": round(float(rng.beta(3.0, 3.0)), 6),
                "is_new": bool(rng.random() < 0.16),
                "embedding_text_seed": int(rng.integers(0, 2**31 - 1)),
                "embedding_image_seed": int(rng.integers(0, 2**31 - 1)),
            }
        )
    return pd.DataFrame(rows)


def _ensure_placeholder_asset(
    image_dir: Path, asset_cache: dict[tuple[str, str], str], category: str, color: str
) -> str:
    key = (category, color)
    if key in asset_cache:
        return asset_cache[key]
    path = image_dir / f"{category}_{color}.png"
    if not path.exists():
        try:
            from PIL import Image, ImageDraw

            image = Image.new("RGB", (160, 200), (245, 245, 242))
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle(
                (28, 30, 132, 170),
                radius=12,
                fill=_color_to_rgb(color),
                outline=(30, 30, 30),
                width=3,
            )
            draw.text((24, 176), category[:12], fill=(20, 20, 20))
            image.save(path)
        except Exception:
            txt_path = path.with_suffix(".txt")
            txt_path.write_text(f"{category} {color}", encoding="utf-8")
            path = txt_path
    asset_cache[key] = str(path)
    return str(path)


def _color_to_rgb(color: str) -> tuple[int, int, int]:
    palette = {
        "black": (28, 28, 30),
        "white": (235, 235, 228),
        "gray": (136, 140, 145),
        "navy": (24, 48, 88),
        "green": (42, 130, 96),
        "red": (205, 64, 70),
        "pink": (232, 123, 164),
        "silver": (181, 188, 196),
        "cream": (230, 218, 192),
        "cyan": (32, 164, 188),
    }
    return palette[color]


def _generate_users(count: int, products: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    categories = sorted(products["category_l1"].unique())
    persona_weights = np.array([0.17, 0.18, 0.17, 0.16, 0.16, 0.16])
    base_date = datetime(2024, 1, 1, tzinfo=UTC)

    rows: list[dict[str, Any]] = []
    for idx in range(count):
        persona = str(rng.choice(PERSONAS, p=persona_weights))
        profile = PERSONA_PROFILE[persona]
        category_count = 3 if persona == "top_category_loyalist" else 2
        rows.append(
            {
                "user_id": f"U{idx + 1:08d}",
                "persona": persona,
                "age_bucket": str(rng.choice(AGE_BUCKETS, p=[0.22, 0.34, 0.22, 0.14, 0.08])),
                "gender": str(rng.choice(GENDERS, p=[0.48, 0.44, 0.04, 0.04])),
                "preferred_categories": list(
                    rng.choice(categories, size=category_count, replace=False)
                ),
                "price_sensitivity": _jitter(profile["price_sensitivity"], rng),
                "trend_affinity": _jitter(profile["trend_affinity"], rng),
                "category_loyalty": _jitter(profile["category_loyalty"], rng),
                "exploration_rate": _jitter(profile["exploration_rate"], rng, scale=0.04),
                "session_frequency": _jitter(profile["session_frequency"], rng),
                "created_at": (base_date + timedelta(days=int(rng.integers(0, 640)))).isoformat(),
            }
        )
    return pd.DataFrame(rows)


def _jitter(value: float, rng: np.random.Generator, scale: float = 0.08) -> float:
    return round(float(np.clip(value + rng.normal(0, scale), 0.01, 0.99)), 6)


def _generate_events(
    count: int, users: pd.DataFrame, products: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    product_lookup = products.set_index("product_id")
    query_examples = {
        category: frame[["color", "category_l1", "category_l3", "style_tags"]].to_dict("records")
        for category, frame in products.groupby("category_l1")
    }
    products_by_category = {
        category: frame.sort_values("popularity_prior", ascending=False)["product_id"].to_numpy()
        for category, frame in products.groupby("category_l1")
    }
    all_products = products.sort_values("popularity_prior", ascending=False)[
        "product_id"
    ].to_numpy()

    start = datetime(2026, 1, 1, tzinfo=UTC)
    events: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    event_id = 1
    session_id = 1

    while event_id <= count:
        user = users.iloc[int(rng.integers(0, len(users)))]
        persona = str(user["persona"])
        target_len = int(np.clip(rng.poisson(_session_lambda(persona)) + 2, 3, 18))
        target_len = min(target_len, count - event_id + 1)
        sid = f"S{session_id:09d}"
        session_start = start + timedelta(minutes=int(rng.integers(0, 120 * 24 * 60)))
        last_product: str | None = None
        last_category: str | None = None
        converted = False
        entry_source = str(rng.choice(SOURCES, p=[0.36, 0.42, 0.16, 0.06]))

        session_start_index = len(events)
        for offset in range(target_len):
            event_type = _choose_event_type(offset, last_product is not None, persona, rng)
            product_id: str | None = None
            query: str | None = None
            query_category: str | None = None
            rank_position: int | None = None
            price_at_event: int | None = None

            if event_type == "search":
                query_category = _choose_category_for_user(user, rng)
                query = _make_query(query_category, query_examples, rng)
            else:
                product_id = _choose_product_for_user(
                    user=user,
                    products_by_category=products_by_category,
                    all_products=all_products,
                    rng=rng,
                )
                product = product_lookup.loc[product_id]
                last_product = product_id
                last_category = str(product["category_l1"])
                query_category = last_category
                price_at_event = int(product["price"])
                rank_position = int(rng.integers(1, 80))
                if event_type == "purchase":
                    converted = True

            timestamp = session_start + timedelta(seconds=offset * int(rng.integers(18, 150)))
            source = (
                entry_source
                if offset == 0
                else str(rng.choice(SOURCES, p=[0.25, 0.50, 0.18, 0.07]))
            )
            events.append(
                {
                    "event_id": f"E{event_id:010d}",
                    "user_id": str(user["user_id"]),
                    "session_id": sid,
                    "event_type": event_type,
                    "product_id": product_id,
                    "query": query,
                    "query_intent_category": query_category,
                    "timestamp": timestamp.isoformat(),
                    "rank_position": rank_position,
                    "source": source,
                    "device": str(rng.choice(DEVICES, p=[0.72, 0.22, 0.06])),
                    "persona": persona,
                    "price_at_event": price_at_event,
                    "dwell_time_sec": _dwell_time(event_type, rng),
                    "is_positive": event_type in ("cart", "purchase"),
                    "event_weight": EVENT_WEIGHT[event_type],
                }
            )
            event_id += 1
            if event_id > count:
                break

        session_events = events[session_start_index:]
        sessions.append(
            {
                "session_id": sid,
                "user_id": str(user["user_id"]),
                "persona": persona,
                "started_at": session_events[0]["timestamp"],
                "ended_at": session_events[-1]["timestamp"],
                "num_events": len(session_events),
                "entry_source": entry_source,
                "converted": converted,
                "last_category": last_category,
                "last_product_id": last_product,
            }
        )
        session_id += 1

    return pd.DataFrame(events), pd.DataFrame(sessions)


def _session_lambda(persona: str) -> float:
    return {
        "trendsetter": 6.0,
        "pragmatist": 5.5,
        "value_seeker": 7.0,
        "top_category_loyalist": 4.8,
        "impulse_buyer": 3.2,
        "careful_explorer": 9.0,
    }[persona]


def _choose_event_type(
    offset: int, has_product_context: bool, persona: str, rng: np.random.Generator
) -> str:
    if offset == 0:
        return str(rng.choice(["search", "view"], p=[0.58, 0.42]))
    profile = PERSONA_PROFILE[persona]
    purchase_p = 0.025 * profile["purchase_bias"]
    cart_p = 0.095 * (1.15 if has_product_context else 0.75)
    if persona == "careful_explorer":
        probs = [0.34, 0.52, cart_p, purchase_p]
    elif persona == "impulse_buyer":
        probs = [0.12, 0.60, 0.17, purchase_p]
    else:
        probs = [0.20, 0.61, cart_p, purchase_p]
    total = sum(probs)
    return str(rng.choice(EVENT_TYPES, p=[p / total for p in probs]))


def _choose_category_for_user(user: pd.Series, rng: np.random.Generator) -> str:
    preferred = list(user["preferred_categories"])
    if rng.random() < 0.78:
        return str(rng.choice(preferred))
    return str(rng.choice(list(CATEGORY_TREE)))


def _make_query(
    category: str, query_examples: dict[str, list[dict[str, Any]]], rng: np.random.Generator
) -> str:
    examples = query_examples[category]
    sample = examples[int(rng.integers(0, len(examples)))]
    template = str(
        rng.choice(
            [
                "{color} {category}",
                "{style} {category}",
                "{color} {style} {category}",
                "{color} {leaf}",
            ]
        )
    )
    return template.format(
        color=sample["color"],
        category=category,
        style=sample["style_tags"][0],
        leaf=str(sample["category_l3"]).replace("_", " "),
    )


def _choose_product_for_user(
    user: pd.Series,
    products_by_category: dict[str, np.ndarray],
    all_products: np.ndarray,
    rng: np.random.Generator,
) -> str:
    if rng.random() < float(user["category_loyalty"]) or rng.random() > float(
        user["exploration_rate"]
    ):
        category = str(rng.choice(list(user["preferred_categories"])))
        pool = products_by_category.get(category, all_products)
    else:
        pool = all_products
    max_window = min(len(pool), max(50, int(len(pool) * 0.18)))
    if rng.random() < 0.68:
        return str(pool[int(rng.integers(0, max_window))])
    return str(pool[int(rng.integers(0, len(pool)))])


def _dwell_time(event_type: str, rng: np.random.Generator) -> float | None:
    if event_type == "search":
        return None
    mean = {"view": 42, "cart": 78, "purchase": 115}[event_type]
    return round(float(rng.gamma(shape=2.0, scale=mean / 2.0)), 3)


def _build_search_queries(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    query_idx = 1
    for _, session in events.groupby("session_id", sort=False):
        records = list(session.itertuples(index=False))
        future_products: list[list[str]] = [[] for _ in records]
        rolling: list[str] = []
        for idx in range(len(records) - 1, -1, -1):
            product_id = records[idx].product_id
            if product_id is not None and not pd.isna(product_id):
                rolling = [str(product_id), *rolling[:4]]
            future_products[idx] = list(dict.fromkeys(rolling[:5]))

        for idx, row in enumerate(records):
            if row.event_type != "search" or not future_products[idx]:
                continue
            rows.append(
                {
                    "query_id": f"Q{query_idx:010d}",
                    "query": row.query,
                    "user_id": row.user_id,
                    "timestamp": row.timestamp,
                    "positive_product_ids": future_products[idx],
                    "category_intent": row.query_intent_category,
                    "persona": row.persona,
                }
            )
            query_idx += 1
    return pd.DataFrame(rows)


def _build_reco_interactions(
    events: pd.DataFrame, products: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    positives = events[events["product_id"].notna()].copy()
    positives["label"] = positives["event_type"].map({"view": 0.2, "cart": 0.7, "purchase": 1.0})
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

    product_ids = products["product_id"].to_numpy()
    positive_pairs = set(zip(positives["user_id"], positives["product_id"], strict=False))
    negative_count = min(len(rows), max(1, len(events) // 3))
    users = positives["user_id"].to_numpy()
    for idx in range(negative_count):
        user_id = str(users[int(rng.integers(0, len(users)))])
        product_id = str(product_ids[int(rng.integers(0, len(product_ids)))])
        attempts = 0
        while (user_id, product_id) in positive_pairs and attempts < 10:
            product_id = str(product_ids[int(rng.integers(0, len(product_ids)))])
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


def _build_manifest(
    config: MarsConfig,
    seed: int,
    paths: dict[str, str],
    products: pd.DataFrame,
    users: pd.DataFrame,
    events: pd.DataFrame,
    sessions: pd.DataFrame,
    search_queries: pd.DataFrame,
    reco_interactions: pd.DataFrame,
) -> dict[str, Any]:
    config_fingerprint = hashlib.sha256(
        f"{config.active_mode}:{config.mode.products}:{config.mode.users}:"
        f"{config.mode.events}:{seed}".encode()
    ).hexdigest()[:16]
    return {
        "schema_version": "simulator.v1",
        "generator_version": "mars.data.simulator.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": config.active_mode,
        "seed": seed,
        "config_hash": config_fingerprint,
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
        },
        "files": paths,
        "time_range": {
            "min": str(events["timestamp"].min()),
            "max": str(events["timestamp"].max()),
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
        "output_format": {
            name: Path(path).suffix.removeprefix(".") for name, path in paths.items()
        },
    }
