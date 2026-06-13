from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import pandas as pd

from mars.data.raw_simulator.catalog import (
    build_top_category_reference_prices,
    generate_products,
    get_top_category_price_tier,
)
from mars.data.raw_simulator.config import load_config
from mars.data.raw_simulator.personas import load_persona_profiles, sample_persona_name


def normalize_distribution(distribution: dict[str, float]) -> dict[str, float]:
    total = sum(distribution.values())
    if total <= 0:
        raise ValueError("Distribution total must be positive.")
    return {key: value / total for key, value in distribution.items()}


def build_price_based_distribution(
    top_category_reference_prices: dict[str, float],
    affordability_alpha: float,
) -> dict[str, float]:
    raw_scores = {
        category: 1.0 / (max(reference_price, 1e-6) ** affordability_alpha)
        for category, reference_price in top_category_reference_prices.items()
    }
    return normalize_distribution(raw_scores)


def build_prior_distribution(
    category_prior: dict[str, float],
    top_categories: list[str],
) -> dict[str, float]:
    if not category_prior:
        uniform = 1.0 / len(top_categories)
        return {category: uniform for category in top_categories}

    filled = {category: float(category_prior.get(category, 0.0)) for category in top_categories}
    return normalize_distribution(filled)


def build_top_category_anchor_distribution(
    preferred_top_category: str,
    top_categories: list[str],
    anchor_weight: float = 4.0,
) -> dict[str, float]:
    raw_scores = {category: 1.0 for category in top_categories}
    if preferred_top_category in raw_scores:
        raw_scores[preferred_top_category] = anchor_weight
    return normalize_distribution(raw_scores)


def apply_noise_to_distribution(
    distribution: dict[str, float],
    rng: random.Random,
    noise_cfg: dict[str, Any],
) -> dict[str, float]:
    mode = str(noise_cfg.get("mode", "blend_uniform"))
    strength = float(noise_cfg.get("strength", 0.0))

    if strength <= 0:
        return distribution

    categories = list(distribution.keys())

    if mode == "blend_uniform":
        random_vector = {category: rng.random() + 1e-6 for category in categories}
        random_distribution = normalize_distribution(random_vector)

        blended = {
            category: (1.0 - strength) * distribution[category]
            + strength * random_distribution[category]
            for category in categories
        }
        return normalize_distribution(blended)

    raise ValueError(f"Unsupported noise mode: {mode}")


def get_top_categories(
    distribution: dict[str, float],
    top_k: int,
) -> list[str]:
    ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    return [category for category, _ in ranked[:top_k]]


def adjust_budget_for_price_tier(
    budget_min: int,
    budget_max: int,
    price_tier: str,
    budget_adjustment_cfg: dict[str, Any],
) -> tuple[int, int]:
    cfg = budget_adjustment_cfg[price_tier]

    adjusted_min = int(
        max(
            budget_min * float(cfg["min_multiplier"]),
            int(cfg["min_floor"]),
        )
    )
    adjusted_max = int(
        max(
            budget_max * float(cfg["max_multiplier"]),
            int(cfg["max_floor"]),
        )
    )

    if adjusted_min > adjusted_max:
        adjusted_min, adjusted_max = adjusted_max, adjusted_min

    return adjusted_min, adjusted_max


def sample_preferred_top_category(
    preferred_top_category_pool: list[str],
    rng: random.Random,
) -> str:
    if not preferred_top_category_pool:
        raise ValueError("preferred_top_category_pool must not be empty.")
    return rng.choice(preferred_top_category_pool)


def _normalize_article_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(10)


def _get_raw_data_dir(config: dict[str, Any]) -> Path:
    simulator_output_cfg = config["simulator"].get("output", {})
    raw_dir = simulator_output_cfg.get("raw_dir")

    if raw_dir:
        return Path(raw_dir)

    return Path(config["paths"]["data_dir"]) / "raw"


def _get_output_format(config: dict[str, Any]) -> str:
    return str(config["simulator"].get("output_format", "csv"))


def _build_default_filename(stem: str, output_format: str) -> str:
    return f"{stem}.{output_format}"


def _resolve_simulator_file_path(
    config: dict[str, Any],
    file_key: str,
    default_stem: str,
) -> Path:
    raw_dir = _get_raw_data_dir(config)
    simulator_output_cfg = config["simulator"].get("output", {})
    output_format = _get_output_format(config)

    configured_name = simulator_output_cfg.get(file_key)
    if configured_name:
        return raw_dir / configured_name

    return raw_dir / _build_default_filename(default_stem, output_format)


def generate_users(config: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    simulator_cfg = config["simulator"]
    profiles = load_persona_profiles(config)
    user_count = int(simulator_cfg["scale"]["users"])

    catalog_cfg = simulator_cfg["catalog"]
    top_category_reference_prices = build_top_category_reference_prices(config)

    allowed_top_categories = list(
        catalog_cfg.get("external_hm", {}).get("allowed_top_categories", [])
    )

    if allowed_top_categories:
        top_categories = [
            category
            for category in allowed_top_categories
            if category in top_category_reference_prices
        ]
    else:
        top_categories = list(top_category_reference_prices.keys())

    if not top_categories:
        raise ValueError("No top categories available for user generation.")

    noise_cfg = simulator_cfg["preference_noise"]
    budget_adjustment_cfg = simulator_cfg["budget_adjustment_by_tier"]
    top_k_categories = int(simulator_cfg["user_preference"]["top_k_categories"])

    users: list[dict[str, Any]] = []

    for idx in range(user_count):
        persona_name = sample_persona_name(profiles, rng)
        profile = profiles[persona_name]

        preferred_top_category = ""
        preferred_top_category_tier = ""
        budget_min = profile.budget_min
        budget_max = profile.budget_max

        preferred_price_tiers = list(getattr(profile, "preferred_price_tiers", []))
        preferred_top_category_pool = list(getattr(profile, "preferred_top_category_pool", []))

        if profile.distribution_mode == "price_based":
            base_distribution = build_price_based_distribution(
                top_category_reference_prices={
                    category: top_category_reference_prices[category] for category in top_categories
                },
                affordability_alpha=float(profile.affordability_alpha or 1.0),
            )

        elif profile.distribution_mode == "prior_based":
            base_distribution = build_prior_distribution(
                category_prior=profile.category_prior,
                top_categories=top_categories,
            )

        elif profile.distribution_mode == "top_category_anchor":
            pool = preferred_top_category_pool or top_categories
            preferred_top_category = sample_preferred_top_category(pool, rng)
            preferred_top_category_tier = get_top_category_price_tier(
                config,
                preferred_top_category,
            )

            budget_min, budget_max = adjust_budget_for_price_tier(
                budget_min=budget_min,
                budget_max=budget_max,
                price_tier=preferred_top_category_tier,
                budget_adjustment_cfg=budget_adjustment_cfg,
            )

            base_distribution = build_top_category_anchor_distribution(
                preferred_top_category=preferred_top_category,
                top_categories=top_categories,
            )

        else:
            raise ValueError(
                f"Unsupported distribution_mode for user generation: {profile.distribution_mode}"
            )

        final_distribution = apply_noise_to_distribution(
            distribution=base_distribution,
            rng=rng,
            noise_cfg=noise_cfg,
        )

        top_preferred_categories = get_top_categories(
            distribution=final_distribution,
            top_k=top_k_categories,
        )

        record: dict[str, Any] = {
            "user_id": f"U{idx + 1:06d}",
            "persona": persona_name,
            "price_sensitivity": profile.price_sensitivity,
            "base_conversion_rate": profile.base_conversion_rate,
            "preferred_top_categories": "|".join(top_preferred_categories),
            "preferred_top_category": preferred_top_category,
            "preferred_top_category_tier": preferred_top_category_tier,
            "preferred_price_tiers": "|".join(preferred_price_tiers),
            "budget_min": budget_min,
            "budget_max": budget_max,
            "signup_days_ago": rng.randint(0, 365),
        }

        for category, value in final_distribution.items():
            record[f"category_pref_{category}"] = round(value, 4)

        users.append(record)

    return users


def save_dataframe(df: pd.DataFrame, output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "csv":
        final_path = (
            output_path if output_path.suffix == ".csv" else output_path.with_suffix(".csv")
        )
        df.to_csv(final_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def print_day2_summary(products_df: pd.DataFrame, users_df: pd.DataFrame) -> None:
    print("\n[Products by price_tier]")
    print(products_df["price_tier"].value_counts())

    print("\n[Average price by price_tier]")
    print(products_df.groupby("price_tier")["price"].mean().round(6))

    print("\n[Products by top_category]")
    print(products_df["top_category"].value_counts())

    print("\n[Users by persona]")
    print(users_df["persona"].value_counts())

    print("\n[Average user category preference by persona]")
    preference_columns = [
        column for column in users_df.columns if column.startswith("category_pref_")
    ]
    grouped = users_df.groupby("persona")[preference_columns].mean().round(3)
    print(grouped)

    print("\n[Top-category loyalist preferred top_category]")
    loyalists = users_df[users_df["persona"] == "top_category_loyalist"]
    if not loyalists.empty:
        print(loyalists["preferred_top_category"].value_counts())


def print_day3_summary(events_df: pd.DataFrame) -> None:
    print("\n[Events by type]")
    print(events_df["event_type"].value_counts())

    print("\n[Events by persona]")
    persona_event_counts = events_df.groupby(["persona", "event_type"]).size().unstack(fill_value=0)
    print(persona_event_counts)

    total_sessions = events_df["session_id"].nunique()
    total_users = events_df["user_id"].nunique()

    print("\n[Entity summary]")
    print(f"total_events: {len(events_df)}")
    print(f"total_sessions: {total_sessions}")
    print(f"total_users: {total_users}")

    print("\n[Session length summary]")
    session_lengths = events_df.groupby("session_id").size()
    print(session_lengths.describe().round(2))

    view_counts = events_df[events_df["event_type"] == "view"].groupby("session_id").size()
    if not view_counts.empty:
        print("\n[Views per session summary]")
        print(view_counts.describe().round(2))

    search_sessions = set(events_df.loc[events_df["event_type"] == "search", "session_id"])
    cart_sessions = set(events_df.loc[events_df["event_type"] == "cart", "session_id"])
    purchase_sessions = set(events_df.loc[events_df["event_type"] == "purchase", "session_id"])

    total_search_sessions = max(len(search_sessions), 1)

    print("\n[Session conversion summary]")
    print(f"cart_session_rate: {len(cart_sessions) / total_search_sessions:.3f}")
    print(f"purchase_session_rate: {len(purchase_sessions) / total_search_sessions:.3f}")

    purchase_df = events_df[events_df["event_type"] == "purchase"]
    if not purchase_df.empty:
        print("\n[Purchase by top_category]")
        print(purchase_df["top_category"].value_counts())

        print("\n[Purchase by persona]")
        print(purchase_df["persona"].value_counts())

        print("\n[Purchase by price_tier]")
        print(purchase_df["price_tier"].value_counts())


def run_day2_sample_generation() -> dict[str, Any]:
    config = load_config()
    rng = random.Random(int(config["app"]["random_seed"]))

    output_format = _get_output_format(config)

    products = generate_products(config, rng)
    users = generate_users(config, rng)

    products_df = pd.DataFrame(products)
    users_df = pd.DataFrame(users)

    if "product_id" in products_df.columns:
        products_df["product_id"] = _normalize_article_id(products_df["product_id"])

    products_path = _resolve_simulator_file_path(config, "products_file", "products")
    users_path = _resolve_simulator_file_path(config, "users_file", "users")

    save_dataframe(products_df, products_path, output_format)
    save_dataframe(users_df, users_path, output_format)

    print_day2_summary(products_df, users_df)

    return {
        "products": len(products_df),
        "users": len(users_df),
        "output_format": output_format,
        "products_path": str(products_path),
        "users_path": str(users_path),
        "data_dir": str(_get_raw_data_dir(config)),
    }


def run_day3_event_generation() -> dict[str, Any]:
    config = load_config()
    rng = random.Random(int(config["app"]["random_seed"]))
    output_format = _get_output_format(config)

    products_path = _resolve_simulator_file_path(config, "products_file", "products")
    users_path = _resolve_simulator_file_path(config, "users_file", "users")
    events_path = _resolve_simulator_file_path(config, "events_file", "events")

    if not products_path.exists():
        raise FileNotFoundError(f"products file not found: {products_path}")
    if not users_path.exists():
        raise FileNotFoundError(f"users file not found: {users_path}")

    products_df = pd.read_csv(
        products_path,
        dtype={
            "product_id": str,
            "product_code": str,
            "price_tier": str,
            "top_category": str,
            "mid_category": str,
            "leaf_category": str,
        },
    )
    if "product_id" in products_df.columns:
        products_df["product_id"] = _normalize_article_id(products_df["product_id"])

    users_df = pd.read_csv(
        users_path,
        dtype={
            "user_id": str,
            "persona": str,
            "preferred_top_categories": str,
            "preferred_top_category": str,
            "preferred_top_category_tier": str,
            "preferred_price_tiers": str,
        },
    )

    from mars.data.raw_simulator.events import generate_events, validate_events

    events_df = generate_events(
        users_df=users_df,
        products_df=products_df,
        config=config,
        rng=rng,
    )

    validation_issues = validate_events(events_df)
    if validation_issues:
        print("\n[Event validation issues]")
        for issue in validation_issues[:20]:
            print("-", issue)
        raise ValueError(f"Event validation failed with {len(validation_issues)} issue(s).")

    save_dataframe(events_df, events_path, output_format)
    print_day3_summary(events_df)

    return {
        "events": len(events_df),
        "output_format": output_format,
        "events_path": str(events_path),
        "products_path": str(products_path),
        "users_path": str(users_path),
        "validation_passed": True,
    }


if __name__ == "__main__":
    start = time.perf_counter()
    result = run_day3_event_generation()
    print("[Simulator Day3/Day4 Checked]", result)
    print(f"[Elapsed] {time.perf_counter() - start:.2f} sec")
