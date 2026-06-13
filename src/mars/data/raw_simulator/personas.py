from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


def normalize_distribution(distribution: dict[str, float]) -> dict[str, float]:
    total = sum(distribution.values())
    if total <= 0:
        raise ValueError("Distribution total must be positive.")
    return {key: value / total for key, value in distribution.items()}


@dataclass(frozen=True)
class PersonaProfile:
    name: str
    ratio: float
    distribution_mode: str
    affordability_alpha: float | None
    category_prior: dict[str, float]
    price_sensitivity: float
    base_conversion_rate: float
    preferred_price_tiers: list[str]
    preferred_top_category_pool: list[str]
    budget_min: int
    budget_max: int


def load_persona_profiles(config: dict[str, Any]) -> dict[str, PersonaProfile]:
    persona_cfg = config["simulator"]["personas"]
    profiles: dict[str, PersonaProfile] = {}

    for name, values in persona_cfg.items():
        raw_prior = values.get("category_prior", {})
        normalized_prior = (
            normalize_distribution({k: float(v) for k, v in raw_prior.items()}) if raw_prior else {}
        )

        profile = PersonaProfile(
            name=name,
            ratio=float(values["ratio"]),
            distribution_mode=str(values["distribution_mode"]),
            affordability_alpha=(
                float(values["affordability_alpha"]) if "affordability_alpha" in values else None
            ),
            category_prior=normalized_prior,
            price_sensitivity=float(values["price_sensitivity"]),
            base_conversion_rate=float(values["base_conversion_rate"]),
            preferred_price_tiers=list(values.get("preferred_price_tiers", [])),
            preferred_top_category_pool=list(values.get("preferred_top_category_pool", [])),
            budget_min=int(values["budget_min"]),
            budget_max=int(values["budget_max"]),
        )

        if profile.distribution_mode not in {
            "price_based",
            "prior_based",
            "top_category_anchor",
        }:
            raise ValueError(
                f"Unsupported distribution_mode for persona '{name}': {profile.distribution_mode}"
            )

        profiles[name] = profile

    total_ratio = sum(profile.ratio for profile in profiles.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Persona ratios must sum to 1.0, got {total_ratio:.6f}")

    return profiles


def sample_persona_name(
    profiles: dict[str, PersonaProfile],
    rng: random.Random,
) -> str:
    names = list(profiles.keys())
    weights = [profiles[name].ratio for name in names]
    return rng.choices(names, weights=weights, k=1)[0]
