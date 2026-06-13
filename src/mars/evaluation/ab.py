from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import erf, sqrt
from typing import Any

import pandas as pd

from mars.evaluation.metrics import ctr, cvr


@dataclass(frozen=True)
class ABAssignment:
    experiment_key: str
    user_id: str
    bucket: str


@dataclass(frozen=True)
class ABReport:
    experiment_key: str
    buckets: dict[str, dict[str, float | int]]
    uplift: dict[str, float]
    p_value: float
    confidence_interval_95: tuple[float, float]
    significant: bool
    method: str = "two_proportion_z_test"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confidence_interval_95"] = list(self.confidence_interval_95)
        return payload


def assign_bucket(
    user_id: str,
    experiment_key: str = "default",
    buckets: tuple[str, ...] = ("control", "treatment"),
    weights: tuple[float, ...] | None = None,
) -> ABAssignment:
    if not buckets:
        raise ValueError("At least one bucket is required.")
    if weights is not None and len(weights) != len(buckets):
        raise ValueError("weights must match buckets length.")

    digest = sha256(f"{experiment_key}:{user_id}".encode()).hexdigest()
    value = int(digest[:12], 16) / float(0xFFFFFFFFFFFF)

    if weights is None:
        index = min(int(value * len(buckets)), len(buckets) - 1)
        bucket = buckets[index]
    else:
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must sum to a positive value.")
        running = 0.0
        bucket = buckets[-1]
        for candidate, weight in zip(buckets, weights, strict=True):
            running += weight / total
            if value <= running:
                bucket = candidate
                break
    return ABAssignment(experiment_key=experiment_key, user_id=user_id, bucket=bucket)


def two_proportion_z_test(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
) -> float:
    if trials_a <= 0 or trials_b <= 0:
        return 1.0
    rate_a = successes_a / trials_a
    rate_b = successes_b / trials_b
    pooled = (successes_a + successes_b) / (trials_a + trials_b)
    se = sqrt(max(pooled * (1.0 - pooled) * (1.0 / trials_a + 1.0 / trials_b), 0.0))
    if se == 0:
        return 1.0
    z_value = (rate_b - rate_a) / se
    return float(2.0 * (1.0 - _normal_cdf(abs(z_value))))


def confidence_interval_for_difference(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
    z_value: float = 1.96,
) -> tuple[float, float]:
    if trials_a <= 0 or trials_b <= 0:
        return (0.0, 0.0)
    rate_a = successes_a / trials_a
    rate_b = successes_b / trials_b
    diff = rate_b - rate_a
    se = sqrt(
        max(
            rate_a * (1.0 - rate_a) / trials_a + rate_b * (1.0 - rate_b) / trials_b,
            0.0,
        )
    )
    return (float(diff - z_value * se), float(diff + z_value * se))


def build_ab_report(
    events: pd.DataFrame,
    experiment_key: str = "default",
    bucket_column: str = "ab_group",
    control_bucket: str = "control",
    treatment_bucket: str = "treatment",
) -> ABReport:
    if events.empty or bucket_column not in events.columns:
        return _empty_report(experiment_key, control_bucket, treatment_bucket)

    bucket_stats = {
        bucket: _bucket_stats(events[events[bucket_column].astype(str) == bucket])
        for bucket in sorted(str(bucket) for bucket in events[bucket_column].dropna().unique())
    }
    bucket_stats.setdefault(control_bucket, _bucket_stats(pd.DataFrame()))
    bucket_stats.setdefault(treatment_bucket, _bucket_stats(pd.DataFrame()))

    control = bucket_stats[control_bucket]
    treatment = bucket_stats[treatment_bucket]
    success_key = (
        "conversions"
        if int(control["conversions"]) + int(treatment["conversions"]) > 0
        else "clicks"
    )
    p_value = two_proportion_z_test(
        int(control[success_key]),
        int(control["impressions"]),
        int(treatment[success_key]),
        int(treatment["impressions"]),
    )
    interval = confidence_interval_for_difference(
        int(control[success_key]),
        int(control["impressions"]),
        int(treatment[success_key]),
        int(treatment["impressions"]),
    )
    return ABReport(
        experiment_key=experiment_key,
        buckets=bucket_stats,
        uplift={
            "ctr": float(treatment["ctr"] - control["ctr"]),
            "cvr": float(treatment["cvr"] - control["cvr"]),
        },
        p_value=p_value,
        confidence_interval_95=interval,
        significant=p_value < 0.05,
    )


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _empty_report(experiment_key: str, control_bucket: str, treatment_bucket: str) -> ABReport:
    return ABReport(
        experiment_key=experiment_key,
        buckets={
            control_bucket: _bucket_stats(pd.DataFrame()),
            treatment_bucket: _bucket_stats(pd.DataFrame()),
        },
        uplift={"ctr": 0.0, "cvr": 0.0},
        p_value=1.0,
        confidence_interval_95=(0.0, 0.0),
        significant=False,
    )


def _bucket_stats(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty or "event_type" not in frame.columns:
        return {
            "impressions": 0,
            "clicks": 0,
            "conversions": 0,
            "ctr": 0.0,
            "cvr": 0.0,
        }

    event_type = frame["event_type"].astype(str)
    impressions = int(
        event_type.isin(["impression", "recommend", "search", "view", "cart", "purchase"]).sum()
    )
    clicks = int(event_type.isin(["view", "cart", "purchase"]).sum())
    conversions = int(event_type.eq("purchase").sum())
    return {
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions,
        "ctr": ctr(clicks, impressions),
        "cvr": ctr(conversions, impressions),
        "purchase_per_click": cvr(conversions, clicks),
    }
