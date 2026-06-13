from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_ALLOWED_TOP_CATEGORIES = ["Ladieswear", "Baby/Children", "Divided", "Menswear", "Sport"]


def normalize_product_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(10)


def normalize_product_id_token(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(10)


def load_hnm_search_qrel_product_ids(
    qrels_path: str | Path,
    *,
    include_negatives: bool,
) -> tuple[set[str], dict[str, Any]]:
    qrels = pd.read_csv(
        qrels_path,
        dtype={"query_id": str, "positive_ids": str, "negative_ids": str},
    )

    def collect(column: str) -> set[str]:
        if column not in qrels.columns:
            return set()
        ids: set[str] = set()
        for value in qrels[column].dropna().astype(str):
            for token in value.replace(",", " ").split():
                token = token.strip()
                if token and token.lower() not in {"nan", "none", "null"}:
                    ids.add(normalize_product_id_token(token))
        return ids

    positive_ids = collect("positive_ids")
    negative_ids = collect("negative_ids")
    selected_ids = positive_ids | negative_ids if include_negatives else positive_ids
    return selected_ids, {
        "qrels_path": str(qrels_path),
        "rows": int(len(qrels)),
        "positive_unique_product_ids": int(len(positive_ids)),
        "negative_unique_product_ids": int(len(negative_ids)),
        "selected_unique_product_ids": int(len(selected_ids)),
        "include_negatives": bool(include_negatives),
    }


def non_empty(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column in columns:
        if column not in frame.columns:
            return pd.Series(False, index=frame.index)
        values = frame[column].fillna("").astype(str).str.strip()
        mask &= values.ne("") & ~values.str.lower().isin({"nan", "none", "null", "unknown"})
    return mask


def capped_sqrt_quotas(counts: dict[str, int], target: int) -> dict[str, int]:
    remaining = {key for key, value in counts.items() if value > 0}
    quotas = {key: 0 for key in counts}
    remaining_target = target

    while remaining:
        weight_sum = sum(math.sqrt(counts[key]) for key in remaining)
        raw = {key: remaining_target * math.sqrt(counts[key]) / weight_sum for key in remaining}
        capped = [key for key, value in raw.items() if value >= counts[key]]
        if capped:
            for key in capped:
                quotas[key] = counts[key]
                remaining_target -= counts[key]
                remaining.remove(key)
            continue

        for key, value in raw.items():
            quotas[key] = int(math.floor(value))
        while sum(quotas.values()) < target:
            candidates = [
                (raw[key] - math.floor(raw[key]), counts[key], key)
                for key in remaining
                if quotas[key] < counts[key]
            ]
            if not candidates:
                break
            _, _, key = max(candidates)
            quotas[key] += 1
        break

    if sum(quotas.values()) != target:
        raise RuntimeError(f"quota sum mismatch: {sum(quotas.values())} != {target}")
    return quotas


def minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return pd.Series(0.0, index=values.index)
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0)


def build_scores(frame: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    out = frame.copy()
    out["product_id"] = normalize_product_id(out["product_id"])
    out["purchase_count"] = pd.to_numeric(out.get("purchase_count", 0), errors="coerce").fillna(0)
    out["price"] = pd.to_numeric(out.get("price", 0), errors="coerce").fillna(0.0)
    low_cut = float(out["price"].quantile(0.55))
    mid_cut = float(out["price"].quantile(0.97))
    out["price_tier"] = np.select(
        [out["price"].le(low_cut), out["price"].le(mid_cut)],
        ["low_price", "mid_price"],
        default="luxury",
    )
    out["last_purchase_date"] = pd.to_datetime(out.get("last_purchase_date"), errors="coerce")
    out["image_path"] = (
        out.get("image_path", "").fillna("").astype(str).str.replace("\\", "/", regex=False)
    )
    out["image_abs_path"] = out["image_path"].map(
        lambda value: image_root / value if value else None
    )
    out["image_file_size"] = out["image_abs_path"].map(
        lambda path: path.stat().st_size if path is not None and path.exists() else 0
    )

    out["popularity_score"] = minmax(np.log1p(out["purchase_count"]))
    out["image_quality_score"] = minmax(np.log1p(out["image_file_size"].clip(lower=1)))

    metadata_parts = [
        non_empty(out, ["description"]).astype(float),
        non_empty(out, ["color"]).astype(float),
        non_empty(out, ["product_group_name"]).astype(float),
        non_empty(out, ["garment_group_name"]).astype(float),
        non_empty(out, ["department_name"]).astype(float),
    ]
    out["metadata_completeness_score"] = sum(metadata_parts) / len(metadata_parts)
    out["text_richness_score"] = minmax(out.get("description", "").fillna("").astype(str).str.len())

    if out["last_purchase_date"].notna().any():
        latest = out["last_purchase_date"].max()
        age_days = (latest - out["last_purchase_date"]).dt.days.fillna(9999).clip(lower=0)
        out["recency_score"] = 1.0 - minmax(age_days)
    else:
        out["recency_score"] = 0.0

    # Mildly prefer rows with stable price estimates and active purchase history.
    out["price_stability_score"] = 1.0 - minmax(
        (
            pd.to_numeric(out.get("price_max", out["price"]), errors="coerce").fillna(out["price"])
            - pd.to_numeric(out.get("price_min", out["price"]), errors="coerce").fillna(
                out["price"]
            )
        ).abs()
    )
    out["selection_score"] = (
        0.40 * out["popularity_score"]
        + 0.16 * out["metadata_completeness_score"]
        + 0.06 * out["text_richness_score"]
        + 0.18 * out["image_quality_score"]
        + 0.12 * out["recency_score"]
        + 0.08 * out["price_stability_score"]
    )
    return out


def select_group(group: pd.DataFrame, quota: int) -> pd.DataFrame:
    if len(group) <= quota:
        return group.copy()
    ranked = group.sort_values(
        ["selection_score", "purchase_count", "product_id"],
        ascending=[False, False, True],
    ).copy()
    ranked["_diversity_key"] = (
        ranked["mid_category"].fillna("").astype(str)
        + "\x1f"
        + ranked["leaf_category"].fillna("").astype(str)
        + "\x1f"
        + ranked["price_tier"].fillna("").astype(str)
    )
    ranked["_style_key"] = ranked.get("product_code", ranked["product_id"]).fillna("").astype(str)
    ranked["_diversity_round"] = ranked.groupby("_diversity_key", sort=False).cumcount()
    ranked["_style_round"] = ranked.groupby("_style_key", sort=False).cumcount()
    selected = ranked.sort_values(
        ["_diversity_round", "_style_round", "selection_score", "purchase_count", "product_id"],
        ascending=[True, True, False, False, True],
    ).head(quota)
    return selected.drop(
        columns=["_diversity_key", "_style_key", "_diversity_round", "_style_round"]
    )


def mark_new_items(frame: pd.DataFrame, threshold_days: int) -> pd.DataFrame:
    out = frame.copy()
    if "last_purchase_date" not in out.columns or not out["last_purchase_date"].notna().any():
        out["is_new"] = False
        return out
    latest = out["last_purchase_date"].max()
    age_days = (latest - out["last_purchase_date"]).dt.days.fillna(9999).clip(lower=0)
    out["is_new"] = age_days.le(int(threshold_days))
    return out


def selection_report(
    source: pd.DataFrame,
    valid: pd.DataFrame,
    selected: pd.DataFrame,
    quotas: dict[str, int],
    *,
    new_item_days_threshold: int,
    hnm_search_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hard_filters = [
        "allowed top category",
        "has_image == 1",
        "image_path is not blank",
        "image file exists",
        "product_id/name/top_category/mid_category/leaf_category are present",
        "price > 0",
    ]
    if hnm_search_filter:
        hard_filters.append("product_id appears in Microsoft H&M search qrels")

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "target_products": int(len(selected)),
        "criteria": {
            "hard_filters": hard_filters,
            "quota_strategy": "capped sqrt-balanced top_category quota",
            "within_category_strategy": (
                "round-robin by mid_category + leaf_category + price_tier, "
                "style-diversity by product_code, then selection_score"
            ),
            "selection_score": {
                "popularity_score": 0.40,
                "metadata_completeness_score": 0.16,
                "text_richness_score": 0.06,
                "image_quality_score_file_size_proxy": 0.18,
                "recency_score": 0.12,
                "price_stability_score": 0.08,
            },
            "random_seed": 42,
            "new_item_days_threshold": int(new_item_days_threshold),
        },
        "source_counts": {
            "rows": int(len(source)),
            "valid_after_hard_filters": int(len(valid)),
            "selected": int(len(selected)),
        },
        "hnm_search_filter": hnm_search_filter,
        "invalid_counts": {
            "has_image_false": int(
                (
                    pd.to_numeric(source.get("has_image", 0), errors="coerce").fillna(0).astype(int)
                    != 1
                ).sum()
            ),
            "blank_image_path": int(
                source.get("image_path", "").fillna("").astype(str).str.strip().eq("").sum()
            ),
        },
        "quotas": {key: int(value) for key, value in quotas.items()},
        "category_counts_before": {
            key: int(value)
            for key, value in source["top_category"].value_counts().sort_index().items()
        },
        "category_counts_valid": {
            key: int(value)
            for key, value in valid["top_category"].value_counts().sort_index().items()
        },
        "category_counts_selected": {
            key: int(value)
            for key, value in selected["top_category"].value_counts().sort_index().items()
        },
        "price_tier_counts_selected": {
            key: int(value)
            for key, value in selected["price_tier"].value_counts().sort_index().items()
        },
        "new_item_count_selected": int(selected["is_new"].sum())
        if "is_new" in selected.columns
        else 0,
        "mid_category_count_selected": int(selected["mid_category"].nunique()),
        "leaf_category_count_selected": int(selected["leaf_category"].nunique()),
        "score_summary_selected": {
            key: float(value)
            for key, value in selected[
                [
                    "selection_score",
                    "popularity_score",
                    "metadata_completeness_score",
                    "text_richness_score",
                    "image_quality_score",
                    "recency_score",
                    "price_stability_score",
                ]
            ]
            .mean()
            .round(6)
            .items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean balanced 50K H&M product master.")
    parser.add_argument("--input", default="data/external/hm/processed/hm_products_master.csv")
    parser.add_argument(
        "--output", default="data/external/hm/processed/hm_products_master_clean_50k.csv"
    )
    parser.add_argument(
        "--manifest",
        default="data/external/hm/processed/hm_products_master_clean_50k_manifest.json",
    )
    parser.add_argument("--image-root", default="data/external/hm/raw/images")
    parser.add_argument("--target", type=int, default=50_000)
    parser.add_argument("--new-item-days-threshold", type=int, default=7)
    parser.add_argument(
        "--allowed-top-categories", nargs="*", default=DEFAULT_ALLOWED_TOP_CATEGORIES
    )
    parser.add_argument("--hnm-search-qrels", default=None)
    parser.add_argument(
        "--hnm-search-include-negatives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --hnm-search-qrels is set, include negative qrels as catalog-coverage candidates."
        ),
    )
    args = parser.parse_args()

    source = pd.read_csv(args.input, dtype={"product_id": str, "product_code": str})
    scored = build_scores(source, Path(args.image_root))
    scored = mark_new_items(scored, args.new_item_days_threshold)
    hnm_search_product_ids: set[str] | None = None
    hnm_search_filter: dict[str, Any] | None = None
    if args.hnm_search_qrels:
        hnm_search_product_ids, hnm_search_filter = load_hnm_search_qrel_product_ids(
            args.hnm_search_qrels,
            include_negatives=args.hnm_search_include_negatives,
        )

    base_hard = (
        scored["top_category"].isin(args.allowed_top_categories)
        & pd.to_numeric(scored.get("has_image", 0), errors="coerce").fillna(0).astype(int).eq(1)
        & scored["image_path"].fillna("").astype(str).str.strip().ne("")
        & scored["image_abs_path"].map(lambda path: path is not None and path.exists())
        & non_empty(scored, ["product_id", "name", "top_category", "mid_category", "leaf_category"])
        & scored["price"].gt(0)
    )
    hard = base_hard
    if hnm_search_product_ids is not None:
        matched = scored["product_id"].isin(hnm_search_product_ids)
        hnm_search_filter = {
            **(hnm_search_filter or {}),
            "base_hard_filter_matched_rows": int((base_hard & matched).sum()),
            "base_hard_filter_positive_only_note": (
                "Positive qrels alone are below the 50K target after hard filters; "
                "positive+negative qrels are used for catalog coverage."
            ),
        }
        hard &= matched
    valid = scored[hard].copy()
    if len(valid) < args.target:
        raise RuntimeError(
            f"valid rows after hard filters ({len(valid)}) are below target ({args.target}); "
            "relax filters or include negative qrels."
        )
    counts = {key: int(value) for key, value in valid["top_category"].value_counts().items()}
    quotas = capped_sqrt_quotas(counts, args.target)

    selected_parts: list[pd.DataFrame] = []
    for category in args.allowed_top_categories:
        quota = int(quotas.get(category, 0))
        if quota <= 0:
            continue
        selected_parts.append(select_group(valid[valid["top_category"].eq(category)], quota))
    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values(
        ["top_category", "mid_category", "leaf_category", "selection_score", "product_id"],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)
    if len(selected) != args.target:
        raise RuntimeError(f"selected {len(selected)} rows, expected {args.target}")

    output_columns = [column for column in source.columns if column in selected.columns]
    if "is_new" not in output_columns:
        output_columns.append("is_new")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    selected[output_columns].to_csv(args.output, index=False)

    report = selection_report(
        source,
        valid,
        selected,
        quotas,
        new_item_days_threshold=args.new_item_days_threshold,
        hnm_search_filter=hnm_search_filter,
    )
    Path(args.manifest).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
