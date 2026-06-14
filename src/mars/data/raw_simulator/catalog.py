from __future__ import annotations

from typing import Any

import pandas as pd

_HM_CACHE: dict[str, pd.DataFrame] = {}

DEFAULT_TOP_CATEGORIES = [
    "Ladieswear",
    "Baby/Children",
    "Divided",
    "Menswear",
    "Sport",
]


def _normalize_article_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(10)


def _prepare_hm_catalog_df(config: dict[str, Any]) -> pd.DataFrame:
    hm_cfg = config["simulator"]["catalog"]["external_hm"]
    master_path = str(hm_cfg["products_master_path"])

    if master_path in _HM_CACHE:
        return _HM_CACHE[master_path].copy()

    df = pd.read_csv(
        master_path,
        dtype={"product_id": str, "product_code": str},
        parse_dates=["last_purchase_date"],
    )

    df["product_id"] = _normalize_article_id(df["product_id"])
    df["product_code"] = (
        df["product_code"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )

    for col in ["name", "description", "color", "top_category", "mid_category", "leaf_category"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["top_category"] = df["top_category"].replace("", "Unknown")
    df["mid_category"] = df["mid_category"].replace("", "Unknown")
    df["leaf_category"] = df["leaf_category"].replace("", "Unknown")

    allowed_top_categories = list(hm_cfg.get("allowed_top_categories", DEFAULT_TOP_CATEGORIES))
    if allowed_top_categories:
        df = df[df["top_category"].isin(allowed_top_categories)].copy()

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    global_price = float(df["price"].median())
    df["price"] = df["price"].fillna(global_price)

    low_q = float(hm_cfg.get("low_price_max_quantile", 0.55))
    mid_q = float(hm_cfg.get("mid_price_max_quantile", 0.97))
    low_cut = float(df["price"].quantile(low_q))
    mid_cut = float(df["price"].quantile(mid_q))

    def _price_to_tier(price: float) -> str:
        if price <= low_cut:
            return "low_price"
        if price <= mid_cut:
            return "mid_price"
        return "luxury"

    df["price_tier"] = df["price"].map(_price_to_tier)

    df["purchase_count"] = (
        pd.to_numeric(df["purchase_count"], errors="coerce").fillna(0).astype(int)
    )
    max_purchase = max(int(df["purchase_count"].max()), 1)
    df["popularity_seed"] = (df["purchase_count"] / max_purchase).clip(lower=0.0, upper=1.0)

    new_days = int(hm_cfg.get("new_item_days_threshold", 7))
    max_date = df["last_purchase_date"].max()
    if pd.isna(max_date):
        df["is_new"] = False
    else:
        df["is_new"] = (max_date - df["last_purchase_date"]).dt.days <= new_days
        df["is_new"] = df["is_new"].fillna(False)

    for optional_col, default_value in {
        "product_group_name": "",
        "garment_group_name": "",
        "department_name": "",
        "image_path": "",
        "has_image": 0,
        "source": "hm",
    }.items():
        if optional_col not in df.columns:
            df[optional_col] = default_value

    _HM_CACHE[master_path] = df.copy()
    return df


def generate_products(config: dict[str, Any], rng: object | None = None) -> list[dict[str, Any]]:
    df = _prepare_hm_catalog_df(config)
    df = df.sort_values(["purchase_count", "product_id"], ascending=[False, True]).reset_index(
        drop=True
    )
    product_limit = int(config.get("simulator", {}).get("scale", {}).get("products", len(df)))
    df = df.head(max(1, min(product_limit, len(df)))).copy()

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        style_tags = "|".join(
            [
                tag
                for tag in [
                    row.get("product_group_name", ""),
                    row.get("garment_group_name", ""),
                    row.get("department_name", ""),
                ]
                if str(tag).strip()
            ]
        )

        records.append(
            {
                "product_id": str(row["product_id"]).zfill(10),
                "product_code": str(row["product_code"]),
                "price_tier": row["price_tier"],
                "top_category": row["top_category"],
                "mid_category": row["mid_category"],
                "leaf_category": row["leaf_category"],
                "price": float(row["price"]),
                "discount_rate": 0.0,
                "color": row["color"],
                "style_tags": style_tags,
                "popularity_seed": round(float(row["popularity_seed"]), 4),
                "is_new": bool(row["is_new"]),
                "name": row["name"],
                "description": row["description"],
                "image_path": row["image_path"],
                "has_image": int(row["has_image"]),
                "last_purchase_date": (
                    row["last_purchase_date"].date().isoformat()
                    if pd.notna(row["last_purchase_date"])
                    else ""
                ),
                "source": row["source"],
            }
        )

    return records


def build_top_category_reference_prices(config: dict[str, Any]) -> dict[str, float]:
    df = _prepare_hm_catalog_df(config)
    grouped = df.groupby("top_category")["price"].median().to_dict()
    return {str(k): float(v) for k, v in grouped.items()}


def get_top_category_price_tier(config: dict[str, Any], top_category: str) -> str:
    df = _prepare_hm_catalog_df(config)
    matched = df[df["top_category"] == top_category]
    if matched.empty:
        return "mid_price"

    hm_cfg = config["simulator"]["catalog"]["external_hm"]
    low_q = float(hm_cfg.get("low_price_max_quantile", 0.55))
    mid_q = float(hm_cfg.get("mid_price_max_quantile", 0.97))

    low_cut = float(df["price"].quantile(low_q))
    mid_cut = float(df["price"].quantile(mid_q))
    category_median_price = float(matched["price"].median())

    if category_median_price <= low_cut:
        return "low_price"
    if category_median_price <= mid_cut:
        return "mid_price"
    return "luxury"
