from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_article_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(10) if text.isdigit() else text


def _safe_mode(series: pd.Series) -> int | None:
    if series.empty:
        return None
    values = series.mode(dropna=True)
    if values.empty:
        return None
    try:
        return int(values.iloc[0])
    except Exception:
        return None


def load_articles(path: str | Path) -> pd.DataFrame:
    articles = pd.read_csv(path, dtype={"article_id": str, "product_code": str})
    articles["article_id"] = articles["article_id"].map(normalize_article_id)
    articles["product_code"] = articles["product_code"].astype(str).str.strip()
    if "detail_desc" not in articles.columns:
        articles["detail_desc"] = ""
    articles["detail_desc"] = articles["detail_desc"].fillna("").astype(str)
    return articles


def load_transactions(path: str | Path) -> pd.DataFrame:
    transactions = pd.read_csv(
        path,
        dtype={"article_id": str, "customer_id": str},
        parse_dates=["t_dat"],
    )
    transactions["article_id"] = transactions["article_id"].map(normalize_article_id)
    transactions["price"] = pd.to_numeric(transactions["price"], errors="coerce")
    return transactions


def build_transaction_stats(transactions: pd.DataFrame) -> pd.DataFrame:
    grouped = transactions.groupby("article_id", as_index=False).agg(
        price=("price", "median"),
        price_mean=("price", "mean"),
        price_min=("price", "min"),
        price_max=("price", "max"),
        purchase_count=("price", "size"),
        last_purchase_date=("t_dat", "max"),
    )
    sales_channel_mode = (
        transactions.groupby("article_id")["sales_channel_id"]
        .apply(_safe_mode)
        .reset_index(name="sales_channel_mode")
    )
    stats = grouped.merge(sales_channel_mode, on="article_id", how="left")
    stats["price_source"] = "transaction_median"
    return stats


def build_fallback_prices(articles: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    merged = articles.merge(stats[["article_id", "price"]], on="article_id", how="left")
    type_price = (
        merged.groupby("product_type_name", dropna=False)["price"]
        .median()
        .reset_index(name="fallback_price_product_type")
    )
    garment_price = (
        merged.groupby("garment_group_name", dropna=False)["price"]
        .median()
        .reset_index(name="fallback_price_garment_group")
    )
    group_price = (
        merged.groupby("product_group_name", dropna=False)["price"]
        .median()
        .reset_index(name="fallback_price_product_group")
    )
    global_price = merged["price"].median()

    fallback = (
        articles.merge(type_price, on="product_type_name", how="left")
        .merge(garment_price, on="garment_group_name", how="left")
        .merge(group_price, on="product_group_name", how="left")
    )
    fallback["fallback_price"] = (
        fallback["fallback_price_product_type"]
        .fillna(fallback["fallback_price_garment_group"])
        .fillna(fallback["fallback_price_product_group"])
        .fillna(global_price)
    )
    fallback["fallback_source"] = fallback["fallback_price_product_type"].notna().map(
        {True: "product_type_median", False: None}
    )
    mask = fallback["fallback_source"].isna() & fallback["fallback_price_garment_group"].notna()
    fallback.loc[mask, "fallback_source"] = "garment_group_median"
    mask = fallback["fallback_source"].isna() & fallback["fallback_price_product_group"].notna()
    fallback.loc[mask, "fallback_source"] = "product_group_median"
    fallback["fallback_source"] = fallback["fallback_source"].fillna("global_median")
    return fallback[["article_id", "fallback_price", "fallback_source"]]


def build_image_columns(article_ids: pd.Series, images_root: str | Path | None) -> pd.DataFrame:
    root = Path(images_root) if images_root else None
    rows: list[dict[str, Any]] = []
    for article_id in article_ids.astype(str):
        relative = Path(article_id[:3]) / f"{article_id}.jpg" if article_id else Path("")
        if not article_id:
            rows.append({"image_path": "", "has_image": 0})
            continue
        exists = int((root / relative).exists()) if root else 0
        rows.append({"image_path": str(relative).replace("\\", "/"), "has_image": exists})
    return pd.DataFrame(rows, index=article_ids.index)


def build_hm_products_master(
    *,
    articles_path: str | Path,
    transactions_path: str | Path,
    images_root: str | Path | None,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    articles = load_articles(articles_path)
    transactions = load_transactions(transactions_path)
    stats = build_transaction_stats(transactions)
    fallback = build_fallback_prices(articles, stats)

    merged = articles.merge(stats, on="article_id", how="left").merge(
        fallback, on="article_id", how="left"
    )
    missing_price = merged["price"].isna()
    merged.loc[missing_price, "price"] = merged.loc[missing_price, "fallback_price"]
    merged.loc[missing_price, "price_source"] = merged.loc[missing_price, "fallback_source"]
    merged["price_mean"] = merged["price_mean"].fillna(merged["price"])
    merged["price_min"] = merged["price_min"].fillna(merged["price"])
    merged["price_max"] = merged["price_max"].fillna(merged["price"])
    merged["purchase_count"] = merged["purchase_count"].fillna(0).astype(int)

    merged["top_category"] = merged["index_group_name"].fillna("").astype(str)
    merged["mid_category"] = merged["index_name"].fillna("").astype(str)
    merged["leaf_category"] = merged["product_type_name"].fillna("").astype(str)
    merged["color"] = merged["colour_group_name"].fillna("").astype(str)
    merged["name"] = merged["prod_name"].fillna("").astype(str)
    merged["description"] = merged["detail_desc"].fillna("").astype(str)

    image_columns = build_image_columns(merged["article_id"], images_root)
    merged["image_path"] = image_columns["image_path"]
    merged["has_image"] = image_columns["has_image"].astype(int)
    merged["source"] = "hm"

    output_columns = [
        "article_id",
        "product_code",
        "name",
        "description",
        "top_category",
        "mid_category",
        "leaf_category",
        "product_group_name",
        "garment_group_name",
        "department_name",
        "color",
        "price",
        "price_source",
        "price_mean",
        "price_min",
        "price_max",
        "purchase_count",
        "last_purchase_date",
        "sales_channel_mode",
        "image_path",
        "has_image",
        "source",
    ]
    final = merged[output_columns].rename(columns={"article_id": "product_id"}).copy()
    final = final.sort_values(
        ["purchase_count", "product_id"], ascending=[False, True]
    ).reset_index(drop=True)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output, index=False, encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "articles_path": str(articles_path),
        "transactions_path": str(transactions_path),
        "images_root": str(images_root) if images_root else None,
        "output_path": str(output),
        "rows": int(len(final)),
        "has_image": int(final["has_image"].sum()),
        "unique_products": int(final["product_id"].nunique()),
        "source": "hm_kaggle_raw",
    }
    if manifest_path:
        Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Build H&M product master from Kaggle raw CSVs.")
    parser.add_argument("--articles", default="data/external/hm/raw/articles.csv")
    parser.add_argument("--transactions", default="data/external/hm/raw/transactions_train.csv")
    parser.add_argument("--images-root", default="data/external/hm/raw/images")
    parser.add_argument("--output", default="data/external/hm/processed/hm_products_master.csv")
    parser.add_argument("--manifest", default="data/external/hm/processed/hm_products_master_manifest.json")
    args = parser.parse_args()
    build_hm_products_master(
        articles_path=args.articles,
        transactions_path=args.transactions,
        images_root=args.images_root,
        output_path=args.output,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()
