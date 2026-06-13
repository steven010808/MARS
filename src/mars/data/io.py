from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

LIST_COLUMNS = {
    "style_tags",
    "preferred_categories",
    "positive_product_ids",
    "negative_product_ids",
}


def _jsonify_list_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in LIST_COLUMNS.intersection(out.columns):
        out[column] = out[column].map(lambda value: json.dumps(value, ensure_ascii=False))
    return out


def _parse_list_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in LIST_COLUMNS.intersection(out.columns):
        out[column] = out[column].map(
            lambda value: json.loads(value) if isinstance(value, str) and value else value
        )
    return out


def write_table(frame: pd.DataFrame, path_without_suffix: Path) -> Path:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = path_without_suffix.with_suffix(".parquet")
    try:
        frame.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = path_without_suffix.with_suffix(".csv")
        _jsonify_list_columns(frame).to_csv(csv_path, index=False, encoding="utf-8")
        return csv_path


def read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if table_path.suffix == ".parquet":
        return pd.read_parquet(table_path)
    if table_path.suffix == ".csv":
        return _parse_list_columns(pd.read_csv(table_path))
    if table_path.suffix == ".json":
        return pd.read_json(table_path, orient="records")
    raise ValueError(f"Unsupported table format: {table_path}")


def write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
