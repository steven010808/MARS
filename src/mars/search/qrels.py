from __future__ import annotations

import hashlib
from typing import Literal

import pandas as pd

from mars.config.settings import MarsConfig

QrelsSplit = Literal["train", "valid", "test", "all"]


def select_qrels_split(
    queries: pd.DataFrame,
    config: MarsConfig,
    split: QrelsSplit,
) -> pd.DataFrame:
    """Return a stable query-id hash split without changing source order."""

    if queries.empty or split == "all":
        return queries.copy()
    if "query_id" not in queries.columns:
        raise ValueError("Search qrels split requires a query_id column")

    seed, train_ratio, valid_ratio = qrels_split_settings(config)
    buckets = queries["query_id"].astype(str).map(lambda query_id: _split_unit(query_id, seed))
    train_end = train_ratio
    valid_end = train_ratio + valid_ratio

    if split == "train":
        mask = buckets < train_end
    elif split == "valid":
        mask = (buckets >= train_end) & (buckets < valid_end)
    elif split == "test":
        mask = buckets >= valid_end
    else:
        raise ValueError(f"Unknown qrels split: {split}")
    return queries.loc[mask].copy()


def qrels_split_settings(config: MarsConfig) -> tuple[int, float, float]:
    raw_search = config.raw.get("search", {}) if isinstance(config.raw, dict) else {}
    seed = int(raw_search.get("qrels_split_seed", config.seed))
    train_ratio = float(raw_search.get("qrels_train_ratio", 0.8))
    valid_ratio = float(raw_search.get("qrels_valid_ratio", 0.1))
    if train_ratio <= 0 or valid_ratio < 0 or train_ratio + valid_ratio >= 1:
        raise ValueError(
            "Search qrels split ratios must satisfy train > 0, valid >= 0, and train + valid < 1"
        )
    return seed, train_ratio, valid_ratio


def qrels_prior_train_only(config: MarsConfig) -> bool:
    raw_search = config.raw.get("search", {}) if isinstance(config.raw, dict) else {}
    return bool(raw_search.get("qrels_prior_train_only", True))


def evaluation_qrels_split(config: MarsConfig) -> QrelsSplit:
    raw_eval = config.raw.get("evaluation", {}) if isinstance(config.raw, dict) else {}
    split = str(raw_eval.get("search_qrels_split", "test")).lower()
    if split not in {"train", "valid", "test", "all"}:
        raise ValueError(f"Unknown evaluation search_qrels_split: {split}")
    return split  # type: ignore[return-value]


def _split_unit(query_id: str, seed: int) -> float:
    digest = hashlib.blake2b(f"{seed}:{query_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64)
