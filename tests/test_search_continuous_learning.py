from __future__ import annotations

import gzip
import json

import pandas as pd

from mars.config.settings import MarsConfig, PathsConfig
from mars.search.behavior_model import (
    build_query_behavior_model_payload,
    write_query_behavior_model,
)
from mars.search.feedback import build_search_feedback_frame


def test_search_feedback_links_exposure_to_engagement() -> None:
    events = [
        {
            "event_id": "E-search",
            "user_id": "U1",
            "session_id": "S1",
            "event_type": "search",
            "query": "red sneaker",
            "timestamp": "2026-06-06T00:00:00+00:00",
            "metadata": {
                "source_surface": "search",
                "event_role": "exposure",
                "search_id": "Q1",
                "result_product_ids": ["P1", "P2"],
            },
        },
        {
            "event_id": "E-view",
            "user_id": "U1",
            "session_id": "S1",
            "event_type": "view",
            "product_id": "P2",
            "timestamp": "2026-06-06T00:01:00+00:00",
            "metadata": {"source_surface": "search", "search_id": "Q1"},
        },
    ]

    frame = build_search_feedback_frame(
        events,
        catalog_products={"P1", "P2"},
        validation_ratio=0.0,
    )

    positive = frame[(frame["label"] == 1) & (frame["product_id"] == "P2")]
    negative = frame[(frame["label"] == 0) & (frame["product_id"] == "P1")]

    assert len(positive) == 1
    assert int(positive.iloc[0]["weight"]) == 1
    assert int(positive.iloc[0]["rank"]) == 2
    assert len(negative) == 1
    assert set(frame["split"]) == {"train"}


def test_behavior_model_merges_live_feedback_train_only(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    artifacts_dir = tmp_path / "artifacts"
    search_dir = artifacts_dir / "search"
    processed_dir.mkdir(parents=True)
    search_dir.mkdir(parents=True)
    products = pd.DataFrame({"product_id": ["P1", "P2"]})
    products.to_parquet(search_dir / "product_meta.parquet", index=False)
    qrels = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "query": "black jacket",
                "positive_product_ids": ["P1"],
            }
        ]
    )
    qrels.to_parquet(processed_dir / "search_queries.parquet", index=False)
    feedback = pd.DataFrame(
        [
            {
                "query": "new red sneaker",
                "product_id": "P2",
                "label": 1,
                "weight": 5,
                "split": "train",
            },
            {
                "query": "heldout blue shirt",
                "product_id": "P2",
                "label": 1,
                "weight": 5,
                "split": "valid",
            },
        ]
    )
    feedback_path = processed_dir / "search_feedback.parquet"
    feedback.to_parquet(feedback_path, index=False)
    config = MarsConfig(
        paths=PathsConfig(processed_dir=processed_dir, artifacts_dir=artifacts_dir),
        raw={
            "search": {
                "query_prior_top_k": 10,
                "query_token_prior_top_k": 10,
                "qrels_split_seed": 42,
                "qrels_train_ratio": 0.8,
                "qrels_valid_ratio": 0.1,
                "online_learning": {
                    "enabled": True,
                    "feedback_path": str(feedback_path),
                    "weight_multiplier": 1.0,
                    "min_positive_weight": 1.0,
                    "max_query_product_weight": 25,
                },
            }
        },
    )

    payload = build_query_behavior_model_payload(config, feedback_path=feedback_path)
    output = search_dir / "query_behavior_model.json.gz"
    write_query_behavior_model(payload, output)

    with gzip.open(output, "rt", encoding="utf-8") as handle:
        written = json.load(handle)

    assert written["query_prior"]["new red sneaker"] == ["P2"]
    assert "heldout blue shirt" not in written["query_prior"]
    assert written["live_feedback"]["train_positive_rows"] == 1
    assert written["live_feedback"]["valid_rows"] == 1
