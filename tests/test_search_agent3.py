from __future__ import annotations

import base64
import gzip
import io
import json
from dataclasses import replace

import numpy as np
import pandas as pd
from PIL import Image

from mars.config.settings import MarsConfig, PathsConfig, SearchConfig
from mars.retrieval.vector_index import VectorIndex
from mars.search.artifacts import build_search_artifacts
from mars.search.encoders import FallbackEncoder
from mars.search.qrels import select_qrels_split
from mars.search.service import SearchRequest, SearchService, _dominant_image_color_hints


def _write_products(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    paths = []
    for name, color in [
        ("jacket", (10, 10, 10)),
        ("sneaker", (240, 20, 20)),
        ("shirt", (20, 90, 240)),
    ]:
        path = image_dir / f"{name}.png"
        Image.new("RGB", (32, 32), color=color).save(path)
        paths.append(str(path))

    products = pd.DataFrame(
        [
            {
                "product_id": "P00000001",
                "name": "Black Leather Jacket",
                "category_l1": "outer",
                "category_l2": "jacket",
                "category_l3": "leather",
                "price": 129000,
                "color": "black",
                "style_tags": ["minimal", "street"],
                "description": "black leather outer jacket",
                "image_path": paths[0],
            },
            {
                "product_id": "P00000002",
                "name": "Red Running Sneaker",
                "category_l1": "shoes",
                "category_l2": "sneaker",
                "category_l3": "running",
                "price": 89000,
                "color": "red",
                "style_tags": ["sport"],
                "description": "red running shoes",
                "image_path": paths[1],
            },
            {
                "product_id": "P00000003",
                "name": "Blue Oxford Shirt",
                "category_l1": "top",
                "category_l2": "shirt",
                "category_l3": "oxford",
                "price": 59000,
                "color": "blue",
                "style_tags": ["classic"],
                "description": "blue classic shirt",
                "image_path": paths[2],
            },
        ]
    )
    products_path = tmp_path / "products.parquet"
    products.to_parquet(products_path)
    return products_path


def _write_hybrid_anchor_products(tmp_path):
    image_dir = tmp_path / "hybrid_images"
    image_dir.mkdir()
    rows = []
    specs = [
        ("P00001001", "Black Tapered Glove", "Black", "Gloves", (10, 10, 10)),
        ("P00001002", "Paris Glove", "Dark Red", "Gloves", (150, 20, 30)),
        ("P00001003", "Red Sweater", "Red", "Sweater", (210, 30, 30)),
        ("P00001004", "Red Dress", "Red", "Dress", (220, 20, 35)),
    ]
    for product_id, name, color, leaf, rgb in specs:
        path = image_dir / f"{product_id}.png"
        Image.new("RGB", (32, 32), color=rgb).save(path)
        rows.append(
            {
                "product_id": product_id,
                "name": name,
                "category_l1": "fashion",
                "category_l2": "accessories" if leaf == "Gloves" else "clothing",
                "category_l3": leaf.lower(),
                "leaf_category": leaf,
                "price": 29000,
                "color": color,
                "style_tags": [],
                "description": f"{color} {leaf}",
                "image_path": str(path),
            }
        )
    products_path = tmp_path / "hybrid_products.parquet"
    pd.DataFrame(rows).to_parquet(products_path)
    return products_path


def _config(tmp_path) -> MarsConfig:
    return MarsConfig(
        paths=PathsConfig(
            data_dir=tmp_path / "data",
            processed_dir=tmp_path / "data" / "processed",
            raw_dir=tmp_path / "data" / "raw",
            artifacts_dir=tmp_path / "artifacts",
            logs_dir=tmp_path / "logs",
        ),
        search=SearchConfig(encoder_type="fallback", embedding_dim=64, index_type="flat"),
    )


def test_fallback_encoder_is_deterministic() -> None:
    encoder = FallbackEncoder(dim=32, seed=42)
    first = encoder.encode_texts(["black leather jacket"])
    second = encoder.encode_texts(["black leather jacket"])
    assert first.shape == (1, 32)
    assert np.allclose(first, second)
    assert np.isclose(np.linalg.norm(first[0]), 1.0)


def test_vector_index_numpy_search_orders_by_similarity() -> None:
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.1]], dtype=np.float32)
    index = VectorIndex.build(vectors, index_type="flat", prefer_faiss=False)
    ids, scores = index.search(np.asarray([[1.0, 0.0]], dtype=np.float32), 2)
    assert ids.tolist() == [0, 2]
    assert scores[0] >= scores[1]


def test_build_artifacts_and_text_search_with_filters(tmp_path) -> None:
    products_path = _write_products(tmp_path)
    artifact_dir = tmp_path / "artifacts" / "search"
    encoder = FallbackEncoder(dim=64, seed=42)
    artifacts = build_search_artifacts(
        products_path,
        artifact_dir,
        encoder=encoder,
        index_type="flat",
    )
    service = SearchService.from_artifact_dir(
        artifact_dir,
        config=_config(tmp_path),
        encoder=encoder,
    )

    response = service.search(
        SearchRequest(
            search_type="text",
            query="black leather jacket",
            top_k=2,
            filters={"category": "outer"},
        )
    )

    assert artifacts.manifest["product_count"] == 3
    assert response["search_type"] == "text"
    assert response["total_count"] >= 1
    assert response["results"][0]["product_id"] == "P00000001"
    assert {"product_id", "name", "score", "price"}.issubset(response["results"][0])


def test_image_and_hybrid_search_return_required_shape(tmp_path) -> None:
    products_path = _write_products(tmp_path)
    artifact_dir = tmp_path / "artifacts" / "search"
    encoder = FallbackEncoder(dim=64, seed=42)
    build_search_artifacts(products_path, artifact_dir, encoder=encoder, index_type="flat")
    service = SearchService.from_artifact_dir(
        artifact_dir,
        config=_config(tmp_path),
        encoder=encoder,
    )

    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color=(240, 20, 20)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    image_response = service.search(
        SearchRequest(search_type="image", image_base64=encoded, top_k=2)
    )
    hybrid_response = service.search(
        SearchRequest(search_type="hybrid", query="red sneaker", image_base64=encoded, top_k=2)
    )

    for response in [image_response, hybrid_response]:
        assert response["total_count"] == 2
        assert response["latency_ms"] >= 0
        assert {"product_id", "name", "score", "price"}.issubset(response["results"][0])


def test_search_warmup_primes_text_and_image_paths(tmp_path) -> None:
    products_path = _write_products(tmp_path)
    artifact_dir = tmp_path / "artifacts" / "search"
    encoder = FallbackEncoder(dim=64, seed=42)
    build_search_artifacts(products_path, artifact_dir, encoder=encoder, index_type="flat")
    service = SearchService.from_artifact_dir(
        artifact_dir,
        config=_config(tmp_path),
        encoder=encoder,
    )

    summary = service.warmup()

    assert summary["ok"] is True
    assert summary["text"] == "ready"
    assert summary["catalog_image"] == "ready"
    assert summary["uploaded_image"] == "ready"
    assert summary["latency_ms"] >= 0


def test_hybrid_image_search_anchors_text_to_visual_leaf(tmp_path) -> None:
    products_path = _write_hybrid_anchor_products(tmp_path)
    artifact_dir = tmp_path / "artifacts" / "search"
    encoder = FallbackEncoder(dim=64, seed=42)
    build_search_artifacts(products_path, artifact_dir, encoder=encoder, index_type="flat")
    service = SearchService.from_artifact_dir(
        artifact_dir,
        config=_config(tmp_path),
        encoder=encoder,
    )

    response = service.search(
        SearchRequest(
            search_type="hybrid",
            query="red",
            image_path=str(tmp_path / "hybrid_images" / "P00001001.png"),
            top_k=4,
        )
    )

    assert response["results"][0]["product_id"] == "P00001002"
    assert response["results"][0]["name"] == "Paris Glove"


def test_lexical_score_uses_token_boundaries(tmp_path) -> None:
    products_path = _write_hybrid_anchor_products(tmp_path)
    artifact_dir = tmp_path / "artifacts" / "search"
    encoder = FallbackEncoder(dim=64, seed=42)
    build_search_artifacts(products_path, artifact_dir, encoder=encoder, index_type="flat")
    service = SearchService.from_artifact_dir(
        artifact_dir,
        config=_config(tmp_path),
        encoder=encoder,
    )
    black_tapered = service._metadata_records[0]
    red_glove = service._metadata_records[1]

    assert service._lexical_score(black_tapered, ["red"]) == 0.0
    assert service._lexical_score(red_glove, ["red"]) > 0.0


def test_image_color_hint_ignores_white_background() -> None:
    image = Image.new("RGB", (80, 100), color=(245, 245, 245))
    image.paste((220, 20, 30), box=(24, 28, 56, 96))

    assert _dominant_image_color_hints(image)[0] == "red"


def test_qrels_split_is_deterministic_and_disjoint(tmp_path) -> None:
    config = replace(
        _config(tmp_path),
        raw={
            "search": {
                "qrels_split_seed": 42,
                "qrels_train_ratio": 0.8,
                "qrels_valid_ratio": 0.1,
            }
        },
    )
    queries = pd.DataFrame({"query_id": [f"Q{idx:04d}" for idx in range(1000)]})

    train = select_qrels_split(queries, config, "train")
    valid = select_qrels_split(queries, config, "valid")
    test = select_qrels_split(queries, config, "test")

    assert 750 <= len(train) <= 850
    assert 70 <= len(valid) <= 130
    assert 70 <= len(test) <= 130
    assert (
        train["query_id"].tolist()
        == select_qrels_split(queries, config, "train")["query_id"].tolist()
    )
    assert set(train["query_id"]).isdisjoint(valid["query_id"])
    assert set(train["query_id"]).isdisjoint(test["query_id"])
    assert set(valid["query_id"]).isdisjoint(test["query_id"])

    shuffled = queries.sample(frac=1.0, random_state=7)
    assert set(train["query_id"]) == set(select_qrels_split(shuffled, config, "train")["query_id"])
    assert set(valid["query_id"]) == set(select_qrels_split(shuffled, config, "valid")["query_id"])
    assert set(test["query_id"]) == set(select_qrels_split(shuffled, config, "test")["query_id"])


def test_query_prior_uses_only_qrels_train_split(tmp_path) -> None:
    products_path = _write_products(tmp_path)
    artifact_dir = tmp_path / "artifacts" / "search"
    encoder = FallbackEncoder(dim=64, seed=42)
    build_search_artifacts(products_path, artifact_dir, encoder=encoder, index_type="flat")
    config = replace(
        _config(tmp_path),
        raw={
            "search": {
                "query_prior_top_k": 10,
                "query_prior_boost": 50.0,
                "query_token_prior_top_k": 0,
                "qrels_prior_train_only": True,
                "qrels_split_seed": 42,
                "qrels_train_ratio": 0.8,
                "qrels_valid_ratio": 0.1,
            }
        },
    )
    config.paths.processed_dir.mkdir(parents=True)
    queries = pd.DataFrame(
        {
            "query_id": [f"Q{idx:03d}" for idx in range(20)],
            "query": [f"unique query q{idx:02d}" for idx in range(20)],
            "positive_product_ids": [["P00000001"] for _ in range(20)],
        }
    )
    queries.to_parquet(config.paths.processed_dir / "search_queries.parquet", index=False)

    service = SearchService.from_artifact_dir(artifact_dir, config=config, encoder=encoder)
    train_queries = select_qrels_split(queries, config, "train")
    expected_keys = {f"unique query q{idx:02d}" for idx in train_queries.index}

    assert set(service._query_prior_index) == expected_keys  # noqa: SLF001


def test_query_behavior_model_loads_without_qrels_and_rejects_stale_split(tmp_path) -> None:
    products_path = _write_products(tmp_path)
    artifact_dir = tmp_path / "artifacts" / "search"
    encoder = FallbackEncoder(dim=64, seed=42)
    build_search_artifacts(products_path, artifact_dir, encoder=encoder, index_type="flat")
    behavior_path = artifact_dir / "query_behavior_model.json.gz"
    payload = {
        "schema_version": "search-query-behavior.v1",
        "split": "train",
        "seed": 42,
        "train_ratio": 0.8,
        "valid_ratio": 0.1,
        "query_prior_top_k": 10,
        "query_token_prior_top_k": 10,
        "query_prior": {"red sneaker": ["P00000002"]},
        "query_token_prior": {"red": [["P00000002", 3]]},
    }
    with gzip.open(behavior_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    config = replace(
        _config(tmp_path),
        raw={
            "search": {
                "query_prior_top_k": 10,
                "query_token_prior_top_k": 10,
                "query_behavior_model_path": str(behavior_path),
                "query_behavior_model_required": True,
                "qrels_split_seed": 42,
                "qrels_train_ratio": 0.8,
                "qrels_valid_ratio": 0.1,
            }
        },
    )
    service = SearchService.from_artifact_dir(artifact_dir, config=config, encoder=encoder)

    assert service._query_prior_index["red sneaker"] == [1]  # noqa: SLF001
    assert service._query_token_prior_index["red"][1] == 3  # noqa: SLF001

    stale_config = replace(
        config,
        raw={**config.raw, "search": {**config.raw["search"], "qrels_split_seed": 7}},
    )
    try:
        SearchService.from_artifact_dir(artifact_dir, config=stale_config, encoder=encoder)
    except ValueError as exc:
        assert "stale or incompatible" in str(exc)
    else:
        raise AssertionError("A stale query behavior model must be rejected")
