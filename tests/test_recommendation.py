from __future__ import annotations

import json

import pandas as pd
import pytest

from mars.config.settings import (
    MarsConfig,
    RecommendationConfig,
    RecommendationStrategyConfig,
    SearchConfig,
)
from mars.recommendation.artifacts import RecommendationArtifacts, _load_training_events
from mars.recommendation.models import TwoTowerModel, fit_torch_wide_deep_ranker
from mars.recommendation.service import (
    RecommendationService,
    _catalog_exploration_pool,
    _scaled_slots,
)
from mars.recommendation.session import InMemorySessionStore
from mars.recommendation.session_encoder import GRUSessionEncoder, fit_gru_session_encoder


def _artifacts() -> RecommendationArtifacts:
    products = []
    for idx in range(12):
        category = "outer" if idx < 6 else "shoes"
        products.append(
            {
                "product_id": f"P{idx:03d}",
                "name": f"Product {idx}",
                "category_l1": category,
                "price": 50_000 + idx,
                "popularity_prior": 1.0 - idx * 0.03,
                "margin_score": 0.1,
                "is_new": idx >= 10,
            }
        )
    users = {
        "U001": {
            "user_id": "U001",
            "persona": "top_category_loyalist",
            "preferred_categories": ["outer"],
            "price_sensitivity": 0.5,
        }
    }
    item_embeddings = [[1.0 if pos == idx % 4 else 0.0 for pos in range(16)] for idx in range(12)]
    return RecommendationArtifacts(
        version="test",
        embedding_dim=16,
        products=products,
        users=users,
        item_embeddings=item_embeddings,
        popularity_order=[product["product_id"] for product in products],
        trending_order=[product["product_id"] for product in reversed(products)],
        item_index={product["product_id"]: idx for idx, product in enumerate(products)},
        user_histories={"U001": ["P000", "P002", "P004"]},
        two_tower_model={
            "model_type": "torch_two_tower",
            "base_model": "feature_hash_two_tower",
            "embedding_dim": 16,
            "trained_samples": 10,
            "positive_samples": 2,
            "negative_samples": 8,
            "negative_sampling": "random negatives 1:4",
        },
    )


def _config() -> MarsConfig:
    return MarsConfig(
        search=SearchConfig(embedding_dim=16),
        recommendation=RecommendationConfig(
            candidate_k=8,
            final_top_n=5,
            exploration_slots=1,
            max_same_category_streak=2,
            session_recent_n=5,
        ),
    )


def test_recommendation_contract_and_candidate_count() -> None:
    service = RecommendationService(config=_config(), artifacts=_artifacts())

    candidates = service.generate_candidates("U001")
    response = service.recommend("U001", top_n=5, request_id="fixed")

    assert len(candidates) == 8
    assert response["user_id"] == "U001"
    assert set(response["pipeline_latency"]) == {
        "candidate_ms",
        "ranking_ms",
        "reranking_ms",
        "total_ms",
    }
    assert len(response["recommendations"]) == 5
    assert {"product_id", "score", "reason", "is_exploration"} <= set(
        response["recommendations"][0]
    )
    assert any(item["is_exploration"] for item in response["recommendations"])


def test_ab_strategies_use_distinct_slot_mixes() -> None:
    service = RecommendationService(config=_config(), artifacts=_artifacts())

    control = service.recommend("U001", top_n=5, request_id="fixed", strategy="control")
    treatment = service.recommend("U001", top_n=5, request_id="fixed", strategy="treatment")

    assert control["session_context"]["recommendation_strategy"] == "control"
    assert treatment["session_context"]["recommendation_strategy"] == "treatment"
    assert control["session_context"]["recommendation_strategy_label"] == "RankOnlyControl"
    assert treatment["session_context"]["recommendation_strategy_label"] == "ComplementGraphExplore"
    assert control["session_context"]["strategy_mix"]["slots"]["exploration_slots"] == 0
    assert control["session_context"]["strategy_mix"]["session_weight"] == 0.0
    assert treatment["session_context"]["strategy_mix"]["slots"]["transition_slots"] >= 1
    assert any(item["reason"].startswith("RankOnlyControl:") for item in control["recommendations"])
    assert any(
        item["reason"].startswith("ComplementGraphExplore:")
        for item in treatment["recommendations"]
    )
    assert not any(item["is_exploration"] for item in control["recommendations"])
    assert any(item["is_exploration"] for item in treatment["recommendations"])
    assert any(item["product_id"] in {"P010", "P011"} for item in treatment["recommendations"])
    assert control["session_context"]["session_features_enabled"] is False
    assert treatment["session_context"]["session_features_enabled"] is True
    assert control["recommendations"] != treatment["recommendations"]


def test_adaptive_repeat_explore_gate_reports_effective_mix() -> None:
    config = MarsConfig(
        search=SearchConfig(embedding_dim=16),
        recommendation=RecommendationConfig(
            candidate_k=8,
            final_top_n=5,
            exploration_slots=1,
            max_same_category_streak=2,
            session_recent_n=5,
            strategies={
                "treatment": RecommendationStrategyConfig(
                    label="RepeatExploreGate",
                    current_category_slots=3,
                    transition_slots=1,
                    long_term_slots=1,
                    exploration_slots=1,
                    long_term_weight=0.55,
                    session_weight=0.45,
                    transition_boost=0.8,
                    adaptive_gate=True,
                    adaptive_margin=0.12,
                )
            },
        ),
    )
    store = InMemorySessionStore(recent_n=5)
    service = RecommendationService(config=config, artifacts=_artifacts(), session_store=store)
    service.update_event(
        {
            "user_id": "U001",
            "session_id": "S-adaptive",
            "event_type": "view",
            "product_id": "P000",
            "category": "outer",
        }
    )

    response = service.recommend("U001", top_n=5, session_id="S-adaptive", strategy="treatment")
    context = response["session_context"]

    assert context["recommendation_strategy_label"] == "RepeatExploreGate"
    assert context["repeat_explore_gate"]["mode"] in {"repeat", "explore", "neutral"}
    assert "strategy_mix_effective" in context


def test_unknown_user_uses_popularity_fallback() -> None:
    service = RecommendationService(config=_config(), artifacts=_artifacts())

    response = service.recommend("NEW_USER", top_n=3)

    product_ids = [item["product_id"] for item in response["recommendations"]]
    assert product_ids
    assert len(product_ids) == len(set(product_ids))
    assert response["recommendations"][0]["reason"] in {
        "cold_start_popularity",
        "exploration:relevance",
        "exploration:diversity",
        "exploration:novelty",
        "exploration:price_match",
    } or response["recommendations"][0]["reason"].startswith("ComplementGraphExplore:")


def test_mab_arm_changes_exploration_product_selection() -> None:
    relevance_service = RecommendationService(config=_config(), artifacts=_artifacts())
    novelty_service = RecommendationService(config=_config(), artifacts=_artifacts())
    for arm in relevance_service.mab.arms:
        relevance_service.mab.stats[arm] = {"successes": 1.0, "trials": 100.0}
        novelty_service.mab.stats[arm] = {"successes": 1.0, "trials": 100.0}
    relevance_service.mab.stats["relevance"] = {"successes": 100.0, "trials": 100.0}
    novelty_service.mab.stats["novelty"] = {"successes": 100.0, "trials": 100.0}

    relevance = relevance_service.recommend("NEW_USER", top_n=5, request_id="same")
    novelty = novelty_service.recommend("NEW_USER", top_n=5, request_id="same")
    relevance_explore = next(
        item for item in relevance["recommendations"] if item["is_exploration"]
    )
    novelty_explore = next(item for item in novelty["recommendations"] if item["is_exploration"])

    assert relevance_explore["arm"] == "relevance"
    assert novelty_explore["arm"] == "novelty"
    assert relevance_explore["product_id"] != novelty_explore["product_id"]


def test_catalog_exploration_pool_is_deterministic_and_request_specific() -> None:
    products = _artifacts().products

    first = _catalog_exploration_pool(products, rotation_token="U1:req-1", limit=4)
    repeated = _catalog_exploration_pool(products, rotation_token="U1:req-1", limit=4)
    other_request = _catalog_exploration_pool(products, rotation_token="U1:req-2", limit=4)
    first_ids = [str(product["product_id"]) for product in first]

    assert first_ids == [str(product["product_id"]) for product in repeated]
    assert len(first_ids) == len(set(first_ids)) == 4
    assert first_ids != [str(product["product_id"]) for product in other_request]


def test_treatment_slot_scaling_keeps_top10_conservative_and_expands_top50_exploration() -> None:
    strategy = RecommendationStrategyConfig(
        label="CoverageTreatment",
        current_category_slots=6,
        transition_slots=2,
        long_term_slots=1,
        exploration_slots=3,
        long_term_weight=0.55,
        session_weight=0.45,
    )

    assert _scaled_slots(strategy, 10)["exploration_slots"] == 2
    assert _scaled_slots(strategy, 50)["exploration_slots"] == 13


def test_treatment_preserves_ranked_prefix_and_places_exploration_at_tail() -> None:
    service = RecommendationService(config=_config(), artifacts=_artifacts())
    context = {"recent_products": ["P000"], "recent_categories": ["outer"]}
    user = service.artifacts.users["U001"]
    candidates = service.generate_candidates("U001", context, candidate_k=8)
    ranked = service.rank_candidates(user, candidates, context)

    reranked = service.rerank(
        "U001",
        ranked,
        top_n=5,
        request_id="tail-check",
        strategy="treatment",
        user=user,
        session_context=context,
    )
    exploration_indices = [idx for idx, item in enumerate(reranked) if item["is_exploration"]]

    assert reranked[0]["product"]["product_id"] == ranked[0]["product"]["product_id"]
    assert exploration_indices == [4]


def test_session_store_updates_context() -> None:
    store = InMemorySessionStore(recent_n=5)
    service = RecommendationService(config=_config(), artifacts=_artifacts(), session_store=store)

    before = service.recommend("U001", top_n=3, session_id="S1")["session_context"]
    after = service.update_event(
        {
            "user_id": "U001",
            "session_id": "S1",
            "event_type": "view",
            "product_id": "P002",
            "category": "outer",
        }
    )

    assert before["num_recent_events"] == 0
    assert after["recent_products"] == ["P002"]
    assert after["recent_categories"] == ["outer"]
    assert after["event_counts"]["view"] == 1
    assert after["num_recent_events"] == 1


def test_session_store_reads_category_from_metadata() -> None:
    store = InMemorySessionStore(recent_n=5)
    service = RecommendationService(config=_config(), artifacts=_artifacts(), session_store=store)

    context = service.update_event(
        {
            "user_id": "U001",
            "session_id": "S-metadata-category",
            "event_type": "view",
            "product_id": "P002",
            "metadata": {"category": "outer"},
        }
    )
    response = service.recommend("U001", top_n=3, session_id="S-metadata-category")

    assert context["recent_categories"] == ["outer"]
    assert response["session_context"]["session_interest"] == "outer"


def test_session_store_isolates_sessions_for_the_same_user() -> None:
    store = InMemorySessionStore(recent_n=5)
    service = RecommendationService(config=_config(), artifacts=_artifacts(), session_store=store)
    service.update_event(
        {
            "user_id": "U001",
            "session_id": "S-first",
            "event_type": "view",
            "product_id": "P002",
            "category": "outer",
        }
    )

    other_session = service.recommend("U001", top_n=3, session_id="S-second")["session_context"]
    user_context = store.get_context("U001")

    assert other_session["num_recent_events"] == 0
    assert other_session["recent_products"] == []
    assert user_context["num_recent_events"] == 1
    assert user_context["recent_products"] == ["P002"]


def test_gru_session_encoder_is_used_for_recent_products() -> None:
    store = InMemorySessionStore(recent_n=5)
    service = RecommendationService(config=_config(), artifacts=_artifacts(), session_store=store)
    service.update_event(
        {
            "user_id": "U001",
            "session_id": "S-gru",
            "event_type": "view",
            "product_id": "P008",
            "category": "shoes",
        }
    )

    response = service.recommend("U001", top_n=3, session_id="S-gru")
    encoder = response["session_context"]["session_encoder"]

    assert isinstance(service.session_encoder, GRUSessionEncoder)
    assert encoder["sequence_length"] == 1
    assert encoder["source_products"] == ["P008"]
    assert encoder["type"] in {"gru_trained", "gru_untrained", "pooled_fallback"}


def test_two_tower_payload_is_preserved_in_artifacts() -> None:
    artifacts = _artifacts()
    artifacts.session_encoder_model = {
        "model_type": "torch_gru_session_encoder",
        "trained_samples": 3,
    }
    restored = RecommendationArtifacts.from_payload(artifacts.to_payload())

    assert restored.two_tower_model
    assert restored.two_tower_model["model_type"] == "torch_two_tower"
    assert restored.two_tower_model["negative_sampling"] == "random negatives 1:4"
    assert restored.session_encoder_model
    assert restored.session_encoder_model["trained_samples"] == 3


def test_two_tower_item_price_and_user_sequence_affect_embeddings() -> None:
    model = TwoTowerModel(embedding_dim=16, seed=42)
    base_product = {"product_id": "P1", "name": "Coat", "category_l1": "outer", "price": 10_000}
    expensive_product = {**base_product, "price": 200_000}
    user = {"user_id": "U1", "preferred_categories": ["outer"]}

    assert model.encode_item(base_product) != model.encode_item(expensive_product)
    assert model.encode_user(user, {"recent_products": ["P1", "P2"]}) != model.encode_user(
        user,
        {"recent_products": ["P2", "P1"]},
    )


def test_ranker_uses_exposure_non_click_negative_labels() -> None:
    pytest.importorskip("torch")
    users = {"U1": {"user_id": "U1", "preferred_categories": ["outer"]}}
    products = {
        "P1": {"product_id": "P1", "category_l1": "outer", "price": 10_000},
        "P2": {"product_id": "P2", "category_l1": "shoes", "price": 50_000},
    }
    events = [
        {
            "event_id": "E1",
            "user_id": "U1",
            "session_id": "S1",
            "event_type": "search",
            "product_id": "P1",
            "metadata": {"event_role": "exposure", "exposure_id": "X1"},
        },
        {
            "event_id": "E2",
            "user_id": "U1",
            "session_id": "S1",
            "event_type": "search",
            "product_id": "P2",
            "metadata": {"event_role": "exposure", "exposure_id": "X2"},
        },
        {
            "event_id": "E3",
            "user_id": "U1",
            "session_id": "S1",
            "event_type": "view",
            "product_id": "P1",
            "metadata": {"event_role": "response", "exposure_id": "X1"},
        },
    ]

    payload = fit_torch_wide_deep_ranker(
        users=users,
        products_by_id=products,
        events=events,
        two_tower=TwoTowerModel(embedding_dim=16, seed=42),
        seed=42,
        max_samples=10,
    )

    assert payload
    assert payload["positive_samples"] >= 1
    assert payload["negative_samples"] >= 1
    assert payload["negative_sampling"].startswith("explicit exposure non-click")


def test_trained_gru_session_encoder_payload_is_loaded() -> None:
    pytest.importorskip("torch")
    embeddings = {
        "P1": [1.0, 0.0, 0.0, 0.0],
        "P2": [0.0, 1.0, 0.0, 0.0],
        "P3": [0.0, 0.0, 1.0, 0.0],
    }
    events = [
        {"user_id": "U1", "session_id": "S1", "event_type": "view", "product_id": "P1"},
        {"user_id": "U1", "session_id": "S1", "event_type": "view", "product_id": "P2"},
        {"user_id": "U1", "session_id": "S1", "event_type": "view", "product_id": "P3"},
    ]
    payload = fit_gru_session_encoder(
        events=events,
        item_embeddings=embeddings,
        embedding_dim=4,
        seed=42,
        max_sequence_length=3,
        max_samples=3,
    )
    encoder = GRUSessionEncoder(
        embedding_dim=4,
        seed=42,
        max_sequence_length=3,
        model_payload=payload,
    )

    result = encoder.encode(["P2", "P1"], embeddings)

    assert payload
    assert result
    assert result.encoder_type == "gru_trained"
    assert result.source_products == ["P1", "P2"]


def test_training_events_include_deduplicated_live_log(tmp_path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame(
        [
            {"event_id": "E-train", "user_id": "U1", "event_type": "view", "product_id": "P1"},
        ]
    ).to_parquet(processed / "train_events.parquet", index=False)
    live_path = tmp_path / "api_events.jsonl"
    live_rows = [
        {"event_id": "E-live", "user_id": "U1", "event_type": "cart", "product_id": "P2"},
        {"event_id": "E-live", "user_id": "U1", "event_type": "cart", "product_id": "P2"},
    ]
    live_path.write_text(
        "\n".join(json.dumps(row) for row in live_rows) + "\n",
        encoding="utf-8",
    )

    events, source = _load_training_events(processed, live_events_path=live_path)

    assert [event["event_id"] for event in events] == ["E-train", "E-live"]
    assert source == "train_events.parquet+api_events.jsonl"
