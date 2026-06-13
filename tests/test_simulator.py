from __future__ import annotations

from pathlib import Path

from mars.config.settings import MarsConfig, ModeConfig, PathsConfig
from mars.data.io import read_json, read_table
from mars.data.simulator import generate_dataset, validate_manifest
from mars.schemas.core import EVENT_TYPES, PERSONAS


def _tiny_config(tmp_path: Path) -> MarsConfig:
    return MarsConfig(
        seed=42,
        active_mode="dev",
        modes={"dev": ModeConfig(products=80, users=24, events=240)},
        paths=PathsConfig(
            data_dir=tmp_path / "data",
            raw_dir=tmp_path / "data" / "raw",
            processed_dir=tmp_path / "data" / "processed",
            artifacts_dir=tmp_path / "artifacts",
            logs_dir=tmp_path / "logs",
        ),
    )


def test_simulator_generates_required_outputs(tmp_path: Path) -> None:
    manifest_path = generate_dataset(config=_tiny_config(tmp_path), clean=True)
    manifest = read_json(manifest_path)

    assert manifest["row_counts"]["products"] == 80
    assert manifest["row_counts"]["users"] == 24
    assert manifest["row_counts"]["events"] == 240
    assert set(manifest["persona_distribution"]) == set(PERSONAS)
    assert set(manifest["event_distribution"]) == set(EVENT_TYPES)
    for path in manifest["files"].values():
        assert Path(path).exists()


def test_simulator_is_reproducible_for_counts_and_distributions(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    first = read_json(generate_dataset(config=config, seed=42, clean=True))
    second = read_json(generate_dataset(config=config, seed=42, clean=True))

    assert first["config_hash"] == second["config_hash"]
    assert first["row_counts"] == second["row_counts"]
    assert first["persona_distribution"] == second["persona_distribution"]
    assert first["event_distribution"] == second["event_distribution"]


def test_validate_manifest_checks_generated_data(tmp_path: Path) -> None:
    manifest_path = generate_dataset(config=_tiny_config(tmp_path), clean=True)
    report = validate_manifest(manifest_path)

    assert report.ok, report.to_pretty_text()


def test_core_tables_have_expected_columns(tmp_path: Path) -> None:
    manifest = read_json(generate_dataset(config=_tiny_config(tmp_path), clean=True))

    products = read_table(manifest["files"]["products"])
    users = read_table(manifest["files"]["users"])
    events = read_table(manifest["files"]["events"])
    sessions = read_table(manifest["files"]["sessions"])
    queries = read_table(manifest["files"]["search_queries"])
    interactions = read_table(manifest["files"]["reco_interactions"])

    assert {"category_l1", "category_l2", "category_l3", "style_tags"}.issubset(products.columns)
    assert {"persona", "preferred_categories", "category_loyalty"}.issubset(users.columns)
    assert {"event_type", "session_id", "query_intent_category", "event_weight"}.issubset(
        events.columns
    )
    assert {"converted", "last_category", "last_product_id"}.issubset(sessions.columns)
    assert {"query", "positive_product_ids", "category_intent"}.issubset(queries.columns)
    assert {"label", "event_weight", "source_event_id"}.issubset(interactions.columns)
