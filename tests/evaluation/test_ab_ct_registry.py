from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from mars.config.settings import MarsConfig, MonitoringConfig, PathsConfig
from mars.ct import CTMonitor, ModelRegistry
from mars.evaluation.ab import assign_bucket, build_ab_report, two_proportion_z_test
from scripts.runtime import worker_loop


def test_assignment_is_deterministic() -> None:
    first = assign_bucket("U0001", "exp1")
    second = assign_bucket("U0001", "exp1")
    assert first == second
    assert first.bucket in {"control", "treatment"}


def test_ab_report_contains_rates_and_significance() -> None:
    events = pd.DataFrame(
        {
            "ab_group": ["control"] * 100 + ["treatment"] * 100,
            "event_type": ["view"] * 10 + ["impression"] * 90 + ["view"] * 30 + ["impression"] * 70,
        }
    )
    report = build_ab_report(events)
    assert report.buckets["control"]["ctr"] == 0.1
    assert report.buckets["treatment"]["ctr"] == 0.3
    assert report.p_value < 0.01
    assert two_proportion_z_test(10, 100, 30, 100) == report.p_value


def test_ct_monitor_and_registry(tmp_path) -> None:
    cfg = MarsConfig(
        paths=PathsConfig(artifacts_dir=tmp_path / "artifacts"),
        monitoring=MonitoringConfig(
            hitrate_threshold=0.2, ctr_threshold=0.03, new_logs_threshold=10
        ),
    )
    metrics = {
        "ab_test": {"buckets": {"treatment": {"ctr": 0.02}}},
        "recommendation": {"hit_rate_at_50": 0.1},
    }
    monitor = CTMonitor(cfg, state_path=tmp_path / "ct_state.json")
    decision = monitor.evaluate(metrics, current_log_count=11)
    assert decision.should_retrain
    assert set(decision.reasons) == {
        "ctr_below_threshold",
        "hitrate_below_threshold",
        "new_logs_threshold_reached",
    }

    registry = ModelRegistry(tmp_path / "models.json")
    entry = registry.register(
        "artifacts/models/v1", metrics_path="artifacts/reports/metrics.json", activate=True
    )
    payload = json.loads((tmp_path / "models.json").read_text(encoding="utf-8"))
    assert entry.version == "v0001"
    assert payload["active_version"] == "v0001"


def test_ct_monitor_tracks_log_sources_independently(tmp_path) -> None:
    cfg = MarsConfig(
        paths=PathsConfig(artifacts_dir=tmp_path / "artifacts"),
        monitoring=MonitoringConfig(
            hitrate_threshold=0.2, ctr_threshold=0.03, new_logs_threshold=10
        ),
    )
    metrics = {
        "ab_test": {"buckets": {"treatment": {"ctr": 1.0}}},
        "recommendation": {"hit_rate_at_50": 0.9},
    }
    monitor = CTMonitor(cfg, state_path=tmp_path / "ct_state.json")

    monitor.evaluate(metrics, current_log_count=100, source_key="processed_events")
    decision = monitor.evaluate(metrics, current_log_count=11, source_key="api_events_jsonl")

    assert decision.should_retrain
    assert decision.snapshot.new_logs == 11
    assert decision.reasons == ["new_logs_threshold_reached"]


def test_ct_monitor_can_accumulate_live_logs_between_worker_checks(tmp_path) -> None:
    cfg = MarsConfig(
        paths=PathsConfig(artifacts_dir=tmp_path / "artifacts"),
        monitoring=MonitoringConfig(
            hitrate_threshold=0.2, ctr_threshold=0.03, new_logs_threshold=10
        ),
    )
    metrics = {
        "ab_test": {"buckets": {"treatment": {"ctr": 1.0}}},
        "recommendation": {"hit_rate_at_50": 0.9},
    }
    monitor = CTMonitor(cfg, state_path=tmp_path / "ct_state.json")

    monitor.evaluate(
        metrics,
        current_log_count=100,
        source_key="api_events_jsonl",
        advance_log_count=False,
    )
    first = monitor.evaluate(
        metrics,
        current_log_count=106,
        source_key="api_events_jsonl",
        advance_log_count=False,
    )
    second = monitor.evaluate(
        metrics,
        current_log_count=111,
        source_key="api_events_jsonl",
        advance_log_count=False,
    )

    assert not first.should_retrain
    assert first.snapshot.new_logs == 6
    assert second.should_retrain
    assert second.snapshot.new_logs == 11


def test_ct_monitor_realigns_when_live_log_is_reset(tmp_path) -> None:
    cfg = MarsConfig(
        paths=PathsConfig(artifacts_dir=tmp_path / "artifacts"),
        monitoring=MonitoringConfig(
            hitrate_threshold=0.2, ctr_threshold=0.03, new_logs_threshold=10
        ),
    )
    metrics = {
        "ab_test": {"buckets": {"treatment": {"ctr": 1.0}}},
        "recommendation": {"hit_rate_at_50": 0.9},
    }
    state_path = tmp_path / "ct_state.json"
    monitor = CTMonitor(cfg, state_path=state_path)

    monitor.evaluate(metrics, current_log_count=100, source_key="api_events_jsonl")
    decision = monitor.evaluate(
        metrics,
        current_log_count=3,
        source_key="api_events_jsonl",
        advance_log_count=False,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    source = state["log_sources"]["api_events_jsonl"]
    assert not decision.should_retrain
    assert decision.snapshot.new_logs == 0
    assert source["last_log_count"] == 3
    assert source["reset_reason"] == "current_log_count_below_last_log_count"
    assert state["log_realignments"][-1]["previous_last_log_count"] == 100


def test_worker_runs_search_refresh_with_promotion_when_enabled(tmp_path, monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(command, capture_output, text, check):
        captured["command"] = command
        assert capture_output is True
        assert text is True
        assert check is False
        return SimpleNamespace(returncode=0, stdout='{"promoted": true}', stderr="")

    monkeypatch.setattr(worker_loop.subprocess, "run", fake_run)
    cfg = MarsConfig(
        active_mode="dev",
        paths=PathsConfig(artifacts_dir=tmp_path / "artifacts"),
        raw={
            "search": {
                "online_learning": {
                    "enabled": True,
                    "auto_refresh": True,
                    "promote_on_pass": True,
                    "max_eval_queries": 123,
                }
            }
        },
    )

    result = worker_loop._run_search_refresh(
        cfg,
        config_path="configs/config.yaml",
        logs_path=Path("logs/api_events.jsonl"),
        enabled=True,
        promote_enabled=True,
    )

    assert result["status"] == "completed"
    assert "--promote" in captured["command"]
    assert "--max-eval-queries" in captured["command"]
    assert "123" in captured["command"]
