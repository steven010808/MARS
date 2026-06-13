from __future__ import annotations

import json

from apps.api import service_adapters
from apps.api.service_adapters import ApiRuntime
from mars.config.settings import MarsConfig, MonitoringConfig, PathsConfig


def test_runtime_reset_live_run_archives_log_and_realigns_ct_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(service_adapters, "_instantiate", lambda service_class, config: None)
    monkeypatch.setattr(ApiRuntime, "_connect_redis", lambda self: None)

    cfg = MarsConfig(
        paths=PathsConfig(
            artifacts_dir=tmp_path / "artifacts",
            logs_dir=tmp_path / "logs",
        )
    )
    log_path = cfg.paths.logs_dir / "api_events.jsonl"
    log_path.parent.mkdir(parents=True)
    log_path.write_text('{"event_type":"search"}\n{"event_type":"view"}\n', encoding="utf-8")

    runtime = ApiRuntime(cfg)
    result = runtime.reset_live_run(reason="unit_test")

    assert result["rotated"] is True
    assert result["previous_lines"] == 2
    assert log_path.exists()
    assert log_path.read_text(encoding="utf-8") == ""

    archive_path = result["archive_path"]
    assert archive_path is not None
    assert "unit_test" in archive_path

    state = json.loads(
        (cfg.paths.artifacts_dir / "registry" / "ct_state.json").read_text(encoding="utf-8")
    )
    source = state["log_sources"]["api_events_jsonl"]
    assert source["last_log_count"] == 0
    assert source["current_log_count"] == 0
    assert source["pending_new_logs"] == 0
    assert state["log_rotations"][-1]["previous_lines"] == 2


def test_runtime_prepare_retrain_state_seeds_near_threshold_without_advancing_baseline(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(service_adapters, "_instantiate", lambda service_class, config: None)
    monkeypatch.setattr(ApiRuntime, "_connect_redis", lambda self: None)

    cfg = MarsConfig(
        paths=PathsConfig(
            artifacts_dir=tmp_path / "artifacts",
            logs_dir=tmp_path / "logs",
        ),
        monitoring=MonitoringConfig(new_logs_threshold=5),
    )
    log_path = cfg.paths.logs_dir / "api_events.jsonl"
    log_path.parent.mkdir(parents=True)
    log_path.write_text('{"event_type":"search"}\n{"event_type":"view"}\n', encoding="utf-8")

    state_path = cfg.paths.artifacts_dir / "registry" / "ct_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "log_sources": {
                    "api_events_jsonl": {
                        "last_log_count": 2,
                        "current_log_count": 2,
                        "pending_new_logs": 0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    runtime = ApiRuntime(cfg)
    result = runtime.prepare_retrain_state()

    assert result["prepared"] is True
    assert result["ready_to_retrain"] is False
    assert result["reason"] == "near_new_logs_threshold"
    assert result["added_events"] == 4
    assert result["last_log_count"] == 2
    assert result["current_log_count"] == 6
    assert result["pending_new_logs"] == 4
    assert result["threshold"] == 5
    assert result["remaining_to_threshold"] == 1
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 6

    state = json.loads(state_path.read_text(encoding="utf-8"))
    source = state["log_sources"]["api_events_jsonl"]
    assert source["last_log_count"] == 2
    assert source["current_log_count"] == 6
    assert source["pending_new_logs"] == 4
    assert state["last_decision"]["should_retrain"] is False
    assert state["last_decision"]["reasons"] == []
