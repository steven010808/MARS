from __future__ import annotations

import json

from apps.api import service_adapters
from apps.api.service_adapters import ApiRuntime
from mars.config.settings import MarsConfig, PathsConfig


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
