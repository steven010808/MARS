from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mars.config import load_config  # noqa: E402
from mars.config.settings import ensure_runtime_dirs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run search-only continuous learning checks from live API logs."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--logs", default="")
    parser.add_argument("--threshold", type=int, default=None)
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, mode=args.mode)
    ensure_runtime_dirs(config)
    logs_path = Path(args.logs) if args.logs else config.paths.logs_dir / "api_events.jsonl"
    state_path = config.paths.artifacts_dir / "registry" / "search_ct_state.json"
    threshold = int(args.threshold or config.monitoring.new_logs_threshold)

    while True:
        state = _read_json(state_path)
        current_count = _count_jsonl(logs_path)
        last_count = int(state.get("last_log_count", 0) or 0)
        new_logs = max(current_count - last_count, 0)
        payload: dict[str, Any] = {
            "checked_at": datetime.now(UTC).isoformat(),
            "log_source": str(logs_path),
            "current_log_count": current_count,
            "last_log_count": last_count,
            "new_logs": new_logs,
            "threshold": threshold,
            "action": "watching",
        }
        if new_logs >= threshold:
            payload["action"] = "refresh_started"
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            result = _run_refresh(args, logs_path)
            payload["action"] = "refresh_completed" if result.returncode == 0 else "refresh_failed"
            payload["returncode"] = result.returncode
            payload["stdout_tail"] = result.stdout[-2000:]
            payload["stderr_tail"] = result.stderr[-2000:]
            if result.returncode == 0:
                payload["last_log_count"] = current_count
                _write_json(state_path, payload)
        else:
            _write_json(state_path, payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        if args.once:
            return 0
        time.sleep(max(args.interval, 1))


def _run_refresh(args: argparse.Namespace, logs_path: Path) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "artifacts" / "refresh_search_behavior_model.py"),
        "--config",
        args.config,
        "--logs",
        str(logs_path),
    ]
    if args.mode:
        command.extend(["--mode", args.mode])
    if not args.no_promote:
        command.append("--promote")
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            total += chunk.count(b"\n")
    return total


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
