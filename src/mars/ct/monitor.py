from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mars.config import MarsConfig


@dataclass(frozen=True)
class MonitorSnapshot:
    checked_at: str
    ctr: float
    cvr: float
    hit_rate: float
    new_logs: int
    thresholds: dict[str, float | int]


@dataclass(frozen=True)
class CTDecision:
    should_retrain: bool
    reasons: list[str]
    snapshot: MonitorSnapshot

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CTMonitor:
    def __init__(self, config: MarsConfig, state_path: str | Path | None = None) -> None:
        self.config = config
        self.state_path = (
            Path(state_path)
            if state_path
            else config.paths.artifacts_dir / "registry" / "ct_state.json"
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        metrics: dict[str, Any],
        current_log_count: int,
        write_state: bool = True,
        source_key: str = "default",
        advance_log_count: bool = True,
    ) -> CTDecision:
        state = self._load_state()
        source_key = str(source_key or "default")
        log_sources = state.get("log_sources", {})
        if not isinstance(log_sources, dict):
            log_sources = {}
        source_state = log_sources.get(source_key, {})
        if not isinstance(source_state, dict):
            source_state = {}
        if "last_log_count" in source_state:
            last_log_count = int(source_state.get("last_log_count", 0))
        elif source_key == "default":
            last_log_count = int(state.get("last_log_count", 0))
        else:
            last_log_count = 0
        new_logs = max(current_log_count - last_log_count, 0)

        ctr_value = _extract(metrics, ("monitoring", "ctr"), default=None)
        if ctr_value is None:
            ctr_value = _extract(metrics, ("system", "ctr"), default=None)
        if ctr_value is None:
            ctr_value = _extract(metrics, ("ab_test", "buckets", "treatment", "ctr"), default=None)
        if ctr_value is None:
            ctr_value = _extract(metrics, ("ab_test", "buckets", "control", "ctr"), default=0.0)
        cvr_value = _extract(metrics, ("monitoring", "cvr"), default=None)
        if cvr_value is None:
            cvr_value = _extract(metrics, ("system", "cvr"), default=None)
        if cvr_value is None:
            cvr_value = _extract(metrics, ("ab_test", "buckets", "treatment", "cvr"), default=None)
        if cvr_value is None:
            cvr_value = _extract(metrics, ("ab_test", "buckets", "control", "cvr"), default=0.0)
        hit_rate = _extract(metrics, ("recommendation", "hit_rate_at_50"), default=0.0)

        reasons: list[str] = []
        enough_ctr_volume = current_log_count >= self.config.monitoring.ctr_min_logs
        enough_new_logs = new_logs >= self.config.monitoring.new_logs_threshold
        if (enough_ctr_volume or enough_new_logs) and float(
            ctr_value
        ) < self.config.monitoring.ctr_threshold:
            reasons.append("ctr_below_threshold")
        if float(hit_rate) < self.config.monitoring.hitrate_threshold:
            reasons.append("hitrate_below_threshold")
        if enough_new_logs:
            reasons.append("new_logs_threshold_reached")

        snapshot = MonitorSnapshot(
            checked_at=datetime.now(UTC).isoformat(),
            ctr=float(ctr_value),
            cvr=float(cvr_value),
            hit_rate=float(hit_rate),
            new_logs=new_logs,
            thresholds={
                "ctr_threshold": self.config.monitoring.ctr_threshold,
                "hitrate_threshold": self.config.monitoring.hitrate_threshold,
                "new_logs_threshold": self.config.monitoring.new_logs_threshold,
                "ctr_min_logs": self.config.monitoring.ctr_min_logs,
            },
        )
        decision = CTDecision(should_retrain=bool(reasons), reasons=reasons, snapshot=snapshot)

        should_advance_log_count = (
            advance_log_count or not source_state or "new_logs_threshold_reached" in reasons
        )

        if write_state:
            next_log_count = int(current_log_count) if should_advance_log_count else last_log_count
            log_sources[source_key] = {
                "last_checked_at": snapshot.checked_at,
                "last_log_count": next_log_count,
                "current_log_count": int(current_log_count),
                "pending_new_logs": new_logs,
            }
            state.update(
                {
                    "last_checked_at": snapshot.checked_at,
                    "last_log_count": next_log_count,
                    "last_log_source": source_key,
                    "log_sources": log_sources,
                    "last_decision": decision.to_dict(),
                }
            )
            self.state_path.write_text(
                json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return decision

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text(encoding="utf-8"))


def _extract(payload: dict[str, Any], path: tuple[str, ...], default: Any) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
