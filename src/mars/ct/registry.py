from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelVersion:
    version: str
    created_at: str
    artifact_path: str
    metrics_path: str | None = None
    status: str = "candidate"
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"active_version": None, "versions": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, payload: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def register(
        self,
        artifact_path: str | Path,
        metrics_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
        activate: bool = False,
    ) -> ModelVersion:
        payload = self.load()
        version = self.next_version()
        entry = ModelVersion(
            version=version,
            created_at=datetime.now(UTC).isoformat(),
            artifact_path=str(artifact_path),
            metrics_path=str(metrics_path) if metrics_path else None,
            status="active" if activate else "candidate",
            metadata=metadata or {},
        )

        versions = payload.setdefault("versions", [])
        if activate:
            for existing in versions:
                if existing.get("status") == "active":
                    existing["status"] = "archived"
            payload["active_version"] = version
        versions.append(asdict(entry))
        self.save(payload)
        return entry

    def activate(self, version: str) -> None:
        payload = self.load()
        found = False
        for entry in payload.get("versions", []):
            if entry.get("version") == version:
                entry["status"] = "active"
                payload["active_version"] = version
                found = True
            elif entry.get("status") == "active":
                entry["status"] = "archived"
        if not found:
            raise ValueError(f"Unknown model version: {version}")
        self.save(payload)

    def next_version(self) -> str:
        return f"v{len(self.load().get('versions', [])) + 1:04d}"
