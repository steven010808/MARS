from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModeConfig:
    products: int
    users: int
    events: int


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path = Path("data")
    processed_dir: Path = Path("data/processed")
    raw_dir: Path = Path("data/raw")
    artifacts_dir: Path = Path("artifacts")
    logs_dir: Path = Path("logs")


@dataclass(frozen=True)
class SearchConfig:
    encoder_type: str = "clip"
    clip_model: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 512
    index_type: str = "hnsw"
    top_k: int = 10
    max_top_k: int = 100
    hybrid_text_weight: float = 0.55
    allow_fallback_encoder: bool = True


@dataclass(frozen=True)
class RecommendationStrategyConfig:
    label: str
    current_category_slots: int
    transition_slots: int
    long_term_slots: int
    exploration_slots: int
    long_term_weight: float
    session_weight: float
    transition_boost: float = 0.45
    adaptive_gate: bool = False
    adaptive_margin: float = 0.15


@dataclass(frozen=True)
class RecommendationConfig:
    candidate_k: int = 300
    final_top_n: int = 10
    embedding_dim: int = 128
    exploration_slots: int = 2
    max_same_category_streak: int = 2
    session_recent_n: int = 20
    long_term_weight: float = 0.7
    session_weight: float = 0.3
    strategies: dict[str, RecommendationStrategyConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class MonitoringConfig:
    hitrate_threshold: float = 0.20
    ctr_threshold: float = 0.03
    new_logs_threshold: int = 10_000
    ctr_min_logs: int = 100


@dataclass(frozen=True)
class MarsConfig:
    seed: int = 42
    active_mode: str = "dev"
    modes: dict[str, ModeConfig] = field(
        default_factory=lambda: {
            "dev": ModeConfig(products=5_000, users=1_000, events=50_000),
            "full": ModeConfig(products=50_000, users=10_000, events=1_000_000),
        }
    )
    paths: PathsConfig = field(default_factory=PathsConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    recommendation: RecommendationConfig = field(default_factory=RecommendationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    redis_url: str = "redis://redis:6379/0"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def mode(self) -> ModeConfig:
        return self.modes[self.active_mode]


def _mode_config(raw: dict[str, Any]) -> ModeConfig:
    return ModeConfig(
        products=int(raw["products"]),
        users=int(raw["users"]),
        events=int(raw["events"]),
    )


def _default_modes_from_root_scale(raw: dict[str, Any]) -> dict[str, ModeConfig]:
    simulator = raw.get("simulator", {}) if isinstance(raw, dict) else {}
    scale = simulator.get("scale", {}) if isinstance(simulator, dict) else {}
    target_scale = simulator.get("target_scale", {}) if isinstance(simulator, dict) else {}
    full_users = int(scale.get("users", target_scale.get("users", 10_000)))
    full_events = int(scale.get("events", target_scale.get("events", 1_000_000)))
    return {
        "dev": ModeConfig(products=5_000, users=1_000, events=50_000),
        "full": ModeConfig(products=50_000, users=full_users, events=full_events),
    }


def load_config(path: str | Path = "configs/config.yaml", mode: str | None = None) -> MarsConfig:
    config_path = Path(path)
    raw: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

    modes = {
        name: _mode_config(value) for name, value in raw.get("modes", {}).items()
    } or _default_modes_from_root_scale(raw)

    paths_raw = raw.get("paths", {})
    search_raw = raw.get("search", {})
    reco_raw = raw.get("recommendation", {})
    monitoring_raw = raw.get("monitoring", {})
    app_raw = raw.get("app", {})
    redis_raw = raw.get("redis", {})
    default_seed = int(app_raw.get("random_seed", raw.get("seed", 42)))
    redis_url = raw.get("redis_url")
    if not redis_url:
        redis_host = str(redis_raw.get("host", "redis"))
        redis_port = int(redis_raw.get("port", 6379))
        redis_db = int(redis_raw.get("db", 0))
        redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"

    active_mode = mode or raw.get("active_mode", "full")
    if active_mode not in modes:
        raise ValueError(f"Unknown mode '{active_mode}'. Available: {sorted(modes)}")

    strategy_defaults = _recommendation_strategy_defaults(reco_raw)

    return MarsConfig(
        seed=default_seed,
        active_mode=active_mode,
        modes=modes,
        paths=PathsConfig(
            data_dir=Path(paths_raw.get("data_dir", "data")),
            processed_dir=Path(paths_raw.get("processed_dir", "data/processed")),
            raw_dir=Path(paths_raw.get("raw_dir", "data/raw")),
            artifacts_dir=Path(paths_raw.get("artifacts_dir", "artifacts")),
            logs_dir=Path(paths_raw.get("logs_dir", "logs")),
        ),
        search=SearchConfig(
            encoder_type=str(search_raw.get("encoder_type", "clip")),
            clip_model=str(search_raw.get("clip_model", "openai/clip-vit-base-patch32")),
            embedding_dim=int(search_raw.get("embedding_dim", 512)),
            index_type=str(search_raw.get("index_type", "hnsw")),
            top_k=int(search_raw.get("top_k", 10)),
            max_top_k=int(search_raw.get("max_top_k", 100)),
            hybrid_text_weight=float(search_raw.get("hybrid_text_weight", 0.55)),
            allow_fallback_encoder=bool(search_raw.get("allow_fallback_encoder", True)),
        ),
        recommendation=RecommendationConfig(
            candidate_k=int(reco_raw.get("candidate_k", 300)),
            final_top_n=int(reco_raw.get("final_top_n", 10)),
            embedding_dim=int(reco_raw.get("embedding_dim", 128)),
            exploration_slots=int(reco_raw.get("exploration_slots", 2)),
            max_same_category_streak=int(reco_raw.get("max_same_category_streak", 2)),
            session_recent_n=int(reco_raw.get("session_recent_n", 20)),
            long_term_weight=float(reco_raw.get("long_term_weight", 0.7)),
            session_weight=float(reco_raw.get("session_weight", 0.3)),
            strategies=strategy_defaults,
        ),
        monitoring=MonitoringConfig(
            hitrate_threshold=float(monitoring_raw.get("hitrate_threshold", 0.20)),
            ctr_threshold=float(monitoring_raw.get("ctr_threshold", 0.03)),
            new_logs_threshold=int(monitoring_raw.get("new_logs_threshold", 10_000)),
            ctr_min_logs=int(monitoring_raw.get("ctr_min_logs", 100)),
        ),
        redis_url=str(redis_url),
        raw=raw,
    )


def _recommendation_strategy_defaults(
    raw: dict[str, Any],
) -> dict[str, RecommendationStrategyConfig]:
    defaults = {
        "baseline_vanilla": {
            "label": "BaselineVanilla",
            "current_category_slots": 0,
            "transition_slots": 0,
            "long_term_slots": 0,
            "exploration_slots": 2,
            "long_term_weight": 0.7,
            "session_weight": 0.3,
            "transition_boost": 0.0,
            "adaptive_gate": False,
            "adaptive_margin": 0.15,
        },
        "control": {
            "label": "MissionPrecision",
            "current_category_slots": 7,
            "transition_slots": 0,
            "long_term_slots": 2,
            "exploration_slots": 1,
            "long_term_weight": 0.75,
            "session_weight": 0.25,
            "transition_boost": 0.0,
        },
        "treatment": {
            "label": "ComplementGraphExplore",
            "current_category_slots": 5,
            "transition_slots": 2,
            "long_term_slots": 1,
            "exploration_slots": 2,
            "long_term_weight": 0.55,
            "session_weight": 0.45,
            "transition_boost": 0.55,
            "adaptive_gate": False,
            "adaptive_margin": 0.15,
        },
    }
    raw_strategies = raw.get("strategies", {}) if isinstance(raw, dict) else {}
    if isinstance(raw_strategies, dict) and "legacy_vanilla" in raw_strategies:
        raw_strategies = dict(raw_strategies)
        raw_strategies.setdefault("baseline_vanilla", raw_strategies["legacy_vanilla"])
    merged: dict[str, RecommendationStrategyConfig] = {}
    strategy_keys = list(defaults)
    if isinstance(raw_strategies, dict):
        strategy_keys.extend(key for key in raw_strategies if key not in defaults)
    for key in strategy_keys:
        fallback = defaults.get(
            key,
            {
                "label": str(key),
                "current_category_slots": 5,
                "transition_slots": 0,
                "long_term_slots": 2,
                "exploration_slots": 1,
                "long_term_weight": float(raw.get("long_term_weight", 0.7)),
                "session_weight": float(raw.get("session_weight", 0.3)),
                "transition_boost": 0.0,
                "adaptive_gate": False,
                "adaptive_margin": 0.15,
            },
        )
        payload = dict(fallback)
        override = raw_strategies.get(key, {}) if isinstance(raw_strategies, dict) else {}
        if isinstance(override, dict):
            payload.update(override)
        merged[key] = RecommendationStrategyConfig(
            label=str(payload.get("label", fallback["label"])),
            current_category_slots=int(
                payload.get("current_category_slots", fallback["current_category_slots"])
            ),
            transition_slots=int(payload.get("transition_slots", fallback["transition_slots"])),
            long_term_slots=int(payload.get("long_term_slots", fallback["long_term_slots"])),
            exploration_slots=int(payload.get("exploration_slots", fallback["exploration_slots"])),
            long_term_weight=float(payload.get("long_term_weight", fallback["long_term_weight"])),
            session_weight=float(payload.get("session_weight", fallback["session_weight"])),
            transition_boost=float(payload.get("transition_boost", fallback["transition_boost"])),
            adaptive_gate=_bool_from_raw(
                payload.get("adaptive_gate", fallback.get("adaptive_gate", False))
            ),
            adaptive_margin=float(
                payload.get("adaptive_margin", fallback.get("adaptive_margin", 0.15))
            ),
        )
    return merged


def _bool_from_raw(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def ensure_runtime_dirs(config: MarsConfig) -> None:
    for path in [
        config.paths.data_dir,
        config.paths.processed_dir,
        config.paths.raw_dir,
        config.paths.artifacts_dir,
        config.paths.artifacts_dir / "search",
        config.paths.artifacts_dir / "recsys",
        config.paths.artifacts_dir / "reports",
        config.paths.artifacts_dir / "registry",
        config.paths.logs_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
