from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mars.data.raw_simulator.config import load_config

REQUIRED_COLUMNS = ["event_id", "session_id", "event_type", "timestamp"]


def _get_output_format(config: dict[str, Any]) -> str:
    return str(config["simulator"].get("output_format", "csv"))


def _build_default_filename(stem: str, output_format: str) -> str:
    return f"{stem}.{output_format}"


def _get_raw_data_dir(config: dict[str, Any]) -> Path:
    simulator_output_cfg = config["simulator"].get("output", {})
    raw_dir = simulator_output_cfg.get("raw_dir")

    if raw_dir:
        return Path(raw_dir)

    return Path(config["paths"]["data_dir"]) / "raw"


def _get_processed_data_dir(config: dict[str, Any]) -> Path:
    simulator_output_cfg = config["simulator"].get("output", {})
    processed_dir = simulator_output_cfg.get("processed_dir")

    if processed_dir:
        return Path(processed_dir)

    return Path(config["paths"]["data_dir"]) / "processed"


def _resolve_events_path(config: dict[str, Any]) -> Path:
    raw_dir = _get_raw_data_dir(config)
    simulator_output_cfg = config["simulator"].get("output", {})
    output_format = _get_output_format(config)

    events_file = simulator_output_cfg.get("events_file")
    if events_file:
        return raw_dir / events_file

    return raw_dir / _build_default_filename("events", output_format)


def _resolve_processed_paths(config: dict[str, Any]) -> dict[str, Path]:
    processed_dir = _get_processed_data_dir(config)
    output_format = _get_output_format(config)

    return {
        "processed_dir": processed_dir,
        "train": processed_dir / _build_default_filename("train_events", output_format),
        "valid": processed_dir / _build_default_filename("valid_events", output_format),
        "test": processed_dir / _build_default_filename("test_events", output_format),
    }


def _validate_input_events(events_df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in events_df.columns]
    if missing:
        raise ValueError(f"events.csv is missing required columns: {missing}")

    if events_df.empty:
        raise ValueError("events.csv is empty.")

    parsed_ts = pd.to_datetime(events_df["timestamp"], errors="coerce")
    if parsed_ts.isna().any():
        bad_count = int(parsed_ts.isna().sum())
        raise ValueError(f"Found {bad_count} invalid timestamp values in events.csv")


def _save_dataframe(df: pd.DataFrame, path: Path, output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "csv":
        df.to_csv(path, index=False)
        return

    raise ValueError(f"Unsupported output format: {output_format}")


def _format_range(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"

    start_ts = pd.to_datetime(df["timestamp"]).min()
    end_ts = pd.to_datetime(df["timestamp"]).max()
    return f"{start_ts.isoformat()} ~ {end_ts.isoformat()}"


def time_based_session_split(
    events_df: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratio_sum = train_ratio + valid_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.6f}")

    events_df = events_df.copy()
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
    events_df = events_df.sort_values(["timestamp", "event_id"]).reset_index(drop=True)

    session_summary = (
        events_df.groupby("session_id", as_index=False)
        .agg(
            session_start=("timestamp", "min"),
            session_end=("timestamp", "max"),
            event_count=("event_id", "count"),
        )
        .sort_values(["session_start", "session_id"])
        .reset_index(drop=True)
    )

    total_events = int(session_summary["event_count"].sum())
    train_target = total_events * train_ratio
    valid_target = total_events * valid_ratio

    session_summary["cumulative_events"] = session_summary["event_count"].cumsum()

    train_session_ids = session_summary.loc[
        session_summary["cumulative_events"] <= train_target, "session_id"
    ].tolist()

    remaining_after_train = session_summary.loc[
        ~session_summary["session_id"].isin(train_session_ids)
    ].copy()

    remaining_after_train["remaining_cumulative"] = remaining_after_train["event_count"].cumsum()

    valid_session_ids = remaining_after_train.loc[
        remaining_after_train["remaining_cumulative"] <= valid_target, "session_id"
    ].tolist()

    assigned_train = set(train_session_ids)
    assigned_valid = set(valid_session_ids)

    if not assigned_train and not session_summary.empty:
        first_session = str(session_summary.iloc[0]["session_id"])
        assigned_train.add(first_session)
        assigned_valid.discard(first_session)

    unassigned_session_ids = [
        sid
        for sid in session_summary["session_id"].tolist()
        if sid not in assigned_train and sid not in assigned_valid
    ]

    if not assigned_valid and len(unassigned_session_ids) > 1:
        assigned_valid.add(unassigned_session_ids[0])
        unassigned_session_ids = unassigned_session_ids[1:]

    assigned_test = set(unassigned_session_ids)

    # test가 비어버리는 극단적인 소규모 상황 방지
    if not assigned_test and assigned_valid:
        moved = sorted(assigned_valid)[-1]
        assigned_valid.remove(moved)
        assigned_test.add(moved)

    train_df = events_df[events_df["session_id"].isin(assigned_train)].copy()
    valid_df = events_df[events_df["session_id"].isin(assigned_valid)].copy()
    test_df = events_df[events_df["session_id"].isin(assigned_test)].copy()

    train_df = train_df.sort_values(["timestamp", "event_id"]).reset_index(drop=True)
    valid_df = valid_df.sort_values(["timestamp", "event_id"]).reset_index(drop=True)
    test_df = test_df.sort_values(["timestamp", "event_id"]).reset_index(drop=True)

    return train_df, valid_df, test_df


def print_split_summary(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    total_events = len(train_df) + len(valid_df) + len(test_df)
    total_sessions = (
        train_df["session_id"].nunique()
        + valid_df["session_id"].nunique()
        + test_df["session_id"].nunique()
    )

    print("\n[Split summary]")
    print(f"total_events: {total_events}")
    print(f"total_sessions: {total_sessions}")

    for name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        event_ratio = (len(df) / total_events) if total_events > 0 else 0.0
        print(f"\n[{name}]")
        print(f"events: {len(df)} ({event_ratio:.3f})")
        print(f"sessions: {df['session_id'].nunique()}")
        print(f"time_range: {_format_range(df)}")

        if not df.empty:
            print(df["event_type"].value_counts())


def run_time_based_split() -> dict[str, Any]:
    config = load_config()
    split_cfg = config["simulator"]["split"]
    output_format = _get_output_format(config)

    events_path = _resolve_events_path(config)
    if not events_path.exists():
        raise FileNotFoundError(f"events file not found: {events_path}")

    events_df = pd.read_csv(events_path)
    _validate_input_events(events_df)

    train_df, valid_df, test_df = time_based_session_split(
        events_df=events_df,
        train_ratio=float(split_cfg["train_ratio"]),
        valid_ratio=float(split_cfg["valid_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
    )

    paths = _resolve_processed_paths(config)

    _save_dataframe(train_df, paths["train"], output_format)
    _save_dataframe(valid_df, paths["valid"], output_format)
    _save_dataframe(test_df, paths["test"], output_format)

    print_split_summary(train_df, valid_df, test_df)

    return {
        "events_path": str(events_path),
        "train_path": str(paths["train"]),
        "valid_path": str(paths["valid"]),
        "test_path": str(paths["test"]),
        "train_events": len(train_df),
        "valid_events": len(valid_df),
        "test_events": len(test_df),
        "train_sessions": int(train_df["session_id"].nunique()),
        "valid_sessions": int(valid_df["session_id"].nunique()),
        "test_sessions": int(test_df["session_id"].nunique()),
    }


if __name__ == "__main__":
    result = run_time_based_split()
    print("\n[Simulator Day5 Split]", result)
