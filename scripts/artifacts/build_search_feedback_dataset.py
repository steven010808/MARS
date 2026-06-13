from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mars.config import load_config  # noqa: E402
from mars.search.feedback import (  # noqa: E402
    build_search_feedback_frame,
    feedback_summary,
    read_event_log,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert live API events into a search feedback dataset."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", default="full")
    parser.add_argument("--logs", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--validation-ratio", type=float, default=None)
    parser.add_argument("--no-negatives", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, mode=args.mode)
    raw_search = config.raw.get("search", {}) if isinstance(config.raw, dict) else {}
    online = raw_search.get("online_learning", {})
    online = online if isinstance(online, dict) else {}
    logs_path = Path(args.logs) if args.logs else config.paths.logs_dir / "api_events.jsonl"
    default_feedback_path = online.get("feedback_path") or (
        config.paths.processed_dir / "search_feedback.parquet"
    )
    output_path = Path(args.output) if args.output else Path(str(default_feedback_path))
    report_path = (
        Path(args.report)
        if args.report
        else config.paths.artifacts_dir / "reports" / "search_feedback_dataset.json"
    )
    validation_ratio = (
        float(args.validation_ratio)
        if args.validation_ratio is not None
        else float(online.get("validation_ratio", 0.2) or 0.2)
    )
    catalog = _catalog_products(config)
    events = read_event_log(logs_path)
    frame = build_search_feedback_frame(
        events,
        catalog_products=catalog,
        validation_ratio=validation_ratio,
        include_unclicked_negatives=not args.no_negatives,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    summary = {
        **feedback_summary(frame),
        "logs_path": str(logs_path),
        "output": str(output_path),
        "validation_ratio": validation_ratio,
        "catalog_products": len(catalog),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _catalog_products(config) -> set[str]:
    for path in (
        config.paths.artifacts_dir / "search" / "product_meta.parquet",
        config.paths.processed_dir / "products.parquet",
    ):
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["product_id"])
        return set(frame["product_id"].astype(str))
    return set()


if __name__ == "__main__":
    raise SystemExit(main())
