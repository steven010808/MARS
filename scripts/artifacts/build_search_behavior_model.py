from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mars.config import load_config  # noqa: E402
from mars.search.behavior_model import (  # noqa: E402
    behavior_model_summary,
    build_query_behavior_model_payload,
    write_query_behavior_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a train-only historical query behavior model for search serving."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", default="full")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--feedback",
        default="",
        help="Optional live search feedback parquet/csv/jsonl to merge into the train-only model.",
    )
    args = parser.parse_args()

    config = load_config(args.config, mode=args.mode)
    output_path = (
        Path(args.output)
        if args.output
        else config.paths.artifacts_dir / "search" / "query_behavior_model.json.gz"
    )
    payload = build_query_behavior_model_payload(
        config,
        feedback_path=args.feedback or None,
    )
    write_query_behavior_model(payload, output_path)
    print(json.dumps(behavior_model_summary(payload, output_path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
