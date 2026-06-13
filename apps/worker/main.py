"""Container entrypoint for metric monitoring and model registry updates."""

from __future__ import annotations

from scripts.runtime.worker_loop import main

if __name__ == "__main__":
    raise SystemExit(main())
