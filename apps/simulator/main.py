"""Container entrypoint for the live customer-behavior simulator."""

from __future__ import annotations

from scripts.runtime.live_simulator_loop import main

if __name__ == "__main__":
    raise SystemExit(main())
