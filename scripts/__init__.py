"""Command-line helpers for MARS.

The project uses a src-layout, so local `python -m scripts...` commands add
`src/` to `sys.path` before submodules import the `mars` package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
