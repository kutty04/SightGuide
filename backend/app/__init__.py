"""SightGuide Backend Application Package."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure backend root is on sys.path so 'app' is resolvable as a top-level package
_backend_dir = str(Path(__file__).resolve().parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

__version__ = "1.0.0"

