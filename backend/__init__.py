"""SightGuide Backend Package."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure backend root is on sys.path so 'app' is accessible as a top-level package
# regardless of whether the process was started from project root or backend dir.
_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
