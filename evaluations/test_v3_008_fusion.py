import sys
import os
from pathlib import Path

_env_file = Path(".env")
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        if "=" in _line and not _line.startswith("#"):
            k, v = _line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

sys.path.insert(0, "backend")
from app.services.fusion_service import fuse_caption_pair

c1 = "a minimal illustration of the logo for downtown cafe"
c2 = "a minimalist poster with the words down on the screen"
res = fuse_caption_pair(c1, c2)
print("STATUS:", res.generation_status)
print("FUSED:", repr(res.fused_caption))
print("ERROR:", repr(res.error_details))
