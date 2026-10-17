"""Live acceptance test against the running FastAPI backend."""

import urllib.request
import json
import uuid
from pathlib import Path

def test_api():
    boundary = uuid.uuid4().hex
    c_type = f"multipart/form-data; boundary={boundary}"
    
    img_path = Path("evaluations/test_portrait_woman.png")
    img_data = img_path.read_bytes()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="test_portrait_woman.png"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("latin1") + img_data + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="lambda_weight"\r\n\r\n'
        f"0.5\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="decoding_strategy"\r\n\r\n'
        f"sample\r\n"
        f"--{boundary}--\r\n"
    ).encode("latin1")

    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/caption/compare",
        data=body,
        headers={"Content-Type": c_type}
    )

    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("API HTTP STATUS:", resp.status)
        print("PROPOSED FUSED CAPTION:")
        print(" ", res.get("proposed_fusion", {}).get("fused_caption"))
        print("CONTROL FUSED CAPTION:")
        print(" ", res.get("control_fusion", {}).get("fused_caption"))
        print("PROPOSED PAIR:")
        p_pair = res.get("proposed_pair", {})
        print("  Anchor (c1*):", p_pair.get("rank_1_caption"), f"(score={p_pair.get('rank_1_clip_score')})")
        print("  Detail (c2*):", p_pair.get("rank_2_caption"), f"(score={p_pair.get('rank_2_clip_score')})")
        print("ALL CANDIDATES:")
        for c in res.get("candidates", []):
            print("  -", c.get("caption"), f"(score={c.get('clip_score')})")

if __name__ == "__main__":
    test_api()
