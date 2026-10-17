"""Live OCR test against running backend."""

import urllib.request
import json
import io
import uuid
from PIL import Image, ImageDraw

def test_ocr(has_text=True):
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    if has_text:
        draw = ImageDraw.Draw(img)
        draw.text((20, 35), "SightGuide OCR Live Test", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    boundary = uuid.uuid4().hex
    c_type = f"multipart/form-data; boundary={boundary}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="test.png"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("latin1") + data + f"\r\n--{boundary}--\r\n".encode("latin1")

    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/ocr/extract",
        data=body,
        headers={"Content-Type": c_type}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print(f"OCR (has_text={has_text}): HTTP {resp.status} | detected={res['has_text']} | text='{res['text']}'")

if __name__ == "__main__":
    test_ocr(True)
    test_ocr(False)
