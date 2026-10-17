"""SightGuide OCR Extraction Service.

Uses rapidocr-onnxruntime for local, CPU-efficient, high-accuracy text detection
and recognition from photographs, screenshots, and document scans without external
binary dependencies.
"""

from __future__ import annotations

import io
import re
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from PIL import Image

try:
    from rapidocr_onnxruntime import RapidOCR
    _RAPIDOCR_AVAILABLE = True
except ImportError:
    _RAPIDOCR_AVAILABLE = False


def _compute_box_rect(box: Any) -> tuple[float, float, float, float]:
    """Compute axis-aligned bounding box (xmin, ymin, xmax, ymax) from polygon points."""
    try:
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        return min(xs), min(ys), max(xs), max(ys)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def _box_overlap(box1: Any, box2: Any) -> tuple[float, float, float, float]:
    """Compute (IoU, coverage, area1, area2) between two bounding boxes.

    Coverage is intersection / min(area1, area2).
    """
    x1_min, y1_min, x1_max, y1_max = _compute_box_rect(box1)
    x2_min, y2_min, x2_max, y2_max = _compute_box_rect(box2)

    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    inter_w = max(0.0, xi_max - xi_min)
    inter_h = max(0.0, yi_max - yi_min)
    inter_area = inter_w * inter_h

    area1 = max(1.0, (x1_max - x1_min) * (y1_max - y1_min))
    area2 = max(1.0, (x2_max - x2_min) * (y2_max - y2_min))

    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    coverage = inter_area / min(area1, area2) if min(area1, area2) > 0 else 0.0
    return iou, coverage, area1, area2


class OCRService:
    _instance: Optional[OCRService] = None

    def __init__(self) -> None:
        self.engine = None
        self.is_available = False
        if _RAPIDOCR_AVAILABLE:
            try:
                self.engine = RapidOCR()
                self.is_available = True
                print("[OCRService] RapidOCR ONNX engine initialized successfully.")
            except Exception as exc:
                print(f"[OCRService] Failed to initialize RapidOCR: {exc}")
                self.is_available = False
        else:
            print("[OCRService] rapidocr-onnxruntime is not installed.")

    @classmethod
    def get_instance(cls) -> OCRService:
        if cls._instance is None:
            cls._instance = OCRService()
        return cls._instance

    def extract_text(self, image_bytes: bytes) -> dict[str, Any]:
        """Extract visible text from image bytes with multi-pass screen and orientation support.

        Returns:
            dict containing:
            - text: concatenated text string
            - has_text: bool
            - lines: list of line dicts (text, confidence, box)
            - execution_time_seconds: float
            - engine: str
        """
        start_time = time.time()
        if not self.is_available or self.engine is None:
            return {
                "text": "",
                "has_text": False,
                "lines": [],
                "error": "OCR engine is not available.",
                "execution_time_seconds": round(time.time() - start_time, 3),
                "engine": "none"
            }

        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            np_img = np.array(pil_img)
            np_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

            # Pass 1: Standard full-image OCR pass without direction-classifier inversion
            # (prevents upright casual handwriting like 'Cat' being falsely flipped 180° into '0')
            std_results, _ = self.engine(np_bgr, use_cls=False)
            merged_items: list[list[Any]] = []
            if std_results:
                for r in std_results:
                    t = str(r[1]).strip()
                    if t:
                        merged_items.append([r[0], t, float(r[2])])

            # Pass 2: Adaptive DBNet region inspection for faint/screen handwriting
            old_box_th = getattr(self.engine.text_det.postprocess_op, "box_thresh", 0.5)
            old_th = getattr(self.engine.text_det.postprocess_op, "thresh", 0.3)
            try:
                self.engine.text_det.postprocess_op.box_thresh = 0.35
                self.engine.text_det.postprocess_op.thresh = 0.25
                dt_boxes, _ = self.engine.text_det(np_bgr)
                if dt_boxes is None or len(dt_boxes) == 0:
                    dt_boxes, _ = self.engine.text_det(cv2.bitwise_not(np_bgr))
            finally:
                self.engine.text_det.postprocess_op.box_thresh = old_box_th
                self.engine.text_det.postprocess_op.thresh = old_th

            if dt_boxes is not None and len(dt_boxes) > 0:
                for box in dt_boxes:
                    p0, p1, p2, p3 = box
                    w = max(np.linalg.norm(p0 - p1), np.linalg.norm(p2 - p3))
                    h = max(np.linalg.norm(p0 - p3), np.linalg.norm(p1 - p2))

                    is_tall = (h > 1.1 * w)
                    is_covered = False
                    covered_conf = 1.0
                    for item in merged_items:
                        iou, cov, _, _ = _box_overlap(box, item[0])
                        if iou > 0.4 or cov > 0.6:
                            is_covered = True
                            covered_conf = item[2]
                            break

                    # Only re-analyze if candidate region was missed, is stacked/tall, or has low confidence
                    if not (is_tall or not is_covered or covered_conf < 0.78):
                        continue

                    crop_w = int(w)
                    crop_h = int(h)
                    if crop_w < 12 or crop_h < 12:
                        continue

                    pts_std = np.array(
                        [[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]],
                        dtype=np.float32
                    )
                    M = cv2.getPerspectiveTransform(box.astype(np.float32), pts_std)
                    M_inv = np.linalg.inv(M)
                    crop = cv2.warpPerspective(np_bgr, M, (crop_w, crop_h))


                    # Run OCR on rectified crop without direction inversion
                    crop_res, _ = self.engine(crop, use_cls=False)
                    if not crop_res:
                        crop_res, _ = self.engine(cv2.bitwise_not(crop), use_cls=False)

                    # Doodle / symbol separation across clean vertical projection gaps
                    # (e.g. 'HELLO HI ♡' where HI is separated by an empty vertical gap from the heart doodle)
                    is_doodle_split = False
                    if crop_w >= 40 and crop_h >= 20:
                        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        _, th_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        if np.mean(th_crop) > 127:
                            th_crop = cv2.bitwise_not(th_crop)
                        col_sums = np.sum(th_crop > 0, axis=0)
                        gaps = []
                        in_gap = False
                        g_start = 0
                        for col in range(int(0.2 * crop_w), int(0.85 * crop_w)):
                            if col_sums[col] <= max(1, 0.03 * crop_h):
                                if not in_gap:
                                    in_gap = True
                                    g_start = col
                            else:
                                if in_gap:
                                    in_gap = False
                                    if col - g_start >= 8:
                                        gaps.append((g_start, col))
                        if in_gap and (int(0.85 * crop_w) - g_start >= 8):
                            gaps.append((g_start, int(0.85 * crop_w)))
                        if gaps:
                            best_gap = max(gaps, key=lambda g: g[1] - g[0])
                            left_seg = crop[:, :best_gap[0]]
                            left_padded = cv2.copyMakeBorder(left_seg, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                            r_left, _ = self.engine(left_padded, use_cls=False)
                            if not r_left:
                                r_left, _ = self.engine(cv2.bitwise_not(left_padded), use_cls=False)
                            if r_left:
                                clean_sub = [x for x in r_left if float(x[2]) > 0.5 and len(str(x[1]).strip()) >= 2]
                                if clean_sub:
                                    crop_res = []
                                    is_doodle_split = True
                                    for item in clean_sub:
                                        adj_box = [[max(0.0, pt[0] - 10.0), max(0.0, pt[1] - 10.0)] for pt in item[0]]
                                        crop_res.append([adj_box, item[1], item[2]])

                    # Vertical separation for bottom doodles (e.g. 'Maddy ♡' where heart is below letters)
                    if (not crop_res or (crop_res and len(crop_res) == 1 and len(crop_res[0][1]) <= 1)) and (crop_h > 0.6 * crop_w):
                        top_h = int(crop_h * 0.65)
                        top_crop = crop[:top_h, :]
                        top_padded = cv2.copyMakeBorder(top_crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                        r_top, _ = self.engine(top_padded, use_cls=False)
                        if not r_top:
                            r_top, _ = self.engine(cv2.bitwise_not(top_padded), use_cls=False)
                        if r_top:
                            clean_top = [x for x in r_top if float(x[2]) > 0.5 and len(str(x[1]).strip()) >= 2]
                            if clean_top:
                                crop_res = []
                                for item in clean_top:
                                    adj_box = [[max(0.0, pt[0] - 12.0), max(0.0, pt[1] - 12.0)] for pt in item[0]]
                                    crop_res.append([adj_box, item[1], item[2]])

                    if crop_res:
                        for sub_item in crop_res:
                            sub_box = np.array(sub_item[0], dtype=np.float32)
                            orig_sub_box = cv2.perspectiveTransform(
                                sub_box.reshape(-1, 1, 2), M_inv
                            ).reshape(-1, 2).tolist()
                            sub_text = str(sub_item[1]).strip()
                            sub_conf = float(sub_item[2])
                            if not sub_text:
                                continue

                            # Deduplicate / collision check
                            collision_idx = -1
                            for idx, exist_item in enumerate(merged_items):
                                iou, cov, _, _ = _box_overlap(exist_item[0], orig_sub_box)
                                if iou > 0.35 or cov > 0.6:
                                    collision_idx = idx
                                    break

                            if collision_idx >= 0:
                                exist = merged_items[collision_idx]
                                exist_text = exist[1]
                                exist_conf = exist[2]

                                # 1. Clean doodle split: replace merged doodle token (e.g. 'Hi' replacing merged 'HiO')
                                if is_doodle_split and sub_conf > 0.80:
                                    merged_items[collision_idx] = [orig_sub_box, sub_text, sub_conf]
                                # 2. Never replace a multi-char word (len >= 3) with a single-letter fragment (len == 1)
                                elif len(exist_text) >= 3 and len(sub_text) == 1:
                                    pass
                                # 3. Never replace a longer word with a shorter subslice if existing conf is solid (> 0.70)
                                elif len(exist_text) > len(sub_text) and exist_conf > 0.70:
                                    pass
                                # 4. Standard replacement when new detection has higher confidence
                                elif sub_conf > exist_conf:
                                    merged_items[collision_idx] = [orig_sub_box, sub_text, sub_conf]
                            else:
                                merged_items.append([orig_sub_box, sub_text, sub_conf])

            elapsed = round(time.time() - start_time, 3)
            if not merged_items:
                return {
                    "text": "",
                    "normalized_text": None,
                    "has_text": False,
                    "lines": [],
                    "message": "No readable text detected in this image.",
                    "execution_time_seconds": elapsed,
                    "engine": "rapidocr-onnxruntime-cpu"
                }

            formatted_text, lines = sort_and_format_ocr_results(merged_items)
            normalized_text, lines = contextually_normalize_ocr(formatted_text, lines, img_h=np_bgr.shape[0])

            return {
                "text": formatted_text,
                "normalized_text": normalized_text,
                "has_text": len(lines) > 0,
                "lines": lines,
                "execution_time_seconds": elapsed,
                "engine": "rapidocr-onnxruntime-cpu"
            }
        except Exception as exc:
            return {
                "text": "",
                "normalized_text": None,
                "has_text": False,
                "lines": [],
                "error": f"OCR processing failed: {exc}",
                "execution_time_seconds": round(time.time() - start_time, 3),
                "engine": "rapidocr-onnxruntime-cpu"
            }



def sort_and_format_ocr_results(raw_items: list) -> tuple[str, list[dict[str, Any]]]:
    """Sort OCR detections into natural reading order and format distinct text regions.

    1. Extracts bounding geometry (ymin, ymax, xmin, xmax, y_center, height).
    2. Groups words/boxes into horizontal lines using adaptive vertical overlap.
    3. Sorts lines top-to-bottom, and boxes within each line left-to-right.
    4. Separates distinct text regions / paragraphs when vertical gap exceeds 1.5x line height.
    5. Normalizes whitespace and assigns reading_order (1-indexed).
    """
    parsed = []
    for item in raw_items:
        # RapidOCR format: [box, text, confidence]
        box = item[0]
        text = str(item[1]).strip()
        conf = round(float(item[2]), 4)
        if not text:
            continue
        try:
            ys = [pt[1] for pt in box]
            xs = [pt[0] for pt in box]
            ymin, ymax = min(ys), max(ys)
            xmin, xmax = min(xs), max(xs)
            y_center = (ymin + ymax) / 2.0
            height = max(1.0, ymax - ymin)
        except Exception:
            ymin, ymax, xmin, xmax, y_center, height = 0.0, 0.0, 0.0, 0.0, 0.0, 10.0

        parsed.append({
            "text": text,
            "confidence": conf,
            "box": box,
            "ymin": ymin,
            "ymax": ymax,
            "xmin": xmin,
            "xmax": xmax,
            "y_center": y_center,
            "height": height
        })

    if not parsed:
        return "", []

    # Sort primarily by vertical coordinate
    parsed.sort(key=lambda item: item["y_center"])

    # Cluster into horizontal lines
    lines: list[list[dict[str, Any]]] = []
    for item in parsed:
        placed = False
        for line in lines:
            line_avg_y = sum(w["y_center"] for w in line) / len(line)
            line_avg_h = sum(w["height"] for w in line) / len(line)
            # If vertical distance is within 45% of average line height, same line
            if abs(item["y_center"] - line_avg_y) <= max(line_avg_h * 0.45, 6.0):
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])

    # Sort lines vertically, and within each line sort tokens left-to-right
    lines.sort(key=lambda line: sum(w["y_center"] for w in line) / len(line))
    for line in lines:
        line.sort(key=lambda w: w["xmin"])

    # Build structured line entries and formatted text with paragraph breaks
    structured_lines: list[dict[str, Any]] = []
    text_paragraphs: list[str] = []
    current_para_lines: list[str] = []
    prev_line_bottom = None
    prev_line_height = None

    order_idx = 1
    for line in lines:
        line_text = " ".join(w["text"] for w in line)
        line_conf = round(sum(w["confidence"] for w in line) / len(line), 4)
        if len(line) == 1 and line[0].get("box"):
            outer_box = line[0]["box"]
        else:
            outer_box = [
                [min(w["xmin"] for w in line), min(w["ymin"] for w in line)],
                [max(w["xmax"] for w in line), min(w["ymin"] for w in line)],
                [max(w["xmax"] for w in line), max(w["ymax"] for w in line)],
                [min(w["xmin"] for w in line), max(w["ymax"] for w in line)],
            ]
        structured_lines.append({
            "text": line_text,
            "confidence": line_conf,
            "box": outer_box,
            "reading_order": order_idx
        })
        order_idx += 1

        line_top = min(w["ymin"] for w in line)
        line_bottom = max(w["ymax"] for w in line)
        line_h = max(1.0, line_bottom - line_top)

        if prev_line_bottom is not None and prev_line_height is not None:
            vertical_gap = line_top - prev_line_bottom
            # If gap between lines is greater than 1.5x previous line height, start new paragraph
            if vertical_gap > prev_line_height * 1.5:
                if current_para_lines:
                    text_paragraphs.append("\n".join(current_para_lines))
                    current_para_lines = []

        current_para_lines.append(line_text)
        prev_line_bottom = line_bottom
        prev_line_height = line_h

    if current_para_lines:
        text_paragraphs.append("\n".join(current_para_lines))

    formatted_text = "\n\n".join(text_paragraphs).strip()
    return formatted_text, structured_lines


def contextually_normalize_ocr(
    raw_formatted_text: str,
    lines: list[dict[str, Any]],
    img_h: Optional[int] = None,
) -> tuple[Optional[str], list[dict[str, Any]]]:
    """Conservative, evidence-based contextual normalization.

    Preserves:
    - Raw predictions in lines[i]['text'] and raw_formatted_text.
    - CJK documents/signs (strictly guarded by Latin ratio > 75% and <= 2 CJK characters).
    - Arbitrary proper nouns / names (never replaces Modu with Maddy, never replaces caddy).

    Normalizes (in lines[i]['normalized_text']):
    - H1 in greeting context -> Hi
    - Isolated CJK homoglyphs (e.g. 工'm -> I'm, or isolated 工 -> I in greeting context)
    - Handwriting pronoun contractions Z'm, T'm, 1'm -> I'm (or Tm following greeting -> Im)
    - Trailing digit 7 in uppercase words -> Y (e.g. DD7 -> DDY)
    - Omits top status bar lines (ymin / img_h < 0.05 matching time/battery/signal) from normalized_text when content lines exist below.
    """
    if not lines or not raw_formatted_text.strip():
        return None, lines

    full_text = " ".join(l.get("text", "") for l in lines)
    total_chars = len([c for c in full_text if not c.isspace()])
    if total_chars == 0:
        return None, lines

    latin_chars = len([c for c in full_text if ('a' <= c <= 'z') or ('A' <= c <= 'Z') or ('0' <= c <= '9')])
    cjk_chars = len([c for c in full_text if '\u4e00' <= c <= '\u9fff'])

    # Strict safeguard: Only allow homoglyph correction if text is predominantly Latin
    allow_cjk_homoglyphs = (cjk_chars <= 2 and (latin_chars / max(1, total_chars)) > 0.75)

    has_greeting = False
    norm_lines_text = []

    # Check if there are content lines below top 15% (screen noise check)
    has_content_below_top = False
    if img_h and img_h > 0:
        for l in lines:
            ymin = None
            if l.get("box"):
                ymin = min(pt[1] for pt in l["box"])
            if ymin is not None and (ymin / img_h) >= 0.15:
                has_content_below_top = True
                break

    for l in lines:
        raw_t = l.get("text", "").strip()
        t = raw_t

        ymin = None
        if l.get("box"):
            ymin = min(pt[1] for pt in l["box"])

        # Check if line is top-margin phone OS status bar or app action toolbar icon
        is_screen_noise = False
        if has_content_below_top and img_h and ymin is not None:
            rel_y = ymin / img_h
            if rel_y < 0.05:
                # Matches clock, battery, wifi/signal icons
                if re.match(r'^\d{1,2}:\d{2}', t) or re.match(r'^[0-9]{1,3}$', t) or t in ['三', 'll', '√', 'J', 'V', '6']:
                    is_screen_noise = True
            elif rel_y < 0.12:
                # App action bar toolbar icons (tick mark, chevron, isolated glyphs)
                if len(t) <= 2 and (t in ['√', 'J', 'V', '1', '6', 'l', '三', 'O'] or not t.isalnum()):
                    is_screen_noise = True

        if is_screen_noise:
            # Omit screen interface noise from normalized output, but raw line.text remains
            l["normalized_text"] = None
            continue

        if re.search(r'\b(hello|hi|hey|dear)\b', t, re.IGNORECASE):
            has_greeting = True

        # Rule 1: H1 -> Hi (digit 1 misrecognized for lowercase i in greeting)
        t = re.sub(r'\bH1\b', 'Hi', t)

        # Rule 2: CJK homoglyphs for 'I' or "I'm"
        if allow_cjk_homoglyphs:
            t = re.sub(r'工[\'\’]?m\b', "I'm", t)
            if has_greeting and t == '工':
                t = 'I'

        # Rule 3: Pronoun contractions Z'm, T'm, 1'm -> I'm (or Tm following greeting -> Im)
        t = re.sub(r'\b[ZT1][\'\’]m\b', "I'm", t)
        if has_greeting or t == 'Tm':
            t = re.sub(r'\bTm\b', 'Im', t)

        # Rule 4: Trailing digit 7 in uppercase words: e.g. DD7 -> DDY
        t = re.sub(r'\b([A-Z]{2,})7\b', r'\g<1>Y', t)

        # Record normalized_text on line if changed or normalized
        l["normalized_text"] = t
        norm_lines_text.append(t)

    normalized_full_text = "\n".join(norm_lines_text).strip() if norm_lines_text else raw_formatted_text
    return normalized_full_text, lines

