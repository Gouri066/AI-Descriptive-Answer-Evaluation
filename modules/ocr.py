"""OCR module: crops answer regions and runs Tesseract (Phase 6).

For text questions  (fill_blank, numerical, shape):
  load gray image → crop region → upscale → Tesseract → text + confidence

For matching questions:
  load gray image → crop region → OpenCV line detection → build "1-C,2-A,3-B"
  (Primary school children draw lines, not write text answers.)
  Falls back to text OCR if no lines are detected.
"""
import os
import re

import cv2
import numpy as np
import pytesseract

# ── Tesseract path (Windows installer default) ──────────────────────────────
_TESS_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_TESS_EXE):
    pytesseract.pytesseract.tesseract_cmd = _TESS_EXE

# ── Tesseract configs ────────────────────────────────────────────────────────
_TEXT_CFG = (
    "--psm 7 --oem 3 "
    "-c tessedit_char_whitelist="
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-., "
)
_NUM_CFG  = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789."

# Answers below this confidence → flagged for teacher review
CONFIDENCE_THRESHOLD = 50


# ── Internal helpers ─────────────────────────────────────────────────────────

def _load_gray(path):
    """Load image as grayscale (handles Windows paths with spaces)."""
    raw = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)


def _crop_region(image, rx, ry, rw, rh):
    """Crop a rectangular region. Coordinates are fractions of image size."""
    h, w = image.shape[:2]
    x1 = max(0, int(rx * w))
    y1 = max(0, int(ry * h))
    x2 = min(w, int((rx + rw) * w))
    y2 = min(h, int((ry + rh) * h))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _run_tesseract(crop, question_type):
    """Run Tesseract on a crop. Returns (text, confidence 0-100)."""
    if crop is None or crop.size == 0:
        return "", 0.0

    # Upscale small crops — Tesseract reads better at ≥80 px height
    h, w = crop.shape[:2]
    if min(h, w) < 80:
        scale = max(2.0, 80 / min(h, w))
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)

    # White border so letters at edges are not clipped
    crop = cv2.copyMakeBorder(crop, 20, 20, 20, 20,
                               cv2.BORDER_CONSTANT, value=255)

    cfg = _NUM_CFG if question_type == "numerical" else _TEXT_CFG

    try:
        data = pytesseract.image_to_data(
            crop, config=cfg,
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return "", 0.0

    words, confs = [], []
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        conf = int(conf)
        if text and conf >= 0:
            words.append(text)
            if conf > 0:
                confs.append(conf)

    recognized = " ".join(words).strip()
    avg_conf   = float(sum(confs) / len(confs)) if confs else 0.0
    return recognized, round(avg_conf, 1)


# ── Matching: line detection ──────────────────────────────────────────────────

def _detect_lines_in_crop(crop, num_pairs):
    """Use OpenCV to find hand-drawn lines connecting left items to right items.

    How it works
    ------------
    1. Binarize and invert the crop so ink is bright (255) on dark (0).
    2. Dilate slightly to join broken pen strokes.
    3. Run Canny edge detection to find ink boundaries.
    4. Run HoughLinesP to find line segments.
    5. Keep only segments that cross from the left half to the right half
       (these are the connecting lines; text strokes are short and stay on one side).
    6. Map each line's Y-positions to row indices on left and right columns.

    Returns
    -------
    dict  {left_row_0indexed: right_row_0indexed}
    Empty dict if no lines detected.
    """
    if crop is None or crop.size == 0:
        return {}

    h, w = crop.shape[:2]
    if h < 10 or w < 10:
        return {}

    # ── Binarize + invert ────────────────────────────────────────────────────
    _, binary = cv2.threshold(crop, 150, 255, cv2.THRESH_BINARY)
    inv = cv2.bitwise_not(binary)          # ink = 255 (bright)

    # Dilate to connect broken pen strokes (children draw shaky lines)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(inv, kernel, iterations=1)

    # ── Edge detection ────────────────────────────────────────────────────────
    edges = cv2.Canny(dilated, threshold1=30, threshold2=100)

    # ── Probabilistic Hough Line Transform ────────────────────────────────────
    # minLineLength: line must span ≥ 20 % of the crop width (filters out text)
    min_len = max(30, int(w * 0.20))
    lines = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi / 180,
        threshold=15,
        minLineLength=min_len,
        maxLineGap=20,
    )

    if lines is None:
        return {}

    mid_x      = w / 2
    row_height = h / num_pairs   # assumed even spacing of items

    mapping = {}

    for seg in lines:
        x1, y1, x2, y2 = seg[0]

        # Keep only lines that cross the midline (left → right)
        if x1 < mid_x and x2 > mid_x:
            left_y, right_y = y1, y2
        elif x2 < mid_x and x1 > mid_x:
            left_y, right_y = y2, y1
        else:
            continue   # both endpoints on same side → text or decoration

        # Row indices (0-based, clamped to valid range)
        left_row  = max(0, min(num_pairs - 1, int(left_y  / row_height)))
        right_row = max(0, min(num_pairs - 1, int(right_y / row_height)))

        # First detected line for each left row wins
        if left_row not in mapping:
            mapping[left_row] = right_row

    return mapping


def _mapping_to_text(mapping, left_labels, right_labels_sorted):
    """Convert {0: 2, 1: 0, 2: 1} → '1-C,2-A,3-B'.

    left_labels         e.g. ['1','2','3']  (order on the left column)
    right_labels_sorted e.g. ['A','B','C']  (alphabetical = top→bottom on sheet)
    """
    parts = []
    for left_row, right_row in sorted(mapping.items()):
        if left_row < len(left_labels) and right_row < len(right_labels_sorted):
            parts.append(f"{left_labels[left_row]}-{right_labels_sorted[right_row]}")
    return ",".join(parts)


def matching_from_lines(gray_image_path, question):
    """Detect matching answer by finding hand-drawn connecting lines.

    Falls back to text OCR if no crossing lines are found
    (handles the rare case where a student writes '1-C' instead of drawing).

    Returns the same dict shape as ocr_question().
    """
    rx = question["region_x"]
    ry = question["region_y"]
    rw = question["region_w"]
    rh = question["region_h"]

    if any(v is None for v in (rx, ry, rw, rh)):
        return {"recognized_text": "", "confidence": 0.0,
                "needs_review": True, "error": "no_region"}

    image = _load_gray(gray_image_path)
    if image is None:
        return {"recognized_text": "", "confidence": 0.0,
                "needs_review": True, "error": "image_not_found"}

    crop = _crop_region(image, rx, ry, rw, rh)
    if crop is None:
        return {"recognized_text": "", "confidence": 0.0, "needs_review": True}

    # Parse correct answer to get labels and pair count
    # e.g.  "1-C,2-A,3-B"  →  left=['1','2','3'],  right sorted=['A','B','C']
    correct = question["correct_answer"]
    raw_pairs = re.findall(r"([A-Za-z0-9]+)-([A-Za-z0-9]+)", correct)
    num_pairs = len(raw_pairs)

    if num_pairs == 0:
        return {"recognized_text": "", "confidence": 0.0, "needs_review": True}

    left_labels         = [p[0] for p in raw_pairs]
    right_labels_sorted = sorted(set(p[1] for p in raw_pairs))   # A < B < C

    # ── 1. Try line detection ─────────────────────────────────────────────────
    mapping = _detect_lines_in_crop(crop, num_pairs)

    if mapping:
        text = _mapping_to_text(mapping, left_labels, right_labels_sorted)
        detected = len(mapping)
        # Confidence: 100 % if all pairs found, proportionally less otherwise
        confidence = round((detected / num_pairs) * 90.0, 1)
        needs_review = detected < num_pairs
        return {
            "recognized_text": text,
            "confidence": confidence,
            "needs_review": needs_review,
            "method": "line_detection",
        }

    # ── 2. Fall back to text OCR (student wrote the answer as text) ───────────
    text, confidence = _run_tesseract(crop, "matching")
    needs_review = (confidence < CONFIDENCE_THRESHOLD) or (not text.strip())
    return {
        "recognized_text": text,
        "confidence": confidence,
        "needs_review": needs_review,
        "method": "ocr_fallback",
    }


# ── Public API ───────────────────────────────────────────────────────────────

def ocr_question(gray_image_path, question):
    """Full pipeline for one question (text types only).

    For matching questions, call matching_from_lines() instead.
    Returns dict: recognized_text, confidence, needs_review
    """
    rx = question["region_x"]
    ry = question["region_y"]
    rw = question["region_w"]
    rh = question["region_h"]

    if any(v is None for v in (rx, ry, rw, rh)):
        return {"recognized_text": "", "confidence": 0.0,
                "needs_review": True, "error": "no_region"}

    image = _load_gray(gray_image_path)
    if image is None:
        return {"recognized_text": "", "confidence": 0.0,
                "needs_review": True, "error": "image_not_found"}

    crop = _crop_region(image, rx, ry, rw, rh)
    text, confidence = _run_tesseract(crop, question["question_type"])

    needs_review = (confidence < CONFIDENCE_THRESHOLD) or (not text.strip())

    return {
        "recognized_text": text,
        "confidence": confidence,
        "needs_review": needs_review,
    }
