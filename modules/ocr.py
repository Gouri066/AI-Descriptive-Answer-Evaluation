"""OCR module: crops answer regions and runs Tesseract with Gemini Vision fallback.

Pipeline for every question:
  1. Load gray image → crop region
  2. Apply 8 different image preprocessing strategies
  3. Run Tesseract on each variant (multiple PSM modes) → pick highest confidence
  4. If Tesseract confidence < threshold → call Gemini Vision API for smudged/bad handwriting
  5. Return best text + confidence

Shape questions:
  → Geometric detection first (circularity, vertices, ellipse fit)
  → Fall back to Tesseract / Gemini if geometric detection is unsure
"""
import base64
import os
import re

import cv2
import numpy as np
import pytesseract
from modules.shape_detection import detect_shape

# ── Tesseract path (Windows installer default) ──────────────────────────────
_TESS_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_TESS_EXE):
    pytesseract.pytesseract.tesseract_cmd = _TESS_EXE

# ── Tesseract configs (multiple PSM modes tried) ─────────────────────────────
_WHITELIST_TEXT = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-., "
)
_WHITELIST_NUM = "0123456789.-"

_TEXT_CFGS = [
    f"--psm 7 --oem 3 -c tessedit_char_whitelist={_WHITELIST_TEXT}",
    f"--psm 8 --oem 3 -c tessedit_char_whitelist={_WHITELIST_TEXT}",
    f"--psm 13 --oem 3 -c tessedit_char_whitelist={_WHITELIST_TEXT}",
    f"--psm 6 --oem 3 -c tessedit_char_whitelist={_WHITELIST_TEXT}",
]
# Backward-compatible alias (some modules may import this name)
_TEXT_CFG = _TEXT_CFGS[0]

_NUM_CFGS = [
    f"--psm 7 --oem 3 -c tessedit_char_whitelist={_WHITELIST_NUM}",
    f"--psm 8 --oem 3 -c tessedit_char_whitelist={_WHITELIST_NUM}",
    f"--psm 13 --oem 3 -c tessedit_char_whitelist={_WHITELIST_NUM}",
]

# If Tesseract confidence below this → try Gemini Vision
CONFIDENCE_THRESHOLD = 50
_GEMINI_FALLBACK_THRESHOLD = 70  # below this confidence → use Gemini


# ── Internal helpers ─────────────────────────────────────────────────────────

def _load_gray(path):
    """Load image as grayscale (handles Windows paths with spaces)."""
    raw = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)


def _crop_region(image, rx, ry, rw, rh):
    """Crop a rectangular region. Coordinates are fractions of image size."""
    h, w = image.shape[:2]
    # Add a small padding so we don't cut off edge strokes
    pad_x = int(rw * w * 0.02)
    pad_y = int(rh * h * 0.02)
    x1 = max(0, int(rx * w) - pad_x)
    y1 = max(0, int(ry * h) - pad_y)
    x2 = min(w, int((rx + rw) * w) + pad_x)
    y2 = min(h, int((ry + rh) * h) + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _tess_one(img_variant, cfg):
    """Run a single Tesseract config on one preprocessed image.
    Returns (text, confidence 0-100).
    """
    padded = cv2.copyMakeBorder(img_variant, 30, 30, 30, 30,
                                cv2.BORDER_CONSTANT, value=255)
    try:
        data = pytesseract.image_to_data(
            padded, config=cfg, output_type=pytesseract.Output.DICT)
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
    if not words:
        return "", 0.0
    recognized = " ".join(words).strip()
    avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
    return recognized, avg_conf


# Confidence at which we stop trying more variants (good enough)
_FAST_EXIT_CONF = 70.0


def _run_tesseract(crop, question_type):
    """Run Tesseract with a smart fast-path strategy.

    Strategy (ordered cheapest → most expensive):
      1. CLAHE + OTSU with PSM 7  → covers most clear handwriting
      2. Adaptive threshold        → covers smudged/uneven ink
      If either gives confidence >= 70: stop immediately.
      3-4. Denoise + OTSU, Sharpened OTSU (only if still low confidence)

    This gives 2 Tesseract calls in the common (good handwriting) case
    and up to 4 for hard cases — vs the old 32.
    """
    if crop is None or crop.size == 0:
        return "", 0.0
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return "", 0.0

    # Ensure grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()

    # Upscale: Tesseract works best at >= 150 px on short side
    short = min(h, w)
    if short < 150:
        scale = max(3.0, 150.0 / short)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)

    is_numeric = question_type in (
    "numerical",
    "addition",
    "subtraction",
    "multiplication",
    "division",
)
   
    # primary_cfg = _NUM_CFGS[0] if is_numeric else _TEXT_CFGS[0]
    # alt_cfg = _NUM_CFGS[1] if is_numeric else _TEXT_CFGS[1]
    
    if is_numeric:
        numeric_cfgs = [
            "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.- -c load_system_dawg=0 -c load_freq_dawg=0",
            "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789.- -c load_system_dawg=0 -c load_freq_dawg=0",
            "--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789.- -c load_system_dawg=0 -c load_freq_dawg=0",
        ]
        primary_cfg = numeric_cfgs[0]
        alt_cfg = numeric_cfgs[1]
    else:
        primary_cfg = _TEXT_CFGS[0]
        alt_cfg = _TEXT_CFGS[1]

    # Initialize these for BOTH numeric and text questions
    best_text = ""
    best_conf = 0.0


    def _update(text, conf):
        nonlocal best_text, best_conf
        if conf > best_conf or (conf == best_conf and len(text) > len(best_text)):
            best_text, best_conf = text, conf

    # ── Fast path: Variant 1 — CLAHE-enhanced + OTSU ─────────────────────
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, v1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t, c = _tess_one(v1, primary_cfg)
    print(f"[OCR TEST] {question_type} | CLAHE+OTSU: {t!r} | confidence={c}")
    _update(t, c)
    if best_conf >= _FAST_EXIT_CONF:
        return best_text, round(best_conf, 1)

    # ── Fast path: Variant 2 — Adaptive threshold (smudged ink) ──────────
    v2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, blockSize=21, C=10)
    t, c = _tess_one(v2, primary_cfg)
    print(f"[OCR TEST] {question_type} | ADAPTIVE: {t!r} | confidence={c}")
    _update(t, c)
    if best_conf >= _FAST_EXIT_CONF:
        return best_text, round(best_conf, 1)

    # ── Slow path: only reached for low-confidence / smudged answers ──────

    # Variant 3 — raw OTSU (baseline)
    print("[DEBUG OCR] Starting Variant 3", flush=True)
    _, v3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t, c = _tess_one(v3, alt_cfg)
    _update(t, c)
    if best_conf >= _FAST_EXIT_CONF:
        return best_text, round(best_conf, 1)

    # Variant 4 — Denoise → OTSU
    print("[DEBUG OCR] Starting Variant 4", flush=True)
    try:
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        _, v4 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t, c = _tess_one(v4, primary_cfg)
        _update(t, c)
    except Exception:
        pass

    return best_text, round(best_conf, 1)



# ── Gemini Vision fallback ───────────────────────────────────────────────────

def _crop_to_png_b64(crop):
    """Encode a crop to a base64 PNG string for the Gemini API."""
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _gemini_ocr(crop, question_type, correct_answer_hint=""):
    """Read handwritten answers using Gemini Vision."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip().strip('"')
    if not api_key or crop is None or crop.size == 0:
        return "", 0.0

    try:
        from google import genai
        from PIL import Image
        import io

        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            return "", 0.0

        pil_img = Image.open(io.BytesIO(buf.tobytes())).convert("RGB")

        prompt = (
            "Read the student's handwritten answer in this image. "
            "The handwriting may be messy, faint, or smudged. "
            "Read only what is visibly written. Do not guess from an answer key. "
            "Return only the answer text. If it cannot be read reliably, "
            "return UNREADABLE."
        )

        qtype = (question_type or "").strip().lower()
        if qtype in (
            "numerical", "addition", "subtraction",
            "multiplication", "division"
        ):
            prompt += " This is a numerical answer; preserve the digits exactly."
        elif qtype in ("true_false",):
            prompt += " The answer should be True or False if readable."
        elif qtype in ("shape", "shape_name"):
            prompt += " Identify the visible drawn shape if possible."

        # client = genai.Client(api_key=api_key)
        client = genai.Client(
        api_key=api_key,
        http_options={"timeout": 15000},
        )
        response = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
            contents=[prompt, pil_img],
        )

        text = (response.text or "").strip()
        text = text.strip("\"'").strip()

        if not text or text.upper() in (
            "UNREADABLE", "UNCLEAR", "CANNOT READ"
        ):
            return "", 0.0

        # This is a fallback result, not a calibrated confidence score.
        return text, 60.0

    except Exception as e:
        print(f"[GEMINI OCR ERROR] {type(e).__name__}: {e}")
        return "", 0.0




# ── Matching: line detection ──────────────────────────────────────────────────

def _detect_lines_in_crop(crop, num_pairs):
    """Use OpenCV to find hand-drawn lines connecting left items to right items."""
    if crop is None or crop.size == 0:
        return {}

    h, w = crop.shape[:2]
    if h < 10 or w < 10:
        return {}

    _, binary = cv2.threshold(crop, 150, 255, cv2.THRESH_BINARY)
    inv = cv2.bitwise_not(binary)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(inv, kernel, iterations=1)
    edges = cv2.Canny(dilated, threshold1=30, threshold2=100)

    min_len = max(30, int(w * 0.20))
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=15, minLineLength=min_len, maxLineGap=20,
    )

    if lines is None:
        return {}

    mid_x = w / 2
    row_height = h / num_pairs
    mapping = {}

    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        if x1 < mid_x and x2 > mid_x:
            left_y, right_y = y1, y2
        elif x2 < mid_x and x1 > mid_x:
            left_y, right_y = y2, y1
        else:
            continue
        left_row = max(0, min(num_pairs - 1, int(left_y / row_height)))
        right_row = max(0, min(num_pairs - 1, int(right_y / row_height)))
        if left_row not in mapping:
            mapping[left_row] = right_row

    return mapping


def _mapping_to_text(mapping, left_labels, right_labels_sorted):
    """Convert {0: 2, 1: 0, 2: 1} → '1-C,2-A,3-B'."""
    parts = []
    for left_row, right_row in sorted(mapping.items()):
        if left_row < len(left_labels) and right_row < len(right_labels_sorted):
            parts.append(f"{left_labels[left_row]}-{right_labels_sorted[right_row]}")
    return ",".join(parts)


def matching_from_lines(gray_image_path, question):
    """Detect matching answer by finding hand-drawn connecting lines.
    Falls back to text OCR if no crossing lines are found.
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
    text, confidence = _run_tesseract(crop, "matching")
    #needs_review = confidence < CONFIDENCE_THRESHOLD or not text.strip()
    needs_review = confidence < 70 or not text.strip()
    return {
        "recognized_text": text,
        "confidence": confidence,
        "needs_review": needs_review,
    }


# ── Public API ───────────────────────────────────────────────────────────────

def ocr_question(gray_image_path, question):
    print("[DEBUG OCR] Entered ocr_question", flush=True)
    """Full pipeline for one question.

    Steps:
      1. Crop the answer region (with small padding).
      2. For shape questions: geometric detection → Gemini if unsure.
      3. For text questions: Tesseract (8 preprocessing variants, multiple PSMs).
      4. If Tesseract confidence < threshold: Gemini Vision fallback.

    Returns dict: recognized_text, confidence, needs_review
    """
    rx = question["region_x"]
    ry = question["region_y"]
    rw = question["region_w"]
    rh = question["region_h"]
    qtype = (question["question_type"] or "").strip().lower()
    correct_hint = question.get("correct_answer", "") if hasattr(question, "get") else (question["correct_answer"] or "")

    if any(v is None for v in (rx, ry, rw, rh)):
        return {"recognized_text": "", "confidence": 0.0,
                "needs_review": True, "error": "no_region"}

    image = _load_gray(gray_image_path)
    if image is None:
        return {"recognized_text": "", "confidence": 0.0,
                "needs_review": True, "error": "image_not_found"}

    crop = _crop_region(image, rx, ry, rw, rh)
    if crop is None or crop.size == 0:
        return {"recognized_text": "", "confidence": 0.0,
                "needs_review": True, "error": "empty_crop"}

    # ── Shape questions: geometric detection first ────────────────────────
    if qtype in ("shape", "shape_name"):
        try:
            shape_res = detect_shape(crop)
            if shape_res:
                return {
                    "recognized_text": shape_res,
                    "confidence": 95.0,
                    "needs_review": False,
                }
        except Exception:
            pass
        # Shape detection failed → try Gemini
        gemini_text, gemini_conf = _gemini_ocr(crop, "shape", correct_hint)
        if gemini_text:
            return {
                "recognized_text": gemini_text,
                "confidence": gemini_conf,
                "needs_review": False,
            }
        # Last resort: OCR
        text, confidence = _run_tesseract(crop, qtype)
        return {
            "recognized_text": text,
            "confidence": confidence,
            "needs_review": confidence < CONFIDENCE_THRESHOLD or not text.strip(),
        }

    # ── Choose OCR mode using the expected answer ─────────────────────
    # Fill-in-the-blank questions can contain either words or numbers.
    expected_first = str(correct_hint or "").split("|")[0].strip()
    is_numeric_answer = bool(
        re.fullmatch(r"-?\d+(?:\.\d+)?", expected_first)
    )

    tess_qtype = "numerical" if is_numeric_answer else qtype

    print(
        f"[DEBUG OCR] Q type={qtype!r}, "
        f"expected={expected_first!r}, "
        f"numeric_mode={is_numeric_answer}",
        flush=True,
    )

    text, confidence = _run_tesseract(crop, tess_qtype)

   
    gemini_text, gemini_conf = "", 0.0

    GEMINI_TRIGGER_CONFIDENCE = 70.0
    should_use_gemini = (
        not text.strip()
        or confidence < GEMINI_TRIGGER_CONFIDENCE
    )
   
    
    if should_use_gemini:
        print("[DEBUG OCR] Starting Gemini fallback", flush=True)

        try:
            gemini_text, gemini_conf = _gemini_ocr(
            crop, tess_qtype, correct_hint
            )
            print(
                f"[DEBUG OCR] Gemini returned {gemini_text!r}, "
                f"confidence={gemini_conf}",
                flush=True,
            )

            if gemini_text and gemini_conf > 0:
                text = gemini_text
                confidence = gemini_conf

        except Exception as e:
            print(
                f"[DEBUG OCR] Gemini fallback failed: {e}",
                flush=True,
            )
    # Low-confidence OCR must be reviewed manually.
    needs_review = (
        confidence < 70
        or not text.strip()
    )

    return {
        "recognized_text": text,
        "confidence": confidence,
        "needs_review": needs_review,
    }

    # return {
    #     "recognized_text": text,
    #     "confidence": confidence,
    #     "needs_review": needs_review,
    # }
