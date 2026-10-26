"""Find out WHY marks are low.   Run:   py debug_scoring.py <submission_id>

It shows, for every question of one uploaded answer sheet:
  * where the answer box really is (page_with_boxes.png)
  * exactly what the box contains (q01_crop.png ...)
  * what OCR reads, and whether that matches the correct answer
Results go in a new folder:  debug_output/submission_<id>/
"""
import os
import re
import sqlite3
import sys

import cv2
import numpy as np
import pytesseract

from config import Config

OUT_ROOT = "debug_output"


def setup_tesseract():
    path = getattr(Config, "TESSERACT_CMD", None) or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path


def read_gray(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def save_png(path, image):
    ok, buf = cv2.imencode(".png", image)
    if ok:
        buf.tofile(path)


def to_number(text):
    table = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5", "B": "8"})
    cleaned = re.sub(r"[^0-9.\-]", "", text.translate(table))
    try:
        return float(cleaned)
    except ValueError:
        return None


def compare(qtype, expected, got):
    """Return (match?, why)."""
    got = got.strip()
    if not got:
        return False, "OCR returned nothing"
    if qtype == "numerical":
        a, b = to_number(got), to_number(expected)
        if a is None:
            return False, "OCR text has no number"
        return (a == b), ("same number" if a == b else f"number differs ({a:g} vs {b:g})")
    if qtype == "matching":
        pairs = lambda s: sorted(f"{n}-{c.upper()}" for n, c in re.findall(r"(\d+)\s*[-_:=.]*\s*([A-Za-z])", s))
        return pairs(got) == pairs(expected), "pairs compared"
    if qtype == "shape":
        return got.lower() == expected.lower(), "shape (OCR is not used for shapes)"
    options = [o.strip().lower() for o in expected.split("|")]
    g = re.sub(r"[^a-z0-9]", "", got.lower())
    ok = g in [re.sub(r"[^a-z0-9]", "", o) for o in options]
    return ok, ("same text" if ok else "text differs")


def clean_ink(binary_crop):
    """Mask (255 = ink) without the printed box border and tiny specks."""
    ink = (binary_crop == 0).astype(np.uint8) * 255
    h, w = ink.shape
    band = 40
    hor = cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(w * 0.8), 20), 1))
    ver = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(int(h * 0.8), 20)))
    h_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, hor)
    v_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, ver)
    h_lines[band:h - band, :] = 0
    v_lines[:, band:w - band] = 0
    lines = cv2.dilate(h_lines | v_lines, np.ones((3, 3), np.uint8), iterations=2)
    ink[lines > 0] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 30:
            ink[labels == i] = 0
    return ink


def ocr_variants(crop, ink, qtype):
    """Try a few sensible Tesseract settings and return (text, confidence)."""
    ys, xs = np.where(ink > 0)
    if len(xs) < 150:
        return [("(box looks empty)", 0.0)], None
    mask = cv2.dilate(ink, np.ones((5, 5), np.uint8))
    cleaned = np.where(mask > 0, crop, 255).astype(np.uint8)
    pad = 6
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, crop.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, crop.shape[0])
    tight = cleaned[y0:y1, x0:x1]
    scale = min(max(80 / max(tight.shape[0], 1), 0.5), 3.0)
    big = cv2.resize(tight, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    big = cv2.copyMakeBorder(big, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)

    configs = ["--oem 1 --psm 7", "--oem 1 --psm 8"]
    if qtype == "numerical":
        configs.append("--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789.-")
    results = []
    for cfg in configs:
        data = pytesseract.image_to_data(big, config=cfg, output_type=pytesseract.Output.DICT)
        words = [(t.strip(), float(c)) for t, c in zip(data["text"], data["conf"]) if t.strip() and float(c) >= 0]
        text = " ".join(w for w, _ in words)
        conf = sum(c for _, c in words) / len(words) / 100 if words else 0.0
        results.append((text, conf))
    return results, big


def main():
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        print("Usage: py debug_scoring.py <submission_id>")
        return
    sid = int(sys.argv[1])
    setup_tesseract()

    db = sqlite3.connect(Config.DATABASE_PATH)
    db.row_factory = sqlite3.Row
    sub = db.execute("SELECT * FROM submissions WHERE id = ?", (sid,)).fetchone()
    if sub is None:
        print(f"No submission with id {sid}.")
        return
    questions = db.execute("SELECT * FROM questions WHERE assessment_id = ? ORDER BY question_no",
                           (sub["assessment_id"],)).fetchall()

    stem = os.path.splitext(sub["image_path"])[0]
    gray_path = os.path.join(Config.UPLOAD_FOLDER, stem + "_gray.png")
    if not os.path.exists(gray_path):
        print("Processed image not found:", gray_path)
        return
    gray = read_gray(gray_path)
    binary = read_gray(os.path.join(Config.UPLOAD_FOLDER, stem + "_binary.png"))
    H, W = gray.shape
    out = os.path.join(OUT_ROOT, f"submission_{sid}")
    os.makedirs(out, exist_ok=True)

    stored = {}
    try:
        for r in db.execute("SELECT * FROM answers WHERE submission_id = ?", (sid,)):
            stored[r["question_id"]] = r
    except sqlite3.Error:
        pass

    print(f"\nSheet: {sub['student_name']} (roll {sub['student_roll_no']}), processed page {W}x{H}px")
    print(f"Questions in assessment: {len(questions)}\n")
    page = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    rows, problems = [], {"no_box": 0, "empty_box": 0, "ocr_wrong": 0, "ok": 0}

    for q in questions:
        no, qtype, exp = q["question_no"], q["question_type"], q["correct_answer"]
        if q["region_x"] is None:
            problems["no_box"] += 1
            rows.append((no, qtype, exp, "-", "NO ANSWER BOX DEFINED", ""))
            continue
        x0, y0 = int(q["region_x"] * W), int(q["region_y"] * H)
        x1, y1 = int((q["region_x"] + q["region_w"]) * W), int((q["region_y"] + q["region_h"]) * H)
        cv2.rectangle(page, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(page, f"Q{no}", (x0 + 2, max(y0 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        crop = gray[max(y0, 0):y1, max(x0, 0):x1]
        crop_bin = binary[max(y0, 0):y1, max(x0, 0):x1]
        if crop.size == 0:
            problems["empty_box"] += 1
            rows.append((no, qtype, exp, "-", "BOX IS OUTSIDE THE PAGE", ""))
            continue
        save_png(os.path.join(out, f"q{no:02d}_crop.png"), crop)

        if qtype == "shape":
            ink_px = int(np.count_nonzero(clean_ink(crop_bin)))
            ok = ink_px > 150
            problems["ok" if ok else "empty_box"] += 1
            rows.append((no, qtype, exp, "(shape)", "drawing found" if ok else "BOX LOOKS EMPTY", ""))
            continue

        ink = clean_ink(crop_bin)
        variants, seen = ocr_variants(crop, ink, qtype)
        if seen is not None:
            save_png(os.path.join(out, f"q{no:02d}_seen_by_ocr.png"), seen)
        best, why_best = None, ""
        for text, conf in variants:
            ok, why = compare(qtype, exp, text)
            if ok:
                best, why_best = (text, conf, True), why
                break
        if best is None:
            text, conf = variants[0]
            ok, why = compare(qtype, exp, text)
            best, why_best = (text, conf, False), why
        text, conf, ok = best
        if text.startswith("(box looks empty)"):
            problems["empty_box"] += 1
            verdict = "BOX LOOKS EMPTY (writing is outside the box?)"
        elif ok:
            problems["ok"] += 1
            verdict = "OK"
        else:
            problems["ocr_wrong"] += 1
            verdict = "OCR WRONG: " + why_best
        s = stored.get(q["id"])
        saved = ""
        if s is not None:
            saved = f"app saved: '{s['recognized_text']}' -> {s['marks_awarded']}"
        rows.append((no, qtype, exp, text.replace("\n", " ")[:22], verdict, saved))

    save_png(os.path.join(out, "page_with_boxes.png"), page)

    print(f"{'Q':>3} {'type':<10} {'correct':<14} {'OCR read':<24} {'verdict':<44} stored in app")
    for no, t, e, g, v, s in rows:
        print(f"{no:>3} {t:<10} {e[:13]:<14} {g:<24} {v:<44} {s}")
    n = len(questions)
    print(f"\nSummary of {n} questions:  OK={problems['ok']}  OCR wrong={problems['ocr_wrong']}  "
          f"empty/outside box={problems['empty_box']}  no box={problems['no_box']}")
    print(f"\nLook at:  {out}\\page_with_boxes.png   (are the red boxes ON the answers?)")
    print(f"          {out}\\q01_crop.png ...        (the box as cut from the page)")
    print(f"          {out}\\q01_seen_by_ocr.png ... (exactly what OCR was given)\n")


main()