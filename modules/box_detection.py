import cv2
import numpy as np
from typing import List, Dict


def detect_answer_boxes(
    image_path: str,
    min_area: int = 1500,
    max_area: int = 25000,
    aspect_range: tuple = (0.6, 1.5),
    kernel_size: int = 5,
) -> List[Dict]:
    """Detect rectangular answer boxes on a uniformly‑grid answer sheet.

    The function is tuned for typical school answer‑sheet layouts where
    each answer space is a clear rectangle. It performs the following steps:

    1. Load the image in grayscale.
    2. Apply Gaussian blur to reduce noise.
    3. Use adaptive threshold (inverse) to obtain a binary image – the
       boxes become white on a black background.
    4. Apply a morphological closing (dilate then erode) to join broken
       edges of the boxes.
    5. Find external contours and filter them by area and aspect ratio.
    6. Sort the remaining boxes top‑to‑bottom then left‑to‑right and assign
       sequential ``question_no`` values.

    Parameters
    ----------
    image_path: str
        Path to the *gray* version of the uploaded answer sheet.
    min_area, max_area: int
        Minimum and maximum contour area (in pixels) that a box can have.
        Increase ``max_area`` for larger boxes; decrease ``min_area`` for
        tiny answer squares.
    aspect_range: tuple (float, float)
        Acceptable width/height ratio. Typical boxes are close to square
        (≈1.0) but a generous range helps if the sheet is scanned at an
        angle.
    kernel_size: int
        Size of the structuring element used for the morphological closing.
        Larger values help when the box borders are faint.

    Returns
    -------
    List[Dict]
        Each dict contains ``question_no``, ``x``, ``y``, ``w`` and ``h``.
    """
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Noise reduction
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Adaptive threshold – makes boxes white (255) on black background
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    # Morphological closing to fill gaps in box borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours of potential boxes
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes: List[Dict] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / float(h)
        if not (aspect_range[0] <= ratio <= aspect_range[1]):
            continue
        boxes.append({"x": x, "y": y, "w": w, "h": h})

    # Sort boxes row‑wise then column‑wise (top‑to‑bottom, left‑to‑right)
    boxes.sort(key=lambda b: (b["y"], b["x"]))
    for i, b in enumerate(boxes, start=1):
        b["question_no"] = i

    return boxes
