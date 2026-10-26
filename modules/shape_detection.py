"""Shape detection module: identifies hand-drawn geometric shapes.

Uses contour analysis, circularity, vertex counting, and ellipse fitting.
Designed to be robust against imperfect/smudged children's drawings.
"""
import cv2
import numpy as np


def detect_shape(crop):
    """Detect basic geometric shapes in a cropped answer region.

    Handles imperfect drawings: wobbly circles, smudged triangles, etc.

    Returns one of:
      circle, oval, square, rectangle, triangle, star, heart, diamond
    or None if detection is not confident enough.
    """
    if crop is None or crop.size == 0:
        return None

    # Ensure grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    h, w = gray.shape

    # --- Multi-threshold approach: try several binarizations ---
    candidates = []

    def _try_contours(binary):
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        # Ignore contours that are just the box border
        # (too close to image edges or covering >80% of image)
        valid = []
        img_area = h * w
        for c in contours:
            area = cv2.contourArea(c)
            if area < 50:
                continue
            if area > img_area * 0.85:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            # Skip if the contour hugs all four edges (it's the border)
            if x < 3 and y < 3 and x + cw > w - 3 and y + ch > h - 3:
                continue
            valid.append(c)
        if not valid:
            return None
        return max(valid, key=cv2.contourArea)

    # Threshold variant 1: OTSU
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, t1 = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidates.append(t1)

    # Threshold variant 2: Adaptive (better for uneven lighting)
    t2 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )
    candidates.append(t2)

    # Threshold variant 3: Dilate to join broken pen strokes
    kernel = np.ones((3, 3), np.uint8)
    t3 = cv2.dilate(t1, kernel, iterations=2)
    candidates.append(t3)

    shape_votes = {}

    for binary in candidates:
        cnt = _try_contours(binary)
        if cnt is None:
            continue
        shape = _classify_contour(cnt)
        if shape:
            shape_votes[shape] = shape_votes.get(shape, 0) + 1

    if not shape_votes:
        return None

    # Return the shape with most votes
    best_shape = max(shape_votes, key=shape_votes.get)
    # Only return if at least one variant agreed
    return best_shape


def _classify_contour(cnt):
    """Classify a contour into a shape name."""
    area = cv2.contourArea(cnt)
    if area < 50:
        return None

    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return None

    # --- Circularity (1.0 = perfect circle) ---
    circularity = 4 * np.pi * (area / (peri * peri))

    # --- Approximate polygon ---
    epsilon = 0.03 * peri   # tighter tolerance for better vertex detection
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    vertices = len(approx)

    # --- Convexity ---
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # --- Bounding box aspect ratio ---
    x, y, bw, bh = cv2.boundingRect(cnt)
    aspect = float(bw) / bh if bh > 0 else 1.0

    # ── Circle / Oval detection ──────────────────────────────────────────
    # Children's circles are rarely perfect; accept circularity >= 0.60
    if circularity >= 0.70 and solidity >= 0.80:
        # Distinguish circle vs oval by aspect ratio
        if 0.80 <= aspect <= 1.25:
            return "circle"
        else:
            return "oval"

    # Ellipse fit for oval shapes (even with low circularity)
    if len(cnt) >= 5:
        try:
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (MA, ma), angle = ellipse
            if MA > 0 and ma > 0:
                ratio = min(MA, ma) / max(MA, ma)
                if ratio >= 0.75 and circularity >= 0.55:
                    if 0.80 <= aspect <= 1.25:
                        return "circle"
                    else:
                        return "oval"
                elif ratio >= 0.50 and circularity >= 0.45:
                    return "oval"
        except Exception:
            pass

    # ── Triangle ────────────────────────────────────────────────────────
    if vertices == 3:
        return "triangle"

    # ── Square / Rectangle ───────────────────────────────────────────────
    if vertices == 4:
        # Use aspect ratio of bounding box
        if 0.85 <= aspect <= 1.15:
            return "square"
        else:
            return "rectangle"

    # ── Star / Diamond ───────────────────────────────────────────────────
    if 5 <= vertices <= 6:
        if solidity < 0.75:
            return "star"
        else:
            return "diamond"

    # ── Fallback heuristics ──────────────────────────────────────────────
    # Many vertices but high circularity → roughly circular
    if vertices > 6 and circularity >= 0.50:
        if 0.80 <= aspect <= 1.25:
            return "circle"
        return "oval"

    # High solidity + many vertices → probably a circle/oval that wasn't caught
    if solidity >= 0.85 and circularity >= 0.45:
        if 0.80 <= aspect <= 1.25:
            return "circle"
        return "oval"

    return None
