"""Image preprocessing for scanned or photographed answer sheets (Phase 5).

Pipeline (matches section 5.4 of the synopsis):
  resize -> grayscale -> lighting correction -> noise removal
  -> skew (tilt) correction -> thresholding
"""
import os

import cv2
import numpy as np

TARGET_WIDTH = 1600      # every sheet is resized to this width (keeps ratio)
MIN_SIDE_PIXELS = 300    # smaller images are rejected as too low quality
MAX_SKEW_DEGREES = 10    # tilt search range: -10 to +10 degrees
SKEW_STEP = 0.5          # tilt search precision in degrees


def decode_image(data):
    """Turn uploaded bytes into an OpenCV image. Returns None if invalid."""
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def load_image(path):
    """Read an image file (safe for Windows paths with special characters)."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def save_png(path, image):
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not save the processed image.")
    buffer.tofile(path)


def processed_names(original_filename):
    """Names of the two processed files made from one uploaded file."""
    stem = os.path.splitext(original_filename)[0]
    return f"{stem}_binary.png", f"{stem}_gray.png"


def resize_to_width(image, width=TARGET_WIDTH):
    height, current = image.shape[:2]
    scale = width / current
    method = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (width, int(round(height * scale))),
                      interpolation=method)


def normalize_lighting(gray):
    """Remove shadows and uneven light: divide the image by its own
    blurred background, so the paper becomes uniformly white."""
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=40)
    return cv2.divide(gray, background, scale=255)


def binarize(gray):
    """Black ink on white paper using a local (adaptive) threshold."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 12
    )


def rotate_image(image, angle, border_value=255):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,
    )


def estimate_skew_angle(binary):
    """Find the rotation (in degrees) that makes the text lines horizontal.
    Method: rotate a small copy by many angles; when lines are horizontal
    the rows are either full of ink or empty, so the row-to-row change in
    ink is largest."""
    small_width = 600
    scale = small_width / binary.shape[1]
    ink = cv2.resize(255 - binary, (small_width, int(binary.shape[0] * scale)),
                     interpolation=cv2.INTER_AREA)
    height, width = ink.shape
    center = (width / 2, height / 2)

    angles = np.arange(-MAX_SKEW_DEGREES, MAX_SKEW_DEGREES + 0.01, SKEW_STEP)
    best_angle, best_score = 0.0, -1.0
    for angle in sorted(angles, key=abs):      # try 0 first: ties keep 0
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(ink, matrix, (width, height),
                                 flags=cv2.INTER_NEAREST, borderValue=0)
        profile = rotated.sum(axis=1, dtype=np.float64)
        score = float(np.sum(np.diff(profile) ** 2))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def process_image(image, binary_path, gray_path):
    """Run the full pipeline and save two files:
       gray_path   - cleaned, straightened grayscale (used later for OCR)
       binary_path - black-and-white version (shown to the teacher)
    Returns a small dictionary describing what was done."""
    original_h, original_w = image.shape[:2]
    if min(original_h, original_w) < MIN_SIDE_PIXELS:
        raise ValueError(
            f"Image is too small ({original_w}x{original_h}). "
            "Please upload a clearer, larger photo or scan."
        )

    resized = resize_to_width(image)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = normalize_lighting(gray)
    gray = cv2.medianBlur(gray, 3)                    # noise removal

    angle = estimate_skew_angle(binarize(gray))
    if angle != 0:
        gray = rotate_image(gray, angle)
    binary = binarize(gray)

    save_png(gray_path, gray)
    save_png(binary_path, binary)
    return {
        "original_size": (original_w, original_h),
        "processed_size": (gray.shape[1], gray.shape[0]),
        "skew_corrected_degrees": angle,
    }