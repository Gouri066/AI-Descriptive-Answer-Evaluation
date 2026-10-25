import cv2
import numpy as np


def make_sheet(angle, shadow, noise, path):
    img = np.full((1754, 1240, 3), 255, np.uint8)
    cv2.rectangle(img, (60, 60), (1180, 1690), (0, 0, 0), 3)
    y = 160
    for i in range(1, 9):
        cv2.putText(img, f"{i}. The day after Monday is  Tuesday  {i * 7}",
                    (110, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 40, 40), 3)
        cv2.rectangle(img, (110, y + 25), (1100, y + 140), (0, 0, 0), 2)
        y += 190

    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, matrix, (w, h), borderValue=(255, 255, 255))

    if shadow:   # dark on the left, bright on the right
        gradient = np.linspace(0.45, 1.0, w, dtype=np.float32)[None, :, None]
        img = (img * gradient).astype(np.uint8)
    if noise:
        grain = np.random.normal(0, 12, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])


make_sheet(0, False, False, "test_straight.jpg")
make_sheet(4, True, True, "test_tilt_4_shadow.jpg")
make_sheet(-7, False, True, "test_tilt_minus7.jpg")
make_sheet(12, False, False, "test_tilt_12_too_much.jpg")
print("Created 4 test sheets in this folder.")