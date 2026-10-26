"""Show the actual region coordinates for the latest submission's questions."""
import sqlite3, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config

db = sqlite3.connect(Config.DATABASE_PATH)
db.row_factory = sqlite3.Row

# Get latest submission
sub = db.execute("SELECT * FROM submissions ORDER BY id DESC LIMIT 1").fetchone()
if not sub:
    print("No submissions.")
    sys.exit()

print(f"Submission ID={sub['id']}, student={sub['student_name']}")
print(f"Image: {sub['image_path']}")
print(f"Processed image: {sub['processed_image_path']}")

# Check if files exist
upload_folder = Config.UPLOAD_FOLDER
for fn in [sub['image_path'], sub['processed_image_path']]:
    if fn:
        full = os.path.join(upload_folder, fn)
        exists = os.path.exists(full)
        print(f"  File '{fn}': {'EXISTS' if exists else 'MISSING'}")
        if exists:
            import cv2, numpy as np
            raw = np.fromfile(full, dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                print(f"    Image size: {img.shape[1]}x{img.shape[0]} (WxH)")

print("\n=== Question Regions ===")
qs = db.execute(
    "SELECT question_no, question_type, correct_answer, region_x, region_y, region_w, region_h "
    "FROM questions WHERE assessment_id=? ORDER BY question_no",
    (sub['assessment_id'],)
).fetchall()

# Get image dimensions for pixel coords
gray_name = sub['processed_image_path']
if gray_name:
    gray_name = gray_name.replace('_binary.png', '_gray.png')
    gray_path = os.path.join(upload_folder, gray_name)
    import cv2, numpy as np
    raw = np.fromfile(gray_path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    H, W = img.shape if img is not None else (0, 0)
else:
    W, H = 0, 0

for q in qs:
    rx, ry, rw, rh = q['region_x'], q['region_y'], q['region_w'], q['region_h']
    if rx is not None and W > 0:
        x0 = int(rx * W); y0 = int(ry * H)
        x1 = int((rx + rw) * W); y1 = int((ry + rh) * H)
        size = f" => pixel [{x0},{y0}] to [{x1},{y1}], crop {x1-x0}x{y1-y0}"
    else:
        size = " => NO BOX" if rx is None else ""
    print(f"  Q{q['question_no']} ({q['question_type']}) correct='{q['correct_answer']}'"
          f"  region=({rx},{ry},{rw},{rh}){size}")

    # Also try to crop and see if there's anything
    if rx is not None and W > 0 and img is not None:
        x0 = max(0, int(rx * W))
        y0 = max(0, int(ry * H))
        x1 = min(W, int((rx + rw) * W))
        y1 = min(H, int((ry + rh) * H))
        crop = img[y0:y1, x0:x1]
        if crop.size > 0:
            # Count dark pixels (ink)
            dark_pixels = int((crop < 128).sum())
            total_pixels = crop.size
            print(f"    Crop has {dark_pixels} dark pixels out of {total_pixels} total ({100*dark_pixels//total_pixels}% dark)")
            # Save crop
            out_dir = "debug_crops"
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, f"q{q['question_no']:02d}_crop.png"), crop)
            print(f"    Saved: {out_dir}/q{q['question_no']:02d}_crop.png")
        else:
            print("    Crop is EMPTY - region is outside the image!")
