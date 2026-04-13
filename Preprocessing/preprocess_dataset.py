import os
import cv2
import csv
import numpy as np
from retinaface import RetinaFace

# ── Paths ────────────────────────────────────────────────────────────────────
REAL_DIR   = r"C:\Users\adams\OneDrive - Coventry University\Disseertation Project\Image Dataset\REAL IMAGES"
FAKEA_DIR  = r"C:\Users\adams\OneDrive - Coventry University\Disseertation Project\Image Dataset\FAKE-A (HyperFace Model)"
FAKEB_DIR  = r"C:\Users\adams\OneDrive - Coventry University\Disseertation Project\Image Dataset\FAKE-B (SimSwap Model)"
OUTPUT_DIR = r"C:\Users\adams\OneDrive - Coventry University\Disseertation Project\Image Dataset\The Evauluation Dataset"
CSV_PATH   = os.path.join(OUTPUT_DIR, "labels.csv")

# ── Settings ─────────────────────────────────────────────────────────────────
CROP_SIZE = 256
MARGIN    = 0.3

# ── Helper: crop one image ────────────────────────────────────────────────────
def crop_face(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [SKIP] Could not read: {image_path}")
        return False

    faces = RetinaFace.detect_faces(image_path)

    if not isinstance(faces, dict) or len(faces) == 0:
        print(f"  [SKIP] No face detected: {image_path}")
        return False

    # Use the first detected face
    face_key = list(faces.keys())[0]
    facial_area = faces[face_key]["facial_area"]
    x1, y1, x2, y2 = facial_area

    # Apply margin
    w = x2 - x1
    h = y2 - y1
    x1 = max(0, int(x1 - w * MARGIN))
    y1 = max(0, int(y1 - h * MARGIN))
    x2 = min(img.shape[1], int(x2 + w * MARGIN))
    y2 = min(img.shape[0], int(y2 + h * MARGIN))

    # Crop and resize
    crop = img[y1:y2, x1:x2]
    crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
    cv2.imwrite(output_path, crop)
    return True

# ── Process all three folders ─────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

datasets = [
    (REAL_DIR,  "REAL", "REAL"),
    (FAKEA_DIR, "FAKE", "A"),
    (FAKEB_DIR, "FAKE", "B"),
]

rows = []   # for the CSV

for folder, label, generator in datasets:
    print(f"\nProcessing: {folder}")
    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    for filename in files:
        src_path = os.path.join(folder, filename)
        base     = os.path.splitext(filename)[0]
        out_name = base + ".png"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        print(f"  Processing: {filename}")
        success = crop_face(src_path, out_path)

        if success:
            rows.append({
                "filename":  out_name,
                "label":     label,
                "generator": generator
            })

# ── Write label CSV ───────────────────────────────────────────────────────────
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "label", "generator"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDone! {len(rows)} images saved to: {OUTPUT_DIR}")
print(f"Label file saved to: {CSV_PATH}")
