import os
import sys
import time
import csv
import cv2
import numpy as np
from PIL import Image
import io

# Add MesoNet to path
sys.path.insert(0, r"C:\MesoNet")
from classifiers import Meso4

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_DIR = r"C:\Users\adams\OneDrive - Coventry University\Disseertation Project\Image Dataset\The Evauluation Dataset"
LABELS_CSV  = os.path.join(DATASET_DIR, "labels.csv")
OUTPUT_CSV  = os.path.join(DATASET_DIR, "results_mesonet_robustness.csv")
WEIGHTS     = r"C:\MesoNet\weights\Meso4_DF.h5"

# ── Transformations ───────────────────────────────────────────────────────────
TRANSFORMATIONS = {
    "jpeg_q90":       lambda img: jpeg_compress(img, 90),
    "jpeg_q50":       lambda img: jpeg_compress(img, 50),
    "blur_light":     lambda img: cv2.GaussianBlur(img, (3, 3), 0),
    "blur_medium":    lambda img: cv2.GaussianBlur(img, (7, 7), 0),
    "noise_low":      lambda img: add_noise(img, 10),
    "noise_medium":   lambda img: add_noise(img, 25),
    "resize_mild":    lambda img: resize_reencode(img, 0.75),
    "resize_strong":  lambda img: resize_reencode(img, 0.50),
}

def jpeg_compress(img, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

def add_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def resize_reencode(img, scale):
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * scale), int(h * scale)))
    return cv2.resize(small, (w, h))

# ── Load MesoNet ──────────────────────────────────────────────────────────────
print("Loading MesoNet...")
model = Meso4()
model.load(WEIGHTS)
print("MesoNet loaded.")

# ── Read labels ───────────────────────────────────────────────────────────────
labels = {}
generators = {}
with open(LABELS_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        labels[row["filename"]]     = row["label"]
        generators[row["filename"]] = row["generator"]

# ── Helper: preprocess for MesoNet ───────────────────────────────────────────
def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((256, 256))
    arr = np.array(img_pil) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print(f"\nRunning robustness evaluation across {len(TRANSFORMATIONS)} conditions...")
results = []

for transform_name, transform_fn in TRANSFORMATIONS.items():
    print(f"\n  Condition: {transform_name}")
    correct = 0

    for filename, true_label in labels.items():
        image_path = os.path.join(DATASET_DIR, filename)
        if not os.path.exists(image_path):
            continue

        img = cv2.imread(image_path)
        img_t = transform_fn(img)
        inp = preprocess(img_t)

        pred = model.predict(inp)
        score = float(pred[0][0])
        fake_score = 1.0 - score
        predicted_label = "FAKE" if fake_score >= 0.50 else "REAL"
        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1

        results.append({
            "detector":        "M1_MesoNet",
            "condition":       transform_name,
            "filename":        filename,
            "true_label":      true_label,
            "generator":       generators[filename],
            "fake_score":      round(fake_score, 4),
            "predicted_label": predicted_label,
            "correct":         str(is_correct),
        })

    print(f"    Correct: {correct}/{len(labels)}")

# ── Save results ──────────────────────────────────────────────────────────────
with open(OUTPUT_CSV, "w", newline="") as f:
    fieldnames = ["detector", "condition", "filename", "true_label", "generator", "fake_score", "predicted_label", "correct"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nDone! Results saved to: {OUTPUT_CSV}")
