import os
import time
import csv
import numpy as np
from PIL import Image
from classifiers import Meso4

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_DIR = r"C:\Users\adams\OneDrive - Coventry University\Disseertation Project\Image Dataset\The Evauluation Dataset"
LABELS_CSV  = os.path.join(DATASET_DIR, "labels.csv")
OUTPUT_CSV  = os.path.join(DATASET_DIR, "results_mesonet.csv")
WEIGHTS     = r"C:\MesoNet\weights\Meso4_DF.h5"

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
        labels[row["filename"]]    = row["label"]
        generators[row["filename"]] = row["generator"]

# ── Helper: preprocess image for MesoNet ─────────────────────────────────────
def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((256, 256))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape: (1, 256, 256, 3)
    return arr

# ── Warm-up run ───────────────────────────────────────────────────────────────
print("Running warm-up...")
sample_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".png")]
if sample_files:
    warmup_path = os.path.join(DATASET_DIR, sample_files[0])
    _ = model.predict(preprocess(warmup_path))
print("Warm-up done.")

# ── Evaluate all images ───────────────────────────────────────────────────────
print(f"\nEvaluating {len(labels)} images...")
results = []

for filename, true_label in labels.items():
    image_path = os.path.join(DATASET_DIR, filename)

    if not os.path.exists(image_path):
        print(f"  [SKIP] File not found: {filename}")
        continue

    img_array = preprocess(image_path)

    # Measure runtime
    start = time.time()
    prediction = model.predict(img_array)
    elapsed = time.time() - start

    # MesoNet outputs a score between 0 and 1
    # Score close to 1 = REAL, close to 0 = FAKE (based on MesoNet's training)
    # We invert so that high score = FAKE to match our threshold policy
    score = float(prediction[0][0])
    fake_score = 1.0 - score  # probability of being FAKE
    predicted_label = "FAKE" if fake_score >= 0.50 else "REAL"

    results.append({
        "filename":        filename,
        "true_label":      true_label,
        "generator":       generators[filename],
        "fake_score":      round(fake_score, 4),
        "predicted_label": predicted_label,
        "correct":         str(predicted_label == true_label),
        "runtime_sec":     round(elapsed, 4)
    })

    print(f"  {filename} | true={true_label} | pred={predicted_label} | score={fake_score:.4f} | time={elapsed:.4f}s")

# ── Save results ──────────────────────────────────────────────────────────────
with open(OUTPUT_CSV, "w", newline="") as f:
    fieldnames = ["filename", "true_label", "generator", "fake_score", "predicted_label", "correct", "runtime_sec"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nDone! Results saved to: {OUTPUT_CSV}")
print(f"Total images evaluated: {len(results)}")
correct = sum(1 for r in results if r["correct"] == "True")
print(f"Correct predictions: {correct}/{len(results)}")
