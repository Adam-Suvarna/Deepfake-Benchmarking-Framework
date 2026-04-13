import os
import time
import csv
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_DIR = r"C:\Users\adams\OneDrive - Coventry University\Disseertation Project\Image Dataset\The Evauluation Dataset"
LABELS_CSV  = os.path.join(DATASET_DIR, "labels.csv")
OUTPUT_CSV  = os.path.join(DATASET_DIR, "results_xceptionnet.csv")
WEIGHTS     = r"C:\Deepfake-Detection\pretrained_model\ffpp_c23.pth"

# ── Load model ────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, r"C:\Deepfake-Detection")
from network.xception import xception

print("Loading XceptionNet...")
model = xception(num_classes=2, pretrained=False)
state_dict = torch.load(WEIGHTS, map_location="cpu", weights_only=False)

# Strip "model." prefix from keys if present
new_state_dict = {}
for k, v in state_dict.items():
    key = k[6:] if k.startswith("model.") else k
    if key == "last_linear.1.weight":
        key = "last_linear.weight"
    elif key == "last_linear.1.bias":
        key = "last_linear.bias"
    new_state_dict[key] = v

model.load_state_dict(new_state_dict)
model.eval()
print("XceptionNet loaded.")

# ── Image preprocessing ───────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ── Read labels ───────────────────────────────────────────────────────────────
labels = {}
generators = {}
with open(LABELS_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        labels[row["filename"]]     = row["label"]
        generators[row["filename"]] = row["generator"]

# ── Warm-up run ───────────────────────────────────────────────────────────────
print("Running warm-up...")
sample_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".png")]
if sample_files:
    warmup_path = os.path.join(DATASET_DIR, sample_files[0])
    img = Image.open(warmup_path).convert("RGB")
    inp = transform(img).unsqueeze(0)
    with torch.no_grad():
        _ = model(inp)
print("Warm-up done.")

# ── Evaluate all images ───────────────────────────────────────────────────────
print(f"\nEvaluating {len(labels)} images...")
results = []

for filename, true_label in labels.items():
    image_path = os.path.join(DATASET_DIR, filename)

    if not os.path.exists(image_path):
        print(f"  [SKIP] File not found: {filename}")
        continue

    img = Image.open(image_path).convert("RGB")
    inp = transform(img).unsqueeze(0)

    start = time.time()
    with torch.no_grad():
        output = model(inp)
    elapsed = time.time() - start

    # Apply sigmoid to get probability
    if output.shape[1] == 1:
        fake_score = float(torch.sigmoid(output[0][0]))
    else:
        probs = torch.softmax(output, dim=1)
        fake_score = float(probs[0][1])

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
