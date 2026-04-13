import os
import sys
import time
import csv
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Add Deepfake-Detection to path
sys.path.insert(0, r"C:\Deepfake-Detection")
from network.xception import xception

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_DIR = r"C:\Users\adams\OneDrive - Coventry University\Disseertation Project\Image Dataset\The Evauluation Dataset"
LABELS_CSV  = os.path.join(DATASET_DIR, "labels.csv")
OUTPUT_CSV  = os.path.join(DATASET_DIR, "results_xceptionnet_robustness.csv")
WEIGHTS     = r"C:\Deepfake-Detection\pretrained_model\ffpp_c23.pth"

# ── Transformations ───────────────────────────────────────────────────────────
def jpeg_compress(img, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

def add_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def resize_reencode(img, scale):
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * scale), int(h * scale)))
    return cv2.resize(small, (w, h))

TRANSFORMATIONS = {
    "jpeg_q90":      lambda img: jpeg_compress(img, 90),
    "jpeg_q50":      lambda img: jpeg_compress(img, 50),
    "blur_light":    lambda img: cv2.GaussianBlur(img, (3, 3), 0),
    "blur_medium":   lambda img: cv2.GaussianBlur(img, (7, 7), 0),
    "noise_low":     lambda img: add_noise(img, 10),
    "noise_medium":  lambda img: add_noise(img, 25),
    "resize_mild":   lambda img: resize_reencode(img, 0.75),
    "resize_strong": lambda img: resize_reencode(img, 0.50),
}

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading XceptionNet...")
device = torch.device("cpu")
model = xception(num_classes=2, pretrained=False)
state_dict = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
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
        img_rgb = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        inp = transform(img_pil).unsqueeze(0)

        with torch.no_grad():
            output = model(inp)

        if output.shape[1] == 1:
            fake_score = float(torch.sigmoid(output[0][0]))
        else:
            probs = torch.softmax(output, dim=1)
            fake_score = float(probs[0][1])

        predicted_label = "FAKE" if fake_score >= 0.50 else "REAL"
        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1

        results.append({
            "detector":        "M2_XceptionNet",
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
