import os
import csv
import cv2
import torch
import timm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_DIR = r"C:\Users\adams\OneDrive - Coventry University\Disseertation Project\Image Dataset\The Evauluation Dataset"
LABELS_CSV  = os.path.join(DATASET_DIR, "labels.csv")
OUTPUT_CSV  = os.path.join(DATASET_DIR, "results_efficientnet_b4_robustness.csv")

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
print("Loading EfficientNet-B4...")
device = torch.device("cpu")
model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=2)
model = model.to(device)
model.eval()
print("EfficientNet-B4 loaded.")

# ── Image preprocessing ───────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
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
        inp = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(inp)

        probs = torch.softmax(output, dim=1)
        fake_score = float(probs[0][1])
        predicted_label = "FAKE" if fake_score >= 0.50 else "REAL"
        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1

        results.append({
            "detector":        "M4_EfficientNetB4",
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
