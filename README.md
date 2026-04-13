# Deepfake Detection Benchmarking Framework

A repeatable benchmarking framework for evaluating and ranking deepfake detection modules, developed as part of a final year dissertation at Coventry University.

## Overview

This framework evaluates five open-source deepfake detection modules under a single consistent protocol across three conditions:
- **Clean performance** — baseline evaluation on unmodified images
- **Robustness** — evaluation under 8 image degradation types (compression, blur, noise, resize)
- **Generalisation** — evaluation across two different fake-generation pipelines

A weighted scoring model combines detection performance (40%), generalisation (25%), robustness (25%), and efficiency (10%) to produce a final score out of 10 for each module.

## Detectors Evaluated

| ID | Detector | Type |
|---|---|---|
| M1 | MesoNet | Specialist deepfake detector |
| M2 | XceptionNet (FF++) | Specialist deepfake detector |
| M3 | SelfBlendedImages (EfficientNet-B4) | Specialist deepfake detector |
| M4 | EfficientNet-B4 (ImageNet) | General vision baseline |
| M5 | EfficientNet-B0 (ImageNet) | General vision baseline |

## Repository Structure
├── preprocess_dataset.py          # Face detection and cropping using RetinaFace
├── evaluate_mesonet.py            # Clean evaluation — MesoNet
├── evaluate_xceptionnet.py        # Clean evaluation — XceptionNet
├── evaluate_sbi.py                # Clean evaluation — SelfBlendedImages
├── evaluate_efficientnet_b4.py    # Clean evaluation — EfficientNet-B4
├── evaluate_efficientnet_b0.py    # Clean evaluation — EfficientNet-B0
├── robustness_mesonet.py          # Robustness evaluation — MesoNet
├── robustness_xceptionnet.py      # Robustness evaluation — XceptionNet
├── robustness_sbi.py              # Robustness evaluation — SelfBlendedImages
├── robustness_efficientnet_b4.py  # Robustness evaluation — EfficientNet-B4
├── robustness_efficientnet_b0.py  # Robustness evaluation — EfficientNet-B0
└── compute_metrics_final.py       # Final scoring and ranking

## Requirements

- Python 3.9 or 3.10
- PyTorch
- TensorFlow / Keras
- timm
- OpenCV
- scikit-learn
- pandas
- numpy
- retina-face

## Results

| Rank | Detector | Final Score |
|---|---|---|
| 1 | MesoNet | 7.62 / 10 |
| 2 | EfficientNet-B0 | 7.47 / 10 |
| 3 | XceptionNet | 4.70 / 10 |
| 4 | EfficientNet-B4 | 4.62 / 10 |
| 5 | SelfBlendedImages | 3.94 / 10 |

## Academic Use Only

This project is strictly for academic and research purposes. All tools and models used are open-source. The evaluation dataset is not publicly released as it contains photographs of the researcher.
