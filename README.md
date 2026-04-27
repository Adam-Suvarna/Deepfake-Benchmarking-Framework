# Deepfake Detection Benchmarking Framework

This is my repeatable benchmarking framework for evaluating and ranking deepfake detection modules, developed as part of a final year dissertation at Coventry University.

## Overview

This framework evaluates five open-source deepfake detection modules under a single consistent protocol across three conditions:
- **Clean performance**: baseline evaluation on unmodified images
- **Robustness**: evaluation under 8 image degradation types (compression, blur, noise, resize)
- **Generalisation**: evaluation across two different fake-generation pipelines

A scoring model combines detection performance (40%), generalisation (25%), robustness (25%), and efficiency (10%) to produce a final score out of 10 for each module.

## Detectors Evaluated

| ID | Detector | Type |
|---|---|---|
| M1 | MesoNet | Specialist deepfake detector |
| M2 | XceptionNet (FF++) | Specialist deepfake detector |
| M3 | SelfBlendedImages (EfficientNet-B4) | Specialist deepfake detector |
| M4 | EfficientNet-B4 (ImageNet) | General vision baseline |
| M5 | EfficientNet-B0 (ImageNet) | General vision baseline |

```
├── Preprocessing/
│   └── preprocess_dataset.py          # Face detection and cropping using RetinaFace
│
├── M1_MesoNet/
│   ├── evaluate_mesonet.py            # Clean evaluation — MesoNet
│   ├── robustness_mesonet.py          # Robustness evaluation — MesoNet
│   ├── results_mesonet.csv            # Clean condition results
│   └── results_mesonet_robustness.csv # Robustness condition results
│
├── M2_XceptionNet/
│   ├── evaluate_xceptionnet.py        # Clean evaluation — XceptionNet
│   ├── robustness_xceptionnet.py      # Robustness evaluation — XceptionNet
│   ├── results_xceptionnet.csv        # Clean condition results
│   └── results_xceptionnet_robustness.csv # Robustness condition results
│
├── M3_SBI/
│   ├── evaluate_sbi.py                # Clean evaluation — SelfBlendedImages
│   ├── robustness_sbi.py              # Robustness evaluation — SelfBlendedImages
│   ├── results_sbi.csv                # Clean condition results
│   └── results_sbi_robustness.csv     # Robustness condition results
│
├── M4_EfficientNet_B4/
│   ├── evaluate_efficientnet_b4.py    # Clean evaluation — EfficientNet-B4
│   ├── robustness_efficientnet_b4.py  # Robustness evaluation — EfficientNet-B4
│   ├── results_efficientnet_b4.csv    # Clean condition results
│   └── results_efficientnet_b4_robustness.csv # Robustness condition results
│
├── M5_EfficientNet_B0/
│   ├── evaluate_efficientnet_b0.py    # Clean evaluation — EfficientNet-B0
│   ├── robustness_efficientnet_b0.py  # Robustness evaluation — EfficientNet-B0
│   ├── results_efficientnet_b0.csv    # Clean condition results
│   └── results_efficientnet_b0_robustness.csv # Robustness condition results
│
└── Final_Results/
    ├── compute_metrics_final.py       # Final scoring and ranking
    ├── labels.csv                     # Dataset label file
    └── final_ranking.csv              # Final weighted scores and ranking
```

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

This project is strictly for academic and research purposes. All tools and models used are open-source. The evaluation dataset is not publicly released as it contains photographs of myself, the researcher.
