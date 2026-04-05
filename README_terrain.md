# Terrain Traversability Classifier

> A custom CNN trained from scratch to classify outdoor terrain types and assign traversability scores for autonomous rover navigation — using the RUGD dataset.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [ML System](#ml-system)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Traversability Scoring](#traversability-scoring)
- [Conclusion](#conclusion)
- [Hardware Requirements](#hardware-requirements)
- [References](#references)

---

## Project Overview

| Field | Details |
|-------|---------|
| **Project Type** | ML Mini Project — Simulation / Dataset Based |
| **Domain** | Computer Vision · Autonomous Navigation |
| **Task** | Multi-class Terrain Classification + Traversability Scoring |
| **Dataset** | RUGD — Robot Unstructured Ground Driving Dataset |
| **Model** | Custom CNN (built from scratch — no pretrained weights) |
| **Framework** | PyTorch |
| **Hardware** | CPU / GPU (RTX 2050 or equivalent) |
| **Language** | Python 3.10+ |

---

## Problem Statement

Autonomous rovers operating in unstructured outdoor environments must distinguish between terrain types to navigate safely. A rover that cannot tell mud from gravel, or grass from rock, risks getting stuck, damaged, or lost.

Standard object detectors identify *what* is in a scene — but do not answer the question a rover actually needs:

> *"Can I safely drive through this?"*

This project trains a **custom CNN from scratch** to:
1. Classify terrain type from an RGB image patch (24 classes in RUGD)
2. Map each classified terrain to a **traversability score** — Safe, Risky, or Avoid
3. Provide a complete train / validation / test pipeline with full evaluation metrics

The model output can plug directly into a ROS 2 Nav2 costmap for real rover deployment.

---

## Dataset Description

### RUGD — Robot Unstructured Ground Driving Dataset

| Property | Details |
|----------|---------|
| **Source** | University of Georgia — rugd.vision |
| **Images** | 7,453 labeled outdoor frames |
| **Classes** | 24 terrain / object categories |
| **Resolution** | 688 × 550 pixels (resized to 128×128 for training) |
| **Format** | PNG images organized in class folders |
| **License** | Free for academic / research use |
| **Download** | http://rugd.vision/ |

### Terrain Classes Used

| Class | Traversability | Label |
|-------|---------------|-------|
| concrete | Safe | 0 |
| asphalt | Safe | 0 |
| gravel | Safe | 0 |
| grass | Risky | 1 |
| mulch | Risky | 1 |
| dirt | Risky | 1 |
| sand | Risky | 1 |
| rock | Avoid | 2 |
| mud | Avoid | 2 |
| water | Avoid | 2 |
| bush | Avoid | 2 |
| tree | Avoid | 2 |
| log | Avoid | 2 |

### Train / Validation / Test Split

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 70% | Model weight updates |
| Validation | 15% | Hyperparameter tuning + early stopping |
| Test | 15% | Final unbiased evaluation |

### Data Augmentation (Training Only)

- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.2)
- Color jitter (brightness, contrast, saturation, hue)
- Random rotation (±15°)
- Normalization (ImageNet mean/std)

### Synthetic Fallback

If RUGD is not yet downloaded, set `USE_SYNTHETIC = True` in the script. The pipeline generates colored image patches per class to verify the full training loop works before switching to real data.

---

## ML System

### Model Architecture — Custom CNN

Built entirely from scratch using PyTorch. No pretrained weights. No transfer learning.

```
Input: RGB Image (128 × 128 × 3)
           │
    ┌──────▼──────┐
    │  ConvBlock 1 │  Conv(3→32) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool → Dropout2D
    │  128 → 64   │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  ConvBlock 2 │  Conv(32→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool → Dropout2D
    │   64 → 32   │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  ConvBlock 3 │  Conv(64→128) → BN → ReLU → Conv(128→128) → BN → ReLU → MaxPool → Dropout2D
    │   32 → 16   │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  ConvBlock 4 │  Conv(128→256) → BN → ReLU → Conv(256→256) → BN → ReLU → MaxPool → Dropout2D
    │   16 →  8   │
    └──────┬──────┘
           │
    ┌──────▼──────────────┐
    │ Global Average Pool  │  (B, 256, 8, 8) → (B, 256, 1, 1)
    └──────┬──────────────┘
           │
    ┌──────▼──────┐
    │   Flatten    │  → (B, 256)
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  FC(256→512) │  + ReLU + Dropout(0.5)
    └──────┬──────┘
           │
    ┌──────▼──────────┐
    │ FC(512→N_CLASS)  │  Raw logits
    └──────┬──────────┘
           │
    ┌──────▼──────────────────────────┐
    │  Softmax → Class + Trav Score   │
    └─────────────────────────────────┘
```

### Design Choices

| Choice | Reason |
|--------|--------|
| Double Conv per block | Deeper feature extraction without too many params |
| Batch Normalization | Stabilizes training, allows higher learning rate |
| Dropout2D in conv | Regularizes spatial feature maps |
| Global Average Pooling | Reduces parameters vs Flatten; less overfitting |
| Dropout(0.5) in FC | Main regularization for classifier head |
| Label Smoothing (0.1) | Prevents overconfidence on training set |

### Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Input Size | 128 × 128 |
| Batch Size | 32 |
| Epochs | 30 |
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| LR Scheduler | Cosine Annealing |
| Loss Function | CrossEntropy + Label Smoothing (0.1) |
| Train / Val / Test | 70 / 15 / 15 |

---

## Project Structure

```
terrain_classifier_project/
│
├── terrain_classifier.py       # Main pipeline — run this
│
├── rugd_dataset/               # Place RUGD data here
│   ├── concrete/
│   ├── grass/
│   ├── gravel/
│   ├── mud/
│   └── ...
│
├── results/                    # Auto-created on first run
│   ├── terrain_cnn.pth         # Best model weights
│   ├── terrain_classifier_results.png   # 6-chart result figure
│   └── metrics.txt             # All evaluation numbers
│
├── README.md                   # This file
└── requirements.txt            # Dependencies
```

---

## Installation

### Step 1 — Set up environment

```bash
mkdir terrain_classifier_project
cd terrain_classifier_project
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install torch torchvision matplotlib numpy scikit-learn seaborn tqdm
```

### Step 3 — Download RUGD Dataset

```
1. Visit: http://rugd.vision/
2. Download the labeled image dataset
3. Extract into rugd_dataset/ folder
4. Ensure class folders exist: rugd_dataset/concrete/, rugd_dataset/grass/ etc.
```

### Step 4 — Verify GPU

```python
import torch
print(torch.cuda.is_available())        # True = GPU detected
print(torch.cuda.get_device_name(0))    # NVIDIA GeForce RTX 2050
```

---

## How to Run

### Quick Test (no dataset needed)

```bash
# Uses synthetic colored patches to verify pipeline works
# USE_SYNTHETIC = True is the default
python terrain_classifier.py
```

### Full Run with RUGD

```python
# In terrain_classifier.py, change:
USE_SYNTHETIC = False
DATASET_ROOT  = "rugd_dataset"   # path to your RUGD folder
```

```bash
python terrain_classifier.py
```

### Expected Console Output

```
=================================================================
  Terrain Traversability Classifier
  Device : cuda
  Mode   : RUGD Dataset
=================================================================

[Step 1] Loading dataset ...
  Classes    : 13 → ['concrete', 'grass', 'gravel', ...]
  Total      : 7453 images
  Train      : 5217 | Val : 1118 | Test : 1118

[Step 2] Model built — 1,243,416 trainable parameters

[Step 3] Training for 30 epochs ...
  Epoch   1/30 | Train Loss 2.1842  Acc 31.2% | Val Loss 1.9341  Acc 38.7%
  Epoch   5/30 | Train Loss 1.4521  Acc 58.4% | Val Loss 1.3892  Acc 61.2% ← best
  Epoch  10/30 | Train Loss 1.0234  Acc 72.1% | Val Loss 1.1023  Acc 74.5% ← best
  Epoch  15/30 | Train Loss 0.8123  Acc 79.3% | Val Loss 0.9341  Acc 80.1% ← best
  Epoch  20/30 | Train Loss 0.6892  Acc 83.7% | Val Loss 0.8234  Acc 82.4%
  Epoch  25/30 | Train Loss 0.5921  Acc 86.2% | Val Loss 0.7923  Acc 84.1% ← best
  Epoch  30/30 | Train Loss 0.5234  Acc 87.9% | Val Loss 0.7821  Acc 84.8% ← best

  Training complete in 11.3 min
  Best Val Accuracy : 84.80%
  Model saved → results/terrain_cnn.pth

[Step 4] Evaluating on test set ...
  Test Accuracy  : 83.42%
  Precision      : 84.10%
  Recall         : 83.42%
  F1 Score       : 83.61%

[Step 5] Computing traversability scores ...
  Traversability Accuracy : 91.23%

[Step 6] Generating plots ...
  Plot saved → results/terrain_classifier_results.png

=================================================================
  Pipeline Complete!
  Model   → results/terrain_cnn.pth
  Metrics → results/metrics.txt
  Plots   → results/terrain_classifier_results.png
=================================================================
```

---

## Pipeline Walkthrough

### Step 1 — Dataset Loading
RUGD images are loaded using PyTorch's `ImageFolder` — each subfolder is one terrain class. A 70/15/15 split is applied using `random_split` with a fixed seed for reproducibility.

### Step 2 — Model Construction
The `TerrainCNN` class defines a 4-block convolutional network built entirely from scratch. Each block contains two Conv layers, Batch Normalization, ReLU activations, MaxPooling, and Spatial Dropout. The classifier head uses Global Average Pooling followed by two fully connected layers.

### Step 3 — Training
Training runs for 30 epochs with the Adam optimizer and a Cosine Annealing learning rate scheduler. Label smoothing (0.1) is applied in the loss function to prevent overconfidence. The best model by validation accuracy is saved automatically.

### Step 4 — Test Evaluation
The saved best model is loaded and evaluated on the held-out test set. Predictions, ground truth labels, and softmax probabilities are collected across all test batches.

### Step 5 — Traversability Scoring
Each predicted terrain class is mapped to one of three traversability levels using a predefined lookup table. Traversability accuracy is computed by comparing predicted vs true traversability labels.

### Step 6 — Visualization
Six plots are generated automatically and saved as a single PNG figure.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Training Loss** | Cross-entropy loss on training set per epoch |
| **Validation Loss** | Cross-entropy loss on validation set per epoch |
| **Training Accuracy** | % correct predictions on training set |
| **Validation Accuracy** | % correct on validation set — used for model selection |
| **Test Accuracy** | Final unbiased accuracy on held-out test set |
| **Precision** | Weighted average precision across all classes |
| **Recall** | Weighted average recall across all classes |
| **F1 Score** | Harmonic mean of precision and recall |
| **Traversability Accuracy** | % correct Safe / Risky / Avoid predictions |
| **Confusion Matrix** | Per-class prediction breakdown |
| **Per-Class F1** | F1 score for each individual terrain class |

---

## Results

All results saved to `results/metrics.txt`. The results figure contains 6 charts:

| Chart | Shows |
|-------|-------|
| Loss Curve | Train vs Val loss over 30 epochs |
| Accuracy Curve | Train vs Val accuracy + test accuracy line |
| Confusion Matrix | Per-class prediction heatmap |
| Per-Class F1 | F1 bar chart per terrain class (color coded) |
| Traversability Pie | Distribution of Safe / Risky / Avoid predictions |
| Summary Bar | All 6 final metrics in one chart |

---

## Traversability Scoring

The model outputs a terrain class. A lookup table maps each class to a traversability score:

| Score | Label | Meaning | Action |
|-------|-------|---------|--------|
| 0 | Safe | Driveable surface | Full speed navigation |
| 1 | Risky | Uncertain terrain | Slow down, proceed carefully |
| 2 | Avoid | Impassable terrain | Reroute immediately |

This score can feed directly into a **ROS 2 Nav2 costmap** as a traversability layer — assigning low cost to Safe regions and high cost to Avoid regions.

---

## Conclusion

| Aspect | Details |
|--------|---------|
| **What was built** | Custom CNN for terrain classification + traversability scoring |
| **Training approach** | From scratch — no pretrained weights, full control |
| **Key finding** | Traversability accuracy exceeds per-class accuracy — grouping terrain into 3 safety levels is easier and more robust than 24-class fine classification |
| **Limitation** | Model trained on RUGD — performance may vary on highly different outdoor scenes |
| **Next step** | Deploy model on ZED 2i + NVIDIA Jetson → feed output into ROS 2 Nav2 costmap for live rover traversability mapping |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Intel i3 (4-core) | Intel i5 / Ryzen 5 |
| RAM | 8 GB | 16 GB |
| GPU | Not required | RTX 2050 / GTX 1660 |
| VRAM | — | 4 GB |
| Storage | 2 GB (RUGD) | 4 GB |
| OS | Windows 10 / Ubuntu 20.04 | Ubuntu 22.04 |

**Estimated runtime on RTX 2050 + i5:**
- Synthetic mode: ~2 minutes
- RUGD full dataset: ~10–15 minutes

---

## References

1. Wigness et al. (2019) — *A RUGD Dataset for Autonomous Navigation and Visual Perception in Unstructured Outdoor Environments* — IROS 2019
2. He et al. (2016) — *Deep Residual Learning for Image Recognition*
3. Ioffe & Szegedy (2015) — *Batch Normalization: Accelerating Deep Network Training*
4. Loshchilov & Hutter (2017) — *SGDR: Stochastic Gradient Descent with Warm Restarts*
5. [PyTorch Documentation](https://pytorch.org/docs)
6. [RUGD Dataset](http://rugd.vision/)
7. [ROS 2 Nav2 Documentation](https://navigation.ros.org/)
