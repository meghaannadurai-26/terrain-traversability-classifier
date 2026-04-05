<div align="center">

# 🤖 Terrain Traversability Classifier
### for Autonomous Rovers

**A deep learning pipeline that classifies outdoor terrain and generates real-time traversability costmaps for autonomous rover navigation.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![ROS2](https://img.shields.io/badge/ROS2-Nav2%20Ready-22314E?style=for-the-badge&logo=ros&logoColor=white)](https://nav2.ros.org/)
[![Dataset: RUGD](https://img.shields.io/badge/Dataset-RUGD-green?style=for-the-badge)](http://rugd.vision/)

[🚀 Quick Start](#-quick-start) · [📊 Results](#-results) · [🧠 Model](#-model-architecture) · [🗺️ ROS2 Integration](#%EF%B8%8F-ros-2-nav2-integration) · [📄 Report](Project_Report.md)

</div>

---

## 🌟 Overview

This project implements a **custom Convolutional Neural Network (CNN)** — built entirely from scratch without any pretrained weights — that classifies terrain patches from camera images into semantic terrain categories and maps them to traversability scores. The output can be directly fed into a **ROS 2 Nav2 costmap** to enable safe, intelligent path planning for autonomous rovers in unstructured outdoor environments.

| Feature | Detail |
|---------|--------|
| 🎯 **Task** | Multi-class terrain classification → traversability scoring |
| 📦 **Dataset** | RUGD (Real Unstructured Ground Dataset) — 10 terrain classes |
| 🧠 **Model** | Custom CNN, built from scratch (no transfer learning) |
| 🏁 **Output** | Safe / Risky / Avoid costmap overlay |
| 🤖 **Target** | ROS 2 Nav2 integration for autonomous rovers |
| ⚡ **Speed** | Real-time sliding-window inference on images & video |

---

## 📊 Results

<div align="center">

![Training Results](results/terrain_classifier_results.png)

*Training curves, confusion matrix, per-class F1, traversability distribution, and evaluation summary*

</div>

### 🏆 Key Metrics

| Metric | Score |
|--------|-------|
| Test Accuracy | See `results/metrics.txt` |
| Traversability Accuracy | See `results/metrics.txt` |
| F1 Score (weighted) | See `results/metrics.txt` |
| Training Time | ~15 epochs |

---

## 🧠 Model Architecture

A **custom CNN built from scratch** — no pretrained weights, full control over every layer:

```
Input Image (3 × 128 × 128)
         │
    ┌────▼────────────────────────────────────┐
    │  ConvBlock × 4                           │
    │  3 → 32 → 64 → 128 → 256 channels       │
    │  [Conv2d → BN → ReLU] × 2               │
    │   + MaxPool2d + Dropout2d               │
    └────────────────────────────────┬────────┘
                                     │
                          Global Average Pooling
                            (256 × 1 × 1)
                                     │
                        FC(512) → ReLU → Dropout
                                     │
                          FC(NUM_CLASSES)
                                     │
                      Terrain Class + Traversability Score
```

### 🗺️ Traversability Mapping

| Score | Label | Color | Terrain Types |
|:-----:|:-----:|:-----:|:-------------|
| 0 | ✅ **Safe** | 🟢 Green | concrete, asphalt, gravel |
| 1 | ⚠️ **Risky** | 🟠 Orange | grass, mulch, dirt, sand |
| 2 | 🚫 **Avoid** | 🔴 Red | rock, mud, water, bush, tree, log |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/sabari-1507/terrain-traversability-classifier.git
cd terrain-traversability-classifier
```

### 2. Install dependencies

```bash
pip install torch torchvision matplotlib numpy scikit-learn seaborn tqdm pillow
```

### 3. Run inference immediately (no training needed!)

The trained model weights (`results/terrain_cnn.pth`) are included. Run inference right away:

```bash
# On a sample image from the included RUGD sample data
python inference_costmap.py

# On your own image
python inference_costmap.py --image path/to/your/image.png
```

---

## 📁 Project Structure

```
terrain-traversability-classifier/
│
├── 📄 terrain_classifier.py     # Main: train CNN from scratch, evaluate, save model
├── 📄 data_preprocessor.py      # Preprocess raw RUGD annotations → patch dataset
├── 📄 inference_costmap.py      # Run model on image → traversability costmap overlay
├── 📄 video_inference.py        # Run model on video footage frame-by-frame
│
├── 📁 results/
│   ├── terrain_cnn.pth                  # ✅ Trained model weights (included!)
│   ├── terrain_classifier_results.png   # Training plots & evaluation charts
│   ├── metrics.txt                      # Full numeric evaluation metrics
│   └── costmaps/                        # Sample costmap output images
│
├── 📁 RUGD_sample-data/         # Sample RUGD images for immediate demo
│   ├── images/                  # Raw RGB frames
│   └── annotations/             # Semantic annotation masks
│
├── 📄 Project_Report.md         # Full ML project report
├── 📄 pipeline_flowchart.md     # System pipeline architecture
├── 📄 README.md
└── 📄 LICENSE
```

---

## 🔧 Full Training Pipeline

> ⚠️ The full RUGD dataset (~5GB) is required for training. Download from [rugd.vision](http://rugd.vision/)

### Step 1 — Preprocess RUGD dataset

```bash
python data_preprocessor.py
```

Extracts terrain patches from RUGD annotation maps into a structured `rugd_dataset/` folder.

### Step 2 — Train the CNN

```bash
python terrain_classifier.py
```

Trains for **15 epochs** with:
- Data augmentation (flips, rotation, color jitter)
- CosineAnnealingLR scheduler
- Label smoothing loss
- Saves best model to `results/terrain_cnn.pth`

### Step 3 — Inference

```bash
# Single image → costmap
python inference_costmap.py --image path/to/image.png

# Video → frame-by-frame traversability
python video_inference.py
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 128 × 128 px |
| Batch size | 32 |
| Epochs | 15 |
| Optimizer | Adam (`lr=1e-3`, `wd=1e-4`) |
| LR Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss (label smoothing=0.1) |
| Train / Val / Test | 70% / 15% / 15% |
| Device | CUDA (auto-fallback to CPU) |

---

## 💻 Load the Model in Your Own Code

```python
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ── Define the model ──────────────────────────────────────────────────────────
class TerrainCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            )
        self.features = nn.Sequential(
            conv_block(3,32), conv_block(32,64),
            conv_block(64,128), conv_block(128,256),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 512),
            nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))

# ── Load weights ──────────────────────────────────────────────────────────────
CLASS_NAMES = ['asphalt', 'bush', 'concrete', 'grass', 'gravel',
               'log', 'mulch', 'rock', 'sand', 'tree']
TRAVERSABILITY = {
    'concrete':0, 'asphalt':0, 'gravel':0,   # Safe
    'grass':1, 'mulch':1, 'dirt':1, 'sand':1, # Risky
    'rock':2, 'mud':2, 'bush':2, 'tree':2, 'log':2  # Avoid
}

model = TerrainCNN(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load("results/terrain_cnn.pth", map_location="cpu"))
model.eval()

# ── Run inference ─────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

img = Image.open("your_image.png").convert("RGB")
with torch.no_grad():
    pred = model(transform(img).unsqueeze(0)).argmax(1).item()

terrain   = CLASS_NAMES[pred]
trav      = TRAVERSABILITY.get(terrain, 1)
labels    = {0:"Safe ✅", 1:"Risky ⚠️", 2:"Avoid 🚫"}
print(f"Terrain: {terrain} → {labels[trav]}")
```

---

## 🗺️ ROS 2 Nav2 Integration

The traversability scores map directly to **Nav2 costmap values**:

```
Score 0 (Safe)  → costmap value:   0  (free space)
Score 1 (Risky) → costmap value: 128  (high cost)
Score 2 (Avoid) → costmap value: 254  (lethal obstacle)
```

The sliding-window costmap generator in `inference_costmap.py` produces colored overlays compatible with this mapping. See [`pipeline_flowchart.md`](pipeline_flowchart.md) for the full system architecture.

---

## 📂 Dataset

This project uses the **RUGD (Real Unstructured Ground Dataset)**:

> Wigness, M., Eum, S., Rogers, J. G., Han, D., & Kwon, H. (2019).
> *A RUGD Dataset for Autonomous Navigation and Visual Perception in Unstructured Outdoor Environments.*
> IROS 2019.

- **Download:** http://rugd.vision/
- **Classes used:** asphalt, bush, concrete, grass, gravel, log, mulch, rock, sand, tree
- Sample data is included in `RUGD_sample-data/` for immediate demo use

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome!

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to your branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Citation

If you use this project in your research or work, please cite it:

```bibtex
@software{terrain_traversability_classifier,
  author    = {Sabari},
  title     = {Terrain Traversability Classifier for Autonomous Rovers},
  year      = {2026},
  url       = {https://github.com/sabari-1507/terrain-traversability-classifier},
  note      = {Custom CNN-based terrain classification with ROS 2 Nav2 integration}
}
```

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [RUGD Dataset](http://rugd.vision/) — Wigness et al., IROS 2019
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [ROS 2 Nav2](https://nav2.ros.org/) — Navigation stack for autonomous robots
- [scikit-learn](https://scikit-learn.org/) — Evaluation metrics

---

<div align="center">

**⭐ Star this repo if you found it useful!**

Made with ❤️ for the Autonomous Robotics community

</div>
