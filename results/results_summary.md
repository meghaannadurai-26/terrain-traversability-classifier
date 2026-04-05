# Terrain Traversability Classifier — Results Summary
**Generated:** April 2026  
**Model:** Custom TerrainCNN (PyTorch, from scratch)  
**Dataset:** RUGD — Robot Unstructured Ground Driving  

---

## Final Metrics

| Metric | Value |
|---|---|
| Best Validation Accuracy | 95.61% |
| **Test Accuracy** | **95.91%** |
| Precision (weighted) | 95.95% |
| Recall (weighted) | 95.91% |
| F1 Score (weighted) | 95.50% |
| **Traversability Accuracy** | **97.61%** |
| Training Time | ~14.4 min (CPU) |
| Model Parameters | 1,309,930 |
| Dataset Size | 13,366 patches |
| Classes | 10 terrain types |
| Train / Val / Test | 9,356 / 2,004 / 2,006 |

---

## Per-Class F1 Breakdown

| Class | Traversability | Precision | Recall | F1-Score | Test Samples |
|---|---|---|---|---|---|
| asphalt   | ✅ Safe  | 0.678 | 0.953 | 0.792 | 64  |
| concrete  | ✅ Safe  | 0.800 | 0.216 | 0.340 | 37  |
| gravel    | ✅ Safe  | 0.904 | 0.990 | 0.945 | 209 |
| grass     | ⚠️ Risky | 0.964 | 0.946 | 0.955 | 426 |
| mulch     | ⚠️ Risky | 0.989 | 0.996 | 0.993 | 283 |
| sand      | ⚠️ Risky | 0.000 | 0.000 | 0.000 | 0   |
| rock      | 🚫 Avoid | 0.991 | 1.000 | 0.995 | 105 |
| bush      | 🚫 Avoid | 0.940 | 0.817 | 0.874 | 115 |
| tree      | 🚫 Avoid | 0.995 | 1.000 | 0.997 | 764 |
| log       | 🚫 Avoid | 0.000 | 0.000 | 0.000 | 3   |

*Note: sand and log have zero F1 due to insufficient test samples — dataset imbalance artifact.*

---

## Saved Artifacts

| File | Description |
|---|---|
| `terrain_cnn.pth` | Best model weights (saved at peak val accuracy) |
| `terrain_classifier_results.png` | 6-panel evaluation chart (Loss, Accuracy, Confusion Matrix, Per-Class F1, Traversability Pie, Summary Bar) |
| `metrics.txt` | Plain-text metrics file |
| `costmaps/costmap_trail-3_02001.png` | Sliding window costmap — trail scene |
| `costmaps/costmap_trail_00001.png` | Sliding window costmap — trail entry |
| `costmaps/costmap_creek_00001.png` | Sliding window costmap — creek/water scene |
| `costmaps/costmap_park-1_00001.png` | Sliding window costmap — park scene |

---

## Traversability Color Legend

| Color | Score | Label | Rover Action |
|---|---|---|---|
| 🟢 Green | 0 | Safe  | Full speed ahead |
| 🟠 Orange | 1 | Risky | Slow down, proceed carefully |
| 🔴 Red | 2 | Avoid | Reroute immediately |
