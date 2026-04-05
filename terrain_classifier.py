"""
================================================================
  Terrain Traversability Classifier — Complete Pipeline
  Dataset  : RUGD (or synthetic fallback for testing)
  Model    : Custom CNN (built from scratch — no pretrained)
  Task     : Multi-class terrain classification
  Output   : Terrain class + Traversability score
================================================================

SETUP:
    pip install torch torchvision matplotlib numpy scikit-learn seaborn tqdm

DATASET:
    Download RUGD from: http://rugd.vision/
    Extract and set DATASET_ROOT below.
    If RUGD is not available, set USE_SYNTHETIC=True to test pipeline
    with synthetic data first, then swap in RUGD.

RUN:
    python terrain_classifier.py
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import os, random, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from tqdm import tqdm
from PIL import Image

# ── 1. Configuration ──────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#  SET THIS to your RUGD dataset path after downloading from rugd.vision
#  Folder structure expected:
#    DATASET_ROOT/
#      concrete/    *.png
#      grass/       *.png
#      gravel/      *.png
#      mud/         *.png
#      ...
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT  = "rugd_dataset"
USE_SYNTHETIC = False       # Using real RUGD patch dataset

IMG_SIZE      = 128         # Input resolution
BATCH_SIZE    = 32
NUM_EPOCHS    = 15          # Full training run
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
# TEST_RATIO  = 0.15 (remainder)
SEED          = 42
SAVE_DIR      = "results"
MODEL_PATH    = os.path.join(SAVE_DIR, "terrain_cnn.pth")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print("=" * 65)
print("  Terrain Traversability Classifier")
print(f"  Device : {DEVICE}")
print(f"  Mode   : {'Synthetic (test)' if USE_SYNTHETIC else 'RUGD Dataset'}")
print("=" * 65)

# ── 2. Traversability Map ─────────────────────────────────────────────────────
# Maps each terrain class → traversability label
# 0 = Safe  |  1 = Risky  |  2 = Avoid
TRAVERSABILITY = {
    "concrete"      : 0,   # Safe
    "asphalt"       : 0,   # Safe
    "gravel"        : 0,   # Safe
    "grass"         : 1,   # Risky
    "mulch"         : 1,   # Risky
    "dirt"          : 1,   # Risky
    "sand"          : 1,   # Risky
    "rock"          : 2,   # Avoid
    "mud"           : 2,   # Avoid
    "water"         : 2,   # Avoid
    "bush"          : 2,   # Avoid
    "tree"          : 2,   # Avoid
    "log"           : 2,   # Avoid
}
TRAV_LABELS = ["Safe", "Risky", "Avoid"]
TRAV_COLORS = ["#4ff7a2", "#f7a24f", "#f74f4f"]


# ── 3. Synthetic Dataset (for testing without RUGD) ───────────────────────────
class SyntheticTerrainDataset(Dataset):
    """
    Generates random colored image patches per terrain class.
    Replace with RUGD ImageFolder once dataset is downloaded.
    Each class gets a distinct mean color to make classification learnable.
    """
    CLASS_COLORS = {
        "concrete" : (180, 180, 180),
        "gravel"   : (150, 130, 110),
        "grass"    : ( 80, 140,  60),
        "dirt"     : (140, 100,  70),
        "mud"      : ( 90,  70,  50),
        "rock"     : (100, 100, 120),
        "water"    : ( 60,  90, 180),
        "sand"     : (210, 190, 140),
    }

    def __init__(self, n_per_class=200, img_size=128, transform=None):
        self.classes     = list(self.CLASS_COLORS.keys())
        self.class_to_idx= {c: i for i, c in enumerate(self.classes)}
        self.transform   = transform
        self.img_size    = img_size
        self.samples     = []
        for cls, color in self.CLASS_COLORS.items():
            for _ in range(n_per_class):
                noise = np.random.randint(-30, 30, (img_size, img_size, 3))
                img   = np.clip(np.array(color) + noise, 0, 255).astype(np.uint8)
                self.samples.append((img, self.class_to_idx[cls]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_array, label = self.samples[idx]
        img = Image.fromarray(img_array)
        if self.transform:
            img = self.transform(img)
        return img, label


# ── 4. Data Transforms ────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ── 5. Load Dataset ───────────────────────────────────────────────────────────
print("\n[Step 1] Loading dataset ...")

if USE_SYNTHETIC:
    full_dataset = SyntheticTerrainDataset(
        n_per_class=250, img_size=IMG_SIZE, transform=train_transform
    )
    class_names = full_dataset.classes
else:
    full_dataset = ImageFolder(DATASET_ROOT, transform=train_transform)
    class_names  = full_dataset.classes

NUM_CLASSES = len(class_names)
total       = len(full_dataset)
n_train     = int(TRAIN_RATIO * total)
n_val       = int(VAL_RATIO   * total)
n_test      = total - n_train - n_val

train_ds, val_ds, test_ds = random_split(
    full_dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED)
)

# Apply val/test transform (no augmentation)
if not USE_SYNTHETIC:
    val_ds.dataset.transform  = val_transform
    test_ds.dataset.transform = val_transform

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)

print(f"  Classes    : {NUM_CLASSES} → {class_names}")
print(f"  Total      : {total} images")
print(f"  Train      : {n_train} | Val : {n_val} | Test : {n_test}")


# ── 6. Custom CNN Model ───────────────────────────────────────────────────────
class TerrainCNN(nn.Module):
    """
    Custom CNN built from scratch — no pretrained weights.
    Architecture:
        4 × ConvBlock (Conv → BN → ReLU → MaxPool)
        Global Average Pooling
        FC(512) → Dropout → FC(NUM_CLASSES)
    """
    def __init__(self, num_classes):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.1),
            )

        self.features = nn.Sequential(
            conv_block(  3,  32),   # 128 → 64
            conv_block( 32,  64),   #  64 → 32
            conv_block( 64, 128),   #  32 → 16
            conv_block(128, 256),   #  16 →  8
        )
        self.gap = nn.AdaptiveAvgPool2d(1)   # → (B, 256, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


model = TerrainCNN(NUM_CLASSES).to(DEVICE)
print(f"\n[Step 2] Model built — {model.count_params():,} trainable parameters")


# ── 7. Loss, Optimizer, Scheduler ────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(),
                       lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)


# ── 8. Training Loop ──────────────────────────────────────────────────────────
def run_epoch(loader, training=True):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if training:
                optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
    return total_loss / total, correct / total * 100


print(f"\n[Step 3] Training for {NUM_EPOCHS} epochs ...")
history = {"train_loss": [], "val_loss": [],
           "train_acc" : [], "val_acc" : []}
best_val_acc = 0.0
t_start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, training=True)
    vl_loss, vl_acc = run_epoch(val_loader,   training=False)
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["val_loss"  ].append(vl_loss)
    history["train_acc" ].append(tr_acc)
    history["val_acc"   ].append(vl_acc)

    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), MODEL_PATH)
        tag = " ← best"
    else:
        tag = ""

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3}/{NUM_EPOCHS} | "
              f"Train Loss {tr_loss:.4f}  Acc {tr_acc:.1f}% | "
              f"Val Loss {vl_loss:.4f}  Acc {vl_acc:.1f}%{tag}")

elapsed = time.time() - t_start
print(f"\n  Training complete in {elapsed/60:.1f} min")
print(f"  Best Val Accuracy : {best_val_acc:.2f}%")
print(f"  Model saved       → {MODEL_PATH}")


# ── 9. Test Evaluation ────────────────────────────────────────────────────────
print("\n[Step 4] Evaluating on test set ...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs   = imgs.to(DEVICE)
        out    = model(imgs)
        probs  = torch.softmax(out, dim=1)
        preds  = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

test_acc = (all_preds == all_labels).mean() * 100
p, r, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="weighted", zero_division=0, labels=range(NUM_CLASSES)
)

print(f"  Test Accuracy  : {test_acc:.2f}%")
print(f"  Precision      : {p*100:.2f}%")
print(f"  Recall         : {r*100:.2f}%")
print(f"  F1 Score       : {f1*100:.2f}%")
print("\n  Per-Class Report:")
print(classification_report(all_labels, all_preds,
                             target_names=class_names, digits=3, zero_division=0, labels=range(NUM_CLASSES)))


# ── 10. Traversability Scoring ────────────────────────────────────────────────
print("[Step 5] Computing traversability scores ...")

def get_traversability(class_name):
    return TRAVERSABILITY.get(class_name.lower(), 1)  # default Risky

pred_trav  = np.array([get_traversability(class_names[p]) for p in all_preds])
true_trav  = np.array([get_traversability(class_names[l]) for l in all_labels])
trav_acc   = (pred_trav == true_trav).mean() * 100
print(f"  Traversability Accuracy : {trav_acc:.2f}%")


# ── 11. Save Metrics ──────────────────────────────────────────────────────────
metrics_path = os.path.join(SAVE_DIR, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write("Terrain Traversability Classifier — Metrics\n")
    f.write("=" * 50 + "\n")
    f.write(f"Dataset Mode      : {'Synthetic' if USE_SYNTHETIC else 'RUGD'}\n")
    f.write(f"Total Images      : {total}\n")
    f.write(f"Train / Val / Test: {n_train} / {n_val} / {n_test}\n")
    f.write(f"Num Classes       : {NUM_CLASSES}\n")
    f.write(f"Model Params      : {model.count_params():,}\n")
    f.write(f"Best Val Accuracy : {best_val_acc:.2f}%\n")
    f.write(f"Test Accuracy     : {test_acc:.2f}%\n")
    f.write(f"Precision         : {p*100:.2f}%\n")
    f.write(f"Recall            : {r*100:.2f}%\n")
    f.write(f"F1 Score          : {f1*100:.2f}%\n")
    f.write(f"Traversability Acc: {trav_acc:.2f}%\n")
    f.write(f"Training Time     : {elapsed/60:.1f} min\n")
print(f"  Metrics saved → {metrics_path}")


# ── 12. Plots ─────────────────────────────────────────────────────────────────
print("\n[Step 6] Generating plots ...")

BG    = "#0f1117"
TEXT  = "#e0e0e0"
GRID  = "#2a2a3a"
BLUE  = "#4f8ef7"
ORG   = "#f7a24f"
GREEN = "#4ff7a2"
epochs_x = range(1, NUM_EPOCHS + 1)

fig = plt.figure(figsize=(18, 12), facecolor=BG)
fig.suptitle("Terrain Traversability Classifier — Results",
             fontsize=17, color=TEXT, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

def style_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TEXT, fontsize=11, pad=8)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.5)

# Plot 1 — Training & Val Loss
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Loss Curve")
ax1.plot(epochs_x, history["train_loss"], color=BLUE, lw=2, label="Train")
ax1.plot(epochs_x, history["val_loss"],   color=ORG,  lw=2, label="Val")
ax1.set_xlabel("Epoch", color=TEXT, fontsize=9)
ax1.set_ylabel("Loss",  color=TEXT, fontsize=9)
ax1.legend(facecolor="#1a1a2a", labelcolor=TEXT, fontsize=9)

# Plot 2 — Training & Val Accuracy
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Accuracy Curve")
ax2.plot(epochs_x, history["train_acc"], color=BLUE,  lw=2, label="Train")
ax2.plot(epochs_x, history["val_acc"],   color=GREEN, lw=2, label="Val")
ax2.axhline(test_acc, color=ORG, lw=1.5, linestyle="--",
            label=f"Test ({test_acc:.1f}%)")
ax2.set_xlabel("Epoch", color=TEXT, fontsize=9)
ax2.set_ylabel("Accuracy (%)", color=TEXT, fontsize=9)
ax2.legend(facecolor="#1a1a2a", labelcolor=TEXT, fontsize=9)

# Plot 3 — Confusion Matrix
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(BG)
cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))
short_names = [c[:6] for c in class_names]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=short_names, yticklabels=short_names,
            ax=ax3, cbar=False, annot_kws={"size": 7})
ax3.set_title("Confusion Matrix", color=TEXT, fontsize=11, pad=8)
ax3.tick_params(colors=TEXT, labelsize=7)
ax3.set_xlabel("Predicted", color=TEXT, fontsize=9)
ax3.set_ylabel("True",      color=TEXT, fontsize=9)

# Plot 4 — Per-Class F1
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4, "Per-Class F1 Score")
_, _, f1_per, _ = precision_recall_fscore_support(
    all_labels, all_preds, average=None, zero_division=0, labels=range(NUM_CLASSES)
)
colors_bar = [GREEN if f >= 0.7 else ORG if f >= 0.5 else "#f74f4f"
              for f in f1_per]
bars = ax4.bar(short_names, f1_per * 100, color=colors_bar, alpha=0.85)
ax4.set_ylim(0, 115)
ax4.set_xlabel("Class", color=TEXT, fontsize=9)
ax4.set_ylabel("F1 (%)", color=TEXT, fontsize=9)
ax4.tick_params(axis="x", rotation=45)
for bar, val in zip(bars, f1_per):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val*100:.0f}", ha="center", va="bottom",
             color=TEXT, fontsize=7)

# Plot 5 — Traversability Distribution
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5, "Traversability Distribution (Predicted)")
trav_counts = [np.sum(pred_trav == i) for i in range(3)]
wedges, texts, autotexts = ax5.pie(
    trav_counts, labels=TRAV_LABELS, colors=TRAV_COLORS,
    autopct="%1.1f%%", startangle=90,
    textprops={"color": TEXT, "fontsize": 9},
    wedgeprops={"edgecolor": BG, "linewidth": 2}
)
for at in autotexts: at.set_color(BG); at.set_fontweight("bold")
ax5.set_facecolor(BG)

# Plot 6 — Summary Metrics Bar
ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6, "Final Evaluation Summary")
metric_names = ["Val Acc", "Test Acc", "Precision", "Recall", "F1", "Trav Acc"]
metric_vals  = [best_val_acc, test_acc, p*100, r*100, f1*100, trav_acc]
m_colors     = [GREEN if v >= 80 else ORG if v >= 60 else "#f74f4f"
                for v in metric_vals]
bars2 = ax6.bar(metric_names, metric_vals, color=m_colors, alpha=0.85)
ax6.set_ylim(0, 115)
ax6.tick_params(axis="x", rotation=30)
ax6.set_ylabel("Score (%)", color=TEXT, fontsize=9)
for bar, val in zip(bars2, metric_vals):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val:.1f}", ha="center", va="bottom",
             color=TEXT, fontsize=8, fontweight="bold")

plot_path = os.path.join(SAVE_DIR, "terrain_classifier_results.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  Plot saved → {plot_path}")

print("\n" + "=" * 65)
print("  Pipeline Complete!")
print(f"  Model   → {MODEL_PATH}")
print(f"  Metrics → {metrics_path}")
print(f"  Plots   → {plot_path}")
print("=" * 65)
