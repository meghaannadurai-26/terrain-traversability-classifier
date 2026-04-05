import os
import glob
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ── 1. Model Definition ───────────────────────────────────────────────────────
class TerrainCNN(nn.Module):
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
        self.gap = nn.AdaptiveAvgPool2d(1)
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

# ── 2. Configuration & Mapping ─────────────────────────────────────────────
CLASS_NAMES = ['asphalt', 'bush', 'concrete', 'grass', 'gravel', 'log', 'mulch', 'rock', 'sand', 'tree']
NUM_CLASSES = len(CLASS_NAMES)

TRAVERSABILITY = {
    "concrete": 0, "asphalt": 0, "gravel": 0,    # Safe
    "grass": 1, "mulch": 1, "dirt": 1, "sand": 1, # Risky
    "rock": 2, "mud": 2, "water": 2, "bush": 2, "tree": 2, "log": 2 # Avoid
}

# (R, G, B) colors for overlay
COLOR_MAP = {
    0: (79, 247, 162),   # Safe: Greenish
    1: (247, 162, 79),   # Risky: Orange/Yellowish
    2: (247, 79, 79)     # Avoid: Reddish
}

PATCH_SIZE = 128
STRIDE = 64
BATCH_SIZE = 64
MODEL_PATH = "results/terrain_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([
    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def main():
    parser = argparse.ArgumentParser(description="Live Inference Costmap Generator")
    parser.add_argument("--image", type=str, help="Path to the input image. If empty, picks a random sample.")
    args = parser.parse_args()

    # Load Model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}. Train the model first.")
    
    print(f"Loading model on {DEVICE}...")
    model = TerrainCNN(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Determine input image
    if args.image and os.path.exists(args.image):
        img_path = args.image
    else:
        sample_dir = "RUGD_sample-data/images"
        images = glob.glob(os.path.join(sample_dir, "*.png"))
        if not images:
            raise FileNotFoundError("No sample images found and no valid --image provided.")
        img_path = random.choice(images)

    print(f"Processing image: {img_path}")
    orig_img = Image.open(img_path).convert("RGB")
    orig_arr = np.array(orig_img)
    H, W, _ = orig_arr.shape

    # Prepare patches
    print("Extracting patches for sliding window...")
    patches = []
    coords = []
    
    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):
            patch = orig_img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
            patches.append(val_transform(patch))
            coords.append((x, y))

    # Inference in batches 
    print(f"Running inference on {len(patches)} patches...")
    predictions = []
    with torch.no_grad():
        for i in range(0, len(patches), BATCH_SIZE):
            batch = torch.stack(patches[i:i+BATCH_SIZE]).to(DEVICE)
            out = model(batch)
            preds = out.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)

    # Reconstruct Map
    print("Reconstructing costmap overlay...")
    heatmap = np.zeros((H, W, 3), dtype=np.float32)
    counts = np.zeros((H, W, 1), dtype=np.float32)

    for (x, y), pred_idx in zip(coords, predictions):
        cls_name = CLASS_NAMES[pred_idx]
        trav_score = TRAVERSABILITY[cls_name]
        color = COLOR_MAP[trav_score]
        
        heatmap[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += color
        counts[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    # Average overlapping areas
    counts[counts == 0] = 1 # avoid dividing by 0
    heatmap /= counts
    heatmap = heatmap.astype(np.uint8)

    # Alpha Blending
    alpha = 0.45
    blended = cv2.addWeighted(orig_arr, 1.0 - alpha, heatmap, alpha, 0) if "cv2" in globals() else (orig_arr * (1 - alpha) + heatmap * alpha).astype(np.uint8)

    # Visualization
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Traversability Costmap Inference", fontsize=16, fontweight="bold")
    
    axs[0].imshow(orig_arr)
    axs[0].set_title("Original Frame")
    axs[0].axis('off')

    axs[1].imshow(heatmap)
    axs[1].set_title("Nav2 Costmap (Green=Safe, Orange=Risky, Red=Avoid)")
    axs[1].axis('off')

    axs[2].imshow(blended)
    axs[2].set_title("Overlay Blended")
    axs[2].axis('off')

    os.makedirs("results/costmaps", exist_ok=True)
    out_file = os.path.join("results/costmaps", "costmap_" + os.path.basename(img_path))
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"Costmap saved successfully to: {out_file}")

if __name__ == "__main__":
    main()
