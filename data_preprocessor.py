import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

ANN_DIR      = 'RUGD_sample-data/annotations'
IMG_DIR      = 'RUGD_sample-data/images'
COLORMAP_PATH= 'RUGD_sample-data/RUGD_annotation-colormap.txt'
OUTPUT_DIR   = 'rugd_dataset'
PATCH_SIZE   = 128
THRESHOLD    = 0.50  # More aggressive
MAX_PATCHES  = 200   # More patches

# Load Colormap
colormap = {}
with open(COLORMAP_PATH, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 5:
            name = parts[1].lower()
            rgb = (int(parts[2]), int(parts[3]), int(parts[4]))
            colormap[rgb] = name

class_map = {"rock-bed": "rock"}
# Categories we WANT to keep for the terrain classifier
target_classes = ["concrete", "asphalt", "gravel", "grass", "mulch", "dirt", 
                  "sand", "rock", "mud", "water", "bush", "tree", "log"]

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
for cls in target_classes:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

def extract_patches():
    files = [f for f in os.listdir(ANN_DIR) if f.endswith('.png')]
    patch_counts = {cls: 0 for cls in target_classes}

    for f in tqdm(files):
        ann_path = os.path.join(ANN_DIR, f)
        img_path = os.path.join(IMG_DIR, f)
        if not os.path.exists(img_path): continue
        
        ann = Image.open(ann_path).convert('RGB')
        img = Image.open(img_path).convert('RGB')
        ann_arr, img_arr = np.array(ann), np.array(img)
        H, W, _ = ann_arr.shape
        
        per_img_counts = {cls: 0 for cls in target_classes}
        
        # Dense grid search
        for y in range(0, H - PATCH_SIZE, 8):
            for x in range(0, W - PATCH_SIZE, 8):
                patch_ann = ann_arr[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                pixels = patch_ann.reshape(-1, 3)
                
                pixels_32 = pixels.astype(np.int32)
                pixels_1d = pixels_32[:, 0] * 65536 + pixels_32[:, 1] * 256 + pixels_32[:, 2]
                colors_1d, counts = np.unique(pixels_1d, return_counts=True)
                idx = np.argmax(counts)
                dom = int(colors_1d[idx])
                dominant_rgb = (dom // 65536, (dom // 256) % 256, dom % 256)
                cls_name = colormap.get(dominant_rgb)
                
                if cls_name:
                    cls_name = class_map.get(cls_name, cls_name)
                    if cls_name in target_classes and counts[idx] >= (PATCH_SIZE**2 * THRESHOLD):
                        if per_img_counts[cls_name] < MAX_PATCHES:
                            per_img_counts[cls_name] += 1
                            patch_counts[cls_name] += 1
                            patch_img = Image.fromarray(img_arr[y:y+PATCH_SIZE, x:x+PATCH_SIZE])
                            patch_img.save(os.path.join(OUTPUT_DIR, cls_name, f"{f[:-4]}_{y}_{x}.png"))

    print("\nPatch extraction complete!")
    for cls, count in patch_counts.items():
        if count > 0:
            print(f"  {cls:15}: {count:5} patches")
        else:
            # Delete empty dirs
            os.rmdir(os.path.join(OUTPUT_DIR, cls))

if __name__ == "__main__":
    extract_patches()
