"""
================================================================
  Terrain Traversability Classifier — Video Inference
  Input  : Any .mp4 / .avi / .mov video file
  Output : Annotated video with traversability costmap overlay
  Run    : python video_inference.py --video path/to/video.mp4
================================================================
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2

# ── 1. Model Definition ───────────────────────────────────────
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
            conv_block(3, 32), conv_block(32, 64),
            conv_block(64, 128), conv_block(128, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 512),
            nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))

# ── 2. Config ─────────────────────────────────────────────────
CLASS_NAMES  = ['asphalt','bush','concrete','grass','gravel',
                'log','mulch','rock','sand','tree']
NUM_CLASSES  = len(CLASS_NAMES)
TRAVERSABILITY = {
    "concrete":0,"asphalt":0,"gravel":0,
    "grass":1,"mulch":1,"dirt":1,"sand":1,
    "rock":2,"mud":2,"water":2,"bush":2,"tree":2,"log":2
}
# BGR colors for OpenCV overlay
TRAV_COLOR_BGR = {
    0: (100, 220, 80),    # Safe  — Green
    1: (50,  160, 230),   # Risky — Orange (BGR)
    2: (60,   60, 210),   # Avoid — Red
}
TRAV_LABEL = {0: "SAFE", 1: "RISKY", 2: "AVOID"}

PATCH_SIZE  = 128
STRIDE      = 64          # smaller = finer map, slower
BATCH_SIZE  = 64
MODEL_PATH  = "results/terrain_cnn.pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_tf = transforms.Compose([
    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ── 3. Helpers ────────────────────────────────────────────────
def build_heatmap(frame_rgb, model, stride=64):
    """Run sliding-window inference and return BGR heatmap."""
    H, W, _ = frame_rgb.shape
    pil_img  = Image.fromarray(frame_rgb)

    patches, coords = [], []
    for y in range(0, H - PATCH_SIZE + 1, stride):
        for x in range(0, W - PATCH_SIZE + 1, stride):
            patches.append(val_tf(pil_img.crop((x, y, x+PATCH_SIZE, y+PATCH_SIZE))))
            coords.append((x, y))

    if not patches:
        return np.zeros((H, W, 3), dtype=np.uint8)

    preds = []
    with torch.no_grad():
        for i in range(0, len(patches), BATCH_SIZE):
            batch = torch.stack(patches[i:i+BATCH_SIZE]).to(DEVICE)
            preds.extend(batch.shape[0] * [None])   # placeholder
            out = model(batch)
            preds[i:i+batch.shape[0]] = out.argmax(1).cpu().numpy().tolist()

    heat   = np.zeros((H, W, 3), dtype=np.float32)
    counts = np.zeros((H, W, 1), dtype=np.float32)
    for (x, y), p in zip(coords, preds):
        trav  = TRAVERSABILITY.get(CLASS_NAMES[p], 1)
        color = TRAV_COLOR_BGR[trav]              # BGR
        heat  [y:y+PATCH_SIZE, x:x+PATCH_SIZE] += color
        counts[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    counts[counts == 0] = 1
    return (heat / counts).astype(np.uint8)


def draw_legend(frame, alpha_map):
    """Overlay a small legend and dominant traversability label."""
    H, W = frame.shape[:2]

    # Count dominant traversability
    safe_px  = np.sum(alpha_map[:,:,1] > 150)   # green channel
    avoid_px = np.sum(alpha_map[:,:,2] > 150)   # red channel
    risky_px = np.sum(alpha_map[:,:,0] > 150)   # blue channel (our orange has high green+red)

    if avoid_px > safe_px and avoid_px > risky_px:
        dom, dom_col = "AVOID ⚠", (60, 60, 210)
    elif safe_px > risky_px:
        dom, dom_col = "SAFE ✓", (100, 220, 80)
    else:
        dom, dom_col = "RISKY ~", (50, 160, 230)

    # Legend box
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (230, 130), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "Traversability", (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA)
    # Swatches
    for i, (label, color) in enumerate([
            ("Safe",  (100,220,80)),
            ("Risky", (50,160,230)),
            ("Avoid", (60,60,210))]):
        y = 50 + i * 24
        cv2.rectangle(frame, (18, y), (38, y+16), color, -1)
        cv2.putText(frame, label, (46, y+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210,210,210), 1, cv2.LINE_AA)

    # Dominant label (bottom bar)
    cv2.rectangle(frame, (0, H-40), (W, H), (20,20,20), -1)
    cv2.putText(frame, f"Scene: {dom}", (10, H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, dom_col, 2, cv2.LINE_AA)
    return frame


def add_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-110, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)


# ── 4. Main ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True,
                        help="Path to input video (.mp4 / .avi / .mov)")
    parser.add_argument("--stride", type=int, default=64,
                        help="Sliding window stride (default 64). Lower = finer but slower.")
    parser.add_argument("--alpha", type=float, default=0.45,
                        help="Heatmap blend alpha 0-1 (default 0.45)")
    parser.add_argument("--skip", type=int, default=1,
                        help="Process every Nth frame (default 1 = all frames)")
    args = parser.parse_args()

    stride = args.stride

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model weights not found. Run terrain_classifier.py first.")

    # Load model
    print(f"Loading model on {DEVICE} ...")
    model = TerrainCNN(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval()

    # Video I/O
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps     = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs("results/videos", exist_ok=True)
    base     = os.path.splitext(os.path.basename(args.video))[0]
    out_path = f"results/videos/{base}_traversability.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, orig_fps / args.skip, (W*2, H))

    print(f"Input  : {args.video}  ({W}×{H}, {total_frames} frames, {orig_fps:.1f} FPS)")
    print(f"Output : {out_path}")
    print(f"Stride : {stride}  |  Alpha : {args.alpha}  |  Skip : {args.skip}")
    print("Press  Q  in the preview window to stop early.\n")

    frame_idx = 0
    last_heat = None

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        frame_idx += 1
        t0 = time.time()

        if frame_idx % args.skip != 0:
            # Re-use last heatmap for skipped frames
            if last_heat is None:
                continue
            heat_bgr = last_heat
        else:
            rgb      = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            heat_bgr = build_heatmap(rgb, model, stride=stride)
            last_heat = heat_bgr

        # Blend
        blended = cv2.addWeighted(bgr, 1 - args.alpha, heat_bgr, args.alpha, 0)
        blended = draw_legend(blended, heat_bgr)
        fps = 1.0 / max(time.time() - t0, 1e-6)
        add_fps(blended, fps)

        # Side-by-side: original | blended
        side = np.hstack([bgr, blended])
        writer.write(side)

        # Progress
        pct = frame_idx / max(total_frames, 1) * 100
        print(f"\r  Frame {frame_idx:>5}/{total_frames}  ({pct:5.1f}%)  FPS≈{fps:4.1f}", end="", flush=True)

        # Live preview (optional — comment out if running headless)
        cv2.imshow("Traversability | Left=Original  Right=Costmap  [Q=quit]", side)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n  Stopped by user.")
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n\nDone! Annotated video saved to:\n  {out_path}")


if __name__ == "__main__":
    main()
