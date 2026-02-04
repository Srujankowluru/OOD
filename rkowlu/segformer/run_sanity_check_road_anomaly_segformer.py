"""
Sanity check: Road Anomaly (EPFL) dataset with SegFormer — same data as DeepLab sanity check,
then run all OOD methods with SegFormer.

Uses RoadAnomaly/frames (default: ../DeepLabV3Plus-Pytorch/RoadAnomaly/frames). Same mask format:
frame_name.labels/labels_semantic.png (background=0, anomaly=2). Binary: 0=ID, 255=OOD.

Usage:
  cd /visinf/projects_students/groupL/rkowlu/segformer
  python run_sanity_check_road_anomaly_segformer.py
  python run_sanity_check_road_anomaly_segformer.py --max-images 20
  python run_sanity_check_road_anomaly_segformer.py --data-dir /path/to/RoadAnomaly/frames

Outputs:
  - road_anomaly_list.txt (image paths)
  - road_anomaly_masks/ (binary masks: 0=ID, 255=OOD)
  - results_road_anomaly_sanity_segformer/ (JSON, CSV, plot, test_images/)
"""

import os
import sys
import subprocess
import argparse
import numpy as np
from glob import glob
from PIL import Image

# Same image extensions as DeepLab sanity check (no videos)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")
OUTPUT_LIST = "road_anomaly_list.txt"
MASK_STAGING_DIR = "road_anomaly_masks"
OUTPUT_DIR = "results_road_anomaly_sanity_segformer"

# Default: same Road Anomaly frames as DeepLab (sibling DeepLabV3Plus-Pytorch/RoadAnomaly/frames)
SEGFORMER_ROOT = os.path.dirname(os.path.abspath(__file__))
DEEPLAB_ROOT = os.path.join(os.path.dirname(SEGFORMER_ROOT), "DeepLabV3Plus-Pytorch")
DEFAULT_FRAMES_DIR = os.path.join(DEEPLAB_ROOT, "RoadAnomaly", "frames")


def find_images_and_masks(root):
    """
    Collect (image_path, mask_path) pairs. Only image files (no videos).
    EPFL Road Anomaly: frame_name.webp + frame_name.labels/labels_semantic.png
    (background=0, anomaly=2 — anything not Cityscapes = OOD).
    """
    root = os.path.abspath(root)
    pairs = []
    for ext in IMG_EXTS:
        for img_path in glob(os.path.join(root, "*" + ext)):
            img_path = os.path.normpath(img_path)
            stem = os.path.splitext(os.path.basename(img_path))[0]
            parent = os.path.dirname(img_path)
            mask_candidates = [
                os.path.join(parent, stem + ".labels", "labels_semantic.png"),
                os.path.join(parent, stem + ".labels", "labels_semantic.PNG"),
                os.path.join(parent, stem + ".png"),
                os.path.join(root, "masks", stem + ".png"),
                os.path.join(root, "labels", stem + ".png"),
                os.path.join(root, "gt", stem + ".png"),
            ]
            mask_path = None
            for m in mask_candidates:
                if os.path.isfile(m):
                    mask_path = m
                    break
            pairs.append((img_path, mask_path))
    return pairs


def prepare_masks(pairs, staging_dir):
    """
    Write staging_dir/stem.png with binary mask for run_all_methods_segformer.
    EPFL: 0 = background (ID), non-zero = anomaly (OOD). Remap: 0 → 0, non-zero → 255.
    """
    os.makedirs(staging_dir, exist_ok=True)
    prepared = 0
    for img_path, mask_path in pairs:
        if mask_path is None:
            continue
        stem = os.path.splitext(os.path.basename(img_path))[0]
        dst = os.path.join(staging_dir, stem + ".png")
        if os.path.abspath(mask_path) == os.path.abspath(dst):
            prepared += 1
            continue
        arr = np.array(Image.open(mask_path).convert("L"))
        binary = np.where(arr > 0, 255, 0).astype(np.uint8)
        Image.fromarray(binary).save(dst)
        prepared += 1
    return prepared


def main():
    parser = argparse.ArgumentParser(
        description="Sanity check: Road Anomaly (EPFL) + SegFormer, all OOD methods"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_FRAMES_DIR,
        help=f"Road Anomaly frames directory (default: {DEFAULT_FRAMES_DIR})",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Max images to run (default: all)")
    parser.add_argument("--gallery-dir", type=str, default="./galleries", help="SegFormer gallery dir")
    parser.add_argument("--max-size", type=int, default=512, help="Max image dimension (default 512 to avoid ReAct/ACTSUB OOM)")
    parser.add_argument("--no-low-mem", action="store_true", help="Disable low-mem (default: low-mem on to avoid ReAct OOM)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 on GPU (ReAct/ASH/ACTSUB run in float32)")
    args = parser.parse_args()

    os.chdir(SEGFORMER_ROOT)

    path = os.path.abspath(args.data_dir)
    if not os.path.isdir(path):
        print("Frames directory not found:", path)
        print("Default expects DeepLabV3Plus-Pytorch/RoadAnomaly/frames (sibling of segformer).")
        sys.exit(1)

    pairs = find_images_and_masks(path)
    with_mask = [(img, m) for img, m in pairs if m is not None]
    without_mask = len(pairs) - len(with_mask)
    if without_mask:
        print(f"Skipped {without_mask} images without mask (only images with masks are used).")

    if not with_mask:
        print("No image+mask pairs found under", path)
        print("Expected: frame_name.webp (or .png/.jpg) and frame_name.labels/labels_semantic.png")
        sys.exit(1)

    print(f"Found {len(with_mask)} images with masks (Road Anomaly, image files only)")

    if args.max_images is not None:
        with_mask = with_mask[: args.max_images]
        print(f"Using first {len(with_mask)} images (--max-images)")

    n_prepared = prepare_masks(with_mask, MASK_STAGING_DIR)
    mask_dir = os.path.abspath(MASK_STAGING_DIR)
    print(f"Prepared {n_prepared} masks in {MASK_STAGING_DIR}/ (binary: 0=ID, 255=OOD)")

    image_paths = [p[0] for p in with_mask]
    list_path = os.path.join(SEGFORMER_ROOT, OUTPUT_LIST)
    with open(list_path, "w") as f:
        for p in image_paths:
            f.write(p + "\n")
    print(f"Wrote {OUTPUT_LIST} ({len(image_paths)} images)")

    cmd = [
        sys.executable,
        "run_all_methods_segformer.py",
        "--image-list", list_path,
        "--mask-dir", mask_dir,
        "--output-dir", OUTPUT_DIR,
        "--gallery-dir", os.path.abspath(args.gallery_dir),
        "--max-size", str(args.max_size),
        "--plot",
        "--save-vis", "20",
    ]
    if not args.no_low_mem:
        cmd.append("--low-mem")
    if args.fp16:
        cmd.append("--fp16")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=SEGFORMER_ROOT)
    print("Done. Results in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
