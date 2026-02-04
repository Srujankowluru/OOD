"""
Sanity check: local Road Anomaly (EPFL) dataset — images + masks, then run all OOD methods.

Uses RoadAnomaly/frames: only image files (skip videos / non-images). Masks from
frame_name.labels/labels_semantic.png (background=0, anomaly=2). Anything not in
Cityscapes (anomaly) is OOD; rest is ID.

Usage:
  cd /visinf/projects_students/groupL/rkowlu/DeepLabV3Plus-Pytorch
  python run_sanity_check_road_anomaly.py
  python run_sanity_check_road_anomaly.py --max-images 20

Outputs:
  - road_anomaly_list.txt (image paths)
  - road_anomaly_masks/ (binary masks: 0=ID, 255=OOD)
  - new_deep_sanity_results/ (JSON, CSV, plot, test_images/)
"""

import os
import sys
import subprocess
import argparse

try:
    import numpy as np
except ModuleNotFoundError:
    print("numpy not found. Use the ood_seg environment and run with its Python:")
    print("  conda activate ood_seg")
    print("  cd /visinf/projects_students/groupL/rkowlu/DeepLabV3Plus-Pytorch")
    print("  $CONDA_PREFIX/bin/python run_sanity_check_road_anomaly.py")
    print("If that fails, reinstall packages in ood_seg:")
    print("  conda activate ood_seg")
    print("  pip install -r requirements.txt")
    sys.exit(1)

from glob import glob
from PIL import Image


def _python_for_subprocess():
    """Use conda env's Python for run_all_methods when CONDA_PREFIX is set (e.g. after conda activate ood_seg)."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        for name in ("python3", "python"):
            exe = os.path.join(conda_prefix, "bin", name)
            if os.path.isfile(exe):
                return exe
    return sys.executable


# Only image extensions (skip videos and other non-image files)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")
OUTPUT_LIST = "road_anomaly_list.txt"
MASK_STAGING_DIR = "road_anomaly_masks"
OUTPUT_DIR = "new_deep_sanity_results"

# Default DeepLab checkpoint for this sanity check
DEFAULT_MODEL_PATH = "/visinf/projects_students/groupL/rkowlu/best_deeplabv3plus_resnet101_cityscapes_os16.pth (1).tar"

# Default: local RoadAnomaly/frames next to this script
DEFAULT_FRAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RoadAnomaly", "frames")


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
            # EPFL: frame.labels/labels_semantic.png; fallbacks for other layouts
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
    Write staging_dir/stem.png with binary mask for run_all_methods.
    run_all_methods uses (mask > 127) for OOD.
    EPFL: 0 = background (ID / Cityscapes), 2 = anomaly (OOD). Remap: 0 → 0, non-zero → 255.
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
        # 0 = ID (Cityscapes-like), non-zero = OOD (anomaly)
        binary = np.where(arr > 0, 255, 0).astype(np.uint8)
        Image.fromarray(binary).save(dst)
        prepared += 1
    return prepared


def main():
    parser = argparse.ArgumentParser(description="Sanity check: Road Anomaly (local) + all OOD methods")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_FRAMES_DIR,
                        help=f"Frames directory (default: {DEFAULT_FRAMES_DIR})")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to run (default: all)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"DeepLab model checkpoint path (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--only-missing", action="store_true",
                        help="Only run methods not in output-dir/results_summary.json; merge new results with existing")
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated methods to run (e.g. Mahalanobis,Mahalanobis++,kNN,VIM); default: all or only missing")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    path = os.path.abspath(args.data_dir)
    if not os.path.isdir(path):
        print("Frames directory not found:", path)
        sys.exit(1)

    pairs = find_images_and_masks(path)
    # Only use frames that have masks (required for proper AUROC; OOD = not in Cityscapes)
    with_mask = [(img, m) for img, m in pairs if m is not None]
    without_mask = len(pairs) - len(with_mask)
    if without_mask:
        print(f"Skipped {without_mask} images without mask (only images with masks are used).")

    if not with_mask:
        print("No image+mask pairs found under", path)
        print("Expected: frame_name.webp (or .png/.jpg) and frame_name.labels/labels_semantic.png")
        sys.exit(1)

    print(f"Found {len(with_mask)} images with masks (only image files, no videos)")

    if args.max_images is not None:
        with_mask = with_mask[: args.max_images]
        print(f"Using first {len(with_mask)} images (--max-images)")

    n_prepared = prepare_masks(with_mask, MASK_STAGING_DIR)
    mask_dir = os.path.abspath(MASK_STAGING_DIR)
    print(f"Prepared {n_prepared} masks in {MASK_STAGING_DIR}/ (binary: 0=ID, 255=OOD)")

    image_paths = [p[0] for p in with_mask]
    with open(OUTPUT_LIST, "w") as f:
        for p in image_paths:
            f.write(p + "\n")
    print(f"Wrote {OUTPUT_LIST} ({len(image_paths)} images with masks)")

    cmd = [
        _python_for_subprocess(),
        "run_all_methods.py",
        "--model", args.model,
        "--image-list", OUTPUT_LIST,
        "--mask-dir", mask_dir,
        "--output-dir", OUTPUT_DIR,
        "--gallery-dir", ".",
        "--plot",
        "--save-vis", "20",
    ]
    if getattr(args, "only_missing", False):
        cmd.append("--only-missing")
    if getattr(args, "methods", None):
        cmd.extend(["--methods", args.methods])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=script_dir)
    print("Done. Results in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
