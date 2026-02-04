"""
Create publication-quality OOD visualizations from existing *_ood.png maps.

1. Overlay: RGB + jet OOD map (alpha=0.5) + colorbar → *_overlay.png
2. Binary mask: threshold (e.g. 95th percentile), binarize, optional smoothing → *_binary.png
   (0 = ID, 1 = OOD; grayscale, no heatmap)

Uses existing *_ood.png in test_images/<method>/ and the image list for RGB paths.
Saves *_overlay.png and *_binary.png in the same folder.

Usage:
  python create_ood_overlays.py --image-list diverse_200_list.txt --test-images-dir results_200/test_images
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None


def build_basename_to_path(image_list_path):
    """Map image basename (no extension) -> full path."""
    out = {}
    with open(image_list_path, "r") as f:
        for line in f:
            path = line.strip()
            if not path:
                continue
            base = os.path.basename(path)
            name = os.path.splitext(base)[0]
            out[name] = path
    return out


def save_overlay(rgb_np, ood_np, out_path, alpha=0.5, cmap="jet"):
    """Overlay OOD map on RGB: plt.imshow(rgb), plt.imshow(ood, cmap=jet, alpha), colorbar."""
    # ood_np: grayscale 0-255 or float; normalize to [0,1] for colormap
    s = np.asarray(ood_np, dtype=np.float64)
    if s.ndim > 2:
        s = s[:, :, 0]
    valid = np.isfinite(s)
    if valid.any():
        lo, hi = np.percentile(s[valid], [1, 99])
        if hi <= lo:
            hi = lo + 1e-6
        s = np.clip((s - lo) / (hi - lo), 0, 1)
    else:
        s = np.zeros_like(s)
    # Upscale OOD map to RGB size for full-resolution overlay
    h_rgb, w_rgb = rgb_np.shape[0], rgb_np.shape[1]
    if s.shape[0] != h_rgb or s.shape[1] != w_rgb:
        s_pil = Image.fromarray((np.clip(s * 255, 0, 255).astype(np.uint8)))
        s_pil = s_pil.resize((w_rgb, h_rgb), Image.BILINEAR)
        s = np.array(s_pil, dtype=np.float64) / 255.0
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(rgb_np)
    im = ax.imshow(s, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="OOD score")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def save_binary_mask(ood_np, out_path, percentile=95, sigma=1.0, threshold=None):
    """Binarize: 0=ID, 1=OOD. Use threshold if given (pixel-level ID 95th pct); else per-image percentile."""
    s = np.asarray(ood_np, dtype=np.float64)
    if s.ndim > 2:
        s = s[:, :, 0]
    valid = np.isfinite(s)
    if not valid.any():
        ood_mask = np.zeros_like(s)
    else:
        if threshold is None:
            threshold = np.percentile(s[valid], percentile)
        ood_mask = (s > threshold).astype(np.float64)
    if sigma > 0 and gaussian_filter is not None:
        ood_mask = gaussian_filter(ood_mask, sigma=sigma)
        ood_mask = (ood_mask > 0.5).astype(np.float64)  # re-binarize after smoothing
    elif sigma > 0:
        pass  # no scipy: skip smoothing
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(ood_mask, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create RGB+OOD overlay figures from existing _ood.png maps")
    parser.add_argument("--image-list", type=str, default="diverse_200_list.txt",
                        help="Image list used to generate the OOD maps (to find RGB paths)")
    parser.add_argument("--test-images-dir", type=str, default="results_200/test_images",
                        help="Root dir containing method folders with *_ood.png")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for OOD overlay (0-1)")
    parser.add_argument("--suffix", type=str, default="_overlay", help="Output suffix for overlay (default: _overlay -> *_overlay.png)")
    parser.add_argument("--no-binary", action="store_true", help="Skip binary mask generation")
    parser.add_argument("--binary-percentile", type=float, default=95,
                        help="Percentile for binary threshold (default: 95, FPR@95 style)")
    parser.add_argument("--binary-sigma", type=float, default=1.0,
                        help="Gaussian smoothing sigma for binary mask (0 = no smoothing)")
    parser.add_argument("--thresholds-file", type=str, default=None,
                        help="JSON with pixel-level ID 95th percentile per method (from run_all_methods; e.g. results_200/method_thresholds.json)")
    args = parser.parse_args()

    # Load pixel-level ID thresholds if available (correct binary calibration)
    method_thresholds = {}
    if args.thresholds_file and os.path.isfile(args.thresholds_file):
        import json
        with open(args.thresholds_file, "r") as f:
            method_thresholds = json.load(f)
        print(f"Loaded {len(method_thresholds)} method thresholds from {args.thresholds_file}")
    else:
        # Infer: test_images is inside output dir; method_thresholds.json is in parent
        parent = os.path.dirname(os.path.normpath(args.test_images_dir))
        default_thresh = os.path.join(parent, "method_thresholds.json")
        if os.path.isfile(default_thresh):
            import json
            with open(default_thresh, "r") as f:
                method_thresholds = json.load(f)
            print(f"Loaded {len(method_thresholds)} method thresholds from {default_thresh}")

    basename_to_path = build_basename_to_path(args.image_list)
    print(f"Loaded {len(basename_to_path)} image paths from {args.image_list}")

    test_dir = args.test_images_dir
    if not os.path.isdir(test_dir):
        print(f"Not a directory: {test_dir}")
        return

    method_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    total = 0
    for method in sorted(method_dirs):
        method_path = os.path.join(test_dir, method)
        ood_files = [f for f in os.listdir(method_path) if f.endswith("_ood.png")]
        for fname in tqdm(ood_files, desc=method, leave=False):
            # Basename of original image (e.g. 4GgyoCqnDCI67PiA8hOB3A)
            stem = fname.replace("_ood.png", "")
            if stem not in basename_to_path:
                continue
            rgb_path = basename_to_path[stem]
            if not os.path.exists(rgb_path):
                continue
            ood_path = os.path.join(method_path, fname)
            out_name = stem + args.suffix + ".png"
            out_path = os.path.join(method_path, out_name)
            try:
                rgb = np.array(Image.open(rgb_path).convert("RGB"))
                ood = np.array(Image.open(ood_path))
                if ood.ndim > 2:
                    ood = ood[:, :, 0]
                save_overlay(rgb, ood, out_path, alpha=args.alpha)
                if not args.no_binary:
                    binary_path = os.path.join(method_path, stem + "_binary.png")
                    # Use pixel-level ID threshold if available (correct calibration)
                    scores_path = os.path.join(method_path, stem + "_scores.npy")
                    thresh = method_thresholds.get(method)
                    if thresh is not None and os.path.isfile(scores_path):
                        raw_scores = np.load(scores_path)
                        save_binary_mask(
                            raw_scores, binary_path,
                            percentile=args.binary_percentile,
                            sigma=args.binary_sigma,
                            threshold=float(thresh),
                        )
                    else:
                        # Fallback: per-image percentile (wrong statistically; use --thresholds-file)
                        if not method_thresholds and total == 0:
                            print("  (No method_thresholds.json: binary masks use per-image percentile; re-run run_all_methods with --save-vis to get pixel-level ID thresholds.)")
                        save_binary_mask(
                            ood, binary_path,
                            percentile=args.binary_percentile,
                            sigma=args.binary_sigma,
                        )
                total += 1
            except Exception as e:
                continue
    print(f"Saved {total} overlay (and binary) images under {test_dir}")


if __name__ == "__main__":
    main()
