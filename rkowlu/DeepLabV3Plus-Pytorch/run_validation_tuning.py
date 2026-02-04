"""
Validation protocol: tune hyperparameters on validation set, then evaluate once on test.

Rule: Never tune on test. Tune on validation → freeze hyperparams → evaluate on test.
See METHODOLOGY.md for full protocol.

Usage:
  # Energy: tune temperature T (controls tail sharpness, FPR@95)
  python run_validation_tuning.py --method Energy --metric AUROC --search "T=0.5,1.0,2.0,5.0"

  # ACTSUB: tune percentile, k (decisive channels), lambda (suppression). 3×2×2 = 12 runs.
  python run_validation_tuning.py --method ACTSUB --metric AUROC --search "percentile=50,60,70 k=32,64 lambda=0.2,0.5"

  # Mahalanobis++: 4 runs (layer3/layer4 x PCA 64/128), ε fixed 1e-2
  python run_validation_tuning.py --method Mahalanobis++

  # Use default grids (no --search)
  python run_validation_tuning.py --method ASH
  python run_validation_tuning.py --method Energy
"""

import torch
import numpy as np
from tqdm import tqdm
import os
import argparse
from sklearn.metrics import roc_auc_score, roc_curve

from run_all_methods import (
    load_model,
    load_galleries,
    process_image,
    DEFAULT_MODEL_PATH,
    DEFAULT_MASK_DIR,
)
from ood_methods import get_ood_score, AVAILABLE_METHODS
from torchvision import transforms
from PIL import Image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default hyperparameter grids per method (tune on validation)
# Energy: T only (no percentile — that is ASH). T controls tail sharpness, FPR@95.
METHOD_GRIDS = {
    "ASH": [{"percentile": p} for p in [65, 80, 90, 95]],
    "ACTSUB": [
        {"percentile": p, "k": k, "lambda_": lam}
        for p in [50, 60, 70]
        for k in [32, 64]
        for lam in [0.2, 0.5]
    ],
    "Energy": [{"T": T} for T in [0.5, 1.0, 2.0, 5.0]],
    "SCALE": [{"T": T, "beta": b} for T in [1.0] for b in [0.3, 0.5, 1.0]],
    "KL_Matching": [{"T": T} for T in [0.5, 1.0, 2.0]],
    "DICE": [{"p": p} for p in [80, 90, 95]],
    "Mahalanobis": [{"alpha": a} for a in [0.001, 0.01, 0.1]],
    "Mahalanobis++": [
        {"layer": layer, "n_components": n} for layer in ["layer3", "layer4"] for n in [64, 128]
    ],
    "VIM": [{"alpha": a} for a in [0.5, 1.0, 2.0]],
    "kNN": [{"k": k} for k in [5, 10, 20, 50]],
}


def _parse_val(s):
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def build_grid_from_search_fixed(search_str, fixed_str):
    """Build list of param dicts from --search 'k1=v1,v2 k2=v3' and --fixed 'k3=v0'."""
    from itertools import product
    search_str = (search_str or "").strip()
    fixed_str = (fixed_str or "").strip()
    fixed = {}
    if fixed_str:
        for part in fixed_str.split():
            if "=" in part:
                key, val = part.split("=", 1)
                fixed[key.strip()] = _parse_val(val.strip())
    if not search_str:
        return [fixed] if fixed else []
    # Parse search: "T=0.5,1.0,2.0,5.0" or "percentile=50,60,70 k=32,64 lambda=0.2,0.5"
    keys_vals = []
    for part in search_str.split():
        if "=" not in part:
            continue
        key, vals_str = part.split("=", 1)
        key = key.strip()
        vals = [_parse_val(v.strip()) for v in vals_str.split(",")]
        keys_vals.append((key, vals))
    if not keys_vals:
        return [fixed] if fixed else []
    keys = [kv[0] for kv in keys_vals]
    val_lists = [kv[1] for kv in keys_vals]
    grid = []
    for combo in product(*val_lists):
        point = dict(zip(keys, combo))
        if "lambda" in point and "lambda_" not in point:
            point["lambda_"] = point.pop("lambda")
        point.update(fixed)
        grid.append(point)
    return grid


def load_image_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def evaluate_scores(scores_list, labels_list, metric="AUROC"):
    """Compute AUROC or FPR@95%TPR from flattened scores and labels."""
    all_scores = np.concatenate([s.flatten() for s in scores_list])
    all_labels = np.concatenate([l.flatten() for l in labels_list])
    valid = np.isfinite(all_scores)
    all_scores = all_scores[valid]
    all_labels = all_labels[valid]
    if len(all_scores) == 0:
        return None
    auroc = roc_auc_score(all_labels, all_scores)
    if auroc < 0.5:
        all_scores = -all_scores
        auroc = roc_auc_score(all_labels, all_scores)
    if metric.upper() == "AUROC":
        return auroc
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    idx = np.where(tpr >= 0.95)[0]
    fpr95 = fpr[idx[0]] if len(idx) > 0 else 1.0
    return fpr95 if metric.upper() == "FPR95" else auroc


def run_method_on_list(model, galleries, image_paths, mask_dir, transform, method, max_size, **method_kwargs):
    """Run one method on a list of images; return (scores_list, labels_list)."""
    scores_list = []
    labels_list = []
    for img_path in tqdm(image_paths, desc=f"{method}", leave=False):
        try:
            scores, _ = process_image(
                model, img_path, transform, method, galleries, max_size=max_size, **method_kwargs
            )
            if scores is None:
                continue
            torch.cuda.empty_cache()
        except Exception:
            continue
        base_name = os.path.basename(img_path).replace(".jpg", ".png")
        mask_path = os.path.join(mask_dir, base_name)
        if not os.path.exists(mask_path):
            continue
        mask_img = Image.open(mask_path).convert("L")
        mask_gt = (np.array(mask_img) > 127).astype(np.uint8)
        sh, sw = scores.shape[0], scores.shape[1]
        if (sw, sh) != mask_gt.shape[::-1]:
            mask_resized = mask_img.resize((sw, sh), Image.NEAREST)
            mask_gt = (np.array(mask_resized) > 127).astype(np.uint8)
        scores_list.append(scores)
        labels_list.append(mask_gt)
    return scores_list, labels_list


def main():
    parser = argparse.ArgumentParser(description="Tune on validation, evaluate on test")
    parser.add_argument("--method", type=str, default="ASH", help="Method name")
    parser.add_argument("--val-list", type=str, default="val_list.txt", help="Validation image list")
    parser.add_argument("--test-list", type=str, default="test_list.txt", help="Test image list")
    parser.add_argument("--mask-dir", type=str, default=DEFAULT_MASK_DIR, help="OOD mask directory")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Model checkpoint")
    parser.add_argument("--gallery-dir", type=str, default="./", help="Gallery directory")
    parser.add_argument("--metric", type=str, default="AUROC", choices=["AUROC", "FPR95"],
                        help="Metric to maximize (AUROC) or minimize (FPR95)")
    parser.add_argument("--max-size", type=int, default=1024, help="Max image dimension")
    parser.add_argument("--max-val", type=int, default=None, help="Max validation images (for speed)")
    parser.add_argument("--max-test", type=int, default=None, help="Max test images")
    parser.add_argument("--search", type=str, default=None,
                        help="Override grid: e.g. 'T=0.5,1.0,2.0,5.0' or 'percentile=50,60,70 k=32,64 lambda=0.2,0.5'")
    parser.add_argument("--fixed", type=str, default=None,
                        help="Fixed params for every run: e.g. 'percentile=65' (Energy ignores unknown params)")
    args = parser.parse_args()

    method = args.method
    if method not in AVAILABLE_METHODS:
        print(f"Unknown method: {method}. Available: {AVAILABLE_METHODS}")
        return
    if args.search:
        grid = build_grid_from_search_fixed(args.search, args.fixed or "")
        print(f"Grid from --search/--fixed: {len(grid)} runs")
    elif method not in METHOD_GRIDS:
        print(f"No grid defined for {method}. Add grid in METHOD_GRIDS in this script.")
        print("Running with default hyperparameters on val then test (no tuning).")
        grid = [{}]
    else:
        grid = METHOD_GRIDS[method]
        if args.fixed:
            fixed = {}
            for part in (args.fixed or "").strip().split():
                if "=" in part:
                    k, v = part.split("=", 1)
                    try:
                        v = int(v) if "." not in v else float(v)
                    except ValueError:
                        pass
                    fixed[k.strip()] = v
            grid = [{**p, **fixed} for p in grid]

    model = load_model(args.model)
    galleries = load_galleries(args.gallery_dir)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_paths = load_image_list(args.val_list)
    test_paths = load_image_list(args.test_list)
    if args.max_val:
        val_paths = val_paths[: args.max_val]
    if args.max_test:
        test_paths = test_paths[: args.max_test]

    # Tune on validation
    best_params = None
    best_val_metric = -np.inf if args.metric.upper() == "AUROC" else np.inf
    for params in grid:
        scores_list, labels_list = run_method_on_list(
            model, galleries, val_paths, args.mask_dir, transform,
            method, args.max_size, **params
        )
        if not scores_list:
            continue
        val_metric = evaluate_scores(scores_list, labels_list, args.metric)
        if val_metric is None:
            continue
        if args.metric.upper() == "AUROC":
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_params = params
        else:
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                best_params = params

    if best_params is None:
        print("No valid validation run. Check val-list and mask-dir.")
        return

    print(f"\nBest validation {args.metric}: {best_val_metric:.4f} with params: {best_params}")

    # Evaluate once on test with best params
    print(f"\nEvaluating on test set ({len(test_paths)} images) with best params...")
    scores_list, labels_list = run_method_on_list(
        model, galleries, test_paths, args.mask_dir, transform,
        method, args.max_size, **best_params
    )
    if not scores_list:
        print("No valid test results.")
        return
    test_auroc = evaluate_scores(scores_list, labels_list, "AUROC")
    test_fpr95 = evaluate_scores(scores_list, labels_list, "FPR95")
    print(f"\n--- Test results (final, no tuning) ---")
    print(f"  AUROC:     {test_auroc:.4f}")
    print(f"  FPR@95%:   {test_fpr95:.4f}")
    print(f"  Best params: {best_params}")


if __name__ == "__main__":
    main()
