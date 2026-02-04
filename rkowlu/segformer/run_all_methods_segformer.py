"""
Run the same OOD methods as DeepLabV3Plus-Pytorch but with SegFormer (HF).
Uses same data (image-list, mask-dir), same Cityscapes 19-class mapping, same output format.
Galleries must be built for SegFormer (different feature dims) and placed in --gallery-dir.

Usage (from this folder or from DeepLabV3Plus-Pytorch):
  python run_all_methods_segformer.py --image-list ../DeepLabV3Plus-Pytorch/val_list.txt \
    --mask-dir /fastdata/groupL/datasets/mapillary/v1.2/validation/ood_masks/ \
    --gallery-dir ./galleries --output-dir ./results_segformer --plot --save-vis 20

Requires: transformers, torch; and sys.path to DeepLabV3Plus-Pytorch for ood_methods.
"""

import os
import sys

# Add DeepLabV3Plus-Pytorch so we reuse ood_methods and evaluation logic
SEGFORMER_ROOT = os.path.dirname(os.path.abspath(__file__))
DEEPLAB_ROOT = os.path.join(os.path.dirname(SEGFORMER_ROOT), "DeepLabV3Plus-Pytorch")
if os.path.isdir(DEEPLAB_ROOT) and DEEPLAB_ROOT not in sys.path:
    sys.path.insert(0, DEEPLAB_ROOT)

import run_all_methods as _run_all_methods
from run_all_methods import (
    load_galleries,
    load_actsub_lambda,
    process_image,
    evaluate_method,
    save_ood_vis,
    _method_folder_name,
    DEFAULT_MASK_DIR,
)
DEVICE = _run_all_methods.DEVICE
from ood_methods import get_ood_score, AVAILABLE_METHODS

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse
import re

# SegFormer wrapper (this folder)
from segformer_wrapper import wrap_segformer_for_ood

NUM_CLASSES = 19
DEFAULT_IMAGE_LIST = os.path.join(DEEPLAB_ROOT, "val_list.txt") if os.path.isdir(DEEPLAB_ROOT) else "val_list.txt"
DEFAULT_OUTPUT_DIR = "./results_segformer"
SEGFORMER_MODEL_NAME = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"


def load_model(model_name=None, device=None, use_fp16=False):
    """Load SegFormer and wrap for OOD. Prefer GPU; use_fp16 reduces VRAM. On CUDA OOM, falls back to CPU."""
    device = device or DEVICE
    name = model_name or SEGFORMER_MODEL_NAME
    print(f"Loading SegFormer: {name}...")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = wrap_segformer_for_ood(model_name=name, device=None)
    try:
        model.to(device)
        if device.type == "cuda" and use_fp16:
            model = model.half()
            print("  Using FP16 on GPU (lower VRAM).")
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            print("  ⚠ CUDA OOM; using CPU (slower). Free GPU memory or use --fp16 --max-size 512 next time.")
            model.to(device)
        else:
            raise
    model.eval()
    return model, device


def main():
    parser = argparse.ArgumentParser(description="Run all OOD methods with SegFormer (same data & mapping as DeepLab)")
    parser.add_argument("--model", type=str, default=SEGFORMER_MODEL_NAME, help="HuggingFace SegFormer model name")
    parser.add_argument("--image-list", type=str, default=DEFAULT_IMAGE_LIST, help="Path to image list")
    parser.add_argument("--mask-dir", type=str, default=DEFAULT_MASK_DIR, help="Directory containing OOD masks")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated methods (default: all)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--gallery-dir", type=str, default="./galleries", help="SegFormer gallery files (.pkl)")
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-vis", type=int, default=0, metavar="N")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mahalanobis-layer", type=str, default="layer4", choices=["layer3", "layer4"])
    parser.add_argument("--cpu", action="store_true", help="Force CPU (use when GPU OOM)")
    parser.add_argument("--fp16", action="store_true", help="Use half precision on GPU (less VRAM, avoids OOM)")
    parser.add_argument("--only-missing", action="store_true", help="Run only methods not in existing results_summary.json (merge with existing)")
    parser.add_argument("--low-mem", action="store_true", help="Keep bottleneck at 1/4 res (saves ~1.7 GiB per image; avoids ReAct/ASH/ACTSUB/SCALE OOM)")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    if args.cpu:
        print("Using CPU (--cpu).")
    else:
        print("Using GPU (CUDA).")
    _run_all_methods.DEVICE = device  # so process_image and others use this device

    os.makedirs(args.output_dir, exist_ok=True)

    if args.methods:
        methods_to_run = [m.strip() for m in args.methods.split(",")]
    else:
        methods_to_run = [m for m in AVAILABLE_METHODS if m not in ["MC_Dropout", "GEM", "RankFeat"]]

    # --only-missing: run only methods not already in results_summary.json
    import json as _json_mod
    existing_results_dict = {}
    existing_method_thresholds = {}
    if args.only_missing:
        results_json = os.path.join(args.output_dir, "results_summary.json")
        if os.path.isfile(results_json):
            with open(results_json, "r") as f:
                existing_results_dict = _json_mod.load(f)
            missing = [m for m in methods_to_run if m not in existing_results_dict]
            if not missing:
                print("No missing methods (all already in results). Exiting.")
                return
            methods_to_run = missing
            print(f"Only-missing mode: running {len(methods_to_run)} methods not in existing results: {methods_to_run}")
            thresh_file = os.path.join(args.output_dir, "method_thresholds.json")
            if os.path.isfile(thresh_file):
                with open(thresh_file, "r") as f:
                    existing_method_thresholds = _json_mod.load(f)
        else:
            print("--only-missing: no existing results_summary.json; running all requested methods.")

    print("\nLoading galleries...")
    galleries = load_galleries(args.gallery_dir)
    actsub_lambda = load_actsub_lambda(args.gallery_dir)
    if "knn" in galleries and "kNN" not in methods_to_run:
        methods_to_run.append("kNN")
    if "vim" in galleries and "VIM" not in methods_to_run:
        methods_to_run.append("VIM")

    print(f"Methods to run: {methods_to_run}")
    print(f"Total: {len(methods_to_run)} methods")

    # ReAct/ASH/ACTSUB need float32: SegFormer encoder (e.g. LayerNorm) yields float32 hidden_states
    # while model.half() keeps Linear weights in half -> "Input type (float) and bias type (c10::Half)".
    use_fp16 = args.fp16
    if use_fp16 and any(m in ("ReAct", "ASH", "ACTSUB") for m in methods_to_run):
        use_fp16 = False
        print("  FP16 disabled (ReAct/ASH/ACTSUB require float32 with SegFormer).")
    model, device = load_model(args.model, device=device, use_fp16=use_fp16)
    _run_all_methods.DEVICE = device  # in case load_model fell back to CPU on OOM
    if args.low_mem:
        model._no_upsample_bottleneck = True
        print("  Low-mem mode: bottleneck at 1/4 res (less VRAM).")

    print("\n--- Feature Dimension Check ---")
    with torch.no_grad():
        dummy = torch.randn(1, 3, 256, 512).to(device)
        logits_d, features_d = model(dummy, return_features=True)
        print(f"Model outputs features with shape: {features_d.shape}")
        if "mahalanobis" in galleries:
            exp_dim = galleries["mahalanobis"]["pca"].n_features_in_
            if features_d.shape[1] != exp_dim:
                print(f"  ⚠ Model: {features_d.shape[1]}, Gallery: {exp_dim}")
            else:
                print(f"  ✓ Mahalanobis dimensions match: {features_d.shape[1]}")
        del dummy, logits_d, features_d
    torch.cuda.empty_cache()

    with open(args.image_list, "r") as f:
        image_paths = [line.strip() for line in f.readlines()]
    args.end = args.end or len(image_paths)
    image_paths = image_paths[args.start : args.end]
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    print(f"\nProcessing {len(image_paths)} images...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Target resolution = first image resized to max_size (no model forward to avoid OOM)
    print("\nDetermining target resolution (logit size)...")
    first_img = Image.open(image_paths[0]).convert("RGB")
    w, h = first_img.size
    if max(w, h) > args.max_size:
        ratio = args.max_size / max(w, h)
        w, h = int(w * ratio), int(h * ratio)
    target_size = (h, w)  # (H, W) for logits
    print(f"  Target resolution for all methods: {target_size} (H, W)")
    del first_img
    torch.cuda.empty_cache()

    all_results = {}
    method_thresholds = {}

    for method in methods_to_run:
        print(f"\n{'='*60}\nProcessing method: {method}\n{'='*60}")
        scores_list = []
        labels_list = []
        n_vis_saved_no_mask = 0
        feature_shape_logged = False
        method_kwargs_extra = {}
        if method.lower() == "actsub":
            method_kwargs_extra["lambda_"] = actsub_lambda
        if method.lower() in ("mahalanobis++", "mahalanobis_plus_plus"):
            method_kwargs_extra["layer"] = args.mahalanobis_layer

        for img_path in tqdm(image_paths, desc=method):
            scores, feat_shape = None, None
            try:
                scores, feat_shape = process_image(
                    model, img_path, transform, method, galleries,
                    max_size=args.max_size, target_size=target_size, **method_kwargs_extra
                )
            except (RuntimeError, ValueError) as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    for retry_size in [max(256, args.max_size // 2), max(256, args.max_size // 4)]:
                        if retry_size >= 256:
                            try:
                                scores, feat_shape = process_image(
                                    model, img_path, transform, method, galleries,
                                    max_size=retry_size, target_size=target_size, **method_kwargs_extra
                                )
                                break
                            except (RuntimeError, ValueError):
                                torch.cuda.empty_cache()
                                continue
                    if scores is None:
                        print(f"  ⚠ OOM on {os.path.basename(img_path)}, skipping...")
                        continue
                else:
                    if args.debug:
                        import traceback
                        traceback.print_exc()
                    continue

            if scores is None:
                continue
            if args.debug and not feature_shape_logged and feat_shape is not None:
                print(f"\n  Feature shape: {feat_shape}, Score shape: {scores.shape}")
                feature_shape_logged = True
            torch.cuda.empty_cache()

            stem = os.path.basename(img_path).rsplit(".", 1)[0]
            mask_path = os.path.join(args.mask_dir, stem + ".png")

            if not os.path.exists(mask_path):
                if args.save_vis and n_vis_saved_no_mask < args.save_vis:
                    vis_dir = os.path.join(args.output_dir, "test_images", _method_folder_name(method))
                    os.makedirs(vis_dir, exist_ok=True)
                    save_ood_vis(scores, os.path.join(vis_dir, stem + "_ood.png"))
                    np.save(os.path.join(vis_dir, stem + "_scores.npy"), np.asarray(scores, dtype=np.float64))
                    n_vis_saved_no_mask += 1
                continue

            mask_img = Image.open(mask_path).convert("L")
            mask_resized = mask_img.resize((target_size[1], target_size[0]), Image.NEAREST)
            mask_gt = (np.array(mask_resized) > 127).astype(np.uint8)
            if scores.shape != mask_gt.shape:
                raise AssertionError(f"Score shape {scores.shape} != mask shape {mask_gt.shape}")
            if args.save_vis and len(scores_list) < args.save_vis:
                vis_dir = os.path.join(args.output_dir, "test_images", _method_folder_name(method))
                os.makedirs(vis_dir, exist_ok=True)
                save_ood_vis(scores, os.path.join(vis_dir, stem + "_ood.png"))
                np.save(os.path.join(vis_dir, stem + "_scores.npy"), np.asarray(scores, dtype=np.float64))
            scores_list.append(scores)
            labels_list.append(mask_gt)

        if len(scores_list) > 0:
            id_scores_list = []
            for s, lbl in zip(scores_list, labels_list):
                flat_s = np.asarray(s, dtype=np.float64).flatten()
                flat_l = np.asarray(lbl, dtype=np.uint8).flatten()
                id_mask = (flat_l == 0) & np.isfinite(flat_s)
                if id_mask.any():
                    id_scores_list.append(flat_s[id_mask])
            if id_scores_list:
                id_pixel_scores = np.concatenate(id_scores_list)
                method_thresholds[_method_folder_name(method)] = float(np.percentile(id_pixel_scores, 95))
            results = evaluate_method(scores_list, labels_list, method)
            if results:
                all_results[method] = results
                print(f"\n{method} Results:")
                print(f"  AUROC: {results['auroc']:.4f}")
                print(f"  AP:    {results['ap']:.4f}")
                print(f"  FPR@95%TPR: {results['fpr_at_95_tpr']:.4f}")
                if results.get("polarity_flipped", False):
                    print(f"  ⚠ Polarity was auto-corrected!")
                print(f"  Score range: {results.get('score_range', 'N/A')}")
                print(f"  OOD ratio: {results.get('ood_ratio', 0):.2%}")
        else:
            if n_vis_saved_no_mask > 0:
                print(f"  No masks: AUROC skipped for {method}; OOD maps saved for {n_vis_saved_no_mask} images in test_images/")
            else:
                print(f"  ⚠ No valid results for {method}")
        del scores_list, labels_list
        torch.cuda.empty_cache()

    # Save results (same format as DeepLab); merge with existing if --only-missing
    results_file = os.path.join(args.output_dir, "results_summary.json")
    results_dict = dict(existing_results_dict)  # start from existing when --only-missing
    for k, v in all_results.items():
        results_dict[k] = {
            "auroc": float(v["auroc"]),
            "ap": float(v["ap"]),
            "fpr_at_95_tpr": float(v["fpr_at_95_tpr"]),
            "polarity_flipped": v.get("polarity_flipped", False),
            "score_range": v.get("score_range", (0, 0)),
            "n_samples": v.get("n_samples", 0),
            "ood_ratio": v.get("ood_ratio", 0),
        }
    with open(results_file, "w") as f:
        _json_mod.dump(results_dict, f, indent=2)
    merged_thresholds = dict(existing_method_thresholds)
    merged_thresholds.update(method_thresholds)
    if merged_thresholds:
        with open(os.path.join(args.output_dir, "method_thresholds.json"), "w") as f:
            _json_mod.dump(merged_thresholds, f, indent=2)
    import csv
    csv_file = os.path.join(args.output_dir, "results_summary.csv")
    with open(csv_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "auroc", "fpr_at_95_tpr", "ap", "n_samples", "ood_ratio"])
        for m, v in results_dict.items():
            w.writerow([m, v["auroc"], v["fpr_at_95_tpr"], v["ap"], v["n_samples"], v["ood_ratio"]])
    print(f"Results saved to {results_file} and {csv_file}")

    if args.plot and results_dict:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            methods = list(results_dict.keys())
            aurocs = [results_dict[m]["auroc"] for m in methods]
            fpr95 = [results_dict[m]["fpr_at_95_tpr"] for m in methods]
            ax[0].barh(methods, aurocs)
            ax[0].set_xlabel("AUROC")
            ax[0].set_title("SegFormer OOD")
            ax[1].barh(methods, fpr95)
            ax[1].set_xlabel("FPR@95% TPR")
            ax[1].set_title("SegFormer OOD")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "metrics_auroc_fpr95.png"), dpi=150)
            plt.close()
            print(f"Plot saved to {args.output_dir}/metrics_auroc_fpr95.png")
        except Exception as e:
            print(f"  ⚠ Could not save plot: {e}")

    print("Done. Results in", args.output_dir)


if __name__ == "__main__":
    main()
