"""
Paper-faithful ActSub λ tuning: ID validation only (no OOD data).

Tunes the balancing scalar λ on an in-distribution validation split.
Fixed: k (derived or num_classes), N=20, gallery fraction 10%, activation shaping.
Only λ is selected via grid on ID val.

Usage:
  python tune_actsub_lambda.py --val-list val_list.txt --gallery-dir .
  python tune_actsub_lambda.py --val-list val_list.txt --max-val 100   # faster

Output: actsub_best_lambda.json in gallery_dir (used by run_all_methods if present).
"""

import os
import json
import pickle
import argparse
import torch
from PIL import Image
from torchvision import transforms
import network.modeling as modeling
from network.ood_wrapper import wrap_model_for_ood
from ood_methods import tune_lambda_actsub

MODEL_PATH = "/visinf/projects_students/groupL/rkowlu/ood_benchmark/models/deeplabv3plus_r101/deeplab_r101.pth"
NUM_CLASSES = 19
LAMBDA_CANDIDATES = (0.2, 0.5, 1.0)  # paper-faithful grid; do not tune on OOD


def main():
    parser = argparse.ArgumentParser(description="Tune ActSub λ on ID validation only")
    parser.add_argument("--val-list", type=str, default="val_list.txt", help="Path to ID validation image list")
    parser.add_argument("--gallery-dir", type=str, default=".", help="Directory containing actsub_gallery.pkl; best λ saved here")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Model checkpoint path")
    parser.add_argument("--max-val", type=int, default=None, help="Max validation images (for speed)")
    parser.add_argument("--max-size", type=int, default=1024, help="Resize image max edge (avoid OOM)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    base_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def transform(pil_img):
        w, h = pil_img.size
        if max(w, h) > args.max_size:
            ratio = args.max_size / max(w, h)
            pil_img = pil_img.resize((int(w * ratio), int(h * ratio)), Image.BILINEAR)
        return base_t(pil_img)

    # Load validation paths
    if not os.path.isfile(args.val_list):
        raise FileNotFoundError(f"Validation list not found: {args.val_list}")
    with open(args.val_list, "r") as f:
        val_paths = [line.strip() for line in f.readlines() if line.strip()]
    if args.max_val is not None:
        val_paths = val_paths[: args.max_val]
    print(f"Using {len(val_paths)} ID validation images from {args.val_list}")

    # Load ActSub gallery
    gallery_path = os.path.join(args.gallery_dir, "actsub_gallery.pkl")
    if not os.path.isfile(gallery_path):
        raise FileNotFoundError(f"ActSub gallery not found: {gallery_path}. Run build_actsub_gallery.py first.")
    with open(gallery_path, "rb") as f:
        gallery_ins = pickle.load(f)
    print(f"Loaded ActSub gallery: {gallery_ins.shape[0]} samples")

    # Load model
    print("Loading model...")
    base_model = modeling.__dict__["deeplabv3plus_resnet101"](num_classes=NUM_CLASSES, output_stride=16)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state") or checkpoint.get("state_dict")
    base_model.load_state_dict(state, strict=False)
    base_model.eval()
    base_model = base_model.to(device)
    model = wrap_model_for_ood(base_model)

    # Tune λ on ID val only (paper-faithful)
    best_lambda = tune_lambda_actsub(
        model,
        val_paths,
        gallery_ins,
        transform,
        device,
        lambda_candidates=LAMBDA_CANDIDATES,
    )
    print(f"Best λ (ID val): {best_lambda}")

    out_path = os.path.join(args.gallery_dir, "actsub_best_lambda.json")
    with open(out_path, "w") as f:
        json.dump({"lambda": best_lambda, "lambda_candidates": list(LAMBDA_CANDIDATES)}, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
