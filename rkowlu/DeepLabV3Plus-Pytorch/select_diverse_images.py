"""
Select N images with diverse predicted classes (not dominated by road/sky).

Uses the segmentation model to get per-pixel predictions, then scores each image
by entropy of the class distribution. Higher entropy = more balanced classes.
Outputs a list of image paths (best N) for use with run_all_methods.py.

Usage:
  python select_diverse_images.py --image-list test_list.txt --output diverse_200_list.txt --n 200
  python run_all_methods.py --image-list diverse_200_list.txt --output-dir results_200 --plot
"""

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import argparse
from torchvision import transforms
try:
    from scipy.stats import entropy as scipy_entropy
except ImportError:
    scipy_entropy = None

from run_all_methods import load_model, DEFAULT_MODEL_PATH, DEFAULT_MASK_DIR

NUM_CLASSES = 19


def _entropy(probs):
    probs = np.asarray(probs, dtype=np.float64) + 1e-10
    probs = probs / probs.sum()
    if scipy_entropy is not None:
        return float(scipy_entropy(probs))
    return float(-np.sum(probs * np.log(probs)))


def diversity_score(pred_flat):
    """Score = entropy of class distribution + 0.01 * num_classes (prefer many classes)."""
    valid = (pred_flat >= 0) & (pred_flat < NUM_CLASSES)
    if valid.sum() == 0:
        return 0.0
    hist, _ = np.histogram(pred_flat[valid], bins=NUM_CLASSES, range=(0, NUM_CLASSES))
    hist = hist.astype(np.float64) + 1e-10
    probs = hist / hist.sum()
    ent = _entropy(probs)
    num_classes = (hist > 0.5).sum()
    return float(ent) + 0.01 * num_classes


def main():
    parser = argparse.ArgumentParser(description="Select diverse images by predicted class entropy")
    parser.add_argument("--image-list", type=str, default="test_list.txt", help="Input image list")
    parser.add_argument("--output", type=str, default="diverse_200_list.txt", help="Output list path")
    parser.add_argument("--n", type=int, default=200, help="Number of images to select")
    parser.add_argument("--mask-dir", type=str, default=DEFAULT_MASK_DIR, help="OOD mask dir (only images with masks)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Model checkpoint")
    parser.add_argument("--max-pool", type=int, default=None, help="Max images to score (default: all)")
    parser.add_argument("--max-size", type=int, default=1024, help="Max image dimension")
    args = parser.parse_args()

    with open(args.image_list, "r") as f:
        all_paths = [line.strip() for line in f.readlines()]

    # Keep only images that have OOD masks (needed for eval)
    paths_with_masks = []
    for p in all_paths:
        base = os.path.basename(p).replace(".jpg", ".png")
        if os.path.exists(os.path.join(args.mask_dir, base)):
            paths_with_masks.append(p)
    print(f"Images with masks: {len(paths_with_masks)} / {len(all_paths)}")

    if args.max_pool:
        paths_with_masks = paths_with_masks[: args.max_pool]

    model = load_model(args.model)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    device = next(model.parameters()).device

    scores_list = []
    for img_path in tqdm(paths_with_masks, desc="Diversity"):
        try:
            image = Image.open(img_path).convert("RGB")
            if max(image.size) > args.max_size:
                ratio = args.max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.BILINEAR)
            x = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(x, return_features=True)
            # [1, C, H, W] -> [H*W]
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().flatten()
            score = diversity_score(pred)
            scores_list.append((img_path, score))
        except Exception:
            continue
        torch.cuda.empty_cache()

    scores_list.sort(key=lambda x: x[1], reverse=True)
    selected = [p for p, _ in scores_list[: args.n]]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for p in selected:
            f.write(p + "\n")

    print(f"Wrote {len(selected)} diverse image paths to {args.output}")
    if scores_list:
        print(f"  Diversity score range: [{scores_list[-1][1]:.3f}, {scores_list[0][1]:.3f}]")


if __name__ == "__main__":
    main()
