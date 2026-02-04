"""
Build ActSub insignificant-subspace gallery (full paper, Eq. 7).

Training activations a^(i) are projected into the insignificant subspace
(V_ins V_ins^T a), spatially averaged, L2-normalized, and stored.
At test time, S→ = -log(1 - mean cosine similarity to this gallery).

Paper: gallery_fraction = 0.1 (fixed 10%). N = 20 neighbors (fixed).
Output: actsub_gallery.pkl - numpy array [N, 256], float32, L2-normalized.
Pipeline: galleries['actsub'] = this array; get_ood_score uses gallery['actsub_ins'].
"""

import torch
import torch.nn.functional as F
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import os
import pickle
import argparse
import network.modeling as modeling
from network.ood_wrapper import wrap_model_for_ood
from torchvision import transforms
from ood_methods import (
    _get_actsub_components,
    project_insignificant,
    ACTSUB_GALLERY_FRACTION,
    select_k_via_norm_balance,
)

# --- Config ---
CITYSCAPES_IMG_DIR = "/fastdata/groupL/datasets/cityscapes/leftImg8bit/train/"
MODEL_PATH = "/visinf/projects_students/groupL/rkowlu/ood_benchmark/models/deeplabv3plus_r101/deeplab_r101.pth"
NUM_CLASSES = 19
OUTPUT_FILE = "actsub_gallery.pkl"
MAX_SIZE_DEFAULT = 1024
K_DECISIVE_DEFAULT = None  # None = use num_classes (19) for segmentation; or set int


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Force CPU (use when GPU is busy)")
    parser.add_argument("--max-size", type=int, default=MAX_SIZE_DEFAULT, help="Resize image max edge (default 1024)")
    parser.add_argument("--gallery-fraction", type=float, default=ACTSUB_GALLERY_FRACTION, help="Fraction of train set for gallery (paper: 0.1)")
    parser.add_argument("--k-auto", action="store_true", help="Select k via Eq. (6) norm balance on train (slower)")
    parser.add_argument("--k", type=int, default=None, help="Decisive components (default: num_classes=19)")
    args = parser.parse_args()

    print("Loading DeepLabV3+ model...")
    base_model = modeling.__dict__["deeplabv3plus_resnet101"](num_classes=NUM_CLASSES, output_stride=16)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state") or checkpoint.get("state_dict")
    base_model.load_state_dict(state, strict=False)
    base_model.eval()

    device = torch.device("cpu")
    if not args.cpu and torch.cuda.is_available():
        try:
            base_model.to("cuda")
            device = torch.device("cuda")
            print("Using GPU.")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("GPU OOM; using CPU.")
    else:
        print("Using CPU." if args.cpu else "Using CPU (no CUDA).")
    base_model = base_model.to(device)
    model = wrap_model_for_ood(base_model)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_files = sorted(glob(os.path.join(CITYSCAPES_IMG_DIR, "*/*.png")))
    n_total = len(image_files)
    n_sample = max(1, int(n_total * args.gallery_fraction))
    np.random.seed(42)
    image_files_sample = np.random.choice(image_files, min(n_sample, n_total), replace=False)
    k_use = args.k if args.k is not None else NUM_CLASSES
    if args.k_auto:
        print("Selecting k via Eq. (6) norm balance...")
        if hasattr(model, "_actsub_components"):
            del model._actsub_components
        k_use = select_k_via_norm_balance(model, list(image_files_sample[: min(100, len(image_files_sample))]), transform, device, max_k=NUM_CLASSES, batch_size=4)
        print(f"  k* = {k_use}")
        if hasattr(model, "_actsub_components"):
            del model._actsub_components
    comp = _get_actsub_components(model, k=k_use)
    P_ins = comp["P_ins"].to(device)
    first_part = comp["first_part"]

    all_feats = []
    print(f"Building ActSub insignificant gallery from {len(image_files_sample)} images (--max-size {args.max_size})...")
    for img_path in tqdm(image_files_sample):
        try:
            image_pil = Image.open(img_path).convert("RGB")
            w, h = image_pil.size
            if max(w, h) > args.max_size:
                ratio = args.max_size / max(w, h)
                image_pil = image_pil.resize((int(w * ratio), int(h * ratio)), Image.BILINEAR)
            inputs = transform(image_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                bottleneck = model.forward_features(inputs)
                feat_256 = first_part(bottleneck)
                feat_ins = project_insignificant(feat_256, P_ins)
                feat_pooled = feat_ins.mean(dim=(2, 3))
                feat_pooled = F.normalize(feat_pooled, p=2, dim=1)
                all_feats.append(feat_pooled.cpu().numpy().astype(np.float32))
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_feats) == 0:
        raise RuntimeError("No features collected; check paths and GPU memory.")

    gallery = np.concatenate(all_feats, axis=0)
    norms = np.linalg.norm(gallery, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    gallery = (gallery / norms).astype(np.float32)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(gallery, f)

    print(f"\nActSub gallery saved to {OUTPUT_FILE}")
    print(f"  shape: {gallery.shape} (insignificant subspace, L2-normalized)")


if __name__ == "__main__":
    main()
