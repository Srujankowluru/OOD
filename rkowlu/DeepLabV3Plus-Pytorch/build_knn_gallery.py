"""
Build kNN gallery from ID (Cityscapes train) bottleneck features.

kNN OOD score = distance to k-th nearest neighbor in normalized feature space.
This script extracts bottleneck features from training images, subsamples pixels,
and saves a normalized feature array [N, C] for run_all_methods / run_validation_tuning.

Output: knn_gallery.pkl - single numpy array of shape (N, 304), float32.
Pipeline expects galleries['knn'] = this array; get_ood_score uses gallery_knn.
"""

import torch
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import os
import pickle
import network.modeling as modeling
from network.ood_wrapper import wrap_model_for_ood
from torchvision import transforms

# --- Config ---
CITYSCAPES_IMG_DIR = "/fastdata/groupL/datasets/cityscapes/leftImg8bit/train/"
MODEL_PATH = "/visinf/projects_students/groupL/rkowlu/ood_benchmark/models/deeplabv3plus_r101/deeplab_r101.pth"
NUM_CLASSES = 19
NUM_IMAGES = 200
MAX_PIXELS_PER_IMAGE = 1500
FEATURE_DIM = 304
OUTPUT_FILE = "knn_gallery.pkl"


def main():
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--cpu", action="store_true", help="Force CPU (use when GPU is busy)")
    _cpu = _p.parse_args().cpu

    print("Loading DeepLabV3+ model...")
    base_model = modeling.__dict__["deeplabv3plus_resnet101"](num_classes=NUM_CLASSES, output_stride=16)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state") or checkpoint.get("state_dict")
    base_model.load_state_dict(state, strict=False)
    base_model.eval()

    device = torch.device("cpu")
    if not _cpu and torch.cuda.is_available():
        try:
            base_model.to("cuda")
            device = torch.device("cuda")
            print("Using GPU.")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("GPU OOM; using CPU (slower but works when GPU is busy).")
    else:
        print("Using CPU." if _cpu else "Using CPU (no CUDA).")
    base_model = base_model.to(device)
    model = wrap_model_for_ood(base_model)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_files = sorted(glob(os.path.join(CITYSCAPES_IMG_DIR, "*/*.png")))
    np.random.seed(42)
    image_files_sample = np.random.choice(image_files, min(NUM_IMAGES, len(image_files)), replace=False)

    all_features_list = []
    print(f"Collecting bottleneck features from {len(image_files_sample)} images (max {MAX_PIXELS_PER_IMAGE} px/image)...")
    for img_path in tqdm(image_files_sample):
        image_pil = Image.open(img_path).convert("RGB")
        inputs = transform(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            bottleneck = model.forward_features(inputs)
        feat = bottleneck.squeeze(0).permute(1, 2, 0).reshape(-1, FEATURE_DIM).cpu().numpy()
        n_keep = min(MAX_PIXELS_PER_IMAGE, feat.shape[0])
        idx = np.random.choice(feat.shape[0], n_keep, replace=False)
        all_features_list.append(feat[idx].astype(np.float32))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gallery = np.concatenate(all_features_list, axis=0)
    # L2-normalize (knn_score normalizes again; saving normalized saves memory at inference)
    norms = np.linalg.norm(gallery, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    gallery = (gallery / norms).astype(np.float32)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(gallery, f)

    print(f"\nkNN gallery saved to {OUTPUT_FILE}")
    print(f"  shape: {gallery.shape}")


if __name__ == "__main__":
    main()
