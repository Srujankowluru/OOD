"""
Build ViM (Virtual Logit Matching) gallery from ID training features.

ViM score = residual (reconstruction error in PCA space) - alpha * energy(logits).
This script collects bottleneck features from Cityscapes training images,
fits PCA, and saves components, mean, and default alpha for run_validation_tuning.

Output: vim_gallery.pkl with keys: components, mean, alpha
- components: (n_components, feature_dim) PCA components for projection/reconstruction
- mean: (feature_dim,) mean of ID features
- alpha: default 1.0 (tune via run_validation_tuning.py --method VIM)
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
from sklearn.decomposition import PCA

# --- Config (match bottleneck from ood_wrapper) ---
CITYSCAPES_IMG_DIR = "/fastdata/groupL/datasets/cityscapes/leftImg8bit/train/"
CITYSCAPES_LBL_DIR = "/fastdata/groupL/datasets/cityscapes/gtFine/train/"
MODEL_PATH = "/visinf/projects_students/groupL/rkowlu/ood_benchmark/models/deeplabv3plus_r101/deeplab_r101.pth"
NUM_CLASSES = 19
NUM_IMAGES = 300
MAX_PIXELS_PER_IMAGE = 2000  # Subsample per image to limit memory
FEATURE_DIM = 304
N_COMPONENTS = 64  # PCA dim; can tune or build multiple galleries
OUTPUT_FILE = "vim_gallery.pkl"
ALPHA_DEFAULT = 1.0

# Label mapping (same as Mahalanobis gallery)
labelid_to_trainid = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255,
    10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255,
    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: 255
}
id_to_trainid_map = np.array([labelid_to_trainid.get(i, 255) for i in range(-1, 34)]).astype(np.uint8)


def convert_labels(label_array):
    return id_to_trainid_map[label_array + 1]


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
        lbl_path = img_path.replace(CITYSCAPES_IMG_DIR, CITYSCAPES_LBL_DIR).replace(
            "_leftImg8bit.png", "_gtFine_labelIds.png"
        )
        if not os.path.exists(lbl_path):
            continue
        image_pil = Image.open(img_path).convert("RGB")
        inputs = transform(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            bottleneck = model.forward_features(inputs)
        # [1, C, H, W] -> [H*W, C]
        feat = bottleneck.squeeze(0).permute(1, 2, 0).reshape(-1, FEATURE_DIM).cpu().numpy()
        n_keep = min(MAX_PIXELS_PER_IMAGE, feat.shape[0])
        idx = np.random.choice(feat.shape[0], n_keep, replace=False)
        all_features_list.append(feat[idx].astype(np.float32))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_features = np.concatenate(all_features_list, axis=0)
    print(f"Total features: {all_features.shape[0]}, dim: {all_features.shape[1]}")

    mean = np.mean(all_features, axis=0).astype(np.float32)
    centered = all_features - mean
    pca = PCA(n_components=min(N_COMPONENTS, centered.shape[0], centered.shape[1]))
    pca.fit(centered)
    components = pca.components_.astype(np.float32)  # (n_components, feature_dim)

    vim_gallery = {
        "components": components,
        "mean": mean,
        "alpha": ALPHA_DEFAULT,
    }
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(vim_gallery, f)

    print(f"\nViM gallery saved to {OUTPUT_FILE}")
    print(f"  components: {components.shape}, mean: {mean.shape}, alpha: {ALPHA_DEFAULT}")


if __name__ == "__main__":
    main()
