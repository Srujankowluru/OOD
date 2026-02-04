"""
Build Mahalanobis gallery for Layer3 features (for Mahalanobis++).

Improvements (OpenOOD / Lee et al. aligned):
  - Confidence-based pixel filtering (conf > 0.9) — reduces boundary/mislabel noise
  - Spatial smoothing (3×3 avg pooling) — reduces pixel noise
  - Lower PCA dimension (64) — less noisy covariance for pixel-level
  - Class-balanced covariance — up to 500 samples per class for cov estimate

Layer3: ResNet backbone layer3, 1024-dim. Output: mahalanobis_gallery_layer3.pkl.
Run: python build_feature_gallery_layer3.py
Then: run_all_methods.py with Mahalanobis++ (layer3) or run_validation_tuning.py --method Mahalanobis++
"""

import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from torchvision import transforms

import network.modeling as modeling
from network.ood_wrapper import wrap_model_for_ood

# --- CONFIG ---
CITYSCAPES_IMG_DIR = "/fastdata/groupL/datasets/cityscapes/leftImg8bit/train/"
CITYSCAPES_LBL_DIR = "/fastdata/groupL/datasets/cityscapes/gtFine/train/"
MODEL_PATH = "/visinf/projects_students/groupL/rkowlu/ood_benchmark/models/deeplabv3plus_r101/deeplab_r101.pth"
NUM_CLASSES = 19
FEATURE_DIM = 1024  # ResNet-101 layer3

# Gallery collection
SAMPLES_PER_CLASS = 2000
NUM_IMAGES = 500

# Improvements (Vamsi / OpenOOD)
CONF_THRESH = 0.9       # Keep only pixels with prediction confidence > this
POOL_KERNEL = 3         # 3×3 avg pooling before flatten (reduces noise)
N_COMPONENTS = 64       # Lower PCA for pixel-level (was 128)
BALANCED_COV_SAMPLES = 500  # Per-class samples for shared covariance (avoids road dominance)

OUTPUT_FILE = "mahalanobis_gallery_layer3.pkl"

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--max-size", type=int, default=1024, help="Resize image max edge (default 1024)")
    args = parser.parse_args()

    print("Loading model...")
    base_model = modeling.__dict__["deeplabv3plus_resnet101"](num_classes=NUM_CLASSES, output_stride=16)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state") or checkpoint.get("state_dict")
    base_model.load_state_dict(state, strict=False)
    base_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    try:
        base_model.to(device)
    except torch.cuda.OutOfMemoryError:
        device = torch.device("cpu")
        base_model.to(device)
        print("GPU OOM; using CPU.")
    model = wrap_model_for_ood(base_model)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Collect features per class (confidence-filtered)
    feature_gallery = {c: [] for c in range(NUM_CLASSES)}
    image_files = sorted(glob(os.path.join(CITYSCAPES_IMG_DIR, "*/*.png")))
    np.random.seed(42)
    image_files_sample = np.random.choice(image_files, min(NUM_IMAGES, len(image_files)), replace=False)

    print(f"Building layer3 gallery: conf>{CONF_THRESH}, pool={POOL_KERNEL}, PCA={N_COMPONENTS}, balanced_cov={BALANCED_COV_SAMPLES}/class")

    current_max_size = args.max_size

    def process_one_image(img_path, max_side):
        lbl_path = img_path.replace(CITYSCAPES_IMG_DIR, CITYSCAPES_LBL_DIR).replace("_leftImg8bit.png", "_gtFine_labelIds.png")
        if not os.path.exists(lbl_path):
            return
        try:
            image_pil = Image.open(img_path).convert("RGB")
            w, h = image_pil.size
            if max(w, h) > max_side:
                ratio = max_side / max(w, h)
                image_pil = image_pil.resize((int(w * ratio), int(h * ratio)), Image.BILINEAR)
            label_img = Image.open(lbl_path)
        except Exception:
            return
        inputs = transform(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, layer3_features = model(inputs, return_layer="layer3")
        layer3_features = F.avg_pool2d(layer3_features, kernel_size=POOL_KERNEL, stride=1, padding=POOL_KERNEL // 2)
        B, Cf, Hf, Wf = layer3_features.shape
        feature_size = (Wf, Hf)
        label_resized = label_img.resize(feature_size, Image.NEAREST)
        label_array_trainids = convert_labels(np.array(label_resized))
        logits_small = F.interpolate(logits, size=(Hf, Wf), mode="bilinear", align_corners=False)
        probs = torch.softmax(logits_small, dim=1)
        conf_map = probs.max(dim=1).values
        features_flat = layer3_features.squeeze(0).permute(1, 2, 0).reshape(-1, Cf).cpu().numpy()
        label_flat = label_array_trainids.flatten()
        conf_flat = conf_map.squeeze(0).flatten().cpu().numpy()
        for c in range(NUM_CLASSES):
            current_count = sum(len(batch) for batch in feature_gallery[c])
            if current_count >= SAMPLES_PER_CLASS:
                continue
            # Prefer high-confidence pixels; fall back to lower conf if none pass (conf>0.9 can be empty)
            class_mask = (label_flat == c) & (conf_flat > CONF_THRESH)
            class_features = features_flat[class_mask]
            if len(class_features) == 0:
                class_mask = (label_flat == c) & (conf_flat > 0.5)
                class_features = features_flat[class_mask]
            if len(class_features) == 0:
                class_mask = (label_flat == c)
                class_features = features_flat[class_mask]
            if len(class_features) == 0:
                continue
            needed = SAMPLES_PER_CLASS - current_count
            n_sample = min(needed, len(class_features))
            indices = np.random.choice(len(class_features), n_sample, replace=False)
            feature_gallery[c].append(class_features[indices].astype(np.float32))

    for img_path in tqdm(image_files_sample, desc="Layer3 + conf filter"):
        try:
            process_one_image(img_path, current_max_size)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if device.type == "cuda":
                print(" GPU OOM; switching to CPU for rest of run (slower but no memory limit).")
                device = torch.device("cpu")
                base_model.to(device)
                model = wrap_model_for_ood(base_model)
                current_max_size = min(args.max_size, 512)
                process_one_image(img_path, current_max_size)
            else:
                raise
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Concatenate per class
    all_features_list = []
    for c in range(NUM_CLASSES):
        if len(feature_gallery[c]) > 0:
            f = np.concatenate(feature_gallery[c], axis=0)
            f = f[:SAMPLES_PER_CLASS]
            feature_gallery[c] = f
            all_features_list.append(f)
        else:
            # Fallback: use any pixels for this class if none passed confidence (rare)
            feature_gallery[c] = np.zeros((1, FEATURE_DIM), dtype=np.float32)

    if len(all_features_list) == 0:
        raise RuntimeError(
            "No features collected. Check: (1) label paths and convert_labels; "
            "(2) CONF_THRESH may be too high—script now falls back to conf>0.5 then any pixel per class."
        )
    all_features = np.concatenate(all_features_list, axis=0)

    # PCA
    print("Fitting PCA (n_components={})...".format(N_COMPONENTS))
    pca = PCA(n_components=N_COMPONENTS, svd_solver="randomized", whiten=False)
    pca.fit(all_features)

    # Class means and PCA-transformed features per class
    class_means = []
    all_features_pca_list = []
    for c in range(NUM_CLASSES):
        f = feature_gallery[c]
        if f.shape[0] < 2:
            f_pca = pca.transform(f)
            class_means.append(f_pca.mean(axis=0).astype(np.float32))
            all_features_pca_list.append(f_pca)
            continue
        f_pca = pca.transform(f)
        class_means.append(f_pca.mean(axis=0).astype(np.float32))
        all_features_pca_list.append(f_pca)

    # Class-balanced covariance (avoid road/background dominance)
    balanced = []
    for c in range(NUM_CLASSES):
        f = all_features_pca_list[c]
        if len(f) > BALANCED_COV_SAMPLES:
            idx = np.random.choice(len(f), BALANCED_COV_SAMPLES, replace=False)
            balanced.append(f[idx])
        elif len(f) > 0:
            balanced.append(f)
    cov_feats = np.concatenate(balanced, axis=0)
    shared_cov = np.cov(cov_feats, rowvar=False).astype(np.float32)

    mahalanobis_data = {
        "pca": pca,
        "means": class_means,
        "cov": shared_cov,
        "feature_dim": FEATURE_DIM,
        "n_components": N_COMPONENTS,
        "samples_per_class": SAMPLES_PER_CLASS,
    }
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(mahalanobis_data, f)

    print(f"Saved {OUTPUT_FILE} (layer3, conf>{CONF_THRESH}, PCA={N_COMPONENTS}, balanced_cov).")
    print("Run: run_all_methods.py with Mahalanobis++ (layer=layer3) or run_validation_tuning.py --method Mahalanobis++")


if __name__ == "__main__":
    main()
