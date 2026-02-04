"""
Build Mahalanobis Gallery with Class-Balanced Subsampling.

Key Settings (IMPROVED):
1. Use 2,000 pixels per class (increased from 1,000) for better covariance estimation
2. Use bottleneck features (304 dim) from decoder, not Layer 4 (2048 dim)
3. Use pseudo-inverse for covariance stability
4. Sample from ~500 images (increased from 100) for more diversity
5. Use 100 PCA components (reduced from 128) for better condition number

The Math: An image is 1024×2048 ≈ 2 million pixels. Training set is 2,975 images.
Total data points ≈ 6 billion. 90% of those pixels are Road, Sky, or Building.
Only 0.01% are Traffic Signs. By subsampling (2,000 pixels per class), we force
the OOD detector to understand what a "Traffic Sign" feature looks like.

With 2,000 samples per class and 100 PCA components, we get a 20:1 ratio which
ensures the covariance matrix is well-conditioned.
"""

import torch
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import os, pickle
import network.modeling as modeling
from network.ood_wrapper import wrap_model_for_ood
from torchvision import transforms
from sklearn.decomposition import PCA

# --- 1. CONFIGURATION ---
CITYSCAPES_IMG_DIR = "/fastdata/groupL/datasets/cityscapes/leftImg8bit/train/"
CITYSCAPES_LBL_DIR = "/fastdata/groupL/datasets/cityscapes/gtFine/train/"
MODEL_PATH = "/visinf/projects_students/groupL/rkowlu/ood_benchmark/models/deeplabv3plus_r101/deeplab_r101.pth"

NUM_CLASSES = 19
SAMPLES_PER_CLASS = 2000  # IMPROVED: Increased from 1000 to 2000 for better covariance estimation
NUM_IMAGES = 500  # IMPROVED: Increased from 100 to 500 for more diversity
FEATURE_DIM = 304  # FIXED: Bottleneck features (48 + 256) from decoder
N_COMPONENTS = 100  # IMPROVED: Reduced from 128 to 100 for better condition number
OUTPUT_FILE = "mahalanobis_gallery_bottleneck.pkl"
MAX_SIZE_DEFAULT = 1024  # Resize image so max edge <= this (reduces GPU memory; use 512 when GPU busy)

# --- Label Mapping ---
labelid_to_trainid = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255,
    10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255,
    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: 255
}
id_to_trainid_map = np.array([labelid_to_trainid.get(i, 255) for i in range(-1, 34)]).astype(np.uint8)

def convert_labels(label_array):
    return id_to_trainid_map[label_array + 1]

# --- 2. LOAD MODEL ---
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--cpu", action="store_true", help="Force CPU (use when GPU is busy)")
_parser.add_argument("--max-size", type=int, default=MAX_SIZE_DEFAULT, help="Resize image so max edge <= this (default 1024; use 512 when GPU busy)")
_args = _parser.parse_args()
_cpu_flag = _args.cpu
_max_size = _args.max_size

print("Loading DeepLabV3+ model...")
base_model = modeling.__dict__["deeplabv3plus_resnet101"](num_classes=NUM_CLASSES, output_stride=16)
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
state = checkpoint.get("model_state") or checkpoint.get("state_dict")
base_model.load_state_dict(state, strict=False)
base_model.eval()

device = torch.device("cpu")
if not _cpu_flag and torch.cuda.is_available():
    try:
        base_model.to("cuda")
        device = torch.device("cuda")
        print("Using GPU.")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("GPU OOM; using CPU (slower but works when GPU is busy).")
else:
    print("Using CPU." if _cpu_flag else "Using CPU (no CUDA).")
base_model = base_model.to(device)

# Wrap model for OOD detection (extracts bottleneck features)
model = wrap_model_for_ood(base_model)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 3. BUILD GALLERY WITH CLASS-BALANCED SUBSAMPLING ---
print(f"Building feature gallery from bottleneck ({FEATURE_DIM} dim)...")
print(f"Target: {SAMPLES_PER_CLASS} pixels per class from {NUM_IMAGES} images")
feature_gallery = {i: [] for i in range(NUM_CLASSES)}
image_files = sorted(glob(os.path.join(CITYSCAPES_IMG_DIR, "*/*.png")))

# Sample images randomly
np.random.seed(42)  # For reproducibility
image_files_sample = np.random.choice(image_files, min(NUM_IMAGES, len(image_files)), replace=False)

print(f"Processing {len(image_files_sample)} images (--max-size {_max_size})...")
for img_path in tqdm(image_files_sample):
    lbl_path = img_path.replace(CITYSCAPES_IMG_DIR, CITYSCAPES_LBL_DIR).replace('_leftImg8bit.png', '_gtFine_labelIds.png')

    if not os.path.exists(lbl_path):
        continue

    image_pil = Image.open(img_path).convert('RGB')
    label_img = Image.open(lbl_path)

    # Resize to reduce GPU memory (Cityscapes can be 2048x1024)
    w, h = image_pil.size
    if max(w, h) > _max_size:
        ratio = _max_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        image_pil = image_pil.resize((new_w, new_h), Image.BILINEAR)
        label_img = label_img.resize((new_w, new_h), Image.NEAREST)

    inputs = transform(image_pil).unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            bottleneck_features = model.forward_features(inputs)  # [1, 304, H, W]
            feature_size = (bottleneck_features.shape[3], bottleneck_features.shape[2])
            label_resized = label_img.resize(feature_size, Image.NEAREST)
            label_array_ids = np.array(label_resized)
            label_array_trainids = convert_labels(label_array_ids)
            features_flat = bottleneck_features.squeeze(0).permute(1, 2, 0).reshape(-1, FEATURE_DIM).cpu()
            label_flat = label_array_trainids.flatten()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        # Retry once at half resolution
        try:
            half_size = max(256, _max_size // 2)
            image_pil = Image.open(img_path).convert('RGB')
            label_img = Image.open(lbl_path)
            w, h = image_pil.size
            if max(w, h) > half_size:
                ratio = half_size / max(w, h)
                image_pil = image_pil.resize((int(w * ratio), int(h * ratio)), Image.BILINEAR)
                label_img = label_img.resize((int(w * ratio), int(h * ratio)), Image.NEAREST)
            inputs = transform(image_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                bottleneck_features = model.forward_features(inputs)
                feature_size = (bottleneck_features.shape[3], bottleneck_features.shape[2])
                label_resized = label_img.resize(feature_size, Image.NEAREST)
                label_array_trainids = convert_labels(np.array(label_resized))
                features_flat = bottleneck_features.squeeze(0).permute(1, 2, 0).reshape(-1, FEATURE_DIM).cpu()
                label_flat = label_array_trainids.flatten()
        except (torch.cuda.OutOfMemoryError, Exception):
            torch.cuda.empty_cache()
            continue
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Class-balanced subsampling (on CPU tensors)
    for c in range(NUM_CLASSES):
        current_count = sum(len(batch) for batch in feature_gallery[c])
        if current_count >= SAMPLES_PER_CLASS:
            continue
        class_mask = label_flat == c
        class_features = features_flat[class_mask]
        if len(class_features) > 0:
            needed = SAMPLES_PER_CLASS - current_count
            if needed > 0:
                n_available = len(class_features)
                n_sample = min(needed, n_available)
                indices = torch.randperm(n_available)[:n_sample]
                samples_to_add = class_features[indices]
                feature_gallery[c].append(samples_to_add)

# --- 4. CONCATENATE AND VERIFY ---
print("\nGallery build complete. Verifying class balance...")
all_features_list = []
for c in range(NUM_CLASSES):
    if len(feature_gallery[c]) > 0:
        feature_gallery[c] = torch.cat(feature_gallery[c], dim=0).numpy().astype(np.float32)
        n_samples = len(feature_gallery[c])
        print(f"  Class {c:2d}: {n_samples:5d} samples")
        all_features_list.append(feature_gallery[c])
    else:
        print(f"  Warning: No features found for class {c}. Using zeros.")
        feature_gallery[c] = np.zeros((1, FEATURE_DIM), dtype=np.float32)

# --- 5. CALCULATE STATISTICS ---
print(f"\nFitting PCA to {N_COMPONENTS} components...")
all_features = np.concatenate(all_features_list, axis=0)
print(f"Total features: {len(all_features)}")

pca = PCA(n_components=N_COMPONENTS)
pca.fit(all_features)

class_means = []
print("Calculating class means and shared covariance...")

all_features_pca_list = []
for c in range(NUM_CLASSES):
    features_class_pca = pca.transform(feature_gallery[c])
    class_mean = np.mean(features_class_pca, axis=0).astype(np.float32)
    class_means.append(class_mean)
    all_features_pca_list.append(features_class_pca)

all_features_pca = np.concatenate(all_features_pca_list, axis=0)

# Calculate shared covariance matrix
shared_cov = np.cov(all_features_pca, rowvar=False).astype(np.float32)

print(f"Covariance matrix shape: {shared_cov.shape}")
print(f"Covariance matrix condition number: {np.linalg.cond(shared_cov):.2e}")

# --- 6. SAVE GALLERY ---
mahalanobis_data = {
    'pca': pca,
    'means': class_means, 
    'cov': shared_cov,
    'feature_dim': FEATURE_DIM,
    'n_components': N_COMPONENTS,
    'samples_per_class': SAMPLES_PER_CLASS,
}

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(mahalanobis_data, f)

print(f"\n--- ALL DONE! ---")
print(f"Mahalanobis gallery saved to {OUTPUT_FILE}")
print(f"  Feature dimension: {FEATURE_DIM}")
print(f"  PCA components: {N_COMPONENTS}")
print(f"  Samples per class: {SAMPLES_PER_CLASS}")
print(f"  Total samples: {len(all_features)}")
