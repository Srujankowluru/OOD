import torch
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import os, pickle
import network.modeling as modeling
from torchvision import transforms
from sklearn.decomposition import PCA

# --- 1. CONFIGURATION ---
CITYSCAPES_IMG_DIR = "/fastdata/groupL/datasets/cityscapes/leftImg8bit/train/"
CITYSCAPES_LBL_DIR = "/fastdata/groupL/datasets/cityscapes/gtFine/train/"
MODEL_PATH = "/visinf/projects_students/groupL/rkowlu/ood_benchmark/models/deeplabv3plus_r101/deeplab_r101.pth"

NUM_CLASSES = 19
SAMPLES_PER_CLASS = 10000
# --- CHANGED: Layer 4 has 2048 channels ---
FEATURE_DIM = 2048 
N_COMPONENTS = 128
# --- CHANGED: New filename ---
OUTPUT_FILE = "mahalanobis_gallery_layer4.pkl"

# --- Label Mapping (Same as before) ---
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
_cpu_flag = _parser.parse_args().cpu

print("Loading DeepLabV3+ model...")
model = modeling.__dict__["deeplabv3plus_resnet101"](num_classes=NUM_CLASSES, output_stride=16)
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
state = checkpoint.get("model_state") or checkpoint.get("state_dict")
model.load_state_dict(state, strict=False)
model.eval()

device = torch.device("cpu")
if not _cpu_flag and torch.cuda.is_available():
    try:
        model.to("cuda")
        device = torch.device("cuda")
        print("Using GPU.")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("GPU OOM; using CPU (slower but works when GPU is busy).")
else:
    print("Using CPU." if _cpu_flag else "Using CPU (no CUDA).")
model = model.to(device)

feature_maps = []
def get_features(module, input, output):
    feature_maps.append(output.detach())

# --- CHANGED: Hook Layer 4 instead of ASPP ---
model.backbone.layer4.register_forward_hook(get_features)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 3. BUILD GALLERY ---
print(f"Building feature gallery from Layer 4 ({FEATURE_DIM} dim)...")
feature_gallery = {i: [] for i in range(NUM_CLASSES)}
image_files = sorted(glob(os.path.join(CITYSCAPES_IMG_DIR, "*/*.png")))
image_files_sample = np.random.choice(image_files, 1000, replace=False)

for img_path in tqdm(image_files_sample):
    lbl_path = img_path.replace(CITYSCAPES_IMG_DIR, CITYSCAPES_LBL_DIR).replace('_leftImg8bit.png', '_gtFine_labelIds.png')

    if not os.path.exists(lbl_path):
        continue

    image_pil = Image.open(img_path).convert('RGB')
    label_img = Image.open(lbl_path)

    inputs = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        feature_maps.clear()
        _ = model(inputs)
        features = feature_maps[0] 

        feature_size = (features.shape[3], features.shape[2])
        label_resized = label_img.resize(feature_size, Image.NEAREST)
        label_array_ids = np.array(label_resized)
        label_array_trainids = convert_labels(label_array_ids)

        features_flat = features.squeeze(0).permute(1, 2, 0).reshape(-1, FEATURE_DIM).cpu()
        label_flat = label_array_trainids.flatten() 

        for c in range(NUM_CLASSES):
            # Calculate current total sample count (not batch count!)
            current_count = sum(len(batch) for batch in feature_gallery[c])
            if current_count < SAMPLES_PER_CLASS:
                class_features = features_flat[label_flat == c]
                if len(class_features) > 0:
                    needed = SAMPLES_PER_CLASS - current_count
                    n_sample = min(needed, len(class_features))
                    indices = torch.randperm(len(class_features))[:n_sample]
                    samples_to_add = class_features[indices]
                    feature_gallery[c].append(samples_to_add)
        
        # Clear GPU cache periodically to prevent memory leaks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# --- 4. CALCULATE STATS ---
print("\nGallery build complete. Calculating statistics...")

all_features_list = []
for c in range(NUM_CLASSES):
    if len(feature_gallery[c]) > 0:
        feature_gallery[c] = torch.cat(feature_gallery[c], dim=0).numpy().astype(np.float32)
        all_features_list.append(feature_gallery[c])
    else:
        print(f"Warning: No features found for class {c}. Using zeros.")
        feature_gallery[c] = np.zeros((1, FEATURE_DIM), dtype=np.float32)

all_features = np.concatenate(all_features_list, axis=0)

print(f"Fitting PCA to {N_COMPONENTS} components...")
pca = PCA(n_components=N_COMPONENTS)
pca.fit(all_features)

class_means = []
print("Calculating class means and shared covariance...")

all_features_pca_list = []
for c in range(NUM_CLASSES):
    features_class_pca = pca.transform(feature_gallery[c])
    class_means.append(np.mean(features_class_pca, axis=0).astype(np.float32))
    all_features_pca_list.append(features_class_pca)

all_features_pca = np.concatenate(all_features_pca_list, axis=0)
shared_cov = np.cov(all_features_pca, rowvar=False).astype(np.float32)

# --- 5. SAVE GALLERY ---
mahalanobis_data = {
    'pca': pca,
    'means': class_means, 
    'cov': shared_cov 
}

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(mahalanobis_data, f)

print(f"\n--- ALL DONE! ---")
print(f"Mahalanobis gallery saved to {OUTPUT_FILE}")
