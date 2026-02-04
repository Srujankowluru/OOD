"""
Build Mahalanobis (bottleneck) gallery for SegFormer using Cityscapes train set.
Same Cityscapes label mapping as DeepLab (19 train classes).
Output: galleries/mahalanobis_gallery_bottleneck.pkl (and layer4 for Mahalanobis++).

Usage (from segformer folder):
  python build_feature_gallery_segformer.py
  python build_feature_gallery_segformer.py --cpu
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
from glob import glob
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from sklearn.decomposition import PCA

# Same Cityscapes paths and mapping as DeepLabV3Plus-Pytorch
CITYSCAPES_IMG_DIR = "/fastdata/groupL/datasets/cityscapes/leftImg8bit/train/"
CITYSCAPES_LBL_DIR = "/fastdata/groupL/datasets/cityscapes/gtFine/train/"
NUM_CLASSES = 19
# SegFormer: bottleneck = decoder input (3072) for Mahalanobis/ReAct/ASH/ACTSUB; layer4 = 512 for Mahalanobis++
FEATURE_DIM_BOTTLENECK = 3072
FEATURE_DIM_LAYER4 = 512
N_COMPONENTS = 100
SAMPLES_PER_CLASS = 5000
NUM_IMAGES = 500
OUTPUT_DIR = "galleries"
OUTPUT_BOTTLENECK = "mahalanobis_gallery_bottleneck.pkl"
OUTPUT_LAYER4 = "mahalanobis_gallery_layer4.pkl"

# Same label mapping as DeepLab (Cityscapes 19 train IDs)
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
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-size", type=int, default=1024)
    args = parser.parse_args()

    from segformer_wrapper import wrap_segformer_for_ood

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Loading SegFormer...")
    model = wrap_segformer_for_ood(device=device)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_files = sorted(glob(os.path.join(CITYSCAPES_IMG_DIR, "*/*.png")))
    np.random.seed(42)
    image_files = list(np.random.choice(image_files, min(NUM_IMAGES, len(image_files)), replace=False))

    feature_gallery = {c: [] for c in range(NUM_CLASSES)}

    print(f"Building bottleneck gallery (dim={FEATURE_DIM_BOTTLENECK}, PCA={N_COMPONENTS})...")
    for img_path in tqdm(image_files):
        lbl_path = img_path.replace(CITYSCAPES_IMG_DIR, CITYSCAPES_LBL_DIR).replace("_leftImg8bit.png", "_gtFine_labelIds.png")
        if not os.path.exists(lbl_path):
            continue
        image_pil = Image.open(img_path).convert("RGB")
        if max(image_pil.size) > args.max_size:
            ratio = args.max_size / max(image_pil.size)
            new_size = (int(image_pil.size[0] * ratio), int(image_pil.size[1] * ratio))
            image_pil = image_pil.resize(new_size, Image.BILINEAR)
        label_img = Image.open(lbl_path)
        if image_pil.size != label_img.size:
            label_img = label_img.resize(image_pil.size, Image.NEAREST)
        inputs = transform(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            # Keep bottleneck at 1/4 res to avoid OOM (no 6GB upsample to full res)
            _, features = model(inputs, return_features=True, upsample_bottleneck=False)
        B, C, H, W = features.shape
        features_flat = features.squeeze(0).permute(1, 2, 0).reshape(-1, C).cpu().numpy().astype(np.float32)
        # Labels at same resolution as features (H, W) = (H_img/4, W_img/4)
        label_array = np.array(label_img.resize((W, H), Image.NEAREST))
        label_train = convert_labels(label_array)
        label_flat = label_train.flatten()
        for c in range(NUM_CLASSES):
            mask = label_flat == c
            if mask.sum() > 0:
                feats_c = features_flat[mask]
                current_count = sum(arr.shape[0] for arr in feature_gallery[c])
                need = SAMPLES_PER_CLASS - current_count
                if need > 0:
                    n_take = min(need, len(feats_c))
                    idx = np.random.choice(len(feats_c), n_take, replace=False)
                    feature_gallery[c].append(feats_c[idx])
        torch.cuda.empty_cache()

    for c in range(NUM_CLASSES):
        if len(feature_gallery[c]) > 0:
            feature_gallery[c] = np.concatenate(feature_gallery[c], axis=0).astype(np.float32)
        else:
            feature_gallery[c] = np.zeros((1, FEATURE_DIM_BOTTLENECK), dtype=np.float32)

    all_features = np.concatenate([feature_gallery[c] for c in range(NUM_CLASSES) if feature_gallery[c].shape[0] > 1], axis=0)
    print(f"Fitting PCA (n_components={N_COMPONENTS}) on {all_features.shape[0]} samples...")
    pca = PCA(n_components=min(N_COMPONENTS, all_features.shape[0], all_features.shape[1]))
    pca.fit(all_features)

    class_means = []
    all_pca_list = []
    for c in range(NUM_CLASSES):
        X = pca.transform(feature_gallery[c])
        class_means.append(np.mean(X, axis=0).astype(np.float32))
        all_pca_list.append(X)
    all_pca = np.concatenate(all_pca_list, axis=0)
    shared_cov = np.cov(all_pca, rowvar=False).astype(np.float32)

    # Same keys as DeepLab pipeline: ood_methods.mahalanobis_score expects 'means' (list) and 'cov'
    gallery = {
        "means": class_means,  # list of 19 arrays (n_components,) for mahalanobis_score
        "cov": shared_cov,
        "pca": pca,
        "feature_gallery": feature_gallery,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_BOTTLENECK)
    with open(out_path, "wb") as f:
        pickle.dump(gallery, f)
    print(f"Saved {out_path}")

    # Build layer4 gallery (512-dim) for Mahalanobis++ separately
    print(f"Building layer4 gallery (dim={FEATURE_DIM_LAYER4}, PCA={N_COMPONENTS})...")
    feature_gallery_l4 = {c: [] for c in range(NUM_CLASSES)}
    for img_path in tqdm(image_files):
        lbl_path = img_path.replace(CITYSCAPES_IMG_DIR, CITYSCAPES_LBL_DIR).replace("_leftImg8bit.png", "_gtFine_labelIds.png")
        if not os.path.exists(lbl_path):
            continue
        image_pil = Image.open(img_path).convert("RGB")
        if max(image_pil.size) > args.max_size:
            ratio = args.max_size / max(image_pil.size)
            new_size = (int(image_pil.size[0] * ratio), int(image_pil.size[1] * ratio))
            image_pil = image_pil.resize(new_size, Image.BILINEAR)
        label_img = Image.open(lbl_path)
        if image_pil.size != label_img.size:
            label_img = label_img.resize(image_pil.size, Image.NEAREST)
        inputs = transform(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            _, features = model(inputs, return_layer="layer4")  # [B, 512, h, w]
        B, C, H, W = features.shape
        features_flat = features.squeeze(0).permute(1, 2, 0).reshape(-1, C).cpu().numpy().astype(np.float32)
        label_array = np.array(label_img.resize((W, H), Image.NEAREST))
        label_train = convert_labels(label_array)
        label_flat = label_train.flatten()
        for c in range(NUM_CLASSES):
            mask = label_flat == c
            if mask.sum() > 0:
                feats_c = features_flat[mask]
                current_count = sum(arr.shape[0] for arr in feature_gallery_l4[c])
                need = SAMPLES_PER_CLASS - current_count
                if need > 0:
                    n_take = min(need, len(feats_c))
                    idx = np.random.choice(len(feats_c), n_take, replace=False)
                    feature_gallery_l4[c].append(feats_c[idx])
        torch.cuda.empty_cache()

    for c in range(NUM_CLASSES):
        if len(feature_gallery_l4[c]) > 0:
            feature_gallery_l4[c] = np.concatenate(feature_gallery_l4[c], axis=0).astype(np.float32)
        else:
            feature_gallery_l4[c] = np.zeros((1, FEATURE_DIM_LAYER4), dtype=np.float32)

    all_l4 = np.concatenate([feature_gallery_l4[c] for c in range(NUM_CLASSES) if feature_gallery_l4[c].shape[0] > 1], axis=0)
    pca_l4 = PCA(n_components=min(N_COMPONENTS, all_l4.shape[0], all_l4.shape[1]))
    pca_l4.fit(all_l4)
    class_means_l4 = [np.mean(pca_l4.transform(feature_gallery_l4[c]), axis=0).astype(np.float32) for c in range(NUM_CLASSES)]
    all_pca_l4 = np.concatenate([pca_l4.transform(feature_gallery_l4[c]) for c in range(NUM_CLASSES)], axis=0)
    shared_cov_l4 = np.cov(all_pca_l4, rowvar=False).astype(np.float32)
    gallery_l4 = {"means": class_means_l4, "cov": shared_cov_l4, "pca": pca_l4, "feature_gallery": feature_gallery_l4}
    out_layer4 = os.path.join(OUTPUT_DIR, OUTPUT_LAYER4)
    with open(out_layer4, "wb") as f:
        pickle.dump(gallery_l4, f)
    print(f"Saved {out_layer4} (use for Mahalanobis++)")
    print("Done. Run: python run_all_methods_segformer.py --gallery-dir ./galleries --image-list ... --mask-dir ...")


if __name__ == "__main__":
    main()
