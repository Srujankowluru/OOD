"""
Build VIM gallery for SegFormer using Cityscapes train bottleneck features (3072-dim).
Saves vim_gallery.pkl with keys: components, mean, alpha (same as DeepLab).
run_all_methods load_galleries expects galleries['vim'] = {'gallery_vim': this_dict}.

Usage (from segformer folder):
  python build_vim_gallery_segformer.py
  python build_vim_gallery_segformer.py --cpu
"""

import os
import pickle
import numpy as np
import torch
from glob import glob
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from sklearn.decomposition import PCA

CITYSCAPES_IMG_DIR = "/fastdata/groupL/datasets/cityscapes/leftImg8bit/train/"
FEATURE_DIM = 3072
NUM_IMAGES = 300
MAX_PIXELS_PER_IMAGE = 2000
N_COMPONENTS = 64
ALPHA_DEFAULT = 1.0
OUTPUT_DIR = "galleries"
OUTPUT_FILE = "vim_gallery.pkl"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--max-size", type=int, default=1024)
    args = p.parse_args()

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

    all_features_list = []
    print(f"Collecting bottleneck features from {len(image_files)} images (max {MAX_PIXELS_PER_IMAGE} px/image)...")
    for img_path in tqdm(image_files):
        image_pil = Image.open(img_path).convert("RGB")
        if max(image_pil.size) > args.max_size:
            ratio = args.max_size / max(image_pil.size)
            new_size = (int(image_pil.size[0] * ratio), int(image_pil.size[1] * ratio))
            image_pil = image_pil.resize(new_size, Image.BILINEAR)
        inputs = transform(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            _, features = model(inputs, return_features=True, upsample_bottleneck=False)
        feat = features.squeeze(0).permute(1, 2, 0).reshape(-1, FEATURE_DIM).cpu().numpy()
        n_keep = min(MAX_PIXELS_PER_IMAGE, feat.shape[0])
        idx = np.random.choice(feat.shape[0], n_keep, replace=False)
        all_features_list.append(feat[idx].astype(np.float32))
        torch.cuda.empty_cache()

    all_features = np.concatenate(all_features_list, axis=0)
    print(f"Total features: {all_features.shape[0]}, dim: {all_features.shape[1]}")

    mean = np.mean(all_features, axis=0).astype(np.float32)
    centered = all_features - mean
    n_comp = min(N_COMPONENTS, centered.shape[0], centered.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(centered)
    components = pca.components_.astype(np.float32)

    vim_gallery = {
        "components": components,
        "mean": mean,
        "alpha": ALPHA_DEFAULT,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    with open(out_path, "wb") as f:
        pickle.dump(vim_gallery, f)
    print(f"Saved {out_path}, components: {components.shape}, mean: {mean.shape}, alpha: {ALPHA_DEFAULT}")


if __name__ == "__main__":
    main()
