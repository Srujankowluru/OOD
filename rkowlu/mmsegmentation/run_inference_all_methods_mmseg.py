import torch
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import mmcv # MMSeg's helper library
from mmseg.apis import init_model, inference_model
from mmengine.registry import init_default_scope
from mmengine.config import Config # New import

# --- 1. OOD METHOD DEFINITIONS (GPU-based) ---
def score_msp(logits):
    probs = F.softmax(logits, dim=1) 
    conf, _ = torch.max(probs, dim=1)
    return (1.0 - conf).cpu().numpy() 

def score_maxlogit(logits):
    conf, _ = torch.max(logits, dim=1)
    return -conf.cpu().numpy()

def score_energy(logits, T=1.0):
    return -T * torch.logsumexp(logits / T, dim=1).cpu().numpy()

def score_entropy(logits):
    probs = F.softmax(logits, dim=1)
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1).cpu().numpy()

def score_react(logits, clip_quantile=0.99):
    logits_flat = logits.flatten()
    sample_size = 1_000_000
    if logits_flat.numel() > sample_size:
        indices = torch.randperm(logits_flat.numel(), device=logits.device)[:sample_size]
        logits_sample = logits_flat[indices]
    else:
        logits_sample = logits_flat
    clip_val = torch.quantile(logits_sample, clip_quantile)
    logits_clipped = torch.clamp(logits, max=clip_val)
    return -torch.logsumexp(logits_clipped, dim=1).cpu().numpy()

OOD_METHODS = {
    "MSP": score_msp, "MaxLogit": score_maxlogit, "Energy": score_energy,
    "Entropy": score_entropy, "ReAct": score_react
}

# --- 2. CONFIGURATION ---
IMAGE_DIR = "/fastdata/groupL/datasets/mapillary/validation/images/"
BASE_OUTPUT_DIR = "/fastdata/groupL/datasets/mapillary/validation/mmseg_deeplab_scores_fullres/" # New folder!

CONFIG_FILE = 'configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024.py'
CHECKPOINT_FILE = 'checkpoints/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-0683251e.pth'

OUTPUT_DIRS = {name: os.path.join(BASE_OUTPUT_DIR, f"{name}_scores") for name in OOD_METHODS}
for dir_path in OUTPUT_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

# --- 3. LOAD MODEL (MMSegmentation Way) ---
print(f"Loading model from: {CONFIG_FILE}...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the config file
cfg = Config.fromfile(CONFIG_FILE)

# MMSeg needs this to find its modules
init_default_scope(cfg.get('default_scope', 'mmseg'))

model = init_model(cfg, CHECKPOINT_FILE, device=device)
print(f"Model loaded successfully. Using device: {device}")

# --- 4. SCRIPT ---
image_files = sorted(glob(os.path.join(IMAGE_DIR, "*.jpg")))
print(f"Found {len(image_files)} images. Starting inference...")

# --- Processing the first 100 images ---
print(f"Processing {len(image_files[:100])} images...")
for image_path in tqdm(image_files[:10]):
    try:
        # We must use mmcv to read the image
        image_bgr = mmcv.imread(image_path)
        image_rgb = mmcv.bgr2rgb(image_bgr)
        image_size = (image_rgb.shape[0], image_rgb.shape[1]) # (height, width)

        with torch.no_grad():
            # --- Run MMSeg Inference ---
            # We call the model's 'encode_decode' method to get logits
            # We need to manually create the 'data_samples' metadata
            img_tensor = torch.from_numpy(image_rgb).permute(2,0,1).unsqueeze(0).to(device).float()
            data_samples = [{'img_path': image_path, 'img_shape': image_size, 'ori_shape': image_size}]

            logits = model.encode_decode(img_tensor, data_samples)

            # --- Upsample logits ONCE (on GPU) ---
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image_size,
                mode="bilinear",
                align_corners=False
            ) # Shape (1, 19, H, W)

            # --- Calculate ALL 5 OOD scores (on GPU) ---
            base_name = os.path.basename(image_path)
            npy_name = os.path.splitext(base_name)[0] + ".npy"

            for method_name, method_func in OOD_METHODS.items():
                ood_score_map = method_func(upsampled_logits)
                output_path = os.path.join(OUTPUT_DIRS[method_name], npy_name)
                np.save(output_path, ood_score_map.squeeze(0))

        torch.cuda.empty_cache() 

    except Exception as e:
        print(f"\n!!! FAILED on image: {image_path} !!! Error: {e}")
        print("Skipping this file.")

print(f"\n--- ALL DONE! ---")
print(f"Generated {len(image_files[:100])} score maps for 5 methods in {BASE_OUTPUT_DIR}")
