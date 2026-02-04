"""
Tune ActSub (ASH) by testing different percentile values.
Loops through p ∈ [65, 80, 90, 95] and prints AUROC for each.
"""

import torch
import numpy as np
from glob import glob
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import network.modeling as modeling
from network.ood_wrapper import wrap_model_for_ood
from ood_methods import ash_score
from sklearn.metrics import roc_auc_score
import os
import json

# Configuration
MODEL_PATH = "/visinf/projects_students/groupL/rkowlu/ood_benchmark/models/deeplabv3plus_r101/deeplab_r101.pth"
NUM_CLASSES = 19
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (adjust these to your actual paths)
LOGITS_DIR = "/fastdata/groupL/datasets/mapillary/validation/logits/"  # Directory with pre-computed logits
MASK_DIR = "/fastdata/groupL/datasets/mapillary/v1.2/validation/ood_masks/"
IMAGE_LIST_PATH = "test_list.txt"  # List of test images

# Percentile values to test
PERCENTILES = [65, 80, 90, 95]

def load_model():
    """Load and wrap the model for OOD detection."""
    print("Loading model...")
    base_model = modeling.__dict__["deeplabv3plus_resnet101"](num_classes=NUM_CLASSES, output_stride=16)
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    state = checkpoint.get("model_state") or checkpoint.get("state_dict")
    base_model.load_state_dict(state, strict=False)
    base_model.to(DEVICE)
    base_model.eval()
    
    model = wrap_model_for_ood(base_model)
    return model

def evaluate_actsub_percentile(model, percentile, image_list, mask_dir):
    """
    Evaluate ActSub (ASH) with a specific percentile value.
    
    Args:
        model: OOD model wrapper
        percentile: Percentile value to test
        image_list: List of image paths
        mask_dir: Directory containing OOD masks
    
    Returns:
        auroc: AUROC score
    """
    all_scores = []
    all_labels = []
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print(f"  Evaluating percentile {percentile}...")
    for img_path in tqdm(image_list[:50], desc=f"p={percentile}"):  # Limit to 50 images for speed
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            x = transform(image).unsqueeze(0).to(DEVICE)
            
            # Get OOD scores using ASH
            ood_scores = ash_score(model, x, percentile=percentile, method='maxlogit', pruning=True)
            
            # Convert to numpy
            if isinstance(ood_scores, torch.Tensor):
                ood_scores = ood_scores.cpu().numpy()
            
            # Load mask
            base_name = os.path.basename(img_path).replace('.jpg', '.png')
            mask_path = os.path.join(mask_dir, base_name)
            
            if not os.path.exists(mask_path):
                continue
            
            mask_img = Image.open(mask_path).convert("L")
            mask_gt = (np.array(mask_img) > 127).astype(np.uint8)
            
            # Resize mask to match score map
            from PIL import Image
            score_size = (ood_scores.shape[1], ood_scores.shape[0])
            mask_resized = mask_img.resize(score_size, Image.NEAREST)
            mask_gt = (np.array(mask_resized) > 127).astype(np.uint8)
            
            # Flatten and collect
            all_scores.append(ood_scores.flatten())
            all_labels.append(mask_gt.flatten())
            
        except Exception as e:
            print(f"    Error processing {img_path}: {e}")
            continue
    
    if len(all_scores) == 0:
        return None
    
    # Concatenate all scores and labels
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    
    # Calculate AUROC
    try:
        auroc = roc_auc_score(all_labels, all_scores)
        return auroc
    except Exception as e:
        print(f"    Error calculating AUROC: {e}")
        return None

def main():
    """Main tuning loop."""
    print("=" * 60)
    print("ActSub (ASH) Percentile Tuning")
    print("=" * 60)
    
    # Load model
    model = load_model()
    
    # Load image list
    if not os.path.exists(IMAGE_LIST_PATH):
        print(f"Error: Image list file not found: {IMAGE_LIST_PATH}")
        return
    
    with open(IMAGE_LIST_PATH, 'r') as f:
        image_list = [line.strip() for line in f.readlines()]
    
    print(f"Loaded {len(image_list)} images from {IMAGE_LIST_PATH}")
    print(f"Testing percentiles: {PERCENTILES}")
    print()
    
    # Test each percentile
    results = {}
    for percentile in PERCENTILES:
        auroc = evaluate_actsub_percentile(model, percentile, image_list, MASK_DIR)
        if auroc is not None:
            results[percentile] = auroc
            print(f"  Percentile {percentile}: AUROC = {auroc:.4f}")
        else:
            print(f"  Percentile {percentile}: FAILED")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary Results:")
    print("=" * 60)
    print(f"{'Percentile':<12} {'AUROC':<10}")
    print("-" * 60)
    for percentile in PERCENTILES:
        if percentile in results:
            print(f"{percentile:<12} {results[percentile]:.4f}")
        else:
            print(f"{percentile:<12} FAILED")
    
    # Find best percentile
    if results:
        best_percentile = max(results, key=results.get)
        best_auroc = results[best_percentile]
        print(f"\nBest percentile: {best_percentile} (AUROC = {best_auroc:.4f})")
    
    # Save results
    output_file = "actsub_tuning_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == '__main__':
    main()
