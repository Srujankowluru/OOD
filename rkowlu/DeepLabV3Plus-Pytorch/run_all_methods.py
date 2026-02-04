"""
Unified Pipeline to Run All OOD Detection Methods.

This script provides a single entry point to run all OOD detection methods
on your dataset with proper feature extraction and evaluation.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os
import pickle
import argparse
import re
from glob import glob
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

import network.modeling as modeling
from network.ood_wrapper import wrap_model_for_ood
from ood_methods import get_ood_score, AVAILABLE_METHODS

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 19

# Default paths (override with command line args)
DEFAULT_MODEL_PATH = "/visinf/projects_students/groupL/rkowlu/ood_benchmark/models/deeplabv3plus_r101/deeplab_r101.pth"
DEFAULT_IMAGE_LIST = "test_list.txt"
DEFAULT_MASK_DIR = "/fastdata/groupL/datasets/mapillary/v1.2/validation/ood_masks/"
DEFAULT_OUTPUT_DIR = "./results_all_methods/"


def load_model(model_path):
    """Load and wrap the model for OOD detection."""
    print(f"Loading model from {model_path}...")
    base_model = modeling.__dict__["deeplabv3plus_resnet101"](num_classes=NUM_CLASSES, output_stride=16)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    state = checkpoint.get("model_state") or checkpoint.get("state_dict")
    if state is None:
        raise KeyError(f"Checkpoint has no 'model_state' or 'state_dict'. Keys: {list(checkpoint.keys())}")
    base_model.load_state_dict(state, strict=False)
    base_model.to(DEVICE)
    base_model.eval()
    
    model = wrap_model_for_ood(base_model)
    return model


def load_galleries(gallery_dir="./"):
    """Load all required galleries."""
    galleries = {}
    
    # Mahalanobis gallery
    maha_path = os.path.join(gallery_dir, "mahalanobis_gallery_bottleneck.pkl")
    if os.path.exists(maha_path):
        with open(maha_path, 'rb') as f:
            galleries['mahalanobis'] = pickle.load(f)
        # Print gallery info for debugging
        pca = galleries['mahalanobis']['pca']
        print(f"  ✓ Loaded Mahalanobis gallery: {pca.n_features_in_} input dims -> {pca.n_components_} PCA dims")
    else:
        print(f"  ⚠ Mahalanobis gallery not found: {maha_path}")
    
    # Layer4 gallery (for Mahalanobis++)
    layer4_path = os.path.join(gallery_dir, "mahalanobis_gallery_layer4.pkl")
    if os.path.exists(layer4_path):
        with open(layer4_path, 'rb') as f:
            galleries['mahalanobis_layer4'] = pickle.load(f)
        pca = galleries['mahalanobis_layer4']['pca']
        print(f"  ✓ Loaded Layer4 gallery: {pca.n_features_in_} input dims -> {pca.n_components_} PCA dims")
    # Layer3 gallery (for Mahalanobis++; build with build_feature_gallery_layer3.py)
    layer3_path = os.path.join(gallery_dir, "mahalanobis_gallery_layer3.pkl")
    if os.path.exists(layer3_path):
        with open(layer3_path, 'rb') as f:
            galleries['mahalanobis_layer3'] = pickle.load(f)
        pca = galleries['mahalanobis_layer3']['pca']
        print(f"  ✓ Loaded Layer3 gallery: {pca.n_features_in_} input dims -> {pca.n_components_} PCA dims")
    
    # kNN gallery (can be same as Mahalanobis features)
    if 'mahalanobis' in galleries:
        # Extract features from Mahalanobis gallery for kNN
        # This is a simplified approach - ideally build separate kNN gallery
        maha_gallery = galleries['mahalanobis']
        if 'pca' in maha_gallery:
            # We'd need original features, but for now skip kNN if no separate gallery
            pass
    
    # VIM gallery
    vim_path = os.path.join(gallery_dir, "vim_gallery.pkl")
    if os.path.exists(vim_path):
        with open(vim_path, 'rb') as f:
            galleries['vim'] = {'gallery_vim': pickle.load(f)}
        print(f"  ✓ Loaded VIM gallery from {vim_path}")

    # kNN gallery
    knn_path = os.path.join(gallery_dir, "knn_gallery.pkl")
    if os.path.exists(knn_path):
        with open(knn_path, 'rb') as f:
            galleries['knn'] = pickle.load(f)
        print(f"  ✓ Loaded kNN gallery from {knn_path}")
    
    # Gram gallery
    gram_path = os.path.join(gallery_dir, "gram_gallery.pkl")
    if os.path.exists(gram_path):
        with open(gram_path, 'rb') as f:
            galleries['gram'] = {'gallery_gram': pickle.load(f)}
        print(f"  ✓ Loaded Gram gallery from {gram_path}")

    # ActSub insignificant-subspace gallery (full paper; optional)
    actsub_path = os.path.join(gallery_dir, "actsub_gallery.pkl")
    if os.path.exists(actsub_path):
        with open(actsub_path, 'rb') as f:
            galleries['actsub'] = pickle.load(f)
        g = galleries['actsub']
        n = g.shape[0] if hasattr(g, 'shape') else len(g)
        print(f"  ✓ Loaded ActSub gallery: {n} samples (insignificant subspace)")
    
    return galleries


def load_actsub_lambda(gallery_dir):
    """Load paper-faithful tuned λ if available (from tune_actsub_lambda.py). Else 1.0."""
    import json
    path = os.path.join(gallery_dir, "actsub_best_lambda.json")
    if not os.path.isfile(path):
        return 1.0
    try:
        with open(path, "r") as f:
            data = json.load(f)
        lam = data.get("lambda", 1.0)
        print(f"  ✓ ActSub using tuned λ = {lam} from {path}")
        return lam
    except Exception:
        return 1.0


def process_image(model, image_path, transform, method, galleries=None, max_size=1024, target_size=None, **method_kwargs):
    """
    Process a single image with specified method.
    All methods output scores at target_size (logit resolution) for consistent comparison.
    
    Args:
        max_size: Maximum dimension for image (to save memory). Images will be resized if larger.
        target_size: (H, W) output resolution for all methods; if None, use logit spatial size from this image.
    
    Returns:
        score_map: numpy array [H, W] at target_size with OOD scores
        feature_shape: tuple of feature shape for debugging
    """
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (W, H)
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.BILINEAR)
        if original_size != new_size and method_kwargs.get('verbose'):
            print(f"  Resized {original_size} -> {new_size}")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    torch.cuda.empty_cache()
    
    if method and (method.lower() == 'mahalanobis++' or method.lower() == 'mahalanobis_plus_plus'):
        layer = method_kwargs.get('layer', 'layer4')
        with torch.no_grad():
            logits, features = model(input_tensor, return_layer=layer)
        feature_shape = features.shape
        gallery_dict = galleries.get('mahalanobis_layer3' if layer == 'layer3' else 'mahalanobis_layer4', {}) if galleries else {}
    else:
        with torch.no_grad():
            logits, features = model(input_tensor, return_features=True)
        feature_shape = features.shape
        gallery_dict = {}
    
    if target_size is None:
        target_size = (logits.shape[2], logits.shape[3])
    
    if galleries and method and method.lower() != 'mahalanobis_plus_plus':
        if method.lower() == 'mahalanobis' and 'mahalanobis' in galleries:
            gallery_dict = galleries['mahalanobis']
        elif method.lower() == 'knn' and 'knn' in galleries:
            gallery_dict = {'gallery_knn': galleries['knn']}
        elif method.lower() == 'vim' and 'vim' in galleries:
            gallery_dict = galleries['vim']
        elif method.lower() == 'gram' and 'gram' in galleries:
            gallery_dict = galleries['gram']
        elif method.lower() == 'actsub' and 'actsub' in galleries:
            gallery_dict = {'actsub_ins': galleries['actsub']}
    
    try:
        method_model = model if method.lower() in ['react', 'ash', 'actsub', 'mc_dropout'] else None
        scores = get_ood_score(
            logits=logits,
            method=method,
            features=features,
            model=method_model,
            input_tensor=input_tensor,
            gallery=gallery_dict,
            target_size=target_size,
            **method_kwargs
        )
        
        # Convert to numpy (detach first to avoid gradient issues)
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        
        # Ensure 2D: Handle different input shapes
        # Expected outputs: [H, W] (2D) or [B, H, W] (3D) or [B, C, H, W] (4D - wrong)
        if scores.ndim == 2:
            # Already 2D, perfect!
            pass
        elif scores.ndim == 3:
            # [B, H, W] - squeeze batch dimension
            scores = scores.squeeze(0)
        elif scores.ndim == 4:
            # [B, C, H, W] - shouldn't happen, but handle it
            scores = scores.squeeze(0).squeeze(0)
        else:
            raise ValueError(f"Unexpected score dimension: {scores.ndim}. Expected 2D [H, W], got shape {scores.shape}")
        
        # Final check: should be 2D now
        if scores.ndim != 2:
            raise ValueError(f"Scores should be 2D [H, W] after processing, but got shape {scores.shape}")
        
        return scores, feature_shape
        
    except Exception as e:
        if "dimension mismatch" in str(e).lower():
            print(f"  ⚠ Error processing {os.path.basename(image_path)} with {method}: {e}")
            import traceback
            traceback.print_exc()
        else:
            print(f"  ⚠ Error processing {os.path.basename(image_path)} with {method}: {e}")
        return None, None


def _method_folder_name(method):
    """Safe folder name (e.g. Mahalanobis++ -> Mahalanobis_plus_plus)."""
    return re.sub(r'[^\w]', '_', method.replace('+', '_plus_'))


def save_ood_vis(scores_np, out_path):
    """Save OOD score map as 8-bit grayscale PNG (higher = more OOD)."""
    s = np.asarray(scores_np, dtype=np.float64)
    valid = np.isfinite(s)
    if not valid.any():
        return
    lo, hi = np.percentile(s[valid], [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    s = np.clip((s - lo) / (hi - lo), 0, 1)
    s = (s * 255).astype(np.uint8)
    Image.fromarray(s).save(out_path)


def evaluate_method(scores_list, labels_list, method_name):
    """Evaluate with auto-polarity correction."""
    if len(scores_list) == 0:
        return None
    
    all_scores = np.concatenate([s.flatten() for s in scores_list])
    all_labels = np.concatenate([l.flatten() for l in labels_list])
    
    # Remove any NaN or Inf
    valid_mask = np.isfinite(all_scores)
    if not valid_mask.all():
        print(f"  ⚠ {method_name}: Removing {(~valid_mask).sum()} invalid scores")
        all_scores = all_scores[valid_mask]
        all_labels = all_labels[valid_mask]
    
    if len(all_scores) == 0:
        return None
    
    try:
        auroc = roc_auc_score(all_labels, all_scores)
        
        # AUTO-FIX: If AUROC < 0.5, scores are inverted!
        polarity_flipped = False
        if auroc < 0.5:
            print(f"  ⚠ {method_name}: AUROC={auroc:.4f} < 0.5 → SCORES INVERTED! Flipping...")
            all_scores = -all_scores
            auroc = roc_auc_score(all_labels, all_scores)
            polarity_flipped = True
        
        ap = average_precision_score(all_labels, all_scores)
        
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        tpr_95_idx = np.where(tpr >= 0.95)[0]
        fpr_at_95_tpr = fpr[tpr_95_idx[0]] if len(tpr_95_idx) > 0 else 1.0
        
        return {
            'method': method_name,
            'auroc': auroc,
            'ap': ap,
            'fpr_at_95_tpr': fpr_at_95_tpr,
            'polarity_flipped': polarity_flipped,
            'score_range': (float(all_scores.min()), float(all_scores.max())),
            'n_samples': len(all_scores),
            'ood_ratio': float(all_labels.mean()),
            'fpr': fpr,
            'tpr': tpr
        }
    except Exception as e:
        print(f"  ⚠ Error evaluating {method_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Run all OOD detection methods')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to model checkpoint')
    parser.add_argument('--image-list', type=str, default=DEFAULT_IMAGE_LIST,
                        help='Path to image list (use test_list.txt for final eval; val_list.txt for tuning). See METHODOLOGY.md.')
    parser.add_argument('--mask-dir', type=str, default=DEFAULT_MASK_DIR,
                        help='Directory containing OOD masks')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for results')
    parser.add_argument('--methods', type=str, default=None,
                        help='Comma-separated list of methods to run (default: all)')
    parser.add_argument('--start', type=int, default=0,
                        help='Start index in image list')
    parser.add_argument('--end', type=int, default=None,
                        help='End index in image list (default: all)')
    parser.add_argument('--gallery-dir', type=str, default='./',
                        help='Directory containing gallery files')
    parser.add_argument('--max-size', type=int, default=1024,
                        help='Maximum image dimension (resize if larger, default=1024)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Max number of images to process (default: all in list)')
    parser.add_argument('--plot', action='store_true',
                        help='Save AUROC and FPR@95 bar chart to output dir')
    parser.add_argument('--save-vis', type=int, default=0, metavar='N',
                        help='Save OOD score maps for first N images per method under output-dir/test_images/<method>/ (e.g. 20)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--mahalanobis-layer', type=str, default='layer4', choices=['layer3', 'layer4'],
                        help='Backbone layer for Mahalanobis++ (default: layer4; use layer3 for improved layer3 gallery)')
    parser.add_argument('--only-missing', action='store_true',
                        help='Only run methods not already in output-dir/results_summary.json; merge new results with existing')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine methods to run (galleries loaded below; we add kNN/VIM after)
    if args.methods:
        methods_to_run = [m.strip() for m in args.methods.split(',')]
    else:
        methods_to_run = AVAILABLE_METHODS.copy()
        methods_to_run = [m for m in methods_to_run if m not in ['MC_Dropout', 'GEM', 'RankFeat']]
    
    # Only-missing: run only methods not in existing results (merge into existing later)
    if args.only_missing:
        results_json = os.path.join(args.output_dir, 'results_summary.json')
        if os.path.isfile(results_json):
            import json
            with open(results_json, 'r') as f:
                existing = json.load(f)
            missing = [m for m in methods_to_run if m not in existing]
            if not missing:
                print('Only-missing: all methods already in results. Exiting.')
                return
            methods_to_run = missing
            print(f'Only-missing: running {len(methods_to_run)} methods not in existing results: {methods_to_run}')
        else:
            print('--only-missing: no existing results_summary.json; running all requested methods.')
    
    # Load galleries first so we can include kNN/VIM when galleries exist
    print("\nLoading galleries...")
    galleries = load_galleries(args.gallery_dir)
    actsub_lambda = load_actsub_lambda(args.gallery_dir)
    if not args.methods:
        if 'knn' in galleries:
            if 'kNN' not in methods_to_run:
                methods_to_run.append('kNN')
        if 'vim' in galleries:
            if 'VIM' not in methods_to_run:
                methods_to_run.append('VIM')
    
    print(f"Methods to run: {methods_to_run}")
    print(f"Total: {len(methods_to_run)} methods")
    
    # Load model
    model = load_model(args.model)
    
    # Dimension check (and optional debug: logit vs feature resolution)
    print("\n--- Feature Dimension Check ---")
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 256, 512).to(DEVICE)
        logits_d, features_d = model(dummy_input, return_features=True)
        print(f"Model outputs features with shape: {features_d.shape}")
        if args.debug:
            print(f"  Logit spatial size: ({logits_d.shape[2]}, {logits_d.shape[3]}) (target_size for all methods)")
            print(f"  Feature spatial size: ({features_d.shape[2]}, {features_d.shape[3]}) (upsampled to target when needed)")
        if 'mahalanobis' in galleries:
            expected_dim = galleries['mahalanobis']['pca'].n_features_in_
            if features_d.shape[1] != expected_dim:
                print(f"⚠ DIMENSION MISMATCH! Model: {features_d.shape[1]}, Gallery: {expected_dim}")
            else:
                print(f"✓ Mahalanobis dimensions match: {features_d.shape[1]}")
        del dummy_input, logits_d, features_d
    torch.cuda.empty_cache()
    
    # Load image list
    with open(args.image_list, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    
    if args.end is None:
        args.end = len(image_paths)
    
    image_paths = image_paths[args.start:args.end]
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    print(f"\nProcessing {len(image_paths)} images...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Determine target resolution once (logit size after resize) for consistent comparison
    print("\nDetermining target resolution (logit size)...")
    first_img = Image.open(image_paths[0]).convert('RGB')
    if max(first_img.size) > args.max_size:
        ratio = args.max_size / max(first_img.size)
        first_size = (int(first_img.size[0] * ratio), int(first_img.size[1] * ratio))
        first_img = first_img.resize(first_size, Image.BILINEAR)
    first_tensor = transform(first_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        first_logits, _ = model(first_tensor, return_features=True)
    target_size = (first_logits.shape[2], first_logits.shape[3])
    print(f"  Target resolution for all methods: {target_size} (H, W)")
    del first_img, first_tensor, first_logits
    torch.cuda.empty_cache()
    
    # Load existing results/thresholds if only-missing (merge new into these)
    import json
    all_results = {}
    method_thresholds = {}
    if args.only_missing:
        results_json = os.path.join(args.output_dir, 'results_summary.json')
        if os.path.isfile(results_json):
            with open(results_json, 'r') as f:
                existing_dict = json.load(f)
            for k, v in existing_dict.items():
                all_results[k] = {
                    'auroc': v['auroc'], 'ap': v['ap'], 'fpr_at_95_tpr': v['fpr_at_95_tpr'],
                    'polarity_flipped': v.get('polarity_flipped', False),
                    'score_range': v.get('score_range', (0, 0)), 'n_samples': v.get('n_samples', 0),
                    'ood_ratio': v.get('ood_ratio', 0),
                }
        thresh_file = os.path.join(args.output_dir, 'method_thresholds.json')
        if os.path.isfile(thresh_file):
            with open(thresh_file, 'r') as f:
                method_thresholds = json.load(f)

    for method in methods_to_run:
        print(f"\n{'='*60}")
        print(f"Processing method: {method}")
        print(f"{'='*60}")
        
        scores_list = []
        labels_list = []
        n_vis_saved_no_mask = 0  # OOD maps saved when mask missing (sanity check without labels)
        
        feature_shape_logged = False
        
        method_kwargs_extra = {}
        if method.lower() == "actsub":
            method_kwargs_extra["lambda_"] = actsub_lambda
        if method.lower() in ("mahalanobis++", "mahalanobis_plus_plus"):
            method_kwargs_extra["layer"] = args.mahalanobis_layer
        for img_path in tqdm(image_paths, desc=method):
            scores, feat_shape = None, None
            try:
                scores, feat_shape = process_image(model, img_path, transform, method, galleries, max_size=args.max_size, target_size=target_size, **method_kwargs_extra)
            except (RuntimeError, ValueError) as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    for retry_size in [max(256, args.max_size // 2), max(256, args.max_size // 4)]:
                        if retry_size >= 256:
                            try:
                                scores, feat_shape = process_image(model, img_path, transform, method, galleries, max_size=retry_size, target_size=target_size, **method_kwargs_extra)
                                break
                            except (RuntimeError, ValueError):
                                torch.cuda.empty_cache()
                                continue
                    if scores is None:
                        print(f"  ⚠ OOM on {os.path.basename(img_path)}, skipping...")
                        torch.cuda.empty_cache()
                        continue
                elif "dimension mismatch" in str(e).lower():
                    print(f"  ⚠ {method} dimension mismatch: {e}")
                    print(f"  ⚠ Skipping {method} for all images")
                    break
                elif "requires gallery" in str(e).lower() or "requires model" in str(e).lower():
                    print(f"  ⚠ {method} requires missing resources: {e}")
                    print(f"  ⚠ Skipping {method} for all images")
                    break
                else:
                    if args.debug:
                        import traceback
                        traceback.print_exc()
                    continue
            
            if scores is None:
                continue
            
            # Log feature shape once per method
            if args.debug and not feature_shape_logged and feat_shape is not None:
                print(f"\n  Feature shape: {feat_shape}")
                print(f"  Score shape: {scores.shape}")
                print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                feature_shape_logged = True
            
            torch.cuda.empty_cache()
            
            # Load mask (mask file = stem.png for any image extension)
            stem = os.path.basename(img_path).rsplit('.', 1)[0]
            mask_path = os.path.join(args.mask_dir, stem + '.png')
            
            if not os.path.exists(mask_path):
                # No mask: still save OOD vis for first N images if requested (e.g. sanity check without labels)
                if args.save_vis and n_vis_saved_no_mask < args.save_vis:
                    vis_dir = os.path.join(args.output_dir, 'test_images', _method_folder_name(method))
                    os.makedirs(vis_dir, exist_ok=True)
                    save_ood_vis(scores, os.path.join(vis_dir, stem + '_ood.png'))
                    np.save(os.path.join(vis_dir, stem + '_scores.npy'), np.asarray(scores, dtype=np.float64))
                    n_vis_saved_no_mask += 1
                continue
            
            mask_img = Image.open(mask_path).convert("L")
            # Resize mask to target_size (same as scores: consistent resolution for all methods)
            mask_resized = mask_img.resize((target_size[1], target_size[0]), Image.NEAREST)
            mask_gt = (np.array(mask_resized) > 127).astype(np.uint8)
            
            if scores.shape != mask_gt.shape:
                raise AssertionError(f"Score shape {scores.shape} != mask shape {mask_gt.shape} (target_size={target_size})")
            
            # Save OOD score visualization for first N images per method (same images across methods)
            stem = os.path.basename(img_path).rsplit('.', 1)[0]
            if args.save_vis and len(scores_list) < args.save_vis:
                vis_dir = os.path.join(args.output_dir, 'test_images', _method_folder_name(method))
                os.makedirs(vis_dir, exist_ok=True)
                base_name = stem + '_ood.png'
                out_path = os.path.join(vis_dir, base_name)
                save_ood_vis(scores, out_path)
                # Save raw scores for correct binary threshold (pixel-level ID threshold)
                np.save(os.path.join(vis_dir, stem + '_scores.npy'), np.asarray(scores, dtype=np.float64))
            
            scores_list.append(scores)
            labels_list.append(mask_gt)
        
        # Evaluate method and compute pixel-level ID threshold (95th percentile of ID pixels)
        if len(scores_list) > 0:
            # Pixel-level ID scores (label 0 = ID) for correct binary threshold
            id_scores_list = []
            for s, lbl in zip(scores_list, labels_list):
                flat_s = np.asarray(s, dtype=np.float64).flatten()
                flat_l = np.asarray(lbl, dtype=np.uint8).flatten()
                id_mask = (flat_l == 0) & np.isfinite(flat_s)
                if id_mask.any():
                    id_scores_list.append(flat_s[id_mask])
            if id_scores_list:
                id_pixel_scores = np.concatenate(id_scores_list)
                threshold_95 = float(np.percentile(id_pixel_scores, 95))
                method_thresholds[_method_folder_name(method)] = threshold_95
            results = evaluate_method(scores_list, labels_list, method)
            if results:
                all_results[method] = results
                print(f"\n{method} Results:")
                print(f"  AUROC: {results['auroc']:.4f}")
                print(f"  AP:    {results['ap']:.4f}")
                print(f"  FPR@95%TPR: {results['fpr_at_95_tpr']:.4f}")
                if results.get('polarity_flipped', False):
                    print(f"  ⚠ Polarity was auto-corrected!")
                print(f"  Score range: {results.get('score_range', 'N/A')}")
                print(f"  OOD ratio: {results.get('ood_ratio', 0):.2%}")
        else:
            if n_vis_saved_no_mask > 0:
                print(f"  No masks: AUROC skipped for {method}; OOD maps saved for {n_vis_saved_no_mask} images in test_images/")
            else:
                print(f"  ⚠ No valid results for {method}")
        # Force-clear GPU memory between methods
        del scores_list, labels_list
        torch.cuda.empty_cache()
    
    # Save results (all_results already contains existing when --only-missing)
    results_file = os.path.join(args.output_dir, "results_summary.json")
    results_dict = {}
    for k, v in all_results.items():
        results_dict[k] = {
            'auroc': float(v['auroc']),
            'ap': float(v['ap']),
            'fpr_at_95_tpr': float(v['fpr_at_95_tpr']),
            'polarity_flipped': v.get('polarity_flipped', False),
            'score_range': v.get('score_range', (0, 0)),
            'n_samples': v.get('n_samples', 0),
            'ood_ratio': v.get('ood_ratio', 0),
        }
    
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    # Save pixel-level ID thresholds (95th percentile) for binary masks in create_ood_overlays
    if method_thresholds:
        thresh_file = os.path.join(args.output_dir, "method_thresholds.json")
        with open(thresh_file, "w") as f:
            json.dump(method_thresholds, f, indent=2)
        print(f"Method thresholds (ID pixel 95th pct) saved to {thresh_file}")

    # Also save CSV for easy tables
    import csv
    csv_file = os.path.join(args.output_dir, "results_summary.csv")
    with open(csv_file, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'auroc', 'fpr_at_95_tpr', 'ap', 'n_samples', 'ood_ratio'])
        for m, v in results_dict.items():
            w.writerow([m, v['auroc'], v['fpr_at_95_tpr'], v['ap'], v['n_samples'], v['ood_ratio']])
    print(f"Results CSV saved to {csv_file}")

    # Optional: plot AUROC and FPR@95
    if args.plot and all_results:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            methods = list(all_results.keys())
            aurocs = [all_results[m]['auroc'] for m in methods]
            fpr95s = [all_results[m]['fpr_at_95_tpr'] for m in methods]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            x = np.arange(len(methods))
            w = 0.35
            ax1.bar(x - w/2, aurocs, w, label='AUROC', color='steelblue')
            ax1.set_xticks(x)
            ax1.set_xticklabels(methods, rotation=45, ha='right')
            ax1.set_ylabel('AUROC')
            ax1.set_ylim(0, 1.05)
            ax1.legend()
            ax1.set_title('AUROC (higher is better)')
            ax2.bar(x - w/2, fpr95s, w, label='FPR@95%TPR', color='coral')
            ax2.set_xticks(x)
            ax2.set_xticklabels(methods, rotation=45, ha='right')
            ax2.set_ylabel('FPR@95%')
            ax2.set_ylim(0, 1.05)
            ax2.legend()
            ax2.set_title('FPR@95% TPR (lower is better)')
            plt.tight_layout()
            plot_path = os.path.join(args.output_dir, 'metrics_auroc_fpr95.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Metrics plot saved to {plot_path}")
        except Exception as e:
            print(f"  ⚠ Could not save plot: {e}")
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'AUROC':<10} {'AP':<10} {'FPR@95%':<12} {'Flipped?'}")
    print("-" * 60)
    
    # Sort by AUROC descending
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['auroc'], reverse=True)
    for method, results in sorted_results:
        flip_marker = "⚠ YES" if results.get('polarity_flipped', False) else "No"
        print(f"{method:<15} {results['auroc']:>9.4f} {results['ap']:>9.4f} {results['fpr_at_95_tpr']:>11.4f} {flip_marker}")
    
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
