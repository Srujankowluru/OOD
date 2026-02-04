"""
OOD Detection Methods Implementation - FIXED VERSION
All methods verified against OpenOOD and original papers.

Resolution: All methods can output at a consistent target_size (logit resolution)
via target_size parameter and _resize_scores_to_target for fair comparison.
"""

import torch
import torch.nn.functional as F
import numpy as np


# =============================================================================
# RESOLUTION UTILITIES (consistent output size across methods)
# =============================================================================

def _get_spatial_size(tensor):
    """Get (H, W) from tensor [B, C, H, W], [B, H, W], or [H, W]."""
    if tensor.dim() == 2:
        return tuple(tensor.shape)
    if tensor.dim() == 3:
        return tuple(tensor.shape[1:])
    if tensor.dim() == 4:
        return tuple(tensor.shape[2:])
    raise ValueError(f"Unexpected tensor dim: {tensor.dim()}")


def _resize_scores_to_target(scores, target_size):
    """
    Resize score map to target (H, W). Preserves torch vs numpy.
    target_size: (H, W).
    """
    if scores is None or target_size is None:
        return scores
    is_numpy = isinstance(scores, np.ndarray)
    if is_numpy:
        t = torch.from_numpy(scores).float()
    else:
        t = scores.float()
    current = _get_spatial_size(t)
    if current == target_size:
        return scores
    orig_dim = t.dim()
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        t = t.unsqueeze(1)
    t = F.interpolate(t, size=target_size, mode='bilinear', align_corners=False)
    if orig_dim == 2:
        t = t.squeeze(0).squeeze(0)
    elif orig_dim == 3:
        t = t.squeeze(1)
    return t.numpy() if is_numpy else t


def msp_score(logits):
    """
    Maximum Softmax Probability (MSP).
    
    Logic: score = torch.max(torch.softmax(logits, dim=1), dim=1).values
    Check: Ensure you apply softmax first! Raw logits are not probabilities.
    
    Args:
        logits: [B, num_classes, H, W] or [num_classes, H, W]
    
    Returns:
        scores: [B, H, W] or [H, W] - Higher score = Higher confidence (ID)
                For OOD detection, use 1.0 - scores (lower confidence = OOD)
    """
    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
    
    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=1)  # [B, num_classes, H, W]
    
    # Get maximum probability across classes
    msp_scores, _ = torch.max(probs, dim=1)  # [B, H, W]
    
    if msp_scores.dim() == 3 and msp_scores.shape[0] == 1:
        msp_scores = msp_scores.squeeze(0)
    
    return msp_scores


def maxlogit_score(logits):
    """
    MaxLogit (Maximum Logit Value).
    
    Logic: score = torch.max(logits, dim=1).values
    Check: Do NOT apply softmax. Softmax normalizes values, hiding the magnitude.
           OOD objects often have low magnitude logits across all classes.
    
    Args:
        logits: [B, num_classes, H, W] or [num_classes, H, W]
    
    Returns:
        scores: [B, H, W] or [H, W] - Higher score = Higher confidence (ID)
                For OOD detection, use -scores (lower magnitude = OOD)
    """
    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
    
    # Get maximum logit value across classes (NO softmax!)
    max_logits, _ = torch.max(logits, dim=1)  # [B, H, W]
    
    if max_logits.dim() == 3 and max_logits.shape[0] == 1:
        max_logits = max_logits.squeeze(0)
    
    return max_logits


def entropy_score(logits, eps=1e-8):
    """
    Entropy-based OOD detection.
    
    Logic: 
        probs = torch.softmax(logits, dim=1)
        score = -torch.sum(probs * torch.log(probs + eps), dim=1)
    
    Check: Add epsilon to prevent log(0) = -inf errors (NaNs).
    
    Args:
        logits: [B, num_classes, H, W] or [num_classes, H, W]
        eps: Small epsilon to prevent log(0)
    
    Returns:
        scores: [B, H, W] or [H, W] - Higher entropy = More uncertain (OOD)
    """
    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
    
    # Apply softmax
    probs = torch.softmax(logits, dim=1)  # [B, num_classes, H, W]
    
    # Calculate entropy: -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)  # [B, H, W]
    
    if entropy.dim() == 3 and entropy.shape[0] == 1:
        entropy = entropy.squeeze(0)
    
    return entropy


def energy_score(logits, T=1.0):
    """
    Energy-based OOD detection.
    
    Logic: score = -T * logsumexp(logits / T, dim=1)
    
    Args:
        logits: [B, num_classes, H, W] or [num_classes, H, W]
        T: Temperature parameter (default 1.0)
    
    Returns:
        scores: [B, H, W] or [H, W] - Higher energy = OOD
    """
    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
    
    # Energy score: -T * log(sum(exp(logits / T)))
    energy_scores = -T * torch.logsumexp(logits / T, dim=1)  # [B, H, W]
    
    if energy_scores.dim() == 3 and energy_scores.shape[0] == 1:
        energy_scores = energy_scores.squeeze(0)
    
    return energy_scores


def react_score(model, x, clip_threshold, method='maxlogit', target_size=None):
    """
    ReAct (Rectified Activation) OOD detection.
    
    Logic: Clip high activations in the feature map before classification.
    
    Args:
        model: OODDeepLab model with forward_features method
        x: Input tensor [B, 3, H, W]
        clip_threshold: Clip threshold (e.g., 90th percentile of training activations)
        method: 'maxlogit' or 'energy' or 'msp'
        target_size: (H, W) for consistent resolution; if None, use input spatial size.
    
    Returns:
        scores: [B, H, W] or [H, W] - OOD scores (higher = more OOD)
    """
    model.eval()
    with torch.no_grad():
        bottleneck_features = model.forward_features(x)
    features_clipped = torch.clamp(bottleneck_features, max=clip_threshold)
    try:
        classifier = model.get_classifier_head()
    except AttributeError:
        try:
            classifier = model.base_model.classifier.classifier
        except AttributeError:
            raise ValueError("ReAct requires model with get_classifier_head()")
    with torch.no_grad():
        logits = classifier(features_clipped)
    out_size = target_size if target_size is not None else x.shape[-2:]
    if logits.shape[-2:] != out_size:
        logits = F.interpolate(logits, size=out_size, mode='bilinear', align_corners=False)
    if method == 'maxlogit':
        scores = -maxlogit_score(logits)
    elif method == 'energy':
        scores = energy_score(logits)
    elif method == 'msp':
        scores = 1.0 - msp_score(logits)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'maxlogit', 'energy', or 'msp'.")
    if target_size is not None and _get_spatial_size(scores) != target_size:
        scores = _resize_scores_to_target(scores, target_size)
    return scores


def ash_score(model, x, percentile=90, method='energy', variant='s', target_size=None, **kwargs):
    """
    ASH (Activation Shaping) OOD detection - FIXED.
    Implements ASH-S (Scale) variant as recommended in paper.
    
    Args:
        model: Model with forward_features and get_classifier_head methods
        x: Input tensor [B, 3, H, W]
        percentile: Percentile of activations to REMOVE (default 90 = keep top 10%)
        method: 'energy', 'maxlogit', or 'msp'
        variant: 's' (scale), 'b' (binarize), 'p' (prune)
        target_size: (H, W) for consistent resolution; if None, use input spatial size.
    
    Returns:
        scores: [B, H, W] - Higher = More OOD
    """
    model.eval()
    with torch.no_grad():
        features = model.forward_features(x)
        B, C, H, W = features.shape
        features_flat = features.view(B, -1)
        k = int(features_flat.shape[1] * (1.0 - percentile / 100.0))
        k = max(k, 1)
        threshold, _ = torch.kthvalue(features_flat, features_flat.shape[1] - k + 1, dim=1, keepdim=True)
        mask = features_flat >= threshold
        if variant == 's':
            s1 = features_flat.sum(dim=1, keepdim=True)
            features_pruned = features_flat * mask.float()
            s2 = features_pruned.sum(dim=1, keepdim=True).clamp(min=1e-6)
            features_shaped = features_pruned * (s1 / s2)
        elif variant == 'b':
            features_shaped = mask.float()
        elif variant == 'p':
            features_shaped = features_flat * mask.float()
        else:
            features_shaped = features_flat * mask.float()
        features_shaped = features_shaped.view(B, C, H, W)
        classifier = model.get_classifier_head()
        logits = classifier(features_shaped)
        out_size = target_size if target_size is not None else x.shape[-2:]
        if logits.shape[-2:] != out_size:
            logits = F.interpolate(logits, size=out_size, mode='bilinear', align_corners=False)
    scores = energy_score(logits) if method == 'energy' else (
        -maxlogit_score(logits) if method == 'maxlogit' else (1.0 - msp_score(logits) if method == 'msp' else energy_score(logits)))
    if method not in ('energy', 'maxlogit', 'msp'):
        raise ValueError(f"Unknown method: {method}")
    if target_size is not None and _get_spatial_size(scores) != target_size:
        scores = _resize_scores_to_target(scores, target_size)
    return scores


# ============================================================================
# ActSub (full paper): decisive subspace (←S) + insignificant subspace (→S)
# - Decisive: SVD of W, project, activation shaping, Energy (no gallery).
# - Insignificant: project training activations, store gallery; test cosine sim (→S).
# - Combined: S↔ = λ · S→ · S← (paper Eq. 10).
# ============================================================================

@torch.no_grad()
def compute_decisive_projection(classifier_weight, k=None):
    """
    Decisive subspace from SVD of classifier weight matrix W (paper Eq.).
    classifier_weight: [num_classes, feat_dim]
    k: number of decisive components (default: min(num_classes, feat_dim))
    Returns P_dec [feat_dim, feat_dim].
    """
    W = classifier_weight.float()
    if W.dim() > 2:
        W = W.squeeze()
    max_k = min(W.shape[0], W.shape[1])
    k = k if k is not None else max_k
    k = min(k, max_k)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    V_dec = Vh[:k].T
    P = V_dec @ V_dec.T
    return P


@torch.no_grad()
def compute_actsub_projections(classifier_weight, k=None):
    """
    Both decisive and insignificant projection matrices from SVD(W).
    Returns (P_dec, P_ins) both [feat_dim, feat_dim].
    """
    W = classifier_weight.float()
    if W.dim() > 2:
        W = W.squeeze()
    max_k = min(W.shape[0], W.shape[1])
    k = k if k is not None else max_k
    k = min(k, max_k)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    V_dec = Vh[:k].T
    V_ins = Vh[k:].T
    P_dec = V_dec @ V_dec.T
    P_ins = V_ins @ V_ins.T
    return P_dec, P_ins


def project_decisive(features, P_dec):
    """Project feature map [B, C, H, W] onto decisive subspace; P_dec [C, C]."""
    B, C, H, W = features.shape
    f = features.permute(0, 2, 3, 1).reshape(-1, C)
    f = f @ P_dec.to(features.device)
    return f.view(B, H, W, C).permute(0, 3, 1, 2)


def project_insignificant(features, P_ins):
    """Project feature map [B, C, H, W] onto insignificant subspace; P_ins [C, C]."""
    B, C, H, W = features.shape
    f = features.permute(0, 2, 3, 1).reshape(-1, C)
    f = f @ P_ins.to(features.device)
    return f.reshape(B, H, W, C).permute(0, 3, 1, 2)


def actsub_shaping_react(features, percentile=90):
    """ReAct-style clipping: clamp activations at per-sample percentile. features: [B, C, H, W]."""
    B, C, H, W = features.shape
    flat = features.reshape(B, -1)
    # quantile() fails on very large tensors; use subset like ReAct
    max_quantile_n = 2**20
    q_val = percentile / 100.0
    thresh_list = []
    for b in range(B):
        f = flat[b : b + 1].float()
        n = f.numel()
        if n <= max_quantile_n:
            t = torch.quantile(f, q_val, dim=1, keepdim=True)
        else:
            perm = torch.randperm(n, device=f.device, dtype=torch.long)[:max_quantile_n]
            t = torch.quantile(f.squeeze(0)[perm].unsqueeze(0), q_val, dim=1, keepdim=True)
        thresh_list.append(t)
    thresh = torch.cat(thresh_list, dim=0).reshape(B, 1, 1, 1)
    return torch.clamp(features, max=thresh)


# Paper-faithful ActSub defaults (do not tune N or gallery fraction)
ACTSUB_TOPK_NEIGHBORS = 20   # N in paper, fixed
ACTSUB_GALLERY_FRACTION = 0.1  # fixed 10%


def actsub_insignificant_score(test_feat_ins, gallery_ins, topk=20, eps=1e-6):
    """
    Paper Eq. (7): S→ = -log(1 - (1/N) Σ cos(ā, ā⁽ⁱ⁾)). N=20 fixed (not tuned).
    test_feat_ins: [B, D] L2-normalized, insignificant-space.
    gallery_ins: [N, D] L2-normalized, on same device or CPU.
    Returns [B] (higher = more OOD when unlike ID).
    """
    device = test_feat_ins.device
    if isinstance(gallery_ins, np.ndarray):
        gallery_ins = torch.from_numpy(gallery_ins).float().to(device)
    else:
        gallery_ins = gallery_ins.float().to(device)
    test_feat_ins = F.normalize(test_feat_ins.float(), p=2, dim=1)
    gallery_ins = F.normalize(gallery_ins, p=2, dim=1)
    sim = test_feat_ins @ gallery_ins.T
    k_actual = min(topk, sim.shape[1])
    vals, _ = torch.topk(sim, k_actual, dim=1)
    mean_sim = vals.mean(dim=1)
    return -torch.log(1.0 - mean_sim + eps)


def actsub_score(model, x, P_dec, P_ins, first_part, last_conv, clip_percentile=90, method='energy',
                 gallery_ins=None, lambda_=1.0, topk=20, target_size=None):
    """
    Full ActSub (paper): S← (decursive energy) and optionally S→ (insignificant cosine).
    Paper Eq. (10): S↔ = (S→ / λ) · S← when gallery_ins is provided (λ tuned on ID val only).
    first_part: bottleneck [B,304,H,W] -> [B,256,H,W]; last_conv: [B,256,H,W] -> logits.
    gallery_ins: [N, 256] L2-normalized insignificant-space training features (optional).
    topk: N=20 fixed in paper (do not tune).
    target_size: (H, W) for consistent resolution; if None, use input spatial size.
    """
    model.eval()
    with torch.no_grad():
        bottleneck = model.forward_features(x)
        features_256 = first_part(bottleneck)
        # Decisive: project, shape, logits, energy (S←)
        features_proj = project_decisive(features_256, P_dec)
        features_shaped = actsub_shaping_react(features_proj, percentile=clip_percentile)
        logits = last_conv(features_shaped)
        out_size = target_size if target_size is not None else x.shape[-2:]
        if logits.shape[-2:] != out_size:
            logits = F.interpolate(logits, size=out_size, mode='bilinear', align_corners=False)
        S_left = energy_score(logits)
        if S_left.dim() == 2:
            S_left = S_left.unsqueeze(0)
        # Insignificant: spatial mean -> [B, D], cosine to gallery (S→), then broadcast
        if gallery_ins is not None and P_ins is not None:
            feat_ins = project_insignificant(features_256, P_ins)
            feat_ins_pooled = feat_ins.mean(dim=(2, 3))
            S_right = actsub_insignificant_score(feat_ins_pooled, gallery_ins, topk=topk)
            S_right = S_right.reshape(-1, 1, 1).expand_as(S_left)
            # Paper Eq. (10): S↔ = (S→ / λ) · S← (λ tuned on ID validation only)
            S_combined = (S_right / (lambda_ + 1e-8)) * S_left
            if S_combined.shape[0] == 1:
                S_combined = S_combined.squeeze(0)
            if target_size is not None and _get_spatial_size(S_combined) != target_size:
                S_combined = _resize_scores_to_target(S_combined, target_size)
            return S_combined
        if method == 'energy':
            out = energy_score(logits)
        elif method == 'maxlogit':
            out = -maxlogit_score(logits)
        elif method == 'msp':
            out = 1.0 - msp_score(logits)
        else:
            out = energy_score(logits)
        if out.dim() == 3 and out.shape[0] == 1:
            out = out.squeeze(0)
        if target_size is not None and _get_spatial_size(out) != target_size:
            out = _resize_scores_to_target(out, target_size)
        return out


def _get_actsub_components(model, k=None):
    """
    One-time: P_dec, P_ins, first_part, last_conv for full ActSub.
    Cached on model._actsub_components.
    """
    if getattr(model, '_actsub_components', None) is not None:
        return model._actsub_components
    head = model.get_classifier_head()
    if not isinstance(head, torch.nn.Sequential):
        raise ValueError("ActSub expects classifier to be nn.Sequential (Conv, BN, ReLU, Conv)")
    first_part = torch.nn.Sequential(head[0], head[1], head[2])
    last_conv = head[3]
    W = last_conv.weight.data.squeeze()
    P_dec, P_ins = compute_actsub_projections(W, k=k)
    comp = {
        'P_dec': P_dec, 'P_ins': P_ins,
        'first_part': first_part, 'last_conv': last_conv,
    }
    model._actsub_components = comp
    return comp


@torch.no_grad()
def select_k_via_norm_balance(model, image_paths, transform, device, max_k=None, batch_size=8):
    """
    Paper Eq. (6): choose k so that ||a_dec|| and ||a_ins|| are balanced on ID data.
    Returns k_star (int). Uses only model weights + ID training activations.
    """
    from PIL import Image
    comp = _get_actsub_components(model, k=max_k or 999)
    first_part = comp['first_part'].to(device)
    last_conv = comp['last_conv']
    W = last_conv.weight.data.squeeze().float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    D = Vh.shape[1]
    max_k = min(max_k or Vh.shape[0], Vh.shape[0])
    V = Vh.T.to(device)
    norms_dec = torch.zeros(max_k, device=device)
    norms_ins = torch.zeros(max_k, device=device)
    count = 0
    n_paths = len(image_paths)
    for start in range(0, n_paths, batch_size):
        batch_paths = image_paths[start:start + batch_size]
        feats_list = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                x = transform(img).unsqueeze(0).to(device)
                bottleneck = model.forward_features(x)
                f = first_part(bottleneck).mean(dim=(2, 3))
                feats_list.append(f)
            except Exception:
                continue
        if not feats_list:
            continue
        feats = torch.cat(feats_list, dim=0)
        for k in range(1, max_k + 1):
            V_dec = V[:, :k]
            V_ins = V[:, k:]
            a_dec = feats @ V_dec @ V_dec.T
            a_ins = feats @ V_ins @ V_ins.T
            norms_dec[k - 1] = norms_dec[k - 1] + a_dec.norm(dim=1).sum()
            norms_ins[k - 1] = norms_ins[k - 1] + a_ins.norm(dim=1).sum()
        count += feats.shape[0]
    if count == 0:
        return min(19, max_k)
    norms_dec = norms_dec / count
    norms_ins = norms_ins / count
    diff = (norms_dec - norms_ins).abs()
    k_star = diff.argmin().item() + 1
    return k_star


@torch.no_grad()
def tune_lambda_actsub(model, val_image_paths, gallery_ins, transform, device, lambda_candidates=(0.2, 0.5, 1.0)):
    """
    Paper-faithful: tune λ on ID validation only. No OOD data.
    Returns best_lambda (float). Lower mean S on ID val = more ID-like = better.
    """
    comp = _get_actsub_components(model)
    P_dec = comp['P_dec'].to(device)
    P_ins = comp['P_ins'].to(device)
    first_part = comp['first_part'].to(device)
    last_conv = comp['last_conv'].to(device)
    if isinstance(gallery_ins, np.ndarray):
        gallery_ins = torch.from_numpy(gallery_ins).float().to(device)
    else:
        gallery_ins = gallery_ins.float().to(device)
    gallery_ins = F.normalize(gallery_ins, p=2, dim=1)
    best_lambda = None
    best_score = float('inf')
    from PIL import Image
    for lam in lambda_candidates:
        scores = []
        for path in val_image_paths:
            try:
                img = Image.open(path).convert('RGB')
                x = transform(img).unsqueeze(0).to(device)
                bottleneck = model.forward_features(x)
                features_256 = first_part(bottleneck)
                f_dec = project_decisive(features_256, P_dec)
                f_dec = actsub_shaping_react(f_dec, percentile=90)
                logits = last_conv(f_dec)
                S_left = -torch.logsumexp(logits, dim=1).mean()
                f_ins = project_insignificant(features_256, P_ins).mean(dim=(2, 3))
                S_right = actsub_insignificant_score(f_ins, gallery_ins, topk=ACTSUB_TOPK_NEIGHBORS).mean()
                S = (S_right / (lam + 1e-8)) * S_left
                scores.append(S.item())
            except Exception:
                continue
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        if avg < best_score:
            best_score = avg
            best_lambda = lam
    return best_lambda if best_lambda is not None else 1.0


def mahalanobis_score(features, gallery, alpha=0.01, n_components=None, target_size=None):
    """
    Mahalanobis Distance OOD detection.
    
    NOTE ON FEATURE CHOICE:
    - Bottleneck features (304 dim, decoder) often give weak results (~0.55 AUROC).
    - Layer4 features (2048 dim, backbone) are more discriminative (~0.63+ AUROC).
    - Use Mahalanobis++ with layer4 gallery for better performance.
    - If using bottleneck, ensure gallery was built from bottleneck features.
    
    Args:
        features: Bottleneck or layer features [B, C, H, W] or [C, H, W]
        gallery: Dictionary with keys: 'pca', 'means', 'cov'
        alpha: Shrinkage parameter for covariance regularization (e.g. 1e-2 fixed)
        n_components: If set, use first n_components of PCA (truncate). For Mahalanobis++ grid (64, 128).
        target_size: (H, W) to upsample scores for consistent resolution; None = keep feature resolution.
    
    Returns:
        scores: [B, H, W] or [H, W] - Mahalanobis distances (higher = OOD)
    """
    import numpy as np
    from sklearn.decomposition import PCA
    
    # Convert to numpy if needed
    if isinstance(features, torch.Tensor):
        if features.dim() == 3:
            features = features.unsqueeze(0)
        n, c, h, w = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy()
    else:
        if features.ndim == 3:
            features = features[np.newaxis, ...]
        n, c, h, w = features.shape
        features_flat = features.transpose(0, 2, 3, 1).reshape(-1, c)
    
    # Apply PCA
    pca = gallery['pca']
    means = gallery['means']
    shared_cov = gallery['cov']
    
    # Check feature dimension matches gallery
    expected_dim = pca.n_features_in_
    actual_dim = features_flat.shape[1]
    if actual_dim != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch! "
            f"Gallery expects {expected_dim} dims, got {actual_dim}. "
            f"Make sure you're using the same feature extraction (bottleneck vs layer4)."
        )
    
    features_pca = pca.transform(features_flat)  # [N, n_components_full]
    n_comp_full = features_pca.shape[1]
    k = n_components if n_components is not None else n_comp_full
    k = min(k, n_comp_full)
    features_pca = features_pca[:, :k]
    means_trunc = [m[:k] for m in means]
    shared_cov_k = shared_cov[:k, :k]
    
    # Regularize covariance
    shared_cov_reg = (1 - alpha) * shared_cov_k + alpha * np.eye(k, dtype=np.float32)
    inv_shared_cov = np.linalg.pinv(shared_cov_reg)
    
    all_dists = []
    for class_mean in means_trunc:
        diff = features_pca - class_mean  # [N, k]
        dist = np.sum(np.dot(diff, inv_shared_cov) * diff, axis=1)
        all_dists.append(dist)
    
    # Minimum distance across all classes
    all_dists = np.array(all_dists)  # [num_classes, N]
    min_dists = np.min(all_dists, axis=0)  # [N]
    
    # Reshape back to spatial dimensions
    scores = min_dists.reshape(n, h, w)
    
    if scores.shape[0] == 1:
        scores = scores.squeeze(0)
    
    if target_size is not None and scores.shape != target_size:
        scores = _resize_scores_to_target(scores, target_size)
    return scores


# Convenience function to get OOD scores (inverted for consistency)
def get_ood_scores(logits, method='msp', **kwargs):
    """
    Get OOD scores from logits.
    
    Args:
        logits: [B, num_classes, H, W] or [num_classes, H, W]
        method: 'msp', 'maxlogit', 'entropy', 'energy'
        **kwargs: Additional arguments for specific methods
    
    Returns:
        scores: [B, H, W] or [H, W] - Higher score = More OOD
    """
    if method == 'msp':
        scores = msp_score(logits)
        # Invert: low confidence = OOD
        return 1.0 - scores
    elif method == 'maxlogit':
        scores = maxlogit_score(logits)
        # Invert: low magnitude = OOD
        return -scores
    elif method == 'entropy':
        scores = entropy_score(logits, **kwargs)
        # High entropy = OOD (no inversion needed)
        return scores
    elif method == 'energy':
        scores = energy_score(logits, **kwargs)
        # High energy = OOD (no inversion needed)
        return scores
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# ADDITIONAL OOD METHODS (from ood_benchmark)
# ============================================================================

def odin_score(logits, model=None, input_tensor=None, temperature=1000.0, epsilon=0.0):
    """
    ODIN (Out-of-Distribution detector for Neural networks).
    Temperature scaling + optional input perturbation (epsilon > 0).
    
    Args:
        logits: [B, num_classes, H, W]
        model: Optional model for input perturbation
        input_tensor: Optional input tensor for perturbation
        temperature: Temperature for scaling (default=1000)
        epsilon: Perturbation magnitude (0.0 = temperature scaling only)
    
    Returns:
        scores: [B, H, W] - Higher = More OOD
    """
    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
    if epsilon > 0 and model is not None and input_tensor is not None:
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        model.eval()
        with torch.enable_grad():
            outputs = model(input_tensor)
            scaled = outputs / temperature
            labels = outputs.argmax(dim=1)
            loss = F.cross_entropy(scaled, labels)
            loss.backward()
        gradient = input_tensor.grad.sign()
        perturbed = input_tensor - epsilon * gradient
        with torch.no_grad():
            logits = model(perturbed)
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=1)
    max_conf, _ = torch.max(probs, dim=1)
    scores = 1.0 - max_conf
    if scores.shape[0] == 1:
        scores = scores.squeeze(0)
    return scores


def knn_score(features, gallery, k=50, target_size=None):
    """
    k-Nearest Neighbors OOD detection.
    Uses chunked gallery and feature processing to avoid OOM (e.g. 300k x 3072 gallery).
    
    Args:
        features: Bottleneck features [B, C, H, W] or [C, H, W]
        gallery: Normalized feature gallery [N_gallery, C] (numpy or torch)
        k: Number of nearest neighbors
        target_size: (H, W) to upsample scores for consistent resolution; None = keep feature resolution.
    
    Returns:
        scores: [B, H, W] or [H, W] - Higher score = More OOD
    """
    if features.dim() == 3:
        features = features.unsqueeze(0)
    
    device = features.device
    B, C, H, W = features.shape
    
    # Keep gallery on CPU; process in chunks to avoid loading 300k x 3072 on GPU
    if isinstance(gallery, torch.Tensor):
        gallery = gallery.cpu().numpy()
    gallery = np.asarray(gallery, dtype=np.float32)
    # Gallery is assumed L2-normalized from build script; re-normalize to be safe
    norms = np.linalg.norm(gallery, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    gallery = gallery / norms

    # Prepare features on GPU
    features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
    features_flat = F.normalize(features_flat, p=2, dim=1)

    N_gallery = gallery.shape[0]
    k_actual = min(k, N_gallery)
    feat_chunk_size = 5000   # pixels per feature chunk
    gallery_chunk_size = 20000  # rows per gallery chunk (~240 MB matmul with 3072 dim)

    scores_list = []
    for f_start in range(0, features_flat.shape[0], feat_chunk_size):
        f_end = min(f_start + feat_chunk_size, features_flat.shape[0])
        chunk = features_flat[f_start:f_end]
        top_k_sim = None
        for g_start in range(0, N_gallery, gallery_chunk_size):
            g_end = min(g_start + gallery_chunk_size, N_gallery)
            gallery_chunk = torch.from_numpy(gallery[g_start:g_end]).float().to(device)
            sim_part = torch.matmul(chunk, gallery_chunk.T)
            if top_k_sim is None:
                top_k_sim, _ = torch.topk(sim_part, k=min(k_actual, sim_part.shape[1]), dim=1)
            else:
                combined = torch.cat([top_k_sim, sim_part], dim=1)
                top_k_sim, _ = torch.topk(combined, k=min(k_actual, combined.shape[1]), dim=1)
            del gallery_chunk, sim_part
            if device.type == "cuda":
                torch.cuda.empty_cache()

        kth_sim = top_k_sim[:, -1]
        knn_distance = 1.0 - kth_sim
        scores_list.append(knn_distance.cpu())

    score_flat = torch.cat(scores_list, dim=0).to(device)
    scores = score_flat.reshape(B, H, W)
    
    if scores.shape[0] == 1:
        scores = scores.squeeze(0)
    
    if target_size is not None and _get_spatial_size(scores) != target_size:
        scores = _resize_scores_to_target(scores, target_size)
    return scores


def vim_score(logits, features, gallery, alpha=None, target_size=None):
    """
    VIM (Virtual Logit Matching) OOD detection.
    
    Args:
        logits: [B, num_classes, H, W]
        features: Bottleneck features [B, C, H, W]
        gallery: Dict with 'components' (PCA), 'mean', 'alpha'
        alpha: Override alpha (e.g. from tuning); else gallery['alpha']
    
    Returns:
        scores: [B, H, W] - Higher score = More OOD
    """
    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
    if features.dim() == 3:
        features = features.unsqueeze(0)
    
    device = logits.device
    B, C, H, W = features.shape
    
    # Get gallery components
    components = torch.from_numpy(gallery['components']).float().to(device)
    mean = torch.from_numpy(gallery['mean']).float().to(device)
    if alpha is None:
        alpha = gallery.get('alpha', 1.0)
    
    # Flatten features
    feat_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
    feat_centered = feat_flat - mean
    
    # Project to principal components
    proj = torch.matmul(feat_centered, components.T)
    recon = torch.matmul(proj, components)
    
    # Residual (reconstruction error)
    residual = torch.norm(feat_centered - recon, dim=1)
    
    # Energy from logits
    logits_small = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
    energy = torch.logsumexp(logits_small, dim=1).flatten()
    
    # VIM score: residual - alpha * energy
    score = residual - alpha * energy
    
    scores = score.reshape(B, H, W)
    if scores.shape[0] == 1:
        scores = scores.squeeze(0)
    
    if target_size is not None and _get_spatial_size(scores) != target_size:
        scores = _resize_scores_to_target(scores, target_size)
    return scores


def gen_score(logits, gamma=0.1, eps=1e-12):
    """
    GEN (Generalized Entropy) OOD detection.
    
    Args:
        logits: [B, num_classes, H, W]
        gamma: Power parameter (default=0.1)
        eps: Small epsilon for numerical stability
    
    Returns:
        scores: [B, H, W] - Higher score = More OOD
    """
    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
    
    probs = torch.softmax(logits, dim=1).clamp(min=eps)
    s = torch.sum(probs * torch.pow(1.0 - probs, gamma), dim=1)
    
    if s.dim() == 3 and s.shape[0] == 1:
        s = s.squeeze(0)
    
    return s


def dice_score(logits, p=90):
    """
    DICE-like scoring (logit-based): sparsify along channel, then energy.
    
    Args:
        logits: [B, num_classes, H, W]
        p: Percentile threshold (default=90)
    
    Returns:
        scores: [B, H, W] - Higher = More OOD
    """
    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
    B, C, H, W = logits.shape
    k = max(1, int(C * (p / 100.0)))
    topk_vals, _ = torch.topk(logits, k, dim=1)
    threshold = topk_vals[:, -1:, :, :]
    mask = logits >= threshold
    masked_logits = logits * mask.float()
    energy = -torch.logsumexp(masked_logits, dim=1)
    if energy.shape[0] == 1:
        energy = energy.squeeze(0)
    return energy


def gram_score(features, gallery=None, target_size=None):
    """
    Gram Matrix OOD detection.
    
    Computes deviation of feature Gram matrix from ID distribution.
    
    IMPORTANT: Without a gallery (mean_gram from ID data), this method is weak.
    The fallback uses per-pixel feature norm variance, which is a rough proxy.
    For proper Gram OOD, build a gallery with build_gram_gallery.py.
    
    Args:
        features: Bottleneck features [B, C, H, W]
        gallery: Dict with 'mean_gram' (mean Gram matrix from ID data)
        target_size: (H, W) to upsample scores for consistent resolution; None = keep feature resolution.
    
    Returns:
        scores: [B, H, W] - Higher score = More OOD
    """
    if features.dim() == 3:
        features = features.unsqueeze(0)
    
    B, C, H, W = features.shape
    device = features.device
    
    # Flatten spatial dimensions
    f_flat = features.view(B, C, -1)
    f_norm = F.normalize(f_flat, p=2, dim=1)
    
    # Compute Gram matrix: G = F @ F^T / (H*W)
    gram = torch.bmm(f_norm, f_norm.transpose(1, 2)) / (H * W)  # (B, C, C)
    
    if gallery is not None and 'mean_gram' in gallery:
        # Compute deviation from mean Gram
        mean_gram = torch.from_numpy(gallery['mean_gram']).float().to(device)
        diff = gram - mean_gram.unsqueeze(0)
        deviation = torch.norm(diff.view(B, -1), p=2, dim=1)  # (B,)
        # Broadcast to spatial dimensions
        scores = deviation.view(B, 1, 1).expand(B, H, W)
    else:
        # Fallback (WEAK without gallery): use feature magnitude variance as proxy.
        # OOD features often have unusual activation patterns (high variance across channels).
        # Per-pixel: std across channels. Higher std = more unusual = more OOD.
        scores = torch.std(features, dim=1)  # [B, H, W]
    
    if scores.shape[0] == 1:
        scores = scores.squeeze(0)
    
    if target_size is not None and _get_spatial_size(scores) != target_size:
        scores = _resize_scores_to_target(scores, target_size)
    return scores


def scale_score(logits, features, T=1.0, beta=0.5):
    """
    SCALE: Normalize logits by feature norm, then energy.
    
    scaled_logits = logits / (norm ** beta), score = energy(scaled_logits / T).
    Hyperparameters: T (temperature), beta (scaling exponent). Tune on validation.
    
    Args:
        logits: [B, num_classes, H, W]
        features: [B, C, H, W] bottleneck features
        T: Temperature (default 1.0)
        beta: Scaling exponent (default 0.5)
    
    Returns:
        scores: [B, H, W] - Higher score = More OOD
    """
    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
    if features.dim() == 3:
        features = features.unsqueeze(0)
    # Same spatial size: prefer downsampling logits to feature size when features are smaller (low-mem)
    if features.shape[-2:] != logits.shape[-2:]:
        feat_h, feat_w = features.shape[-2:]
        log_h, log_w = logits.shape[-2:]
        if feat_h * feat_w <= log_h * log_w:
            logits = F.interpolate(logits, size=(feat_h, feat_w), mode='bilinear', align_corners=False)
        else:
            features = F.interpolate(features, size=(log_h, log_w), mode='bilinear', align_corners=False)
    norm = features.norm(dim=1, keepdim=True).clamp(min=1e-6)
    scaled_logits = logits / (norm ** beta)
    return energy_score(scaled_logits, T=T)


def kl_matching_score(logits, gallery=None, T=1.0, eps=1e-8):
    """
    KL Matching: Compare predictive distribution to reference ID mean softmax.
    
    p = softmax(logits / T), q = reference (from gallery or uniform).
    
    POLARITY FIX: When q = uniform, KL(p||q) = log(C) - entropy(p).
    High KL → confident (sharp) → ID. Low KL → uncertain → OOD.
    So we return **negative KL** when using uniform reference (no gallery),
    so that higher score = more OOD.
    
    With a proper gallery (mean ID softmax), high KL(p||q_id) means p differs
    from typical ID predictions, so high KL → OOD (no negation needed).
    
    Hyperparameters: T. Tune on validation.
    
    Args:
        logits: [B, num_classes, H, W]
        gallery: Dict with 'ref_probs' [num_classes] (mean softmax over ID). If None, use uniform q.
        T: Temperature (default 1.0)
        eps: Small epsilon for log stability
    
    Returns:
        scores: [B, H, W] - Higher score = More OOD
    """
    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
    B, C, H, W = logits.shape
    p = F.softmax(logits / T, dim=1)
    
    use_uniform = gallery is None or 'ref_probs' not in gallery
    if not use_uniform:
        q = torch.from_numpy(np.asarray(gallery['ref_probs'], dtype=np.float32)).to(logits.device)
        q = q.view(1, C, 1, 1).expand(B, C, H, W)
    else:
        q = torch.ones_like(p) / C
    
    kl = torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)), dim=1)
    
    # POLARITY FIX: negate when using uniform (high KL = confident = ID)
    if use_uniform:
        kl = -kl
    
    if kl.shape[0] == 1:
        kl = kl.squeeze(0)
    return kl


def gem_score(features, model=None, input_tensor=None, pseudo_label=None):
    """
    GEM (Gradient Embedding Matching): OOD samples induce unusual gradients.
    
    score = ||grad||. Hyperparameters: gradient layer, distance metric.
    Expensive — subsample pixels. Tune on validation. TODO: full implementation.
    """
    raise NotImplementedError(
        "GEM not yet implemented. Idea: loss = CE(logits, pseudo_label); "
        "grad = autograd(loss, features); score = ||grad||. Tune layer + norm on validation."
    )


def rankfeat_score(features, gallery=None, k=None):
    """
    RankFeat: OOD disrupts relative feature ranking.
    
    score = rank_correlation(f, mean_id_feature). Hyperparameters: top-k.
    Tune k on validation. TODO: full implementation.
    """
    raise NotImplementedError(
        "RankFeat not yet implemented. Idea: rank_id = argsort(mean_id_feature); "
        "rank_x = argsort(f); score = spearman(rank_x, rank_id). Tune k on validation."
    )


def mc_dropout_score(model, input_tensor, n_iter=5, target_size=None):
    """
    Monte Carlo Dropout OOD detection.
    
    Runs model multiple times with dropout enabled and computes variance.
    
    Args:
        model: Model with dropout layers
        input_tensor: Input image [B, 3, H, W]
        n_iter: Number of stochastic forward passes
        target_size: (H, W) to upsample scores for consistent resolution; None = keep logit resolution.
    
    Returns:
        scores: [B, H, W] or [H, W] - Higher variance = More OOD
    """
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_iter):
            out = model(input_tensor)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            preds.append(entropy.unsqueeze(0))
    model.eval()
    if len(preds) == 0:
        return torch.zeros_like(input_tensor[:, 0])
    stacked = torch.cat(preds, dim=0)
    var_map = torch.var(stacked, dim=0)
    if var_map.shape[0] == 1:
        var_map = var_map.squeeze(0)
    if target_size is not None and _get_spatial_size(var_map) != target_size:
        var_map = _resize_scores_to_target(var_map, target_size)
    return var_map


# ============================================================================
# UNIFIED METHOD REGISTRY
# ============================================================================

AVAILABLE_METHODS = [
    'MSP', 'MaxLogit', 'Entropy', 'Energy',
    'Mahalanobis', 'Mahalanobis++', 'ReAct', 'ASH', 'ACTSUB',
    'ODIN', 'kNN', 'VIM', 'GEN', 'DICE', 'Gram', 'MC_Dropout',
    'SCALE', 'KL_Matching', 'GEM', 'RankFeat'  # SCALE/KL implemented; GEM/RankFeat stubs
]


def get_ood_score(logits, method='msp', features=None, model=None, input_tensor=None,
                  gallery=None, target_size=None, **kwargs):
    """
    Unified interface to get OOD scores from any method.
    All methods can output at consistent target_size (logit resolution) for fair comparison.
    
    Args:
        logits: [B, num_classes, H, W] or [num_classes, H, W]
        method: Method name (case-insensitive)
        features: Optional bottleneck features [B, C, H, W]
        model: Optional model (for ODIN, MC_Dropout, ReAct, ASH, ActSub)
        input_tensor: Optional input tensor (for ODIN, MC_Dropout, ReAct, ASH, ActSub)
        gallery: Optional gallery dict (for Mahalanobis, kNN, VIM, Gram, ActSub)
        target_size: (H, W) output resolution; default = logit spatial size. Use for consistent resolution.
        **kwargs: Method-specific parameters
    
    Returns:
        scores: [B, H, W] or [H, W] at target_size - Higher score = More OOD
    """
    method = method.lower().replace("++", "_plus_plus")
    # Default target size = logit spatial size
    if target_size is None:
        if logits.dim() == 3:
            target_size = (logits.shape[1], logits.shape[2])
        else:
            logits_4d = logits.unsqueeze(0) if logits.dim() == 3 else logits
            target_size = (logits_4d.shape[2], logits_4d.shape[3])
    
    def _ensure_target_size(scores):
        """Resize scores to target_size when caller passed a fixed target (e.g. from first image)."""
        if target_size is None or scores is None:
            return scores
        if hasattr(scores, 'dim'):
            current = _get_spatial_size(scores)
        elif hasattr(scores, 'shape'):
            s = scores.shape
            current = s if len(s) == 2 else (s[1], s[2]) if len(s) == 3 else (s[2], s[3])
        else:
            return scores
        if current != target_size:
            return _resize_scores_to_target(scores, target_size)
        return scores

    # Logit-based methods (no features needed) — output at logit resolution, then resize to target_size
    if method == 'msp':
        return _ensure_target_size(get_ood_scores(logits, method='msp'))
    elif method == 'maxlogit':
        return _ensure_target_size(get_ood_scores(logits, method='maxlogit'))
    elif method == 'entropy':
        return _ensure_target_size(get_ood_scores(logits, method='entropy', **kwargs))
    elif method == 'energy':
        return _ensure_target_size(get_ood_scores(logits, method='energy', **kwargs))
    elif method == 'odin':
        return _ensure_target_size(odin_score(logits, model=model, input_tensor=input_tensor, **kwargs))
    elif method == 'gen':
        return _ensure_target_size(gen_score(logits, **kwargs))
    elif method == 'dice':
        return _ensure_target_size(dice_score(logits, **kwargs))
    elif method == 'kl_matching':
        gallery_kl = gallery if gallery else None
        return _ensure_target_size(kl_matching_score(logits, gallery=gallery_kl, **kwargs))
    
    # Feature-based methods (require features)
    if method == 'scale':
        if features is None:
            raise ValueError("SCALE requires features")
        return _ensure_target_size(scale_score(logits, features, **kwargs))
    
    if features is None:
        raise ValueError(f"Method '{method}' requires features but none provided")
    
    if method == 'mahalanobis':
        if gallery is None:
            raise ValueError("Mahalanobis requires gallery")
        return mahalanobis_score(features, gallery, target_size=target_size, **kwargs)
    if method == 'mahalanobis_plus_plus':
        if gallery is None:
            raise ValueError("Mahalanobis++ requires gallery for chosen layer (mahalanobis_layer3 / mahalanobis_layer4)")
        kwargs_pp = {k: v for k, v in kwargs.items() if k != 'layer'}
        return mahalanobis_score(features, gallery, alpha=0.01, target_size=target_size, **kwargs_pp)
    elif method == 'knn':
        if gallery is None or 'gallery_knn' not in gallery:
            raise ValueError("kNN requires gallery with 'gallery_knn' key. Build kNN gallery first.")
        return knn_score(features, gallery['gallery_knn'], target_size=target_size, **kwargs)
    elif method == 'vim':
        vim_g = (gallery or {}).get('gallery_vim', gallery)
        if vim_g is None or 'components' not in vim_g:
            raise ValueError("VIM requires gallery (components, mean, alpha). Build VIM gallery first.")
        return vim_score(logits, features, vim_g, target_size=target_size, **kwargs)
    elif method == 'gram':
        return gram_score(features, gallery, target_size=target_size)
    elif method == 'gem':
        return gem_score(features, model=model, input_tensor=input_tensor, **kwargs)
    elif method == 'rankfeat':
        return rankfeat_score(features, gallery=gallery, **kwargs)
    
    # Model-based methods (require model wrapper)
    if method == 'react':
        if model is None:
            raise ValueError("ReAct requires model")
        clip_threshold = kwargs.get('clip_threshold', None)
        if clip_threshold is None and features is not None:
            flat = features.view(-1).float()
            n = flat.numel()
            # quantile() fails on very large tensors; sample or use kthvalue
            max_quantile_n = 2**20
            if n <= max_quantile_n:
                clip_threshold = torch.quantile(flat, 0.9).item()
            else:
                perm = torch.randperm(n, device=flat.device, dtype=torch.long)[:max_quantile_n]
                clip_threshold = torch.quantile(flat[perm], 0.9).item()
        elif clip_threshold is None:
            clip_threshold = 1.0
        return react_score(model, input_tensor, clip_threshold, target_size=target_size, **kwargs)
    elif method == 'actsub':
        if model is None or input_tensor is None:
            raise ValueError("ActSub requires model and input_tensor")
        comp = _get_actsub_components(model, k=kwargs.get('k'))
        gallery_ins = None
        if gallery is not None and gallery.get('actsub_ins') is not None:
            gallery_ins = gallery['actsub_ins']
        return actsub_score(
            model, input_tensor,
            comp['P_dec'], comp['P_ins'], comp['first_part'], comp['last_conv'],
            clip_percentile=kwargs.get('clip_percentile', 90),
            method=kwargs.get('method', 'energy'),
            gallery_ins=gallery_ins,
            lambda_=kwargs.get('lambda_', kwargs.get('lambda', 1.0)),
            topk=kwargs.get('topk', ACTSUB_TOPK_NEIGHBORS),
            target_size=target_size,
        )
    elif method == 'ash':
        if model is None:
            raise ValueError("ASH requires model")
        kwargs_ash = dict(kwargs)
        if 'lambda' in kwargs_ash and 'lambda_' not in kwargs_ash:
            kwargs_ash['lambda_'] = kwargs_ash.pop('lambda')
        return ash_score(model, input_tensor, target_size=target_size, **kwargs_ash)
    elif method == 'mc_dropout':
        if model is None or input_tensor is None:
            raise ValueError("MC_Dropout requires model and input_tensor")
        return mc_dropout_score(model, input_tensor, target_size=target_size, **kwargs)
    
    raise ValueError(f"Unknown method: {method}. Available: {AVAILABLE_METHODS}")
