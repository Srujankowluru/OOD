"""
Comprehensive verification script for all OOD detection methods.
Tests MSP, MaxLogit, Entropy, Energy, Mahalanobis, ReAct, and ASH.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import network.modeling as modeling
from network.ood_wrapper import wrap_model_for_ood
from ood_methods import (
    msp_score, maxlogit_score, entropy_score, energy_score,
    react_score, ash_score, mahalanobis_score, get_ood_scores
)
import pickle
import os

# Configuration
MODEL_PATH = "/visinf/projects_students/groupL/rkowlu/ood_benchmark/models/deeplabv3plus_r101/deeplab_r101.pth"
NUM_CLASSES = 19
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test image (you can replace this with an actual image path)
TEST_IMAGE_PATH = None  # Set to an actual image path for testing

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

def test_method_shapes(model, x):
    """Test that all methods return correct shapes."""
    print("\n=== Testing Method Shapes ===")
    
    # Get logits and features
    logits, features = model(x, return_features=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    
    # Test each method
    methods = {
        'MSP': lambda: msp_score(logits),
        'MaxLogit': lambda: maxlogit_score(logits),
        'Entropy': lambda: entropy_score(logits),
        'Energy': lambda: energy_score(logits),
    }
    
    for name, method_fn in methods.items():
        try:
            scores = method_fn()
            print(f"  {name:12s}: {scores.shape} ✓")
        except Exception as e:
            print(f"  {name:12s}: ERROR - {e} ✗")
    
    # Test ReAct (requires clip threshold)
    try:
        # Estimate clip threshold (90th percentile of features)
        # Sample a subset to avoid "tensor too large" error
        features_flat = features.view(-1)
        if len(features_flat) > 1000000:  # If tensor is very large, sample
            sample_size = min(1000000, len(features_flat))
            indices = torch.randperm(len(features_flat))[:sample_size]
            features_sample = features_flat[indices]
        else:
            features_sample = features_flat
        clip_threshold = torch.quantile(features_sample, 0.9).item()
        react_scores = react_score(model, x, clip_threshold, method='maxlogit')
        print(f"  {'ReAct':12s}: {react_scores.shape} ✓")
    except Exception as e:
        print(f"  {'ReAct':12s}: ERROR - {e} ✗")
    
    # Test ASH
    try:
        ash_scores = ash_score(model, x, percentile=90, method='maxlogit')
        print(f"  {'ASH':12s}: {ash_scores.shape} ✓")
    except Exception as e:
        print(f"  {'ASH':12s}: ERROR - {e} ✗")
    
    # Test Mahalanobis (if gallery exists)
    gallery_path = "mahalanobis_gallery_bottleneck.pkl"
    if os.path.exists(gallery_path):
        try:
            with open(gallery_path, 'rb') as f:
                gallery = pickle.load(f)
            mahal_scores = mahalanobis_score(features, gallery, alpha=0.0)
            print(f"  {'Mahalanobis':12s}: {mahal_scores.shape} ✓")
        except Exception as e:
            print(f"  {'Mahalanobis':12s}: ERROR - {e} ✗")
    else:
        print(f"  {'Mahalanobis':12s}: SKIPPED (gallery not found)")

def test_method_values(model, x):
    """Test that method values are reasonable."""
    print("\n=== Testing Method Values ===")
    
    logits, features = model(x, return_features=True)
    
    # Test MSP: should be in [0, 1]
    msp = msp_score(logits)
    print(f"MSP range: [{msp.min().item():.4f}, {msp.max().item():.4f}] (expected [0, 1])")
    assert 0 <= msp.min() and msp.max() <= 1, "MSP values out of range!"
    
    # Test Entropy: should be >= 0
    entropy = entropy_score(logits)
    print(f"Entropy range: [{entropy.min().item():.4f}, {entropy.max().item():.4f}] (expected >= 0)")
    assert entropy.min() >= 0, "Entropy values negative!"
    
    # Test MaxLogit: can be any value
    maxlogit = maxlogit_score(logits)
    print(f"MaxLogit range: [{maxlogit.min().item():.4f}, {maxlogit.max().item():.4f}]")
    
    # Test Energy: should be negative (energy is -logsumexp)
    energy = energy_score(logits)
    print(f"Energy range: [{energy.min().item():.4f}, {energy.max().item():.4f}] (expected negative)")
    
    print("All value checks passed! ✓")

def test_softmax_application():
    """Verify that MSP uses softmax but MaxLogit doesn't."""
    print("\n=== Testing Softmax Application ===")
    
    # Create dummy logits
    logits = torch.randn(1, 19, 64, 64)
    
    # MSP should use softmax
    msp = msp_score(logits)
    probs = torch.softmax(logits, dim=1)
    max_probs = torch.max(probs, dim=1)[0]
    assert torch.allclose(msp, max_probs), "MSP should use softmax!"
    print("MSP correctly uses softmax ✓")
    
    # MaxLogit should NOT use softmax
    maxlogit = maxlogit_score(logits)
    max_raw = torch.max(logits, dim=1)[0]
    assert torch.allclose(maxlogit, max_raw), "MaxLogit should NOT use softmax!"
    print("MaxLogit correctly does NOT use softmax ✓")

def test_entropy_epsilon():
    """Verify that entropy uses epsilon to prevent log(0)."""
    print("\n=== Testing Entropy Epsilon ===")
    
    # Create logits that would produce very small probabilities
    logits = torch.randn(1, 19, 64, 64) * 10  # Large logits -> very small probs after softmax
    
    entropy = entropy_score(logits, eps=1e-8)
    
    # Check for NaN or Inf
    assert not torch.isnan(entropy).any(), "Entropy contains NaN!"
    assert not torch.isinf(entropy).any(), "Entropy contains Inf!"
    print("Entropy correctly handles edge cases with epsilon ✓")

def main():
    """Run all verification tests."""
    print("=" * 60)
    print("OOD Detection Methods Verification")
    print("=" * 60)
    
    # Load model
    model = load_model()
    
    # Create dummy input for testing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Use a dummy image or load from file
    if TEST_IMAGE_PATH and os.path.exists(TEST_IMAGE_PATH):
        image = Image.open(TEST_IMAGE_PATH).convert('RGB')
        x = transform(image).unsqueeze(0).to(DEVICE)
    else:
        # Create dummy image
        x = torch.randn(1, 3, 512, 1024).to(DEVICE)
        print("Using dummy input image for testing")
    
    # Run tests
    test_softmax_application()
    test_entropy_epsilon()
    test_method_shapes(model, x)
    test_method_values(model, x)
    
    print("\n" + "=" * 60)
    print("All verification tests completed!")
    print("=" * 60)

if __name__ == '__main__':
    main()
