"""
OOD Detection Model Wrapper for DeepLabV3+.
Extracts bottleneck features (penultimate layer) for OOD detection methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OODDeepLab(nn.Module):
    """
    Wrapper for DeepLabV3+ that can extract bottleneck features.
    
    The bottleneck is the output of the decoder/ASPP before the final classification head.
    For DeepLabV3+, this is the concatenated features [low_level (48) + aspp_output (256)] = 304 channels.
    """
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x, return_features=False, return_layer=None):
        """
        Forward pass with optional feature extraction.
        
        Args:
            x: Input tensor [B, 3, H, W]
            return_features: If True, return (logits, bottleneck_features)
            return_layer: If 'layer3' or 'layer4', return (logits, that layer features) for Mahalanobis++
        
        Returns:
            If return_features=False and return_layer is None: logits
            If return_features=True: (logits, bottleneck_features)
            If return_layer in ('layer3','layer4'): (logits, backbone features for that layer)
        """
        input_shape = x.shape[-2:]
        
        # 1. Backbone/Encoder (layer3='mid', layer4='out', layer1='low_level')
        features = self.base_model.backbone(x)
        
        # 2. Decode/ASPP (The "Penultimate" Layer)
        # For DeepLabV3+, the decode_head processes features and returns logits
        # We need to intercept the bottleneck features
        
        # Get low-level and high-level features
        low_level_feature = self.base_model.classifier.project(features['low_level'])
        output_feature = self.base_model.classifier.aspp(features['out'])
        
        # Interpolate ASPP output to match low-level feature size (contiguous reduces fragmentation)
        output_feature = F.interpolate(
            output_feature,
            size=low_level_feature.shape[2:],
            mode='bilinear',
            align_corners=False
        ).contiguous()
        
        # Bottleneck features: concatenated before final classifier
        bottleneck_features = torch.cat([low_level_feature, output_feature], dim=1)  # [B, 304, H', W']
        
        # 3. Final Classification (Logits)
        logits = self.base_model.classifier.classifier(bottleneck_features)  # [B, num_classes, H', W']
        
        # Interpolate logits to original input size
        logits = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=False).contiguous()
        
        if return_layer in ('layer3', 'layer4'):
            key = 'mid' if return_layer == 'layer3' else 'out'
            layer_feat = features[key]
            return logits, layer_feat
        if return_features:
            return logits, bottleneck_features
        return logits
    
    def forward_features(self, x):
        """
        Extract only bottleneck features without computing logits.
        More efficient when you only need features.
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            bottleneck_features: [B, 304, H, W]
        """
        input_shape = x.shape[-2:]
        
        # Backbone
        features = self.base_model.backbone(x)
        
        # Decode
        low_level_feature = self.base_model.classifier.project(features['low_level'])
        output_feature = self.base_model.classifier.aspp(features['out'])
        output_feature = F.interpolate(
            output_feature,
            size=low_level_feature.shape[2:],
            mode='bilinear',
            align_corners=False
        ).contiguous()
        
        # Bottleneck
        bottleneck_features = torch.cat([low_level_feature, output_feature], dim=1)
        
        # Don't interpolate to full size - keep at feature map resolution to save memory
        # Features are at reduced resolution (typically 1/8 or 1/16 of input)
        # This is sufficient for OOD detection and saves significant memory
        
        return bottleneck_features

    def get_classifier_head(self):
        """
        Return the classifier module that maps bottleneck features to logits.
        Used by ReAct, ASH, ACTSUB (decisive-only).
        """
        return self.base_model.classifier.classifier


def wrap_model_for_ood(base_model):
    """
    Convenience function to wrap a DeepLabV3+ model for OOD detection.
    
    Args:
        base_model: DeepLabV3+ model from modeling.py
    
    Returns:
        OODDeepLab wrapper
    """
    return OODDeepLab(base_model)
