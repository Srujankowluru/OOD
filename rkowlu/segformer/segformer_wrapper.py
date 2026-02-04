"""
OOD Detection Wrapper for SegFormer (Hugging Face).
Exposes logits and features compatible with the DeepLab OOD pipeline.
- Bottleneck = decoder input (3072 dims) so ReAct/ASH/ACTSUB/SCALE work; same H,W as logits.
- return_layer = encoder stage (320/512 dims) for Mahalanobis++.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _load_hf_model(model_name="nvidia/segformer-b5-finetuned-cityscapes-1024-1024"):
    from transformers import AutoModelForSemanticSegmentation
    return AutoModelForSemanticSegmentation.from_pretrained(model_name)


def _get_decoder_input(decode_head, hidden_states):
    """
    Replicate SegformerDecodeHead logic up to (before) linear_fuse.
    Returns [B, 3072, H/4, W/4] = concat of 4 scales (768 each) at same resolution.
    Encoder hidden_states are 3D [B, N, C]; N = H*W (sequence length), C = channel.
    """
    batch_size = hidden_states[-1].shape[0]
    # Reference spatial size: encoder outputs are 3D [B, N, C], so use N = H*W
    h0 = hidden_states[0]
    if h0.ndim == 3:
        n0 = h0.shape[1]
        ref_h = ref_w = int(math.sqrt(n0))
        ref_size = (ref_h, ref_w)
    else:
        ref_size = h0.shape[2:]
    all_hidden = []
    for i, (enc_h, mlp) in enumerate(zip(hidden_states, decode_head.linear_c)):
        if enc_h.ndim == 3:
            # Sequence length N = H*W, not channel C
            seq_len = enc_h.shape[1]
            hw = int(math.sqrt(seq_len))
            enc_h = enc_h.reshape(batch_size, hw, hw, -1).permute(0, 3, 1, 2).contiguous()
        height, width = enc_h.shape[2], enc_h.shape[3]
        # HF SegformerMLP expects 4D [B, C, H, W]; it does flatten(2).transpose(1,2) then Linear
        enc_proj = mlp(enc_h)  # -> [B, H*W, decoder_hidden_size]
        enc_proj = enc_proj.permute(0, 2, 1).reshape(batch_size, -1, height, width)
        enc_proj = F.interpolate(enc_proj, size=ref_size, mode="bilinear", align_corners=False)
        all_hidden.append(enc_proj)
    # HF uses [::-1] so last stage first
    fused = torch.cat(all_hidden[::-1], dim=1)  # [B, 768*4, H/4, W/4]
    return fused


class OODSegFormer(nn.Module):
    """
    - forward(x, return_features=True) -> (logits, bottleneck [B, 3072, H, W]) for ReAct/ASH/ACTSUB/SCALE.
    - forward(x, return_layer='layer3'/'layer4') -> (logits, encoder stage [B, 320/512, h, w]) for Mahalanobis++.
    """

    NUM_CLASSES = 19
    BOTTLENECK_DIM = 3072   # decoder input (768*4)
    LAYER3_DIM = 320
    LAYER4_DIM = 512

    def __init__(self, model_name="nvidia/segformer-b5-finetuned-cityscapes-1024-1024", device=None):
        super().__init__()
        self.model_name = model_name
        self.hf_model = _load_hf_model(model_name)
        self._device = device

    def forward(self, x, return_features=False, return_layer=None, upsample_bottleneck=True):
        input_shape = x.shape[-2:]
        if next(self.parameters()).dtype == torch.float16:
            x = x.half()
        outputs = self.hf_model.segformer(
            x,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        logits = self.hf_model.decode_head(hidden_states)
        logits = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)
        logits = logits.float()

        if return_layer == "layer3":
            feat = hidden_states[-2].float()  # [B, N, C]
            if feat.ndim == 3:
                b, n, c = feat.shape
                hw = int(math.sqrt(n))
                feat = feat.reshape(b, hw, hw, c).permute(0, 3, 1, 2).contiguous()
            return logits, feat
        if return_layer == "layer4":
            feat = hidden_states[-1].float()  # [B, N, C]
            if feat.ndim == 3:
                b, n, c = feat.shape
                hw = int(math.sqrt(n))
                feat = feat.reshape(b, hw, hw, c).permute(0, 3, 1, 2).contiguous()
            return logits, feat
        if return_features:
            # Decoder input (3072 ch); keep same dtype as model so ReAct/ASH classifier gets matching type (e.g. half with --fp16)
            decoder_in = _get_decoder_input(self.hf_model.decode_head, hidden_states)
            do_upsample = upsample_bottleneck and not getattr(self, "_no_upsample_bottleneck", False)
            if do_upsample:
                bottleneck = F.interpolate(decoder_in, size=input_shape, mode="bilinear", align_corners=False)
            else:
                bottleneck = decoder_in  # [B, 3072, H/4, W/4] saves ~1.7 GiB per image
            return logits, bottleneck
        return logits

    def forward_features(self, x):
        """Bottleneck = decoder input; same dtype as model (float16 when --fp16) so classifier head matches."""
        if next(self.parameters()).dtype == torch.float16:
            x = x.half()
        outputs = self.hf_model.segformer(x, output_hidden_states=True, return_dict=True)
        decoder_in = _get_decoder_input(self.hf_model.decode_head, outputs.hidden_states)
        if getattr(self, "_no_upsample_bottleneck", False):
            return decoder_in  # [B, 3072, H/4, W/4] saves VRAM; scores resized to target_size in ood_methods
        input_shape = x.shape[-2:]
        return F.interpolate(decoder_in, size=input_shape, mode="bilinear", align_corners=False)

    def get_classifier_head(self):
        """
        Return nn.Sequential(linear_fuse, batch_norm, activation, classifier) so ActSub
        can use head[3].weight (Conv2d 19 x decoder_hidden_size) for P_dec/P_ins.
        """
        d = self.hf_model.decode_head
        return nn.Sequential(d.linear_fuse, d.batch_norm, d.activation, d.classifier)


def wrap_segformer_for_ood(model_name="nvidia/segformer-b5-finetuned-cityscapes-1024-1024", device=None):
    """Load and wrap SegFormer for OOD detection (same interface as wrap_model_for_ood for DeepLab)."""
    model = OODSegFormer(model_name=model_name, device=device)
    return model
