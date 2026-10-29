"""Multi-scale depth adaptation and residual fusion."""
import torch
from torch import nn
import torch.nn.functional as F


class FusionLevel(nn.Module):
    def __init__(self, channels=256, prior_channels=385, mode="intrinsic", alpha_init=0.0):
        super().__init__()
        if mode not in ("simple", "gated", "intrinsic", "confidence"):
            raise ValueError(mode)
        self.mode = mode
        self.adapter = nn.Sequential(nn.Conv2d(prior_channels, channels, 1),
                                     nn.GroupNorm(32, channels), nn.GELU())
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.gate = None if mode == "simple" else nn.Conv2d(channels * 2, channels, 1)
        self.intrinsic = nn.Linear(channels, channels) if mode in ("intrinsic", "confidence") else None
        # Zero alpha preserves baseline exactly at initialization. Adapters/gates
        # receive nonzero gradients after alpha's first optimization update.

    def forward(self, rgb, prior, embedding, confidence=None):
        prior = F.interpolate(prior, rgb.shape[-2:], mode="bilinear", align_corners=False)
        depth = self.adapter(prior)
        if self.gate is None:
            gate = 1.0
        else:
            logits = self.gate(torch.cat((rgb, depth), dim=1))
            if self.intrinsic is not None:
                logits = logits + self.intrinsic(embedding)[:, :, None, None]
            gate = logits.sigmoid()
        if self.mode == "confidence":
            if confidence is None:
                raise ValueError("A4 requires an explicit prior reliability map")
            gate = gate * F.interpolate(confidence, rgb.shape[-2:], mode="bilinear", align_corners=False)
        return rgb + self.alpha * gate * depth


class MultiScaleFusion(nn.Module):
    def __init__(self, mode, channels=256, levels=4, alpha_init=0.0):
        super().__init__()
        self.levels = nn.ModuleList([FusionLevel(channels, mode=mode, alpha_init=alpha_init)
                                     for _ in range(levels)])

    def forward(self, rgb, priors, embedding, confidence=None):
        if len(rgb) != len(self.levels) or len(priors) != len(rgb):
            raise ValueError("Upstream feature-level interface changed")
        return [layer(x, d, embedding, confidence)
                for layer, x, d in zip(self.levels, rgb, priors)]
