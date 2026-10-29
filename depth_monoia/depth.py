"""Frozen official DA-V2-Small on the exact already-augmented detector image."""
import sys
import torch
from torch import nn
import torch.nn.functional as F
from .config import ROOT
from .provenance import verify_checkpoint


class FrozenDepthPrior(nn.Module):
    def __init__(self, checkpoint=None, input_size=518):
        super().__init__()
        sys.path.insert(0, str(ROOT / "third_party/Depth-Anything-V2"))
        from depth_anything_v2.dpt import DepthAnythingV2
        from depth_anything_v2.util.transform import Resize
        checkpoint = checkpoint or ROOT / "checkpoints/depth_anything_v2_vits.pth"
        self.provenance = verify_checkpoint("dav2_small", checkpoint)
        self.network = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        self.network.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.network.requires_grad_(False)
        self.network.eval()
        self.resize = Resize(input_size, input_size, resize_target=False,
                             keep_aspect_ratio=True, ensure_multiple_of=14,
                             resize_method="lower_bound")

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images):
        # Frozen inference is microbatched to avoid quadratic ViT attention
        # memory at KITTI aspect ratio. Detector batch size remains unchanged.
        chunks = [self._extract(image[None]) for image in images]
        return [torch.cat([chunk[level] for chunk in chunks], dim=0) for level in range(4)]

    def _extract(self, images):
        # MonoIA and DA-V2 use the same RGB ImageNet mean/std. Images here already
        # include all focal warps, flips, crops, mixup, padding and photometric edits.
        # Tensor bicubic is deliberate (recorded), not byte-identical to OpenCV.
        width, height = self.resize.get_size(images.shape[-1], images.shape[-2])
        x = F.interpolate(images.float(), (height, width), mode="bicubic", align_corners=False)
        features = self.network.pretrained.get_intermediate_layers(
            x, self.network.intermediate_layer_idx["vits"], return_class_token=True)
        depth = self.network.depth_head(features, height // 14, width // 14).relu()
        mean = depth.mean(dim=(-2, -1), keepdim=True)
        scale = depth.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
        relative = (depth - mean) / scale
        relative = F.interpolate(relative, (height // 14, width // 14), mode="bilinear", align_corners=False)
        priors = []
        for tokens, _ in features:
            tokens = F.layer_norm(tokens, (tokens.shape[-1],))
            grid = tokens.transpose(1, 2).reshape(x.shape[0], 384, height // 14, width // 14)
            priors.append(torch.cat((grid, relative), dim=1))
        if any(not torch.isfinite(p).all() for p in priors):
            raise FloatingPointError("Nonfinite depth prior")
        return priors
