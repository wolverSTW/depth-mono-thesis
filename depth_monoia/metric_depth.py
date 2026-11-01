"""E4a frozen official DA-V2 metric VKITTI Small; never re-standardize metre depth."""
import importlib
import sys
import types
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .config import ROOT
from .provenance import verify_checkpoint

METRIC_REPRESENTATION = "single_metric_vkitti_depth_in_meters_only"
METRIC_CHECKPOINT_KEY = "dav2_metric_vkitti_small"
METRIC_CHECKPOINT_FILENAME = "depth_anything_v2_metric_vkitti_vits.pth"
METRIC_MAX_DEPTH_M = 80.0
_METRIC_PACKAGE = "depth_monoia_pinned_dav2_metric"


def _metric_model_class():
    """Use a private package namespace; never confuse relative/metric DA-V2 dpt.py."""
    root = ROOT / "third_party/Depth-Anything-V2/metric_depth/depth_anything_v2"
    if not (root / "dpt.py").is_file():
        raise FileNotFoundError("Pinned DA-V2 metric source missing: " + str(root))
    if _METRIC_PACKAGE not in sys.modules:
        package = types.ModuleType(_METRIC_PACKAGE)
        package.__path__ = [str(root)]
        package.__package__ = _METRIC_PACKAGE
        sys.modules[_METRIC_PACKAGE] = package
    module = importlib.import_module(_METRIC_PACKAGE + ".dpt")
    return module.DepthAnythingV2


class FrozenMetricDepthPrior(nn.Module):
    prior_channels = 1
    representation = METRIC_REPRESENTATION

    def __init__(self, checkpoint=None, input_size=518, max_depth_m=80.0):
        super().__init__()
        if float(max_depth_m) != METRIC_MAX_DEPTH_M:
            raise ValueError("Pinned VKITTI-S metric model requires max_depth_m=80")
        path = Path(checkpoint or ROOT / "checkpoints" / METRIC_CHECKPOINT_FILENAME)
        # Hash must match before any torch deserialization.
        self.provenance = verify_checkpoint(METRIC_CHECKPOINT_KEY, path)
        MetricDepthAnythingV2 = _metric_model_class()
        self.network = MetricDepthAnythingV2(
            encoder="vits", features=64,
            out_channels=[48, 96, 192, 384], max_depth=METRIC_MAX_DEPTH_M)
        self.network.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True), strict=True)
        self.network.requires_grad_(False)
        self.network.eval()
        self.max_depth_m = METRIC_MAX_DEPTH_M
        # The resize is shared, but the network and its weights are NOT shared
        # with the relative-depth DA-V2 implementation.
        sys.path.insert(0, str(ROOT / "third_party/Depth-Anything-V2"))
        from depth_anything_v2.util.transform import Resize
        self.resize = Resize(input_size, input_size, resize_target=False,
                             keep_aspect_ratio=True, ensure_multiple_of=14,
                             resize_method="lower_bound")
        self.train(False)  # Keep wrapper and frozen network in eval mode.

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images):
        chunks = [self._extract(image[None]) for image in images]
        return [torch.cat([chunk[level] for chunk in chunks], dim=0)
                for level in range(4)]

    def _extract(self, images):
        # Same already-augmented detector RGB tensor and bicubic preprocessing
        # as E1-E3; this is not byte-equivalent to official cv2.infer_image.
        width, height = self.resize.get_size(images.shape[-1], images.shape[-2])
        x = F.interpolate(images.float(), (height, width), mode="bicubic", align_corners=False)
        # Official metric forward: sigmoid depth head multiplied by 80 metres.
        metric_m = self.network(x).unsqueeze(1)
        if metric_m.ndim != 4 or metric_m.shape[1] != 1:
            raise ValueError("Unexpected metric depth output shape")
        # Downsample to the same patch lattice used by E2 prior fusion.
        metric_m = F.interpolate(metric_m, (height // 14, width // 14),
                                 mode="bilinear", align_corners=False)
        if not torch.isfinite(metric_m).all() or (metric_m < 0).any() or (metric_m > METRIC_MAX_DEPTH_M + 1e-3).any():
            raise FloatingPointError("Invalid metric VKITTI depth (metres)")
        # No per-image z-score, percentile scaling, scale fitting or KITTI GT.
        return [metric_m] * 4
