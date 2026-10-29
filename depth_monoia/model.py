import sys
import torch
from torch import nn
from .config import ROOT
from .depth import FrozenDepthPrior
from .fusion import MultiScaleFusion
from .checkpoint import rng_state, restore_rng


def upstream_path():
    ops = str(ROOT / "build/ops")
    if ops not in sys.path:
        sys.path.insert(0, ops)
    source = str(ROOT / "MonoIA")
    if source not in sys.path:
        sys.path.insert(0, source)


class DepthMonoIA(nn.Module):
    """Hooks extend the original forward without copying or editing it.

    One GPU per process only. Context is cleared even after a failed forward.
    DataParallel, reentrant forwards and activation checkpointing are unsupported.
    """
    def __init__(self, detector, depth_cfg):
        super().__init__()
        self.detector = detector
        self.prior = FrozenDepthPrior(input_size=depth_cfg["input_size"])
        self.fusion = MultiScaleFusion(depth_cfg["mode"], detector.hidden_dim,
                                       detector.num_feature_levels, depth_cfg["alpha_init"])
        self._context = None
        self._bypass = False
        self._hooks = [detector.connector.register_forward_hook(self._capture_embedding),
                       detector.region_head.register_forward_pre_hook(self._fuse)]

    def _capture_embedding(self, module, args, output):
        if self._bypass:
            return
        if self._context is None:
            raise RuntimeError("Missing per-forward depth context")
        self._context["embedding"] = output

    def _fuse(self, module, args):
        if self._bypass:
            return None
        ctx = self._context
        return (self.fusion(args[0], ctx["priors"], ctx["embedding"], ctx.get("confidence")),)

    def baseline_forward(self, *args, **kwargs):
        self._bypass = True
        try:
            return self.detector(*args, **kwargs)
        finally:
            self._bypass = False

    def forward(self, images, calibs, targets, img_sizes, dn_args=None):
        if self._context is not None:
            raise RuntimeError("Reentrant/DataParallel forward is unsupported")
        self._context = {"priors": self.prior(images)}
        try:
            if self.fusion.levels[0].mode == "confidence":
                # Flip consistency is a relative-prior reliability heuristic,
                # not a calibrated distance-confidence or uncertainty model.
                flipped = self.prior(images.flip(-1))
                a = self._context["priors"][-1][:, -1:]
                b = flipped[-1][:, -1:].flip(-1)
                self._context["confidence"] = torch.exp(-(a - b).abs())
            return self.detector(images, calibs, targets, img_sizes, dn_args)
        finally:
            self._context = None


def build_model(cfg):
    upstream_path()
    from lib.helpers.model_helper import build_model as original_builder
    detector, criterion = original_builder(cfg["model"])
    if cfg["experiment"]["id"] == "A0":
        return detector, criterion
    # Extra module initialization must not shift the original dataloader,
    # augmentation or dropout random streams between comparable experiments.
    state = rng_state()
    try:
        extended = DepthMonoIA(detector, cfg["depth"])
    finally:
        restore_rng(state)
    return extended, criterion
