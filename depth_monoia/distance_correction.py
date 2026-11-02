"""E4b query-level metric-depth residual head; no KITTI labels used as inputs.

The metric prior is *evidence*, not object-centre ground truth. The residual
is trained in the unchanged MonoIA depth-target domain, which may include
training crop scaling. No conversion between physical Z and training Z is
assumed. Predicted centre UV and the prior share the detector image frame.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ObjectDepthCorrection(nn.Module):
    """Correct only query depth; keep log variance and auxiliary layers intact."""

    FEATURE_DIM = 10

    def __init__(self, hidden_dim=64, max_delta_m=10.0):
        super().__init__()
        if not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError("Positive integer hidden_dim required")
        if not 0.0 < float(max_delta_m) <= 80.0:
            raise ValueError("max_delta_m must be in (0, 80]")
        self.max_delta_m = float(max_delta_m)
        self.mlp = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Exact E4a depth at initialization; gradients reach the final layer
        # on step 1 and deeper MLP layers after the first optimizer step.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, outputs, metric_map):
        required = ("pred_depth", "pred_boxes", "pred_3d_dim", "pred_logits")
        if not isinstance(outputs, dict) or any(k not in outputs for k in required):
            raise ValueError("MonoIA query prediction keys missing")
        z_unc = outputs["pred_depth"]
        boxes, dims, logits = (outputs[k] for k in ("pred_boxes", "pred_3d_dim", "pred_logits"))
        if (z_unc.ndim != 3 or z_unc.shape[-1] != 2 or boxes.shape != (*z_unc.shape[:2], 6)
                or dims.shape != (*z_unc.shape[:2], 3) or logits.ndim != 3
                or logits.shape[:2] != z_unc.shape[:2] or logits.shape[-1] < 1):
            raise ValueError("Unexpected MonoIA query output shape")
        if (not isinstance(metric_map, torch.Tensor) or metric_map.ndim != 4 or
                metric_map.shape[:2] != (z_unc.shape[0], 1) or
                metric_map.shape[-2] < 1 or metric_map.shape[-1] < 1 or
                metric_map.device != z_unc.device or metric_map.dtype != z_unc.dtype):
            raise ValueError("Metric map batch/channel/device/dtype mismatch")
        for tensor in (z_unc, boxes, dims, logits, metric_map):
            if not torch.isfinite(tensor).all():
                raise FloatingPointError("Nonfinite E4b inputs")
        if (metric_map < 0).any() or (metric_map > 80.001).any():
            raise ValueError("Expected 0-80 m metric VKITTI prior")

        # MonoIA's normalized projected 3D object centre (u,v) is already in
        # the *augmented detector image* frame used for metric DA-V2 input.
        uv = boxes[..., :2].detach()
        valid = ((uv >= 0) & (uv <= 1)).all(dim=-1, keepdim=True)
        grid = uv.mul(2).sub(1).unsqueeze(2)  # [B, Q, 1, 2]
        sampled = F.grid_sample(
            metric_map.detach(), grid, mode="bilinear", padding_mode="border",
            align_corners=False).squeeze(-1).transpose(1, 2)  # [B,Q,1]
        z = z_unc[..., :1]
        # Detaching the auxiliary features keeps correction supervision from
        # perturbing class/position/dimension heads through the sampling path.
        with torch.no_grad():
            confidence = logits.sigmoid().amax(-1, keepdim=True)
            features = torch.cat((
                z.detach() / 80.0, sampled / 80.0, (sampled - z.detach()) / 80.0,
                uv, z_unc[..., 1:2].detach().clamp(-10, 10),
                confidence, dims.detach() / 10.0,
            ), dim=-1)
        if features.shape[-1] != self.FEATURE_DIM:
            raise RuntimeError("E4b feature schema changed")
        delta = self.max_delta_m * self.mlp(features).tanh()
        delta = torch.where(valid, delta, torch.zeros_like(delta))
        corrected = torch.cat((z + delta, z_unc[..., 1:2]), dim=-1)
        return {**outputs, "pred_depth": corrected}
