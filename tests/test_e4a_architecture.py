"""E4a isolated metric-depth architecture checks, not KITTI benchmark evidence."""
import importlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
import yaml

from depth_monoia.config import CONFIGS, ROOT, RUN_NAMES, resolve_config
from depth_monoia.fusion import MultiScaleFusion
from depth_monoia.metric_depth import (
    FrozenMetricDepthPrior, METRIC_REPRESENTATION, METRIC_CHECKPOINT_KEY,
    METRIC_CHECKPOINT_FILENAME, METRIC_MAX_DEPTH_M,
)
from depth_monoia.provenance import checkpoint_info


class _ResizeStub:
    def get_size(self, w, h):
        return w, h


class _MetricStub(nn.Module):
    def __init__(self, value=12.5):
        super().__init__()
        self.value = value
    def forward(self, x):
        return torch.full((x.shape[0], x.shape[-2], x.shape[-1]), self.value,
                          dtype=x.dtype, device=x.device)


class E4aArchitectureTests(unittest.TestCase):
    def test_registry_and_e2_inheritance(self):
        self.assertEqual(CONFIGS["E4a"], "configs/experiments/E4a.yaml")
        self.assertEqual(RUN_NAMES["E4a"], "E4a_metric_vkitti_gated_fusion")
        e4 = yaml.safe_load((ROOT / CONFIGS["E4a"]).read_text())
        self.assertEqual(e4["extends"], "E2.yaml")
        self.assertEqual(e4["depth"]["representation"], METRIC_REPRESENTATION)
        self.assertEqual(e4["depth"]["metric_units"], "metres")
        self.assertEqual(e4["depth"]["max_depth_m"], METRIC_MAX_DEPTH_M)
        self.assertNotIn("correction", e4["depth"])

    @unittest.skipUnless((ROOT / "MonoIA/config/monoia.yaml").is_file(),
                         "Pinned MonoIA parent config unavailable in isolated package")
    def test_resolved_e4a_e2_training_protocol_parity(self):
        e2 = resolve_config(ROOT / CONFIGS["E2"])
        e4 = resolve_config(ROOT / CONFIGS["E4a"])
        for key in ("dataset", "model", "optimizer", "lr_scheduler", "trainer", "random_seed"):
            self.assertEqual(e2[key], e4[key], key)
        self.assertEqual(e4["depth"]["prior_channels"], 1)
        self.assertEqual(e4["depth"]["mode"], "gated")

    def test_official_checkpoint_identity_is_separate(self):
        spec = checkpoint_info(METRIC_CHECKPOINT_KEY)
        self.assertEqual(spec["sha256"],
                         "9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb")
        self.assertEqual(Path(spec["path"]).name, METRIC_CHECKPOINT_FILENAME)
        self.assertNotEqual(spec["sha256"], checkpoint_info("dav2_small")["sha256"])
        self.assertEqual(FrozenMetricDepthPrior.prior_channels, 1)

    def test_metres_preserved_no_per_image_standardization(self):
        # Isolated tensor algebra only; synthetic values are NOT KITTI data/results.
        prior = FrozenMetricDepthPrior.__new__(FrozenMetricDepthPrior)
        nn.Module.__init__(prior)
        prior.resize = _ResizeStub()
        prior.network = _MetricStub(12.5)
        prior.max_depth_m = 80.0
        images = torch.zeros(2, 3, 28, 42)
        out = prior(images)
        self.assertEqual(len(out), 4)
        for depth in out:
            self.assertEqual(tuple(depth.shape), (2, 1, 2, 3))
            self.assertTrue(torch.equal(depth, torch.full_like(depth, 12.5)))
        self.assertFalse(any(p.requires_grad for p in prior.parameters()))

    def test_metric_values_outside_range_fail(self):
        prior = FrozenMetricDepthPrior.__new__(FrozenMetricDepthPrior)
        nn.Module.__init__(prior)
        prior.resize = _ResizeStub()
        prior.network = _MetricStub(81.0)
        prior.max_depth_m = 80.0
        with self.assertRaises(FloatingPointError):
            prior(torch.zeros(1, 3, 28, 42))

    def test_fusion_parameter_and_initialization_comparable_to_e2(self):
        torch.manual_seed(57)
        e2 = MultiScaleFusion("gated", prior_channels=1)
        e4 = MultiScaleFusion("gated", prior_channels=1)
        self.assertEqual(sum(p.numel() for p in e2.parameters()), sum(p.numel() for p in e4.parameters()))
        rgb = [torch.randn(1, 256, 3, 5) for _ in range(4)]
        metric = [torch.ones(1, 1, 2, 3) * 12.5 for _ in range(4)]
        for original, output in zip(rgb, e4(rgb, metric, torch.zeros(1, 256))):
            torch.testing.assert_close(original, output, rtol=0, atol=0)

    def test_vit_metric_namespace_differs_from_relative(self):
        from depth_monoia.metric_depth import _metric_model_class
        root = ROOT / "third_party/Depth-Anything-V2/metric_depth/depth_anything_v2/dpt.py"
        if not root.is_file():
            self.skipTest("Pinned metric source unavailable in isolated package")
        metric_cls = _metric_model_class()
        relative_root = str(ROOT / "third_party/Depth-Anything-V2")
        if relative_root not in sys.path:
            sys.path.insert(0, relative_root)
        import depth_anything_v2.dpt as relative
        self.assertIsNot(metric_cls, relative.DepthAnythingV2)
        self.assertIn("depth_monoia_pinned_dav2_metric", metric_cls.__module__)


    @unittest.skipUnless(
        (ROOT / "checkpoints/depth_anything_v2_metric_vkitti_vits.pth").is_file() and
        (ROOT / "third_party/Depth-Anything-V2/metric_depth/depth_anything_v2/dpt.py").is_file(),
        "Authentic verified metric checkpoint and pinned upstream source unavailable")
    def test_authentic_official_metric_weight_strict_load(self):
        # This loads genuine official weights (no synthetic KITTI examples/results).
        # Repository manifest SHA256 is checked before weights_only=True deserialization.
        prior = FrozenMetricDepthPrior(input_size=518)
        self.assertFalse(prior.training)
        self.assertEqual(prior.network.max_depth, 80.0)
        self.assertFalse(any(p.requires_grad for p in prior.network.parameters()))
        self.assertEqual(prior.provenance["sha256"], checkpoint_info(METRIC_CHECKPOINT_KEY)["sha256"])
        prior.train(True)
        self.assertFalse(prior.training)

    def test_metric_provenance_precedes_deserialization(self):
        from depth_monoia.metric_depth import _metric_model_class
        with patch("depth_monoia.metric_depth.verify_checkpoint", side_effect=ValueError("unverified")):
            with patch("depth_monoia.metric_depth._metric_model_class") as constructor:
                with self.assertRaisesRegex(ValueError, "unverified"):
                    FrozenMetricDepthPrior(checkpoint=Path("not_a_real_weight.pth"))
                constructor.assert_not_called()

if __name__ == "__main__":
    unittest.main()
