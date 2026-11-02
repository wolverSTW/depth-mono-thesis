"""E4b isolated CPU contract tests; NOT KITTI training or evaluation evidence."""
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from depth_monoia.config import CONFIGS, RUN_NAMES, ROOT, resolve_config, load_config
from depth_monoia.distance_correction import ObjectDepthCorrection
from depth_monoia.model import DepthMonoIA


def _outputs(batch=2, queries=3):
    # Isolated tensor contract fixtures, not KITTI samples or reported metrics.
    z = torch.full((batch, queries, 1), 25.0, requires_grad=True)
    variance = torch.zeros(batch, queries, 1)
    boxes = torch.full((batch, queries, 6), 0.5)
    dims = torch.full((batch, queries, 3), 2.0)
    logits = torch.full((batch, queries, 3), 0.5)
    return {
        "pred_depth": torch.cat((z, variance), -1),
        "pred_boxes": boxes, "pred_3d_dim": dims, "pred_logits": logits,
        "aux_outputs": [{"pred_depth": torch.full((batch, queries, 2), 7.)}],
    }


class _PriorStub(nn.Module):
    prior_channels = 1
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, images):
        prior = torch.full((images.shape[0], 1, 2, 3), 22., device=images.device)
        return [prior] * 4


class _DetectorStub(nn.Module):
    hidden_dim = 32
    num_feature_levels = 4
    def __init__(self):
        super().__init__()
        self.connector = nn.Identity()
        self.region_head = nn.Identity()
        self.depth_scalar = nn.Parameter(torch.tensor(25.))

    def forward(self, images, calibs, targets, img_sizes, dn_args=None):
        b = images.shape[0]
        self.connector(torch.zeros(b, 32, device=images.device))
        features = [torch.ones(b, 32, 2, 3, device=images.device) for _ in range(4)]
        self.region_head(features)
        outputs = _outputs(b, 3)
        # Keep a trainable predicted depth path for criterion integration checks.
        outputs["pred_depth"] = torch.stack((
            self.depth_scalar.expand(b, 3), torch.zeros(b, 3, device=images.device)), -1)
        return outputs


class E4bArchitectureTests(unittest.TestCase):
    def test_e4b_config_extends_e4a_and_registry(self):
        self.assertIn("E4b", CONFIGS)
        self.assertIn("E4b", RUN_NAMES)
        import yaml
        raw = yaml.safe_load((ROOT / CONFIGS["E4b"]).read_text())
        self.assertEqual(raw["extends"], "E4a.yaml")
        self.assertEqual(raw["experiment"]["id"], "E4b")
        self.assertEqual(raw["depth"]["distance_correction"]["auxiliary"], "unchanged")
        self.assertNotIn("distance_correction", yaml.safe_load((ROOT / CONFIGS["E4a"]).read_text())["depth"])

    @unittest.skipUnless((ROOT / "MonoIA/config/monoia.yaml").is_file(),
                         "Pinned MonoIA config absent in isolated source package")
    def test_resolved_protocol_e4a_e4b_parity_and_rejection(self):
        e4a = resolve_config(ROOT / CONFIGS["E4a"])
        e4b = resolve_config(ROOT / CONFIGS["E4b"])
        for field in ("dataset", "model", "optimizer", "lr_scheduler", "trainer", "random_seed"):
            self.assertEqual(e4a[field], e4b[field], field)
        self.assertEqual(e4b["depth"]["representation"], e4a["depth"]["representation"])
        self.assertEqual(e4b["depth"]["checkpoint_key"], "dav2_metric_vkitti_small")
        with patch("depth_monoia.config.load_config", return_value={
                **e4b, "depth": {**e4b["depth"], "distance_correction": {
                    **e4b["depth"]["distance_correction"], "auxiliary": "corrected"}}}):
            with self.assertRaises(ValueError):
                resolve_config(ROOT / CONFIGS["E4b"])

    def test_exact_initial_depth_identity_and_untouched_metadata(self):
        inputs = _outputs()
        previous = inputs["pred_depth"].clone()
        aux = inputs["aux_outputs"]
        head = ObjectDepthCorrection()
        result = head(inputs, torch.full((2, 1, 3, 5), 21.0))
        self.assertTrue(torch.equal(result["pred_depth"], previous))
        self.assertIs(result["aux_outputs"], aux)
        self.assertIs(result["pred_boxes"], inputs["pred_boxes"])
        self.assertTrue(torch.equal(result["pred_depth"][..., 1], previous[..., 1]))
        self.assertIsNot(result, inputs)

    def test_two_step_gradient_connectivity_and_metric_influence(self):
        torch.manual_seed(42)
        head = ObjectDepthCorrection(hidden_dim=16)
        optim = torch.optim.SGD(head.parameters(), lr=0.01)
        for step in range(2):
            optim.zero_grad()
            result = head(_outputs(), torch.full((2, 1, 4, 4), 18.0))
            loss = ((result["pred_depth"][..., 0] - 29.0) ** 2).mean()
            loss.backward()
            self.assertGreater(head.mlp[-1].weight.grad.abs().sum().item(), 0)
            if step == 1:
                self.assertGreater(head.mlp[0].weight.grad.abs().sum().item(), 0)
            optim.step()
        base = _outputs()
        pred1 = head(base, torch.full((2, 1, 4, 4), 12.))["pred_depth"]
        pred2 = head(base, torch.full((2, 1, 4, 4), 33.))["pred_depth"]
        self.assertGreater((pred1[..., 0] - pred2[..., 0]).abs().sum().item(), 0)

    def test_out_of_bounds_coordinates_use_unmodified_z(self):
        head = ObjectDepthCorrection()
        with torch.no_grad():
            head.mlp[-1].bias.fill_(1.0)
        outputs = _outputs()
        outputs["pred_boxes"][0, 0, 0] = -0.1
        outputs["pred_boxes"][0, 1, 1] = 1.1
        result = head(outputs, torch.full((2, 1, 3, 4), 26.))
        self.assertTrue(torch.equal(result["pred_depth"][0, :2], outputs["pred_depth"][0, :2]))
        self.assertNotEqual(result["pred_depth"][0, 2, 0].item(), outputs["pred_depth"][0, 2, 0].item())

    def test_spatial_sampling_uses_projected_query_center(self):
        head = ObjectDepthCorrection()
        outputs = _outputs(batch=1, queries=2)
        outputs["pred_boxes"][0, 0, :2] = torch.tensor([0.25, 0.5])
        outputs["pred_boxes"][0, 1, :2] = torch.tensor([0.75, 0.5])
        values = torch.tensor([[[[10., 30.], [10., 30.]]]])
        sampled = torch.nn.functional.grid_sample(values, outputs["pred_boxes"][..., :2].mul(2).sub(1).unsqueeze(2),
                                                   align_corners=False, padding_mode="border").squeeze(-1).transpose(1, 2)
        torch.testing.assert_close(sampled.squeeze(), torch.tensor([10., 30.]))

    def test_nonfinite_or_malformed_inputs_are_rejected(self):
        head = ObjectDepthCorrection()
        for invalid in (float("nan"), float("inf")):
            outputs = _outputs()
            outputs["pred_depth"][0, 0, 0] = invalid
            with self.assertRaises(FloatingPointError):
                head(outputs, torch.ones(2, 1, 3, 3))
        with self.assertRaises(ValueError):
            head(_outputs(), torch.ones(2, 2, 3, 3))
        with self.assertRaises(ValueError):
            head(_outputs(), torch.full((2, 1, 3, 3), 81.))

    def test_train_eval_parity_deterministic_head(self):
        head = ObjectDepthCorrection()
        with torch.no_grad():
            head.mlp[-1].bias.fill_(0.3)
        outputs = _outputs()
        prior = torch.ones(2, 1, 3, 4)
        head.train(True)
        train_result = head(outputs, prior)["pred_depth"]
        head.eval()
        val_result = head(outputs, prior)["pred_depth"]
        self.assertTrue(torch.equal(train_result, val_result))

    def test_frozen_prior_namespace_and_runner_provenance(self):
        import inspect
        from depth_monoia import runner
        source = inspect.getsource(runner.preparation)
        self.assertIn('in ("E4a", "E4b")', source)
        self.assertIn('METRIC_CHECKPOINT_KEY', source)
        from depth_monoia.model import DepthMonoIA
        self.assertIn('self.distance_correction(outputs', inspect.getsource(DepthMonoIA.forward))

    def test_wrapper_corrects_output_before_existing_loss_interface(self):
        import yaml
        correction_cfg = yaml.safe_load((ROOT / CONFIGS["E4b"]).read_text())["depth"]["distance_correction"]
        cfg = {"representation": "single_metric_vkitti_depth_in_meters_only", "input_size": 518,
               "max_depth_m": 80.0, "prior_channels": 1, "mode": "gated", "alpha_init": 0.0,
               "distance_correction": correction_cfg}
        detector = _DetectorStub()
        with patch("depth_monoia.metric_depth.FrozenMetricDepthPrior", _PriorStub):
            model = DepthMonoIA(detector, cfg)
        self.assertIsInstance(model.distance_correction, ObjectDepthCorrection)
        self.assertFalse(any(p.requires_grad for p in model.prior.parameters()))
        images = torch.zeros(1, 3, 28, 42)
        output = model(images, None, None, None)
        self.assertEqual(tuple(output["pred_depth"].shape), (1, 3, 2))
        self.assertTrue(torch.equal(output["pred_depth"][..., 0], torch.full((1, 3), 25.)))
        with torch.no_grad():
            model.distance_correction.mlp[-1].bias.fill_(0.5)
        output2 = model(images, None, None, None)
        self.assertGreater((output2["pred_depth"][..., 0] - output["pred_depth"][..., 0]).abs().sum().item(), 0)
        self.assertTrue(torch.equal(output2["pred_depth"][..., 1], output["pred_depth"][..., 1]))
        self.assertIsNone(model._context)

    def test_baseline_e4a_mode_does_not_construct_correction(self):
        cfg = {"representation": "single_metric_vkitti_depth_in_meters_only", "input_size": 518,
               "max_depth_m": 80.0, "prior_channels": 1, "mode": "gated", "alpha_init": 0.0}
        with patch("depth_monoia.metric_depth.FrozenMetricDepthPrior", _PriorStub):
            model = DepthMonoIA(_DetectorStub(), cfg)
        self.assertIsNone(model.distance_correction)

    def test_correction_head_parameter_count_and_restriction(self):
        head = ObjectDepthCorrection(hidden_dim=64, max_delta_m=10)
        self.assertEqual(sum(p.numel() for p in head.parameters()), 10*64 + 64 + 64*64 + 64 + 64+1)
        with self.assertRaises(ValueError):
            ObjectDepthCorrection(hidden_dim=0)
        with self.assertRaises(ValueError):
            ObjectDepthCorrection(max_delta_m=0)


if __name__ == "__main__":
    unittest.main()
