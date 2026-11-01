"""Isolated E2 dense-depth-only unit contracts; NOT KITTI benchmark evidence."""
import unittest
import torch
from torch import nn
from depth_monoia.depth import FrozenDepthPrior
from depth_monoia.fusion import MultiScaleFusion
from depth_monoia.config import ROOT, CONFIGS, RUN_NAMES, resolve_config
from test_e1_architecture import _prior_for_isolated_test


class E2ArchitectureTests(unittest.TestCase):
    def test_e2_registry_and_config_file(self):
        self.assertIn("E2", CONFIGS)
        self.assertIn("E2", RUN_NAMES)
        self.assertTrue((ROOT / CONFIGS["E2"]).is_file())

    @unittest.skipUnless((ROOT / "MonoIA/config/monoia.yaml").is_file(),
                         "Pinned original MonoIA config not bundled with isolated review")
    def test_e2_resolved_config_and_legacy_contract(self):
        cfg = resolve_config(ROOT / CONFIGS["E2"])
        self.assertEqual(cfg["experiment"]["id"], "E2")
        self.assertEqual(cfg["depth"]["mode"], "gated")
        self.assertEqual(cfg["depth"]["representation"], "single_standardized_relative_depth_map_only")
        self.assertEqual(cfg["depth"]["prior_channels"], 1)
        self.assertTrue(cfg["depth"]["frozen"])
        self.assertEqual(cfg["depth"]["alpha_init"], 0.0)
        self.assertEqual(resolve_config(ROOT / CONFIGS["E1"])["depth"]["prior_channels"], 384)
        legacy = resolve_config(ROOT / CONFIGS["A2"])
        self.assertEqual(legacy["depth"].get("prior_channels", 385), 385)
        self.assertEqual(
            MultiScaleFusion("gated").levels[0].adapter[0].in_channels,
            385,
        )
        # A wrong E2 channel contract must fail before a model is built.
        import tempfile
        import yaml
        with tempfile.TemporaryDirectory() as temporary:
            invalid = dict(cfg)
            invalid["depth"] = dict(cfg["depth"], prior_channels=384)
            config_path = __import__("pathlib").Path(temporary) / "invalid_e2.yaml"
            config_path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "representation and channel"):
                resolve_config(config_path)

    def test_dense_depth_only_four_single_channel_maps_and_head_called(self):
        prior = _prior_for_isolated_test("single_standardized_relative_depth_map_only")
        prior.prior_channels = 1  # E1 fixture bypasses production __init__.
        # The test fixture's depth head returns a constant field; it verifies
        # decoder execution and shape, not depth accuracy or KITTI performance.
        inputs = torch.ones(2, 3, 28, 56)
        maps = prior(inputs)
        self.assertEqual(len(maps), 4)
        self.assertTrue(all(x.shape == (2, 1, 2, 4) for x in maps))
        self.assertEqual(prior.network.pretrained.calls, 2)
        self.assertEqual(prior.network.depth_head.calls, 2)
        self.assertTrue(all(torch.equal(maps[0], x) for x in maps[1:]))
        self.assertTrue(torch.isfinite(maps[0]).all().item())
        self.assertEqual(prior.prior_channels, 1)

    def test_no_encoder_normalization_in_e2(self):
        prior = _prior_for_isolated_test("single_standardized_relative_depth_map_only")
        prior.prior_channels = 1  # E1 fixture bypasses production __init__.
        original = torch.nn.functional.layer_norm
        def forbidden(*args, **kwargs):
            raise AssertionError("E2 must not normalize or fuse encoder tokens")
        torch.nn.functional.layer_norm = forbidden
        try:
            maps = prior(torch.ones(1, 3, 28, 56))
            self.assertEqual(len(maps), 4)
        finally:
            torch.nn.functional.layer_norm = original

    def test_fusion_exact_initial_parity_and_channel_rejection(self):
        torch.manual_seed(21)
        fusion = MultiScaleFusion("gated", prior_channels=1)
        self.assertEqual(fusion.levels[0].adapter[0].weight.shape, (256, 1, 1, 1))
        rgb = [torch.randn(1, 256, 3, 5) for _ in range(4)]
        maps = [torch.randn(1, 1, 2, 4) for _ in range(4)]
        result = fusion(rgb, maps, torch.randn(1, 256))
        for before, after in zip(rgb, result):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            fusion(rgb, [torch.randn(1, 384, 2, 4) for _ in range(4)], torch.randn(1, 256))

    def test_e2_two_step_gradient_connectivity(self):
        torch.manual_seed(22)
        fusion = MultiScaleFusion("gated", prior_channels=1)
        rgb = [torch.randn(1, 256, 3, 5) for _ in range(4)]
        maps = [torch.randn(1, 1, 2, 4) for _ in range(4)]
        embedding = torch.randn(1, 256)
        optimizer = torch.optim.SGD(fusion.parameters(), lr=0.1)
        for _ in range(2):
            optimizer.zero_grad()
            sum(y.square().mean() for y in fusion(rgb, maps, embedding)).backward()
            optimizer.step()
        for name, param in fusion.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all().item(), name)
            self.assertGreater(param.grad.abs().sum().item(), 0, name)

    def test_fusion_capacity_difference_recorded(self):
        e1 = MultiScaleFusion("gated", prior_channels=384)
        e2 = MultiScaleFusion("gated", prior_channels=1)
        n1 = sum(p.numel() for p in e1.parameters() if p.requires_grad)
        n2 = sum(p.numel() for p in e2.parameters() if p.requires_grad)
        self.assertEqual(n1 - n2, 4 * 256 * (384 - 1))
        self.assertNotEqual(n1, n2)  # Representation confound, NOT an AP conclusion.


if __name__ == "__main__":
    unittest.main()
