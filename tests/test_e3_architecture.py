"""E3 isolated intrinsic-gate ablation contracts; not KITTI results."""
import unittest
from pathlib import Path
import yaml
import torch
from depth_monoia.config import ROOT, CONFIGS, RUN_NAMES, load_config, resolve_config
from depth_monoia.fusion import MultiScaleFusion


class E3ArchitectureTests(unittest.TestCase):
    def test_registered_and_config_inherits_e2(self):
        self.assertIn("E3", CONFIGS)
        self.assertIn("E3", RUN_NAMES)
        self.assertEqual(Path(CONFIGS["E3"]).as_posix(), "configs/experiments/E3.yaml")
        # Inspect the local E2/E3 YAML override contract even when the pinned
        # MonoIA parent config is absent from this isolated source package.
        cfg = yaml.safe_load((ROOT / CONFIGS["E3"]).read_text(encoding="utf-8"))
        e2 = yaml.safe_load((ROOT / CONFIGS["E2"]).read_text(encoding="utf-8"))
        self.assertEqual(cfg["extends"], "E2.yaml")
        self.assertEqual(cfg["experiment"]["id"], "E3")
        self.assertEqual(cfg["depth"], {"mode": "intrinsic"})
        self.assertEqual(e2["depth"]["mode"], "gated")
        self.assertEqual(e2["depth"]["representation"], "single_standardized_relative_depth_map_only")
        self.assertEqual(e2["depth"]["prior_channels"], 1)

    @unittest.skipUnless((ROOT / "MonoIA/config/monoia.yaml").is_file(),
                         "Pinned original MonoIA config absent in isolated source package")
    def test_resolved_e2_e3_only_depth_mode_changes(self):
        e2 = resolve_config(ROOT / CONFIGS["E2"])
        e3 = resolve_config(ROOT / CONFIGS["E3"])
        self.assertEqual(e3["experiment"]["id"], "E3")
        self.assertEqual(e3["depth"]["mode"], "intrinsic")
        self.assertEqual(e3["depth"]["representation"], e2["depth"]["representation"])
        self.assertEqual(e3["depth"]["prior_channels"], 1)
        for key in ("dataset", "model", "optimizer", "lr_scheduler", "trainer", "random_seed"):
            self.assertEqual(e3[key], e2[key], key)

    def test_gate_structure_parameter_delta(self):
        e2 = MultiScaleFusion("gated", prior_channels=1)
        e3 = MultiScaleFusion("intrinsic", prior_channels=1)
        self.assertEqual(len(e3.levels), 4)
        for layer in e3.levels:
            self.assertEqual(layer.adapter[0].in_channels, 1)
            self.assertIsNotNone(layer.intrinsic)
            self.assertEqual(tuple(layer.intrinsic.weight.shape), (256, 256))
            self.assertEqual(tuple(layer.intrinsic.bias.shape), (256,))
        for layer in e2.levels:
            self.assertIsNone(layer.intrinsic)
        added = sum(p.numel() for p in e3.parameters()) - sum(p.numel() for p in e2.parameters())
        self.assertEqual(added, 4 * (256 * 256 + 256))
        self.assertEqual(added, 263168)

    def test_zero_alpha_exact_identity_in_both_modes(self):
        torch.manual_seed(82)
        rgb = [torch.randn(1, 256, 3, 5) for _ in range(4)]
        depth = [torch.randn(1, 1, 2, 4) for _ in range(4)]
        emb = torch.randn(1, 256)
        for mode in ("gated", "intrinsic"):
            model = MultiScaleFusion(mode, prior_channels=1)
            output = model(rgb, depth, emb)
            for a, b in zip(rgb, output):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_intrinsic_embedding_actually_modulates_fusion(self):
        torch.manual_seed(83)
        fusion = MultiScaleFusion("intrinsic", prior_channels=1)
        # Isolated tensor algebra, NOT KITTI data.
        rgb = [torch.randn(1, 256, 3, 5) for _ in range(4)]
        depth = [torch.randn(1, 1, 2, 4) for _ in range(4)]
        with torch.no_grad():
            for layer in fusion.levels:
                layer.alpha.fill_(1.0)
        zero = torch.zeros(1, 256)
        other = torch.ones(1, 256)
        a = fusion(rgb, depth, zero)
        b = fusion(rgb, depth, other)
        self.assertTrue(any((x - y).abs().max().item() > 1e-7 for x, y in zip(a, b)))
        # E2's gate, by construction, ignores embedding.
        e2 = MultiScaleFusion("gated", prior_channels=1)
        with torch.no_grad():
            for layer in e2.levels:
                layer.alpha.fill_(1.0)
        c = e2(rgb, depth, zero)
        d = e2(rgb, depth, other)
        for x, y in zip(c, d):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_intrinsic_projection_has_two_step_gradients(self):
        torch.manual_seed(84)
        model = MultiScaleFusion("intrinsic", prior_channels=1)
        rgb = [torch.randn(1, 256, 3, 5) for _ in range(4)]
        depth = [torch.randn(1, 1, 2, 4) for _ in range(4)]
        emb = torch.randn(1, 256)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(2):
            opt.zero_grad()
            sum(o.square().mean() for o in model(rgb, depth, emb)).backward()
            opt.step()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all().item(), name)
            self.assertGreater(param.grad.abs().sum().item(), 0, name)
        self.assertTrue(any(name.endswith("intrinsic.weight") for name, _ in model.named_parameters()))

    def test_prior_channel_rejection(self):
        fusion = MultiScaleFusion("intrinsic", prior_channels=1)
        rgb = [torch.randn(1, 256, 3, 5) for _ in range(4)]
        priors = [torch.randn(1, 384, 2, 4) for _ in range(4)]
        with self.assertRaises(RuntimeError):
            fusion(rgb, priors, torch.zeros(1, 256))

    @unittest.skipUnless((ROOT / "MonoIA/config/monoia.yaml").is_file(),
                         "Pinned original MonoIA config absent in isolated source package")
    def test_inherited_e2_depth_head_unchanged(self):
        e2 = load_config(ROOT / CONFIGS["E2"])["depth"]
        e3 = load_config(ROOT / CONFIGS["E3"])["depth"]
        self.assertEqual(e3["representation"], e2["representation"])
        self.assertEqual(e3["prior_channels"], e2["prior_channels"])
        self.assertEqual(e3["input_size"], e2["input_size"])
        self.assertEqual(e3["frozen"], e2["frozen"])


if __name__ == "__main__":
    unittest.main()
