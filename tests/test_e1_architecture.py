"""Isolated E1 encoder-only unit contracts; NOT KITTI benchmark evidence."""
import unittest
import torch
from torch import nn
from depth_monoia.depth import FrozenDepthPrior
from depth_monoia.fusion import MultiScaleFusion
from depth_monoia.config import ROOT, CONFIGS, RUN_NAMES, resolve_config


class _Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def get_intermediate_layers(self, x, layers, return_class_token=True):
        self.calls += 1
        assert return_class_token is True
        b, _, h, w = x.shape
        assert h % 14 == 0 and w % 14 == 0
        # Mathematically isolated tensor fixture, not mock KITTI data.
        tokens = torch.arange(384, dtype=x.dtype, device=x.device).repeat(b, (h // 14) * (w // 14), 1)
        return [(tokens + float(i), None) for i in range(4)]


class _DepthHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, features, h, w):
        self.calls += 1
        return torch.ones(features[0][0].shape[0], 1, h * 14, w * 14)


class _Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = _Pretrained()
        self.depth_head = _DepthHead()
        self.intermediate_layer_idx = {"vits": [2, 5, 8, 11]}


class _Resize:
    def get_size(self, width, height):
        return width, height


def _prior_for_isolated_test(representation):
    prior = FrozenDepthPrior.__new__(FrozenDepthPrior)
    nn.Module.__init__(prior)
    prior.representation = representation
    prior.prior_channels = 384 if representation == "four_DINO_intermediates_only" else 385
    prior.network = _Network()
    prior.resize = _Resize()
    prior.network.requires_grad_(False)
    return prior


class E1ArchitectureTests(unittest.TestCase):
    def test_e1_registered(self):
        self.assertIn("E1", CONFIGS)
        self.assertIn("E1", RUN_NAMES)

    @unittest.skipUnless((ROOT / "MonoIA/config/monoia.yaml").is_file(),
                         "Pinned original MonoIA config is not in this review package")
    def test_e1_resolved_config(self):
        config = resolve_config(ROOT / CONFIGS["E1"])
        self.assertEqual(config["experiment"]["id"], "E1")
        self.assertEqual(config["depth"]["mode"], "gated")
        self.assertEqual(config["depth"]["prior_channels"], 384)
        self.assertEqual(config["depth"]["representation"], "four_DINO_intermediates_only")
        self.assertTrue(config["depth"]["frozen"])
        self.assertEqual(config["depth"]["alpha_init"], 0)
        self.assertEqual(resolve_config(ROOT / CONFIGS["A2"])["depth"]["representation"],
                         "four_DINO_intermediates_plus_standardized_relative_depth")

    def test_encoder_only_never_calls_depth_head(self):
        prior = _prior_for_isolated_test("four_DINO_intermediates_only")
        priors = prior(torch.ones(2, 3, 28, 56))
        self.assertEqual(len(priors), 4)
        self.assertTrue(all(x.shape == (2, 384, 2, 4) for x in priors))
        self.assertEqual(prior.network.pretrained.calls, 2)  # microbatch 1
        self.assertEqual(prior.network.depth_head.calls, 0)
        self.assertTrue(all(torch.isfinite(x).all().item() for x in priors))
        prior.train(True)
        self.assertFalse(prior.training)
        self.assertFalse(prior.network.training)

    def test_legacy_mode_still_uses_depth_head(self):
        prior = _prior_for_isolated_test("four_DINO_intermediates_plus_standardized_relative_depth")
        priors = prior(torch.ones(1, 3, 28, 56))
        self.assertTrue(all(x.shape == (1, 385, 2, 4) for x in priors))
        self.assertEqual(prior.network.depth_head.calls, 1)
        self.assertTrue(all(torch.isfinite(x).all().item() for x in priors))

    def test_fusion_parameter_shape_and_exact_zero_alpha_identity(self):
        torch.manual_seed(5)
        e1 = MultiScaleFusion("gated", prior_channels=384)
        legacy = MultiScaleFusion("gated")
        self.assertEqual(e1.levels[0].adapter[0].weight.shape, (256, 384, 1, 1))
        self.assertEqual(legacy.levels[0].adapter[0].weight.shape, (256, 385, 1, 1))
        rgb = [torch.randn(1, 256, 3, 5) for _ in range(4)]
        prior = [torch.randn(1, 384, 2, 4) for _ in range(4)]
        y = e1(rgb, prior, torch.randn(1, 256))
        for a, b in zip(rgb, y):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "expected input"): # wrong channels fail in convolution
            e1(rgb, [torch.randn(1, 385, 2, 4) for _ in range(4)], torch.randn(1, 256))

    def test_e1_two_step_gradient_connectivity(self):
        torch.manual_seed(6)
        fusion = MultiScaleFusion("gated", prior_channels=384)
        rgb = [torch.randn(1, 256, 3, 5) for _ in range(4)]
        prior = [torch.randn(1, 384, 2, 4) for _ in range(4)]
        emb = torch.randn(1, 256)
        optim = torch.optim.SGD(fusion.parameters(), lr=0.1)
        for _ in range(2):
            optim.zero_grad()
            out = fusion(rgb, prior, emb)
            sum(x.square().mean() for x in out).backward()
            optim.step()
        for name, p in fusion.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all().item(), name)
            self.assertGreater(p.grad.abs().sum().item(), 0, name)


if __name__ == "__main__":
    unittest.main()
