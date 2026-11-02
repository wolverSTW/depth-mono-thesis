"""Permanent E4b depth-safety engineering tests.

No KITTI dataset, training, or benchmark measurements are used.
"""

import unittest

import torch

from depth_monoia.distance_correction import ObjectDepthCorrection


class E4bDepthSafetyTests(unittest.TestCase):

    def setUp(self):
        self.head = ObjectDepthCorrection(
            hidden_dim=64,
            max_delta_m=10.0,
        )
        self.prior = torch.full((1, 1, 8, 8), 5.0)

    @staticmethod
    def outputs(z, u=0.5):
        return {
            "pred_depth": torch.tensor([[[float(z), 0.0]]]),
            "pred_boxes": torch.tensor(
                [[[float(u), 0.5, 0.1, 0.1, 0.1, 0.1]]]
            ),
            "pred_3d_dim": torch.ones(1, 1, 3),
            "pred_logits": torch.zeros(1, 1, 3),
        }

    def set_bias(self, bias):
        with torch.no_grad():
            self.head.mlp[-1].bias.fill_(float(bias))

    def test_initial_identity(self):
        for z in (-2.0, 0.0, 0.001, 0.01, 2.0, 80.0):
            with self.subTest(z=z):
                data = self.outputs(z)
                result = self.head(data, self.prior)
                self.assertTrue(
                    torch.equal(
                        result["pred_depth"],
                        data["pred_depth"],
                    )
                )

    def test_nonpositive_depth_unchanged(self):
        for bias in (-5.0, -1.0, 1.0, 5.0):
            self.set_bias(bias)

            for z in (-2.0, 0.0):
                with self.subTest(bias=bias, z=z):
                    data = self.outputs(z)
                    result = self.head(data, self.prior)

                    self.assertTrue(
                        torch.equal(
                            result["pred_depth"],
                            data["pred_depth"],
                        )
                    )

    def test_negative_residual_preserves_positive_depth(self):
        for bias in (-20.0, -5.0, -1.0):
            self.set_bias(bias)

            for z in (0.001, 0.01, 0.1, 2.0, 10.0, 40.0, 80.0):
                with self.subTest(bias=bias, z=z):
                    result = self.head(
                        self.outputs(z), self.prior
                    )["pred_depth"]

                    new_z = result[0, 0, 0].item()

                    self.assertTrue(
                        torch.isfinite(result).all().item()
                    )
                    self.assertGreater(new_z, 0.0)
                    self.assertLessEqual(new_z, z)

    def test_positive_residual_available(self):
        self.set_bias(1.0)

        data = self.outputs(2.0)
        result = self.head(data, self.prior)["pred_depth"]

        self.assertGreater(
            result[0, 0, 0].item(), 2.0
        )
        self.assertTrue(
            torch.equal(
                result[..., 1:],
                data["pred_depth"][..., 1:],
            )
        )

    def test_out_of_bounds_identity(self):
        self.set_bias(1.0)

        data = self.outputs(2.0, u=1.5)
        result = self.head(data, self.prior)

        self.assertTrue(
            torch.equal(
                result["pred_depth"],
                data["pred_depth"],
            )
        )

    def test_gradient_connectivity(self):
        self.set_bias(-1.0)
        self.head.zero_grad(set_to_none=True)

        result = self.head(
            self.outputs(2.0), self.prior
        )["pred_depth"]

        result[..., 0].sum().backward()

        grad = self.head.mlp[-1].bias.grad

        self.assertIsNotNone(grad)
        self.assertTrue(
            torch.isfinite(grad).all().item()
        )
        self.assertGreater(
            grad.abs().sum().item(), 0.0
        )


if __name__ == "__main__":
    unittest.main()
