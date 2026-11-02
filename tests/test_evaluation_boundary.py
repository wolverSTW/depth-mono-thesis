"""CPU-only boundary fixtures; no KITTI data or benchmark metric evidence."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from depth_monoia.evaluation import infer, validate_detections
from depth_monoia.model import upstream_path


def detections(batch=1, count=3):
    values = np.zeros((batch, count, 37), dtype=np.float32)
    values[..., 1] = 0.8
    values[..., 2:6] = 0.5
    values[..., 6] = 20.0
    values[..., 31:34] = 1.0
    values[..., 34:36] = 0.5
    values[..., 36] = 1.0
    return values


class EvaluationBoundaryTests(unittest.TestCase):
    def validate(self, values):
        return validate_detections(values, [42], 3, 0.2)

    def test_valid_positive_depth_passes_exactly_unchanged(self):
        values = detections(2)
        values[..., 6] = [[0.001, 20.0, 100.0], [80.0, 30.0, 1.0]]
        values[..., 1] = [0.9, 0.2, 0.8]  # Keep order, including equality.
        values[..., -1] = 2.0  # No confidence clipping or factor-based filter.
        for data in (values.copy(), torch.from_numpy(values.copy())):
            with self.subTest(kind=type(data).__name__):
                before = np.asarray(data).tobytes()
                self.assertIs(validate_detections(data, [42, 43], 3, 0.2), data)
                self.assertEqual(np.asarray(data).tobytes(), before)

    def assert_bad_depth(self, depth):
        values = detections()
        values[0, 1, 6] = depth
        before = values.tobytes()
        with self.assertRaisesRegex(ValueError, "image 000042: 1 invalid detections"):
            self.validate(values)
        self.assertEqual(values.tobytes(), before)

    def test_selected_zero_depth_is_rejected(self):
        self.assert_bad_depth(0.0)

    def test_selected_negative_depth_is_rejected(self):
        self.assert_bad_depth(-1.0)

    def test_selected_nonfinite_depth_is_rejected(self):
        for depth in (np.nan, np.inf, -np.inf):
            with self.subTest(depth=depth):
                self.assert_bad_depth(depth)

    def test_all_nonfinite_scores_are_rejected(self):
        for score in (np.nan, np.inf, -np.inf):
            with self.subTest(score=score):
                values = detections()
                values[0, 0, 1] = score
                with self.assertRaisesRegex(ValueError, "image 000042: 1 invalid detections"):
                    self.validate(values)

    def test_threshold_selection_matches_upstream_scalar_comparison(self):
        for dtype in (np.float32, np.float64):
            for threshold in (0.2, 0.7):
                center = dtype(threshold)
                for score in (np.nextafter(center, dtype(0)), center,
                              np.nextafter(center, dtype(1))):
                    with self.subTest(dtype=dtype, threshold=threshold, score=score):
                        values = detections().astype(dtype)
                        values[0, 0, 1] = score
                        values[0, 0, 6] = 0.0
                        if score < threshold:
                            self.assertIs(validate_detections(values, [42], 3, threshold), values)
                        else:
                            with self.assertRaisesRegex(ValueError, "image 000042: 1 invalid detections"):
                                validate_detections(values, [42], 3, threshold)

    def test_selected_non_depth_numeric_fields_are_rejected(self):
        for field in set(range(37)) - {1, 6}:
            for bad in (np.nan, np.inf, -np.inf):
                with self.subTest(field=field, bad=bad):
                    values = detections()
                    values[0, 0, field] = bad
                    with self.assertRaisesRegex(ValueError, "image 000042: 1 invalid detections"):
                        self.validate(values)

    def test_shape_contract_checks_batch_detection_and_field_dimensions(self):
        for shape in ((3, 37), (2, 3, 37), (1, 2, 37), (1, 3, 36), (1, 3, 38), (1, 3, 37, 1)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "shape for images 000042.*invalid detection count unknown"):
                    self.validate(np.zeros(shape))

    def test_invalid_counts_are_per_detection_and_per_image(self):
        values = detections(2)
        values[0, 0, [1, 6, 31]] = np.nan  # Three fields, one detection.
        values[0, 1, 6] = -1.0
        values[1, 2, 6] = 0.0
        with self.assertRaises(ValueError) as error:
            validate_detections(values, [42, 43], 3, 0.2)
        self.assertEqual(str(error.exception),
                         "Invalid KITTI detections (nonfinite score/selected fields or camera-Z <= 0); "
                         "image 000042: 2 invalid detections; image 000043: 1 invalid detections")

    def test_non_real_fields_and_nonfinite_threshold_fail_closed(self):
        for values in (detections().astype(complex), detections().astype(str)):
            with self.assertRaisesRegex(ValueError, "dtype for images 000042; 3 invalid detections"):
                self.validate(values)
        for threshold in (np.nan, np.inf, -np.inf):
            with self.assertRaisesRegex(ValueError, "threshold must be finite"):
                validate_detections(detections(), [42], 3, threshold)


class InferencePublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        upstream_path()
        from lib.helpers import decode_helper
        cls.upstream = decode_helper

    @staticmethod
    def calibration():
        # Unit-only decoder collaborator; never used for benchmark evaluation.
        return SimpleNamespace(
            img_to_rect=lambda x, y, z: np.array([[x, y, z]]),
            alpha2ry=lambda alpha, x: alpha)

    def test_below_threshold_finite_rows_preserve_pinned_decoder_semantics(self):
        values = detections()
        values[0, :, 1] = [np.nextafter(np.float32(0.2), np.float32(0)), 0.2, 0.9]
        values[0, :, 0] = [0, 1, 2]
        values[0, :, 6] = [-1.0, 2.0, 30.0]
        values[0, :, -1] = [1.0, 0.01, 2.0]
        info = {"img_id": np.array([42]), "img_size": np.array([[100, 60]])}
        args = (info, [self.calibration()], np.zeros((3, 3)), 0.2)
        expected = self.upstream.decode_detections(values.copy(), *args)
        before = values.tobytes()
        checked = validate_detections(values, [42], 3, 0.2)
        self.assertEqual(values.tobytes(), before)
        actual = self.upstream.decode_detections(checked.copy(), *args)
        self.assertEqual(actual, expected)
        self.assertEqual([row[0] for row in actual[42]], [1, 2])
        self.assertLess(actual[42][0][-1], 0.2)  # Final score is not the selection score.
        self.assertGreater(actual[42][1][-1], 1.0)

    def run_infer(self, destination, arrays, ids=None, decoder=None, extract=None):
        ids = ids or list(range(42, 42 + len(arrays)))
        batches = [(torch.zeros(1, 3, 2, 2), torch.zeros(1), {},
                    {"img_id": torch.tensor([image_id]), "img_size": torch.tensor([[100, 60]])})
                   for image_id in ids]
        loader = Mock()
        loader.__iter__ = Mock(return_value=iter(batches))
        loader.dataset = SimpleNamespace(max_objs=50, cls_mean_size=np.zeros((3, 3)),
                                         class_name=["Car", "Pedestrian", "Cyclist"],
                                         get_calib=lambda image_id: self.calibration())
        decoder = decoder or Mock(wraps=self.upstream.decode_detections)
        with patch.object(torch.Tensor, "cuda", lambda value: value), \
                patch.object(self.upstream, "extract_dets_from_outputs",
                             side_effect=extract or [torch.from_numpy(a) for a in arrays]), \
                patch.object(self.upstream, "decode_detections", decoder):
            infer(Mock(), loader, {"tester": {"topk": 3, "threshold": 0.2}}, destination)
        return decoder

    def test_existing_prediction_files_are_preserved_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "predictions"
            destination.mkdir()
            prior = destination / "000042.txt"
            prior.write_bytes(b"existing evidence sentinel\n")
            bad = detections()
            bad[..., 6] = 0.0
            decoder = Mock()
            with self.assertRaises(FileExistsError):
                self.run_infer(destination, [bad], decoder=decoder)
            decoder.assert_not_called()
            self.assertEqual(prior.read_bytes(), b"existing evidence sentinel\n")
            self.assertEqual(list(destination.iterdir()), [prior])

    def test_existing_empty_prediction_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "predictions"
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                self.run_infer(destination, [detections()])
            self.assertTrue(destination.is_dir())
            self.assertEqual(list(destination.iterdir()), [])

    def test_late_validation_failure_never_publishes_partial_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prior = root / "prior_predictions"
            prior.mkdir()
            evidence = prior / "000042.txt"
            evidence.write_bytes(b"prior evidence sentinel\n")
            bad = detections()
            bad[0, 2, 6] = 0.0
            decoder = Mock(wraps=self.upstream.decode_detections)
            with self.assertRaisesRegex(ValueError, "image 000043: 1 invalid detections"):
                self.run_infer(root / "new_predictions", [detections(), bad], decoder=decoder)
            self.assertEqual(decoder.call_count, 1)  # Invalid batch never reaches decoder.
            self.assertEqual(evidence.read_bytes(), b"prior evidence sentinel\n")
            self.assertEqual(list(root.iterdir()), [prior])

    def test_success_publishes_original_format_only_after_all_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "predictions"
            def decode(*args):
                self.assertFalse(destination.exists())
                return self.upstream_decoder(*args)
            self.upstream_decoder = self.upstream.decode_detections
            decoder = self.run_infer(destination, [detections(), detections()], decoder=Mock(side_effect=decode))
            self.assertEqual(decoder.call_count, 2)
            self.assertEqual(sorted(p.name for p in destination.iterdir()), ["000042.txt", "000043.txt"])
            rows = self.upstream_decoder(detections(), {"img_id": [42], "img_size": [[100, 60]]},
                                         [self.calibration()], np.zeros((3, 3)), 0.2)[42]
            expected = "".join("Car 0.0 0" + "".join(f" {v:.2f}" for v in row[1:]) + "\n" for row in rows)
            self.assertEqual((destination / "000042.txt").read_text(), expected)

    def test_destination_created_during_inference_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "predictions"
            def extract(*args, **kwargs):
                destination.mkdir()
                (destination / "sentinel").write_bytes(b"concurrent evidence")
                return torch.from_numpy(detections())
            with self.assertRaises(FileExistsError):
                self.run_infer(destination, [detections()], extract=extract)
            self.assertEqual((destination / "sentinel").read_bytes(), b"concurrent evidence")
            self.assertEqual(list(Path(temporary).iterdir()), [destination])


if __name__ == "__main__":
    unittest.main()
