"""Published split enforcement; cached real lists are not KITTI G1 evidence."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from depth_monoia.config import ROOT
from depth_monoia.kitti_download import prepare
from depth_monoia.provenance import split_protocol, verify_published_splits


class PublishedProtocol(unittest.TestCase):
    def test_undefined_protocol_cannot_be_silently_relabelled(self):
        for protocol in (None, "original_monoia", "random", ""):
            with self.assertRaisesRegex(ValueError, "explicitly approved"):
                split_protocol(protocol)

    def test_source_overrides_fail_before_network_or_dataset_creation(self):
        spec = json.loads((ROOT / "configs/vast/kitti_sources.example.json").read_text())
        spec["authorized_official_downloads"] = True
        spec["splits"] = {"train": {"sha256": "0" * 64}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sources.json"
            source.write_text(json.dumps(spec))
            with patch("depth_monoia.kitti_download._transfer") as network:
                with self.assertRaisesRegex(ValueError, "redefine the published split"):
                    prepare(source, root / "dataset", root / "cache", root / "manifest.json")
                network.assert_not_called()
            self.assertFalse((root / "dataset").exists())

    def real_lists(self):
        directory = ROOT / "data/protocols/kitti_3dop_chen"
        if not all((directory / (split + ".txt")).is_file() for split in ("train", "val")):
            self.skipTest("Real pinned split-byte cache absent; no network or fabricated lists used")
        return directory

    def test_real_pinned_bytes_counts_membership_and_order(self):
        checked = verify_published_splits(self.real_lists())
        self.assertEqual(checked["splits"]["train"]["count"], 3712)
        self.assertEqual(checked["splits"]["val"]["count"], 3769)

    def test_same_membership_reordered_or_relabelled_is_rejected(self):
        source = self.real_lists()
        protocol = split_protocol("kitti_3dop_chen")
        manifest = {"protocol": protocol["protocol"], "split_source": protocol["split_source"],
                    **{name: {key: entry[key] for key in ("count", "sha256")}
                       for name, entry in protocol["splits"].items()}}
        manifest["train"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "redefine the published split"):
            verify_published_splits(source, manifest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(source / "val.txt", root / "val.txt")
            ids = (source / "train.txt").read_bytes().split(b"\n")
            (root / "train.txt").write_bytes(b"\n".join(reversed(ids)))
            with self.assertRaisesRegex(ValueError, "raw-byte/order SHA-256 mismatch"):
                verify_published_splits(root)
