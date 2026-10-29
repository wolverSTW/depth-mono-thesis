"""Acquisition safety tests use project source bytes, not fake KITTI samples."""
from pathlib import Path
import shutil
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from depth_monoia.config import ROOT
from depth_monoia.kitti_download import download, extract_verified_zip, https_url, prepare
from depth_monoia.provenance import sha256


class AcquisitionSafety(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = ROOT / "README.md"

    def tearDown(self):
        self.temporary.cleanup()

    def test_verified_cache_skips_network_and_rejects_changed_bytes(self):
        cached = self.root / "README.md"
        shutil.copyfile(self.source, cached)
        spec = {"sha256": sha256(self.source), "sha256_source": "Existing repository README bytes"}
        with patch("depth_monoia.kitti_download.subprocess.run") as network:
            self.assertEqual(download(spec, cached), cached)
            network.assert_not_called()
        cached.write_bytes(cached.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Existing file fails SHA-256"):
            download(spec, cached)

    def test_source_zip_roundtrip_rerun_and_conflict(self):
        archive = self.root / "source.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.write(self.source, "docs/README.md")
        destination = self.root / "extracted"
        first = extract_verified_zip(archive, destination)
        self.assertEqual(first["new_files"], 1)
        self.assertEqual((destination / "docs/README.md").read_bytes(), self.source.read_bytes())
        self.assertEqual(extract_verified_zip(archive, destination)["new_files"], 0)
        (destination / "docs/README.md").write_bytes(b"corrupt source-file copy")
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            extract_verified_zip(archive, destination)

    def test_zip_traversal_alias_and_symlink_rejected_before_writes(self):
        for name in ("../escape", "./alias", "C:/escape", "link"):
            archive = self.root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.write(self.source, "README.md")
                entry = zipfile.ZipInfo(name)
                if name == "link":
                    entry.create_system = 3
                    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                handle.writestr(entry, self.source.read_bytes())
            target = self.root / "rejected"
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                extract_verified_zip(archive, target)
            self.assertFalse((target / "README.md").exists())

    def test_wrong_kitti_archive_layout_is_rejected(self):
        archive = self.root / "source.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.write(self.source, "README.md")
        with self.assertRaisesRegex(ValueError, "official KITTI"):
            extract_verified_zip(archive, self.root / "rejected", "image_2")

    def test_missing_provenance_fails_before_network_or_dataset_creation(self):
        with patch("depth_monoia.kitti_download.download") as network:
            with self.assertRaisesRegex(ValueError, "authorization"):
                prepare(ROOT / "configs/vast/kitti_sources.example.json", self.root / "dataset",
                        self.root / "downloads", self.root / "manifest.json")
            network.assert_not_called()
        self.assertFalse((self.root / "dataset").exists())

    def test_plaintext_or_embedded_password_urls_rejected(self):
        for url in ("http://www.cvlibs.net/private", "https://user:password@www.cvlibs.net/private", "file:///private"):
            with self.assertRaisesRegex(ValueError, "HTTPS"):
                https_url(url)

    def test_transfer_hash_mismatch_retains_partial_without_leaking_url_to_argv(self):
        url = "https://www.cvlibs.net/authorized-download?session=local-test-only"
        target = self.root / "source.txt"
        spec = {"url": url, "sha256": sha256(self.source), "sha256_source": "Existing repository README bytes"}
        def transferred(command, **kwargs):
            self.assertNotIn(url, command)
            self.assertIn(url, kwargs["input"])
            # Different genuine source bytes; not a KITTI/model fixture.
            Path(command[command.index("--output") + 1]).write_bytes((ROOT / "AGENTS.md").read_bytes())
            return SimpleNamespace(returncode=0)
        with patch("depth_monoia.kitti_download.shutil.which", return_value="curl"), patch("depth_monoia.kitti_download.subprocess.run", side_effect=transferred):
            with self.assertRaisesRegex(ValueError, "Downloaded SHA-256 mismatch"):
                download(spec, target)
        self.assertFalse(target.exists())
        self.assertTrue(target.with_name("source.txt.part").is_file())
