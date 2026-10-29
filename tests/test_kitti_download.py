"""Acquisition safety tests use project source bytes, not fake KITTI samples."""
from pathlib import Path
import json
import shutil
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from depth_monoia.config import ROOT
from depth_monoia import kitti_download as acquisition
from depth_monoia.kitti_download import download, download_archive, extract_verified_zip, https_url, prepare
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


class ArchiveRecovery(unittest.TestCase):
    """ZIPs contain repository README bytes only; no KITTI sample fixtures."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.zip"
        with zipfile.ZipFile(self.source, "w") as archive:
            archive.write(ROOT / "README.md", "README.md")
        self.target = self.root / "cache/archive.zip"
        self.receipts = self.root / "cache/verified_archives.json"
        self.spec = {"url": "https://example.org/source-byte-test-only.zip",
                     "download_source": "Offline repository-source ZIP fixture"}

    def tearDown(self):
        self.temporary.cleanup()

    def transfer(self, spec, partial, auth=None):
        shutil.copyfile(self.source, partial)

    def fetch(self):
        return download_archive(self.spec, self.target, self.receipts)

    def receipt(self):
        return json.loads(self.receipts.read_text())[self.target.name]

    def test_crc_failure_never_publishes_final_file_or_receipt(self):
        # Damage the first stored README byte, keeping a readable ZIP directory.
        content = bytearray(self.source.read_bytes())
        with zipfile.ZipFile(self.source) as archive:
            info = archive.getinfo("README.md")
            offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
        content[offset] ^= 1
        self.source.write_bytes(content)
        with patch.object(acquisition, "_transfer", side_effect=self.transfer):
            with self.assertRaisesRegex(ValueError, "ZIP CRC failed before publication"):
                self.fetch()
        self.assertFalse(self.target.exists())
        self.assertFalse(self.receipts.exists())
        self.assertTrue(self.target.with_name("archive.zip.part").is_file())

    def test_interruption_after_receipt_before_publication_recovers_without_network(self):
        with patch.object(acquisition, "_transfer", side_effect=self.transfer), patch.object(
                acquisition, "_publish_file", side_effect=InterruptedError("publication interrupted")):
            with self.assertRaises(InterruptedError):
                self.fetch()
        self.assertFalse(self.target.exists())
        self.assertEqual(self.receipt()["state"], "verified_pending_publish")
        with patch.object(acquisition, "_transfer") as network:
            self.fetch()
            network.assert_not_called()
        self.assertEqual(self.receipt()["state"], "complete")
        self.assertEqual(self.target.read_bytes(), self.source.read_bytes())

    def test_interruption_after_publication_before_complete_receipt_recovers(self):
        write = acquisition.write_json
        def interrupted(path, receipts):
            if receipts[self.target.name]["state"] == "complete":
                raise InterruptedError("receipt completion interrupted")
            write(path, receipts)
        with patch.object(acquisition, "_transfer", side_effect=self.transfer), patch.object(
                acquisition, "write_json", side_effect=interrupted):
            with self.assertRaises(InterruptedError):
                self.fetch()
        self.assertTrue(self.target.is_file())
        self.assertEqual(self.receipt()["state"], "verified_pending_publish")
        with patch.object(acquisition, "_transfer") as network:
            self.fetch()
            self.fetch()  # Ordinary complete-cache rerun also avoids network.
            network.assert_not_called()
        self.assertEqual(self.receipt()["state"], "complete")
        self.assertEqual(self.target.read_bytes(), self.source.read_bytes())

    def test_unknown_final_cache_is_rejected_not_silently_adopted(self):
        self.target.parent.mkdir()
        shutil.copyfile(self.source, self.target)
        with patch.object(acquisition, "_transfer") as network:
            with self.assertRaisesRegex(ValueError, "Unknown cached archive"):
                self.fetch()
            network.assert_not_called()
        self.assertFalse(self.receipts.exists())
        self.assertEqual(self.target.read_bytes(), self.source.read_bytes())

    def test_verified_final_with_conflicting_staging_preserves_both_and_receipt(self):
        with patch.object(acquisition, "_transfer", side_effect=self.transfer):
            self.fetch()
        partial = self.target.with_name("archive.zip.part")
        # Another valid source-byte ZIP: CRC alone must not authorize its reuse.
        with zipfile.ZipFile(partial, "w") as archive:
            archive.write(ROOT / "AGENTS.md", "AGENTS.md")
        final_bytes, staged_bytes = self.target.read_bytes(), partial.read_bytes()
        with zipfile.ZipFile(partial) as archive:
            self.assertIsNone(archive.testzip())
        entry = self.receipt()
        for state in ("complete", "verified_pending_publish", None):
            with self.subTest(receipt_state=state or "legacy"):
                current = dict(entry)
                if state is None:
                    current.pop("state")
                    current.pop("schema")
                else:
                    current["state"] = state
                acquisition.write_json(self.receipts, {self.target.name: current})
                receipt_bytes = self.receipts.read_bytes()
                with patch.object(acquisition, "_transfer") as network:
                    with self.assertRaisesRegex(ValueError, "Staged archive conflicts"):
                        self.fetch()
                    network.assert_not_called()
                self.assertEqual(self.target.read_bytes(), final_bytes)
                self.assertEqual(partial.read_bytes(), staged_bytes)
                self.assertEqual(self.receipts.read_bytes(), receipt_bytes)

    def test_verified_final_with_identical_staging_recovers_for_valid_receipts(self):
        with patch.object(acquisition, "_transfer", side_effect=self.transfer):
            self.fetch()
        partial = self.target.with_name("archive.zip.part")
        shutil.copyfile(self.target, partial)  # Separate files, identical verified bytes.
        entry = self.receipt()
        original = self.target.read_bytes()
        for state in ("complete", "verified_pending_publish", None):
            with self.subTest(receipt_state=state or "legacy"):
                current = dict(entry)
                if state is None:
                    current.pop("state")
                    current.pop("schema")
                else:
                    current["state"] = state
                acquisition.write_json(self.receipts, {self.target.name: current})
                with patch.object(acquisition, "_transfer") as network:
                    self.fetch()
                    network.assert_not_called()
                self.assertEqual(self.receipt()["state"], "complete")
                self.assertEqual(self.target.read_bytes(), original)
                self.assertEqual(partial.read_bytes(), original)

    def test_interruption_between_link_and_staging_unlink_recovers(self):
        link = acquisition.os.link
        def interrupted(partial, target):
            link(partial, target)
            raise InterruptedError("staging unlink not reached")
        with patch.object(acquisition, "_transfer", side_effect=self.transfer), patch.object(
                acquisition.os, "link", side_effect=interrupted):
            with self.assertRaises(InterruptedError):
                self.fetch()
        self.assertTrue(self.target.with_name("archive.zip.part").is_file())
        self.assertEqual(self.receipt()["state"], "verified_pending_publish")
        self.assertEqual(self.target.with_name("archive.zip.part").read_bytes(), self.source.read_bytes())
        with patch.object(acquisition, "_transfer") as network:
            self.fetch()
            network.assert_not_called()
        self.assertEqual(self.receipt()["state"], "complete")
        self.assertEqual(self.target.read_bytes(), self.source.read_bytes())
        self.assertEqual(self.target.with_name("archive.zip.part").read_bytes(), self.source.read_bytes())

    def test_legacy_verified_receipt_reuses_cache_but_rejects_hash_override(self):
        self.target.parent.mkdir()
        shutil.copyfile(self.source, self.target)
        entry = {"sha256": sha256(self.source), "bytes": self.source.stat().st_size,
                 "zip_crc": "PASS", "expected_sha256_supplied": False,
                 "download_source": self.spec["download_source"]}
        acquisition.write_json(self.receipts, {self.target.name: entry})
        with patch.object(acquisition, "_transfer") as network:
            self.fetch()
            network.assert_not_called()
        self.assertFalse(self.receipt()["expected_sha256_supplied"])
        self.spec.update(sha256="0" * 64, sha256_source="Conflicting fixture expectation")
        with self.assertRaisesRegex(ValueError, "conflicts with its verification receipt"):
            self.fetch()
        self.assertEqual(self.target.read_bytes(), self.source.read_bytes())

    def test_trusted_expected_hash_allows_unknown_cache_after_crc_check(self):
        self.target.parent.mkdir()
        shutil.copyfile(self.source, self.target)
        self.spec.update(sha256=sha256(self.source), sha256_source="Known source-byte fixture")
        with patch.object(acquisition, "_transfer") as network:
            self.fetch()
            network.assert_not_called()
        self.assertTrue(self.receipt()["expected_sha256_supplied"])
        self.assertEqual(self.receipt()["zip_crc"], "PASS")

    def test_receipt_conflicts_reject_changed_final_and_staged_bytes(self):
        with patch.object(acquisition, "_transfer", side_effect=self.transfer), patch.object(
                acquisition, "_publish_file", side_effect=InterruptedError()):
            with self.assertRaises(InterruptedError):
                self.fetch()
        partial = self.target.with_name("archive.zip.part")
        partial.write_bytes((ROOT / "AGENTS.md").read_bytes())
        before = partial.read_bytes()
        receipt_before = self.receipts.read_bytes()
        self.assertEqual(self.receipt()["state"], "verified_pending_publish")
        with patch.object(acquisition, "_transfer") as network:
            with self.assertRaisesRegex(ValueError, "Staged archive conflicts"):
                self.fetch()
            network.assert_not_called()
        self.assertEqual(partial.read_bytes(), before)
        self.assertEqual(self.receipts.read_bytes(), receipt_before)
        self.assertFalse(self.target.exists())
        # Isolate the final-file conflict from the already-tested staging conflict.
        partial.write_bytes(self.source.read_bytes())
        self.target.write_bytes(before)
        with self.assertRaisesRegex(ValueError, "Archive SHA-256/size mismatch"):
            self.fetch()
        self.assertEqual(self.target.read_bytes(), before)

    def test_racing_conflicting_destination_is_never_overwritten(self):
        publish = acquisition._publish_file
        other = (ROOT / "AGENTS.md").read_bytes()
        def raced(partial, target):
            target.write_bytes(other)
            publish(partial, target)
        with patch.object(acquisition, "_transfer", side_effect=self.transfer), patch.object(
                acquisition, "_publish_file", side_effect=raced):
            with self.assertRaises(FileExistsError):
                self.fetch()
        self.assertEqual(self.target.read_bytes(), other)
        self.assertEqual(self.receipt()["state"], "verified_pending_publish")

    def test_unsafe_archive_members_fail_before_publication(self):
        with zipfile.ZipFile(self.source, "w") as archive:
            archive.write(ROOT / "README.md", "../README.md")
        with patch.object(acquisition, "_transfer", side_effect=self.transfer):
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                self.fetch()
        self.assertFalse(self.target.exists())
        self.assertFalse(self.receipts.exists())
