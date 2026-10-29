"""Operational tests use real source-file bytes, never dummy KITTI or metrics."""
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from depth_monoia.artifacts import RunLock, archive_files, restore_archive
from depth_monoia.config import ROOT, resolve_config
from depth_monoia.provenance import sha256
from depth_monoia.reporting import aggregate


class Operations(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "source"
        self.run.mkdir()
        self.source = self.run / "README.md"
        shutil.copyfile(ROOT / "README.md", self.source)
        self.archive = self.root / "snapshot.tar.gz"

    def tearDown(self):
        self.temp.cleanup()

    def archive_source(self):
        return archive_files(self.run, [self.source], self.archive, {"kind": "source-file operational test"})

    def test_archive_roundtrip_and_no_overwrite(self):
        self.archive_source()
        destination = self.root / "restored"
        result = restore_archive(self.archive, destination)
        self.assertTrue(result["verified"])
        self.assertEqual(self.source.read_bytes(), (destination / "README.md").read_bytes())
        with self.assertRaises(FileExistsError):
            restore_archive(self.archive, destination)
        with self.assertRaises(FileExistsError):
            self.archive_source()

    def test_corrupt_archive_is_rejected_before_restore(self):
        self.archive_source()
        with self.archive.open("ab") as handle:
            handle.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            restore_archive(self.archive, self.root / "must_not_exist")
        self.assertFalse((self.root / "must_not_exist").exists())

    def test_member_traversal_and_links_are_rejected(self):
        for attack in ("traversal", "symlink"):
            archive = self.root / (attack + ".tar.gz")
            with tarfile.open(archive, "w:gz") as tar:
                payload = json.dumps({"schema": 1, "files": {}}).encode()
                info = tarfile.TarInfo("SNAPSHOT.json")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
                bad = tarfile.TarInfo("run/../../escape" if attack == "traversal" else "run/link")
                if attack == "symlink":
                    bad.type, bad.linkname = tarfile.SYMTYPE, "../../escape"
                tar.addfile(bad)
            archive.with_name(archive.name + ".sha256").write_text(sha256(archive))
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                restore_archive(archive, self.root / attack)
        self.assertFalse((self.root / "escape").exists())

    def test_run_lock_excludes_another_process(self):
        code = "from depth_monoia.artifacts import RunLock\nimport sys\ntry:\n with RunLock(sys.argv[1], timeout=0): pass\nexcept TimeoutError:\n sys.exit(0)\nsys.exit(1)"
        with RunLock(self.run):
            process = subprocess.run([sys.executable, "-B", "-c", code, str(self.run)], cwd=ROOT, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr.decode())

    def test_overrides_are_explicit_and_roundtrip(self):
        cfg = resolve_config(ROOT / "configs/proposed/A3.yaml", overrides={"batch_size": 8, "workers": 0, "seed": 445})
        self.assertEqual(cfg["execution"]["protocol_changes"]["batch_size"], {"original": 16, "resolved": 8})
        self.assertEqual(cfg["execution"]["workers"], 0)
        import yaml
        saved = self.root / "config.yaml"
        saved.write_text(yaml.safe_dump(cfg))
        self.assertEqual(resolve_config(saved, overrides={"batch_size": 8, "workers": 0, "seed": 445}), cfg)
        with self.assertRaises(ValueError):
            resolve_config(saved, overrides={"workers": -1})

    def test_no_evaluations_produce_only_missing_measurements(self):
        output = self.root / "comparison.csv"
        aggregate(self.run, output)
        with output.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row["evaluation_status"] == "not_evaluated" and row["AP3D_Moderate"] == "" for row in rows))
