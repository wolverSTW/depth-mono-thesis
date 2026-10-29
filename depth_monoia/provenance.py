"""Fail-closed data, source and checkpoint provenance."""
import hashlib
import json
import platform
import re
import subprocess
from importlib import metadata
from pathlib import Path
from .config import ROOT


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", "-c", "core.excludesFile=", "-c",
                                   "safe.directory=" + Path(cwd).resolve().as_posix(), *args],
                                   cwd=cwd, text=True, stderr=subprocess.PIPE).strip()


def check_sources():
    spec = json.loads((ROOT / "docs/reproducibility/sources.json").read_text())
    result = {}
    for folder, source in spec["sources"].items():
        actual = git("rev-parse", "HEAD", cwd=ROOT / folder)
        if actual != source["commit"] or git("status", "--porcelain", cwd=ROOT / folder):
            raise ValueError("Modified/unpinned upstream: " + folder)
        result[folder] = actual
    return result


def checkpoint_info(name):
    return json.loads((ROOT / "docs/reproducibility/sources.json").read_text())["checkpoints"][name]


def verify_checkpoint(name, path=None):
    spec = checkpoint_info(name)
    path = Path(path or ROOT / spec["path"])
    actual = sha256(path)
    if actual != spec["sha256"]:
        raise ValueError("Checkpoint SHA-256 mismatch: " + name)
    return {"source": spec["url"], "sha256": actual, "bytes": path.stat().st_size}


def split_ids(path):
    ids = Path(path).read_text().splitlines()
    if not ids or any(not re.fullmatch(r"\d{6}", item) for item in ids):
        raise ValueError("Expected nonempty six-digit KITTI split: " + str(path))
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate split IDs: " + str(path))
    return ids


def verify_dataset(root, manifest):
    root = Path(root)
    manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if not manifest.get("split_source") or "REQUIRED" in manifest["split_source"]:
        raise ValueError("Verified split_source is required")
    result, all_ids = {}, {}
    content = hashlib.sha256()
    for split in ("train", "val"):
        path = root / "ImageSets" / (split + ".txt")
        ids = split_ids(path)
        digest = sha256(path)
        if len(ids) != manifest[split]["count"] or digest != manifest[split]["sha256"]:
            raise ValueError("Split count/hash mismatch: " + split)
        for sample_id in ids:
            for folder, suffix in (("image_2", ".png"), ("calib", ".txt"), ("label_2", ".txt")):
                sample = root / "training" / folder / (sample_id + suffix)
                if not sample.is_file() or sample.stat().st_size == 0:
                    raise FileNotFoundError(sample)
                # Annotation/calibration changes invalidate prior gate evidence
                # and resume identity. Images get size inventory, not 12GB rehash.
                content.update((sample.relative_to(root).as_posix() + ":" + str(sample.stat().st_size)).encode())
                if folder != "image_2":
                    content.update(bytes.fromhex(sha256(sample)))
        result[split] = {"count": len(ids), "sha256": digest}
        all_ids[split] = set(ids)
    if all_ids["train"] & all_ids["val"]:
        raise ValueError("Training/validation leakage: overlapping IDs")
    if all_ids["train"] | all_ids["val"] != {f"{i:06d}" for i in range(7481)}:
        raise ValueError("Final experiments require the complete designated KITTI split")
    return {"splits": result, "split_source": manifest["split_source"],
            "dataset_source": manifest["source"], "annotation_calibration_inventory_sha256": content.hexdigest(),
            "image_verification": "split coverage and byte sizes; image-content authentication remains operator responsibility"}


def environment():
    import torch
    packages = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "Pillow", "PyYAML",
                 "opencv-python", "scikit-image", "scikit-learn", "numba", "llvmlite"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "missing"
    system_tools = {}
    if torch.cuda.is_available():
        for name, command in {"nvidia_driver_inventory": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                              "cuda_toolkit": ["nvcc", "--version"], "compiler": ["c++", "--version"]}.items():
            try:
                system_tools[name] = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
            except (OSError, subprocess.SubprocessError) as error:
                system_tools[name] = "Unavailable: " + type(error).__name__
    return {"platform": platform.platform(), "python": platform.python_version(), "system_tools": system_tools,
            "packages": packages, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain"))}
