"""Authorized official KITTI acquisition; no embedded download URLs or split guesses."""
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
from urllib.parse import urlsplit
import zipfile
import zlib

from .artifacts import RunLock, write_json
from .provenance import sha256, split_ids, verify_dataset

OFFICIAL_SOURCE = "https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d"
ARCHIVES = {"image_2": "data_object_image_2.zip", "calib": "data_object_calib.zip",
            "label_2": "data_object_label_2.zip"}


def required_text(value, label):
    if not isinstance(value, str) or not value.strip() or "REQUIRED" in value:
        raise ValueError("Missing trusted " + label + "; fill local.kitti_sources.json first")
    return value


def digest_spec(spec, label, required=True):
    digest = spec.get("sha256", "")
    if not required and digest in (None, ""):
        return None
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise ValueError("A trusted SHA-256 is required for " + label)
    required_text(spec.get("sha256_source"), label + " SHA-256 provenance")
    return digest.lower()


def https_url(value):
    value = required_text(value, "authorized download URL")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or any(ord(c) < 32 for c in value):
        raise ValueError("Download URLs must use HTTPS without embedded login credentials")
    return value


def download(spec, target, auth=None, require_digest=True):
    """Resume into .part; authenticate bytes before publication. Never log signed URLs."""
    target = Path(target)
    expected = digest_spec(spec, target.name, required=require_digest)
    if target.exists():
        if expected is None:
            raise ValueError("Existing archive has no verification receipt or expected hash; supply its trusted SHA-256 before reuse: " + str(target))
        if not target.is_file() or target.is_symlink() or sha256(target) != expected:
            raise ValueError("Existing file fails SHA-256; preserve/investigate it before retrying: " + str(target))
        print("Verified cached file: " + target.name)
        return target
    url = https_url(spec.get("url"))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if partial.is_symlink():
        raise ValueError("Refusing a symlink download destination")
    if expected and partial.exists() and sha256(partial) == expected:
        os.replace(partial, target)
        return target
    # curl's config goes over stdin, so signed URL/cookie paths are not in argv
    # or console logs. Only explicit cookie/netrc authentication is supported.
    def quoted(value):
        if any(ord(c) < 32 for c in value):
            raise ValueError("Control character in download configuration")
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'
    settings = ["url = " + quoted(url)]
    for key, option in (("cookie_file", "cookie"), ("netrc_file", "netrc-file")):
        if (auth or {}).get(key):
            path = Path(auth[key])
            if not path.is_file():
                raise FileNotFoundError("Authentication file is unavailable: " + str(path))
            settings.append(option + " = " + quoted(str(path)))
            if key == "netrc_file":
                settings.append("netrc")
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required; run scripts/vast/setup.sh first")
    print("Downloading authorized file: " + target.name)
    completed = subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                                "--retry", "3", "--connect-timeout", "30", "--proto", "=https",
                                "--proto-redir", "=https", "--continue-at", "-", "--output", str(partial),
                                "--config", "-"], input="\n".join(settings) + "\n", text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"Download failed for {target.name} (curl exit {completed.returncode}); check authorization, URL expiry and server range support. Partial bytes retained. URL/credentials suppressed.")
    if expected and sha256(partial) != expected:
        raise ValueError("Downloaded SHA-256 mismatch for " + target.name + "; partial file retained, no extraction performed")
    os.replace(partial, target)
    return target


def checked_members(archive):
    """Validate every ZIP member before writing any; ZIP payloads need not be KITTI."""
    seen, members = set(), []
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        mode = member.external_attr >> 16
        canonical = path.as_posix()
        if (not path.parts or path.is_absolute() or ".." in path.parts or "\\" in member.filename
                or member.filename.rstrip("/") != canonical or ":" in member.filename or canonical in seen
                or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR))):
            raise ValueError("Unsafe/duplicate ZIP member: " + member.filename)
        seen.add(canonical)
        members.append(member)
    return members


def crc32_file(path):
    crc = 0
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            crc = zlib.crc32(block, crc)
    return crc & 0xffffffff


def extract_verified_zip(path, root, kind=None, verify_crc=True):
    """CRC-check ZIP before extraction; atomically publish files, preserve conflicts."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        members = checked_members(archive)
        files = [member for member in members if not member.is_dir()]
        if not files:
            raise ValueError("Archive has no regular files")
        if kind:
            suffix = "png" if kind == "image_2" else "txt"
            splits = "training" if kind == "label_2" else "(?:training|testing)"
            pattern = rf"{splits}/{kind}/\d{{6}}\.{suffix}"
            if any(not re.fullmatch(pattern, member.filename) for member in files):
                raise ValueError("Archive structure differs from official KITTI " + kind)
        # Reject escaping directory links and conflicts before writing anything.
        pending = []
        for member in files:
            target = root / member.filename
            if root not in target.resolve().parents or target.is_symlink():
                raise ValueError("Unsafe extraction destination: " + member.filename)
            if target.exists():
                if not target.is_file() or target.stat().st_size != member.file_size or crc32_file(target) != member.CRC:
                    raise ValueError("Existing extracted file differs; refusing overwrite: " + str(target))
            else:
                pending.append(member)
        required = sum(member.file_size for member in pending)
        if shutil.disk_usage(root).free < required + 1024**3:
            raise OSError("Insufficient disk space for archive extraction plus 1 GiB headroom")
        bad = archive.testzip() if verify_crc else None
        if bad:
            raise ValueError("ZIP CRC failure before extraction: " + bad)
        for member in pending:
            target = root / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".kitti-", dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as output, archive.open(member) as source:
                    shutil.copyfileobj(source, output)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return {"files": len(files), "new_files": len(pending), "uncompressed_bytes": sum(m.file_size for m in files)}


def prepare(source_spec, dataset_root, download_root, manifest_path):
    spec_path = Path(source_spec).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("official_source") != OFFICIAL_SOURCE or spec.get("authorized_official_downloads") is not True:
        raise ValueError("Confirm official KITTI acquisition/authorization in the source specification; no mirror or login bypass is supplied")
    split_source = required_text(spec.get("split_source"), "original MonoIA split provenance")
    # Validate all provenance before spending time/bandwidth on large archives.
    for kind in ARCHIVES:
        entry = spec.get("archives", {}).get(kind, {})
        required_text(entry.get("download_source"), kind + " official URL provenance")
        digest_spec(entry, kind, required=False)
    for split, count in (("train", 3712), ("val", 3769)):
        entry = spec.get("splits", {}).get(split, {})
        digest_spec(entry, split)
        if entry.get("count") != count:
            raise ValueError("Original designated split count required: " + split)
    root, cache = Path(dataset_root).resolve(), Path(download_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    auth = dict(spec.get("authentication", {}))
    for key in ("cookie_file", "netrc_file"):
        if auth.get(key):
            auth[key] = str((spec_path.parent / auth[key]).resolve())
    with RunLock(cache, name=".kitti-preparation.lock", timeout=0):
        receipt_path = cache / "verified_archives.json"
        receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        splits = {}
        for split in ("train", "val"):
            entry = spec["splits"][split]
            if entry.get("path"):
                source = (spec_path.parent / entry["path"]).resolve()
                if sha256(source) != entry["sha256"].lower():
                    raise ValueError("Original split SHA-256 mismatch: " + split)
            else:
                source = download(entry, cache / (split + ".txt"), auth)
            ids = split_ids(source)
            if len(ids) != entry["count"]:
                raise ValueError("Original split count mismatch: " + split)
            splits[split] = (source, set(ids))
        if splits["train"][1] & splits["val"][1] or splits["train"][1] | splits["val"][1] != {f"{i:06d}" for i in range(7481)}:
            raise ValueError("Split leakage or incomplete designated KITTI coverage")
        archives, extraction = {}, {}
        # Authenticate/CRC-check ALL archives before extracting any of them.
        for kind, name in ARCHIVES.items():
            entry = dict(spec["archives"][kind])
            expected = digest_spec(entry, kind, required=False)
            if expected is None and name in receipts:
                entry.update(sha256=receipts[name]["sha256"], sha256_source="Previous verified official acquisition receipt")
            path = download(entry, cache / name, auth, require_digest=False)
            with zipfile.ZipFile(path) as archive:
                checked_members(archive)
                if archive.testzip():
                    raise ValueError("ZIP CRC failed: " + name)
            receipts[name] = {"sha256": sha256(path), "bytes": path.stat().st_size, "zip_crc": "PASS",
                              "expected_sha256_supplied": expected is not None,
                              "download_source": spec["archives"][kind]["download_source"]}
            # Publish after each completed download; an interrupted next archive
            # must not force downloading an already checked 12 GB file again.
            write_json(receipt_path, receipts)
            archives[kind] = path
        for kind, path in archives.items():
            extraction[kind] = extract_verified_zip(path, root, kind, verify_crc=False)
        split_dir = root / "ImageSets"
        split_dir.mkdir(exist_ok=True)
        for split, (source, _) in splits.items():
            target = split_dir / (split + ".txt")
            if root not in target.resolve().parents or target.is_symlink():
                raise ValueError("Unsafe split destination")
            if target.exists() and sha256(target) != sha256(source):
                raise ValueError("Existing split differs; refusing overwrite: " + split)
            if not target.exists():
                fd, temporary = tempfile.mkstemp(prefix=".split-", dir=split_dir)
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(source.read_bytes())
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, target)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
        manifest = {"dataset": "KITTI 3D Object Detection", "source": OFFICIAL_SOURCE, "split_source": split_source,
                    **{split: {"count": spec["splits"][split]["count"], "sha256": sha256(splits[split][0])} for split in splits},
                    "archives": {kind: receipts[path.name] for kind, path in archives.items()},
                    "acquisition": "Operator-authorized official HTTPS downloads; ZIP CRC and SHA-256 recorded before extraction. Expected archive hashes compared when supplied; recorded hashes alone do not independently authenticate a source. Signed URLs/authentication excluded."}
        destination = Path(manifest_path).resolve()
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if any(existing.get(key) != manifest[key] for key in ("source", "split_source", "train", "val")):
                raise ValueError("Existing manifest identity differs; preserve it and choose a new destination")
            if "archives" in existing and existing["archives"] != manifest["archives"]:
                raise ValueError("Existing archive provenance differs; refusing overwrite")
            manifest = {**existing, **manifest}  # Preserve manually added fields.
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            candidate = Path(temporary) / "manifest.json"
            write_json(candidate, manifest)
            verified = verify_dataset(root, candidate)
        write_json(destination, manifest)
        print(json.dumps({"dataset_root": str(root), "manifest": str(destination), "verification": verified,
                          "extraction": extraction}, indent=2))
        return manifest
