"""Coordinated run writes and verified, non-overwriting artifact snapshots."""
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import threading
import time
from .provenance import sha256


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class RunLock:
    """Process-safe advisory lock; released by the OS if a process dies.

    All cooperating run writers share the same instance within a process.
    Use a local filesystem with advisory-lock support, not a notebook Drive FUSE mount.
    """
    def __init__(self, root, timeout=600, name=".artifacts.lock"):
        self.path = Path(root) / name
        self.timeout, self.depth = timeout, 0
        self.local = threading.RLock()

    def __enter__(self):
        self.local.acquire()
        try:
            if self.depth == 0:
                self.handle = self.path.open("a+b")
                if self.path.stat().st_size == 0:
                    self.handle.write(b"0")
                    self.handle.flush()
                start = time.monotonic()
                while True:
                    try:
                        if os.name == "nt":
                            import msvcrt
                            self.handle.seek(0)
                            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
                        else:
                            import fcntl
                            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except (BlockingIOError, OSError):
                        if time.monotonic() - start >= self.timeout:
                            self.handle.close()
                            raise TimeoutError("Run artifact lock is busy: " + str(self.path))
                        time.sleep(0.1)
            self.depth += 1
            return self
        except BaseException:
            self.local.release()
            raise

    def __exit__(self, *exc):
        try:
            self.depth -= 1
            if self.depth == 0:
                if os.name == "nt":
                    import msvcrt
                    self.handle.seek(0)
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
                self.handle.close()
        finally:
            self.local.release()


def write_json(path, value, lock=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock or nullcontext():
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def append_event(path, event, lock):
    with lock:
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp_utc": timestamp(), **event}, allow_nan=False) + "\n")


def archive_files(root, files, archive, metadata=None):
    """Caller holds the run lock. Archive only supplied regular, non-symlink files."""
    root, archive = Path(root).resolve(), Path(archive).resolve()
    if archive.exists() or archive.with_name(archive.name + ".sha256").exists():
        raise FileExistsError("Backup already exists: " + str(archive))
    if archive == root or root in archive.parents:
        raise ValueError("Backup must be outside its source run")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".tmp")
    inventory = {}
    try:
        with tarfile.open(temporary, "w:gz") as tar:
            for original in sorted(set(Path(p) for p in files)):
                if original.is_symlink():
                    raise ValueError("Symlink run artifact is not supported")
                path = original.resolve()
                relative = path.relative_to(root).as_posix()
                if path.is_symlink() or root not in path.resolve().parents or not path.is_file():
                    raise ValueError("Not a safe regular run artifact: " + str(path))
                inventory[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
                tar.add(path, arcname="run/" + relative, recursive=False)
            snapshot = {"schema": 1, "created_utc": timestamp(), "files": inventory,
                        "metadata": metadata or {}}
            payload = json.dumps(snapshot, indent=2, allow_nan=False).encode()
            info = tarfile.TarInfo("SNAPSHOT.json")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        os.replace(temporary, archive)
        checksum = archive.with_name(archive.name + ".sha256")
        checksum.write_text(sha256(archive) + "  " + archive.name + "\n", encoding="utf-8")
        return {"archive": str(archive), "sha256": sha256(archive), "files": len(inventory)}
    finally:
        if temporary.exists():
            temporary.unlink()


def backup_run(run, archive):
    run = Path(run).resolve()
    if not (run / "metadata.json").is_file():
        raise ValueError("Expected a new-layout executed run with metadata.json")
    with RunLock(run):
        checkpoints = {p.name: sha256(p) for p in (run / "checkpoints").glob("*.pth")}
        if checkpoints:
            committed = json.loads((run / "checkpoint_hashes.json").read_text())
            if checkpoints != committed:
                raise ValueError("Checkpoint publication was interrupted or changed; use the last verified backup")
        files = [p for name in ("metadata.json", "config.yaml", "gates.json", "completed.json", "training_state.json", "checkpoint_hashes.json")
                 for p in [run / name] if p.is_file()]
        for folder in ("checkpoints", "logs"):
            files.extend(p for p in (run / folder).rglob("*") if p.is_file() and p.suffix != ".tmp")
        for marker in (run / "evaluation").rglob("evaluation_metadata.json"):
            evidence = json.loads(marker.read_text(encoding="utf-8"))
            if evidence.get("status") in ("complete", "benchmark_complete"):
                files.extend(p for p in marker.parent.rglob("*") if p.is_file() and p.suffix != ".tmp")
        meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        return archive_files(run, files, archive, {"identity": meta["identity"],
                              "experiment": meta["config"]["experiment"]["id"]})


def restore_archive(archive, destination):
    """Never extract links, special files or traversal paths; never overwrite a run."""
    archive, destination = Path(archive).resolve(), Path(destination).resolve()
    expected = archive.with_name(archive.name + ".sha256").read_text().split()[0]
    if sha256(archive) != expected:
        raise ValueError("Archive SHA-256 mismatch")
    if destination.exists():
        raise FileExistsError("Restore destination must not exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar, tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or "SNAPSHOT.json" not in names:
            raise ValueError("Duplicate archive members or missing manifest")
        for member in members:
            parts = PurePosixPath(member.name).parts
            if (not member.isfile() or member.name.startswith("/") or ".." in parts
                    or "\\" in member.name or ":" in member.name):
                raise ValueError("Unsafe archive member: " + member.name)
        snapshot = json.load(tar.extractfile("SNAPSHOT.json"))
        if snapshot.get("schema") != 1 or set(names) != {"SNAPSHOT.json"} | {"run/" + name for name in snapshot["files"]}:
            raise ValueError("Archive inventory mismatch")
        staging = Path(temporary) / "run"
        staging.mkdir()
        for relative, info in snapshot["files"].items():
            member = tar.getmember("run/" + relative)
            target = staging / relative
            if staging.resolve() not in target.resolve().parents or member.size != info["bytes"]:
                raise ValueError("Invalid artifact path/size")
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with tar.extractfile(member) as source, target.open("xb") as handle:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    handle.write(block)
            if digest.hexdigest() != info["sha256"]:
                raise ValueError("Artifact SHA-256 mismatch: " + relative)
        write_json(staging / "RESTORE_PROVENANCE.json", snapshot)
        os.rename(staging, destination)
    return {"destination": str(destination), "files": len(snapshot["files"]), "verified": True}
