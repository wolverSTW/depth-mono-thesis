"""Linux single-GPU execution using upstream model/loss/optimizer/epoch code."""
import hashlib
import json
import logging
import platform
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
import torch
import yaml
from . import checkpoint
from .artifacts import RunLock, write_json, timestamp
from .config import ROOT, resolve_config
from .events import TrainingEvents
from .provenance import check_sources, environment, verify_dataset, verify_checkpoint, sha256
from .model import build_model, upstream_path
from .evaluation import infer, benchmark, auxiliary, profile
from .reporting import export_benchmark, evaluation_metadata
from .training_session import StopRequest, run_epoch_session, validate_stop_after_epoch


def require_cuda():
    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise RuntimeError("G2–G4 require Linux CUDA; CPU execution cannot replace them")
    import torchvision
    if (torch.__version__.split("+")[0], torchvision.__version__.split("+")[0], np.__version__, torch.version.cuda) != ("2.4.1", "0.19.1", "1.26.4", "12.1"):
        raise RuntimeError("Use documented torch 2.4.1 / torchvision 0.19.1 / numpy 1.26.4 / CUDA 12.1")


def preparation(config, dataset_root, manifest, overrides=None, checkpoint_dir=None):
    cfg = resolve_config(config, dataset_root, overrides)
    sources = check_sources()
    data = verify_dataset(cfg["dataset"]["root_dir"], manifest)
    if cfg["dataset"].get("protocol") != data["protocol"]:
        raise ValueError("Configuration and dataset manifest split protocol differ")
    prior_path = Path(checkpoint_dir or ROOT / "checkpoints") / "depth_anything_v2_vits.pth"
    prior = verify_checkpoint("dav2_small", prior_path) if cfg["experiment"]["id"] != "A0" else None
    code = {p.relative_to(ROOT).as_posix(): sha256(p) for folder in ("depth_monoia", "scripts") for p in sorted((ROOT / folder).rglob("*.py"))}
    identity = {"config": cfg, "sources": sources, "source_manifest_sha256": sha256(ROOT / "docs/reproducibility/sources.json"),
                "extension_code": code, "data": data, "prior": prior}
    canonical = json.loads(json.dumps(identity))
    canonical["config"]["dataset"]["root_dir"] = "<KITTI_ROOT>"
    canonical["config"]["model"]["focal_embedding_path"] = "<PINNED_MONOIA_EMBEDDINGS>"
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    protocol = {"dataset_protocol": data["protocol"],
                "per_device_batch_size": cfg["dataset"]["batch_size"],
                "devices": cfg["execution"]["devices"], "gradient_accumulation_steps": 1,
                "effective_batch_size": cfg["dataset"]["batch_size"] * cfg["execution"]["devices"],
                "precision": cfg["execution"]["precision"], "seed": cfg["random_seed"],
                "optimizer": cfg["optimizer"], "scheduler": cfg["lr_scheduler"],
                "protocol_changes": cfg["execution"].get("protocol_changes", {})}
    return cfg, {"identity": digest, **identity, "training_protocol": protocol, "environment": environment()}


def components(cfg, checkpoint_dir=None):
    require_cuda()
    upstream_path()
    from lib.helpers.utils_helper import set_random_seed
    from lib.helpers.dataloader_helper import build_dataloader
    from lib.helpers.optimizer_helper import build_optimizer
    from lib.helpers.scheduler_helper import build_lr_scheduler
    set_random_seed(cfg["random_seed"])
    train, val = build_dataloader(cfg["dataset"], workers=cfg["execution"]["workers"])
    model, criterion = build_model(cfg, checkpoint_dir)
    model.cuda()
    optimizer = build_optimizer(cfg["optimizer"], model)
    scheduler, warmup = build_lr_scheduler(cfg["lr_scheduler"], optimizer, last_epoch=-1)
    return model, criterion, optimizer, scheduler, warmup, train, val


def record_hub(metadata):
    from lib.models.monoia.backbone import donwload_url
    name = donwload_url[int(metadata["config"]["model"]["backbone"].replace("resnet", ""))].rsplit("/", 1)[-1]
    path = Path(torch.hub.get_dir()) / "checkpoints" / name
    metadata["backbone_cache"] = {name: sha256(path)}
    operator = Path(sys.modules["MultiScaleDeformableAttention"].__file__)
    metadata["compiled_ops"] = {operator.name: sha256(operator)}


class Tee:
    def __init__(self, terminal, handle, lock):
        self.terminal, self.handle, self.lock = terminal, handle, lock

    def write(self, text):
        self.terminal.write(text)
        with self.lock:
            self.handle.write(text)
            self.handle.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        with self.lock:
            self.handle.flush()


def train(config, dataset_root, manifest, run_dir, gate_evidence, resume=None, overrides=None, checkpoint_dir=None,
          stop_after_epoch=None):
    cfg, metadata = preparation(config, dataset_root, manifest, overrides, checkpoint_dir)
    validate_stop_after_epoch(stop_after_epoch, cfg["trainer"]["max_epoch"])
    gate = json.loads(Path(gate_evidence).read_text())
    if gate.get("identity") != metadata["identity"] or any(gate.get(g) != "PASS" for g in ("G1", "G2", "G3", "G4")):
        raise ValueError("All four gates must PASS for this exact configuration before full training")
    metadata["verification_gates"] = gate
    # A4 unlocked after documented A0-A3 evidence review.
    require_cuda()
    if metadata["environment"]["dirty"]:
        raise ValueError("Commit reviewed source/configuration before full training")
    run = Path(run_dir).resolve()
    if resume:
        if not run.is_dir() or Path(resume).resolve().parent != run / "checkpoints":
            raise ValueError("Resume must use this existing run's checkpoints directory")
        if json.loads((run / "metadata.json").read_text())["identity"] != metadata["identity"]:
            raise ValueError("Existing run identity differs")
    else:
        run.mkdir(parents=True, exist_ok=False)
    for folder in ("checkpoints", "logs", "evaluation"):
        (run / folder).mkdir(exist_ok=True)
    lock = RunLock(run)
    lease = RunLock(run, timeout=0, name=".training.lock")
    lease.__enter__()
    try:
        return _train_run(cfg, metadata, run, lock, checkpoint_dir, resume, stop_after_epoch)
    finally:
        lease.__exit__(None, None, None)


def _train_run(cfg, metadata, run, lock, checkpoint_dir, resume, stop_after_epoch=None):
    validate_stop_after_epoch(stop_after_epoch, cfg["trainer"]["max_epoch"])
    model, criterion, optimizer, scheduler, warmup, train_loader, val_loader = components(cfg, checkpoint_dir)
    record_hub(metadata)
    actual_assets = {key: metadata[key] for key in ("backbone_cache", "compiled_ops")}
    if metadata["verification_gates"].get("initialization_assets") != actual_assets:
        raise ValueError("Initialization/operator bytes changed since the GPU gates; rerun G1–G4")
    metadata["created_utc"] = timestamp()
    if not resume:
        with lock:
            (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            write_json(run / "metadata.json", metadata)
            write_json(run / "gates.json", metadata["verification_gates"])
    start, best, global_step = 0, float("-inf"), 0
    if resume:
        with lock:
            hashes = json.loads((run / "checkpoint_hashes.json").read_text())
            if hashes.get(Path(resume).name) != sha256(resume):
                raise ValueError("Resume checkpoint differs from committed hash; restore a verified snapshot")
            start, best, global_step = checkpoint.resume(resume, model, optimizer, scheduler, warmup, metadata)
    validate_stop_after_epoch(stop_after_epoch, cfg["trainer"]["max_epoch"], start)
    if (run / "completed.json").exists() and start < cfg["trainer"]["max_epoch"]:
        raise ValueError("Cannot rewind a completed run in place; preserve its completion evidence")
    from lib.helpers.trainer_helper import Trainer
    logger = logging.getLogger("depth_monoia")
    logger.setLevel(logging.INFO)
    trainer = Trainer(dict(cfg["trainer"], save_path=str(run)), model, optimizer, train_loader, val_loader,
                      scheduler, warmup, logger, criterion, cfg["experiment"]["id"])
    events = TrainingEvents(run, lock, criterion, optimizer, global_step)
    stop = StopRequest()
    events.event("resumed" if resume else "started", source_commit=metadata["environment"]["git_commit"],
                 environment=metadata["environment"], initialization_assets=actual_assets, next_epoch=start,
                 stop_after_epoch=stop_after_epoch)
    def commit_epoch(epoch):
        nonlocal best
        events.epoch = epoch + 1
        np.random.seed(np.random.get_state()[1][0] + epoch)
        trainer.epoch = epoch
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        with (run / "logs/training.log").open("a", encoding="utf-8") as handle:
            with redirect_stdout(Tee(sys.stdout, handle, lock)):
                trainer.train_one_epoch(epoch)
        if stop.reason is not None:
            events.event("stop_requested", policy="finish current epoch validation and checkpoint")
        if warmup is not None and epoch < 5:
            warmup.step()
        else:
            scheduler.step()
        epoch_dir = run / "evaluation" / f"epoch_{epoch + 1:03d}"
        with lock:
            if epoch_dir.exists():
                epoch_dir.rename(epoch_dir.with_name(epoch_dir.name + f"_interrupted_{time.time_ns()}"))
            epoch_dir.mkdir()
        infer(model, val_loader, cfg, epoch_dir / "predictions")
        evaluation = benchmark(val_loader.dataset, epoch_dir / "predictions", epoch_dir / "metrics.json")
        export_benchmark(epoch_dir, evaluation)
        score = evaluation["Car_3d_moderate_R40"]
        improved, best = score > best, max(best, score)
        next_epoch = epoch + 1
        with lock:
            if improved:
                checkpoint.save(run / "checkpoints/best.pth", model, optimizer, scheduler, warmup,
                                next_epoch, best, metadata, events.global_step)
            checkpoint.save(run / "checkpoints/last.pth", model, optimizer, scheduler, warmup,
                            next_epoch, best, metadata, events.global_step)
            hashes = {p.name: sha256(p) for p in (run / "checkpoints").glob("*.pth")}
            write_json(run / "checkpoint_hashes.json", hashes)
            write_json(epoch_dir / "evaluation_metadata.json", evaluation_metadata(cfg, metadata,
                {"sha256": hashes["last.pth"], "epoch": next_epoch}, "benchmark_complete", "independent_checkpoint"))
            write_json(run / "training_state.json", {"next_epoch": next_epoch, "global_step": events.global_step,
                "best_AP3D_Moderate_R40": best, "status": "epoch_committed", "timestamp_utc": timestamp()})
            events.event("checkpoint_saved", checkpoint_sha256=hashes, improved=improved,
                AP3D_Moderate_R40=score, epoch_and_validation_seconds=time.perf_counter() - tick,
                peak_GPU_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
                next_lr=[g["lr"] for g in optimizer.param_groups])
    return run_epoch_session(run, lock, events, start, cfg["trainer"]["max_epoch"],
                             commit_epoch, stop, stop_after_epoch)


def evaluate(config, dataset_root, manifest, weights, run_dir, official=False, overrides=None, checkpoint_dir=None, benchmark_only=False, gate_evidence=None):
    cfg, metadata = preparation(config, dataset_root, manifest, overrides, checkpoint_dir)
    require_cuda()
    model, _, _, _, _, _, val_loader = components(cfg, checkpoint_dir)
    weights = Path(weights).resolve()
    owner = weights.parent.parent if weights.parent.name == "checkpoints" else weights.parent
    with RunLock(owner):
        gate = json.loads(Path(gate_evidence).read_text()) if gate_evidence else None
        if official:
            if cfg["experiment"]["id"] != "A0":
                raise ValueError("Official MonoIA weights require A0")
            info = verify_checkpoint("monoia_val", weights)
            category = "official_pretrained"
        else:
            state = torch.load(weights, map_location="cpu", weights_only=False)
            if state.get("schema") != 2 or state["metadata"]["identity"] != metadata["identity"]:
                raise ValueError("Checkpoint configuration/code/data identity mismatch")
            model.load_state_dict(state["model_state"], strict=True)
            gate = gate or state["metadata"].get("verification_gates")
            info = {"sha256": sha256(weights), "epoch": state["next_epoch"], "global_step": state["global_step"],
                    "full_training_completed": state["next_epoch"] == cfg["trainer"]["max_epoch"]}
            category = "independent_checkpoint"
        if not gate or gate.get("identity") != metadata["identity"] or any(gate.get(g) != "PASS" for g in ("G1", "G2", "G3", "G4")):
            raise ValueError("Evaluation requires matching G1–G4; official weights need --gates for A0")
        if official:
            checkpoint.load_weights(weights, model, verified_official=True)
        metadata["verification_gates"] = gate
    info["path"] = str(weights)
    output = Path(run_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    record_hub(metadata)
    if benchmark_only:
        profile(model, val_loader, output / "resource_metrics.json")
        write_json(output / "benchmark_metadata.json", evaluation_metadata(cfg, metadata, info, "resource_measured", category))
        return
    infer(model, val_loader, cfg, output / "predictions")
    ap = benchmark(val_loader.dataset, output / "predictions", output / "metrics.json")
    export_benchmark(output, ap)
    distance = auxiliary(val_loader.dataset, output / "predictions", output / "distance_metrics.json")
    confidence = {key: distance.pop(key) for key in ("detection_ECE", "ECE_bins", "ECE_status", "detections")}
    write_json(output / "distance_metrics.json", distance)
    write_json(output / "confidence_metrics.json", confidence)
    resources = profile(model, val_loader, output / "resource_metrics.json")
    sessions = []
    if not official and (owner / "logs/events.jsonl").is_file():
        for line in (owner / "logs/events.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "session_ended":
                sessions.append(event["duration_seconds"])
    resources["training_duration_seconds_recorded"] = sum(sessions) if sessions else None
    resources["training_duration_scope"] = "Sum of recorded session_ended events, including validation; abrupt-kill time may be missing"
    write_json(output / "resource_metrics.json", resources)
    # Completion metadata is published last. Backup/aggregation ignore unfinished evaluations.
    write_json(output / "evaluation_metadata.json", evaluation_metadata(cfg, metadata, info, "complete", category))
