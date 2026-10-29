"""Linux single-GPU runner, reusing upstream loss/optimizer/scheduler/epoch code."""
import hashlib
import json
import logging
import platform
import time
import sys
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
import torch
import yaml
from . import checkpoint
from .config import ROOT, resolve_config
from .provenance import check_sources, environment, verify_dataset, verify_checkpoint, sha256
from .model import build_model, upstream_path
from .evaluation import infer, benchmark, auxiliary, profile


def require_cuda():
    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise RuntimeError("G2–G4 require Linux CUDA; CPU/dummy execution cannot replace them")
    import torchvision
    if (torch.__version__.split("+")[0], torchvision.__version__.split("+")[0],
        np.__version__, torch.version.cuda) != ("2.4.1", "0.19.1", "1.26.4", "12.1"):
        raise RuntimeError("Use the documented torch 2.4.1 / torchvision 0.19.1 / numpy 1.26.4 / CUDA 12.1 environment")


def preparation(config, dataset_root, manifest):
    cfg = resolve_config(config, dataset_root)
    sources = check_sources()
    data = verify_dataset(cfg["dataset"]["root_dir"], manifest)
    prior = verify_checkpoint("dav2_small") if cfg["experiment"]["id"] != "A0" else None
    code = {p.relative_to(ROOT).as_posix(): sha256(p) for folder in ("depth_monoia", "scripts")
            for p in sorted((ROOT / folder).glob("*.py"))}
    identity = {"config": cfg, "sources": sources, "extension_code": code, "data": data, "prior": prior}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return cfg, {"identity": digest, **identity, "environment": environment()}


def components(cfg):
    require_cuda()
    upstream_path()
    from lib.helpers.utils_helper import set_random_seed
    from lib.helpers.dataloader_helper import build_dataloader
    from lib.helpers.optimizer_helper import build_optimizer
    from lib.helpers.scheduler_helper import build_lr_scheduler
    set_random_seed(cfg["random_seed"])
    train, val = build_dataloader(cfg["dataset"], workers=cfg["execution"]["workers"])
    model, criterion = build_model(cfg)
    model.cuda()
    optimizer = build_optimizer(cfg["optimizer"], model)
    scheduler, warmup = build_lr_scheduler(cfg["lr_scheduler"], optimizer, last_epoch=-1)
    return model, criterion, optimizer, scheduler, warmup, train, val


def record_hub(metadata):
    metadata["backbone_cache"] = {p.name: sha256(p) for p in
                                 (Path(torch.hub.get_dir()) / "checkpoints").glob("*ResNet50*")}
    metadata["compiled_ops"] = {p.name: sha256(p) for p in (ROOT / "build/ops").glob("*.so")}


class Tee:
    def __init__(self, terminal, handle):
        self.terminal, self.handle = terminal, handle

    def write(self, text):
        self.terminal.write(text)
        self.handle.write(text)
        self.handle.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.handle.flush()


def train(config, dataset_root, manifest, run_dir, gate_evidence, resume=None):
    cfg, metadata = preparation(config, dataset_root, manifest)
    gate = json.loads(Path(gate_evidence).read_text())
    if gate.get("identity") != metadata["identity"] or any(gate.get(g) != "PASS" for g in ("G1", "G2", "G3", "G4")):
        raise ValueError("All four gates must PASS for this exact configuration before full training")
    if cfg["experiment"]["id"] == "A4":
        raise ValueError("A4 execution is intentionally locked until A0–A3 real evidence is reviewed")
    require_cuda()
    if metadata["environment"]["dirty"]:
        raise ValueError("Commit the reviewed source/configuration before full training")
    run = Path(run_dir).resolve()
    if resume:
        if not run.is_dir() or Path(resume).resolve().parent != run:
            raise ValueError("Resume must use an existing run and its checkpoint")
    else:
        run.mkdir(parents=True, exist_ok=False)
    model, criterion, optimizer, scheduler, warmup, train_loader, val_loader = components(cfg)
    record_hub(metadata)
    if resume:
        existing = json.loads((run / "provenance.json").read_text())
        if existing["identity"] != metadata["identity"]:
            raise ValueError("Existing run identity differs")
    else:
        (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (run / "provenance.json").write_text(json.dumps(metadata, indent=2))
    logger = logging.getLogger("depth_monoia")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(run / "train.log"))
    logger.addHandler(logging.StreamHandler())
    from lib.helpers.trainer_helper import Trainer
    trainer_cfg = dict(cfg["trainer"], save_path=str(run))
    trainer = Trainer(trainer_cfg, model, optimizer, train_loader, val_loader,
                      scheduler, warmup, logger, criterion, cfg["experiment"]["id"])
    start, best = 0, float("-inf")
    if resume:
        start, best = checkpoint.resume(resume, model, optimizer, scheduler, warmup, metadata)
    total_start = time.perf_counter()
    for epoch in range(start, cfg["trainer"]["max_epoch"]):
        np.random.seed(np.random.get_state()[1][0] + epoch)  # original epoch seed policy
        trainer.epoch = epoch
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        with (run / "batch_losses.log").open("a") as handle:
            with redirect_stdout(Tee(sys.stdout, handle)):
                trainer.train_one_epoch(epoch)  # exact original criterion/optimizer step
        if warmup is not None and epoch < 5:
            warmup.step()
        else:
            scheduler.step()
        epoch_dir = run / f"eval_epoch_{epoch + 1:03d}"
        # An interrupted validation is discarded only by renaming it for review.
        if epoch_dir.exists():
            epoch_dir.rename(run / (epoch_dir.name + f"_interrupted_{time.time_ns()}"))
        epoch_dir.mkdir()
        infer(model, val_loader, cfg, epoch_dir / "data")
        evaluation = benchmark(val_loader.dataset, epoch_dir / "data", epoch_dir / "AP.json")
        score = evaluation["Car_3d_moderate_R40"]
        improved = score > best
        best = max(best, score)
        if improved:
            checkpoint.save(run / "checkpoint_best.pth", model, optimizer, scheduler, warmup,
                            epoch + 1, best, metadata)
        checkpoint.save(run / "checkpoint.pth", model, optimizer, scheduler, warmup,
                        epoch + 1, best, metadata)
        event = {"epoch": epoch + 1, "AP3D_Moderate_R40": score, "best": best,
                 "epoch_and_validation_seconds": time.perf_counter() - tick,
                 "session_seconds": time.perf_counter() - total_start,
                 "peak_GPU_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
                 "next_lr": [g["lr"] for g in optimizer.param_groups]}
        with (run / "epochs.jsonl").open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        logger.info(json.dumps(event))
    (run / "completed.json").write_text(json.dumps({
        "completed_epochs": cfg["trainer"]["max_epoch"], "best_AP3D_Moderate_R40": best,
        "last_session_seconds": time.perf_counter() - total_start,
        "status": "Training complete; final held-out evaluation and reporting still required"}, indent=2))


def evaluate(config, dataset_root, manifest, weights, run_dir, official=False):
    cfg, metadata = preparation(config, dataset_root, manifest)
    if official:
        if cfg["experiment"]["id"] != "A0":
            raise ValueError("Official MonoIA weights require A0")
        metadata["evaluated_checkpoint"] = verify_checkpoint("monoia_val", weights)
    require_cuda()
    model, _, _, _, _, _, val_loader = components(cfg)
    # Official checkpoint bytes were authenticated above. Local full training
    # states are trusted by the operator. Never load an untrusted pickle.
    if official:
        checkpoint.load_weights(weights, model, verified_official=True)
    else:
        state = torch.load(weights, map_location="cpu", weights_only=False)
        if state.get("schema") != 1 or state["metadata"]["identity"] != metadata["identity"]:
            raise ValueError("Checkpoint provenance/configuration mismatch")
        model.load_state_dict(state["model_state"], strict=True)
        metadata["evaluated_checkpoint"] = {"sha256": sha256(weights), "metadata": state["metadata"]}
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=False)
    record_hub(metadata)
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2))
    infer(model, val_loader, cfg, output / "data")
    benchmark(val_loader.dataset, output / "data", output / "AP.json")
    auxiliary(val_loader.dataset, output / "data", output / "distance_calibration.json")
    profile(model, val_loader, output / "resources.json")
