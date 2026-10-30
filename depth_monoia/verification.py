"""One command for static checks and four fail-closed research gates."""
import ast
import importlib
import json
import tempfile
import math
import random
import numpy as np
from pathlib import Path
import torch
from .config import ROOT, load_config, CONFIGS
from .provenance import check_sources, environment, verify_checkpoint, sha256
from .fusion import MultiScaleFusion


def static_checks():
    files = [p for folder in ("depth_monoia", "scripts", "tests") for p in (ROOT / folder).rglob("*.py")]
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    # No CUDA extension import on CPU; import the actual new modules instead.
    for module in ("config", "provenance", "fusion", "depth", "model", "checkpoint", "evaluation", "runner", "artifacts", "events", "reporting", "kitti_download"):
        importlib.import_module("depth_monoia." + module)
    configs = {name: load_config(ROOT / path) for name, path in CONFIGS.items()}
    for name, cfg in configs.items():
        if cfg["experiment"]["id"] != name:
            raise ValueError("Experiment ID mismatch")
        a0 = configs["A0"]
        for key in ("dataset", "model", "optimizer", "lr_scheduler", "trainer", "random_seed"):
            if cfg[key] != a0[key]:
                raise ValueError("Uncontrolled ablation setting: " + key)
    # Isolated tensor algebra only: no dummy dataset, detector or benchmark.
    torch.manual_seed(17)
    for mode in ("simple", "gated", "intrinsic", "confidence"):
        fusion = MultiScaleFusion(mode)
        rgb = [torch.randn(1, 256, 3, 5) for _ in range(4)]
        priors = [torch.randn(1, 385, 2, 3) for _ in range(4)]
        embedding = torch.randn(1, 256)
        confidence = torch.rand(1, 1, 2, 3)
        initial = fusion(rgb, priors, embedding, confidence)
        for before, after in zip(rgb, initial):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        optimizer = torch.optim.SGD(fusion.parameters(), lr=0.1)
        for _ in range(2):
            optimizer.zero_grad()
            sum(x.square().mean() for x in fusion(rgb, priors, embedding, confidence)).backward()
            optimizer.step()
        for name, p in fusion.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.abs().sum() > 0:
                raise AssertionError("Fusion gradient missing: " + mode + "/" + name)
    import unittest
    import io
    import shutil
    import subprocess
    import os
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    stream = io.StringIO()
    tested = unittest.TextTestRunner(stream=stream).run(suite)
    if not tested.wasSuccessful():
        raise AssertionError(stream.getvalue())
    bash = (Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "/missing/bash"))
    shell_files = sorted((ROOT / "scripts").rglob("*.sh"))
    shell_status = "Pending: Bash unavailable"
    if bash.is_file():
        for script in shell_files:
            subprocess.run([str(bash), "-n", script.as_posix()], capture_output=True, check=True)
        shell_status = f"PASS: {len(shell_files)} Bash syntax checks; Linux execution remains pending"
    return {"syntax_files": len(files), "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in files},
            "configuration_count": len(configs), "operational_tests": tested.testsRun,
            "shell_syntax": shell_status,
            "isolated_fusion_checks": "PASS: exact zero-alpha identity and two-step gradients; no KITTI gate evidence"}


def run(config, dataset_root, manifest, output, gpu=False, reference_predictions=None, reference_metrics=None, overrides=None, checkpoint_dir=None):
    from .runner import preparation, components, record_hub
    result = {"static": static_checks(), "sources": check_sources(), "environment": environment(),
              "G1": "BLOCKED", "G2": "BLOCKED", "G3": "BLOCKED", "G4": "BLOCKED", "details": {}}
    try:
        prior_path = Path(checkpoint_dir or ROOT / "checkpoints") / "depth_anything_v2_vits.pth"
        result["dav2_checkpoint"] = verify_checkpoint("dav2_small", prior_path)
        from .depth import FrozenDepthPrior
        prior = FrozenDepthPrior(prior_path)
        prior.train(True)
        if prior.training or any(p.requires_grad for p in prior.parameters()):
            raise AssertionError("DA-V2 freeze/eval contract failed")
        result["static"]["official_depth_load"] = "PASS: strict official checkpoint load, all weights frozen, eval mode retained"
        del prior
    except (OSError, ValueError) as error:
        result["dav2_checkpoint"] = str(error)
    cfg = metadata = None
    try:
        if not manifest:
            raise ValueError("Supply verified --manifest and --dataset-root on the GPU machine")
        cfg, metadata = preparation(config, dataset_root, manifest, overrides, checkpoint_dir)
        result["identity"] = metadata["identity"]
        result["data"] = metadata["data"]
        result["configuration"] = cfg
        result["training_protocol"] = metadata["training_protocol"]
        result["G1"] = "PASS"
    except (OSError, ValueError) as error:
        result["details"]["G1"] = str(error)
    if gpu and result["G1"] == "PASS":
        if cfg["experiment"]["id"] == "A4":
            raise ValueError("A4 runtime verification is locked until A0–A3 work")
        try:
            from .runner import require_cuda
            require_cuda()
            model, criterion, optimizer, scheduler, warmup, train_loader, val_loader = components(cfg, checkpoint_dir)
            record_hub(metadata)
            result["initialization_assets"] = {k: metadata[k] for k in ("backbone_cache", "compiled_ops")}
            # Exercise the configured batch and original training augmentations.
            # A single sample cannot establish capacity for baseline batch 16.
            batch = next(iter(train_loader))
            if len(batch[0]) != cfg["dataset"]["batch_size"]:
                raise ValueError("G2 requires a complete configured training batch")
            result["details"]["integration_train_ids"] = batch[3]["img_id"].tolist()
            capacity = result["details"]["capacity"] = {}
            g2_g3(model, criterion, optimizer, scheduler, warmup, batch, metadata, capacity)
            result["G2"] = result["G3"] = "PASS"
        except Exception as error:
            result["details"]["G2_G3"] = type(error).__name__ + ": " + str(error)
            if isinstance(error, torch.cuda.OutOfMemoryError):
                result["details"]["capacity"] = {
                    **result["details"].get("capacity", {}), "status": "CUDA_OOM",
                    "configured_batch_size": cfg["dataset"]["batch_size"],
                    "gpu": torch.cuda.get_device_name(),
                    "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "action": "Stopped. Batch size/precision unchanged. Review protocol before an explicit batch reduction."}
        try:
            if result["details"].get("capacity", {}).get("status") == "CUDA_OOM":
                raise ValueError("G4 not run after configured-batch CUDA OOM; no training permitted")
            if not reference_predictions or not reference_metrics:
                raise ValueError("G4 requires real validation reference predictions and independently obtained KITTI R40 metrics")
            from .evaluation import benchmark
            from .model import upstream_path
            upstream_path()
            from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
            dataset = KITTI_Dataset("val", cfg["dataset"], data_augmentation=False)
            expected = json.loads(Path(reference_metrics).read_text())
            if not expected.get("reference_source") or expected.get("split_sha256") != metadata["data"]["splits"]["val"]["sha256"]:
                raise ValueError("Reference source and matching validation split hash are mandatory")
            with tempfile.TemporaryDirectory() as temp:
                actual = benchmark(dataset, reference_predictions, Path(temp) / "AP.json")
            for metric in ("3d", "bev"):
                for difficulty in ("easy", "moderate", "hard"):
                    key = f"Car_{metric}_{difficulty}_R40"
                    observed, target = actual["metrics"][key], expected["metrics"][key]
                    if not math.isfinite(observed) or not math.isfinite(target) or abs(observed - target) > 1e-3:
                        raise AssertionError("KITTI reference metric mismatch: " + key)
            result["G4"] = "PASS"
            result["details"]["evaluation_reference"] = expected["reference_source"]
        except Exception as error:
            result["details"]["G4"] = type(error).__name__ + ": " + str(error)
    else:
        result["details"]["G2_G3_G4"] = "Pending: Linux CUDA, real KITTI and reference evaluation evidence"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return all(result[g] == "PASS" for g in ("G1", "G2", "G3", "G4"))


def g2_g3(model, criterion, optimizer, scheduler, warmup, batch, metadata, capacity=None):
    from .checkpoint import save, resume, rng_state
    from lib.helpers.trainer_helper import Trainer
    images, calibs, targets, info = batch
    images, calibs = images.cuda(), calibs.cuda()
    targets = {k: v.cuda() for k, v in targets.items()}
    sizes = targets["img_size"]
    model.eval()
    with torch.no_grad():
        actual = model(images, calibs, targets, sizes)
        if hasattr(model, "baseline_forward"):
            baseline = model.baseline_forward(images, calibs, targets, sizes)
            for key in ("pred_logits", "pred_boxes", "pred_depth", "pred_3d_dim", "pred_angle"):
                torch.testing.assert_close(actual[key], baseline[key], rtol=0, atol=0)
            del baseline
        del actual
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    prepared = Trainer.prepare_targets(None, targets, len(images))
    def step():
        model.train()
        criterion.train()
        optimizer.zero_grad()
        outputs = model(images, calibs, targets, sizes)
        losses = criterion(outputs, prepared, None)
        loss = sum(v * criterion.weight_dict[k] for k, v in losses.items() if k in criterion.weight_dict)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite real-data integration loss")
        loss.backward()
        optimizer.step()
        return loss.detach()
    step()
    step()
    torch.cuda.synchronize()
    if capacity is None:
        capacity = {}
    capacity.update({"status": "TRAINING_STEPS_FIT", "batch_size": len(images), "input_shape": list(images.shape),
                "dtype": str(images.dtype), "gpu": torch.cuda.get_device_name(),
                "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "scope": "two real configured-batch training steps, including optimizer state; not a full-run VRAM guarantee"})
    if hasattr(model, "fusion"):
        for name, p in model.fusion.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.abs().sum() > 0:
                raise AssertionError("Real-data gradient missing: " + name)
        if model.prior.training or any(p.grad is not None for p in model.prior.parameters()):
            raise AssertionError("DA-V2 must remain frozen in eval mode")
    scheduler.step()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "resume.pth"
        save(path, model, optimizer, scheduler, warmup, 1, 0.0, metadata, global_step=2)
        # Preserve exact model state at the checkpoint boundary.
        saved_model_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        saved_scheduler = scheduler.state_dict()
        saved_rng = rng_state()
        expected_loss = step()
        expected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        # Perturb all restored state; avoid a round trip that would pass merely
        # because state remained in memory. Resume must repair these changes.
        scheduler.step()
        np.random.random()
        random.random()
        torch.rand(1)
        torch.rand(1, device="cuda")
        next_epoch, _, global_step = resume(path, model, optimizer, scheduler, warmup, metadata)
        assert next_epoch == 1 and global_step == 2
        assert scheduler.state_dict() == saved_scheduler
        restored_rng = rng_state()
        assert restored_rng["python"] == saved_rng["python"]
        assert (restored_rng["numpy"][1] == saved_rng["numpy"][1]).all()
        assert restored_rng["numpy"][2:] == saved_rng["numpy"][2:]
        assert torch.equal(restored_rng["torch"], saved_rng["torch"])
        assert all(torch.equal(a, b) for a, b in zip(restored_rng["cuda"], saved_rng["cuda"]))
        # A checkpoint must restore model tensors exactly BEFORE
        # testing numerical reproducibility of the next GPU step.
        for name, tensor in model.state_dict().items():
            if not torch.equal(
                tensor.detach().cpu(), saved_model_state[name]
            ):
                raise AssertionError(
                    "Checkpoint restoration mismatch before replay: "
                    + name
                )
        print(
            "PASS: Model state restored bitwise exactly",
            flush=True
        )

        actual_loss = step()
        torch.testing.assert_close(expected_loss, actual_loss, rtol=1e-5, atol=1e-6)
        for key, value in model.state_dict().items():
            try:
                # Exact checkpoint model restoration was checked above.
                # This comparison allows bounded FP32 CUDA replay drift;
                # it does NOT establish bitwise-identical training replay.
                torch.testing.assert_close(
                    value.cpu(),
                    expected[key],
                    rtol=1e-5,
                    atol=5e-6
                )
            except AssertionError as error:
                raise AssertionError(
                    "Post-resume GPU replay mismatch in tensor "
                    + key + ": " + str(error)
                ) from error
    return capacity
