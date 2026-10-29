"""Single CLI for research execution, operational artifacts and comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depth_monoia.config import ROOT, CONFIGS, RUN_NAMES, resolve_config
from depth_monoia.provenance import checkpoint_info, verify_checkpoint, verify_dataset, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "train", "evaluate", "benchmark", "evaluate-all", "fetch-checkpoint",
                        "prepare-dataset", "verify-gpu", "resolve-config", "backup", "restore", "aggregate"])
    parser.add_argument("--config")
    parser.add_argument("--experiment", choices=list(CONFIGS))
    parser.add_argument("--dataset-root")
    parser.add_argument("--manifest")
    parser.add_argument("--run-dir")
    parser.add_argument("--output-root", default=str(ROOT / "experiments"))
    parser.add_argument("--run-name", default="run_001")
    parser.add_argument("--output")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--reference-predictions")
    parser.add_argument("--reference-metrics")
    parser.add_argument("--gates")
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints"))
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--name", choices=["dav2_small", "monoia_val"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--archive")
    parser.add_argument("--destination")
    parser.add_argument("--search-root", default=str(ROOT / "experiments"))
    parser.add_argument("--category", choices=["independent_checkpoint", "official_pretrained"], default="independent_checkpoint")
    args = parser.parse_args()
    config = Path(args.config or ROOT / CONFIGS[args.experiment or "A3"]).resolve()
    overrides = {"batch_size": args.batch_size, "workers": args.workers, "seed": args.seed}
    def required(*names):
        for name in names:
            if getattr(args, name) is None:
                parser.error(args.command + " requires --" + name.replace("_", "-"))
    if args.command == "fetch-checkpoint":
        required("name")
        spec = checkpoint_info(args.name)
        target = Path(args.checkpoint_dir) / Path(spec["path"]).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_suffix(".download")
            urllib.request.urlretrieve(spec["url"], temporary)
            verify_checkpoint(args.name, temporary)
            os.replace(temporary, target)
        print(json.dumps(verify_checkpoint(args.name, target), indent=2))
        return
    if args.command in ("backup", "restore", "aggregate"):
        from depth_monoia.artifacts import backup_run, restore_archive
        from depth_monoia.reporting import aggregate
        if args.command == "backup":
            required("run_dir", "archive")
            result = backup_run(args.run_dir, args.archive)
        elif args.command == "restore":
            required("archive", "destination")
            result = restore_archive(args.archive, args.destination)
        else:
            result = aggregate(args.search_root, args.output or ROOT / "results/model_comparison.csv", args.category)
        print(json.dumps(result, indent=2))
        return
    if args.command == "verify-gpu":
        from depth_monoia.runner import require_cuda
        from depth_monoia.model import upstream_path
        require_cuda()
        subprocess.run(["nvidia-smi"], check=True)
        toolkit = subprocess.check_output(["nvcc", "--version"], text=True)
        print(toolkit)
        if "release 12.1," not in toolkit:
            raise RuntimeError("Build/use the documented CUDA toolkit 12.1")
        upstream_path()
        import MultiScaleDeformableAttention
        print(json.dumps(environment(), indent=2))
        return
    cfg = resolve_config(config, args.dataset_root, overrides)
    if args.experiment and cfg["experiment"]["id"] != args.experiment:
        parser.error("--experiment conflicts with --config")
    if args.command == "resolve-config":
        from depth_monoia.artifacts import write_json
        if args.output:
            write_json(args.output, cfg)
        print(json.dumps(cfg, indent=2))
        return
    if args.command == "prepare-dataset":
        required("manifest")
        print(json.dumps(verify_dataset(cfg["dataset"]["root_dir"], args.manifest), indent=2))
        return
    if args.command == "verify":
        from depth_monoia.verification import run
        success = run(config, args.dataset_root, args.manifest, args.output or ROOT / "results/verification.json",
                      args.gpu, args.reference_predictions, args.reference_metrics, overrides, args.checkpoint_dir)
        sys.exit(0 if success else 2)
    if args.command == "evaluate-all":
        from depth_monoia.runner import evaluate
        required("manifest")
        found = sorted(Path(args.search_root).rglob("checkpoints/best.pth"))
        if not found:
            raise FileNotFoundError("No real best checkpoints found; nothing evaluated")
        import yaml
        for weights in found:
            run_root = weights.parent.parent
            source_cfg = yaml.safe_load((run_root / "config.yaml").read_text())
            exp = source_cfg["experiment"]["id"]
            if exp == "A4":
                continue
            settings = {"batch_size": source_cfg["dataset"]["batch_size"],
                        "workers": source_cfg["execution"]["workers"], "seed": source_cfg["random_seed"]}
            evaluate(run_root / "config.yaml", args.dataset_root, args.manifest, weights,
                     run_root / "evaluation/final_best", overrides=settings, checkpoint_dir=args.checkpoint_dir)
        return
    required("manifest")
    run_dir = args.run_dir or str(Path(args.output_root) / RUN_NAMES[cfg["experiment"]["id"]] / args.run_name)
    if args.command == "train":
        required("gates")
        from depth_monoia.runner import train
        train(config, args.dataset_root, args.manifest, run_dir, args.gates, args.resume, overrides, args.checkpoint_dir)
    else:
        required("checkpoint", "run_dir")
        from depth_monoia.runner import evaluate
        evaluate(config, args.dataset_root, args.manifest, args.checkpoint, run_dir,
                 args.official, overrides, args.checkpoint_dir, benchmark_only=args.command == "benchmark", gate_evidence=args.gates)


if __name__ == "__main__":
    main()
