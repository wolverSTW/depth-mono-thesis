"""Single entry point: verification, training, evaluation and checkpoint retrieval."""
import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depth_monoia.config import ROOT
from depth_monoia.provenance import checkpoint_info, verify_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "train", "evaluate", "fetch-checkpoint"])
    parser.add_argument("--config", default=str(ROOT / "configs/proposed/A3.yaml"))
    parser.add_argument("--dataset-root")
    parser.add_argument("--manifest")
    parser.add_argument("--run-dir")
    parser.add_argument("--output", default=str(ROOT / "results/verification.json"))
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--reference-predictions")
    parser.add_argument("--reference-metrics")
    parser.add_argument("--gates")
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint")
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--name", choices=["dav2_small", "monoia_val"])
    args = parser.parse_args()
    if args.command == "fetch-checkpoint":
        if not args.name:
            parser.error("fetch-checkpoint requires --name")
        spec = checkpoint_info(args.name)
        target = ROOT / spec["path"]
        if target.exists():
            print(json.dumps(verify_checkpoint(args.name), indent=2))
            return
        temporary = target.with_suffix(".download")
        urllib.request.urlretrieve(spec["url"], temporary)
        verify_checkpoint(args.name, temporary)
        os.replace(temporary, target)
        print(json.dumps(verify_checkpoint(args.name), indent=2))
        return
    if args.command == "verify":
        from depth_monoia.verification import run
        success = run(args.config, args.dataset_root, args.manifest, args.output,
                      args.gpu, args.reference_predictions, args.reference_metrics)
        sys.exit(0 if success else 2)
    if not args.manifest or not args.run_dir:
        parser.error("training/evaluation require --manifest and --run-dir")
    if args.command == "train":
        if not args.gates:
            parser.error("training requires --gates")
        from depth_monoia.runner import train
        train(args.config, args.dataset_root, args.manifest, args.run_dir, args.gates, args.resume)
    else:
        if not args.checkpoint:
            parser.error("evaluation requires --checkpoint")
        from depth_monoia.runner import evaluate
        evaluate(args.config, args.dataset_root, args.manifest, args.checkpoint,
                 args.run_dir, args.official)


if __name__ == "__main__":
    main()
