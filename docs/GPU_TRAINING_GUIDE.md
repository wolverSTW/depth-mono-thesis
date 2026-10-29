# Full training, monitoring and evaluation

Vast.ai is the exclusive GPU provider. The [single Vast.ai runbook](VAST_AI_RUNBOOK.md)
gives exact A0/A1/A2/A3 commands, the configured-batch G2/G3 capacity check,
evaluation paths and Windows backup/replacement-instance recovery commands.

Status: commands prepared, not GPU verified or executed. First complete
GPU_SETUP_GUIDE.md and matching G1–G4. Source must be committed/clean. All commands
run from the project root in the activated environment.

Shell wrappers accept `--project-dir`, `--python`, `--gpu-device`; the remaining
options go to the actual Python CLI (`python scripts/research.py --help`). Set
KITTI_ROOT, OUTPUT_ROOT and CHECKPOINT_DIR to real paths as in the setup guide.
Each run has a unique name; existing directories are never overwritten.

```bash
bash scripts/gpu/train_baseline.sh --manifest local.dataset.json \
  --checkpoint-dir "$CHECKPOINT_DIR" --gates results/A0_gates.json \
  --output-root "$OUTPUT_ROOT" --run-name run_001 --gpu-device 0

bash scripts/gpu/train_proposed.sh --manifest local.dataset.json \
  --checkpoint-dir "$CHECKPOINT_DIR" --gates results/A3_gates.json \
  --output-root "$OUTPUT_ROOT" --run-name run_001 --gpu-device 0

bash scripts/gpu/train_ablation.sh --experiment A1 --manifest local.dataset.json \
  --checkpoint-dir "$CHECKPOINT_DIR" --gates results/A1_gates.json \
  --output-root "$OUTPUT_ROOT" --run-name run_001 --gpu-device 0
```

Repeat the ablation command for A2. A4 stays locked. Optional `--config FILE`,
`--batch-size N`, `--workers N`, `--seed N` are implemented, not imagined upstream
arguments. Defaults preserve batch 16/seed 444. An explicit batch/seed change is
recorded under execution.protocol_changes, invalidates old gates and must be
applied consistently. Resolve before running gates:

```bash
python scripts/research.py resolve-config --experiment A3 --workers 4
```

The runner calls original MonoIA Trainer.train_one_epoch and its builder/loss/
optimizer/scheduler; only orchestration/logging/atomic state handling are extended.
Use one GPU per process, float32. Multi-GPU and AMP are unsupported. Do not combine
paid runs without approval or assume the original batch fits a given GPU.

## Monitoring and stopping

Use tmux on the authorized Vast.ai instance, with one training command in the
session (`tmux new -s depth-monoia`). Activate Conda and source local.vast.env in
that shell as the runbook specifies. Monitor artifacts and copy them off-host:

```bash
tail -f "$OUTPUT_ROOT/A3_intrinsic_fusion/run_001/logs/events.jsonl"
tail -f "$OUTPUT_ROOT/A3_intrinsic_fusion/run_001/logs/training.log"
nvidia-smi
df -h "$OUTPUT_ROOT"
```

Events record epoch/global step/actual loss/LR/time/memory/checkpoint saves and
interruptions. Session IDs distinguish replayed steps after an interruption.
First Ctrl+C or SIGTERM requests stop after current epoch validation/checkpoint;
that can take time. A second signal aborts and replays the unfinished epoch from
last.pth on resume. Provider hard-kill deadlines may interrupt any cleanup.
No mid-epoch exact resume is claimed. No completed.json is emitted for early stop.

```bash
bash scripts/gpu/resume_training.sh --experiment A3 --manifest local.dataset.json \
  --checkpoint-dir "$CHECKPOINT_DIR" --gates results/A3_gates.json \
  --run-dir "$OUTPUT_ROOT/A3_intrinsic_fusion/run_001" \
  --resume "$OUTPUT_ROOT/A3_intrinsic_fusion/run_001/checkpoints/last.pth"
```

Resume rejects missing/incompatible state. It never silently starts at epoch zero.
Code/data hashes must match; relocating equivalent source/dataset paths is allowed.
See CHECKPOINT_MANAGEMENT.md for publication and recovery limitations.

## Real checkpoint evaluation and measurement

```bash
bash scripts/gpu/evaluate_model.sh --experiment A3 --manifest local.dataset.json \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --checkpoint "$OUTPUT_ROOT/A3_intrinsic_fusion/run_001/checkpoints/best.pth" \
  --run-dir "$OUTPUT_ROOT/A3_intrinsic_fusion/run_001/evaluation/final_best"

bash scripts/gpu/evaluate_all.sh --search-root "$OUTPUT_ROOT" \
  --manifest local.dataset.json --checkpoint-dir "$CHECKPOINT_DIR"

bash scripts/gpu/benchmark_model.sh --experiment A3 --manifest local.dataset.json \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --checkpoint "$OUTPUT_ROOT/A3_intrinsic_fusion/run_001/checkpoints/best.pth" \
  --run-dir results/A3_resources

python scripts/aggregate_results.py --search-root "$OUTPUT_ROOT" --output results/model_comparison.csv
```

Evaluate-all discovers real best.pth files and uses each run's saved config. It
refuses missing checkpoints or existing output directories, preserving evidence.
Use individual evaluation with a new output directory to rerun deliberately.
Official validation-weight A0 is separate:

```bash
bash scripts/gpu/evaluate_model.sh --experiment A0 --manifest local.dataset.json \
  --checkpoint "$CHECKPOINT_DIR/MonoIA_KITTI_Val.pth" --official \
  --gates results/A0_gates.json \
  --run-dir results/official_A0
```

By default aggregation includes independent checkpoints only; `--category
official_pretrained` writes a separate official-checkpoint comparison. Compare only
equal comparison_group hashes; missing measurements remain blank. Full GFLOPs
stays unavailable until custom CUDA coverage is validated. Epoch AP artifacts do
not imply final distance/calibration/resource evaluation has completed.

Evaluation enforces G1–G4 too: local trained checkpoints embed the accepted gate
record; official A0 needs the explicit gate file. No critical gate is bypassed.
