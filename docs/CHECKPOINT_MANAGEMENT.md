# Checkpoints and run artifacts

For the exact Vast.ai paths/commands and off-host recovery procedure, see
[VAST_AI_RUNBOOK.md](VAST_AI_RUNBOOK.md), steps 8–10. Each experiment has its own
run directory. New runs refuse existing destinations; best/last in the active run
advance intentionally, while previous runs and named backup snapshots remain intact.

No training checkpoints exist yet. This layout is created only by real execution:

```text
experiments/A3_intrinsic_fusion/run_001/
  metadata.json
  config.yaml
  checkpoint_hashes.json
  training_state.json
  checkpoints/best.pth
  checkpoints/last.pth
  logs/training.log
  logs/events.jsonl
  evaluation/epoch_001/predictions/...
  evaluation/final_best/
    predictions/...
    metrics.json
    metrics.csv
    kitti_eval.txt
    distance_metrics.json
    confidence_metrics.json
    resource_metrics.json
    evaluation_metadata.json
```

best.pth is selected after completed validation by Car AP3D Moderate R40 (highest
score; ties retain earlier best). last.pth stores the last committed epoch state.
Extensions remain .pth. Schema 2 includes model/optimizer/scheduler/optional warmup,
next epoch, global step, best metric, complete RNG and configuration/source/data/
checkpoint/environment metadata. AMP is unsupported, so no scaler state exists.

Write a temporary file, flush/fsync, then replace atomically. Run artifact locks
coordinate checkpoint/hash publication, logs and backup; a separate training lease
prevents simultaneous resumes into one run. Backup refuses hash-manifest mismatch,
including a hard crash during multi-file publication. Individual checkpoint files
are atomic; best/last/hash publication is not a filesystem-wide crash transaction.
Use the last verified archive if publication was interrupted. Do not edit hashes
to bypass the failure. Exact mid-epoch continuation is unsupported.

G3 tests save/resume and next-step agreement on real KITTI. It is Pending locally.
No placeholder best/last/metrics files are created. Full-state loading requires a
trusted run; official MonoIA weights are authenticated against official hashes
before their pickled metadata can be loaded. Raw DA-V2 weights use weights-only.

Epoch evaluations save AP and raw predictions. Full evaluation adds distance,
confidence and resources and publishes evaluation_metadata.json last. Backup and
comparison include completed evidence; unfinished directories stay available for
inspection and are excluded. Retained per-epoch predictions can be large: monitor
storage and archive before any approved pruning.
