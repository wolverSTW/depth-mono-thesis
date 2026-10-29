# Recover on another compatible GPU machine

Recovery is prepared, not tested on a GPU. No live trained run is available here.

1. Obtain an authorized Linux machine with the documented CUDA/toolchain stack.
   Restore source at the recorded commit, including pinned submodules. Verify
   `git status`, `git rev-parse HEAD` and `git submodule status`. Do not assume the
   latest develop commit matches an older experiment.
2. Recreate the Python environment/build ops using GPU_SETUP_GUIDE.md. Fetch the
   same official pretrained bytes and acquire the same trusted KITTI content/
   split files/manifest. Rebuilt operator/hardware/software details are recorded;
   cross-hardware bitwise equality is not promised.
3. Transfer the archive and sidecar, then restore into a new directory:

```bash
bash scripts/gpu/restore_artifacts.sh \
  --archive /backups/A3_run001_snapshot01.tar.gz \
  --destination /persistent/experiments/A3_intrinsic_fusion/run_001
```

4. Inspect RESTORE_PROVENANCE.json, metadata.json, checkpoint_hashes.json and
   training_state.json. Do not remove genuine OS-held locks. Locks release after
   process death; a remaining lock file alone is harmless. Do not resume a run
   that is still training elsewhere.
5. Rerun G1–G4 on the target environment with the same config/overrides and real
   reference evaluation evidence, using paths appropriate to the new machine.
   Run identity excludes machine-specific source/dataset path prefixes while
   retaining code/data/config/weight hashes. Hardware metadata remains distinct.
6. Resume explicitly from checkpoints/last.pth with the restored saved config:

```bash
bash scripts/gpu/resume_training.sh \
  --config /persistent/experiments/A3_intrinsic_fusion/run_001/config.yaml \
  --dataset-root /datasets/KITTIDataset --manifest local.dataset.json \
  --checkpoint-dir /persistent/pretrained --gates results/A3_recovery_gates.json \
  --run-dir /persistent/experiments/A3_intrinsic_fusion/run_001 \
  --resume /persistent/experiments/A3_intrinsic_fusion/run_001/checkpoints/last.pth
```

An unfinished epoch is replayed from the last committed epoch. Session IDs explain
repeated global step numbers in surviving partial logs. Missing or mismatched
checkpoint/code/data identity fails; never silently start over or change hashes.
If a publication/hash mismatch exists, recover the previous verified snapshot
instead of guessing which files are consistent. Back up again after recovery.
