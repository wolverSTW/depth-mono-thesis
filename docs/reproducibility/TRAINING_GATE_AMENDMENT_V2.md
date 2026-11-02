# Training-first gate amendment v2 (Phase 25.2)

Status: proposed local change pending review/commit; NOT real GPU verification.

## Scope

- `training_preflight`: real G1, G2 and G3 must PASS for the exact source/config/data identity.
- G4 is **not run** in this scope, remains `BLOCKED` in the gate report and is
  recorded as `PENDING_INDEPENDENT_G4` in training metadata.
- `full_evaluation`: original G1–G4 PASS contract is unchanged and remains the
  default for direct `research.py verify` invocations and final `runner.evaluate`.
- Neither synthetic reference metrics nor fabricated G4 PASS is permitted.
- No silent dataset, batch, precision, optimizer, loss, architecture or split changes.
- Every training epoch's validation scores are provisional until independently verified.
- Source + identity + initialization asset checks and full-state resume are retained.

## Scientific limitation

Running E0–E4b training is NOT the same as confirming benchmark reproduction.
No final AP3D/APBEV thesis claims before real-data independent evaluator agreement,
checkpoint provenance review and valid validation protocol.

## Operational contract

On Vast.ai after approved source commit, dataset setup, manifest and GPU environment:

```bash
bash scripts/vast/train_a0.sh --verify-only
# Only if real G1/G2/G3 PASS and batch/capacity is approved:
bash scripts/vast/train_a0.sh --run-name e0_full_kitti
```

Both commands run the exact configured-batch CUDA preflight. The second command
starts actual training only after passing. G4 remains BLOCKED.
Do not use this protocol's training gate report to satisfy final evaluation.
