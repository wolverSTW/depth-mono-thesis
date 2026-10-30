# Preliminary training: implementation and reproducibility review

Date: 2026-10-30. Branch: `develop`. Reviewed parent:
`918fa22346d63a766b4568457b8e960fdd488307` (`918fa22`).
Status: implemented and CPU tested; human review pending. No real training,
KITTI download, GPU operation, paid resource, commit or push was performed.

## F1 follow-up: signal before the requested boundary

Corrected the status condition: preliminary_stopped requires the committed epoch
to have reached stop_after_epoch. SIGINT or SIGTERM after committed epoch 7 with
stop_after_epoch=20 now records `status: stopped`, `next_epoch: 7`, and the actual
signal name in stop_reason. The requested boundary remains recorded as 20.
Normal committed epoch-20 stopping still records preliminary_stopped; epoch-250
completion is unchanged. The stop_reason assignment and all checkpoint/config/
identity-computation/gate logic are unchanged; existing source-hash checks apply.

Two new regression tests exercise SIGINT and SIGTERM through the installed
handlers at epoch 7. They check that only epochs 1–7 commit, status is stopped,
the signal name survives in state/events, and completed.json is absent. Existing
normal epoch-20 and full-horizon tests also pass. The complete current suite has
**43 passing tests**, including 14 session tests; all 23 Bash syntax checks pass.
All five resolved configurations and the prior 16 protected hashes were compared
again and are unchanged. No real detector training or benchmark was performed.

Exact follow-up changes: `depth_monoia/training_session.py` (status condition),
`tests/test_training_session.py` (two regressions), this report and
`PROJECT_STATUS.md`; new actual CPU evidence:
[PRELIMINARY_TRAINING_F1_LOCAL_VERIFICATION.json](reproducibility/PRELIMINARY_TRAINING_F1_LOCAL_VERIFICATION.json).
The prior 41-test evidence and configuration-comparison report are preserved.
No dataset download, GPU operation, staging, commit or push. G1–G4 remain BLOCKED.

## Behavior and integration

`--stop-after-epoch 20` is an optional absolute one-based session boundary.
The loop still uses the configured 250-epoch horizon. The existing per-epoch
callback performs training, scheduler/warmup stepping, real validation inference,
R40 evaluation, best/last checkpoint publication, hash receipt, evaluation metadata,
training-state publication and checkpoint event before returning to the boundary
controller. Only then can the session stop. Failure in that callback does not
publish a successful preliminary stop or full completion.

At epoch 20, training_state.json records `preliminary_stopped`,
`stop_reason: stop_after_epoch`, `stop_after_epoch: 20`, `next_epoch: 20`, and
the actual global step/best score already committed by the runner. Events record
the session boundary and reason. completed.json is absent until epoch 250
actually commits. Resume without the boundary continues at epoch 21. A later
absolute boundary, e.g. 40, is supported. At boundary 250 normal completion applies.

Boundaries outside integer 1–250 are rejected. Resume at or beyond the requested
boundary is rejected before an epoch runs or checkpoint/status files change;
omit the boundary or choose a later one. Completed runs cannot be rewound in
place, preventing an old completed.json from describing a partial continuation.

The CLI forwards the option separately from configuration overrides. `verify`
accepts it for boundary validation only, so the Vast wrapper can reject invalid
values before GPU verification. It does not truncate or relax G1–G4. All four
train_aN wrappers already delegate to train.sh; that shared parser required an
update. The generic scripts/gpu/workflow.sh already forwards arguments unchanged.
Four local wrapper --help calls passed. No training command was executed.
See [training commands](GPU_TRAINING_GUIDE.md) and
[Vast runbook](VAST_AI_RUNBOOK.md).

## Configuration and source comparison

| Setting | Reviewed A0–A3 | After implementation |
| --- | --- | --- |
| Configured full horizon | 250 epochs | Identical |
| Optimizer | Original custom AdamW, LR 0.0002, weight decay 0.0001 | Identical |
| Scheduler | Original LambdaLR/step policy, decays 85/125/165/205, factor 0.5 | Identical |
| Warmup | Disabled | Identical; optional implementation state preserved |
| Seed / batch / precision | 444 / 16 / float32, one GPU | Identical |
| Augmentation, loss, architecture | Existing per-experiment MonoIA/fusion configuration | Identical |
| Dataset | kitti_3dop_chen, original pinned split bytes/order | Identical |
| Evaluation / best selection | Existing Car R40 metrics and Moderate 3D selection | Identical |

The before/after comparison found **exact equality of all resolved A0–A3 config
objects**, and unchanged hashes for 16 protected files, including every config,
checkpoint implementation, provenance/evaluation/verification implementation,
protocol pins and both actual split files. No source pin was changed. The existing
source check confirms clean MonoIA and DA-V2 submodules at their recorded revisions.
An AST comparison against reviewed HEAD found the per-epoch callback identical to
the old loop body, except for the stop-signal representation and removal of the
old boundary-control statement into the session controller.

Machine-readable comparison:
[PRELIMINARY_TRAINING_CONFIGURATION_COMPARISON.json](reproducibility/PRELIMINARY_TRAINING_CONFIGURATION_COMPARISON.json).
This verifies configuration/source preservation, not real-data execution.

## Checkpoint and resume guarantees

The unchanged schema-2 writer saves model weights, optimizer state, scheduler
state, optional warmup state (None under the approved configuration), Python/
NumPy/PyTorch/CUDA RNG states, actual global step, next epoch, best value and
configuration/source/data/environment/gate metadata. Temporary-file flush/fsync
and atomic replacement, artifact locking and checkpoint-hash validation remain.
Use last.pth for continuation; best.pth can be from an earlier epoch.

The runtime boundary is absent from saved config and identity inputs; only events
and operational training status record it. Removing/changing it between sessions
therefore preserves experiment identity and gate applicability under the same
reviewed implementation, data and assets. The existing strict resume identity,
checkpoint hash, initialization/operator-asset and G1–G4 checks remain mandatory.
No optimizer or scheduler is restarted on resume: the original builders construct
them and schema-2 loading restores their saved state before the next training epoch.

Implementation code itself remains part of identity. Checkpoints/gates generated
under the earlier 918fa22 implementation are not silently migrated to the updated
code identity. Review/transfer the final implementation before starting preliminary
training, then use that same implementation for resume. This change does not
weaken compatibility checks to accept older/different code.

Individual checkpoints are atomic; the existing best/last/hash/metadata sequence
is not one filesystem-wide crash transaction. A hard crash in that sequence may
require the last verified backup. A crash after commitment but before publishing
the preliminary status can leave epoch_committed in training_state.json; the
verified last.pth is authoritative. Unfinished epochs replay; committed epoch 20
does not replay. No exact mid-epoch or cross-hardware bitwise guarantee is made.
Back up verified artifacts off-host; Vast container storage remains disposable.

## CPU verification and limits

| Check | Result |
| --- | --- |
| Complete CPU regression suite | PASS: 43 tests, including 14 session tests |
| Syntax/import checks | PASS: 22 Python files |
| Inherited configuration controls | PASS: 5 configurations, including locked A4 |
| Bash syntax | PASS: 23 scripts |
| Existing isolated fusion/gradient and official frozen-depth loading | PASS; no KITTI performance claim |
| A0–A3 comparison, protected hashes, actual split checks | PASS |
| Four Vast task help entry points | PASS; argument routing inspected |
| Whitespace/diff check | PASS |
| Remaining implemented CPU failures | None |
| G1–G4 | BLOCKED; no full dataset/CUDA/reference execution |

The existing consolidated command ran the complete suite:

```powershell
& 'C:\Users\swunnthut.wonn\.conda\envs\depth_monoia\python.exe' -B scripts/research.py verify --experiment A0 --output results/PRELIMINARY_TRAINING_F1_LOCAL_VERIFICATION.json
```

Actual CLI exit **2** reflects BLOCKED G1–G4, while all static/unit checks passed.
The raw report was preserved byte-for-byte in
[PRELIMINARY_TRAINING_F1_LOCAL_VERIFICATION.json](reproducibility/PRELIMINARY_TRAINING_F1_LOCAL_VERIFICATION.json).
The original [41-test report](reproducibility/PRELIMINARY_TRAINING_LOCAL_VERIFICATION.json)
remains available as the implementation's earlier snapshot.
Local environment: Windows 10, Python 3.9.25, PyTorch 2.8.0 CPU. This is not the
prepared Linux CUDA environment. Reports honestly record the dirty worktree at
the reviewed parent; no nonexistent implementation commit is invented.

The session tests exercise the actual production boundary controller, CLI routing,
signal handlers and schema-2 save/resume. The scalar-parameter test uses genuine
AdamW state and the pinned upstream scheduler builder, stops/serializes at 20,
resumes at 21 and compares exact weights/optimizer/scheduler states with
uninterrupted scalar updates through epoch 206, crossing all four decay points.
Python/NumPy/torch random draws influence those scalar updates, so the equality
also checks RNG restoration. Optional warmup is exercised using an in-memory
unit-only config copy; experiment files retain warmup disabled.

Coverage includes callback/publication failure, premature completion prevention,
same checkpoint identity/config before and after resume, invalid/reached/passed
boundaries, full-horizon completion, completed-run rewind rejection, first
SIGINT/SIGTERM finishing the current callback, second-signal abort, restoration
of prior handlers, and refusal of failed gate evidence despite a stop argument.
No passing gates, KITTI images/labels/detections or benchmark values are fabricated.
All scalar checkpoint/status fixtures are temporary unit artifacts, never thesis
training/evaluation evidence. Signal handlers are invoked through their installed
Python handlers in these CPU tests; real Linux process/provider interruption and
end-to-end detector resume still require G2/G3.

## Remaining acceptance blockers

- **G1:** complete authentic official KITTI acquisition and full manifest acceptance
  are pending. The local default dataset and local.dataset.json are absent. The
  private source specification was not opened or its credentials exposed here.
- **G2:** actual Linux CUDA build/operators and a real configured-batch
  forward/backward acceptance run; batch-16 fit on RTX 4090 24 GB remains unknown.
- **G3:** full-state next-step agreement on real KITTI in the actual GPU runtime.
- **G4:** independently executed R40 reference metrics and real predictions for
  the pinned validation split; no copied table or synthetic benchmark substitute.
- Human review, committed/clean source transfer and explicit owner approval remain
  required before paid Vast operations. Full training and evaluation remain unstarted.

Exact original MonoIA split equivalence remains unverified. The approved named
published kitti_3dop_chen protocol is preserved. GFLOPs custom-op coverage remains
a later reporting limitation, unrelated to this session stop feature.

## Original preliminary implementation file inventory (before F1)

Modified:

- `depth_monoia/runner.py`
- `scripts/research.py`
- `scripts/vast/train.sh`
- `docs/GPU_TRAINING_GUIDE.md`
- `docs/CHECKPOINT_MANAGEMENT.md`
- `docs/VAST_AI_RUNBOOK.md`
- `PROJECT_STATUS.md`

Created:

- `depth_monoia/training_session.py`
- `tests/test_training_session.py`
- `docs/PRELIMINARY_TRAINING_REVIEW.md`
- `docs/reproducibility/PRELIMINARY_TRAINING_LOCAL_VERIFICATION.json`
- `docs/reproducibility/PRELIMINARY_TRAINING_CONFIGURATION_COMPARISON.json`

The two pre-existing untracked review patches are preserved. No config,
checkpoint schema, downloader, source pin or earlier evidence report is edited.
Nothing staged, committed or pushed. Changes await human review.
