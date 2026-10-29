# Depth-MonoIA research engineering instructions

## Purpose and session start

Build reproducible MonoIA/DA-V2 experiments to test depth-prior fusion. Improvement
and novelty are hypotheses. This existing project is authorized for continuation.
Do not restart it or touch neighboring thesis projects unless explicitly asked.

At each new session read this file, PROJECT_STATUS.md, then only the documents
relevant to the next incomplete milestone. Architecture: docs/ARCHITECTURE.md;
execution: docs/GPU_TRAINING_GUIDE.md; protocol: docs/EVALUATION_PROTOCOL.md.

## Scientific constraints

- Use real KITTI only. No dummy training datasets, invented metrics, copied
  published results presented as local evidence, or validation/test label leakage.
- Preserve pinned submodules, original A0 architecture, intrinsic embeddings,
  losses, optimizer, scheduler, augmentation and benchmark protocol. A1/A2 omit
  only additional intrinsic conditioning in the new fusion module.
- Freeze official checksum-verified DA-V2-Small initially. Relative depth is not
  metric range. Detection confidence is not distance uncertainty.
- Do not execute A4 before A0–A3 function on real data; the optional legacy A4
  implementation is locked. No unverified performance/novelty claims.

## Implementation and verification

Use Python 3.9-compatible code and existing upstream utilities. Keep new orchestration
outside submodules. Defaults preserve the baseline; explicit batch/seed changes
must be recorded and applied consistently. No silent AMP or multi-GPU changes.
Local development is CPU-only. Vast.ai is the exclusive GPU training provider.
GPU work requires Linux CUDA and verified off-host backups; container storage is
temporary and a Vast volume must not be assumed portable across hosts. Follow
docs/VAST_AI_RUNBOOK.md. Never rent or start paid training without owner approval.

One command, `python scripts/research.py verify`, combines source/import/config
checks and relevant operational tests. Do not repeatedly test unchanged code.
Mandatory gates: G1 provenance; G2 real KITTI forward/backward; G3 full-state resume;
G4 KITTI-compatible evaluation versus independently executed reference evidence.
Never bypass a failed gate. Isolated tests are not thesis results.

## Training and artifacts

Use full designated train split and held-out val. Only real completed validation
selects best.pth by Car AP3D Moderate R40. last.pth is the latest committed epoch
state. Save model/optimizer/scheduler/RNG/config/seed/commit/global step. Resume
must fail on incompatible identity, never silently start at zero. AMP is unsupported.
Checkpoint saving, backup and log writers must share the run artifact lock. Backup
only completed evaluations; restore into a new directory and verify hashes.

## GitHub and editable reports

Remote is https://github.com/wolverSTW/depth-mono-thesis.git. Preserve history and
the develop convention. Review status/diff, sensitive files, exclusions and human
report edits before meaningful commits. Ask the owner before every push; never
force-push. Paid GPU resources also require explicit owner approval.
Datasets, checkpoints, archives, credentials and generated runs stay out of Git.

Read existing reports and relevant diffs before targeted edits. Preserve manual
text, citations and evidence; ask if a change conflicts with manual edits. Keep
PROJECT_STATUS.md and reports synchronized after coherent milestones, not every
minor edit. Missing evidence remains Pending/Not yet evaluated.

## Definition of Done

Distinguish implemented, locally verified, GPU verified, fully trained and evaluated.
Code/prepared commands alone cannot complete M5/M6. Every completed experiment
needs G1–G4, full real training, checkpoint provenance, real evaluation and backed-up
artifacts. End each coherent milestone with checks, reviewed commit, status/blockers
and next action. Continue independent work when external data/GPU blocks a gate.
