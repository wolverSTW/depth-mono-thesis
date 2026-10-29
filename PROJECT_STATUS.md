# Project status

Updated: 2026-10-29. Editable research record; do not replace manual sections.

## Current milestone

M1 complete. M2–M4 source implementation prepared; real-data GPU acceptance is
pending. M5–M6 operational bootstrap implemented and locally checked. Real GPU
execution remains blocked by missing data/hardware/reference evidence.
Branch: `develop`. M1: `c3e7a5691e5155007fe1def666d490ad909331f4`.
Implementation: `006d83e326d91d5dad996e2158cc18575a11e5b4`.
Exact report/final milestone hashes: `git log --oneline`.

## Completed

- Original project folders untouched; new project initialized separately.
- Official MonoIA and Depth Anything V2 cloned as pinned submodules.
- Licenses, official checkpoint revisions/hashes and proposed CUDA dependencies recorded.
- Frozen official DA-V2-Small loading, relative-depth/latent adapters, simple/gated/
  intrinsic-aware fusion and optional flip-consistency reliability implemented.
- A0–A4 configs inherit original benchmark settings; A0 remains unmodified.
- Single entry point prepares four gates, original epoch training, atomic complete
  save/resume, KITTI-compatible evaluation, auxiliary distance/ECE and resources.
- README and six editable reports describe exact commands and evidence limits.
- Continued this existing project with explicit owner authorization on 2026-10-29.
- Project AGENTS.md, root status, requested research/GPU documentation and reports
  organized without discarding prior report content.
- Thirteen GPU wrappers plus a shared dispatcher expose setup, data checks,
  weights, training/resume, evaluation/benchmark and artifact backup/restore.
- New run layout uses best.pth/last.pth, actual global-step/loss events, completed
  evaluation JSON/CSV/raw text and a comparison aggregator with missing values blank.
- Artifact locks, checksum inventories and non-overwriting safe restore implemented;
  recovery identity permits relocating equivalent code/dataset paths.
- No push, paid GPU allocation or training performed.

## Pending and blockers

- G1: real KITTI data and split provenance will be supplied on the GPU machine;
  the user confirmed these are intentionally absent from this laptop.
- Linux CUDA environment and compiled deformable attention are required for G2–G4.
- G4 additionally needs real reference predictions and independently executed
  KITTI R40 metrics; no golden metrics have been invented.
- Full-model GFLOPs require validated custom-op coverage; profiler counts are partial.
- Proposed dependency set is source-compatible in intent, not GPU-validated yet.

## Evidence

Local inspection: Python 3.9.25, PyTorch 2.8.0+cpu, torchvision 0.23.0,
NumPy 2.0.2. CUDA unavailable. This differs from the prepared Linux environment.

Consolidated CPU checks passed syntax/imports, five inherited configs, exact
zero-alpha fusion identity and two-step gradients for all four variants. The
official DA-V2-Small checkpoint hash and strict model load passed; its weights
remain frozen in eval mode. Both upstream source pins are unchanged.
Initial evidence: `docs/reproducibility/INITIAL_LOCAL_VERIFICATION.json`, executed immediately
before the implementation commit. Its dirty flag records then-uncommitted work;
it does not indicate completed KITTI gates. G1–G4 remain BLOCKED/Pending.

Operational review: six local tests and fourteen Bash syntax checks passed during
implementation, including archive round-trip/corruption/traversal/locking and
configuration persistence. Final continuation evidence is saved as
docs/reproducibility/LOCAL_VERIFICATION.json, with hashes of the checked Python files.
The Bash dispatcher also forwarded configuration options successfully. Aggregation
found zero executed evaluations and left all five experiment measurements missing.
No checkpoint, synthetic benchmark table, model result backup or GPU claim was created.

## Important changes for review

New code is confined to depth_monoia/, configs/, scripts/ and tests/. Upstream submodules
remain unchanged. Checkpoints, datasets and generated run outputs are ignored.
The original absolute embedding path and CUDA build architectures are resolved
in external infrastructure. Manual report sections remain editable. Next:
owner review/push approval, then real-data G1–G4 on an authorized Linux CUDA machine.

## Next tasks and actual execution status

Obtain push approval for the reviewed local bootstrap commit. On an authorized
GPU machine acquire real KITTI/split/reference evidence, run G1–G4, reproduce A0,
then train/evaluate A3/A1/A2 and verify real artifact backups. Training: Not started.
Evaluation: Not started. GPU verified: No. Full training completed: No.
