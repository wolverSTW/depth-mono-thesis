# Project status

Updated: 2026-10-29. Editable research record; do not replace manual sections.

## Current milestone

M1 complete. M2–M4 source implementation prepared; real-data GPU acceptance is
pending. M5–M6 operational bootstrap implemented and locally checked. Real GPU
execution remains blocked by missing data/hardware/reference evidence. Vast.ai
is the exclusive training provider; deployment preparation is complete locally,
with configured-batch Linux CUDA acceptance pending.
Branch: `develop`. M1: `c3e7a5691e5155007fe1def666d490ad909331f4`.
Implementation: `006d83e326d91d5dad996e2158cc18575a11e5b4`.
Operational bootstrap: `ed871c0d46d84fb122f6fe07ac3c428dd739168c`.
Approved bootstrap push: `89c6b999098b93f9c7b20e46c72bd2de03dbbb20`,
verified at origin/develop. Vast preparation continues from that commit; its exact
local milestone hash is in `git log -1` (no new push authorized/performed).
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
- Bootstrap develop commit 89c6b99 pushed with approval and remote SHA verified.
  No paid GPU allocation or training performed.
- Vast-specific beginner runbook and links in existing setup/training/checkpoint/
  download/recovery guides added without replacing manual research reports.
- G2/G3 now use the configured real training batch (default 16), rather than one
  sample. G2 records input/dtype/device and peak allocated/reserved CUDA bytes.
  The initial eval outputs are released before measuring training peak memory.
- Setup rejects a non-12.1 toolkit early, requires the developer toolkit/NVVM and
  limits default extension build parallelism to four jobs. Source pins unchanged.

## Pending and blockers

- Before rental: obtain real KITTI archives, trusted original split files/hash
  provenance and independent G4 prediction/metric evidence off-host. These inputs
  are intentionally absent from the CPU laptop; no random splits are substituted.
- Make the exact new preparation commit available via separately approved push
  or a Git bundle before deployment. Do not deploy 89c6b99 expecting the new gate.
- Owner approval of an actual Vast offer/budget is required before paid resources.
- Linux CUDA environment and compiled deformable attention are required for G2–G4.
- G4 additionally needs real reference predictions and independently executed
  KITTI R40 metrics; no golden metrics have been invented.
- Full-model GFLOPs require validated custom-op coverage; profiler counts are partial.
- PyTorch's published torch 2.4.1/vision 0.19.1 cu121 pairing, CUDA 12.1 driver
  requirements, Numba 0.60 toolkit requirements and the versioned developer image
  were checked against primary documentation. Actual build/runtime still pending.
- Planning only: full A100 80 GB, allocated 8–16 vCPUs/64 GB RAM, 300 GB disk;
  low-confidence 32–64 GiB VRAM allowance, not a measured bound. 24 GB fit is not
  established. See VAST_AI_RUNBOOK.md for image digest, sources and caveats.

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

Vast preparation checks: consolidated CPU verification again passed 17 Python
syntax/import checks, five inherited configs, six operational tests, fourteen
Bash syntax checks, isolated fusion identity/gradients and strict frozen official
DA-V2 loading. Evidence: docs/reproducibility/VAST_LOCAL_VERIFICATION.json. The
nonzero overall verify result correctly reflects BLOCKED G1–G4, not a GPU pass.
Five PowerShell and fourteen Bash runbook blocks also parsed without executing
their deployment/training commands. No new verification infrastructure added.

## Important changes for review

New code is confined to depth_monoia/, configs/, scripts/ and tests/. Upstream submodules
remain unchanged. Checkpoints, datasets and generated run outputs are ignored.
The original absolute embedding path and CUDA build architectures are resolved
in external infrastructure. Manual report sections remain editable. Next:
owner review, acquire real inputs, transfer the recorded preparation commit, then
real-data G1–G4 on an approved Vast Linux CUDA machine.

## Next tasks and actual execution status

The previous bootstrap push is complete. Review the local Vast preparation commit;
publish only with separate approval or transfer via Git bundle. Once real inputs
and paid-operation approval are available, execute VAST_AI_RUNBOOK.md: build and
run configured-batch G1–G4, reproduce A0, then train/evaluate A3/A1/A2 and transfer
verified backups off-host. Complete custom-op GFLOPs coverage before reporting a
full-GFLOPs thesis comparison; this does not block training. Training: Not started.
Evaluation: Not started. GPU verified: No. Full training completed: No.
