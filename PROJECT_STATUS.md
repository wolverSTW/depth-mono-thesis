# Project status

Updated: 2026-10-29. Editable research record; do not replace manual sections.

## Current milestone

M1 complete. M2–M4 source implementation prepared; real-data GPU acceptance is
pending. M5–M6 operational bootstrap implemented and locally checked. Real GPU
execution remains blocked by missing data/hardware/reference evidence. Vast.ai
is the exclusive training provider. Current target is RTX 4090 24 GB; one-command
task workflow is implemented locally, with real configured-batch Linux CUDA
acceptance pending. The RTX 4090 workflow is now committed at `50eac764a015e696f3deba3efe563470e42407f7`;
the source/split audit and published 3DOP protocol/F3 changes below remain local
and uncommitted for human review.
Branch: `develop`. M1: `c3e7a5691e5155007fe1def666d490ad909331f4`.
Implementation: `006d83e326d91d5dad996e2158cc18575a11e5b4`.
Operational bootstrap: `ed871c0d46d84fb122f6fe07ac3c428dd739168c`.
Approved bootstrap push: `89c6b999098b93f9c7b20e46c72bd2de03dbbb20`,
verified at origin/develop. Vast preparation continues from that commit; its exact
local milestone hash is `4cf0f97b954e89cefc023d589e465c8e4467e328` (not pushed).
The RTX 4090 workflow subsequently reached `50eac764a015e696f3deba3efe563470e42407f7`
before this audit session. No commit/push is performed by the audit.
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
- Seven scripts/vast task entry points use the existing research.py/modules:
  setup, prepare_kitti, train_a0/a1/a2/a3, evaluate_all. Shared shell helpers only
  manage paths, Conda and command routing; no model/training code is copied.
- Direct official-server KITTI acquisition supports operator-authorized HTTPS
  URLs, cookie/netrc authentication, curl range resume, CRC/recorded SHA-256 and
  optional trusted expected archive hashes. Matching cached files are reused;
  unsafe ZIP members/conflicting existing files fail. The approved published
  kitti_3dop_chen split now has verified immutable sources/hashes; the manifest is generated only
  after real dataset verification. No real dataset download has occurred here.
- Training task commands run existing G1–G4 at the intended batch before full
  training; --verify-only stops after gates. CUDA OOM saves memory/config evidence,
  exits and leaves batch/precision unchanged. Timestamped run names isolate runs;
  --resume uses the saved config and existing full-state/checkpoint hash checks.
- Gate and run metadata explicitly record per-device/effective batch,
  accumulation=1, optimizer/scheduler, seed and protocol changes. AMP and gradient
  accumulation are unsupported; model architecture/source pins remain unchanged.

## Pending and blockers

- Before rental: obtain official KITTI download authorization/URLs and independent
  G4 prediction/metric evidence for the pinned kitti_3dop_chen validation hash.
  KITTI will download directly on the GPU server; laptop data is not required.
  Published split provenance is verified. Exact original MonoIA split equivalence
  remains unverified, a scope limitation rather than a blocker for the approved protocol.
- Review the uncommitted published-protocol/F3 changes. No automatic commit/push.
  Transfer that exact reviewed commit via separately approved push or Git bundle before
  training. 50eac764 contains scripts/vast, but not these new protocol/F3 changes.
- Owner approval of an actual Vast offer/budget is required before paid resources.
- Linux CUDA environment and compiled deformable attention are required for G2–G4.
- G4 additionally needs real reference predictions and independently executed
  KITTI R40 metrics; no golden metrics have been invented.
- Full-model GFLOPs require validated custom-op coverage; profiler counts are partial.
- PyTorch's published torch 2.4.1/vision 0.19.1 cu121 pairing, CUDA 12.1 driver
  requirements, Numba 0.60 toolkit requirements and the versioned developer image
  were checked against primary documentation. Actual build/runtime still pending.
- User-selected target: RTX 4090 24 GB, allocated 8–16 vCPUs/64 GB RAM and 300 GB
  disk planning budget. Original batch 16 fit is unknown; run actual G2/G3 before
  full training. Explicit batch reduction changes the original protocol and must
  be applied equally to comparisons, not presented as an exact reproduction.
  No measured total VRAM estimate or GPU fit claim is made.

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

RTX 4090 task-workflow evidence: docs/reproducibility/RTX4090_LOCAL_VERIFICATION.json
records the current CPU-safe consolidated run. It checks 19 Python files/imports,
five configs, 13 operational tests, 23 Bash scripts and existing isolated fusion/
strict official-depth loading. New acquisition safety tests use only genuine
repository source bytes (no dummy KITTI or model results). Seven task --help
invocations and four config targets were checked; 16 Bash and four PowerShell
runbook blocks parsed. Local verify's nonzero result denotes BLOCKED G1–G4.
Actual Linux setup, authenticated data download, GPU memory/OOM handling,
full-state GPU resume and real KITTI evaluation remain unexecuted.

## Important changes for review

New code is confined to depth_monoia/, configs/, scripts/ and tests/. Upstream submodules
remain unchanged. Checkpoints, datasets and generated run outputs are ignored.
The original absolute embedding path and CUDA build architectures are resolved
in external infrastructure. Manual report sections remain editable. Next:
owner review, acquire real inputs, transfer the recorded preparation commit, then
real-data G1–G4 on an approved Vast Linux CUDA machine.

## Next tasks and actual execution status

The previous bootstrap push is complete. Review the uncommitted protocol/F3 changes;
do not commit or push until review. Once the reviewed source, authentic inputs
and paid-operation approval are available, execute VAST_AI_RUNBOOK.md: setup,
direct official KITTI preparation, intended-batch G1–G4, A0 under the published
3DOP/Chen split (or an explicitly labeled reduced-batch comparison), then
A3/A1/A2 and evaluation. Transfer
verified backups off-host. Complete custom-op GFLOPs coverage before reporting a
full-GFLOPs thesis comparison; this does not block training. Training: Not started.
Evaluation: Not started. GPU verified: No. Full training completed: No.

## KITTI source and split audit — 2026-10-29

See docs/KITTI_SOURCE_AND_SPLIT_AUDIT.md for the evidence matrix, exact missing
inputs, proposed fixes and acceptance criteria. Audited HEAD: 50eac764a015e696f3deba3efe563470e42407f7.
The prior RTX 4090 changes were already committed when this session began; the
earlier uncommitted-work notes describe the preceding implementation session.

Verified: original loader paths, three official package categories and official
registration/download policy. The pinned evaluator README contains a real MV3D
split citation, but a header-only request to that exact archive returned 404.
The inspected MonoIA/MonoCoP/MonoDETR/MonoDGP trees contain no split lists; no
explicit original-use confirmation or exact author-verified split hashes were
found in the inspected paper/site/models/issues. Exact MonoIA train/val provenance
remains BLOCKED; matching 3712/3769 counts is insufficient evidence.

Source-compatible extraction and ordinary rerun checks do not prove real-archive
compatibility. An offline source-byte probe confirmed a crash-recovery gap between
final archive publication and verification receipt: safe refusal on retry when no
expected hash exists. The report proposes a staged receipt/publication fix; no
downloader/model code or configuration was changed during this audit.

No KITTI dataset/split payload downloaded locally, authenticated login attempted,
GPU operation started, author contacted, commit created or push performed. The
audit adds only the report and this status update. Real acquisition/extraction,
G1–G4 and original-protocol reproduction remain unverified/blocked.

## Approved published split and F3 implementation — 2026-10-29

The owner subsequently approved `kitti_3dop_chen`: the published 3DOP/Chen
research partition, not an official KITTI train/val split or a claim of exact
original MonoIA reproduction. This supersedes the missing-original-split
requirement for the approved comparisons; the historical audit above is preserved.

Source VERIFIED: original Frustum PointNets author repository, commit
`2ffdd345e1fce4775ecb508d207e0ad465bcca80`, kitti/image_sets/train.txt and val.txt.
Paper attribution follows Frustum PointNets → MV3D → 3DOP, corroborated by
CenterNet's explicit 3DOP link and byte-identical OpenPCDet files at an immutable
revision. Actual raw lists verify 3712/3769 unique six-digit IDs, disjoint and
covering 000000–007480. LF separators/no final newline and original ordering are
preserved. The Frustum PointNets supplement's inconsistent 3717 count is recorded.
Exact source paths, SHA-256, blob identities and caveats:
docs/KITTI_3DOP_PROTOCOL.md and docs/reproducibility/kitti_3dop_chen.json.

Implemented: explicit protocol in the shared A0–A3 configuration, immutable
automatic split acquisition, pinned manifest validation, protocol-aware run/gate
identity and metadata. Old unspecified manifests/gates/checkpoints cannot be
silently relabelled. Official image/calibration/3D-label acquisition remains the
only dataset path; no annotation conversion. Architecture, optimizer, losses,
schedule, epochs, batch and metrics are unchanged.

F3 repaired locally: staged ZIP safety/layout/CRC and hash verification precede a
durable pending receipt, atomic no-overwrite file publication and atomic complete
receipt. Matching staged/final bytes recover after publication interruptions;
unknown caches, conflicting hashes and existing-file conflicts are refused.
Legacy valid cache receipts remain usable. Ordinary reruns retain manual manifest
fields and verified extracted files. Real official-archive extraction remains pending.

CPU evidence: docs/reproducibility/KITTI_3DOP_LOCAL_VERIFICATION.json records 20
Python syntax files/imports, five inherited configurations, **27 passing tests**,
23 Bash syntax checks, existing fusion identity/gradient and strict frozen official
DA-V2 loading checks. Both upstream pins remain clean/unchanged. A separate actual
configuration comparison confirms only dataset.protocol differs from HEAD's
resolved common config. Executed split checks are in
docs/reproducibility/KITTI_3DOP_SPLIT_VERIFICATION.json. The consolidated command
exits 1 because G1–G4 correctly remain BLOCKED; this is not a GPU pass.
See docs/KITTI_3DOP_VERIFICATION.md for PASS/FAIL/BLOCKED acceptance details.

Remaining inputs: authorized official archive URLs/session access; real acquired
dataset/manifest for G1; reviewed source transfer; independent real G4 predictions
and R40 reference values for the exact validation hash; approval of paid Vast work;
actual Linux CUDA build and configured-batch G2/G3 acceptance. Batch-16 fit on
24 GB remains unknown. Full GFLOPs coverage remains a reporting limitation.
Exact original MonoIA membership and validation-checkpoint overlap remain unverified.

Only the small source ID lists were acquired into ignored local data storage.
No KITTI image/label/calibration/archive downloaded to Windows; no fabricated
predictions/metrics, paid instance, GPU training, commit or push. Branch/HEAD
remain develop/50eac764. Changes await human review.

## F3-01 conditional-review correction — 2026-10-29

Fixed the valid-final-archive branch bypassing a conflicting `.part`: staging
SHA-256 and byte size must now match the receipt before either recovery path.
Conflicts preserve both files and receipt without transfer, deletion or overwrite.
Identical staged/final bytes still recover, including an interrupted hard-link
publication. Complete, pending and legacy verification receipts remain supported;
unknown archives cannot be accepted on ZIP CRC alone.

Complete CPU suite/static verification: **29 tests passed**, 20 Python syntax
files/imports, five configurations, 23 Bash syntax checks and existing fusion/
official-depth-load checks passed. New tests cover conflicting and identical
staging beside a valid final archive for complete/pending/legacy receipts;
pending-conflict and interrupted-publication assertions were strengthened.
Actual evidence: docs/reproducibility/KITTI_3DOP_F3_01_LOCAL_VERIFICATION.json.
Exit 1 correctly denotes BLOCKED G1–G4; no CPU check failure or GPU pass.

Before/after hashes confirm the published protocol definition/source pins,
original train/val bytes and ordering, all configurations and baseline training
settings/code are unchanged. Earlier evidence and manual review patch are preserved.
Follow-up changes: downloader, acquisition tests, verification document, this
status file and one new CPU evidence JSON. No commit/staging/push, KITTI archive
download, paid resource or GPU training. Human review remains pending.
Remaining blockers are unchanged: official dataset authorization/acquisition/G1,
independent real G4 reference evidence, reviewed source transfer and owner approval,
then actual Linux CUDA build, configured-batch capacity and full-state resume/G2–G3.
Exact original MonoIA split equivalence remains unverified; the approved research
protocol remains kitti_3dop_chen.
