# Reproducibility record

Updated: 2026-10-29. Reports remain human-readable and manually editable.

## Source history

Branch: `develop`. M1: `c3e7a5691e5155007fe1def666d490ad909331f4`.
Later milestone hashes: `git log --oneline`. Executed runs save their parent
commit, branch, dirty state and extension-code hashes.
Implementation commit: `006d83e326d91d5dad996e2158cc18575a11e5b4`.
Operational bootstrap commit: `ed871c0d46d84fb122f6fe07ac3c428dd739168c`.
The latest local verification's Python hashes match this implementation; its
parent-commit/dirty flag records the real check before milestone commit creation.
MonoIA: `69d6ee30ca5e44848727387dd44af6d56d62c24f`.
DA-V2: `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`.
Both remain unmodified. Bootstrap 89c6b99 was pushed with owner approval; Vast
preparation 4cf0f97 remains local. RTX 4090 tasks reached 50eac764; published-split
and F3 updates remain uncommitted for human review.

## Software and hardware

Local laptop: Windows CPU, existing depth_monoia Conda environment; no packages
were installed/changed. Python 3.9.25, torch 2.8.0 CPU, torchvision 0.23.0 and
NumPy 2.0.2 are local versions, not upstream CUDA-compatibility evidence.
`docs/reproducibility/INITIAL_LOCAL_VERIFICATION.json` records the original consolidated
check and its commit/state at execution.
The check ran immediately before the implementation commit, while the tested
source was staged; its dirty flag is preserved as recorded.
The latest continuation check is recorded separately as LOCAL_VERIFICATION.json.
The existing project was continued with explicit owner authorization, retaining
its Git history and prior reports. Current workflow commits are in `git log`.

Prepared GPU stack: Linux, Python 3.9, torch 2.4.1, torchvision 0.19.1, CUDA toolkit
12.1, NumPy 1.26.4; remaining pins in requirements/linux-cu121.txt. GPU model,
driver/compiler and actual resolved GPU environment are Pending. Runs record
packages, GPU name, CUDA/cuDNN, backbone-cache and compiled-operator hashes.
The selected target is now Vast.ai RTX 4090 24 GB. G2/G3 at the intended batch,
not the advertised card size, determines initial capacity acceptance. Gate reports
record actual peaks and explicit OOM; no automatic batch/precision fallback.

## Data and checkpoints

KITTI data acquisition: Pending. The owner-approved `kitti_3dop_chen` published
split source, raw hashes and all 7481 IDs are verified; see KITTI_3DOP_PROTOCOL.md
and reproducibility/kitti_3dop_chen.json. Exact original MonoIA split equivalence
remains unverified. Preparation generates local.dataset.json only after actual
official acquisition and complete dataset checks. Validation labels never
enter training. Identity includes split hashes, annotation/calibration inventory,
config, upstream/code hashes and prior checkpoint hash. Full image authentication
relies on trusted acquisition; images receive coverage/size checks.
The new scripts/vast/prepare_kitti.sh generates this same manifest from real
official-server downloads and verified pinned published splits. Archive receipts record
SHA-256/CRC and distinguish expected-hash verification from a recorded hash of an
authorized HTTPS download. Signed URLs/cookies/netrc credentials are excluded.
Verified staged receipts precede no-clobber final archive publication and recover
matching staged/final bytes on rerun. Unknown caches and identity conflicts fail.
Protocol ID and definition hash participate in run/gate identity; old gates and
checkpoints cannot silently become evidence for a different split protocol.

DA-V2 Small official bytes match SHA-256
`715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`.
All immutable official checkpoint sources are recorded in sources.json. MonoIA
validation-weight evaluation is Pending. Only trusted local generated full-state
pickle checkpoints may be loaded. Weight provenance is not training completion.

## Seeds, commands and recovery

README.md contains exact setup/verify/train/resume/evaluate commands. Seed 444
uses the original powered Python/NumPy/torch/CUDA policy and epoch NumPy reseeding.
Workers restart per epoch as upstream. Atomic epoch-boundary recovery replays
unfinished work from the last checkpoint. G3 checks numerical agreement of the
next step, not promised bitwise reproducibility across different hardware.

G1–G4 all remain Pending here. Their evidence must match code/config/data/prior
identity before full training. Isolated tensor checks are not KITTI evidence.
Run/gate metadata explicitly records batch, effective batch, one device,
accumulation=1, optimizer/scheduler, seed, precision and protocol deviations.
If batch 16 OOMs, an explicit reduced batch must be shared by all compared variants
and labeled a deviation from the original training protocol; accumulation/AMP
are not silently added. See VAST_AI_RUNBOOK.md for task commands and safe resume.
No training, benchmark claim, test submission or cloud resource was started.
Archive actual configs, logs, gates and checkpoint hashes with each real result.

## Operational continuation

GPU_SETUP_GUIDE.md, GPU_TRAINING_GUIDE.md, CHECKPOINT_MANAGEMENT.md,
ARTIFACT_DOWNLOAD_GUIDE.md and GPU_RECOVERY_GUIDE.md document the implemented CLI
and wrappers. Local tests use genuine source-file bytes to exercise backup/
restore, corruption/path rejection and OS locking; they do not create fake model
checkpoints or KITTI results. Linux deployment/full-state resume still require G2/G3.
Gates now also record actual backbone cache and imported operator hashes. Training
rejects changed bytes since verification. Source/data root relocation is allowed
without weakening content/config identity. Runtime batch/seed changes are explicit.
