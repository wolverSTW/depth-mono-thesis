# Vast.ai RTX 4090: full KITTI deployment runbook

Prepared 2026-10-29. Vast.ai is the exclusive training provider. These commands
have been inspected locally, **not executed on a GPU**. Do not rent an instance
or execute paid GPU work until the owner approves it. Continue the existing
`develop` project; do not initialize another project or change the pinned sources.
Current reference: `4cf0f97b954e89cefc023d589e465c8e4467e328`. The new task wrappers
are uncommitted pending owner review; they must be reviewed/committed and then
published with approval or transferred with history before full training.

## 1. Before paying

Have these inputs ready off the GPU machine:

- Authorized official download URLs for KITTI object-detection left RGB,
  calibration and training labels. The GPU server downloads these directly;
  Windows dataset upload is not required. Obtain links/authentication through the
  [KITTI account/download page](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
  No URLs are guessed and no login is bypassed. LiDAR, stereo/right RGB, raw KITTI
  and temporal images are unnecessary.
- Original designated `train.txt` and `val.txt`, their independently checked
  provenance and SHA-256 hashes. Intended counts are 3712/3769, disjoint and
  covering all 7481 labeled images. The pinned MonoIA repository does not supply
  these files. Do not generate a random split or assume arbitrary online lists
  reproduce the original protocol. Keep their original bytes/line endings.
- `local.kitti_sources.json`, copied from `configs/vast/kitti_sources.example.json`
  and populated as in step 5. Dataset preparation generates the existing dataset
  manifest only after verifying actual data and splits.
- Real validation predictions in KITTI text format and independent KITTI-compatible
  R40 reference measurements for those exact predictions and split. Name them
  `val_predictions/` and `val_reference.json`. The latter must contain
  `reference_source`, `split_sha256`, and `metrics` with all six keys
  `Car_3d_easy_R40`, `Car_3d_moderate_R40`, `Car_3d_hard_R40`,
  `Car_bev_easy_R40`, `Car_bev_moderate_R40`, `Car_bev_hard_R40`.
  Use Car IoU 0.7. Published scores or results from this same evaluator alone
  are not independent reference evidence. This missing input currently blocks G4.
- The reviewed/committed new task-wrapper changes available on GitHub **after
  separate push approval**, or a Git bundle containing them. The current reference
  commit alone does not contain `scripts/vast/`.
- A Windows SSH client/key, sufficient off-host backup space (plan 100 GB free,
  more if retaining many full snapshots), and a reviewed rental budget.

No GPU is needed to obtain authorization and provenance. A full training run requires all four
existing gates; preparing an account/template is not GPU verification.

## 2. Hardware, image and storage selection

Authorized target: **one NVIDIA RTX 4090 24 GB**, Linux x86-64/amd64,
8 allocated vCPUs minimum (16 preferred), 64 GB allocated host RAM, and **300 GB
SSD/NVMe container disk**. Use the full GPU allocation, not a shared-memory offer.
Select an on-demand offer with direct SSH, suitable transfer bandwidth and a
contract duration you can monitor. There is no measured training-time estimate.
Check allocated resources, not the host's advertised total or burst resources.
[RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/),
[Vast execution environment](https://docs.vast.ai/guides/instances/docker-environment).

**VRAM is unmeasured:** selecting a 24 GB card is not evidence that batch 16 fits.
The four retained DA-V2 prior tensors alone are
about 0.42 GiB at the default resolution/batch, excluding detector activations,
DA-V2 workspaces, optimizer, gradients and allocator overhead. A1–A3 run frozen
DA-V2 with microbatch 1 but train the detector with batch 16. There is no defensible
measured total VRAM estimate here. Run the real configured-batch gates on the
RTX 4090 first. On OOM the script saves failure/memory evidence and stops; it
never retries with a smaller batch or AMP. See step 6 for the explicit protocol
tradeoff if a smaller batch is necessary.

300 GB is a capacity budget, not measured usage or unlimited retention:

| Use | Planning allowance |
| --- | ---: |
| Image, isolated environment, build and caches | 50 GB |
| KITTI ZIPs plus extracted files | 35 GB |
| Pretrained and active best/last states | 15 GB |
| Four runs' retained epoch/final predictions and logs | 80 GB |
| One local snapshot and restore staging | 50 GB |
| Free headroom | 70 GB |

Predictions accumulate every epoch. Monitor usage and offload snapshots; retaining
multiple complete archives on-instance can exceed this budget. Vast container
disk is fixed at creation. `/workspace` is just a path, **not durable storage**.
Destroying the instance deletes its container data. A volume is host-local, not
a cross-host recovery solution. Keep verified off-host copies.
[Vast storage rules](https://docs.vast.ai/guides/instances/storage/types).

Create a **custom Vast template**, for example `Depth-MonoIA CUDA12.1`, with:

| Setting | Value |
| --- | --- |
| Docker image | `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel` |
| Launch mode | SSH (not Entrypoint); use direct SSH when offered |
| On-start command | `mkdir -p /workspace; touch /root/.no_auto_tmux` |
| Automatic training/download commands | None |
| Container disk | 300 GB |

The [versioned official image](https://hub.docker.com/layers/pytorch/pytorch/2.4.1-cuda12.1-cudnn9-devel/images/sha256-9057ed8f4550b37a9715e59ff484fa5502a34e94a5159e02d8174f55ca8be29b?context=explore)
is amd64, Ubuntu 22.04, CUDA 12.1.1 developer toolkit and Conda. Registry manifest
digest verified on 2026-10-29:
`sha256:9057ed8f4550b37a9715e59ff484fa5502a34e94a5159e02d8174f55ca8be29b`.
Use that digest as the image reference if supported by the template UI; otherwise
use the exact versioned tag and retain this digest in deployment notes. No local
image pull or Vast boot has been tested. Avoid `latest`/automatic image selection.
Vast's SSH launch mode supplies SSH startup for custom images; Entrypoint mode
does not. [Template settings](https://docs.vast.ai/guides/templates/template-settings).

Use host driver >=530.30.02 as a conservative selection rule for this stack;
CUDA 12.1's minor-compatibility minimum is Linux 525.60.13. Newer compatible
drivers are allowed. Do not install host drivers inside the container.
[NVIDIA CUDA 12.1 release notes](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/index.html).

## 3. Connect from Windows, after rental approval

Create a dedicated key only if you do not already have a suitable key. Upload
only its `.pub` content to Vast's SSH-key settings; keep the private key local.
Copy the instance's SSH hostname and mapped port from Vast's Connect panel.
These values cannot be known before allocation. In Windows PowerShell:

```powershell
$TaskKey = "$env:USERPROFILE\.ssh\depth_monoia_vast"
# Run once if this key does not already exist; choose a passphrase:
ssh-keygen -t ed25519 -f $TaskKey -C depth-monoia-vast
Get-Content "$TaskKey.pub"
$VastHost = Read-Host 'SSH hostname/IP from Vast Connect'
$VastPort = Read-Host 'SSH port from Vast Connect'
ssh -i $TaskKey -p $VastPort "root@${VastHost}"
```

Use the username shown by the Connect panel if it differs from root. Confirm
the host key when connecting; do not disable SSH host-key checking.

## 4. Clone and set up the authorized instance

All Linux commands below run in Bash over SSH as root in the documented developer
image. Stop at any error. The new wrappers must first be reviewed/committed and
made available via approved push or Git bundle; reference commit 4cf0f97 alone
does not contain them.

```bash
set -euo pipefail
apt-get update
apt-get install -y git
mkdir -p /workspace/depth-monoia
cd /workspace/depth-monoia
git clone --branch develop --recurse-submodules https://github.com/wolverSTW/depth-mono-thesis.git source
cd source
read -r -p 'Reviewed deployment commit SHA: ' DEPLOY_COMMIT
git checkout --detach "$DEPLOY_COMMIT"
git submodule update --init --recursive
cp -n configs/vast/environment.example.env local.vast.env
bash scripts/vast/setup.sh
```

Without push approval, create/transfer a Git bundle on Windows **after review and
commit**, preserving existing history. Public pinned submodules fetch separately.

```powershell
git -C .\Depth-MonoIA-Thesis bundle create ..\depth-monoia-develop.bundle develop
scp -i $TaskKey -P $VastPort .\depth-monoia-develop.bundle "root@${VastHost}:/workspace/depth-monoia/"
```

Replace the GitHub clone with the following, using the same new destination,
then perform the same commit checkout/submodule/setup sequence:

```bash
git clone --branch develop /workspace/depth-monoia/depth-monoia-develop.bundle /workspace/depth-monoia/source
cd /workspace/depth-monoia/source
git remote set-url origin https://github.com/wolverSTW/depth-mono-thesis.git
```

`setup.sh` installs build/download tools, creates Python 3.9, installs torch
2.4.1/torchvision 0.19.1 cu121 and requirements/linux-cu121.txt, builds original
CUDA ops for the actual GPU, checks RTX 4090 hardware and records the environment.
It fetches SHA-256-verified DA-V2-Small weights and starts **no training**.
[Official PyTorch pairing](https://pytorch.org/get-started/previous-versions/).
The original builder fetches its original ResNet backbone on first construction
and records the cache bytes. A0 training does not use trained MonoIA val weights.

Each task automatically activates Conda and sources ignored `local.vast.env`.
CUDA_HOME points to `/usr/local/cuda`, including nvcc/NVVM/NVRTC needed by the
Numba KITTI evaluator, not only runtime wheels. Do not install another arbitrary
toolkit. Review any Conda channel terms rather than silently accepting them.
[Numba requirements](https://numba.readthedocs.io/en/0.60.0/cuda/overview.html).
Defaults are derived from TASK_ROOT=/workspace/depth-monoia:

| Data | Path beneath TASK_ROOT |
| --- | --- |
| Source | `source` |
| KITTI | `datasets/KITTIDataset` |
| Download cache | `downloads/kitti` |
| Manifest | `evidence/local.dataset.json` |
| Reference inputs | `evidence/val_predictions`, `evidence/val_reference.json` |
| Pretrained / experiments / environment logs | `pretrained`, `experiments`, `setup_logs` |

## 5. Direct official KITTI download and manifest generation

Prepare the private source specification **on the GPU server**:

```bash
cp -n configs/vast/kitti_sources.example.json local.kitti_sources.json
chmod 600 local.kitti_sources.json
# Edit this file through your SSH editor / VS Code Remote SSH before proceeding.
bash scripts/vast/prepare_kitti.sh
```

Fill these fields before the final command:

- Set `authorized_official_downloads` true only after obtaining official KITTI
  access and reviewing its terms. Copy exact authorized HTTPS ZIP URLs for
  image_2, calib and label_2 from the official account. Record public, nonsecret
  acquisition provenance in `download_source`. No S3 links/mirrors are invented.
- Signed URLs are supported. Alternatively set `authentication.cookie_file`
  (Netscape cookie file) or `netrc_file` to an existing authenticated session file.
  Paths are relative to the JSON directory or absolute. Keep authentication
  files outside Git and `chmod 600`. curl receives URL/auth settings via stdin;
  signed URLs/credentials are suppressed in argv, logs and public metadata.
  Never put secrets in a provenance field.
- Archive `sha256`/`sha256_source` are optional when a trusted expected hash is
  available; if supplied it must match. Otherwise source trust relies on the
  operator-authorized official HTTPS download. SHA-256 is computed and ZIP CRC
  verified **before extraction**, then a verification receipt is saved. A recorded
  hash alone is not independent source authentication. No nonexistent published
  official archive checksum is required.
- Both original splits require a trusted expected SHA-256 and its provenance,
  count 3712/3769, and an immutable trusted `url` or a local `path`. Set
  `split_source` to evidence these are the original designated MonoIA lists.
  Hashing an arbitrary list does not establish original-protocol provenance.
  These split inputs remain missing here; no random split is substituted.

curl uses retries and `.part` range resume where supported. Renew expired signed
URLs in the ignored JSON. Unsupported range resume fails clearly; inspect/preserve
partial files before manually restarting a download. Cached archives reuse a
matching expected hash or previous verified receipt and are not downloaded again.
Existing matching extracted files are retained; conflicting files are not overwritten.
CRC, unsafe ZIP paths/links and disk capacity are checked. Individual files publish
atomically; the existing artifact lock serializes preparation.

All three full official object archives are required. The image/calibration ZIPs
may include testing files; training uses only the designated train IDs and held-out
validation. Expected layout is `training/{image_2,calib,label_2}/000000.*` plus
`ImageSets/{train,val}.txt` under KITTI_ROOT. No test/validation labels enter training.
Preparation calls the existing `verify_dataset`, verifies hashes, split counts,
disjoint coverage of all 7481 labeled IDs and required files, then generates the
existing manifest. It records archive hashes/CRC/public provenance while excluding
URLs/authentication. Extra manual manifest fields survive; identity conflicts fail.
No genuine KITTI download/extraction has been executed on this CPU laptop.

## 6. RTX 4090 capacity and existing G1?G4 gates

Before a long run, explicitly request real-data GPU verification only:

```bash
bash scripts/vast/train_a0.sh --verify-only
bash scripts/vast/train_a3.sh --verify-only
```

G2 exercises the intended batch (16 by default), real forward/backward/optimizer
steps, and records input/GPU/dtype plus peak allocated/reserved bytes in
`details.capacity`. G3 verifies full-state restore/next-step agreement. G4 uses
independent real KITTI R40 reference evidence. Unique reports live under
`/workspace/depth-monoia/evidence/gates/`. CUDA OOM records `CUDA_OOM`, memory and
the unchanged batch; training stops. No automatic smaller-batch/AMP retry occurs.
A representative batch that fits is not a full-run peak guarantee.
If reference evidence is still missing, --verify-only can measure real G2/G3
capacity with a verified dataset, but G4 remains BLOCKED and the command exits
nonzero. It cannot permit full training. A normal training task refuses missing
reference inputs before GPU work.

If batch 16 OOMs, review the scientific tradeoff before explicitly testing a
smaller batch. Eight is an example, **not** a verified fit:

```bash
bash scripts/vast/train_a0.sh --verify-only --batch-size 8
bash scripts/vast/train_a3.sh --verify-only --batch-size 8
```

For a controlled comparison set `export VAST_BATCH_SIZE=8` in local.vast.env and
apply the SAME setting to new A0/A1/A2/A3 runs. Label this a protocol deviation,
not an exact original-batch reproduction. Batch reduction changes update count
and gradient noise; no original-protocol equivalence is claimed. Effective batch
is the configured batch: one GPU, accumulation=1. Optimizer/LR/scheduler remain
original unless separately reviewed; no silent LR scaling, accumulation, seed or
precision change. Metadata records batch/effective batch/optimizer/scheduler/seed/
precision and protocol_changes. Compare equal comparison_group hashes.
A genuine DataLoader shared-memory failure can use explicit `--workers 0` or
`VAST_WORKERS=0`; do not add speculative overrides. Missing split provenance and
independent G4 inputs still block both original and reduced-batch training.

## 7. One task command per experiment

Only on an approved instance with reviewed, committed and clean source:

```bash
tmux new -s depth-monoia
cd /workspace/depth-monoia/source
bash scripts/vast/train_a0.sh
```

Run each remaining task separately, serially on the one GPU:

```bash
bash scripts/vast/train_a1.sh
bash scripts/vast/train_a2.sh
bash scripts/vast/train_a3.sh
bash scripts/vast/evaluate_all.sh
```

Setup/data tasks never invoke training. Explicit train_aN commands run fresh
matching G1?G4 before full training; --verify-only stops after gates. Wrappers
reuse research.py and project modules; no training/evaluation implementation is
copied. Configs remain configs/baseline/A0.yaml, configs/ablation/A1.yaml,
configs/ablation/A2.yaml, configs/proposed/A3.yaml.
Defaults are batch 16/250 epochs/seed 444/1280x384/original augmentation/loss/
optimizer/scheduler, float32, single GPU. A4 stays locked. Unique UTC timestamp
run names prevent overwriting previous runs. To use the run_001 examples below,
explicitly add `--run-name run_001`. Always use the actual printed run directory.
A failed gate exits nonzero before creating the training run directory.

Detach with Ctrl+B then D; reconnect with `tmux attach -t depth-monoia`.
Monitor the actual run (substitute its printed timestamp or explicit run name):

```bash
RUN_DIR=/workspace/depth-monoia/experiments/A3_intrinsic_fusion/run_001
tail -f "$RUN_DIR/logs/events.jsonl"
# After ending tail:
nvidia-smi
df -h /workspace/depth-monoia
```

First Ctrl+C inside training (or SIGTERM to its Python PID) requests an epoch-
boundary stop after validation/checkpoint. Wait for exit. A second signal/hard
kill replays the incomplete epoch on resume; no exact mid-epoch resume is claimed.

## 8. Checkpoints, evaluation and machine-readable outputs

The one-task evaluation command is `bash scripts/vast/evaluate_all.sh`. It uses
each actual run's saved config. It refuses existing final output directories;
do not rerun it over completed outputs expecting overwrite. Deliberate re-evaluation
uses the lower-level CLI with a new destination. The advanced commands below use
explicit run_001 names; substitute actual timestamped names when applicable.
Initialize the advanced shell variables after reconnecting:

```bash
source scripts/vast/common.sh
vast_activate
```

Run directories under `$OUTPUT_ROOT`:

| Experiment | Directory |
| --- | --- |
| A0 | `A0_baseline/run_001` |
| A1 | `A1_simple_fusion/run_001` |
| A2 | `A2_gated_fusion/run_001` |
| A3 | `A3_intrinsic_fusion/run_001` |

Each has `checkpoints/best.pth`, `checkpoints/last.pth`,
`checkpoint_hashes.json`, `metadata.json`, `config.yaml`, `gates.json`,
`training_state.json`, `logs/training.log`, `logs/events.jsonl`.
Best is highest completed Car AP3D Moderate R40; last is latest committed epoch.
Full-state checkpoints contain optimizer/scheduler/warmup/RNG/next epoch/global
step. Files use atomic replace under a shared artifact lock. Active-run best/last
update as training progresses; previous run directories and named backup snapshots
are not overwritten. A multi-file hard-crash mismatch requires a prior valid snapshot.

After real training, evaluate the baseline and proposed model individually:

```bash
bash scripts/gpu/evaluate_model.sh --experiment A0 --dataset-root "$KITTI_ROOT" --manifest "$MANIFEST" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --checkpoint "$OUTPUT_ROOT/A0_baseline/run_001/checkpoints/best.pth" \
  --run-dir "$OUTPUT_ROOT/A0_baseline/run_001/evaluation/final_best"
bash scripts/gpu/evaluate_model.sh --experiment A3 --dataset-root "$KITTI_ROOT" --manifest "$MANIFEST" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --checkpoint "$OUTPUT_ROOT/A3_intrinsic_fusion/run_001/checkpoints/best.pth" \
  --run-dir "$OUTPUT_ROOT/A3_intrinsic_fusion/run_001/evaluation/final_best"
```

For A1/A2 substitute the corresponding experiment/directory above. Alternatively,
use the existing batch command **instead of** individual commands on those same
output directories (existing outputs are intentionally refused):

```bash
bash scripts/gpu/evaluate_all.sh --search-root "$OUTPUT_ROOT" \
  --dataset-root "$KITTI_ROOT" --manifest "$MANIFEST" --checkpoint-dir "$CHECKPOINT_DIR"
python scripts/aggregate_results.py --search-root "$OUTPUT_ROOT" --output "$TASK_ROOT/model_comparison.csv"
```

Every `evaluation/final_best/` stores actual `predictions/`, `metrics.json`,
`metrics.csv`, `kitti_eval.txt`, `distance_metrics.json`, `confidence_metrics.json`,
`resource_metrics.json`, `evaluation_metadata.json`. The completion marker is
published last. Epoch folders contain detection AP/predictions only.

Metrics: Car AP3D/APBEV Easy/Moderate/Hard, IoU 0.7, R40; matched-object camera-Z
and Euclidean-centre distance MAE/RMSE with recall; detection confidence ECE with
15 bins when scores are in [0,1]. MonoIA scores can exceed 1, in which case ECE
is explicitly inapplicable (no clipping or invented calibration).

Resource evaluation counts parameters (including frozen parameters) in millions,
records real hardware/input/batch, CUDA-synchronized inference latency with 5 warmups
and 20 measured iterations, and peak memory. Batch latency and amortized per-image
latency are distinct from batch-1 latency. **Full-model GFLOPs are still unavailable**:
`GFLOPs` remains null; `partial_GFLOPs_per_batch` is profiler coverage only, excluding
unsupported custom operations. This is a reporting limitation, not a reason to
invent a number or block training. Validated complete operator counting remains
necessary before a thesis full-GFLOPs comparison. No metric values are fabricated.

## 9. Off-host backups and Windows artifact download

These examples use run_001 only if explicitly selected. Substitute the actual
run directory printed by train_aN.sh, and `source scripts/vast/common.sh` plus
`vast_activate` in a fresh Linux shell before invoking lower-level backup tools.

Back up after the first committed epoch, at subsequent useful epoch boundaries
(prefer every committed epoch when transfer speed permits), and after final
evaluation. A running backup uses the same artifact lock as writers; archive
creation can pause their publication. It includes best/last, hashes, metadata,
config, gates, logs and completed evaluations. Local snapshots alone do not
protect against host loss. Untransferred epochs can be lost.

Linux (A3 example, substitute A0/A1/A2 directory for other runs):

```bash
SNAPSHOT=A3_run001_$(date -u +%Y%m%dT%H%M%SZ).tar.gz
bash scripts/gpu/backup_artifacts.sh \
  --run-dir "$OUTPUT_ROOT/A3_intrinsic_fusion/run_001" \
  --archive "$TASK_ROOT/backups/$SNAPSHOT"
printf '%s\n' "$SNAPSHOT"
sha256sum "$TASK_ROOT/backups/$SNAPSHOT"
# Once per deployment / whenever evidence changes; checkpoint archives exclude these:
SUPPORT=deployment_support_$(date -u +%Y%m%dT%H%M%SZ).tar.gz
tar -czf "$TASK_ROOT/backups/$SUPPORT" -C "$TASK_ROOT" evidence setup_logs -C "$KITTI_ROOT" ImageSets
sha256sum "$TASK_ROOT/backups/$SUPPORT" > "$TASK_ROOT/backups/$SUPPORT.sha256"
printf '%s\n' "$SUPPORT"
```

Windows PowerShell (retain the source Git commit/bundle, original data and inputs
locally too). Use names printed by the Linux commands:

```powershell
$BackupRoot = 'D:\research-backups\depth-monoia'
New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
$Snapshot = Read-Host 'Exact run snapshot filename printed on Linux'
scp -i $TaskKey -P $VastPort "root@${VastHost}:/workspace/depth-monoia/backups/$Snapshot" $BackupRoot
scp -i $TaskKey -P $VastPort "root@${VastHost}:/workspace/depth-monoia/backups/$Snapshot.sha256" $BackupRoot
$LocalArchive = Join-Path $BackupRoot $Snapshot
$ExpectedHash = ((Get-Content "$LocalArchive.sha256" -Raw).Trim() -split '\s+')[0]
if ((Get-FileHash -Algorithm SHA256 $LocalArchive).Hash -ine $ExpectedHash) { throw 'Archive SHA-256 mismatch' }
# Run in the existing local repository with its Python 3.9 environment active:
python scripts/research.py restore --archive $LocalArchive --destination (Join-Path $BackupRoot ($Snapshot + '_verified'))
if ($LASTEXITCODE -ne 0) { throw 'Snapshot restore verification failed' }
$Support = Read-Host 'Exact deployment_support archive filename printed on Linux'
scp -i $TaskKey -P $VastPort "root@${VastHost}:/workspace/depth-monoia/backups/$Support" $BackupRoot
scp -i $TaskKey -P $VastPort "root@${VastHost}:/workspace/depth-monoia/backups/$Support.sha256" $BackupRoot
$LocalSupport = Join-Path $BackupRoot $Support
$SupportHash = ((Get-Content "$LocalSupport.sha256" -Raw).Trim() -split '\s+')[0]
if ((Get-FileHash -Algorithm SHA256 $LocalSupport).Hash -ine $SupportHash) { throw 'Support archive SHA-256 mismatch' }
```

Restore verifies both whole-archive and each member SHA-256, rejects unsafe paths/
links and refuses existing destinations; it works without a GPU. Do not run the
run-restore tool against the separate support tar (it has no SNAPSHOT inventory).
Keep the original evidence/data locally; the support tar is an extra copy.
With server-side acquisition, retain the source URLs/provenance and a secure
off-server copy of your private source specification separately; renew expired
official links after host loss. The support tar preserves actual original
ImageSets split bytes, public manifest, reference evidence and environment logs.
Run snapshots exclude full KITTI/pretrained files; recovery re-downloads them
from authorized original sources and checks the same recorded bytes/hashes.
Keep at least the latest and preceding verified off-host run snapshots if space
permits. No credentials, data, checkpoint or generated archives belong in Git.
Any deletion/pruning is a separate reviewed action; this guide issues none.

Before destroying an instance: finish the requested epoch-boundary stop, make a
fresh snapshot, download and verify/restore it, download support evidence and all
final evaluations, then use Vast's console. A stop still is not an off-host backup;
storage billing may continue. No script here destroys or purchases instances.

## 10. Interrupted run / replacement instance recovery

Same surviving instance: activate/source the environment, retain source/data/
operator bytes and use the original gates. Resume explicitly:

The beginner task wrapper can run fresh gates and resume the saved configuration
in one explicit command (use the actual run path):

```bash
bash scripts/vast/train_a3.sh --resume /workspace/depth-monoia/experiments/A3_intrinsic_fusion/run_001
```

The lower-level equivalent with previously matching gate evidence is:

```bash
RUN_DIR="$OUTPUT_ROOT/A3_intrinsic_fusion/run_001"
bash scripts/gpu/resume_training.sh --config "$RUN_DIR/config.yaml" \
  --dataset-root "$KITTI_ROOT" --manifest "$MANIFEST" --checkpoint-dir "$CHECKPOINT_DIR" \
  --gates "$RUN_DIR/gates.json" --run-dir "$RUN_DIR" \
  --resume "$RUN_DIR/checkpoints/last.pth"
```

Replacement instance, only after its rental is approved:

1. Repeat steps 2–5 with the **recorded source commit**, same environment, KITTI
   bytes/splits/manifest, pretrained weights, and original reference inputs.
   Rebuild CUDA ops for the target GPU. A previous host's volume is not assumed
   available. Copy the verified archive and sidecar from Windows:

```powershell
# Reassign VastHost/VastPort for the replacement first; create directories via step 4.
scp -i $TaskKey -P $VastPort $LocalArchive "$LocalArchive.sha256" "root@${VastHost}:/workspace/depth-monoia/backups/"
```

2. On Linux, use the exact transferred snapshot filename (not a wildcard), and
   restore into a **new/nonexistent** run directory:

```bash
read -r -p 'Exact transferred run snapshot filename: ' SNAPSHOT
RUN_DIR="$OUTPUT_ROOT/A3_intrinsic_fusion/run_001"
bash scripts/gpu/restore_artifacts.sh --archive "$TASK_ROOT/backups/$SNAPSHOT" --destination "$RUN_DIR"
cat "$RUN_DIR/RESTORE_PROVENANCE.json"
cat "$RUN_DIR/training_state.json"
```

3. Rerun the existing gates for the restored saved config, then resume:

```bash
python scripts/research.py verify --config "$RUN_DIR/config.yaml" --gpu \
  --dataset-root "$KITTI_ROOT" --manifest "$MANIFEST" --checkpoint-dir "$CHECKPOINT_DIR" \
  --reference-predictions "$TASK_ROOT/evidence/val_predictions" \
  --reference-metrics "$TASK_ROOT/evidence/val_reference.json" \
  --output "$TASK_ROOT/evidence/A3_recovery_gates.json"
bash scripts/gpu/resume_training.sh --config "$RUN_DIR/config.yaml" \
  --dataset-root "$KITTI_ROOT" --manifest "$MANIFEST" --checkpoint-dir "$CHECKPOINT_DIR" \
  --gates "$TASK_ROOT/evidence/A3_recovery_gates.json" --run-dir "$RUN_DIR" \
  --resume "$RUN_DIR/checkpoints/last.pth"
```

Identity is path-independent for equivalent code/data/config but retains hashes.
New gate evidence records the rebuilt operator and current initialization cache.
Cross-hardware bitwise equality is not promised. Do not run two writers/resumes
on one experiment. A leftover lock filename after process death is harmless;
do not bypass a live OS-held lock. Hash/publication mismatch means restore the
preceding verified snapshot, not edit hashes. Resume restores model, optimizer,
scheduler/warmup, RNG, epoch/global step; incomplete epochs replay. Back up again
after a successful recovery.

## Audit outcome and approval boundary

Implemented: seven task entry points reusing the existing runner, authenticated
official-source acquisition with safe cached reruns, original-split verification,
manifest generation, independent A0–A3 outputs and explicit-batch/OOM reporting.
Existing full-state resume, real evaluation/export and verified off-server
backup/restore remain in use. No model architecture was replaced.

Remaining external requirements: official download authorization/URLs; trusted
original split provenance/hashes; independent G4 reference evidence; reviewed and
committed wrapper changes available with history; and approval of a paid RTX 4090
offer. Archive expected hashes are optional when unavailable; authenticated
official acquisition plus CRC/recorded hashes is explicitly distinguished from
independent source authentication. Linux build, real dataset download, G1–G4 and
batch-16 fit on 24 GB remain pending. Full-GFLOPs coverage is a reporting limitation.
No GPU verification, training, performance improvement or paid operation is claimed.
