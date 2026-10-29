# Vast.ai: full KITTI deployment runbook

Prepared 2026-10-29. Vast.ai is the exclusive training provider. These commands
have been inspected locally, **not executed on a GPU**. Do not rent an instance
or execute paid GPU work until the owner approves it. Continue the existing
`develop` project; do not initialize another project or change the pinned sources.

## 1. Before paying

Have these inputs ready off the GPU machine:

- Official KITTI object-detection left RGB, calibration and training-label ZIPs.
  Obtain them through the [KITTI account/download page](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
  LiDAR, stereo/right RGB, raw KITTI and temporal images are unnecessary.
- Original designated `train.txt` and `val.txt`, their independently checked
  provenance and SHA-256 hashes. Intended counts are 3712/3769, disjoint and
  covering all 7481 labeled images. The pinned MonoIA repository does not supply
  these files. Do not generate a random split or assume arbitrary online lists
  reproduce the original protocol. Keep their original bytes/line endings.
- `local.dataset.json`, populated from `docs/reproducibility/dataset.example.json`:
  replace REQUIRED fields with the actual trusted split source and hashes.
- Real validation predictions in KITTI text format and independent KITTI-compatible
  R40 reference measurements for those exact predictions and split. Name them
  `val_predictions/` and `val_reference.json`. The latter must contain
  `reference_source`, `split_sha256`, and `metrics` with all six keys
  `Car_3d_easy_R40`, `Car_3d_moderate_R40`, `Car_3d_hard_R40`,
  `Car_bev_easy_R40`, `Car_bev_moderate_R40`, `Car_bev_hard_R40`.
  Use Car IoU 0.7. Published scores or results from this same evaluator alone
  are not independent reference evidence. This missing input currently blocks G4.
- This preparation commit available on GitHub **after separate push approval**,
  or a Git bundle containing it. `89c6b99` is the preceding bootstrap commit;
  it does not contain the configured-batch capacity gate in this runbook.
- A Windows SSH client/key, sufficient off-host backup space (plan 100 GB free,
  more if retaining many full snapshots), and a reviewed rental budget.

No GPU is needed to acquire these inputs. A full training run requires all four
existing gates; preparing an account/template is not GPU verification.

## 2. Hardware, image and storage selection

Planning recommendation: **one full NVIDIA A100 80 GB**, Linux x86-64/amd64,
8 allocated vCPUs minimum (16 preferred), 64 GB allocated host RAM, and **300 GB
SSD/NVMe container disk**. Avoid MIG partitions and fractional-memory offers.
Select an on-demand offer with direct SSH, suitable transfer bandwidth and a
contract duration you can monitor. There is no measured training-time estimate.
Check allocated resources, not the host's advertised total or burst resources.
[A100 specifications](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/PB-10577-001_v02.pdf?ncid=no-ncid),
[Vast execution environment](https://docs.vast.ai/guides/instances/docker-environment).

**VRAM estimates, not measurements:** budget roughly 32–64 GiB for planning
A0–A3 FP32/batch-16 trials, with low confidence; this is neither a measured
minimum nor an upper bound. The four retained DA-V2 prior tensors alone are
about 0.42 GiB at the default resolution/batch, excluding detector activations,
DA-V2 workspaces, optimizer, gradients and allocator overhead. A1–A3 run frozen
DA-V2 with microbatch 1 but train the detector with batch 16. An 80 GB offer
provides headroom; even its fit remains unverified. **24 GB is not established
as sufficient.** Consider 48 GB only after an actual configured-batch gate on
that hardware. Do not silently reduce A0's batch or enable AMP to fit.

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

## 4. Bootstrap source and CUDA environment (Linux SSH terminal)

Stop at any failed command. This runbook uses Bash and the root SSH account.
For an explicitly non-root template, prefix apt commands with sudo.

```bash
set -euo pipefail
apt-get update
apt-get install -y build-essential git ninja-build unzip rsync tmux libgl1 libglib2.0-0 ca-certificates
export TASK_ROOT=/workspace/depth-monoia
mkdir -p "$TASK_ROOT"/{incoming,evidence,pretrained,experiments,backups,setup_logs}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvcc --version
free -h
df -h "$TASK_ROOT" /dev/shm
source /opt/conda/etc/profile.d/conda.sh
git clone --branch develop --recurse-submodules https://github.com/wolverSTW/depth-mono-thesis.git "$TASK_ROOT/source"
cd "$TASK_ROOT/source"
read -r -p 'Approved preparation commit SHA (from local git log): ' DEPLOY_COMMIT
git checkout --detach "$DEPLOY_COMMIT"
git submodule update --init --recursive
test -z "$(git status --porcelain)"
git rev-parse HEAD
git submodule status
```

The preparation SHA must be reachable on the remote before the clone above.
If publication is not approved, make a bundle on Windows after the local commit:

```powershell
git -C .\Depth-MonoIA-Thesis bundle create ..\depth-monoia-develop.bundle develop
# Transfer the bundle to the instance's incoming directory using scp, as below.
```

Then substitute these Linux commands for the GitHub clone (same new destination):

```bash
git clone --branch develop "$TASK_ROOT/incoming/depth-monoia-develop.bundle" "$TASK_ROOT/source"
cd "$TASK_ROOT/source"
git remote set-url origin https://github.com/wolverSTW/depth-mono-thesis.git
# Continue with DEPLOY_COMMIT checkout and submodule update above.
```

Bundles preserve project history; public pinned submodules are fetched separately.
Do not copy a source-only ZIP or edit source on the GPU. Keep the bundle off-host.

Create a session environment file (ignored by Git, contains no credentials):

```bash
cat > local.vast.env <<'EOF'
export TASK_ROOT=/workspace/depth-monoia
export KITTI_ROOT=$TASK_ROOT/datasets/KITTIDataset
export OUTPUT_ROOT=$TASK_ROOT/experiments
export CHECKPOINT_DIR=$TASK_ROOT/pretrained
export MANIFEST=$TASK_ROOT/evidence/local.dataset.json
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_HOME=$TASK_ROOT/torch_cache
export MAX_JOBS=4
export GPU_DEVICE=0
EOF
source local.vast.env
bash scripts/gpu/setup_environment.sh --env-name depth-monoia-gpu 2>&1 | tee "$TASK_ROOT/setup_logs/setup.log"
conda activate depth-monoia-gpu
python -m pip freeze > "$TASK_ROOT/setup_logs/pip-freeze.txt"
git rev-parse HEAD > "$TASK_ROOT/setup_logs/source_commit.txt"
git submodule status > "$TASK_ROOT/setup_logs/submodules.txt"
nvidia-smi > "$TASK_ROOT/setup_logs/nvidia-smi.txt"
nvcc --version > "$TASK_ROOT/setup_logs/nvcc.txt"
```

The script creates Python 3.9, installs torch 2.4.1/torchvision 0.19.1 cu121 and
`requirements/linux-cu121.txt` (NumPy 1.26.4, Numba 0.60.0), then compiles original
deformable attention for the actual GPU into `build/ops`. It checks CUDA/toolkit
versions; the image's base Python is not assumed to be 3.9. The equivalent torch
install is `python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121`.
[Official version pairing](https://pytorch.org/get-started/previous-versions/).

CUDA_HOME must point to the complete toolkit, including NVVM/NVRTC used by the
Numba KITTI evaluator, not only CUDA runtime wheels. The devel image supplies it.
[Numba 0.60 CUDA requirements](https://numba.readthedocs.io/en/0.60.0/cuda/overview.html).
Do not install a second arbitrary CUDA toolkit. If Conda requests channel terms,
review them before proceeding; do not silently accept terms on someone's behalf.

After every reconnect or new tmux shell:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate depth-monoia-gpu
cd /workspace/depth-monoia/source
source local.vast.env
```

## 5. Transfer and prepare KITTI, evidence and pretrained weights

In a second Windows PowerShell terminal (reuse host/port/key variables), adapt
only the local input locations to where you actually saved the authentic files:

```powershell
$DataInput = 'D:\KITTI'
$EvidenceInput = 'D:\depth-monoia-evidence'
scp -i $TaskKey -P $VastPort "$DataInput\data_object_image_2.zip" "$DataInput\data_object_calib.zip" "$DataInput\data_object_label_2.zip" "root@${VastHost}:/workspace/depth-monoia/incoming/"
scp -i $TaskKey -P $VastPort "$EvidenceInput\train.txt" "$EvidenceInput\val.txt" "$EvidenceInput\local.dataset.json" "$EvidenceInput\val_reference.json" "root@${VastHost}:/workspace/depth-monoia/evidence/"
scp -r -i $TaskKey -P $VastPort "$EvidenceInput\val_predictions" "root@${VastHost}:/workspace/depth-monoia/evidence/"
```

Back in Linux, from source with the environment activated:

```bash
mkdir -p "$KITTI_ROOT/ImageSets"
unzip -n "$TASK_ROOT/incoming/data_object_image_2.zip" -d "$KITTI_ROOT"
unzip -n "$TASK_ROOT/incoming/data_object_calib.zip" -d "$KITTI_ROOT"
unzip -n "$TASK_ROOT/incoming/data_object_label_2.zip" -d "$KITTI_ROOT"
cp -n "$TASK_ROOT/evidence/train.txt" "$KITTI_ROOT/ImageSets/train.txt"
cp -n "$TASK_ROOT/evidence/val.txt" "$KITTI_ROOT/ImageSets/val.txt"
sha256sum "$KITTI_ROOT/ImageSets/"{train,val}.txt
bash scripts/gpu/prepare_dataset.sh --dataset-root "$KITTI_ROOT" --manifest "$MANIFEST"
bash scripts/gpu/download_pretrained.sh --name dav2_small --checkpoint-dir "$CHECKPOINT_DIR"
# Optional, separate official-checkpoint evaluation; not A0 training initialization:
bash scripts/gpu/download_pretrained.sh --name monoia_val --checkpoint-dir "$CHECKPOINT_DIR"
```

Expected layout is `$KITTI_ROOT/training/{image_2,calib,label_2}/000000.*` and
`$KITTI_ROOT/ImageSets/{train,val}.txt`. Each downloader verifies the immutable
official SHA-256 recorded in `docs/reproducibility/sources.json`. The MonoIA builder
also downloads its original pretrained ResNet backbone to `$TORCH_HOME/checkpoints`
on first construction; retain network access and this cache. Its actual bytes
are hashed in the gate/run metadata. A0 is independently trained with that
original backbone initialization, **not** fine-tuned from MonoIA validation weights.

## 6. Existing mandatory gates, including actual batch capacity

Execute serially. G2 now uses a complete batch from the real augmented training
loader (default 16), performs forward/backward/optimizer steps, and records input
shape plus peak allocated/reserved CUDA memory in `details.capacity`. G3 verifies
full-state restoration and next-step agreement; G4 checks independent reference
evidence. Gates mutate only temporary models, not experiment checkpoints.

```bash
for EXP in A0 A3 A1 A2; do
  python scripts/research.py verify --experiment "$EXP" --gpu \
    --dataset-root "$KITTI_ROOT" --manifest "$MANIFEST" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --reference-predictions "$TASK_ROOT/evidence/val_predictions" \
    --reference-metrics "$TASK_ROOT/evidence/val_reference.json" \
    --output "$TASK_ROOT/evidence/${EXP}_gates.json"
done
```

Stop on any failure. No PASS is claimed here. A gate's representative batch is
not proof of peak memory across every augmented batch or the full 250 epochs.
Leave headroom and monitor training. On OOM, select more memory before modifying
the reproduction protocol. No silent AMP, batch reduction or multi-GPU change.
If shared-memory/DataLoader failure actually occurs, an explicit `--workers 0`
is supported; use the same setting for gates, training and evaluation and record
it. Do not add such overrides speculatively. G4 also exercises the CUDA evaluator.

## 7. Full training: four independent configurations and run directories

Only after paid training approval and PASS gates, run `tmux new -s depth-monoia`,
then activate/source the environment as in step 4. Run **one experiment at a time**.
Default settings remain batch 16, 250 epochs, seed 444, original 1280x384 detector
input, original augmentation/loss/optimizer/scheduler and float32. A0's original
MonoIA model and epoch implementation are preserved; fusion is external to it.

```bash
# A0: Original MonoIA -- configs/baseline/A0.yaml
bash scripts/gpu/train_baseline.sh --dataset-root "$KITTI_ROOT" --manifest "$MANIFEST" \
  --checkpoint-dir "$CHECKPOINT_DIR" --gates "$TASK_ROOT/evidence/A0_gates.json" \
  --output-root "$OUTPUT_ROOT" --run-name run_001 --gpu-device 0

# A3: DA-V2 Intrinsic-Aware Gated Fusion -- configs/proposed/A3.yaml
bash scripts/gpu/train_proposed.sh --dataset-root "$KITTI_ROOT" --manifest "$MANIFEST" \
  --checkpoint-dir "$CHECKPOINT_DIR" --gates "$TASK_ROOT/evidence/A3_gates.json" \
  --output-root "$OUTPUT_ROOT" --run-name run_001 --gpu-device 0

# A1: DA-V2 Simple Fusion -- configs/ablation/A1.yaml
bash scripts/gpu/train_ablation.sh --experiment A1 --dataset-root "$KITTI_ROOT" --manifest "$MANIFEST" \
  --checkpoint-dir "$CHECKPOINT_DIR" --gates "$TASK_ROOT/evidence/A1_gates.json" \
  --output-root "$OUTPUT_ROOT" --run-name run_001 --gpu-device 0

# A2: DA-V2 Gated Fusion -- configs/ablation/A2.yaml
bash scripts/gpu/train_ablation.sh --experiment A2 --dataset-root "$KITTI_ROOT" --manifest "$MANIFEST" \
  --checkpoint-dir "$CHECKPOINT_DIR" --gates "$TASK_ROOT/evidence/A2_gates.json" \
  --output-root "$OUTPUT_ROOT" --run-name run_001 --gpu-device 0
```

These are the full designated training split, not train+val and not smoke tests.
A4 remains locked. Choose a new run name for a new experiment; existing run
directories are refused. Detach tmux with Ctrl+B then D; `tmux attach -t depth-monoia`
reconnects. From a separate SSH shell:

```bash
tail -f "$OUTPUT_ROOT/A3_intrinsic_fusion/run_001/logs/events.jsonl"
# Other useful views, after ending tail with Ctrl+C:
tail -n 40 "$OUTPUT_ROOT/A3_intrinsic_fusion/run_001/logs/training.log"
nvidia-smi
df -h "$TASK_ROOT"
```

First Ctrl+C **inside the training shell**, or SIGTERM to its Python PID, requests
an epoch-boundary stop after validation and checkpoint commit. Wait for process
exit. Second interrupt/hard termination abandons the current epoch; resume replays
it. Provider deadline/host loss may prevent cleanup. No exact mid-epoch resume.

## 8. Checkpoints, evaluation and machine-readable outputs

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
tar -czf "$TASK_ROOT/backups/$SUPPORT" -C "$TASK_ROOT" evidence setup_logs
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

```bash
RUN_DIR="$OUTPUT_ROOT/A3_intrinsic_fusion/run_001"
bash scripts/gpu/resume_training.sh --config "$RUN_DIR/config.yaml" \
  --dataset-root "$KITTI_ROOT" --manifest "$MANIFEST" --checkpoint-dir "$CHECKPOINT_DIR" \
  --gates "$TASK_ROOT/evidence/A3_gates.json" --run-dir "$RUN_DIR" \
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

Implemented: independent A0–A3 configs/runs, safe full-state save/resume, actual
evaluation/export, locked checksum backups/restores. This preparation additionally
fixes the one-sample capacity gate and checks the complete CUDA 12.1 toolkit early.
Only configured-batch Linux CUDA execution can establish fit/compatibility.

Genuine pre-rental blockers: trusted KITTI/split/reference inputs ready to transfer;
this exact preparation commit available remotely or via bundle; owner approval
of the selected paid offer/budget. GPU-only build/G2–G4 and actual VRAM measurement
are pending acceptance work on that authorized machine. Full-GFLOPs coverage is
a remaining reporting limitation. No GPU verification, full training, measured
performance improvement or paid operation is claimed.
