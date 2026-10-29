# Artifact backup and Windows download

Vast.ai is the exclusive GPU provider. [VAST_AI_RUNBOOK.md](VAST_AI_RUNBOOK.md)
steps 9–10 provide the actual `/workspace/depth-monoia` snapshot, mapped-port SSH,
Windows SHA-256/restore check and replacement-instance commands. The generic
paths below illustrate the same existing tools; they are not assumed mounts.

Do not treat a stopped GPU instance as an off-machine backup. Archives contain
checkpoints, resolved configs, metadata, completed evaluation outputs and logs;
KITTI data and pretrained downloads are excluded. Source commits must also be
available through the approved Git remote or a separately transferred Git bundle.

On the GPU machine, use a destination outside the run:

```bash
bash scripts/gpu/backup_artifacts.sh \
  --run-dir "$OUTPUT_ROOT/A3_intrinsic_fusion/run_001" \
  --archive /persistent/backups/A3_run001_snapshot01.tar.gz
```

The backup takes the shared artifact lock, verifies the committed checkpoint hash
manifest, excludes unfinished evaluations, builds a per-file SHA-256 inventory and
publishes an archive plus `.sha256` sidecar. Writers may wait while a large archive
is created. Existing archive names are refused; use a new snapshot name. Hashes
detect corruption, not adversarial authenticity; retain them through a trusted
transfer/account. Use a local POSIX run filesystem with working advisory locks.

From Windows PowerShell, substitute your actual host/port/key paths. Do not put
private keys or tokens in this repository:

```powershell
scp -P 22 user@host:/persistent/backups/A3_run001_snapshot01.tar.gz D:\research-backups\
scp -P 22 user@host:/persistent/backups/A3_run001_snapshot01.tar.gz.sha256 D:\research-backups\
Get-FileHash -Algorithm SHA256 D:\research-backups\A3_run001_snapshot01.tar.gz
python scripts/research.py restore `
  --archive D:\research-backups\A3_run001_snapshot01.tar.gz `
  --destination D:\research-backups\A3_run001_verified
```

Create D:\research-backups beforehand and compare the checksum with the sidecar.
The restore command verifies both archive and each file, rejects traversal/links
and refuses existing destinations. This restore check is useful on CPU and does
not run a model. Use the actual host/port shown in Vast's Connect panel; do not
assume port 22 is externally mapped to the container's SSH port.

Keep at least one verified off-machine copy before terminating a resource. Inspect
restored metadata/checkpoint_hashes and retain raw evidence; do not commit large
archives to Git. No remote transfer or backup of model results has occurred here.
