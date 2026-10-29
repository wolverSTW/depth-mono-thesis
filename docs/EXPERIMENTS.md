# Experiment protocol

Approved membership: `kitti_3dop_chen`, common to A0–A3 through configs/common.yaml.
See [published split evidence](KITTI_3DOP_PROTOCOL.md). Describe comparisons as
using the published research protocol; exact original MonoIA split is unverified.

| ID | Configuration | Added fusion | Execution |
|---|---|---|---|
| A0 | configs/baseline/A0.yaml | None | Prepared |
| A1 | configs/ablation/A1.yaml | Simple residual | Prepared |
| A2 | configs/ablation/A2.yaml | RGB/depth gate | Prepared |
| A3 | configs/proposed/A3.yaml | Intrinsic-aware gate | Prepared |
| A4 | configs/ablation/A4.yaml | Optional prior reliability | Locked |

All retain original MonoIA intrinsics. Inherit train/val, augmentations, batch 16,
250 epochs, custom AdamW lr 0.0002, decay at 85/125/165/205 and seed 444. Explicit
batch/seed overrides must match all compared runs and appear in the resolved config
and gate identity; they define a documented protocol revision, not exact baseline
reproduction. Worker count is configurable and recorded. No mixed precision.

Use best completed Car AP3D Moderate R40 for selection; record best and last
epoch/global step/seed/hardware/duration. Official validation-weight A0 is separate
from independent A0 training. Do not evaluate leaderboard-trained weights on val.
See CHECKPOINT_MANAGEMENT.md and EVALUATION_PROTOCOL.md.

Run A0/A3 first if constrained, then A1/A2. A4 cannot run until A0–A3 real-data
functionality is demonstrated; this bootstrap does not unlock it. Actual ablation
evidence belongs in ../reports/ABLATION_REPORT.md. All results are Pending.
