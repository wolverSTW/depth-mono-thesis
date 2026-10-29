# Evaluation results

Updated: 2026-10-29. **Not yet evaluated on KITTI. No benchmark numbers available.**

| Measure | Actual status |
|---|---|
| AP3D Easy / Moderate / Hard | Not yet evaluated |
| APBEV Easy / Moderate / Hard | Not yet evaluated |
| Camera Z-depth MAE/RMSE | Not yet evaluated |
| Euclidean box-centre range MAE/RMSE | Not yet evaluated |
| Detection ECE / confidence distribution | Not yet evaluated |
| Distance confidence / uncertainty | No calibrated model implemented; do not report |
| Total / trainable parameters | Full detector not measured |
| Full GFLOPs | Pending validated custom CUDA coverage |
| Latency / peak GPU memory | Not measured |
| Training duration | No training executed |

## Evaluation protocol

Main benchmark: pinned upstream KITTI-compatible evaluator, Car BEV/3D IoU 0.7,
AP_R40, original Easy/Moderate/Hard and DontCare conventions. Prediction and GT
lists share exact split order. Original topk 50, pre-depth-score threshold 0.2
and two-decimal prediction formatting are retained for baseline comparison.
G4 compares real reference outputs against independently obtained KITTI metrics.

Auxiliary distance/ECE: score-descending greedy one-to-one 3D IoU>0.7 matching for
Moderate Car, separate from AP assignment. Upstream valid/ignored filters apply;
matches to ignored GT are excluded, duplicates count as false positives. Report
matched count/eligible-GT recall alongside errors. Z is camera-axis depth;
Euclidean range uses geometric centres converted from KITTI bottom-centre locations.

ECE uses 15 equal-width bins over retained final detection scores. Original scores
combine sigmoid classification with exp(-predicted depth deviation), so can exceed
one. ECE is inapplicable then; scores are not clipped. Detection confidence is not
distance confidence. No calibrated distance uncertainty is reported.

Resources include hardware, float32, actual batch/image shape, warmups, batch
latency and amortized per-image latency (not single-image latency). Frozen weights
count toward total parameters. Torch-profiler FLOPs are explicitly partial:
unsupported/custom operations prevent a validated full-model GFLOPs claim.

Published results, official-weight evaluation and independently trained A0 are
separate evidence types. Add reviewed real logs/paths/hashes, including negative
findings, after execution.
