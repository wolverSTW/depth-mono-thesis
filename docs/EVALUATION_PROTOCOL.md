# Evaluation protocol

Validation uses the pinned `kitti_3dop_chen` 3769-frame list; see
[split provenance](KITTI_3DOP_PROTOCOL.md). Independent G4 reference inputs must
match its SHA-256. Exact original MonoIA split/checkpoint overlap is unverified.

On the approved RTX 4090 environment, `bash scripts/vast/evaluate_all.sh` activates
the pinned environment and delegates to research.py using each actual trained
run's saved configuration/checkpoint. It then exports model_comparison.csv.
No training is started. Existing final output directories are refused to preserve
evidence; choose a new output via the lower-level CLI for deliberate reruns.
GPU synchronization, 5 warmups/20 repeated-real-batch latency measurements and
parameter counts retain the methodology below. Full GFLOPs remains null when
custom-op coverage is unvalidated; any partial profiler number is labeled partial.

Primary metrics use pinned upstream KITTI-compatible evaluation: Car, IoU 0.7
for 3D and BEV, AP_R40 Easy/Moderate/Hard, original difficulty/DontCare behavior,
topk 50, original 0.2 filter and two-decimal prediction serialization. Predictions
and labels follow the same verified split order. G4 requires independently
executed real KITTI reference outputs, never invented expected values.

Auxiliary distance metrics use score-descending one-to-one 3D IoU>0.7 matching on
Moderate Car, with upstream valid/ignored filters. This differs from AP assignment.
Report matched count/eligible-GT recall together with camera-Z MAE/RMSE in metres.
Euclidean range uses geometric box centres after converting KITTI bottom-centre
locations, and is reported separately.

Retain actual prediction confidence in raw KITTI files. ECE uses 15 equal-width
bins only if final scores are valid probabilities in [0,1]. MonoIA's depth-score
factor can violate this; mark ECE inapplicable rather than clipping. Detection
confidence is not distance confidence. No calibrated distance uncertainty model
has been implemented, so uncertainty metrics are unavailable.

Measure total/trainable parameters, hardware, float32, actual batch/image size,
warmup/count, batch wall latency, amortized image latency, peak GPU memory and
training duration. Full-model GFLOPs is Pending until custom-op coverage is
validated; partial profiler counts must retain their label.

Each executed evaluation writes predictions, JSON/CSV metrics, raw evaluator text,
distance/confidence/resource details and checkpoint/data/code metadata. Only
completed, mutually comparable real evaluations enter model_comparison.csv.
Reports distinguish published, official-checkpoint and independent-training results.

Protocol source: [KITTI benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
