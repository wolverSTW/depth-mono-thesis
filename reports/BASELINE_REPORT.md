# Baseline reproduction

Updated: 2026-10-29. No baseline training/evaluation has executed here.

## Original configuration

MonoIA commit `69d6ee30ca5e44848727387dd44af6d56d62c24f`, MIT.
`configs/baseline/A0.yaml` inherits upstream `config/monoia.yaml`.
The original builder, intrinsic embeddings, architecture, losses, custom optimizer,
scheduler and epoch optimization code remain unchanged.

| Setting | Prepared value |
|---|---|
| Split | train / val; exact original split files pending |
| Batch size / epochs | 16 / 250 |
| Seed | 444; original Python/NumPy/torch/CUDA seed-power policy |
| Input width / height | 1280 / 384, original focal/crop/padding handling |
| Classes / reporting | Original 3-class heads; KITTI Car reporting |
| Optimizer | Original custom AdamW, lr 0.0002, weight decay 0.0001 |
| Schedule | Decay 0.5 at 85, 125, 165, 205; no warmup |
| Augmentations | Original photometric/crop/flip/mixup and target focal list |
| Evaluation | Original decoding, topk 50, threshold 0.2; Car IoU 0.7 AP_R40 |

## Infrastructure differences

Resolve the author's absolute focal-embedding path to the pinned `.npy` file;
select GPU 0 via CUDA_VISIBLE_DEVICES. Compile the same CUDA source for the actual
GPU architecture. Explicit run directories and atomic complete state replace
upstream timestamp-only naming/incomplete scheduler/RNG resume bookkeeping.
Validation predictions/GT are read in identical split order. No A0 layers, losses,
optimization updates or filtering rules were altered.

## Dataset, environment and checkpoint provenance

The original repo omits ImageSets. Intended common counts 3712/3769 are not verified
locally; obtain original designated split files and provenance before G1.
Prepared environment: Linux/Python 3.9/PyTorch 2.4.1/CUDA 12.1, not GPU-tested here.
Official validation checkpoint revision/hash are in `sources.json`; it has not
been evaluated locally. Actual ResNet initialization cache hashes are saved by
each GPU run. Leaderboard checkpoints are excluded from held-out val evaluation.

## Actual evidence

| Category | Status |
|---|---|
| Published results | External: official MonoIA Model Zoo; not local evidence |
| Official pretrained validation-weight evaluation | Not yet evaluated |
| Independently trained A0 | Not trained |
| Differences from published results | Pending comparable real execution |

Exact preparation, training and evaluation commands are in README.md.

## Operational bootstrap update

The original Trainer.train_one_epoch remains in use. External criterion/optimizer
hooks observe actual loss/global step without replacing optimization code. The
new run layout uses checkpoints/best.pth and last.pth with complete state; fixed
settings remain inherited. Official validation-weight evaluation requires A0 G1–G4
evidence and remains unexecuted.
