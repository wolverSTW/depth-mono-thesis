# Linux CUDA setup and deployment

Status: prepared commands; no GPU instance was created or GPU command executed here.
Never rent paid resources without owner approval. Scripts configure an existing
authorized machine; they do not purchase/provision one.

## Select and inspect the machine

Use a Linux NVIDIA machine supporting the documented PyTorch 2.4.1/CUDA 12.1 stack,
with a CUDA 12.1 **developer toolkit** (nvcc), sufficient RAM/VRAM and persistent
output storage. The required VRAM has not been measured; do not claim a particular
GPU will fit batch 16. Run G2 then a capacity check on real data. Any necessary
batch revision must be explicit and identical across compared experiments.

On the selected machine:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvcc --version
df -h
free -h
```

CUDA 12.1 lists Linux driver >=525.60.13 for minor-version compatibility; this is
not a guarantee for every GPU/compiler combination. Check the actual device and
toolkit support before purchase. [NVIDIA release notes](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/index.html).
The CUDA version shown by nvidia-smi describes driver support, not the installed
nvcc version. Do not install host drivers inside a rented container.

For an Ubuntu-based developer image, if permitted:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git ninja-build unzip rsync tmux libgl1 libglib2.0-0
```

Conda must be installed first using its [official platform instructions](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html).
Choose a CUDA 12.1 developer image if nvcc is missing; the setup script does not
silently replace a provider's drivers/toolkit.

## Obtain source and build

The repository must first be pushed with owner approval, or transferred with its
Git history/submodules. Current development is local; no push is assumed.

```bash
git clone --recurse-submodules https://github.com/wolverSTW/depth-mono-thesis.git
cd depth-mono-thesis
git checkout develop
git submodule update --init --recursive
bash scripts/gpu/setup_environment.sh --env-name depth-monoia-gpu
conda activate depth-monoia-gpu
bash scripts/gpu/verify_gpu.sh --gpu-device 0
```

Setup installs Python 3.9, official torch 2.4.1/torchvision 0.19.1 cu121 wheels and
requirements/linux-cu121.txt, then builds original MonoIA CUDA sources for the
actual GPU. Defaults match the official MonoIA core stack and
[PyTorch's versioned commands](https://pytorch.org/get-started/previous-versions/).
It refuses missing prerequisites and does not introduce AMP. Record its output,
`python -m pip freeze`, `nvidia-smi`, nvcc and compiler versions in persistent logs.
Full compatibility is still pending GPU execution.

## Weights, real data and gates

Set locations to your actual mounted storage. Example layout:

```bash
export KITTI_ROOT=/datasets/KITTIDataset
export OUTPUT_ROOT=/persistent/depth-monoia/experiments
export CHECKPOINT_DIR=/persistent/depth-monoia/pretrained
mkdir -p "$OUTPUT_ROOT" "$CHECKPOINT_DIR"
bash scripts/gpu/download_pretrained.sh --name dav2_small --checkpoint-dir "$CHECKPOINT_DIR"
bash scripts/gpu/download_pretrained.sh --name monoia_val --checkpoint-dir "$CHECKPOINT_DIR"
bash scripts/gpu/prepare_dataset.sh --dataset-root "$KITTI_ROOT" --manifest local.dataset.json
```

KITTI requires licensed official image_2/label_2/calib files and original designated
ImageSets/train.txt and val.txt. Populate local.dataset.json from the documented
example with independently verified split hashes/source/counts. No script invents
splits, downloads labels without authorization, or rewrites calibration.

For each A0–A3, run the four gates, using independently executed real reference
KITTI predictions/metrics:

```bash
python scripts/research.py verify --experiment A3 --gpu \
  --dataset-root "$KITTI_ROOT" --manifest local.dataset.json \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --reference-predictions /evidence/val_predictions \
  --reference-metrics /evidence/val_reference.json --output results/A3_gates.json
```

G4 JSON requirements are in README.md. No reference metrics are supplied here.
Only matching PASS evidence permits training. Hardware preflight alone is not
G1–G4. See GPU_TRAINING_GUIDE.md for training, monitoring, checkpoints and evaluation.

## Provider-specific boundaries

- **Vast.ai:** select storage and an appropriate developer image in the console
  only after approval. Container storage is deleted with instance destruction;
  volumes persist but are tied to a physical host. Keep a verified off-machine
  archive too. [Official storage guide](https://docs.vast.ai/guides/instances/storage/types).
- **Colab/Pro:** GPU availability and runtime limits vary. Inspect actual versions
  and run through the supported notebook interface. Do not use free-tier SSH or
  bypass its usage limits. Treat the VM as temporary; copy verified archives to
  durable storage before disconnecting. [Official FAQ](https://research.google.com/colaboratory/faq.html).
- **Kaggle:** inspect the actual notebook image, accelerator, internet/storage
  controls and output-retention rules in the account. Do not assume its Python,
  CUDA or torch match this project. If an isolated compatible environment/toolkit
  cannot be created, do not silently alter pins. [Official runtime image source](https://github.com/Kaggle/docker-python).

Before terminating any machine: request an epoch-boundary stop, verify the last
checkpoint, make/download/restore-check the archive, and only then use the provider
console to stop/destroy the instance. Scripts never terminate instances. Follow
ARTIFACT_DOWNLOAD_GUIDE.md and GPU_RECOVERY_GUIDE.md.
