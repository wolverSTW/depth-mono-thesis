# Reproducibility record

Updated: 2026-10-29. Reports remain human-readable and manually editable.

## Source history

Branch: `develop`. M1: `c3e7a5691e5155007fe1def666d490ad909331f4`.
Later milestone hashes: `git log --oneline`. Executed runs save their parent
commit, branch, dirty state and extension-code hashes.
Implementation commit: `006d83e326d91d5dad996e2158cc18575a11e5b4`.
MonoIA: `69d6ee30ca5e44848727387dd44af6d56d62c24f`.
DA-V2: `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`.
Both remain unmodified. No remote push has occurred.

## Software and hardware

Local laptop: Windows CPU, existing depth_monoia Conda environment; no packages
were installed/changed. Python 3.9.25, torch 2.8.0 CPU, torchvision 0.23.0 and
NumPy 2.0.2 are local versions, not upstream CUDA-compatibility evidence.
`docs/reproducibility/LOCAL_VERIFICATION.json` records the actual consolidated
check and its commit/state at execution.
The check ran immediately before the implementation commit, while the tested
source was staged; its dirty flag is preserved as recorded.

Prepared GPU stack: Linux, Python 3.9, torch 2.4.1, torchvision 0.19.1, CUDA toolkit
12.1, NumPy 1.26.4; remaining pins in requirements/linux-cu121.txt. GPU model,
driver/compiler and actual resolved GPU environment are Pending. Runs record
packages, GPU name, CUDA/cuDNN, backbone-cache and compiled-operator hashes.

## Data and checkpoints

KITTI data and designated split provenance: Pending. Fill local.dataset.json from
the example using real data and original split evidence. Validation labels never
enter training. Identity includes split hashes, annotation/calibration inventory,
config, upstream/code hashes and prior checkpoint hash. Full image authentication
relies on trusted acquisition; images receive coverage/size checks.

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
No training, benchmark claim, test submission or cloud resource was started.
Archive actual configs, logs, gates and checkpoint hashes with each real result.
