# Project status

Updated: 2026-10-29. Editable research record; do not replace manual sections.

## Current milestone

M1: clean repository created in the approved subdirectory. Remote was empty when
inspected. Branch: `develop`. Exact milestone commits: `git log --oneline`.

## Completed

- Original project folders untouched; new project initialized separately.
- Official MonoIA and Depth Anything V2 cloned as pinned submodules.
- Licenses, official checkpoint revisions/hashes and proposed CUDA dependencies recorded.
- No push, paid GPU allocation or training performed.

## Pending and blockers

- M2–M4 implementation and consolidated local checks.
- Real KITTI data and split provenance are not available at inspected paths.
- Linux CUDA environment and compiled deformable attention are required for G2–G4.
- Proposed dependency set is source-compatible in intent, not GPU-validated yet.

## Evidence

Local inspection: Python 3.9.25, PyTorch 2.8.0+cpu, torchvision 0.23.0,
NumPy 2.0.2. CUDA unavailable. This differs from the prepared Linux environment.
