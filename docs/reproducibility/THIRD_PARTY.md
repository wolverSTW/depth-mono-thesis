# Third-party sources

The Git submodules retain all upstream notices and history. MonoIA is MIT licensed;
individual inherited DETR/Deformable DETR files retain their Apache/Facebook notices.
Depth Anything V2 code and Small weights are Apache-2.0; Base/Large/Giant weights
have different, noncommercial terms. Only Small is configured here.

No upstream implementation is copied into the extension package. A0 uses the
original builder and forward. Extension hooks operate only in A1–A4. Checkpoints
and KITTI data stay outside version control. KITTI access and usage terms apply.

Checkpoint revision IDs and SHA-256 values were obtained from the official
Hugging Face model API on 2026-10-29. A URL alone is not proof that local bytes match;
the verification command checks the bytes. The upstream ResNet50-vd initialization
is downloaded by MonoIA itself; each run records the torch hub cache file hashes.
