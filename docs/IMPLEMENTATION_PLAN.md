# Milestones and acceptance

| Milestone | Objective and implementation | Dependencies | Acceptance and deliverable |
|---|---|---|---|
| M1 | Instructions, status, docs, safe Git layout | Approved location/remote | Reviewed docs and preserved history; continuation authorized 2026-10-29 |
| M2 | Pinned original builder/config/training/evaluator | Official source and split/checkpoint evidence | Prepared A0; real baseline reproduction remains pending G1–G4/training |
| M3 | Frozen verified DA-V2-Small, normalization/alignment/adapters | M2, official weights | Local strict load/freeze verified; actual KITTI GPU acceptance pending |
| M4 | Simple/gated/intrinsic fusion and A0–A4 configs | M3, inspected interfaces | Local zero-alpha/gradient checks; G2 acceptance pending; A4 locked |
| M5 | GPU setup/train/resume/logging/backup/recovery workflow | Linux CUDA, real KITTI, gates, authorized resource | Actual full A0–A3 runs with complete state/logs and verified backups; not executed |
| M6 | Evaluation, resource export, comparison and final reports | Real checkpoints and completed evaluations | Reviewed raw predictions/metrics/provenance, actual comparison and reports; not executed |

Current work implements operational M5/M6 prerequisites. Finish independent local
implementation and consolidated checks before waiting on GPU/data. User approval
is required for paid resources and GitHub push, not routine reversible edits.
