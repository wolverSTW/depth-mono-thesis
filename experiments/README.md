# Experiment records

Run directories are created as `A0_baseline/run_001`, ..., `A3_intrinsic_fusion/run_001` and ignored
by Git. Configs remain in `configs/`. A4 is optional and execution-locked pending
A0–A3 functionality. No runs exist yet.

Archive actual configs, provenance, gate records, logs and checkpoint hashes on
durable storage. Transfer reviewed evidence into the editable reports after real
execution; datasets and large checkpoints must not enter Git.

Output root/run name are configurable. Actual files appear only after execution;
see ../docs/CHECKPOINT_MANAGEMENT.md for the checkpoint/log/evaluation layout.
