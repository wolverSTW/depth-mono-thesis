# Ablation study

Updated: 2026-10-29. No ablation model has been trained/evaluated here.

| ID | Architecture | Configuration | Actual status |
|---|---|---|---|
| A0 | Original MonoIA | configs/baseline/A0.yaml | Prepared; GPU gates pending |
| A1 | DA-V2 + simple residual | configs/ablation/A1.yaml | Implemented; GPU gates pending |
| A2 | DA-V2 + RGB/depth gate | configs/ablation/A2.yaml | Implemented; GPU gates pending |
| A3 | DA-V2 + intrinsic-aware gate | configs/proposed/A3.yaml | Implemented; GPU gates pending |
| A4 | A3 + flip-consistency reliability | configs/ablation/A4.yaml | Optional; execution locked |

## Controls and training settings

All configs inherit the same original split names, full designated training set,
intrinsic embeddings, augmentations, 250 epochs, batch 16, custom AdamW and decay
schedule. Seed 444 is prepared for initial comparable runs. Record additional
seeds when compute permits; one run does not establish statistical significance.
DA-V2 checkpoint, preprocessing, adapters and zero-alpha initialization are shared
across A1–A4. A1/A2 retain MonoIA's original intrinsics and exclude only additional
gate conditioning.

## Pending analysis and limitations

Accuracy, distance errors, ECE, resources and component contributions are Pending.
Compute component deltas only after comparable real runs exist. No complete
ablation study is claimed. Prioritize A0/A3, then A1/A2; A4 waits for functional
A0–A3 real-data evidence and review. G1–G4 must pass before full training.
Document any memory-driven setting revision consistently across experiments and
distinguish it from an exact upstream reproduction. No paid GPU was started.
