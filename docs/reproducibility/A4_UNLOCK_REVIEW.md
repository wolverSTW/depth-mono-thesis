# A4 Preliminary Research Unlock Review

## Evidence

All values below were read from actual KITTI validation
evaluation artifacts, not manually generated model results.

| Experiment | Best AP3D Moderate R40 (%) |
|---|---:|
| A0 | 11.383556 |
| A1 | 11.108024 |
| A2 | 11.933117 |
| A3 | 10.548573 |

## Preliminary Findings

- A2 has the highest AP3D Moderate among A0-A3.
- A1 has the lowest reported matched-object Camera-Z MAE.
- A2 and A3 show substantial best-to-last AP deterioration.
- Depth-prior experiments have higher inference cost than A0.
- Matched-object distance metrics are conditional on
  successful 3D IoU matching.

## A4 Research Decision

Proceed with A4 flip-consistency prior-reliability ablation
for a maximum of 20 training epochs.

This is preliminary experimentation, not final paper
benchmark reproduction.

Retain dataset provenance, checkpoint integrity,
source cleanliness, G1-G4 GPU verification gates,
and strict epoch-boundary stopping.

A4 reliability is a relative depth-prior heuristic,
not calibrated distance confidence.
