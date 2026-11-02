# E4b — frozen metric-depth evidence with learned query-Z residual

Status: **implementation proposal / CPU-only verification**. E4b has **not** been trained on KITTI, has no AP3D/APBEV/distance measurements, and is not a leaderboard submission.

## Controls and independence

- E4b inherits E4a config; same original MonoIA, train/val split (`kitti_3dop_chen`), augmentation, batch size, seed, losses, optimizer, schedule, calibration preprocessing and frozen verified VKITTI-S weights.
- The only architectural addition is a trainable, 4,929-parameter query-level residual MLP. Its last layer is zero-initialized for **exact E4a initial depth parity**, not an expected final accuracy.
- A new experiment ID, run name, independent checkpoint and provenance identity are required. Original `MonoIA` and DA-V2 upstream code remains untouched.
- E4b must use `dav2_metric_vkitti_small` in both initialization provenance and GPU readiness verification (not relative `dav2_small`).

## Input/output contract

- The frozen metric prior provides one `[B,1,Hm,Wm]` depth map in nominal metres from the *same already-augmented detector image*. This is not a measured sensor depth map, and VKITTI-domain scale may be inaccurate on KITTI.
- Predicted normalized projected 3D centre `pred_boxes[...,0:2]` is sampled via `grid_sample` with `align_corners=False`, mapping `(u,v)` to `(2u-1,2v-1)` in the network image frame. Out-of-range query centres are bypassed without clamping the query position to an apparently valid box.
- For each query the ten numeric inputs are: initial Z/80, sampled depth/80, difference/80, u, v, clipped log variance, max sigmoid class confidence, and 3 predicted dimension channels/10.
- These evidence features are detached from MonoIA's predictions and the frozen prior. The *identity* path from initial predicted Z to corrected Z remains differentiable. The correction head uses the existing MonoIA `loss_depth` via `outputs['pred_depth']` and is also used by KITTI decode; it does **not** replace/deform uncertainty `[...,1]` or alter aux decoder layers.
- Residual `10*tanh(MLP(inputs))` is constrained to ±10 nominal depth units. This is an explicit hyperparameter, not an empirical best value.

## Known risks / prerequisites before GPU expenditure

1. With `depth_scale=normal` and random cropping, training Z targets are **encoded**, `object_camera_Z*crop_scale`; the metric prior is physical-image evidence, not ground-truth camera centre-Z. The residual learns the *existing MonoIA target domain*, without asserting an analytic camera-Z conversion. Validation uses no random crop. This is an architectural limitation to measure rather than hide.
2. Dataset `random_mixup3d=0.3`, random flip, photometric augmentation, and focal-normalizing affine transforms remain **identical to E4a**. A blended image can make the metric prior less reliable. No ground-truth depth, crop scale, donor identity or label is passed to the correction head. Compare E4b against E4a under the same protocol.
3. Metric DA-V2 was trained on synthetic VKITTI2 and has a different domain from real KITTI; the prior is evidence only. An object surface pixel's metric Z is not the 3D box centre-Z.
4. `pred_depth[...,1]` is retained from E4a, so KITTI confidence treatment remains unchanged; **the score is not newly calibrated** for corrected depths.
5. Run dedicated CPU tests and complete regression on local machine. The full real KITTI provenance/splits and G1–G4 CUDA gates are not established by these isolated tests. Do not report benchmark metrics, run full GPU training, or declare KITTI readiness until all gates pass and the reviewed source commit is clean.
6. Do not add metric depth as a stand-alone supervision target, use val labels for training, or change E4a/E4b settings to improve comparison without registering a separate controlled experiment.
