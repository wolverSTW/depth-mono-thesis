"""Upstream KITTI evaluation, explicit split ordering, auxiliary distance/ECE."""
import json
from pathlib import Path
import numpy as np
import torch
from .model import upstream_path
from .provenance import split_ids, sha256


def annotations(dataset, prediction_dir):
    upstream_path()
    from lib.datasets.kitti.kitti_eval_python import kitti_common as kitti
    ids = [int(s) for s in split_ids(dataset.split_file)]
    expected = {f"{i:06d}.txt" for i in ids}
    actual = {p.name for p in Path(prediction_dir).glob("*.txt")}
    if actual != expected:
        raise ValueError("Prediction coverage differs from validation split")
    # Read both lists in identical split-file order, never directory order.
    return (kitti.get_label_annos(dataset.label_dir, ids),
            kitti.get_label_annos(str(prediction_dir), ids))


def benchmark(dataset, prediction_dir, output):
    upstream_path()
    from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
    gt, dt = annotations(dataset, prediction_dir)
    text, values, moderate = get_official_eval_result(gt, dt, "Car")
    if not np.isfinite(moderate) or any(not np.isfinite(v) for v in values.values()):
        raise FloatingPointError("Nonfinite KITTI evaluation metrics")
    result = {"protocol": "KITTI-compatible upstream evaluator; AP_R40; Car 3D/BEV IoU=0.7",
              "split_sha256": sha256(dataset.split_file),
              "metrics": {k: float(v) for k, v in values.items()},
              "Car_3d_moderate_R40": float(moderate), "raw": text}
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def auxiliary(dataset, prediction_dir, output):
    """Separate from AP: greedy score-ordered one-to-one 3D matching, Moderate Car.

    Eligible and ignored objects/detections use upstream clean_data. Match valid
    GT first, ignore matches to ignored GT; no DontCare removal for 3D, matching
    upstream's 3D convention. ECE is precision calibration on retained detections.
    """
    upstream_path()
    from lib.datasets.kitti.kitti_eval_python.eval import clean_data, calculate_iou_partly
    gt, dt = annotations(dataset, prediction_dir)
    overlaps, _, _, _ = calculate_iou_partly(dt, gt, 2)
    z_errors, range_errors, scores, correct = [], [], [], []
    eligible_count, matched_count = 0, 0
    for g, d, iou in zip(gt, dt, overlaps):
        nvalid, ignored_g, ignored_d, _ = clean_data(g, d, 0, 1)
        eligible_count += nvalid
        used = set()
        for di in np.argsort(-d["score"], kind="stable"):
            if ignored_d[di] != 0:
                continue
            candidates = [gi for gi in range(len(g["name"]))
                          if ignored_g[gi] == 0 and gi not in used and iou[di, gi] > 0.7]
            matched = bool(candidates)
            if matched:
                gi = max(candidates, key=lambda index: iou[di, index])
                used.add(gi)
                matched_count += 1
                z_errors.append(float(d["location"][di, 2] - g["location"][gi, 2]))
                # KITTI locations are bottom centres. Convert to geometric box
                # centres for Euclidean range; dimensions are [length,height,width].
                pc, gc = d["location"][di].copy(), g["location"][gi].copy()
                pc[1] -= d["dimensions"][di, 1] / 2
                gc[1] -= g["dimensions"][gi, 1] / 2
                range_errors.append(float(np.linalg.norm(pc) - np.linalg.norm(gc)))
            elif any(ignored_g[gi] == 1 and iou[di, gi] > 0.7 for gi in range(len(g["name"]))):
                continue
            scores.append(float(d["score"][di]))
            correct.append(float(matched))
    def errors(values):
        if not values:
            return {"MAE_m": None, "RMSE_m": None}
        values = np.asarray(values)
        return {"MAE_m": float(np.abs(values).mean()), "RMSE_m": float(np.sqrt((values ** 2).mean()))}
    result = {"protocol": "auxiliary Moderate Car greedy score-ordered one-to-one 3D IoU>0.7; not AP matching",
              "matched": matched_count, "eligible_gt": eligible_count,
              "recall": matched_count / eligible_count if eligible_count else None,
              "camera_Z": errors(z_errors), "Euclidean_box_centre_range": errors(range_errors),
              "distance_uncertainty": "Not evaluated: no calibrated uncertainty model",
              "detection_ECE": None, "ECE_bins": 15, "detections": len(scores)}
    if scores and all(0 <= s <= 1 and np.isfinite(s) for s in scores):
        s, y = np.asarray(scores), np.asarray(correct)
        bins = np.minimum((s * 15).astype(int), 14)
        result["detection_ECE"] = float(sum(np.mean(bins == b) * abs(s[bins == b].mean() - y[bins == b].mean())
                                               for b in range(15) if (bins == b).any()))
        result["ECE_status"] = "Evaluated on retained final detection scores, not distance confidence"
    else:
        result["ECE_status"] = "Not applicable: empty detections or upstream final scores outside [0,1]; no clipping"
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def infer(model, loader, cfg, prediction_dir):
    upstream_path()
    from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detections
    prediction_dir = Path(prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=False)
    model.eval()
    with torch.no_grad():
        for images, calibs, targets, info in loader:
            images, calibs = images.cuda(), calibs.cuda()
            outputs = model(images, calibs, targets, info["img_size"].cuda(), dn_args=0)
            dets = extract_dets_from_outputs(outputs, K=loader.dataset.max_objs,
                                            topk=cfg["tester"]["topk"]).cpu().numpy()
            calibrations = [loader.dataset.get_calib(int(i)) for i in info["img_id"]]
            info = {k: v.numpy() for k, v in info.items()}
            decoded = decode_detections(dets, info, calibrations, loader.dataset.cls_mean_size,
                                        cfg["tester"]["threshold"])
            for sample_id, rows in decoded.items():
                with (prediction_dir / f"{sample_id:06d}.txt").open("w") as handle:
                    for row in rows:
                        # Same formatting/threshold/decoding as upstream Tester.
                        name = loader.dataset.class_name[int(row[0])]
                        handle.write(name + " 0.0 0" + "".join(f" {v:.2f}" for v in row[1:]) + "\n")


def profile(model, loader, output, warmup=5, steps=20):
    """Real-image full-model wall latency; frozen weights counted in total params."""
    import time
    model.eval()
    images, calibs, targets, info = next(iter(loader))
    images, calibs, sizes = images.cuda(), calibs.cuda(), info["img_size"].cuda()
    timings = []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for step in range(warmup + steps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(images, calibs, targets, sizes)
            torch.cuda.synchronize()
            if step >= warmup:
                timings.append(time.perf_counter() - start)
    result = {"total_parameters_M": sum(p.numel() for p in model.parameters()) / 1e6,
              "trainable_parameters_M": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
              "latency_batch_ms": float(np.mean(timings) * 1000),
              "latency_amortized_image_ms": float(np.mean(timings) * 1000 / len(images)),
              "batch_size": len(images), "input_BCHW": list(images.shape), "precision": "float32",
              "GPU": torch.cuda.get_device_name(), "peak_GPU_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
              "warmup": warmup, "repeated_real_batch_steps": steps,
              "GFLOPs": None, "GFLOPs_status": "Pending: custom CUDA operator FLOP coverage requires validation"}
    try:
        from torch.profiler import profile as torch_profile, ProfilerActivity
        with torch.no_grad(), torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as trace:
            model(images, calibs, targets, sizes)
        result["partial_GFLOPs_per_batch"] = sum(event.flops for event in trace.key_averages()) / 1e9
        result["FLOP_scope"] = "PyTorch-supported operators only; deformable attention and unsupported operations excluded. Not a full-model GFLOPs measurement."
    except Exception as error:
        result["partial_GFLOPs_status"] = type(error).__name__ + ": " + str(error)
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
