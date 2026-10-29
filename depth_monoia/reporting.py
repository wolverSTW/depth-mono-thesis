"""Export executed evidence and aggregate without inventing missing measurements."""
import csv
import hashlib
import json
import math
from pathlib import Path
from .artifacts import write_json, timestamp
from .config import RUN_NAMES


def comparison_group(cfg, data, metadata):
    dataset = {k: v for k, v in cfg["dataset"].items() if k != "root_dir"}
    model = {k: v for k, v in cfg["model"].items() if k != "focal_embedding_path"}
    depth = {k: v for k, v in cfg["depth"].items() if k != "mode"}
    protocol = {"dataset": dataset, "data": data, "model": model, "depth": depth,
                "sources": metadata["sources"], "source_manifest_sha256": metadata["source_manifest_sha256"],
                "extension_code": metadata["extension_code"], "software": metadata["environment"]["packages"],
                **{key: cfg[key] for key in ("random_seed", "optimizer", "lr_scheduler", "trainer", "tester", "execution")}}
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def export_benchmark(directory, result):
    directory = Path(directory)
    with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(sorted(result["metrics"].items()))
    (directory / "kitti_eval.txt").write_text(result["raw"], encoding="utf-8")


def evaluation_metadata(cfg, metadata, checkpoint_info, status, category):
    return {"status": status, "experiment_id": cfg["experiment"]["id"],
            "category": category, "timestamp_utc": timestamp(), "split": "val",
            "checkpoint": checkpoint_info, "identity": metadata["identity"],
            "source_commit": metadata["environment"]["git_commit"],
            "sources": metadata["sources"], "extension_code": metadata["extension_code"],
            "initialization_assets": {key: metadata.get(key, {}) for key in ("backbone_cache", "compiled_ops")},
            "verification_gates": metadata["verification_gates"],
            "comparison_group": comparison_group(cfg, metadata["data"], metadata),
            "data": metadata["data"], "configuration": cfg, "environment": metadata["environment"]}


def aggregate(search_root, output, category="independent_checkpoint"):
    fields = ["experiment_id", "evaluation_status", "category", "comparison_group",
              "AP3D_Easy", "AP3D_Moderate", "AP3D_Hard", "APBEV_Easy", "APBEV_Moderate", "APBEV_Hard",
              "distance_MAE_m", "distance_RMSE_m", "detection_ECE", "parameters_M", "GFLOPs",
              "latency_batch_ms", "latency_amortized_image_ms", "checkpoint_sha256", "evaluation_directory"]
    rows, observed = [], set()
    for marker in sorted(Path(search_root).rglob("evaluation_metadata.json")):
        meta = json.loads(marker.read_text(encoding="utf-8"))
        if meta.get("status") != "complete" or meta.get("category") != category:
            continue
        exp = meta["experiment_id"]
        if exp not in RUN_NAMES:
            raise ValueError("Unknown experiment in evaluation: " + str(marker))
        metrics = json.loads((marker.parent / "metrics.json").read_text())["metrics"]
        row = {"experiment_id": exp, "evaluation_status": "evaluated", "category": category,
               "comparison_group": meta["comparison_group"], "checkpoint_sha256": meta["checkpoint"]["sha256"],
               "evaluation_directory": str(marker.parent.resolve())}
        for prefix, source in (("AP3D", "3d"), ("APBEV", "bev")):
            for difficulty in ("Easy", "Moderate", "Hard"):
                value = metrics[f"Car_{source}_{difficulty.lower()}_R40"]
                if not math.isfinite(value):
                    raise ValueError("Nonfinite AP evidence: " + str(marker))
                row[f"{prefix}_{difficulty}"] = value
        distance = json.loads((marker.parent / "distance_metrics.json").read_text())
        confidence = json.loads((marker.parent / "confidence_metrics.json").read_text())
        resources = json.loads((marker.parent / "resource_metrics.json").read_text())
        row.update(distance_MAE_m=distance["camera_Z"]["MAE_m"], distance_RMSE_m=distance["camera_Z"]["RMSE_m"],
                   detection_ECE=confidence["detection_ECE"], parameters_M=resources["total_parameters_M"],
                   GFLOPs=resources.get("GFLOPs"), latency_batch_ms=resources["latency_batch_ms"],
                   latency_amortized_image_ms=resources["latency_amortized_image_ms"])
        rows.append(row)
        observed.add(exp)
    rows.extend({"experiment_id": exp, "evaluation_status": "not_evaluated", "category": category}
                for exp in RUN_NAMES if exp not in observed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"output": str(output), "evaluated_records": len(rows) - (len(RUN_NAMES) - len(observed)),
            "note": "Compare only equal comparison_group values; missing values are blank. No published results are imported."}
