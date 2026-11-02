from copy import deepcopy
from pathlib import Path
import os
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {"A0": "configs/baseline/A0.yaml", "A1": "configs/ablation/A1.yaml",
           "A2": "configs/ablation/A2.yaml", "A3": "configs/proposed/A3.yaml", "A4": "configs/ablation/A4.yaml",
           "E1": "configs/experiments/E1.yaml", "E2": "configs/experiments/E2.yaml",
           "E3": "configs/experiments/E3.yaml", "E4a": "configs/experiments/E4a.yaml",
           "E4b": "configs/experiments/E4b.yaml"}
RUN_NAMES = {"A0": "A0_baseline", "A1": "A1_simple_fusion", "A2": "A2_gated_fusion",
             "A3": "A3_intrinsic_fusion", "A4": "A4_confidence_fusion",
             "E1": "E1_encoder_only_gated_fusion", "E2": "E2_dense_depth_only_gated_fusion",
             "E3": "E3_dense_depth_intrinsic_gate", "E4a": "E4a_metric_vkitti_gated_fusion",
             "E4b": "E4b_metric_vkitti_distance_correction"}


def merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = cfg.pop("extends", None)
    if parent:
        cfg = merge(load_config(path.parent / parent), cfg)
    return cfg


def resolve_config(path, dataset_root=None, overrides=None):
    cfg = load_config(path)
    changes = deepcopy(cfg["execution"].get("protocol_changes", {}))
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key == "batch_size":
            if value < 1:
                raise ValueError("Batch size must be positive")
            if value != cfg["dataset"]["batch_size"]:
                changes[key] = {"original": changes.get(key, {}).get("original", cfg["dataset"]["batch_size"]), "resolved": value}
            cfg["dataset"]["batch_size"] = value
        elif key == "workers":
            if value < 0:
                raise ValueError("Worker count must be nonnegative")
            cfg["execution"]["workers"] = value
        elif key == "seed":
            if not 0 <= value < 65536:
                raise ValueError("Seed must fit original NumPy seed**2 policy")
            if value != cfg["random_seed"]:
                changes[key] = {"original": changes.get(key, {}).get("original", cfg["random_seed"]), "resolved": value}
            cfg["random_seed"] = value
        else:
            raise ValueError("Unsupported runtime override: " + key)
    cfg["execution"]["protocol_changes"] = changes
    cfg["dataset"]["root_dir"] = str(Path(dataset_root or os.environ.get(
        "KITTI_ROOT", cfg["dataset"]["root_dir"])).resolve())
    cfg["model"]["focal_embedding_path"] = str(ROOT / "MonoIA/focal_list_feat.npy")
    if cfg["dataset"]["train_split"] != "train" or cfg["dataset"]["test_split"] != "val":
        raise ValueError("Thesis runner requires disjoint train/val; no trainval training")
    if cfg["model"]["device"] != "cuda":
        raise ValueError("MonoIA training/evaluation requires CUDA")
    if cfg["execution"]["devices"] != 1 or cfg["execution"]["precision"] != "float32":
        raise ValueError("Audited runner supports one GPU per process and float32 only")
    if cfg["execution"].get("gradient_accumulation_steps", 1) != 1:
        raise ValueError("Gradient accumulation is unsupported; effective batch equals the configured batch")
    expected_modes = {"A0": None, "A1": "simple", "A2": "gated", "A3": "intrinsic",
                      "A4": "confidence", "E1": "gated", "E2": "gated", "E3": "intrinsic", "E4a": "gated", "E4b": "gated"}
    experiment_id = cfg["experiment"]["id"]
    if experiment_id not in expected_modes or cfg["depth"]["mode"] != expected_modes[experiment_id]:
        raise ValueError("Experiment/fusion mode mismatch")
    if cfg["depth"]["alpha_init"] != 0.0 or not cfg["depth"]["frozen"]:
        raise ValueError("Initial experiments require zero-alpha fusion and frozen DA-V2")
    # Strict, explicit representation contract prevents accidentally running E1
    # with the 385-channel encoder+relative-depth prior (or vice versa).
    representation_contracts = {
        "E1": ("four_DINO_intermediates_only", 384),
        "E2": ("single_standardized_relative_depth_map_only", 1),
        "E3": ("single_standardized_relative_depth_map_only", 1),
        "E4a": ("single_metric_vkitti_depth_in_meters_only", 1),
        "E4b": ("single_metric_vkitti_depth_in_meters_only", 1),
    }
    expected_representation, expected_channels = representation_contracts.get(
        experiment_id, ("four_DINO_intermediates_plus_standardized_relative_depth", 385)
    )
    if (cfg["depth"].get("representation") != expected_representation or
            cfg["depth"].get("prior_channels", 385) != expected_channels):
        raise ValueError("Experiment/prior representation and channel contract mismatch")
    if experiment_id in ("E4a", "E4b") and (cfg["depth"].get("checkpoint_key") != "dav2_metric_vkitti_small" or
                                       cfg["depth"].get("max_depth_m") != 80.0 or
                                       not cfg["depth"].get("metric_units") == "metres"):
        raise ValueError("E4a metric checkpoint, units and max-depth contract mismatch")
    correction = cfg["depth"].get("distance_correction")
    if experiment_id == "E4b":
        if not isinstance(correction, dict) or set(correction) != {
            "hidden_dim", "max_delta_m", "feature_schema", "auxiliary"}:
            raise ValueError("E4b correction configuration missing/ambiguous")
        if (type(correction["hidden_dim"]) is not int or correction["hidden_dim"] < 1 or
                type(correction["max_delta_m"]) not in (int, float) or
                not 0.0 < correction["max_delta_m"] <= 80.0 or
                correction["feature_schema"] != "query_uv_z_logvar_class_dims_v1" or
                correction["auxiliary"] != "unchanged"):
            raise ValueError("E4b distance correction contract mismatch")
    elif correction is not None:
        raise ValueError("Distance correction must only be enabled for E4b")
    return cfg
