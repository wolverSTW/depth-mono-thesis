from copy import deepcopy
from pathlib import Path
import os
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {"A0": "configs/baseline/A0.yaml", "A1": "configs/ablation/A1.yaml",
           "A2": "configs/ablation/A2.yaml", "A3": "configs/proposed/A3.yaml", "A4": "configs/ablation/A4.yaml"}
RUN_NAMES = {"A0": "A0_baseline", "A1": "A1_simple_fusion", "A2": "A2_gated_fusion",
             "A3": "A3_intrinsic_fusion", "A4": "A4_confidence_fusion"}


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
    expected_modes = {"A0": None, "A1": "simple", "A2": "gated", "A3": "intrinsic", "A4": "confidence"}
    if cfg["depth"]["mode"] != expected_modes[cfg["experiment"]["id"]]:
        raise ValueError("Experiment/fusion mode mismatch")
    if cfg["depth"]["alpha_init"] != 0.0 or not cfg["depth"]["frozen"]:
        raise ValueError("Initial experiments require zero-alpha fusion and frozen DA-V2")
    return cfg
