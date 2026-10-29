from copy import deepcopy
from pathlib import Path
import os
import yaml

ROOT = Path(__file__).resolve().parents[1]


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


def resolve_config(path, dataset_root=None):
    cfg = load_config(path)
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
