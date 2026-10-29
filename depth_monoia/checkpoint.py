"""Atomic epoch-boundary resume, including RNG and scheduler state."""
import os
import random
from pathlib import Path
import numpy as np
import torch


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def save(path, model, optimizer, scheduler, warmup, next_epoch, best, metadata):
    path = Path(path)
    state = {"schema": 1, "model_state": model.state_dict(),
             "optimizer_state": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "warmup": warmup.state_dict() if warmup is not None else None,
             "next_epoch": next_epoch, "best": best, "rng": rng_state(), "metadata": metadata}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def resume(path, model, optimizer, scheduler, warmup, expected):
    # Only load trusted locally generated training states (contain NumPy RNG).
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != 1 or state["metadata"]["identity"] != expected["identity"]:
        raise ValueError("Resume configuration/source/data identity mismatch")
    model.load_state_dict(state["model_state"], strict=True)
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler"])
    if warmup is not None:
        warmup.load_state_dict(state["warmup"])
    restore_rng(state["rng"])
    return state["next_epoch"], state["best"]


def load_weights(path, model, verified_official=False):
    # Official MonoIA files may contain NumPy scalar optimizer/best-result
    # metadata. Full loading is permitted only after their official hash was
    # authenticated by the caller. DA-V2 raw weights use weights_only=True.
    state = torch.load(path, map_location="cpu", weights_only=not verified_official)
    model.load_state_dict(state.get("model_state", state), strict=True)
