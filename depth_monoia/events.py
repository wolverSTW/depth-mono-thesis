"""Observe upstream criterion/optimizer without replacing the epoch implementation."""
import time
import uuid
import torch
from .artifacts import append_event


class TrainingEvents:
    def __init__(self, run, lock, criterion, optimizer, global_step=0):
        self.path, self.lock = run / "logs/events.jsonl", lock
        self.criterion, self.global_step, self.epoch = criterion, global_step, 0
        self.start = time.perf_counter()
        self.session_id = uuid.uuid4().hex
        self.hooks = [criterion.register_forward_hook(self._loss),
                      optimizer.register_step_post_hook(self._step)]

    def _loss(self, module, inputs, output):
        total = sum(value * module.weight_dict[key] for key, value in output.items() if key in module.weight_dict)
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite training loss; previous committed checkpoint is preserved")
        self.loss = float(total.detach())

    def _step(self, optimizer, args, kwargs):
        self.global_step += 1
        append_event(self.path, {"event": "train_step", "epoch": self.epoch,
                     "session_id": self.session_id,
                     "global_step": self.global_step, "loss": self.loss,
                     "lr": [g["lr"] for g in optimizer.param_groups],
                     "session_seconds": time.perf_counter() - self.start,
                     "GPU_allocated_MiB": torch.cuda.memory_allocated() / 2**20}, self.lock)

    def event(self, name, **values):
        append_event(self.path, {"event": name, "epoch": self.epoch,
                     "session_id": self.session_id, "global_step": self.global_step, **values}, self.lock)

    def close(self):
        for hook in self.hooks:
            hook.remove()
