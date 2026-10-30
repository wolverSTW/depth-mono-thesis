"""CPU session/checkpoint tests, never KITTI/detector/benchmark evidence.

The scalar parameter exercises genuine optimizer/scheduler/RNG serialization.
No images, labels, detections, benchmark metrics or passing gate evidence exist
in these isolated temporary fixtures.
"""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import runpy
import signal
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import numpy as np
import torch

from depth_monoia import checkpoint, runner
from depth_monoia.artifacts import RunLock, write_json
from depth_monoia.config import CONFIGS, ROOT, load_config
from depth_monoia.training_session import StopRequest, run_epoch_session, validate_stop_after_epoch

spec = importlib.util.spec_from_file_location("session_test_upstream_scheduler", ROOT / "MonoIA/lib/helpers/scheduler_helper.py")
upstream_scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream_scheduler)


class Events:
    def __init__(self):
        self.items, self.closed = [], False

    def event(self, name, **fields):
        self.items.append((name, fields))

    def close(self):
        self.closed = True


class TrainingSession(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run = Path(self.temporary.name)
        self.lock = RunLock(self.run)
        self.events, self.stop = Events(), StopRequest()
        self.saved_rng = checkpoint.rng_state()

    def tearDown(self):
        checkpoint.restore_rng(self.saved_rng)
        self.temporary.cleanup()

    def commit(self, epoch):
        # A publication marker only: no fake evaluation or detector outputs.
        write_json(self.run / "training_state.json", {"next_epoch": epoch + 1, "global_step": epoch + 1}, self.lock)

    def session(self, callback, start=0, total=250, boundary=None):
        return run_epoch_session(self.run, self.lock, self.events, start, total, callback, self.stop, boundary)

    def test_stops_after_twentieth_committed_epoch_without_completion(self):
        committed = []
        def commit(epoch):
            self.assertFalse((self.run / "completed.json").exists())
            self.commit(epoch)
            committed.append(epoch + 1)
        self.assertEqual(self.session(commit, boundary=20), 20)
        self.assertEqual(committed, list(range(1, 21)))
        self.assertFalse((self.run / "completed.json").exists())
        state = json.loads((self.run / "training_state.json").read_text())
        self.assertEqual((state["next_epoch"], state["status"], state["stop_reason"]),
                         (20, "preliminary_stopped", "stop_after_epoch"))
        self.assertTrue(self.events.closed)

    def test_callback_failure_at_stop_boundary_never_marks_commit_or_completion(self):
        def commit(epoch):
            if epoch == 19:
                raise OSError("Interrupted epoch publication")
            self.commit(epoch)
        with self.assertRaisesRegex(OSError, "publication"):
            self.session(commit, boundary=20)
        self.assertEqual(json.loads((self.run / "training_state.json").read_text())["next_epoch"], 19)
        self.assertFalse((self.run / "completed.json").exists())
        self.assertNotIn("stopped_at_epoch_boundary", [name for name, _ in self.events.items])

    def test_resume_at_epoch_21_and_completion_only_at_full_horizon(self):
        self.session(self.commit, boundary=20)
        resumed = []
        def commit(epoch):
            resumed.append(epoch + 1)
            self.commit(epoch)
        self.session(commit, start=20, boundary=21)
        self.assertEqual(resumed, [21])
        self.assertFalse((self.run / "completed.json").exists())
        self.session(self.commit, start=21, total=250)
        self.assertEqual(json.loads((self.run / "completed.json").read_text())["completed_epochs"], 250)

    def test_stop_at_full_horizon_is_completed(self):
        self.assertEqual(self.session(self.commit, start=249, boundary=250), 250)
        self.assertTrue((self.run / "completed.json").is_file())

    def test_invalid_and_already_reached_boundaries_preserve_artifacts(self):
        self.commit(20)
        before = (self.run / "training_state.json").read_bytes()
        for boundary in (0, -1, 251, 20.5, "20", True, False):
            with self.subTest(boundary=boundary), self.assertRaisesRegex(ValueError, "integer"):
                validate_stop_after_epoch(boundary, 250)
        for start in (20, 21, 50):
            with self.subTest(start=start), self.assertRaisesRegex(ValueError, "already reached"):
                self.session(lambda epoch: self.fail("Must not train past requested stop"), start=start, boundary=20)
        self.assertEqual((self.run / "training_state.json").read_bytes(), before)

    def test_completed_runs_cannot_be_rewound_in_place(self):
        write_json(self.run / "completed.json", {"completed_epochs": 250})
        with self.assertRaisesRegex(ValueError, "rewind"):
            self.session(self.commit, start=20, boundary=21)

    def test_first_sigint_or_sigterm_finishes_current_commit_and_restores_handlers(self):
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        for sig in previous:
            with self.subTest(signal=sig):
                self.stop, self.events = StopRequest(), Events()
                calls = []
                def commit(epoch):
                    if epoch == 2:
                        signal.getsignal(sig)(sig, None)
                    self.commit(epoch)
                    calls.append(epoch + 1)
                self.assertEqual(self.session(commit, boundary=20), 3)
                self.assertEqual(calls, [1, 2, 3])
                state = json.loads((self.run / "training_state.json").read_text())
                self.assertEqual(state["stop_reason"], signal.Signals(sig).name)
                self.assertFalse((self.run / "completed.json").exists())
                self.assertEqual({s: signal.getsignal(s) for s in previous}, previous)

    def test_second_signal_aborts_and_preserves_previous_commit(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=sig):
                self.stop, self.events = StopRequest(), Events()
                def commit(epoch):
                    if epoch == 1:
                        self.stop(sig, None)
                        self.stop(sig, None)
                    self.commit(epoch)
                with self.assertRaisesRegex(KeyboardInterrupt, "Second stop signal"):
                    self.session(commit, boundary=20)
                self.assertEqual(json.loads((self.run / "training_state.json").read_text())["next_epoch"], 1)
                self.assertTrue(self.events.closed)

    def _check_signal_stop_at_epoch_seven(self, sig):
        committed = []
        def commit(epoch):
            if epoch == 6:
                signal.getsignal(sig)(sig, None)
            self.commit(epoch)
            committed.append(epoch + 1)
        self.assertEqual(self.session(commit, boundary=20), 7)
        self.assertEqual(committed, list(range(1, 8)))
        state = json.loads((self.run / "training_state.json").read_text())
        self.assertEqual(state["status"], "stopped")
        self.assertEqual(state["stop_reason"], signal.Signals(sig).name)
        self.assertEqual(state["next_epoch"], 7)
        self.assertEqual(state["stop_after_epoch"], 20)
        self.assertFalse((self.run / "completed.json").exists())
        stopped = [fields for name, fields in self.events.items if name == "stopped_at_epoch_boundary"]
        self.assertEqual(len(stopped), 1)
        self.assertEqual(stopped[0]["stop_reason"], signal.Signals(sig).name)
        self.assertEqual(stopped[0]["next_epoch"], 7)

    def test_sigint_at_epoch_seven_before_requested_twenty_is_stopped(self):
        self._check_signal_stop_at_epoch_seven(signal.SIGINT)

    def test_sigterm_at_epoch_seven_before_requested_twenty_is_stopped(self):
        self._check_signal_stop_at_epoch_seven(signal.SIGTERM)

    def test_checkpoint_resume_preserves_identity_rng_optimizer_scheduler_and_warmup(self):
        for use_warmup in (False, True):
            with self.subTest(warmup=use_warmup):
                cfg = load_config(ROOT / CONFIGS["A0"])
                # Optional branch coverage only, not an experiment config change.
                cfg["lr_scheduler"]["warmup"] = use_warmup
                scheduler_cfg = cfg["lr_scheduler"]
                identity = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
                metadata = {"identity": identity, "config": copy.deepcopy(cfg)}
                def components():
                    model = torch.nn.Linear(1, 1, bias=False)
                    model.weight.data.fill_(1.0)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"],
                                                   weight_decay=cfg["optimizer"]["weight_decay"])
                    scheduler, warmup = upstream_scheduler.build_lr_scheduler(scheduler_cfg, optimizer, -1)
                    return model, optimizer, scheduler, warmup
                def step(parts, epoch):
                    model, optimizer, scheduler, warmup = parts
                    optimizer.zero_grad()
                    (model.weight.square().sum() * (random.random() + float(np.random.rand()) + torch.rand(()))).backward()
                    optimizer.step()
                    (warmup if warmup is not None and epoch < 5 else scheduler).step()
                random.seed(444)
                np.random.seed(444)
                torch.manual_seed(444)
                original = components()
                path = self.run / "scalar_state.pth"
                def commit_original(epoch):
                    step(original, epoch)
                    checkpoint.save(path, *original, epoch + 1, None, metadata, global_step=epoch + 1)
                    self.commit(epoch)
                self.assertEqual(self.session(commit_original, boundary=20), 20)
                saved = torch.load(path, map_location="cpu", weights_only=False)
                self.assertEqual(saved["scheduler"], original[2].state_dict())
                self.assertEqual(saved["warmup"], original[3].state_dict() if original[3] else None)
                # Cross all four original LR decay boundaries after restoration.
                for epoch in range(20, 206):
                    step(original, epoch)
                resumed = components()
                start, best, global_step = checkpoint.resume(path, *resumed, metadata)
                self.assertEqual((start, best, global_step), (20, None, 20))
                self.assertEqual(saved["metadata"]["identity"], identity)
                self.assertEqual(metadata["config"], cfg)
                self.assertEqual(resumed[2].state_dict(), saved["scheduler"])
                if resumed[3]:
                    self.assertEqual(resumed[3].state_dict(), saved["warmup"])
                resumed_epochs = []
                def commit_resumed(epoch):
                    step(resumed, epoch)
                    resumed_epochs.append(epoch + 1)
                    checkpoint.save(self.run / "resumed_scalar_state.pth", *resumed, epoch + 1,
                                    None, metadata, global_step=epoch + 1)
                    self.commit(epoch)
                self.session(commit_resumed, start=start, boundary=21)
                self.assertEqual(resumed_epochs, [21])
                restored = torch.load(self.run / "resumed_scalar_state.pth", map_location="cpu", weights_only=False)
                self.assertEqual(restored["metadata"]["identity"], saved["metadata"]["identity"])
                self.assertEqual(restored["metadata"]["config"], saved["metadata"]["config"])
                self.assertEqual((restored["next_epoch"], restored["global_step"]), (21, 21))
                for epoch in range(21, 206):
                    step(resumed, epoch)
                torch.testing.assert_close(resumed[0].weight, original[0].weight, rtol=0, atol=0)
                self.assertEqual(resumed[2].state_dict(), original[2].state_dict())
                self.assertEqual(resumed[1].param_groups[0]["lr"], original[1].param_groups[0]["lr"])
                for key, value in original[1].state_dict()["state"][0].items():
                    torch.testing.assert_close(resumed[1].state_dict()["state"][0][key], value, rtol=0, atol=0)

    def test_cli_forwards_boundary_separately_from_configuration_overrides(self):
        for experiment in ("A0", "A1", "A2", "A3"):
            with self.subTest(experiment=experiment), patch.object(sys, "argv", ["research.py", "train",
                    "--experiment", experiment, "--manifest", "unused-routing-only", "--gates", "unused-routing-only",
                    "--stop-after-epoch", "20"]), patch.object(runner, "train") as target:
                runpy.run_path(str(ROOT / "scripts/research.py"), run_name="__main__")
                self.assertEqual(target.call_args.kwargs["stop_after_epoch"], 20)
                self.assertNotIn("stop_after_epoch", target.call_args.args[6])

    def test_invalid_cli_boundary_fails_before_training(self):
        for value in ("0", "-1", "251", "20.5", "invalid"):
            with self.subTest(value=value), patch.object(sys, "argv", ["research.py", "train", "--stop-after-epoch", value]), \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                runpy.run_path(str(ROOT / "scripts/research.py"), run_name="__main__")
            self.assertEqual(error.exception.code, 2)

    def test_preliminary_boundary_cannot_bypass_failed_gates(self):
        cfg = load_config(ROOT / CONFIGS["A0"])
        path = self.run / "failing_gates.json"
        write_json(path, {"identity": "unit-failure-only"})  # No fabricated PASS gates.
        with patch.object(runner, "preparation", return_value=(cfg, {"identity": "unit-failure-only"})), \
                patch.object(runner, "require_cuda") as cuda:
            with self.assertRaisesRegex(ValueError, "All four gates"):
                runner.train(None, None, None, self.run / "unused", path, stop_after_epoch=20)
            cuda.assert_not_called()
