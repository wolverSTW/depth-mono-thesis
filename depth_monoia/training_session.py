"""Session-only epoch stops; checkpoint/configuration identity stays with the runner."""
import json
import signal
import time

from .artifacts import timestamp, write_json


def validate_stop_after_epoch(boundary, max_epoch, start_epoch=0):
    if boundary is None:
        return
    if type(boundary) is not int or not 1 <= boundary <= max_epoch:
        raise ValueError("stop-after-epoch must be an integer from 1 through configured trainer.max_epoch")
    if start_epoch >= boundary:
        raise ValueError("Checkpoint already reached or passed stop-after-epoch; omit it or choose a later boundary")


class StopRequest:
    def __init__(self):
        self.reason = None

    def __call__(self, signum, frame):
        if self.reason is not None:
            raise KeyboardInterrupt("Second stop signal: replay unfinished epoch on resume")
        self.reason = signal.Signals(signum).name


def run_epoch_session(run, lock, events, start, max_epoch, commit_epoch, stop,
                      stop_after_epoch=None):
    """commit_epoch returns only after validation and checkpoint/hash publication.

    This function never truncates the scheduler's configured horizon. Epoch numbers
    at the CLI are one-based; checkpoint next_epoch is the next zero-based index.
    """
    validate_stop_after_epoch(stop_after_epoch, max_epoch, start)
    if (run / "completed.json").exists() and start < max_epoch:
        raise ValueError("Cannot rewind a completed run in place; preserve its completion evidence")
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    next_epoch, tick_session = start, time.perf_counter()
    try:
        for epoch in range(start, max_epoch):
            commit_epoch(epoch)
            next_epoch = epoch + 1
            reason = stop.reason
            if reason is None and stop_after_epoch is not None and next_epoch >= stop_after_epoch:
                reason = "stop_after_epoch"
            if reason is not None and next_epoch < max_epoch:
                with lock:
                    state = json.loads((run / "training_state.json").read_text(encoding="utf-8"))
                    if state.get("next_epoch") != next_epoch:
                        raise ValueError("Epoch publication marker does not match the stop boundary")
                    write_json(run / "training_state.json", {**state,
                        "status": "preliminary_stopped" if stop_after_epoch is not None and next_epoch >= stop_after_epoch else "stopped",
                        "stop_reason": reason, "stop_after_epoch": stop_after_epoch,
                        "timestamp_utc": timestamp()})
                    events.event("stopped_at_epoch_boundary", next_epoch=next_epoch,
                                 stop_reason=reason, stop_after_epoch=stop_after_epoch)
                break
        if next_epoch == max_epoch:
            with lock:
                state = json.loads((run / "training_state.json").read_text(encoding="utf-8"))
                if state.get("next_epoch") != next_epoch:
                    raise ValueError("Epoch publication marker does not match full completion")
                write_json(run / "completed.json", {"completed_epochs": next_epoch,
                    **{key: state[key] for key in ("global_step", "best_AP3D_Moderate_R40") if key in state},
                    "timestamp_utc": timestamp(), "last_session_seconds": time.perf_counter() - tick_session})
                write_json(run / "training_state.json", {**state, "status": "completed",
                    "stop_reason": "full_schedule_completed", "timestamp_utc": timestamp()})
                events.event("training_completed")
    except BaseException as error:
        events.event("interrupted", error_type=type(error).__name__, message=str(error),
                     recovery="Resume last.pth; unfinished epoch is replayed")
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        events.event("session_ended", duration_seconds=time.perf_counter() - tick_session)
        events.close()
    return next_epoch
