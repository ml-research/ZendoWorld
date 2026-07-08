"""Append-only compute log for paper-ready reporting.

Usage from the experiment driver:

    import compute_log
    compute_log.start(args=vars(args))   # parent process only
    ...
    compute_log.end()                    # parent process only (atexit ok)

Render call sites and prompter generate paths emit per-event lines via
record_render() and record_tokens(). Worker subprocesses inherit the
log path through the ZENDO_COMPUTE_LOG env var and append to the same
file (POSIX guarantees atomic writes under PIPE_BUF; events are well
under that).

Summarize after the run:

    python compute_log.py --summarize logs/compute_<run>.jsonl --out summary.json
"""
import argparse
import json
import os
import socket
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

_LOG_ENV = "ZENDO_COMPUTE_LOG"


def seed_log_path(player_tag: str, seed, log_root: str = "logs") -> str:
    """Stable per-(player, seed) log file path."""
    return os.path.abspath(
        os.path.join(log_root, f"compute_{player_tag}_seed_{seed}.jsonl")
    )


def _log_path() -> Optional[str]:
    return os.environ.get(_LOG_ENV)


def _emit(event: dict) -> None:
    path = _log_path()
    if not path:
        return
    event = {"ts": time.time(), "pid": os.getpid(), **event}
    line = json.dumps(event, default=str)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def _probe_gpus() -> list:
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        return [
            {"index": i, "name": torch.cuda.get_device_name(i)}
            for i in range(torch.cuda.device_count())
        ]
    except Exception:
        return []


def _probe_energy_mj() -> dict:
    """Best-effort per-GPU total energy in millijoules via NVML.
    Returns {} if pynvml is unavailable or the driver doesn't support it."""
    try:
        import pynvml
        pynvml.nvmlInit()
        out = {}
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                out[i] = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            except Exception:
                pass
        pynvml.nvmlShutdown()
        return out
    except Exception:
        return {}


def start(args: Optional[dict] = None, scope: Optional[dict] = None) -> None:
    """Emit an experiment_start event into the currently-active log file."""
    if not _log_path():
        return
    _emit({
        "type": "experiment_start",
        "hostname": socket.gethostname(),
        "gpus": _probe_gpus(),
        "energy_mj_at_start": _probe_energy_mj(),
        "args": args or {},
        "scope": scope or {},
    })


def end(scope: Optional[dict] = None) -> None:
    """Emit an experiment_end event into the currently-active log file."""
    if not _log_path():
        return
    _emit({
        "type": "experiment_end",
        "energy_mj_at_end": _probe_energy_mj(),
        "scope": scope or {},
    })


@contextmanager
def use_log(path: str, args: Optional[dict] = None, scope: Optional[dict] = None):
    """Temporarily redirect compute-log writes to `path`.

    Sets ZENDO_COMPUTE_LOG so child processes spawned inside the block
    (e.g. Blender render subprocesses) also write to this file. Emits a
    paired start/end. Restores the prior log path on exit.
    """
    prev = os.environ.get(_LOG_ENV)
    os.environ[_LOG_ENV] = path
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    start(args=args, scope=scope)
    try:
        yield path
    finally:
        end(scope=scope)
        if prev is None:
            os.environ.pop(_LOG_ENV, None)
        else:
            os.environ[_LOG_ENV] = prev


def record_render(duration_s: float) -> None:
    _emit({"type": "render", "duration_s": float(duration_s)})


def record_tokens(n: int, model: str = "") -> None:
    if n <= 0:
        return
    _emit({"type": "tokens", "n": int(n), "model": model})


@contextmanager
def timed_render():
    t0 = time.time()
    try:
        yield
    finally:
        record_render(time.time() - t0)


def _energy_delta_mj(s_event: dict, e_event: dict) -> float:
    e0 = s_event.get("energy_mj_at_start") or {}
    e1 = e_event.get("energy_mj_at_end") or {}
    if not e0 or not e1:
        return 0.0
    total = 0.0
    for k, v in e0.items():
        v1 = e1.get(k, e1.get(str(k), v))
        total += float(v1) - float(v)
    return total


def summarize(path: str) -> dict:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    if not events:
        return {}

    starts = [e for e in events if e["type"] == "experiment_start"]
    ends = [e for e in events if e["type"] == "experiment_end"]
    renders = [e for e in events if e["type"] == "render"]
    tokens = [e for e in events if e["type"] == "tokens"]

    # Pair starts/ends within each pid in order. Multiple workers may write
    # to the same file (different tasks, same seed); pairing per-pid keeps
    # nesting clean.
    by_pid_starts: dict = defaultdict(list)
    by_pid_ends: dict = defaultdict(list)
    for e in starts:
        by_pid_starts[e.get("pid", 0)].append(e)
    for e in ends:
        by_pid_ends[e.get("pid", 0)].append(e)

    active_seconds = 0.0
    energy_mj = 0.0
    for pid, ss in by_pid_starts.items():
        es = by_pid_ends.get(pid, [])
        ss = sorted(ss, key=lambda x: x["ts"])
        es = sorted(es, key=lambda x: x["ts"])
        for s_ev, e_ev in zip(ss, es):
            active_seconds += e_ev["ts"] - s_ev["ts"]
            energy_mj += _energy_delta_mj(s_ev, e_ev)

    wall_start = min(e["ts"] for e in starts) if starts else min(e["ts"] for e in events)
    wall_end = max(e["ts"] for e in ends) if ends else max(e["ts"] for e in events)
    elapsed_s = wall_end - wall_start

    gpus = starts[0]["gpus"] if starts else []
    n_gpus = len(gpus)

    by_model: dict = {}
    for e in tokens:
        key = e.get("model", "")
        by_model[key] = by_model.get(key, 0) + e["n"]

    energy_kwh = energy_mj / 1e6 / 3600.0 if energy_mj > 0 else None

    scope = starts[0].get("scope", {}) if starts else {}

    return {
        "scope": scope,
        "start_iso": datetime.fromtimestamp(wall_start).isoformat(),
        "end_iso": datetime.fromtimestamp(wall_end).isoformat(),
        "elapsed_wall_hours": elapsed_s / 3600.0,
        "active_wall_hours": active_seconds / 3600.0,
        "n_gpus": n_gpus,
        "gpu_hours_active": active_seconds * n_gpus / 3600.0,
        "gpu_hours_elapsed": elapsed_s * n_gpus / 3600.0,
        "gpus": gpus,
        "n_sessions": min(sum(len(v) for v in by_pid_starts.values()),
                          sum(len(v) for v in by_pid_ends.values())),
        "n_renders": len(renders),
        "render_seconds_total": sum(e["duration_s"] for e in renders),
        "n_tokens_total": sum(e["n"] for e in tokens),
        "tokens_by_model": by_model,
        "energy_kwh": energy_kwh,
        "host": starts[0].get("hostname", "") if starts else "",
        "args": starts[0].get("args", {}) if starts else {},
    }


def main():
    import glob as _glob

    p = argparse.ArgumentParser()
    p.add_argument(
        "--summarize",
        required=True,
        help="Path or glob pattern matching JSONL log file(s)",
    )
    p.add_argument("--out", help="Optional path to write JSON summary")
    a = p.parse_args()

    paths = sorted(_glob.glob(a.summarize)) or [a.summarize]
    if len(paths) == 1:
        result = summarize(paths[0])
    else:
        result = {os.path.basename(p): summarize(p) for p in paths}

    text = json.dumps(result, indent=2)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
