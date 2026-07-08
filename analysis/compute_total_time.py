"""
compute_total_time.py

Reads gamestate JSON files and sums up turn_durations to report total
wall-clock time.

Usage
-----
    python analysis/compute_total_time.py
"""

import json
import re
from pathlib import Path


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s:.1f}s"
    if m > 0:
        return f"{m}m {s:.1f}s"
    return f"{s:.1f}s"


def _sum_durations(data: dict) -> tuple[float, int] | None:
    """Return (total_seconds, n_turns) from a loaded state dict, or None."""
    durations = data.get("turn_durations")
    if not durations:
        return None
    if isinstance(durations, dict):
        return sum(float(v) for v in durations.values()), len(durations)
    if isinstance(durations, list):
        return sum(float(v) for v in durations), len(durations)
    return None


def compute_total_time(base_dir: Path, participant_id: str | None = None):
    """Sum turn_durations across task state files in *base_dir*.

    Parameters
    ----------
    base_dir : Path
        Directory (possibly with seed sub-dirs) containing task state JSONs.
    participant_id : str or None
        If given, only count files matching ``task_*_state_s_<id>.json``.
        If None, count all ``*.json`` files that contain turn_durations.

    Returns
    -------
    dict with keys: total_seconds, n_games, n_turns
    """
    if participant_id is not None:
        pattern = f"task_*_state_s_{participant_id}.json"
    else:
        pattern = "*.json"

    json_files = list(base_dir.rglob(pattern))

    total_seconds = 0.0
    n_games = 0
    n_turns = 0

    for path in json_files:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue

        result = _sum_durations(data)
        if result is None:
            continue

        total_seconds += result[0]
        n_games += 1
        n_turns += result[1]

    return {
        "total_seconds": total_seconds,
        "n_games": n_games,
        "n_turns": n_turns,
    }


def compute_per_game_times(base_dir: Path):
    """Return per-seed, per-task game durations.

    Returns
    -------
    dict[str, dict[int, float]]
        {seed_name: {task_id: game_seconds}}
    """
    _re_task = re.compile(r"task_(\d+)_state\.json$")
    result: dict[str, dict[int, float]] = {}

    subdirs = sorted(p for p in base_dir.iterdir() if p.is_dir())
    # If subdirs contain task files, treat them as seeds
    if subdirs and any(list(sd.glob("task_*_state.json")) for sd in subdirs):
        for sd in subdirs:
            seed_times: dict[int, float] = {}
            for f in sorted(sd.glob("task_*_state.json")):
                m = _re_task.match(f.name)
                if not m:
                    continue
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                except Exception:
                    continue
                dur = _sum_durations(data)
                if dur is not None:
                    seed_times[int(m.group(1))] = dur[0]
            if seed_times:
                result[sd.name] = seed_times
    else:
        # Flat directory – treat as a single "seed"
        seed_times = {}
        for f in sorted(base_dir.glob("task_*_state.json")):
            m = _re_task.match(f.name)
            if not m:
                continue
            try:
                with open(f) as fh:
                    data = json.load(fh)
            except Exception:
                continue
            dur = _sum_durations(data)
            if dur is not None:
                seed_times[int(m.group(1))] = dur[0]
        if seed_times:
            result[base_dir.name] = seed_times

    return result


# ---------------------------------------------------------------------------
# Agent totals
# ---------------------------------------------------------------------------

AGENTS = [
    ("Oracle Agent",   Path("gamestates/gamestates_zendo_symbolic")),
    ("VLM Agent",      Path("gamestates/gamestates_fullgpt_symbolic")),
    ("Bayesian Agent", Path("gamestates/gamestates_scientist_symbolic")),
    ("VLP Agent",      Path("gamestates/gamestates_vlp_symbolic")),
    ("Random Oracle Agent",      Path("gamestates/gamestates_random_zendo_symbolic")),
]

for agent_name, agent_dir in AGENTS:
    if not agent_dir.exists():
        print(f"{agent_name}: directory not found ({agent_dir})")
        continue
    r = compute_total_time(agent_dir)
    avg = format_duration(r["total_seconds"] / r["n_turns"]) if r["n_turns"] else "N/A"
    print(
        f"{agent_name:<20}  {format_duration(r['total_seconds']):>15}"
        f"  ({r['n_games']} games, {r['n_turns']} turns, avg {avg} / turn)"
    )

    # Per-seed, per-game breakdown
    per_game = compute_per_game_times(agent_dir)
    for seed_name in sorted(per_game):
        tasks = per_game[seed_name]
        seed_total = sum(tasks.values())
        print(f"  {seed_name:<18}  {format_duration(seed_total):>15}  ({len(tasks)} games)")
        for tid in sorted(tasks):
            print(f"    task {tid:<3}  {format_duration(tasks[tid]):>15}")
    print()

# ---------------------------------------------------------------------------
# Human participants (per participant)
# ---------------------------------------------------------------------------

HUMAN_DIR = Path("gamestates/gamestates_study_")

if HUMAN_DIR.exists():
    # Discover unique participant IDs from filenames
    _re_pid = re.compile(r"task_\d+_state_s_(\w+)\.json$")
    pids = sorted({
        m.group(1)
        for f in HUMAN_DIR.glob("task_*_state_s_*.json")
        if (m := _re_pid.search(f.name))
    })

    print(f"\n{'— Human participants ':—<60}")
    print(f"Total unique participants: {len(pids)}")
    print()

    # Count how often each task was played
    _re_task_id = re.compile(r"task_(\d+)_state_s_\w+\.json$")
    task_counts: dict[int, int] = {}
    for f in HUMAN_DIR.glob("task_*_state_s_*.json"):
        m = _re_task_id.search(f.name)
        if m:
            tid = int(m.group(1))
            task_counts[tid] = task_counts.get(tid, 0) + 1

    print(f"{'Task':<10}  {'Times played':>12}")
    print("-" * 24)
    for tid in sorted(task_counts):
        print(f"Task {tid:<5}  {task_counts[tid]:>12}")
    print()

    print(f"{'Participant':<20}  {'Total time':>15}  {'Tasks':>5}  {'Turns':>5}  {'Avg/task':>12}")
    print("-" * 70)

    for pid in pids:
        r = compute_total_time(HUMAN_DIR, participant_id=pid)
        avg_per_task = format_duration(r["total_seconds"] / r["n_games"]) if r["n_games"] else "N/A"
        print(
            f"{pid:<20}  {format_duration(r['total_seconds']):>15}"
            f"  {r['n_games']:>5}  {r['n_turns']:>5}  {avg_per_task:>12}"
        )

    # Grand total across all participants
    r_all = compute_total_time(HUMAN_DIR)
    avg_all = format_duration(r_all["total_seconds"] / r_all["n_games"]) if r_all["n_games"] else "N/A"
    print("-" * 70)
    print(
        f"{'ALL HUMANS':<20}  {format_duration(r_all['total_seconds']):>15}"
        f"  {r_all['n_games']:>5}  {r_all['n_turns']:>5}  {avg_all:>12}"
    )
else:
    print(f"Human study directory not found ({HUMAN_DIR})")
