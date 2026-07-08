"""
compute_eig.py

Computes Expected Information Gain (EIG) for every player-proposed example
in gamestates_zendo_symbolic and gamestates_random_zendo_symbolic.

Definition
----------
Given a current hypothesis distribution {h_i, p_i} (PCFG-weighted programs
consistent with examples seen so far), the EIG of proposing structure x is:

    P_+(x) = Σ_{h_i : h_i(x) = True}  p_i   (normalised)
    EIG(x)  = H(Bernoulli(P_+)) = -P_+ log2 P_+ - (1-P_+) log2 (1-P_+)

High EIG (≈ 1 bit) means the hypothesis set is split 50/50 on x → maximally
informative.  Low EIG means all current hypotheses agree → little to learn.

The same metric is computed for GM-provided examples so they can serve as a
baseline.

Resumability
------------
Results are appended to the output CSV one row at a time.  On restart the
script reads already-finished (player_tag, task_idx, seed, example_idx) keys
and skips them, so no work is repeated even if the process is killed.

Usage
-----
    python analysis/compute_eig.py \\
        --output analysis/eig_results.csv \\
        --top-n  20 \\
        --max-depth 5
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import torch

DEBUG = False  # set via --debug flag


def dlog(*args, **kwargs):
    """Print only when DEBUG is enabled."""
    if DEBUG:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Difficulty — imported from the canonical source
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_by_difficulty import classify_difficulty

def extract_task_and_run_id(state_path: Path):
    """
    Returns (task_idx, run_id)

    Supports:
      NEW: task_<n>_state_s_<id>.json
      OLD: seed_<id>/task_<n>_state*.json
    """
    name = state_path.name

    # --- NEW format ---
    m = re.search(r"task_(\d+)_state_s_([^\.]+)", name)
    if m:
        return int(m.group(1)), m.group(2)

    # --- OLD format ---
    m = re.search(r"task_(\d+)_state", name)
    task_idx = int(m.group(1)) if m else -1

    # extract seed from parent directory like seed_42
    parent = state_path.parent.name
    m2 = re.search(r"(\d+)", parent)
    run_id = m2.group(1) if m2 else parent

    return task_idx, run_id

def resolve_examples_file(state_path: Path, n: int, run_id: str):
    base_dir = state_path.parent
    root_dir = base_dir.parent  # for flat layout fallback

    candidates = []

    # --- NEW flat layout (same dir as state file) ---
    candidates += [
        base_dir / f"examples_{n}_s_{run_id}_completed.pt",
        base_dir / f"examples_{n}_s_{run_id}.pt",
    ]

    # --- FLAT but maybe one level up ---
    candidates += [
        root_dir / f"examples_{n}_s_{run_id}_completed.pt",
        root_dir / f"examples_{n}_s_{run_id}.pt",
    ]

    # --- OLD layout (inside seed dir) ---
    candidates += [
        base_dir / f"examples_{n}_completed.pt",
        base_dir / f"examples_{n}.pt",
    ]

    # --- FINAL fallback (flat, no id) ---
    candidates += [
        root_dir / f"examples_{n}_completed.pt",
        root_dir / f"examples_{n}.pt",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None

# ---------------------------------------------------------------------------
# Path helpers (shared with analyze_information_gain.py)
# ---------------------------------------------------------------------------

def parse_path_label(path_str: str):
    stem = Path(path_str).stem
    for part in stem.split("_"):
        if part.lower() == "true":
            return True
        if part.lower() == "false":
            return False
    return None


def parse_path_index(path_str: str):
    try:
        return int(Path(path_str).stem.split("_")[-1])
    except (ValueError, IndexError):
        return None


def is_player_proposed(path_str: str) -> bool:
    return Path(path_str).name.lower().startswith("player_")


# ---------------------------------------------------------------------------
# Program search – returns list of (Program, pcfg_probability)
# ---------------------------------------------------------------------------

def run_search_with_probs(examples: list, dsl, cfg, top_n: int,
                          timeout: int = 60, model=None):
    """
    Run heap search and return [(program, pcfg_probability), ...].
    Tries decreasing accuracy thresholds until at least one program is found.

    If *model* (a BigramsPredictor) is provided, a learned PCFG conditioned on
    *examples* is used instead of the uniform prior, which dramatically speeds
    up finding programs for complex rules.
    """
    from experiment_helper import make_program_checker_with_accuracy, merge_grammars
    import experiments.run_experiment as _re
    from experiments.run_experiment import run_algorithm

    _re.timeout = timeout

    if not examples:
        return []

    checker = make_program_checker_with_accuracy(dsl, examples)

    if model is not None:
        try:
            import torch
            with torch.no_grad():
                raw_grammars = model([examples])
            type_request = next(iter(model.cfg_dictionary))
            learned_pcfgs = model.reconstruct_grammars(raw_grammars, [type_request])
            learned_pcfg  = learned_pcfgs[0].normalise()
            pcfg = merge_grammars(learned_pcfg, cfg)
            dlog("      [search] using learned PCFG", flush=True)
        except Exception as e:
            print(f"      [warn] learned PCFG failed ({e}), falling back to uniform")
            pcfg = cfg.CFG_to_Uniform_PCFG()
    else:
        pcfg = cfg.CFG_to_Uniform_PCFG()

    n = len(examples)
    for t in range(n):
        threshold = 1.0 - (t / n)
        dlog(f"      [search] threshold={threshold:.3f} (attempt {t+1}/{n})", flush=True)
        raw = run_algorithm(
            is_correct_program=checker,
            pcfg=pcfg,
            algo_index=0,
            accuracy=threshold,
            incorrect_rules=[],
            amount=top_n,
        )
        # Each element: (program, search_time, eval_time, nb_programs, cumul_prob, accuracy, probability)
        # Index 5 = accuracy, index 6 = pcfg probability.
        # Keep only programs with accuracy == 1.0 regardless of which threshold run found them.
        valid   = [(r[0], float(r[6])) for r in raw if r[0] is not None]
        perfect = [(r[0], float(r[6])) for r in raw if r[0] is not None and float(r[5]) >= 1.0]
        dlog(f"        raw results: {len(raw)}  |  non-null: {len(valid)}  |  perfect-acc: {len(perfect)}", flush=True)
        if valid and not perfect:
            best_acc = max(float(r[5]) for r in raw if r[0] is not None)
            dlog(f"        best accuracy in this run: {best_acc:.3f} (below 1.0, not kept)", flush=True)
        if perfect:
            dlog(f"        → returning {len(perfect)} perfect programs", flush=True)
            return perfect

    dlog("        → no perfect programs found across all thresholds", flush=True)
    return []


# ---------------------------------------------------------------------------
# EIG computation
# ---------------------------------------------------------------------------

def _bernoulli_entropy(p: float) -> float:
    """Shannon entropy of a Bernoulli(p) in bits."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def compute_eig(hypothesis_set, structure_tensor, dsl) -> dict:
    """
    Compute EIG for a single proposed structure.

    Parameters
    ----------
    hypothesis_set : list of (Program, float)  — (program, unnormalised_weight)
    structure_tensor : tensor   — the proposed structure
    dsl : DSL

    Returns
    -------
    dict with keys: eig, p_positive, n_hypotheses, total_weight
    """
    if not hypothesis_set:
        return {"eig": float("nan"), "p_positive": float("nan"),
                "n_hypotheses": 0, "total_weight": 0.0,
                "n_eval_true": 0, "n_eval_false": 0,
                "n_eval_error": 0, "n_eval_other": 0}

    total_w = sum(w for _, w in hypothesis_set)
    if total_w <= 0.0:
        # Fall back to uniform weights
        hypothesis_set = [(p, 1.0) for p, _ in hypothesis_set]
        total_w = float(len(hypothesis_set))

    positive_w = 0.0
    n_true = n_false = n_error = n_other = 0
    for prog, w in hypothesis_set:
        try:
            result = prog.eval_naive(dsl, (structure_tensor, None))
            if result is True:
                positive_w += w
                n_true += 1
            elif result is False:
                n_false += 1
            else:
                # unexpected return type (e.g. int, None, closure) — log and skip
                print(f"      [EIG warn] unexpected eval result: {type(result).__name__} = {result!r}")
                n_other += 1
        except Exception as e:
            print(f"      [EIG error] {type(e).__name__}: {e}")
            n_error += 1

    print(f"      [EIG] true={n_true} false={n_false} error={n_error} other={n_other}")

    p_pos = positive_w / total_w
    eig = _bernoulli_entropy(p_pos)

    return {
        "eig": eig,
        "p_positive": p_pos,
        "n_hypotheses": len(hypothesis_set),
        "total_weight": total_w,
        "n_eval_true":  n_true,
        "n_eval_false": n_false,
        "n_eval_error": n_error,
        "n_eval_other": n_other,
    }


# ---------------------------------------------------------------------------
# CSV checkpoint helpers
# ---------------------------------------------------------------------------

MIN_PERFECT_HYPOTHESES = 2  # skip task if fewer perfect-accuracy programs found

# Task index that must run WITHOUT the learned model
NO_MODEL_TASK = 14

FIELDNAMES = [
    "player_tag", "task_idx", "seed", "example_idx",
    "proposed_by",   # "player" or "gm"
    "label",
    "correct_program", "difficulty",
    "n_hypotheses", "total_weight", "p_positive", "eig",
    "n_eval_true", "n_eval_false", "n_eval_error", "n_eval_other",
    "skipped_task",  # 1 = sentinel row marking whole task as skipped (< MIN_PERFECT_HYPOTHESES)
]


def load_done_keys(csv_path: Path) -> tuple[set, set]:
    """
    Returns:
        done_examples : set of (player_tag, task_idx, seed, example_idx) already computed
        skipped_tasks : set of (player_tag, task_idx, seed) where the task was skipped
                        because fewer than MIN_PERFECT_HYPOTHESES programs were found
    """
    done_examples = set()
    skipped_tasks = set()
    if not csv_path.exists():
        return done_examples, skipped_tasks
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("skipped_task") == "1":
                skipped_tasks.add((row["player_tag"], row["task_idx"], row["seed"]))
            else:
                done_examples.add((row["player_tag"], row["task_idx"], row["seed"], row["example_idx"]))
    return done_examples, skipped_tasks


def _migrate_csv_if_needed(csv_path: Path):
    """If the CSV exists but has a different schema than FIELDNAMES, rewrite it."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        if list(existing_fields) == FIELDNAMES:
            return
        # Schema mismatch — rewrite with new fieldnames, filling missing cols with ""
        rows = list(reader)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in FIELDNAMES})


def append_row(csv_path: Path, row: dict):
    """Append a single row to the CSV, writing the header if the file is new."""
    _migrate_csv_if_needed(csv_path)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Tensor normalisation
# ---------------------------------------------------------------------------

def _ensure_tensor(t):
    """
    If *t* is already a tensor, return it unchanged.
    If it is a Prolog structure saved as a string repr of a list
    (e.g. "['item(0, ...)', ...]"), parse and convert via
    prolog_strings_to_tensor.
    Otherwise return None.
    """
    import ast
    if isinstance(t, torch.Tensor):
        return t
    if isinstance(t, str) and "item" in t:
        try:
            structure = ast.literal_eval(t)
        except Exception:
            return None
        from data.pieces2tensor import prolog_strings_to_tensor
        converted = prolog_strings_to_tensor([structure])
        return converted[0] if converted else None
    if isinstance(t, (list, tuple)) and t and isinstance(t[0], str) and "item" in t[0]:
        from data.pieces2tensor import prolog_strings_to_tensor
        converted = prolog_strings_to_tensor([list(t)])
        return converted[0] if converted else None
    return None


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------

def analyse_file(state_path: Path, dsl, cfg, top_n: int,
                 player_tag: str, done_examples: set, skipped_tasks: set,
                 csv_path: Path, timeout: int = 60, model=None):
    with open(state_path) as f:
        state = json.load(f)

    task_idx, run_id = extract_task_and_run_id(state_path)
    seed = run_id
    task_key = (player_tag, str(task_idx), seed)

    if task_key in skipped_tasks:
        print(f"  [skip] task {task_idx} previously marked as skipped (< {MIN_PERFECT_HYPOTHESES} perfect programs)")
        return 0

    correct_program = state.get("correct_program") or state.get("correct_Program") or ""
    difficulty = classify_difficulty(correct_program)

    # Load tensor data
    tensor_path = resolve_examples_file(state_path, task_idx, run_id)

    if tensor_path is None:
        print(f"  [skip] no examples file found for task {task_idx} (id={run_id})")
        return 0

    raw = torch.load(tensor_path, map_location="cpu", weights_only=False)
    pairs = []
    for i, item in enumerate(raw):
        if not (isinstance(item, (tuple, list)) and len(item) == 2):
            print(f"  [warn] malformed example at index {i}, skipping")
            continue

        t, l = item[0], bool(item[1])
        t = _ensure_tensor(t)

        if t is None:
            print(f"  [warn] example {i} has no valid tensor, skipping")
            continue

        pairs.append((t, l))

    paths = state.get("paths", [])
    if not paths:
        print(f"  [skip] no paths in {state_path.name}")
        return 0

    # Parse and sort chronologically by filename index
    parsed = []
    for path in paths:
        idx = parse_path_index(path)
        if idx is None or idx >= len(pairs):
            continue
        tensor, stored_label = pairs[idx]
        label = parse_path_label(path)
        if label is None:
            label = stored_label
        parsed.append((idx, tensor, label, is_player_proposed(path)))
    parsed.sort(key=lambda x: x[0])

    task_eigs = []
    n_written = 0
    for pos, (idx, tensor, label, is_player) in enumerate(parsed):
        # Only compute EIG for player-proposed examples
        if not is_player:
            continue

        key = (player_tag, str(task_idx), seed, str(idx))
        if key in done_examples:
            print(f"    [player] example {idx}: already done, skipping")
            continue

        # Need at least one prior example to have a hypothesis set
        before_examples = [
            (t, l) for _, t, l, _ in parsed[:pos] if t is not None
        ]
        if not before_examples:
            print(f"    [player] example {idx}: no prior examples, skipping")
            continue

        if tensor is None:
            print(f"    [player] example {idx}: tensor missing, skipping")
            continue

        print(f"    [player] example {idx}: running search on {len(before_examples)} prior examples …", flush=True)
        hypothesis_set = run_search_with_probs(before_examples, dsl, cfg, top_n,
                                               timeout=timeout, model=model)
        n_perfect = len(hypothesis_set)
        print(f"      → {n_perfect} perfect hypotheses found", flush=True)

        if n_perfect < MIN_PERFECT_HYPOTHESES:
            print(f"      → only {n_perfect} perfect programs (< {MIN_PERFECT_HYPOTHESES}): "
                  f"writing skip marker and abandoning task", flush=True)
            sentinel = {
                "player_tag":      player_tag,
                "task_idx":        task_idx,
                "seed":            seed,
                "example_idx":     idx,
                "proposed_by":     "player",
                "label":           label,
                "correct_program": correct_program,
                "difficulty":      difficulty,
                "n_hypotheses":    n_perfect,
                "total_weight":    "",
                "p_positive":      "",
                "eig":             "",
                "n_eval_true":     "",
                "n_eval_false":    "",
                "n_eval_error":    "",
                "n_eval_other":    "",
                "skipped_task":    1,
            }
            append_row(csv_path, sentinel)
            skipped_tasks.add(task_key)
            break

        eig_info = compute_eig(hypothesis_set, tensor, dsl)
        eig_val = eig_info["eig"]

        row = {
            "player_tag":      player_tag,
            "task_idx":        task_idx,
            "seed":            seed,
            "example_idx":     idx,
            "proposed_by":     "player",
            "label":           label,
            "correct_program": correct_program,
            "difficulty":      difficulty,
            "n_hypotheses":    eig_info["n_hypotheses"],
            "total_weight":    round(eig_info["total_weight"], 8),
            "p_positive":      round(eig_info["p_positive"], 6) if not math.isnan(eig_info["p_positive"]) else "",
            "eig":             round(eig_val, 6) if not math.isnan(eig_val) else "",
            "n_eval_true":     eig_info["n_eval_true"],
            "n_eval_false":    eig_info["n_eval_false"],
            "n_eval_error":    eig_info["n_eval_error"],
            "n_eval_other":    eig_info["n_eval_other"],
            "skipped_task":    0,
        }
        append_row(csv_path, row)
        done_examples.add(key)
        n_written += 1
        if not math.isnan(eig_val):
            task_eigs.append(eig_val)

    # Task-level summary
    n_player = len(task_eigs)
    if n_player > 0:
        avg_eig = sum(task_eigs) / n_player
        eig_strs = ", ".join(f"{v:.4f}" for v in task_eigs)
        print(f"  [task {task_idx}] player examples: {n_player}  |  EIG per example: [{eig_strs}]  |  avg EIG: {avg_eig:.4f}")
    else:
        print(f"  [task {task_idx}] no scoreable player examples")

    return n_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute EIG for zendo_symbolic game data")
    parser.add_argument("--output",    default="analysis/eig_results.csv")
    parser.add_argument("--top-n",     type=int, default=20,
                        help="Programs to collect per search run (default: 20)")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--tasks",     type=int, nargs="+", default=None,
                        help="Only process these task indices (e.g. --tasks 0 1 2). Default: all tasks.")
    parser.add_argument("--seeds",          type=str, nargs="+", default=None,
                        help="Only process these seed directories. Default: all seeds.")
    parser.add_argument("--timeout",        type=int, default=60,
                        help="Per-search timeout in seconds (default: 60).")
    parser.add_argument("--debug",          action="store_true",
                        help="Print verbose search diagnostics.")
    args = parser.parse_args()

    global DEBUG
    DEBUG = args.debug

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from grammar import dsl as dsl_module
    from type_system import BOOL, Arrow
    from type_system import List as TypeList

    # DSL used by the learned model (all tasks except NO_MODEL_TASK)
    from DSL import zendo as zendo_base
    dsl_model = dsl_module.DSL(zendo_base.semantics, zendo_base.primitive_types, None)
    type_request_model = Arrow(TypeList(zendo_base.PIECE), BOOL)
    cfg_model = dsl_model.DSL_to_CFG(type_request_model, max_program_depth=args.max_depth)

    # DSL used for task NO_MODEL_TASK (uniform prior, extended primitives)
    from DSL import zendo_extended as zendo_ext
    dsl_ext = dsl_module.DSL(zendo_ext.semantics, zendo_ext.primitive_types, None)
    type_request_ext = Arrow(TypeList(zendo_ext.PIECE), BOOL)
    cfg_ext = dsl_ext.DSL_to_CFG(type_request_ext, max_program_depth=args.max_depth)

    from model_loader import __build_generic_zendo_model
    _, learned_model = __build_generic_zendo_model(
        dsl=dsl_model,
        max_program_depth=args.max_depth,
        size_max=11, size_hidden=64,
        embedding_output_dimension=78,
        number_layers_RNN=1,
        autoload=True,
        name="model_weights/bigramsPredictor_variable.weights",
    )
    learned_model.eval()
    print(f"Loaded BigramsPredictor (zendo DSL). Task {NO_MODEL_TASK} uses zendo_extended + uniform.\n")

    dirs = [
        ("zendo_symbolic",     Path("gamestates/gamestates_zendo_symbolic")),
        ("vlp_symbolic",       Path("gamestates/gamestates_vlp_symbolic")),
        ("fullgpt_symbolic",   Path("gamestates/gamestates_fullgpt_symbolic")),
        ("scientist_symbolic", Path("gamestates/gamestates_scientist_symbolic")),
        ("random_zendo_symbolic", Path("gamestates/gamestates_random_zendo_symbolic")),
    ]

    csv_path = Path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already-finished keys for resumability
    done_examples, skipped_tasks = load_done_keys(csv_path)
    print(f"Resuming: {len(done_examples)} examples already computed, "
          f"{len(skipped_tasks)} tasks previously skipped.\n")

    total_written = 0
    for player_tag, base_dir in dirs:
        if not base_dir.exists():
            print(f"[warn] directory not found: {base_dir}")
            continue

        state_files = sorted(base_dir.rglob("task_*_state*.json"))
        if args.seeds is not None:
            state_files = [f for f in state_files if f.parent.name in args.seeds]
        if args.tasks is not None:
            state_files = [f for f in state_files
                           if any(f"task_{t}_state" in f.name for t in args.tasks)]
        print(f"\n[{player_tag}]  {len(state_files)} state files in {base_dir}")

        for state_path in state_files:
            m = re.search(r"task_(\d+)_state", state_path.name)
            task_idx = int(m.group(1)) if m else -1
            if task_idx == NO_MODEL_TASK:
                dsl, cfg, model_for_task = dsl_ext, cfg_ext, None
                mode_str = "zendo_extended + uniform"
            else:
                dsl, cfg, model_for_task = dsl_model, cfg_model, learned_model
                mode_str = "zendo + model"

            print(f"  {state_path.relative_to(base_dir)} [{mode_str}] …", flush=True)
            n = analyse_file(state_path, dsl, cfg, args.top_n,
                             player_tag, done_examples, skipped_tasks, csv_path,
                             timeout=args.timeout,
                             model=model_for_task)
            print(f"    → {n} new rows written")
            total_written += n

    print(f"\nDone. Total new rows written: {total_written}")
    print(f"Full results at: {csv_path}")

    # Summary (player examples only) — saved to files for figure generation
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, on_bad_lines="skip")
        df = df[df["proposed_by"] == "player"].copy()
        df["eig"] = pd.to_numeric(df["eig"], errors="coerce")

        out_dir = csv_path.parent

        # Per-task summary: one row per (player_tag, task_idx, seed)
        per_task = (
            df.groupby(["player_tag", "task_idx", "seed", "difficulty"])["eig"]
            .agg(n_examples="count", avg_eig="mean")
            .reset_index()
        )
        per_task_path = out_dir / "eig_per_task.csv"
        per_task.to_csv(per_task_path, index=False)
        print(f"\nSaved per-task summary → {per_task_path}")
        print(per_task.to_string(index=False))

        # Overall summary: one row per player_tag
        overall = (
            df.groupby("player_tag")["eig"]
            .agg(n_examples="count", avg_eig="mean")
            .reset_index()
        )
        overall_path = out_dir / "eig_overall.csv"
        overall.to_csv(overall_path, index=False)
        print(f"\nSaved overall summary  → {overall_path}")
        print(overall.to_string(index=False))

    except Exception as e:
        print(f"(Summary failed: {e})")


if __name__ == "__main__":
    main()
