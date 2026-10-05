import zendoworld.serialization as legacy_pickle
"""
compute_eig.py

Computes Expected Information Gain (EIG) for every player-proposed example in
a run directory of Zendo gamestates.

Definition
----------
For each player-proposed example at position k, a fresh program search is run
over *exactly the examples the player had seen before proposing it* (positions
0..k-1, both GM- and player-provided).  That search returns the batch of
accuracy-1.0 programs for that example index, each with its PCFG probability.
The EIG of the proposed structure x is then measured against that batch:

    P_+(x) = Σ_{h_i : h_i(x) = True}  p_i   (normalised)
    EIG(x)  = H(Bernoulli(P_+)) = -P_+ log2 P_+ - (1-P_+) log2 (1-P_+)

High EIG (≈ 1 bit) means the hypothesis set is split 50/50 on x → maximally
informative.  Low EIG means all current hypotheses agree → little to learn.
``n_eval_true`` / ``n_eval_false`` record the unweighted split of that batch;
``p_positive`` is the PCFG-weighted version that ``eig`` is derived from.

Only player-proposed examples are scored.  There is no GM baseline row.
The JSON `examples` count bounds the saved tensor prefix: extra trailing
examples from subsequently corrected games are ignored.

Provenance
----------
Which example belongs to the player and which to the game master is
reconstructed from ``turn_descriptions``, replaying the append order of
``src/zendoworld/game/states.py::step``.  See ``reconstruct_provenance``.  The ``gm_*`` /
``player_*`` image filenames are deliberately NOT used: they come from
``run_experiment.classify_label``, a heuristic on the source image path that
mislabels GM counterexamples rendered under ``outputs/rendering/`` and tags
every example ``gm`` in runs whose scenes are not rendered there (all symbolic
runs, where ``paths`` keeps raw dataset paths and empty strings).

Resumability
------------
Results are appended to the output CSV one row at a time.  On restart the
script reads already-finished (player_tag, task_idx, seed, example_idx) keys
and skips them, so no work is repeated even if the process is killed.

Usage
-----
    python -m zendoworld.analysis.compute_eig \\
        --dirs   zendo_symbolic=outputs/gamestates/gamestates_zendo_symbolic \\
        --output analysis/eig_results.csv \\
        --top-n  20 \\
        --max-depth 5
"""

import argparse
import concurrent.futures
import csv
import json
import math
import multiprocessing
import os
import re
import sys
import tempfile
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

from zendoworld.analysis.evaluate_by_difficulty import classify_difficulty

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
# Provenance: who proposed each example, reconstructed from turn_descriptions
# ---------------------------------------------------------------------------

# Descriptions emitted by the Turn.LABEL step in src/zendoworld/game/states.py.  Each LABEL
# step appends exactly one example: the scene the player proposed.
_LABEL_MARKERS = ("Quiz mode correct", "Quiz mode incorrect", "Tell mode")


def _turn_number(desc: str):
    match = re.match(r"Turn (\d+)", desc)
    return int(match.group(1)) if match else None


def reconstruct_provenance(state: dict):
    """Return ["gm" | "player", ...], one entry per example, in append order.

    Replays every ``state.examples.append`` in ``src/zendoworld/game/states.py::step``:

    * ``game.play_game_state`` seeds the game with ``gm.initial_examples()``,
      which returns exactly one positive and one negative GM example.
    * Turn.PROPOSE with no input logs "proposed no input." and appends one GM
      example from ``gm.get_next_example()``.
    * Turn.LABEL appends the player's proposed scene, then logs the quiz/tell
      description.  In quiz mode one description is logged *per player*, so
      only the first LABEL description of a turn counts as an example.
    * Turn.GUESS logs "Guessing rule: ..." per guess and appends the GM's
      counterexample from ``gm.disprove_guess()``.  "Not guessing rule"
      appends nothing.
    * A won game ends ON its final guess -- either the guess was correct, or no
      counterexample could be found -- so that last guess appends nothing.

    The "proposed input based on rule:" description is deliberately ignored: it
    is only written when the player reported a non-empty rule, so players that
    propose with an empty rule string (e.g. the random baseline) log nothing at
    PROPOSE even though the example is still appended at LABEL.

    The caller must check the returned length against the real example count;
    a mismatch means the log cannot be trusted for that task.
    """
    provenance = ["gm", "gm"]          # gm.initial_examples(): 1 pos + 1 neg
    counterexample_positions = []
    labelled_turns = set()

    descriptions = state.get("turn_descriptions", []) or []
    for log_index, desc in enumerate(descriptions):
        if "proposed no input" in desc:
            exhausted = "examples exhausted" in state.get("game_over_reason", "").lower()
            if not (exhausted and log_index == len(descriptions) - 1):
                provenance.append("gm")
        elif any(marker in desc for marker in _LABEL_MARKERS):
            turn = _turn_number(desc)
            if turn is None or turn not in labelled_turns:
                labelled_turns.add(turn)
                provenance.append("player")
        elif "Not guessing rule" in desc:
            continue
        elif "Guessed invalid rule" in desc:
            # Legacy description, no longer emitted; the guess was rejected
            # before a counterexample could be produced.
            continue
        elif ": Guessing rule" in desc:
            counterexample_positions.append(len(provenance))
            provenance.append("gm")

    if state.get("won") and counterexample_positions:
        del provenance[counterexample_positions[-1]]

    return provenance


# ---------------------------------------------------------------------------
# Program search – returns list of (Program, pcfg_probability)
# ---------------------------------------------------------------------------

def run_search_with_probs(examples: list, dsl, cfg, top_n: int,
                          timeout: int = 60, model=None):
    """
    Run heap search and return [(program, pcfg_probability), ...], keeping only
    programs that are accuracy-1.0 on *examples*.

    If *model* (a BigramsPredictor) is provided, a learned PCFG conditioned on
    *examples* is used instead of the uniform prior, which dramatically speeds
    up finding programs for complex rules.
    """
    from zendoworld.experiments.helpers import make_program_checker_with_accuracy, merge_grammars
    import zendoworld.synthesis.search_runner as _re
    from zendoworld.synthesis.search_runner import run_algorithm

    _re.timeout = timeout

    # The cached evaluator must stay off here.  It memoises results on the
    # Program instance keyed by example index (src/zendoworld/synthesis/program.py::Variable.eval), and
    # the Variable instances belong to the CFG we build once and reuse for
    # every task (src/zendoworld/synthesis/grammar/dsl.py::DSL_to_CFG).  Index 3 therefore keeps
    # returning the input tensor of whichever task this process searched first,
    # so every later search scores candidates against the wrong scenes and
    # reports them as accuracy 1.0.  Within one task the prefixes are nested so
    # the staleness is invisible; across tasks it silently corrupts every rule
    # batch.  eval_naive takes no cache, and is what compute_eig() uses too, so
    # the accuracy filter and the EIG evaluation now agree.
    _re.use_heap_search_cached_eval = False

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

    # A single pass at accuracy 1.0 is all that can help.  run_algorithm's
    # `accuracy` is only an early-stop test: it always retains the best `amount`
    # candidates by accuracy, so lowering the threshold can only make it stop
    # sooner and return a subset of what this pass finds.  It can never surface
    # a perfect program that this pass missed, and the retry loop it replaces
    # burned one full timeout per prior example to prove that.
    raw = run_algorithm(
        is_correct_program=checker,
        pcfg=pcfg,
        algo_index=0,
        accuracy=1.0,
        incorrect_rules=[],
        amount=top_n,
        timeout=timeout,
    )
    # Each element: (program, search_time, eval_time, nb_programs, cumul_prob, accuracy, probability)
    # Index 5 = accuracy, index 6 = pcfg probability.
    valid   = [r for r in raw if r[0] is not None]
    perfect = [(r[0], float(r[6])) for r in valid if float(r[5]) >= 1.0]
    dlog(f"        raw results: {len(raw)}  |  non-null: {len(valid)}  |  perfect-acc: {len(perfect)}", flush=True)
    if valid and not perfect:
        best_acc = max(float(r[5]) for r in valid)
        dlog(f"        best accuracy: {best_acc:.3f} (below 1.0, not kept)", flush=True)
    return perfect


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

    if any(not math.isfinite(w) or w < 0 for _, w in hypothesis_set):
        raise ValueError("Hypothesis weights must be finite and nonnegative")
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
            if isinstance(result, torch.Tensor) and result.numel() == 1 and result.dtype == torch.bool:
                result = result.item()
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

    # An unevaluable rule has unknown prediction, not a negative prediction.
    # Keep diagnostics but do not report a partial-batch entropy as valid EIG.
    p_pos = float("nan") if n_error or n_other else positive_w / total_w
    eig = _bernoulli_entropy(p_pos) if math.isfinite(p_pos) else float("nan")

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

COMPUTATION_VERSION = "prefix_search_v4_json_cutoff"

# Task index that must run WITHOUT the learned model
NO_MODEL_TASK = 14

FIELDNAMES = [
    "player_tag", "task_idx", "seed", "example_idx",
    "proposed_by",   # "player" or "gm"
    "label",
    "correct_program", "difficulty",
    "n_hypotheses", "total_weight", "p_positive", "eig",
    "n_eval_true", "n_eval_false", "n_eval_error", "n_eval_other",
    "skipped_task",  # 1 = sentinel row marking the whole task as skipped
    "skip_reason",   # diagnostic for an unavailable example or game
    "status", "computation_version",
]


def _skip_sentinel(player_tag, task_idx, seed, correct_program, difficulty,
                   reason, example_idx="", label="", n_hypotheses=""):
    """Build an unavailable-data row; callers can mark it per-example.

    Game-level sentinels stop retries when chronology cannot be established.
    Per-example failures use skipped_task=0 and a specific status instead.
    """
    return {
        "player_tag":      player_tag,
        "task_idx":        task_idx,
        "seed":            seed,
        "example_idx":     example_idx,
        "proposed_by":     "player",
        "label":           label,
        "correct_program": correct_program,
        "difficulty":      difficulty,
        "n_hypotheses":    n_hypotheses,
        "total_weight":    "",
        "p_positive":      "",
        "eig":             "",
        "n_eval_true":     "",
        "n_eval_false":    "",
        "n_eval_error":    "",
        "n_eval_other":    "",
        "skipped_task":    1,
        "skip_reason":     reason,
        "status":          "data_error",
        "computation_version": COMPUTATION_VERSION,
    }


def load_done_keys(csv_path: Path) -> tuple[set, set]:
    """
    Returns:
        done_examples : set of (player_tag, task_idx, seed, example_idx) already computed
        skipped_tasks : games stopped for unsafe chronology or empty search.
        Versioned results cannot be mixed with older computations.
    """
    done_examples = set()
    skipped_tasks = set()
    if not csv_path.exists():
        return done_examples, skipped_tasks
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("computation_version") != COMPUTATION_VERSION:
                raise ValueError(f"{csv_path} contains results from an older EIG computation. "
                                 "Use a new output filename to avoid mixing incompatible results.")
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
    row = dict(row, computation_version=COMPUTATION_VERSION)
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
        from zendoworld.datasets.pieces2tensor import prolog_strings_to_tensor
        converted = prolog_strings_to_tensor([structure])
        return converted[0] if converted else None
    if isinstance(t, (list, tuple)) and t and isinstance(t[0], str) and "item" in t[0]:
        from zendoworld.datasets.pieces2tensor import prolog_strings_to_tensor
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
        print(f"  [skip] task {task_idx} previously marked as skipped")
        return 0

    correct_program = state.get("correct_program") or state.get("correct_Program") or ""
    difficulty = classify_difficulty(correct_program)

    # Load tensor data
    tensor_path = resolve_examples_file(state_path, task_idx, run_id)

    if tensor_path is None:
        print(f"  [skip] no examples file found for task {task_idx} (id={run_id})")
        return 0

    raw = torch.load(tensor_path, map_location="cpu", weights_only=False, pickle_module=legacy_pickle)
    # Corrected games can end earlier than the originally saved tensor file.
    # The JSON count is authoritative; only trailing examples may be removed.
    example_count = state.get("examples", len(raw))
    if (isinstance(example_count, bool) or not isinstance(example_count, int)
            or example_count < 0 or example_count > len(raw)):
        reason = (f"invalid/unavailable JSON example count {example_count!r}; "
                  f"tensor file has {len(raw)} examples")
        print(f"  [skip] task {task_idx}: {reason}")
        append_row(csv_path, _skip_sentinel(
            player_tag, task_idx, seed, correct_program, difficulty, reason))
        skipped_tasks.add(task_key)
        return 0
    if len(raw) > example_count:
        print(f"  [trim] task {task_idx}: ignoring {len(raw) - example_count} "
              f"trailing tensors beyond JSON example count {example_count}")
        raw = raw[:example_count]
    # Keep this list positionally aligned with `raw`: position == append order
    # == the index provenance and example_idx refer to.  Unusable entries
    # become None placeholders rather than being dropped, because compacting
    # the list would shift every later example onto the wrong index.
    pairs = [None] * len(raw)
    for i, item in enumerate(raw):
        if not (isinstance(item, (tuple, list)) and len(item) == 2):
            print(f"  [warn] malformed example at index {i}, leaving a gap")
            continue

        t, l = item[0], bool(item[1])
        t = _ensure_tensor(t)

        if t is None:
            print(f"  [warn] example {i} has no valid tensor, leaving a gap")
            continue

        pairs[i] = (t, l)

    # Who proposed each example, from the turn log rather than the image
    # filenames.  The reconstruction is only trusted when it accounts for
    # exactly as many examples as the retained JSON-bounded prefix; otherwise
    # the log and the examples disagree, and guessing would silently score the
    # wrong scenes.
    provenance = reconstruct_provenance(state)
    if len(provenance) != len(pairs):
        reason = (f"provenance mismatch: turn log reconstructs "
                  f"{len(provenance)} examples, retained tensor prefix has {len(pairs)}")
        print(f"  [skip] task {task_idx}: {reason}")
        append_row(csv_path, _skip_sentinel(
            player_tag, task_idx, seed, correct_program, difficulty, reason))
        skipped_tasks.add(task_key)
        return 0

    # Chronological: position in the list IS the append order.  Labels come
    # from the tensor file, which is authoritative.
    parsed = [
        (idx, pair[0] if pair else None, pair[1] if pair else None, who == "player")
        for idx, (pair, who) in enumerate(zip(pairs, provenance))
    ]
    n_player_total = sum(1 for _, _, _, is_player in parsed if is_player)
    print(f"    provenance: {len(parsed)} examples, {n_player_total} player-proposed")

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

        def unavailable(status, reason):
            row = _skip_sentinel(player_tag, task_idx, seed, correct_program,
                                 difficulty, reason, example_idx=idx, label=label,
                                 n_hypotheses=0)
            row.update(skipped_task=0, status=status)
            append_row(csv_path, row)
            done_examples.add(key)

        if tensor is None:
            unavailable("missing_current_example", "Current example has no usable tensor")
            n_written += 1
            continue
        if any(t is None for _, t, _, _ in parsed[:pos]):
            unavailable("incomplete_history", "A preceding example has no usable tensor")
            n_written += 1
            continue

        # Everything the player had seen before proposing this example,
        # GM- and player-provided alike.
        before_examples = [
            (t, l) for _, t, l, _ in parsed[:pos]
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

        # Stop on the first empty batch as an explicit computation policy.
        # A bounded search failure does not prove the version space is empty.
        if not hypothesis_set:
            reason = ("Search returned zero consistent rules within its budget; "
                      "remaining indices stopped by policy, not scored as zero EIG")
            row = _skip_sentinel(player_tag, task_idx, seed, correct_program,
                                 difficulty, reason, example_idx=idx, label=label,
                                 n_hypotheses=0)
            row["status"] = "no_hypotheses"
            append_row(csv_path, row)
            skipped_tasks.add(task_key)
            print(f"      {reason}; retaining {n_written} earlier rows", flush=True)
            break

        # One consistent rule is a valid degenerate distribution (EIG = 0).
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
            "skip_reason":     ("Search returned no consistent rules" if not hypothesis_set else
                                "Some rules could not be evaluated" if eig_info["n_eval_error"] or eig_info["n_eval_other"] else ""),
            "status":          ("no_hypotheses" if not hypothesis_set else
                                "evaluation_error" if eig_info["n_eval_error"] or eig_info["n_eval_other"] else "ok"),
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
# Parallel processing
# ---------------------------------------------------------------------------

_WORKER_RESOURCES = None


def _init_eig_worker():
    """Prevent each worker from starting a full CPU thread team of its own."""
    torch.set_num_threads(1)


def _parallel_analyse_file(job):
    """Analyse one task/seed file and write only to this process's shard."""
    (state_path, player_tag, output_index, shard_dir, top_n, timeout,
     done_examples, skipped_tasks) = job
    dsl_model, cfg_model, learned_model, dsl_ext, cfg_ext = _WORKER_RESOURCES
    state_path = Path(state_path)
    task_idx, _ = extract_task_and_run_id(state_path)
    if task_idx == NO_MODEL_TASK:
        dsl, cfg, model_for_task = dsl_ext, cfg_ext, None
    else:
        dsl, cfg, model_for_task = dsl_model, cfg_model, learned_model

    shard_path = Path(shard_dir) / f"output_{output_index}_worker_{os.getpid()}.csv"
    n_written = analyse_file(
        state_path, dsl, cfg, top_n, player_tag,
        set(done_examples), set(skipped_tasks), shard_path,
        timeout=timeout, model=model_for_task,
    )
    return n_written, str(shard_path)


def _numeric_sort_part(value):
    text = str(value)
    match = re.search(r"-?\d+", text)
    return (0, int(match.group())) if match else (1, text)


def merge_csv_shards(csv_path: Path, shard_paths):
    """Merge worker shards with existing results, deduplicate, and sort."""
    _migrate_csv_if_needed(csv_path)
    rows = []
    for path in [csv_path, *sorted(map(Path, shard_paths))]:
        if not path.exists() or path.stat().st_size == 0:
            continue
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))

    unique = {}
    for row in rows:
        key = (
            row.get("player_tag", ""), row.get("task_idx", ""),
            row.get("seed", ""), row.get("example_idx", ""),
            row.get("skipped_task", ""),
        )
        unique[key] = {field: row.get(field, "") for field in FIELDNAMES}

    ordered = sorted(
        unique.values(),
        key=lambda row: (
            _numeric_sort_part(row["seed"]),
            _numeric_sort_part(row["task_idx"]),
            row["player_tag"],
            _numeric_sort_part(row["example_idx"]),
        ),
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(ordered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, csv_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _task_seed_grid(value):
    """Parse ``--expect-grid 28x5`` into (n_tasks, n_seeds)."""
    match = re.fullmatch(r"(\d+)\s*x\s*(\d+)", value.strip().lower())
    if not match:
        raise argparse.ArgumentTypeError(
            f"--expect-grid must look like 28x5 (tasks x seeds), got {value!r}")
    return int(match.group(1)), int(match.group(2))


def check_grid(state_files, want_tasks, want_seeds):
    """Return a list of complaints if the discovered files are not a full grid.

    Only the run *count* per task matters, not which seed directories supplied
    the runs: interchangeable seeds (scientist_qwen ran tasks 22-27 under
    seed_5 rather than seed_73) still form a complete grid.
    """
    coverage = {}
    for state_path in state_files:
        task_idx, _ = extract_task_and_run_id(state_path)
        coverage.setdefault(task_idx, set()).add(state_path.parent.name)

    problems = []
    if len(coverage) != want_tasks:
        missing_tasks = [t for t in range(want_tasks) if t not in coverage]
        extra_tasks = sorted(t for t in coverage if t >= want_tasks or t < 0)
        detail = f"expected {want_tasks} tasks, found {len(coverage)}"
        if missing_tasks:
            detail += f"; missing task indices {missing_tasks}"
        if extra_tasks:
            detail += f"; unexpected task indices {extra_tasks}"
        problems.append(detail)
    short = {task: sorted(seeds) for task, seeds in sorted(coverage.items())
             if len(seeds) != want_seeds}
    if short:
        shown = ", ".join(f"task {task} has {len(seeds)} ({', '.join(seeds)})"
                          for task, seeds in list(short.items())[:6])
        suffix = f" (+{len(short) - 6} more)" if len(short) > 6 else ""
        problems.append(f"{len(short)} tasks do not have exactly {want_seeds} "
                        f"runs: {shown}{suffix}")
    expected_total = want_tasks * want_seeds
    if len(state_files) != expected_total:
        problems.append(f"expected {expected_total} state files, found {len(state_files)}")
    return problems


def main():
    parser = argparse.ArgumentParser(description="Compute EIG for zendo_symbolic game data")
    parser.add_argument("--output", nargs="+", required=True,
                        help="One output CSV per --dirs entry, same order and count, e.g. "
                             "--dirs fullgpt=... scientist=... --output eig_fullgpt.csv eig_scientist.csv. "
                             "Each gets its own resumability state and per-task/overall summary files, "
                             "so separate --dirs entries never share (or race on) an output file.")
    parser.add_argument("--top-n",     type=int, default=20,
                        help="Programs to collect per search run (default: 20)")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--tasks",     type=int, nargs="+", default=None,
                        help="Only process these task indices (e.g. --tasks 0 1 2). Default: all tasks.")
    parser.add_argument("--seeds",          type=str, nargs="+", default=None,
                        help="Only process these seed directories. Default: all seeds.")
    parser.add_argument("--timeout",        type=int, default=60,
                        help="Per-search timeout in seconds (default: 60).")
    parser.add_argument("--workers",        type=int, default=1,
                        help="Parallel task/seed worker processes (default: 1). "
                             "Each worker writes an isolated CSV shard; shards are "
                             "merged and sorted after processing.")
    parser.add_argument("--debug",          action="store_true",
                        help="Print verbose search diagnostics.")
    parser.add_argument("--expect-grid", type=_task_seed_grid, default=None,
                        help="Fail fast unless every --dirs entry holds a complete "
                             "TASKSxSEEDS grid of state files, e.g. --expect-grid 28x5. "
                             "Reports the missing task/seed pairs instead of silently "
                             "computing EIG over a partial run.")
    parser.add_argument("--dirs", nargs="+", required=True,
                        help="One or more player_tag=path pairs to process, e.g. "
                             "--dirs fullgpt_qwen_ablation=gamestates_qwen_ablation/gamestates_fullgpt "
                             "scientist_qwen_ablation=gamestates_qwen_ablation/gamestates_scientist. "
                             "Each entry gets its own --output file (matched by position) -- if you "
                             "instead run separate invocations simultaneously, still give each its "
                             "own --output, since two processes sharing one file would race on it.")
    args = parser.parse_args()

    global DEBUG, _WORKER_RESOURCES
    DEBUG = args.debug
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    from zendoworld.synthesis.grammar import dsl as dsl_module
    from zendoworld.synthesis.type_system import BOOL, Arrow
    from zendoworld.synthesis.type_system import List as TypeList

    # DSL used by the learned model (all tasks except NO_MODEL_TASK)
    from zendoworld.synthesis.dsl import zendo as zendo_base
    dsl_model = dsl_module.DSL(
        zendo_base.semantics, zendo_base.primitive_types, None, constants_range=3
    )
    type_request_model = Arrow(TypeList(zendo_base.PIECE), BOOL)
    cfg_model = dsl_model.DSL_to_CFG(type_request_model, max_program_depth=args.max_depth)

    # DSL used for task NO_MODEL_TASK (uniform prior, extended primitives)
    from zendoworld.synthesis.dsl import zendo_extended as zendo_ext
    dsl_ext = dsl_module.DSL(
        zendo_ext.semantics, zendo_ext.primitive_types, None, constants_range=3
    )
    type_request_ext = Arrow(TypeList(zendo_ext.PIECE), BOOL)
    cfg_ext = dsl_ext.DSL_to_CFG(type_request_ext, max_program_depth=args.max_depth)

    from zendoworld.synthesis.predictions.loader import __build_generic_zendo_model
    _, learned_model = __build_generic_zendo_model(
        dsl=dsl_model,
        max_program_depth=args.max_depth,
        size_max=11, size_hidden=64,
        embedding_output_dimension=78,
        number_layers_RNN=1,
        autoload=True,
        name="weights/synthesis/bigramsPredictor_variable.weights",
    )
    learned_model.eval()
    _WORKER_RESOURCES = (dsl_model, cfg_model, learned_model, dsl_ext, cfg_ext)
    print(f"Loaded BigramsPredictor (zendo DSL). Task {NO_MODEL_TASK} uses zendo_extended + uniform.\n")

    if len(args.output) != len(args.dirs):
        parser.error(f"--output must have exactly one entry per --dirs entry: "
                     f"got {len(args.dirs)} --dirs but {len(args.output)} --output")

    dirs = []
    for entry, output_str in zip(args.dirs, args.output):
        tag, sep, path_str = entry.partition("=")
        if not sep or not path_str:
            parser.error(f"--dirs entries must be TAG=PATH, got: {entry!r}")
        dirs.append((tag, Path(path_str), Path(output_str)))

    grand_total_written = 0
    for player_tag, base_dir, csv_path in dirs:
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Load already-finished keys for resumability (per output file)
        done_examples, skipped_tasks = load_done_keys(csv_path)
        print(f"\n[{player_tag}] Resuming: {len(done_examples)} examples already computed, "
              f"{len(skipped_tasks)} tasks previously skipped.")

        if not base_dir.exists():
            print(f"[warn] directory not found: {base_dir}")
            continue

        # Ignore backups/variants such as ``task_14_state_overlap.json``.
        # They are not independent runs and may resolve to the canonical
        # tensor file, causing the same task to be analysed twice.
        state_files = sorted(
            f for f in base_dir.rglob("task_*_state.json")
            if re.fullmatch(r"task_\d+_state\.json", f.name)
        )
        if args.seeds is not None:
            state_files = [f for f in state_files if f.parent.name in args.seeds]
        if args.tasks is not None:
            requested_tasks = set(args.tasks)
            state_files = [
                f for f in state_files
                if extract_task_and_run_id(f)[0] in requested_tasks
            ]
        seeds_seen = sorted({f.parent.name for f in state_files})
        tasks_seen = {extract_task_and_run_id(f)[0] for f in state_files}
        print(f"[{player_tag}]  {len(state_files)} state files in {base_dir} "
              f"({len(tasks_seen)} tasks x {len(seeds_seen)} seeds: {', '.join(seeds_seen)})")
        if args.expect_grid is not None:
            problems = check_grid(state_files, *args.expect_grid)
            if problems:
                parser.error(f"[{player_tag}] {base_dir} is not a complete "
                             f"{args.expect_grid[0]}x{args.expect_grid[1]} grid:\n  - "
                             + "\n  - ".join(problems))
            print(f"[{player_tag}]  grid check passed: "
                  f"{args.expect_grid[0]} tasks x {args.expect_grid[1]} seeds, no gaps")

        total_written = 0
        if args.workers == 1 or not state_files:
            for state_path in state_files:
                task_idx, _ = extract_task_and_run_id(state_path)
                if task_idx == NO_MODEL_TASK:
                    dsl, cfg, model_for_task = dsl_ext, cfg_ext, None
                    mode_str = "zendo_extended + uniform"
                else:
                    dsl, cfg, model_for_task = dsl_model, cfg_model, learned_model
                    mode_str = "zendo + model"

                print(f"  {state_path.relative_to(base_dir)} [{mode_str}] …", flush=True)
                n = analyse_file(
                    state_path, dsl, cfg, args.top_n, player_tag,
                    done_examples, skipped_tasks, csv_path,
                    timeout=args.timeout, model=model_for_task,
                )
                print(f"    → {n} new rows written")
                total_written += n
        else:
            worker_count = min(args.workers, len(state_files))
            print(f"[{player_tag}] Processing task/seed files with {worker_count} workers")
            with tempfile.TemporaryDirectory(prefix="compute-eig-shards-") as shard_dir:
                jobs = [
                    (
                        str(state_path), player_tag, 0, shard_dir,
                        args.top_n, args.timeout,
                        frozenset(done_examples), frozenset(skipped_tasks),
                    )
                    for state_path in state_files
                ]
                shard_paths = set()
                failures = []
                # fork shares the read-only DSL, grammars, and predictor loaded
                # above. Workers perform inference only and write separate files.
                context = multiprocessing.get_context("fork")
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=context,
                    initializer=_init_eig_worker,
                ) as executor:
                    futures = {
                        executor.submit(_parallel_analyse_file, job): job[0]
                        for job in jobs
                    }
                    for future in concurrent.futures.as_completed(futures):
                        state_path = futures[future]
                        try:
                            n, shard_path = future.result()
                            total_written += n
                            shard_paths.add(Path(shard_path))
                            print(f"  [done] {state_path}: {n} new rows", flush=True)
                        except Exception as exc:
                            failures.append((state_path, exc))
                            print(f"  [error] {state_path}: {exc}", flush=True)

                # Discover shards directly as well, so rows written before a
                # worker failure are not lost merely because its future raised.
                shard_paths.update(Path(shard_dir).glob("output_0_worker_*.csv"))
                merge_csv_shards(csv_path, shard_paths)
                if failures:
                    details = "; ".join(f"{path}: {exc}" for path, exc in failures)
                    raise RuntimeError(
                        f"{len(failures)} EIG worker job(s) failed; successful "
                        f"shards were merged. Re-run to resume. {details}"
                    )

        grand_total_written += total_written
        print(f"[{player_tag}] Done. New rows written: {total_written}")
        print(f"[{player_tag}] Full results at: {csv_path}")

        # Summary (player examples only) — saved to files for figure generation.
        # Named after this csv_path so multiple --dirs/--output pairs writing
        # into the same directory don't overwrite each other's summaries.
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, on_bad_lines="skip")
            df = df[df["proposed_by"] == "player"].copy()

            # Drop the sentinel rows themselves -- they carry no EIG -- but keep
            # every example that WAS scored, including those of a task that
            # stopped early.  A task whose search dried up at example k still
            # has valid EIG for examples 0..k-1 and that information is kept.
            if "skipped_task" in df.columns:
                is_sentinel = pd.to_numeric(
                    df["skipped_task"], errors="coerce"
                ) == 1
                n_stopped = int(is_sentinel.sum())
                df = df[~is_sentinel].copy()
                if n_stopped:
                    print(f"[{player_tag}] {n_stopped} task/seed run(s) stopped "
                          f"early; their already-scored examples are kept")

            df["eig"] = pd.to_numeric(df["eig"], errors="coerce")

            out_dir = csv_path.parent
            stem = csv_path.stem

            # Per-task summary: one row per (player_tag, task_idx, seed)
            per_task = (
                df.groupby(["player_tag", "task_idx", "seed", "difficulty"])["eig"]
                .agg(n_examples="count", avg_eig="mean")
                .reset_index()
            )
            per_task_path = out_dir / f"{stem}_per_task.csv"
            per_task.to_csv(per_task_path, index=False)
            print(f"[{player_tag}] Saved per-task summary → {per_task_path}")
            print(per_task.to_string(index=False))

            # Overall summary: one row per player_tag
            overall = (
                df.groupby("player_tag")["eig"]
                .agg(n_examples="count", avg_eig="mean")
                .reset_index()
            )
            overall_path = out_dir / f"{stem}_overall.csv"
            overall.to_csv(overall_path, index=False)
            print(f"[{player_tag}] Saved overall summary  → {overall_path}")
            print(overall.to_string(index=False))

        except Exception as e:
            print(f"[{player_tag}] (Summary failed: {e})")

    print(f"\nDone. Total new rows written across all --dirs: {grand_total_written}")


if __name__ == "__main__":
    main()
