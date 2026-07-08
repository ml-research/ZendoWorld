import json
from pathlib import Path
import subprocess
from call_vision_model import call_vision_model
from data.create_programs import convert_prolog_to_dsl
from data.create_prolog import dsl_to_prolog
from data.pieces2tensor import prolog_strings_to_tensor
from data.tensor2piece import tensor_to_prolog_strings
from experiment_helper import (
    task_set2zendodataset,
    task_set2zendodataset_vlp,
)
from experiments.run_experiment import canonicalize_program, gather_data, normalize_program_structure
import random
import re
from generation.render import render_scene
from grammar import dsl
from models.prompters import get_prompter
from program import Program, strip_trailing_var0
from prompts.zendo.player import (
    query_structure_prompt,
    propose_structure_images_header,
    propose_structure_text_prompt,
    guess_rule_dsl_images_header,
    guess_rule_nl_images_header,
    guess_rule_dsl_text_prompt,
    detect_pieces_header,
    guess_rule_nl_text_prompt,
)
import torch
import time
from collections import Counter
import math
import ast
import numpy as np
import sys

from type_system import BOOL, Arrow, IMG, List, OBJECT, PROPERTY, ACTION
from DSL.vlp_dsl import get_dsl as get_vlp_dsl
from DSL.vlp_dsl_symbolic import get_dsl as get_vlp_dsl_symbolic


def load_api_key(path="./model/api.key"):
    with open(path, "r") as f:
        return f.read().strip()

def normalize_rule(rule):
    rule = strip_trailing_var0(rule)
    norm_rule = normalize_program_structure(rule)
    canonical_rule = canonicalize_program(norm_rule)
    return str(canonical_rule)

PREDICATE_TO_IDX_VAL = {
    "IS_RED":       (1, 0),
    "IS_BLUE":      (1, 1),
    "IS_YELLOW":    (1, 2),
    "IS_BLOCK":     (2, 0),
    "IS_WEDGE":     (2, 1),
    "IS_PYRAMID":   (2, 2),
    "IS_UPRIGHT":   (3, 0),
    "IS_UPSIDE_DOWN": (3, 1),
    "IS_FLAT":        (3, 2),
    "IS_CHEESECAKE":  (3, 3),
    "IS_HORIZONTAL":  (3, 2),
    "IS_VERTICAL":    (3, 0),
}

AMOUNT_PREDICATES = ["EVEN", "ODD", "EITHER_OR"]

def extract_predicates(program_str):
    preds = [word.rstrip(')') for word in program_str.split() if word.startswith("IS_")]
    if preds:
        return preds
    match = re.search(r'\(\s*(\w+)', program_str)
    if match:
        return [match.group(1)]
    return []


def parse_either_or_args(rule_str: str):
    """Extract the two integer args from an EITHER_OR rule like (EITHER_OR 2 3 var0) -> (2, 3)."""
    match = re.search(r'\(EITHER_OR\s+(\d+)\s+(\d+)', rule_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def call_prolog_subprocess_with_retries(n, query, prolog_file, retries=10, delay=2):
    """Run a Prolog query in a subprocess with retries, returning the JSON-parsed result or None."""
    for attempt in range(retries):
        try:
            abs_path = Path(prolog_file).resolve().as_posix()
            result = subprocess.check_output(
            [sys.executable, 'call_generate_prolog.py', str(n), query, abs_path],
            timeout=60,
            stderr=subprocess.STDOUT
        )
            return json.loads(result)
        except subprocess.TimeoutExpired:
            print(f"Timeout on attempt {attempt + 1}/{retries}")
        except subprocess.CalledProcessError as e:
            print(f"Subprocess failed on attempt {attempt + 1}/{retries}:\n", e.output.decode())
        except json.JSONDecodeError as e:
            print(f"JSON decode failed on attempt {attempt + 1}/{retries}:", e)
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}/{retries}:", e)

        if attempt < retries - 1:
            time.sleep(delay)

    print("All retry attempts failed.", query)
    return None


def _augment_prompt_with_failures(base_prompt, failed_proposals):
    """Append a 'do not propose these again' section if there are previous failures.

    Handles both the text-only prompt form (str) and the image form ((str, paths)).
    Returning the original prompt object when ``failed_proposals`` is empty keeps
    the prompter's memory cache effective on the first attempt.
    """
    if not failed_proposals:
        return base_prompt
    failure_note = (
        "\n\nThe following structures were proposed previously and FAILED to "
        "render or were duplicates of existing examples. DO NOT propose any of "
        "these again — pick something materially different:\n"
        + "\n".join(f"- {s}" for s in failed_proposals)
    )
    if isinstance(base_prompt, tuple) and len(base_prompt) == 2:
        text, paths = base_prompt
        return text + failure_note, paths
    return base_prompt + failure_note


class ZendoPlayerInterface:
    """Interface for Zendo players.

    Subclasses (or ZendoPlayer) override only the methods they need:
    - observe(example):  receive a new (input, label) pair; update internal state.
    - decide_guess(state) -> {"type": "guess_rule", "rule": Program}: guess the rule during GUESS phase.
    - guess_label(input_scene) -> bool: predict the label of a proposed input during QUIZ phase.
    - react(state) -> {"type": "propose_input", "input": (INPUT, PATH), "mode": "QUIZ"|"TELL", "rule": str}:
        propose a new input during PROPOSE phase.
    - quiz_correct() / quiz_incorrect(): feedback hook after a QUIZ guess.
    """

    def __init__(self, player_id, cfg, dsl, model=None):
        self.player_id = player_id

    def observe(self, example) -> None: ...
    def decide_guess(self, state) -> dict: ...
    def guess_label(self, input_scene) -> bool: ...
    def react(self, state) -> dict: ...
    def quiz_correct(self) -> None: ...
    def quiz_incorrect(self) -> None: ...

class ZendoPlayer:
    def __init__(self, player_id, task_idx, model, dsl, cfg, bar=5e-7, prefer_valid=True, min_examples=7, images=True, gs_threshold=0, vision_model=None, genai_client=None, use_dsl=True, seed=1):
        self.id = player_id
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.examples = []
        self.model = model
        self.dsl = dsl
        self.cfg = cfg
        self.pad_values = [7, 3, 3, 4, 7, 7, 7, 7, 7, 7, 7]
        self.guessing_stones = 0
        self.bar = bar
        self.incorrect_rules = []
        self.previous_guesses = []
        self.task_idx = task_idx
        self.use_model = model is not None
        self.prefer_valid = prefer_valid
        self.min_examples = min_examples
        self.create_images = images
        self.last_label = None
        self.gs_threshold = gs_threshold
        self.top_guess = None
        self.vision_model = vision_model
        self.genai_client = genai_client
        self.use_dsl = use_dsl
    def observe(self, example):
        if example == None or example[0] is None or example[0] == "":
            return
        if example[0][0] is None and self.vision_model is not None:
            image_tensor = call_vision_model(self.vision_model, example[1])
            example = ((image_tensor, example[0][1]), example[1])
        self.examples.append(example)

    def wrong_rule(self, rule):
        if rule not in self.incorrect_rules:
            self.incorrect_rules.append(normalize_rule(rule))
        self.top_guess = None

    def quiz_correct(self):
        self.guessing_stones += 1

    def quiz_incorrect(self):
        self.top_guess = None

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _search_programs(self, examples):
        """Run program search with decreasing accuracy thresholds, returning the candidates list."""
        dataset = task_set2zendodataset(
            [["", examples]], self.model, self.dsl, self.cfg, use_model=self.use_model
        )
        candidates = [(None, 0.0, 0.0, 0, 0.0, 0.0, 0.0)]
        for t in range(len(examples)):
            required_accuracy = 1 - (t / len(examples))
            print(f"Gathering data for accuracy {required_accuracy:.2f}...")
            data = gather_data(dataset, 0, accuracy=required_accuracy, incorrect_rules=self.incorrect_rules)
            candidates = data[0][1]
            if candidates != [(None, 0.0, 0.0, 0, 0.0, 0.0, 0.0)]:
                break
        return candidates

    def _candidate_path(self):
        return Path(str(self.task_idx)) / Path(str(self.seed)) / Path(str(self.id)) / str(len(self.examples))

    def _render_scene(self, scene, candidate_path, label=None, rule=""):
        """Render a Prolog scene to image or tensor; returns (None, None, None, None) on render failure so callers can retry."""
        if self.create_images:
            full_input_path = Path("generation") / Path("output") / (str(candidate_path) + ".png")
            new_input = render_scene(scene, path=candidate_path)
            if new_input is not None:
                return new_input, full_input_path, label, rule
            return None, None, None, None
        else:
            return prolog_strings_to_tensor([scene])[0], "", label, rule


    def guess_label(self, input_scene):
        if self.last_label is not None:
            label = self.last_label
            self.last_label = None
            return label
        examples, _ = zip(*self.examples)
        print(self.examples[0], examples[0])
        candidates = self._search_programs(examples)
        top_rule = candidates[0][0]
        try:
            top_rule = strip_trailing_var0(top_rule)
            prog_fn = top_rule.eval(dsl=self.dsl, environment=(None, None), i=0)
            self.top_guess = top_rule
            return prog_fn(input_scene[0])
        except Exception as e:
            print(f"Error evaluating rule {top_rule}: {e}")
            top_rule = candidates[1][0]
            try:
                top_rule = strip_trailing_var0(top_rule)
                prog_fn = top_rule.eval(dsl=self.dsl, environment=(None, None), i=0)
                self.top_guess = top_rule
                return prog_fn(input_scene[0])
            except Exception as e:
                print(f"Error evaluating rule {top_rule} again: {e}")
                return False

    def guess_labels(self, input_paths):
        input_scenes = [call_vision_model(self.vision_model, p) for p in input_paths]
        examples, _ = zip(*self.examples)
        candidates = self._search_programs(examples)
        top_rule = candidates[0][0]
        labels = []
        for input_scene in input_scenes:
            try:
                top_rule = strip_trailing_var0(top_rule)
                prog_fn = top_rule.eval(dsl=self.dsl, environment=(None, None), i=0)
                labels.append(prog_fn(input_scene))
            except Exception as e:
                print(f"Error evaluating rule {top_rule}: {e}")
                top_rule = candidates[1][0]
                try:
                    top_rule = strip_trailing_var0(top_rule)
                    prog_fn = top_rule.eval(dsl=self.dsl, environment=(None, None), i=0)
                    labels.append(prog_fn(input_scene))
                except Exception as e:
                    print(f"Error evaluating rule {top_rule} again: {e}")
                    labels.append(False)
        return labels, top_rule

    def decide_guess(self, state):
        if self.guessing_stones <= 0 or len(self.examples) < self.min_examples:
            return None
        rule = self.guess_rule()
        if rule is None:
            print(f"Player {self.id} could not find a rule")
            return None
        self.guessing_stones -= 1
        print(f"Player {self.id} guessed rule: {rule}")
        return {"type": "guess_rule", "rule": rule}

    def guess_rule(self):
        if self.top_guess is not None:
            guess = self.top_guess
            self.top_guess = None
            return guess
        examples, _ = zip(*self.examples)
        candidates = self._search_programs(examples)
        candidate_rule, *_ = candidates[0]
        if candidate_rule is None:
            candidates = self._search_programs(examples)
            candidate_rule, *_ = candidates[0]
        if candidate_rule is None:
            print(f"Player {self.id} could not find any valid rule in the dataset.")
            return None
        return candidate_rule
    
    def react(self, state):
        turn = state.current_turn.name
        if turn == "PROPOSE":
            return self._react_propose(state)
        elif turn == "LABEL":
            return self._react_label(state)
        elif turn == "GUESS":
            return self._react_guess(state)
        return None

    def _react_propose(self, state):
        proposed_input, path, label, rule = self.propose_input()
        self.last_label = label
        amount_players = len(state.player_guess_tokens)
        if proposed_input is None:
            print("Failed to propose input, returning None.")
            return {"type": "propose_input", "input": None, "mode": "TELL"}
        mode = "QUIZ" if (label is not None and self.guessing_stones <= self.gs_threshold) or amount_players == 1 else "TELL"
        return {"type": "propose_input", "input": (proposed_input, path), "mode": mode, "rule": rule}

    def _react_label(self, state):
        label = self.guess_label(state.input_scene)
        return {"type": "guess_label", "label": label}

    def _react_guess(self, state):
        action = self.decide_guess(state)
        if action is None:
            return {"type": "no_guess"}
        return action

    def propose_input(self):
        print(f"Proposing input based on {len(self.examples)} current examples...")
        examples, _ = zip(*self.examples)
        candidates = self._search_programs(examples)

        valid_candidates = [
            (prog, prob)
            for prog, *_, prob in candidates
            if normalize_rule(prog) not in self.incorrect_rules
        ]
        print(f"Valid candidates found: {valid_candidates}")
        if not valid_candidates:
            print("All candidate rules are in wrong_rules.")
            return None, None, None, None

        candidate_path = self._candidate_path()
        top_rule, _ = valid_candidates[0]
        self.top_guess = top_rule
        inner_query = dsl_to_prolog(top_rule)
        validity_order = [("valid", True), ("invalid", False)] if self.prefer_valid else [("invalid", False), ("valid", True)]

        if len(valid_candidates) == 1:
            for validity, label in validity_order:
                for _ in range(15):
                    prolog_str = f"generate_{validity}_structure([{inner_query}], Structure)"
                    _res = call_prolog_subprocess_with_retries(1, prolog_str, "rules/rules.pl")
                    if _res is None:
                        continue
                    scene = _res[0]
                    if scene is None:
                        continue
                    result = self._render_scene(scene, candidate_path, label, str(top_rule))
                    if result[0] is not None:
                        return result
                    print(f"Render failed for {validity} scene, generating new scene...", scene)
            print("Failed to generate both valid and invalid scenes.")
            return self._propose_random_valid(top_rule, inner_query, candidate_path)

        # Primary disprove strategy: invalid for top rule, valid for second.
        second_rule, _ = valid_candidates[1]
        second_query = dsl_to_prolog(second_rule)
        disprove_query = f"generate_disproving_structure([{inner_query}], [{second_query}], Structure)"
        print(f"Trying disprove strategy (not rule1 and rule2)...")
        for attempt in range(6):
            _res = call_prolog_subprocess_with_retries(1, disprove_query, "rules/rules.pl")
            if _res is None:
                continue
            scene = _res[0]
            if scene is None:
                continue
            try:
                proposed_input = prolog_strings_to_tensor([scene])[0]
            except Exception as e:
                print(f"Failed to convert disprove scene: {e}")
                continue
            result = self._render_scene(scene, candidate_path, False, str(top_rule))
            if result[0] is not None:
                print(f"Disprove scene found on attempt {attempt + 1}.")
                return result
            print(f"Render failed for disprove scene, retrying...", scene)

        for i, (validity, label) in enumerate(validity_order):
            print(f"Trying to generate a '{validity}' scene...")
            for j in range(6):
                prolog_str = f"generate_{validity}_structure([{inner_query}], Structure)"
                _res = call_prolog_subprocess_with_retries(1, prolog_str, "rules/rules.pl")
                if _res is None:
                    print("Prolog returned None for scene generation.")
                    continue
                scene = _res[0]
                if scene is None:
                    print("Prolog returned None for scene generation.")
                    continue

                try:
                    proposed_input = prolog_strings_to_tensor([scene])[0]
                except Exception as e:
                    print(f"Failed to convert scene, generating new scene: {e}")
                    continue

                eval_results = []
                for prog, _ in valid_candidates:
                    try:
                        strip_trailing_var0(prog)
                        prog_fn = prog.eval(dsl=self.dsl, environment=(None, None), i=0)
                        eval_results.append(prog_fn(proposed_input))
                    except Exception as e:
                        print("Evaluation error:", e)
                        eval_results.append(False)

                counts = Counter(eval_results)
                _, most_common_count = counts.most_common(1)[0]
                num_disagreeing = len(eval_results) - most_common_count

                large_set_discrim = len(valid_candidates) > 3 and num_disagreeing >= len(valid_candidates) // 2 - 1
                small_set_discrim = len(valid_candidates) <= 3 and num_disagreeing >= 1
                is_fallback = i == 1 and j == 29

                if large_set_discrim:
                    print(f"Discriminating input found: {num_disagreeing} disagreements out of {len(valid_candidates)}")
                elif small_set_discrim:
                    print("Input distinguishes among small candidate set.")
                elif is_fallback:
                    print(f"Failed to find a discriminating input after 30 attempts for {validity} scene. Returning last attempt.")
                else:
                    continue

                result = self._render_scene(scene, candidate_path, label, str(top_rule))
                if result[0] is not None:
                    return result
                print(f"Render failed for discriminating scene, generating new scene...", scene)

        print("No discriminating input found from either validity. Falling back...")
        return self._propose_random_valid(top_rule, inner_query, candidate_path)

    def _propose_random_valid(self, top_rule, inner_query, candidate_path):
        print("Attempting random fallback scene generation...")
        for query in [
            f"generate_valid_structure([{inner_query}], Structure)",
            "generate_valid_structure([], Structure)",
        ]:
            for _ in range(5):
                _res = call_prolog_subprocess_with_retries(1, query, "rules/rules.pl")
                if _res is None:
                    continue
                scene = _res[0]
                if scene is None:
                    continue
                try:
                    proposed_input = prolog_strings_to_tensor([scene])[0]
                    strip_trailing_var0(top_rule)
                    prog_fn = top_rule.eval(dsl=self.dsl, environment=(None, None), i=0)
                    label = bool(prog_fn(proposed_input))
                except Exception as e:
                    print(f"Fallback eval error: {e}")
                    continue
                result = self._render_scene(scene, candidate_path, label, str(top_rule))
                if result[0] is not None:
                    print("Fallback random scene succeeded.")
                    return result
        print("Fallback random scene generation failed entirely.")
        return None, None, None, None

class RandomZendoPlayer(ZendoPlayer):
    def _react_propose(self, state):
        proposed_input, path, label, rule = self.propose_input()
        amount_players = len(state.player_guess_tokens)
        if proposed_input is None:
            print("Failed to propose input, returning None.")
            return {"type": "propose_input", "input": None, "mode": "TELL"}
        mode = "QUIZ" if (label is not None and self.guessing_stones <= self.gs_threshold) or amount_players == 1 else "TELL"
        return {"type": "propose_input", "input": (proposed_input, path), "mode": mode, "rule": rule}
    
    def propose_input(self):
        print(f"Proposing input based on {len(self.examples)} current examples...")
        candidate_path = self._candidate_path()
        for _ in range(100):
            prolog_str = f"generate_valid_structure([], Structure)"
            _res = call_prolog_subprocess_with_retries(1, prolog_str, "rules/rules.pl")
            scene = _res[0] if _res is not None else None
            if scene is not None:
                try:
                    result = self._render_scene(scene, candidate_path, False, "")
                    if result[0] is not None:
                        return result
                except Exception as e:
                    print(f"Failed to convert Prolog scene to tensor:", e)
                    return None, None, None, None
            print("Failed to generate both valid and invalid scenes.")
            return None, None, None, None
        print("No discriminating input found from either validity. Falling back...")
        return None, None, None, None

COLOR_IDX = 1
SHAPE_IDX = 2
ORIENT_IDX = 3
max_values = {
    COLOR_IDX: 3,
    SHAPE_IDX: 3,
    ORIENT_IDX: 4,
}


def random_piece_like(piece: torch.Tensor, rng=None) -> torch.Tensor:
    if rng is None:
        rng = random
    attr_idx = rng.choice([COLOR_IDX, SHAPE_IDX, ORIENT_IDX])
    current_val = int(piece[attr_idx].item())
    candidates = [v for v in range(max_values[attr_idx]) if v != current_val]
    new_val = rng.choice(candidates)

    new_piece = piece.clone()
    new_piece[attr_idx] = new_val
    return new_piece

class HeuristicZendoPlayer(ZendoPlayer):
    PAD_VALS = torch.tensor([7, 3, 3, 4, 7, 7, 7, 7, 7, 7, 7, -1, -1, -1, -1], dtype=torch.int64)

    def is_padding(self, piece):
        return torch.all(piece == self.PAD_VALS)
    
    def non_padded_indices(self, structure):
        return [i for i, p in enumerate(structure) if not self.is_padding(p)]
    
    def _react_propose(self, state):
        proposed_input, path = self.propose_input()
        if proposed_input is None:
            print("Failed to propose input, returning None.")
            return {"type": "propose_input", "input": None, "mode": "TELL"}
        return {"type": "propose_input", "input": (proposed_input, path), "mode": "QUIZ"}

    def ensure_size(self, structure: torch.Tensor) -> torch.Tensor:
        """Ensure structure has shape (7, 15); right-pad short columns with -1."""
        rows, target_cols = 7, 15
        current_cols = structure.size(1)

        if current_cols < target_cols:
            pad_cols = target_cols - current_cols
            pad_tensor = torch.full((rows, pad_cols), -1, dtype=structure.dtype, device=structure.device)
            structure = torch.cat([structure, pad_tensor], dim=1)

        return structure

    def propose_input(self, max_attempts=15):
        for _ in range(max_attempts):
            examples, _ = zip(*self.examples)
            print(f"Proposing input based on {len(examples)} current examples...")
            base_structure, _ = self.rng.choice(examples)
            base_structure = self.ensure_size(base_structure)
            mutation = self.rng.choice(self.heuristics)
            new_input = mutation(base_structure)
            duplicate = any(
                (new_input.shape == ex.shape) and torch.equal(new_input, ex)
                for (ex, _) in examples
            )

            if duplicate or new_input is None:
                print(f"Heuristic Player proposed a duplicate structure, retrying...")
                continue

            if self.create_images:
                candidate_path = self._candidate_path()
                full_input_path = Path("generation") / Path("output") / (str(candidate_path) + ".png")
                new_input_rendered = render_scene(tensor_to_prolog_strings([new_input])[0], path=candidate_path)
                if new_input_rendered is not None:
                    return new_input_rendered, full_input_path
            else:
                return new_input, ""
        return None, ""

    def reduce_by_one(self, structure):
        structure = structure.clone()
        indices = self.non_padded_indices(structure)
        if len(indices) <= 1:
            return structure
        idx_to_remove = self.rng.choice(indices)
        structure = torch.cat([structure[:idx_to_remove], structure[idx_to_remove+1:], self.PAD_VALS.unsqueeze(0)], dim=0)
        return structure

    def substitute_one_piece(self, structure):
        structure = structure.clone()
        indices = self.non_padded_indices(structure)
        if not indices:
            return structure
        i = self.rng.choice(indices)
        new_piece = self.random_piece_like(structure[i])
        structure[i] = new_piece
        return structure

    def homogenize_attribute(self, structure):
        structure = structure.clone()
        indices = self.non_padded_indices(structure)
        if not indices:
            return structure
        attr_idx = self.rng.choice([COLOR_IDX, SHAPE_IDX, ORIENT_IDX])
        val = self.rng.choice([int(structure[i][attr_idx].item()) for i in indices])
        if val == 3:
            val = 2
        for i in indices:
            structure[i][attr_idx] = val
        return structure
    
    def single_piece_structure(self, structure):
        structure = structure.clone()
        N, D = structure.shape
        indices = self.non_padded_indices(structure)
        if not indices:
            return structure

        i = self.rng.choice(indices)
        selected_piece = structure[i].clone()
        if D >= 11:
            selected_piece[4:11] = 8
            if D > 11:
                selected_piece[11:] = -1
        else:
            selected_piece[4:] = 8

        padding = self.PAD_VALS.unsqueeze(0).repeat(6, 1)
        new_structure = torch.cat([selected_piece.unsqueeze(0), padding], dim=0)
        return new_structure
    
    def spread_structure(self, structure):
        structure = structure.clone()
        indices = self.non_padded_indices(structure)
        for i in indices:
            structure[i][4:11] = 8
        return structure

    def random_piece_like(self, piece: torch.Tensor) -> torch.Tensor:
        new_piece = None
        while new_piece is None or torch.equal(new_piece, piece):
            attr_idx = self.rng.choice([COLOR_IDX, SHAPE_IDX, ORIENT_IDX])
            current_val = int(piece[attr_idx].item())
            max_values = {
                COLOR_IDX: 3,
                SHAPE_IDX: 3,
                ORIENT_IDX: 4
            }
            candidates = [v for v in range(max_values[attr_idx]) if v != current_val]
            new_val = self.rng.choice(candidates)
            new_piece = piece.clone()
            new_piece[attr_idx] = new_val
            if new_piece[2] != 1 and new_piece[3] == 3:
                new_piece = None
                continue
        return new_piece

    @property
    def heuristics(self):
        return [
            self.reduce_by_one,
            self.substitute_one_piece,
            self.homogenize_attribute,
            self.single_piece_structure,
            self.spread_structure
        ]

class GPTQueryZendoPlayer(HeuristicZendoPlayer):
    def _normalize_item_str(self, s: str) -> str:
        return re.sub(r'\s+', '', s.strip().lower())

    def _canonicalize_items(self, items: list[str]) -> tuple[str, ...]:
        # Sort so item-list order doesn't affect equality.
        return tuple(sorted(self._normalize_item_str(x) for x in items))


    def _react_propose(self, state):
        proposed_input, rule, path = self.propose_input()
        if proposed_input is None:
            print("Failed to propose input, returning None.")
            return {"type": "propose_input", "input": None, "mode": "TELL"}
        return {"type": "propose_input", "input": (proposed_input, path), "mode": "QUIZ", "rule": rule}

    def build_zendo_prompt_from_examples(self, examples, top_rules):
        tensors = [t for t, _ in examples]
        labels = [l for _, l in examples]
        structure_strs = tensor_to_prolog_strings(tensors)
        formatted = [(label, str(items)) for label, items in zip(labels, structure_strs)]
        positives = [s for l, s in formatted if l == 1]
        negatives = [s for l, s in formatted if l == 0]
        prompt_text = query_structure_prompt.format(
            top_rules="\n".join(map(str, top_rules)),
            positives="\n".join(positives),
            negatives="\n".join(negatives),
        )
        return prompt_text, []

    def query_gpt_for_structure(self, prompt, seed=None):
        """Send a structure-proposal prompt; ``prompt`` may be text or a (text, paths) tuple.

        ``seed`` lets the caller bump the seed per retry so the prompter's
        memory cache (keyed on (prompt_text, seed)) and the API's seeded
        sampling both diverge across attempts.
        """
        print("Querying LLM for structure generation...")
        if not hasattr(self, 'prompter') or self.prompter is None:
            self.prompter = get_prompter("gpt-5-mini", "zendo", self.seed, reasoning=False, sampling=False)

        if isinstance(prompt, tuple) and len(prompt) == 2:
            prompt_text, paths = prompt
        else:
            prompt_text, paths = prompt, []

        call_seed = self.seed if seed is None else seed

        try:
            if paths:
                response_text = self.prompter.prompt_with_images(
                    prompt_text=prompt_text, paths=paths, seed=call_seed,
                )
            else:
                response_text = self.prompter.prompt_with_text(
                    prompt_text=prompt_text, seed=call_seed,
                )
            outputs = response_text
            lines = [line for line in response_text.splitlines() if not line.strip().startswith("#")]
            response_text = "\n".join(lines).strip()
            if response_text.startswith("```python"):
                response_text = response_text.strip("`").split("python", 1)[-1].strip()
            if response_text.endswith("```"):
                response_text = response_text.rsplit("```", 1)[0].strip()
            print("LLM response:", response_text, type(response_text))
            if not isinstance(response_text, str):
                print("LLM response is not a string.")
                return None
            try:
                parsed = ast.literal_eval(response_text)
            except Exception as e:
                print("Failed to parse response using ast.literal_eval:", e)
                return None
            if not isinstance(parsed, list) or len(parsed) != 2:
                print("Unexpected format. Expected: [[item strings...], label]")
                return None
            items, label = parsed
            if not isinstance(items, list) or not all(isinstance(s, str) for s in items):
                print("Invalid item list.")
                return None
            input_tensor = prolog_strings_to_tensor([items])[0]
            print("LLM response:", response_text, "Parsed tensor:", input_tensor)
            if input_tensor is None:
                print("Failed to parse LLM response into tensor.")
                return None
            print("LLM response successfully parsed into tensor.")
            return input_tensor
        except Exception as e:
            print("Failed to query LLM:", e)
            return None

    def propose_input(self, max_retries: int = 10):
        print(f"Proposing input based on {len(self.examples)} current examples...")
        examples, _ = zip(*self.examples)
        candidates = self._search_programs(examples)

        valid_candidates = [
            prog
            for prog, *_ in candidates
            if normalize_rule(prog) not in self.incorrect_rules
        ]
        base_prompt = self.build_zendo_prompt_from_examples(examples, valid_candidates[:2])
        candidate_path = self._candidate_path()
        # Structures the LLM proposed that didn't survive rendering / dedup
        # checks. We feed these back into the prompt on retry so the prompter's
        # memory cache misses (different prompt text -> different cache key)
        # and the LLM has explicit context for what to avoid. Without this, a
        # render failure leads to an infinite loop on the same cached response.
        failed_proposals: list[str] = []
        for attempt in range(1, max_retries + 1):
            prompt = _augment_prompt_with_failures(base_prompt, failed_proposals)
            # Bump the seed per retry so the prompter's memory cache key
            # (prompt_text, seed) differs even when the failure note hasn't
            # nudged the LLM off its previous output.
            attempt_seed = self.seed + attempt - 1
            structure = self.query_gpt_for_structure(prompt, seed=attempt_seed)
            if structure is None:
                print(f"Attempt {attempt}: GPT failed to produce a valid structure.")
                continue

            duplicate = any(
                (structure.shape == ex.shape) and torch.equal(structure, ex)
                for (ex, _) in examples
            )

            structure_str = str(tensor_to_prolog_strings([structure])[0])
            if duplicate:
                print(f"Attempt {attempt}: GPT proposed a duplicate structure, retrying...")
                failed_proposals.append(structure_str)
                continue

            if self.create_images:
                full_input_path = Path("generation") / Path("output") / (str(candidate_path) + ".png")
                new_input_rendered = render_scene(tensor_to_prolog_strings([structure])[0], path=candidate_path)
                if new_input_rendered is not None:
                    print(f"Novel structure generated on attempt {attempt}.")
                    return new_input_rendered, str(valid_candidates[0]), full_input_path
                else:
                    print(f"Attempt {attempt}: Failed to render structure to image.")
                    failed_proposals.append(structure_str)
                    continue
            else:
                print(f"Novel structure generated on attempt {attempt}.")
                return structure, str(valid_candidates[0]), ""

        print("Failed to generate a novel structure after retries.")
        return None, None, ""
        
class FullGPTZendoPlayer(GPTQueryZendoPlayer):
    def __init__(self, player_id, task_idx, model, dsl, cfg, bar=5e-7, prefer_valid=True, min_examples=7, images=True, gs_threshold=0, vision_model=None, genai_client=None, use_dsl=False, seed=1, prompter=None):
        self.id = player_id
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.examples = []
        self.model = model
        self.dsl = dsl
        self.cfg = cfg
        self.pad_values = [7, 3, 3, 4, 7, 7, 7, 7, 7, 7, 7]
        self.guessing_stones = 0
        self.bar = bar
        self.incorrect_rules = []
        self.task_idx = task_idx
        self.use_model = model is not None
        self.last_label = None
        self.previous_guesses = []
        self.min_examples = min_examples
        self.create_images = images
        self.gs_threshold = gs_threshold
        self.vision_model = vision_model
        self.genai_client = genai_client
        self.use_dsl = use_dsl
        if prompter is None:
            prompter = get_prompter("gpt-5-mini", "zendo", seed, reasoning=False, sampling=False)
        self.prompter = prompter

    def _react_propose(self, state):
        proposed_input, label, path = self.propose_input()
        self.last_label = label
        if proposed_input is None:
            print("Failed to propose input, returning None.")
            return {"type": "propose_input", "input": None, "mode": "TELL"}
        return {"type": "propose_input", "input": (proposed_input, path), "mode": "QUIZ"}

    def extract_list_literal(self, text):
        match = re.search(r"(\[\s*\[.*?\]\s*,\s*[01]\s*\])", text, re.DOTALL)
        if match:
            return match.group(1)
        return None
    
    def extract_python_block(self, text):
        match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def extract_prolog_block(self, text):
        match = re.search(r"```prolog\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    def extract_generic_code_block(self, text):
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def query_gpt_for_structure(self, prompt, seed=None):
        """Send a structure / rule-guessing prompt; ``prompt`` may be text or a (text, paths) tuple.

        ``seed`` lets the caller bump the seed per retry so the prompter's
        memory cache (keyed on (prompt_text, seed)) and the API's seeded
        sampling both diverge across attempts.
        """
        print("Querying LLM for structure/rule generation...")
        if isinstance(prompt, tuple) and len(prompt) == 2:
            prompt_text, paths = prompt
        else:
            prompt_text, paths = prompt, []
        call_seed = self.seed if seed is None else seed
        try:
            if paths:
                response_text = self.prompter.prompt_with_images(
                    prompt_text=prompt_text, paths=paths, seed=call_seed,
                )
            else:
                response_text = self.prompter.prompt_with_text(
                    prompt_text=prompt_text, seed=call_seed,
                )
            outputs = response_text
            # Open-weight chat models (Gemma, Qwen, ...) often wrap the answer
            # in prose plus a fenced ```python / ```prolog block. Pull the
            # block from anywhere in the response; if no fence is found, fall
            # back to using the whole text (which matches the prior GPT path).
            fence_match = re.search(
                r"```(?:python|prolog|json)?\s*\n?(.*?)(?:```|\Z)",
                response_text,
                re.DOTALL,
            )
            if fence_match:
                response_text = fence_match.group(1)
            lines = [line for line in response_text.splitlines() if not line.strip().startswith("#")]
            response_text = "\n".join(lines).strip()
            if response_text:
                return response_text
            raw = outputs
            code = self.extract_python_block(raw)
            if code is None:
                code = self.extract_list_literal(raw)
            if code is None:
                code = self.extract_prolog_block(raw)
            if code is None:
                code = self.extract_generic_code_block(raw)
            return code
        except Exception as e:
            print("Failed to query LLM:", e)
            return None

    def propose_input(self, max_retries=10):
        print(f"Proposing input based on {len(self.examples)} current examples...")
        examples, paths = zip(*self.examples)
        base_prompt = self.build_zendo_prompt_from_examples(examples, paths, True)
        # See _augment_prompt_with_failures: same prompt across retries hits the
        # prompter's memory cache and yields the same broken structure forever.
        failed_proposals: list[str] = []
        for i in range(max_retries):
            try:
                prompt = _augment_prompt_with_failures(base_prompt, failed_proposals)
                # Bump the seed per retry so the prompter's memory cache key
                # (prompt_text, seed) differs even when the failure note hasn't
                # nudged the LLM off its previous output.
                attempt_seed = self.seed + i
                response_text = self.query_gpt_for_structure(prompt, seed=attempt_seed)
                if type(response_text) is str:
                    try:
                        parsed = ast.literal_eval(response_text)
                    except Exception as e:
                        print("Failed to parse response using ast.literal_eval:", e, response_text)
                        continue

                    if not isinstance(parsed, list) or len(parsed) != 2:
                        print("Unexpected format. Expected: [[item strings...], label]", response_text)
                        continue

                    items, label = parsed
                    if not isinstance(items, list) or not all(isinstance(s, str) for s in items):
                        print("Invalid item list.", response_text)
                        continue
                else:
                    print("Model response is not a string.", response_text)
                    items = response_text[0]
                input_tensor = prolog_strings_to_tensor([items])[0]
                if input_tensor is None:
                    print("Failed to parse model response into tensor.")
                    continue
                else:
                    duplicate = any(
                        (input_tensor.shape == ex.shape) and torch.equal(input_tensor, ex)
                        for (ex, _) in examples
                    )

                    if duplicate:
                        print(f"Attempt {i}: Model proposed a duplicate structure, retrying...")
                        failed_proposals.append(str(items))
                        continue
                    if self.create_images:
                        print("Model response successfully parsed into tensor.")
                        candidate_path = self._candidate_path()
                        full_input_path = Path("generation") / Path("output") / (str(candidate_path) + ".png")
                        try:
                            new_input = render_scene(items, path=candidate_path)
                            if new_input is None:
                                print("Failed to render scene, returning None.")
                                failed_proposals.append(str(items))
                                continue
                            return new_input, label, full_input_path
                        except Exception as e:
                            print(f"Failed to convert Prolog scene to tensor:", e)
                            failed_proposals.append(str(items))
                            continue
                    else:
                        return input_tensor, label, ""
            except Exception as e:
                print("Failed to generate input:", e)
                return None, None, None
        return None, None, None

    def guess_label(self, input_scene):
        print(f"Guessing label for input scene: {self.last_label}")
        guess = bool(self.last_label)
        self.last_label = None
        return guess

    def guess_rule(self, max_attempts=5):
        print(f"Guessing rule based on {len(self.examples)} examples...")
        examples, paths = zip(*self.examples)
        prompt = self.build_zendo_prompt_guess_rule(examples, paths, True)
        for i in range(max_attempts):
            response_text = self.query_gpt_for_structure(prompt)
            if type(response_text) is str and self.use_dsl:
                try:
                    self.previous_guesses.append(response_text)
                    program = convert_prolog_to_dsl(response_text, self.cfg)
                except Exception as e:
                    print("Failed to parse response into DSL:", e)
                    if response_text in self.incorrect_rules:
                        continue
                    return response_text

                if program is not None and normalize_rule(program) not in self.incorrect_rules:
                    print("returning", str(program))
                    return program
            else:
                print("LLM response is not a string.")
                self.previous_guesses.append(response_text)
                return response_text
            print("Failed to parse LLM response into a rule.")
            return None

    def _collect_examples(self, examples, paths, use_paths):
        """Split observed examples into (positives_text, negatives_text, positives_paths, negatives_paths).

        Path lists are populated only when ``use_paths`` is True.
        """
        tensors = [t for t, _ in examples]
        labels = [l for _, l in examples]
        structure_strs = tensor_to_prolog_strings(tensors)
        if use_paths and paths and len(paths) != len(examples):
            raise ValueError("paths length must match len(examples) when use_paths=True.")
        positives_text, negatives_text = [], []
        positives_paths, negatives_paths = [], []
        for label, struct_str, img_path in zip(
            labels, structure_strs, paths if use_paths else [None] * len(examples)
        ):
            if label == 1:
                positives_text.append(str(struct_str))
                if img_path and img_path != "":
                    positives_paths.append(str(img_path))
            else:
                negatives_text.append(str(struct_str))
                if img_path and img_path != "":
                    negatives_paths.append(str(img_path))
        return positives_text, negatives_text, positives_paths, negatives_paths

    def _build_image_prompt(self, header_text, positives_paths, negatives_paths):
        """Build a (prompt_text, paths) pair where ordering is described in text since paths is a flat list."""
        n_pos, n_neg = len(positives_paths), len(negatives_paths)
        order_note = (
            f"\n\nThe attached images are ordered: the first {n_pos} are "
            f"POSITIVE examples and the last {n_neg} are NEGATIVE examples."
        )
        prompt_text = header_text + order_note
        paths = list(positives_paths) + list(negatives_paths)
        return prompt_text, paths

    def build_zendo_prompt_guess_rule(self, examples, paths, use_paths=True):
        print(self.previous_guesses)
        positives_text, negatives_text, positives_paths, negatives_paths = \
            self._collect_examples(examples, paths, use_paths)
        prev_guesses_note = (
            "You might have already guessed some rules, but they were incorrect or incomplete. "
            "DO NOT output these rules again. These are the rules you guessed so far:\n"
            + "\n".join(self.previous_guesses)
        )
        if not use_paths or not (positives_paths or negatives_paths):
            if self.use_dsl:
                return guess_rule_dsl_text_prompt.format(
                    positives="\n".join(positives_text),
                    negatives="\n".join(negatives_text),
                    previous_guesses="\n".join(self.previous_guesses),
                ), []
            return guess_rule_nl_text_prompt.format(
                positives="\n".join(positives_text),
                negatives="\n".join(negatives_text),
                previous_guesses="\n".join(self.previous_guesses),
            ), []
        header = guess_rule_dsl_images_header if self.use_dsl else guess_rule_nl_images_header
        return self._build_image_prompt(
            header + "\n\n" + prev_guesses_note,
            positives_paths,
            negatives_paths,
        )

    def build_zendo_prompt_from_examples(self, examples, paths, use_paths=True):
        positives_text, negatives_text, positives_paths, negatives_paths = \
            self._collect_examples(examples, paths, use_paths)
        if use_paths and (positives_paths or negatives_paths):
            return self._build_image_prompt(
                propose_structure_images_header,
                positives_paths,
                negatives_paths,
            )
        return propose_structure_text_prompt.format(
            positives="\n".join(positives_text),
            negatives="\n".join(negatives_text),
        ), []


class GPTVisionModel(FullGPTZendoPlayer):
    def detect_image(self, path, json_path):
        print(f"Detecting pieces in image {path}...")
        prompt = self.build_detection_prompt_from_examples(path)
        if prompt is None:
            print("No examples provided for detection.")
            return
        response_text = self.query_gpt_for_structure(prompt)
        # response_text = "['item(0, yellow, pyramid, upright, grounded)', 'item(1, green, pyramid, upright, grounded)', 'item(2, yellow, wedge, flat, pointing(0))']"
        if type(response_text) is str:
            try:
                parsed = ast.literal_eval(response_text)
            except Exception as e:
                print("Failed to parse response using ast.literal_eval:", e, response_text)

            if not isinstance(parsed, list):
                print("Unexpected format. Expected: [item strings...]", response_text)
            print("Parsed:", parsed)
            items = parsed
            if not isinstance(items, list) or not all(isinstance(s, str) for s in items):
                print("Invalid item list.", response_text)
            else:
                try:
                    input_tensor = prolog_strings_to_tensor([items])[0]
                    if input_tensor is None:
                        print("Failed to parse GPT-4o response into tensor.")
                except Exception as e:
                    print("Error occurred while converting items to tensor:", e)
                    input_tensor = None
            if input_tensor is None:
                record = {
                    "image_path": str(path),
                    "gpt_output": response_text,
                    "parsed_items": items,
                }
            else:
                record = {
                    "image_path": str(path),
                    "gpt_output": response_text,
                    "parsed_items": items,
                    "tensor": _to_jsonable(input_tensor),
                }
            json_path = Path(json_path)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with json_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    def build_detection_prompt_from_examples(self, path):
        if not path:
            return None
        return detect_pieces_header, [str(path)]

def _to_jsonable(x):
    try:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
    except Exception:
        pass
    try:
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    return x

class VLPZendoPlayer(ZendoPlayer):
    def __init__(
        self,
        player_id,
        task_idx,
        model,
        dsl,
        cfg,
        zendo_cfg,
        bar=5e-7,
        prefer_valid=True,
        min_examples=7,
        images=True,
        gs_threshold=0,
        vision_model=None,
        genai_client=None,
        prompter=None,
        discovery_examples=4,
        n_objects=8,
        n_properties=12,
        n_actions=0,
        n_sceneries=0,
        seed=1,
        initial_variables=None,
        symbolic=False,
    ):
        self.id = player_id
        self.symbolic = symbolic
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.examples = []
        self.model = model
        self.dsl = dsl
        self.cfg = cfg
        self.zendo_cfg = zendo_cfg
        self.pad_values = [7, 3, 3, 4, 7, 7, 7, 7, 7, 7, 7]
        self.guessing_stones = 0
        self.bar = bar
        self.incorrect_rules = []
        self.previous_guesses = []
        self.task_idx = task_idx
        self.use_model = model is not None
        self.prefer_valid = prefer_valid
        self.min_examples = min_examples
        self.create_images = images
        self.last_label = None
        self.gs_threshold = gs_threshold
        self.top_guess = None
        self.second_guess = None
        self.already_guessed = []
        self.vision_model = vision_model
        self.genai_client = genai_client
        self.prompter = prompter
        self.n_objects = max(1, int(n_objects))
        self.n_properties = max(1, int(n_properties))
        self.n_actions = max(0, int(n_actions))
        self.n_sceneries = max(0, int(n_sceneries))
        self._discovery_count = 0

        if initial_variables is not None:
            # Skip discovery; use the provided variables and rebuild the DSL.
            self.discovered_variables = {
                "objects": list(initial_variables.get("objects", [])),
                "properties": list(initial_variables.get("properties", [])),
                "actions": list(initial_variables.get("actions", [])),
            }
            self.discovery_rounds = 0
            self._rebuild_dsl_from_variables()
            print(
                f"[VLPZendoPlayer] Initialised with pre-specified variables: "
                f"{self.discovered_variables}. Discovery disabled."
            )
        else:
            self.discovery_rounds = max(1, int(discovery_examples))
            self.discovered_variables = {
                "objects": [],
                "properties": [],
                "actions": [],
            }

    def observe(self, example):
        self.examples.append(example)
        self._maybe_discover_variables()

    @staticmethod
    def _sorted_unique(values):
        return sorted({v for v in values if isinstance(v, str) and v.strip()})

    @staticmethod
    def _dedupe_keep_order(values):
        seen = set()
        out = []
        for x in values:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    @staticmethod
    def _parse_incomplete_list(s):
        s = s.split("[", 1)[-1]
        s = "[" + s
        if not s.endswith("]"):
            s += "]"
        try:
            return ast.literal_eval(s)
        except Exception:
            s = s.lstrip("[").rstrip("]")
            return [p.strip().strip("'").strip('"') for p in s.split(",") if p.strip()]

    def _prompt_with_text(self, prompt_text, max_new_tokens=1500, seed=None):
        return self.prompter.prompt_with_text(
            prompt_text=prompt_text,
            max_new_tokens=max_new_tokens,
            seed=self.seed if seed is None else seed,
        )

    def _prompt_with_images(self, prompt_text, paths, max_new_tokens=1500, seed=None):
        # Some prompters don't accept overwrite_memory / seed.
        call_seed = self.seed if seed is None else seed
        try:
            return self.prompter.prompt_with_images(
                prompt_text=prompt_text,
                paths=paths,
                max_new_tokens=max_new_tokens,
                overwrite_memory=False,
                seed=call_seed,
            )
        except TypeError:
            try:
                return self.prompter.prompt_with_images(
                    prompt_text=prompt_text,
                    paths=paths,
                    max_new_tokens=max_new_tokens,
                    seed=call_seed,
                )
            except TypeError:
                return self.prompter.prompt_with_images(
                    prompt_text=prompt_text,
                    paths=paths,
                    max_new_tokens=max_new_tokens,
                )

    def _load_prompt_or_default(self, prompt_path, fallback_text):
        if prompt_path is not None and Path(prompt_path).exists():
            with open(prompt_path, "r") as f:
                return f.read()
        return fallback_text

    def _already_discovered_header(self, category):
        """Prompt header listing already-discovered variables; empty before the first discovery round."""
        if self._discovery_count <= 1:
            return ""

        existing = self.discovered_variables.get(category, [])
        if not existing:
            return ""

        return (
            f"\n## Already Discovered\n"
            f"The following {category} have already been identified in previous "
            f"images: {existing}\n"
            f"Do NOT include any of these in your response. Only return {category} "
            f"that are NEW and not already in the list above. "
            f"If you do not see anything new, return an empty list [].\n\n"
        )

    def _discover_objects(
        self,
        image_paths,
        n_min_properties=None,
        prompter=None,
        prompt_path=None,
    ):
        if prompter is None:
            raise ValueError("Prompter is not provided. Please provide a prompter.")

        if prompt_path is None:
            prompt_path = "vlp/prompts/discovery/objects_old_new.txt"

        fallback_prompt = (
            "You are provided with images. Identify object categories that appear.\n"
            f"Return exactly {n_min_properties} object names when possible.\n"
            "Output only:\n```python\nobjects = [...]\n```"
        )
        print(f"Discovering objects with prompt: {fallback_prompt}")
        prompt_template = self._load_prompt_or_default(prompt_path, fallback_prompt)
        prompt = self._already_discovered_header("objects") + prompt_template.replace("{n}", str(n_min_properties))
        print(f"Final object discovery prompt:\n{prompt}\nWith image paths: {image_paths}")
        if self.symbolic:
            prompt = prompt + "\n\nScene descriptions:\n" + "\n".join(image_paths)
            response = self._prompt_with_text(prompt_text=prompt, max_new_tokens=1500)
        else:
            response = self._prompt_with_images(prompt_text=prompt, paths=image_paths, max_new_tokens=1500)
            print(f"Raw response for object discovery: {response}")
        original_response = response
        try:
            response = response.replace("\n", "")
            response = response.split("objects =")[-1]
            response = response.split("```")[0]
            response = response.split("[")[-1]
            response = response.split("]")[0]
            objects = eval(f"[{response}]")
            print(f"Parsed objects: {objects}")
        except Exception:
            try:
                objects = self._parse_incomplete_list(original_response)
                print(f"Parsed objects from incomplete list: {objects}")
            except Exception:
                objects = []

        objects = [x for x in objects if isinstance(x, str)]
        return self._dedupe_keep_order(objects)

    def _discover_properties(
        self,
        image_paths,
        objects,
        n_min_properties=None,
        prompter=None,
        prompt_path=None,
    ):
        if prompter is None:
            raise ValueError("Prompter is not provided. Please provide a prompter.")

        if prompt_path is None:
            prompt_path = "vlp/prompts/discovery/properties.txt"

        fallback_prompt = (
            "You are provided with images. Identify visual properties of objects.\n"
            f"Relevant objects: {objects}\n"
            f"Return exactly {n_min_properties} properties when possible.\n"
            "Output only:\n```python\nproperties = [...]\n```"
        )
        prompt_template = self._load_prompt_or_default(prompt_path, fallback_prompt)
        prompt = prompt_template.replace("{n}", str(n_min_properties))
        prompt = prompt.replace("{objects}", str(objects))
        prompt = self._already_discovered_header("properties") + prompt

        if self.symbolic:
            prompt = prompt + "\n\nScene descriptions:\n" + "\n".join(image_paths)
            response = self._prompt_with_text(prompt_text=prompt, max_new_tokens=1500)
        else:
            response = self._prompt_with_images(prompt_text=prompt, paths=image_paths, max_new_tokens=1500)
        original_response = response
        try:
            response = response.replace("\n", "")
            response = response.split("properties =")[-1]
            response = response.split("```")[0]
            response = response.split("[")[-1]
            response = response.split("]")[0]
            properties = eval(f"[{response}]")
        except Exception:
            try:
                properties = self._parse_incomplete_list(original_response)
            except Exception:
                properties = []

        properties = [x for x in properties if isinstance(x, str)]
        return self._dedupe_keep_order(properties)

    def _discover_actions(
        self,
        image_paths,
        objects,
        n_min_actions=None,
        prompter=None,
        prompt_path=None,
    ):
        if prompter is None:
            raise ValueError("Prompter is not provided. Please provide a prompter.")

        if prompt_path is None:
            prompt_path = "vlp/prompts/discovery/actions.txt"

        fallback_prompt = (
            "You are provided with images. Identify actions/relations in the scene.\n"
            f"Relevant objects: {objects}\n"
            f"Return exactly {n_min_actions} actions when possible.\n"
            "Output only:\n```python\nactions = [...]\n```"
        )
        prompt_template = self._load_prompt_or_default(prompt_path, fallback_prompt)
        prompt = prompt_template.replace("{n}", str(n_min_actions))
        prompt = prompt.replace("{objects}", str(objects))
        prompt = self._already_discovered_header("actions") + prompt

        if self.symbolic:
            prompt = prompt + "\n\nScene descriptions:\n" + "\n".join(image_paths)
            response = self._prompt_with_text(prompt_text=prompt, max_new_tokens=1500)
        else:
            response = self._prompt_with_images(prompt_text=prompt, paths=image_paths, max_new_tokens=1500)
        original_response = response
        try:
            response = response.replace("\n", "")
            response = response.split("actions =")[-1]
            response = response.split("```")[0]
            response = response.split("[")[-1]
            response = response.split("]")[0]
            actions = eval(f"[{response}]")
        except Exception:
            try:
                actions = self._parse_incomplete_list(original_response)
            except Exception:
                actions = []

        actions = [x for x in actions if isinstance(x, str)]
        return self._dedupe_keep_order(actions)

    def _variable_discovery(self, train_images):
        print(f"Discovering {self.n_objects} objects...")
        objects = self._discover_objects(
            train_images,
            n_min_properties=self.n_objects,
            prompter=self.prompter,
        )
        print(f"Discovered objects: {objects}")
        if len(objects) == 0:
            objects = []

        print("Discovering properties...")
        properties = self._discover_properties(
            train_images,
            objects,
            n_min_properties=self.n_properties,
            prompter=self.prompter,
        )
        print(f"Discovered properties: {properties}")

        if self.n_actions > 0:
            print("Discovering actions...")
            actions = self._discover_actions(
                train_images,
                objects,
                n_min_actions=self.n_actions,
                prompter=self.prompter,
            )
            print(f"Discovered actions: {actions}")
        else:
            actions = []

        variables = {
            "objects": [o for o in objects if isinstance(o, str)],
            "properties": [p for p in properties if isinstance(p, str)],
            "actions": [a for a in actions if isinstance(a, str)],
        }
        return variables

    @staticmethod
    def _add_variables_to_dsl(problem_semantics, problem_primitive_types, variables):
        objects = variables.get("objects", [])
        properties = variables.get("properties", [])
        actions = variables.get("actions", [])

        for obj in objects:
            if obj in problem_semantics:
                continue
            problem_semantics[obj] = obj
            problem_primitive_types[obj] = OBJECT

        for prop in properties:
            if prop in problem_semantics:
                continue
            problem_semantics[prop] = prop
            problem_primitive_types[prop] = PROPERTY

        for action in actions:
            if action in problem_semantics:
                continue
            problem_semantics[action] = action
            problem_primitive_types[action] = ACTION

        return problem_semantics, problem_primitive_types

    def _rebuild_dsl_from_variables(self):
        # Method exists so the `dsl` module import isn't shadowed by a `dsl` parameter.
        _get_dsl = get_vlp_dsl_symbolic if self.symbolic else get_vlp_dsl
        semantics, primitive_types = _get_dsl(self.prompter, self.discovered_variables, seed=self.seed)
        semantics, primitive_types = self._add_variables_to_dsl(
            semantics, primitive_types, self.discovered_variables
        )
        self.dsl = dsl.DSL(semantics, primitive_types, None)

    def _maybe_discover_variables(self):
        if self._discovery_count >= self.discovery_rounds:
            return
        if self.prompter is None:
            return

        # Get the input (image path or scene description) from the most recently observed example
        _example, path = self.examples[-1]
        if path is None:
            return
        input_repr = str(path)

        self._discovery_count += 1
        round_num = self._discovery_count
        try:
            variables = self._variable_discovery([input_repr])

            new_objects = [
                o for o in variables.get("objects", [])
                if o not in self.discovered_variables["objects"]
            ]
            new_properties = [
                p for p in variables.get("properties", [])
                if p not in self.discovered_variables["properties"]
            ]
            new_actions = [
                a for a in variables.get("actions", [])
                if a not in self.discovered_variables["actions"]
            ]

            if not new_objects and not new_properties and not new_actions:
                print(
                    f"[VLPZendoPlayer] Discovery round {round_num}/{self.discovery_rounds}"
                )
                return

            self.discovered_variables["objects"].extend(new_objects)
            self.discovered_variables["properties"].extend(new_properties)
            self.discovered_variables["actions"].extend(new_actions)

            self._rebuild_dsl_from_variables()

            print(
                f"[VLPZendoPlayer] Discovery round {round_num}/{self.discovery_rounds}: "
                f"+{len(new_objects)} objects, +{len(new_properties)} properties, "
                f"+{len(new_actions)} actions. "
                f"Totals: {self.discovered_variables}"
            )
        except Exception as e:
            print(
                f"[VLPZendoPlayer] Discovery round {round_num}/{self.discovery_rounds} "
                f"failed: {e}"
            )

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _vlp_examples(self):
        input_output, image_paths = zip(*self.examples)
        labels = [l for _, l in input_output]
        inputs = [inp for inp, _ in input_output]

        if self.symbolic:
            reprs = []
            for inp in inputs:
                if isinstance(inp, str):
                    reprs.append(inp)
                else:
                    try:
                        reprs.append(str(tensor_to_prolog_strings([inp])[0]))
                    except Exception:
                        reprs.append(str(inp))
        else:
            reprs = [str(p) if isinstance(p, Path) else p for p in image_paths]

        return [(repr_, label) for repr_, label in zip(reprs, labels)]

    def _search_programs_vlp(self, examples):
        dataset = task_set2zendodataset_vlp([["", examples]], self.model, self.dsl, self.cfg, self.zendo_cfg)
        candidates = [(None, 0.0, 0.0, 0, 0.0, 0.0, 0.0)]
        for t in range(len(examples)):
            required_accuracy = 1 - (t / len(examples))
            print(f"Gathering data for accuracy {required_accuracy:.2f}...")
            data = gather_data(dataset, 0, accuracy=required_accuracy, incorrect_rules=self.already_guessed)
            candidates = data[0][1]
            if candidates != [(None, 0.0, 0.0, 0, 0.0, 0.0, 0.0)]:
                break
        return candidates

    # ─────────────────────────────────────────────────────────────────────────

    def guess_label(self, input_scene):
        if self.last_label is not None:
            label = self.last_label
            self.last_label = None
            print(f"Using last proposed label: {label}")
            return label

        # symbolic: (tensor, scene_desc); image: (tensor, image_path).
        input_repr = input_scene[1] if self.symbolic else input_scene[0]

        examples = self._vlp_examples()
        candidates = self._search_programs_vlp(examples)
        top_rule = candidates[0][0]
        second_rule = candidates[1][0] if len(candidates) > 1 else None
        try:
            top_rule = strip_trailing_var0(top_rule)
            prog_fn = top_rule.eval(dsl=self.dsl, environment=(None, None), i=0)
            self.top_guess = top_rule

            if second_rule is not None:
                second_rule = strip_trailing_var0(second_rule)
                self.second_guess = second_rule

            return prog_fn(input_repr)

        except Exception as e:
            print(f"Error evaluating rule {top_rule}: {e}")
            top_rule = candidates[1][0]
            try:
                top_rule = strip_trailing_var0(top_rule)
                prog_fn = top_rule.eval(dsl=self.dsl, environment=(None, None), i=0)
                self.top_guess = top_rule
                return prog_fn(input_repr)
            except Exception as e:
                print(f"Error evaluating rule {top_rule} again: {e}")
                return False

    def guess_labels(self, input_paths):
        examples = self._vlp_examples()
        candidates = self._search_programs_vlp(examples)
        top_rule = candidates[0][0]
        labels = []
        for input_path in input_paths:
            try:
                top_rule = strip_trailing_var0(top_rule)
                prog_fn = top_rule.eval(dsl=self.dsl, environment=(None, None), i=0)
                labels.append(prog_fn(input_path))
            except Exception as e:
                print(f"Error evaluating rule {top_rule}: {e}")
                top_rule = candidates[1][0]
                try:
                    top_rule = strip_trailing_var0(top_rule)
                    prog_fn = top_rule.eval(dsl=self.dsl, environment=(None, None), i=0)
                    labels.append(prog_fn(input_path))
                except Exception as e:
                    print(f"Error evaluating rule {top_rule} again: {e}")
                    labels.append(False)
        return labels, top_rule

    def decide_guess(self, state):
        if self.guessing_stones <= 0 or len(self.examples) < self.min_examples:
            return None
        rule = self.guess_rule()
        if rule is None:
            print(f"Player {self.id} could not find a rule")
            return None
        self.guessing_stones -= 1
        print(f"Player {self.id} guessed rule: {rule}")
        return {"type": "guess_rule", "rule": rule}

    def wrong_rule(self, rule):
        # `rule` is a Zendo DSL Program (converted by the game master), not a VLP DSL
        # program. Duplicate-guess filtering uses self.already_guessed (VLP strings).
        if isinstance(rule, Program):
            normalized = normalize_rule(rule)
        else:
            normalized = str(rule)
        if normalized not in self.incorrect_rules:
            self.incorrect_rules.append(normalized)
        self.top_guess = None
        self.second_guess = None

    def quiz_incorrect(self):
        self.top_guess = None
        self.second_guess = None

    def guess_rule(self):
        if self.top_guess is not None:
            guess = self.top_guess
            self.top_guess = None
            guess_str = str(guess)
            norm = normalize_rule(guess)
            if norm not in self.already_guessed:
                self.already_guessed.append(norm)
            return guess_str
        examples = self._vlp_examples()
        candidates = self._search_programs_vlp(examples)
        candidate_rule, *_ = candidates[0]
        if candidate_rule is None:
            candidates = self._search_programs_vlp(examples)
            candidate_rule, *_ = candidates[0]
        if candidate_rule is None:
            print(f"Player {self.id} could not find any valid rule in the dataset.")
            return None
        guess_str = str(candidate_rule)
        norm = normalize_rule(guess_str)
        if norm not in self.already_guessed:
            self.already_guessed.append(norm)
        return guess_str

    def _react_propose(self, state):
        proposed_input, path, label, rule = self.propose_input()
        self.last_label = label
        amount_players = len(state.player_guess_tokens)
        if proposed_input is None:
            print("Failed to propose input, returning None.")
            return {"type": "propose_input", "input": None, "mode": "TELL"}
        mode = "QUIZ" if (label is not None and self.guessing_stones <= self.gs_threshold) or amount_players == 1 else "TELL"
        return {"type": "propose_input", "input": (proposed_input, path), "mode": mode, "rule": rule}

    def _is_valid_structure_tensor(self, structure: torch.Tensor):
        if not isinstance(structure, torch.Tensor):
            return False, "response is not a tensor"
        if structure.ndim != 2:
            return False, "tensor must be rank-2"
        if structure.shape[0] != 7:
            return False, f"tensor must have 7 rows, got {structure.shape[0]}"
        if structure.shape[1] not in (11, 15):
            return False, f"tensor must have 11 or 15 columns, got {structure.shape[1]}"

        # Non-padding rows should represent valid objects.
        non_pad_rows = 0
        for i in range(structure.shape[0]):
            row = structure[i]
            is_pad = (
                int(row[0].item()) == 7
                and int(row[1].item()) == 3
                and int(row[2].item()) == 3
                and int(row[3].item()) == 4
            )
            if is_pad:
                continue

            non_pad_rows += 1
            obj_id = int(row[0].item())
            color = int(row[1].item())
            shape = int(row[2].item())
            orientation = int(row[3].item())
            touching = row[4:10].tolist()
            pointing = int(row[10].item())

            if not (0 <= obj_id <= 6):
                return False, f"invalid object id {obj_id}"
            if color not in (0, 1, 2):
                return False, f"invalid color index {color}"
            if shape not in (0, 1, 2):
                return False, f"invalid shape index {shape}"
            if orientation not in (0, 1, 2, 3):
                return False, f"invalid orientation index {orientation}"
            # Index 8 is the buffer/none sentinel.
            if any((int(t) < 0 or int(t) > 8) for t in touching):
                return False, "invalid touching ids"
            if pointing not in (0, 1, 2, 3, 4, 5, 6, 7, 8):
                return False, f"invalid pointing id {pointing}"

        if non_pad_rows == 0:
            return False, "structure contains only padding"
        return True, ""

    def ensure_size(self, structure: torch.Tensor) -> torch.Tensor:
        """Ensure structure has shape (7, 15); right-pad short columns with -1."""
        rows, target_cols = 7, 15
        current_cols = structure.size(1)

        if current_cols < target_cols:
            pad_cols = target_cols - current_cols
            pad_tensor = torch.full((rows, pad_cols), -1, dtype=structure.dtype, device=structure.device)
            structure = torch.cat([structure, pad_tensor], dim=1)

        return structure

    def _parse_vlp_structure_response(self, response):
        print(f"Parsing VLP response: {response}")

        if response is None:
            return None, None, "empty model response"

        if isinstance(response, torch.Tensor):
            parsed_tensor = response
            parsed_label = None
        else:
            response_text = str(response).strip()
            if not response_text:
                return None, None, "empty text response"

            # Strip chat-template end-of-turn sentinels that leak through
            # some HF text-gen setups (Qwen, Llama-3, etc.). Without this the
            # trailing sentinel sits after the closing ``` and the fence
            # stripper below can't find it.
            for sentinel in ("<|im_end|>", "<|endoftext|>",
                             "<|eot_id|>", "<|end|>", "</s>"):
                response_text = response_text.replace(sentinel, "")
            response_text = response_text.strip()

            lines = response_text.splitlines()
            lines = [line for line in lines if not line.strip().startswith("#")]
            response_text = "\n".join(lines).strip()

            # Prefer extracting the first ```python ...``` block anywhere in
            # the response -- the model often prefixes the block with
            # commentary or chain-of-thought. Fall back to start/end stripping
            # if no fenced block is found.
            fenced = re.search(r"```(?:python)?\s*\n?(.*?)```",
                               response_text, flags=re.DOTALL)
            if fenced:
                response_text = fenced.group(1).strip()
            else:
                if response_text.startswith("```python"):
                    response_text = response_text.strip("`").split("python", 1)[-1].strip()
                elif response_text.startswith("```"):
                    response_text = response_text.strip("`").strip()
                if response_text.endswith("```"):
                    response_text = response_text.rsplit("```", 1)[0].strip()

            try:
                parsed = ast.literal_eval(response_text)
            except Exception:
                return None, None, f"response is not valid Python literal \n{response_text}\n"

            if not isinstance(parsed, list) or len(parsed) != 2:
                return None, None, "expected [[item(...), ...], label]"

            items, label = parsed
            if not isinstance(items, list) or not all(isinstance(s, str) for s in items):
                return None, None, "items must be a list of strings"
            if not isinstance(label, (int, bool)) or int(label) not in (0, 1):
                return None, None, "label must be 0 or 1"

            try:
                parsed_tensor = prolog_strings_to_tensor([items])[0]
            except Exception as e:
                return None, None, f"failed to parse items: {e}"
            parsed_label = int(label)

        parsed_tensor = self.ensure_size(parsed_tensor)
        is_valid, reason = self._is_valid_structure_tensor(parsed_tensor)
        if not is_valid:
            return None, None, reason
        return parsed_tensor, parsed_label, ""

    def guess_label(self, input_scene):
        image_path = str(input_scene[1]) if isinstance(input_scene[1], Path) else input_scene[1]

        if self.last_label is not None:
            label = self.last_label
            self.last_label = None
            return label

        examples = self._vlp_examples()
        candidates = self._search_programs_vlp(examples)
        top_rule = candidates[0][0]
        try:
            self.top_guess = top_rule
            return top_rule.eval_naive(self.dsl, [image_path])
        except Exception as e:
            print(f"Error evaluating rule {top_rule}: {e}")
            top_rule = candidates[1][0]
            try:
                top_rule = strip_trailing_var0(top_rule)
                prog_fn = top_rule.eval(dsl=self.dsl, environment=(None, None), i=0)
                self.top_guess = top_rule
                return prog_fn(image_path)
            except Exception as e:
                print(f"Error evaluating rule {top_rule} again: {e}")
                return False


    def propose_input(self, max_retries: int = 10):
        print(f"Proposing input based on {len(self.examples)} current examples...")

        examples = self._vlp_examples()
        candidates = self._search_programs_vlp(examples)

        valid_candidates = [
            prog
            for prog, *_ in candidates
            if normalize_rule(prog) not in self.incorrect_rules
        ]

        prompt, image_paths = self.build_zendo_prompt_from_examples(examples, valid_candidates[:2])
        # Track every proposal that failed to parse or render so we can feed
        # the full history back into the prompt. Showing only the most recent
        # response (the old behavior) plus a constant seed meant the model
        # kept regenerating the same broken scene -- both the prompter's
        # memory cache and the API's seeded sampling were pinned.
        failed_proposals: list[str] = []
        response = ""
        for attempt in range(1, max_retries + 1):
            current_prompt = _augment_prompt_with_failures(prompt, failed_proposals)
            if attempt > 1:
                current_prompt = (
                    f"{current_prompt}\n\n"
                    f"Attempt {attempt} of {max_retries}. Your previous answer "
                    f"was invalid or could not be rendered.\nPrevious answer: {response}\n"
                    "Pick a materially different structure and return ONLY one "
                    "valid python block in the required format."
                )

            # Per-attempt seed so the prompter's memory cache (keyed on
            # (prompt_text, seed)) and the API's seeded sampling both diverge.
            attempt_seed = self.seed + attempt - 1
            if self.symbolic:
                response = self._prompt_with_text(
                    prompt_text=current_prompt,
                    max_new_tokens=1500,
                    seed=attempt_seed,
                )
            else:
                response = self._prompt_with_images(
                    prompt_text=current_prompt,
                    paths=image_paths,
                    max_new_tokens=1500,
                    seed=attempt_seed,
                )
            structure, _, parse_error = self._parse_vlp_structure_response(response)
            if structure is None:
                print(f"Attempt {attempt}: invalid VLP structure response ({parse_error}). Retrying...")
                if response:
                    failed_proposals.append(str(response).strip())
                continue

            if self.symbolic:
                scene_desc = str(tensor_to_prolog_strings([structure])[0])
                print(f"Novel structure generated on attempt {attempt}.")
                top_label = None
                top_candidate = valid_candidates[0] if valid_candidates else None
                if top_candidate is not None:
                    try:
                        top_candidate = strip_trailing_var0(top_candidate)
                        top_label = top_candidate.eval_naive(dsl=self.dsl, environment=[scene_desc])
                        self.top_guess = top_candidate
                    except Exception as e:
                        print(f"Error evaluating top candidate {top_candidate}: {e}")
                self.last_label = top_label
                return structure, scene_desc, top_label, str(valid_candidates[0]) if valid_candidates else ""
            elif self.create_images:
                candidate_path = self._candidate_path()
                full_input_path = Path("generation") / Path("output") / (str(candidate_path) + ".png")
                full_input_path.parent.mkdir(parents=True, exist_ok=True)
                structure_items = tensor_to_prolog_strings([structure])[0]
                new_input_rendered = render_scene(structure_items, path=candidate_path)
                if new_input_rendered is not None:
                    print(f"Novel structure generated on attempt {attempt}.")
                    top_label = None
                    top_candidate = valid_candidates[0] if valid_candidates else None
                    if top_candidate is not None:
                        try:
                            top_candidate = strip_trailing_var0(top_candidate)
                            top_label = top_candidate.eval_naive(dsl=self.dsl, environment=[str(full_input_path)])
                            self.top_guess = top_candidate
                            return new_input_rendered, full_input_path, top_label, str(valid_candidates[0])
                        except Exception as e:
                            print(f"Error evaluating top candidate {top_candidate} on {full_input_path}: {e}")
                    else:
                        continue
                else:
                    print(f"Attempt {attempt}: Failed to render structure to image.")
                    # Remember the un-renderable structure so the next attempt's
                    # prompt explicitly tells the model not to repeat it.
                    failed_proposals.append(str(structure_items))
                    continue
            else:
                print(f"Novel structure generated on attempt {attempt}.")
                return structure, str(valid_candidates[0]), ""

        print("Failed to generate a novel structure after retries.")
        return None, None, "", None

    def build_zendo_prompt_from_examples(self, examples, top_rules):
        raw_paths = [t for t, _ in examples]
        labels = [l for _, l in examples]

        # The prompt below tells the model "first N are positive, next M are
        # negative", so the images must be sent in that order. Previously
        # image_paths kept the interleaved order they were collected in, so
        # the model's Pos/Neg indexing was off — visible in reasoning traces
        # where it mislabels which scenes are positive vs negative.
        positive_paths = [p for p, l in zip(raw_paths, labels) if l == 1]
        negative_paths = [p for p, l in zip(raw_paths, labels) if l != 1]
        image_paths = positive_paths + negative_paths

        positive_count = len(positive_paths)
        negative_count = len(negative_paths)

        top_rules_str = "\n".join(map(str, top_rules))

        base_prompt = f"""You are a Zendo player. Your job is to generate a new structure example to gain new knowledge about the hidden rule.
You are given a few positive and negative examples. Each structure consists of a list of items with the format:
"item(ID, color, shape, orientation, interaction)".

The pieces can have colors: red, blue, yellow; shapes: block, wedge, pyramid; orientations: upright, upside_down, flat, cheesecake, doorstop.
Wedges are never flat but instead can be doorstop or cheesecake, while the two other shapes can be flat but not cheesecake or doorstop.
Interactions can be: grounded, touching(ID), pointing(ID) and on_top_of(ID), where ID is the first field of another piece, e.g. "pointing(2)" means this piece is pointing to the piece with ID 2.
You may propose up to 7 pieces in your structure.
The current top rule hypotheses are:
{top_rules_str}

Positive examples: First {positive_count} examples are positive, meaning they follow the hidden rule, while the next {negative_count} examples are negative, meaning they do not follow the hidden rule.
"""

        if self.symbolic:
            pos_block = "\n".join(f"  {s}" for s in positive_paths) if positive_paths else "  (none)"
            neg_block = "\n".join(f"  {s}" for s in negative_paths) if negative_paths else "  (none)"
            examples_section = f"Positive examples:\n{pos_block}\n\nNegative examples:\n{neg_block}\n"
        else:
            examples_section = ""

        prompt = base_prompt + examples_section + """
Here are examples of valid answers:
```python
[["item(0, red, block, upright, grounded)", "item(1, blue, wedge, doorstop, touching(0))"], 1]
```
```python
[["item(0, yellow, pyramid, upside_down, grounded)", "item(1, blue, wedge, doorstop, on_top_of(0))", "item(2, red, block, upright, pointing(0))"], 0]
```
STRICT OUTPUT REQUIREMENTS — read carefully:
- Do NOT write any reasoning, analysis, commentary, hypotheses, or "let me think" prose.
- Do NOT discuss the positive/negative examples, the rule, or your guess.
- Do NOT echo the example answers above.
- Output EXACTLY ONE ```python fenced block and NOTHING else (no text before, no text after).
- The fenced block must contain a single Python literal in this format, where label is 1 for valid and 0 for invalid:

```python
[["item(ID, color, shape, orientation, interaction)", ...], label]
```
"""
        return prompt, image_paths


class VLPWeightedZendoPlayer(VLPZendoPlayer):
    """VLP player with two-stage VLM-weighted search over a macro-augmented DSL.

    Phase 1 (offline, see train_vlp_dsl.py) builds a macro library JSON file.
    Phase 2 (this class) loads the library, queries the VLM for "intuitions"
    about which predicates a task is likely about, and runs the heap search
    twice: first over a PCFG boosted toward those predicates, then (if no
    candidate clears the accuracy threshold) over the full unboosted PCFG.
    """

    def __init__(self, *args, macro_library_path=None,
                 vlp_weight_factor=10.0, vlp_weighted_budget=60.0,
                 vlp_intuition_examples=4, **kwargs):
        super().__init__(*args, **kwargs)
        from DSL.vlp_dsl_macros import MacroLibrary
        self._macro_library_path = macro_library_path
        self.macro_library = (
            MacroLibrary.load(macro_library_path)
            if macro_library_path else MacroLibrary.empty()
        )
        self.vlp_weight_factor = float(vlp_weight_factor)
        self.vlp_weighted_budget = float(vlp_weighted_budget)
        self.vlp_intuition_examples = int(vlp_intuition_examples)
        self._intuitions_by_task = {}
        self._base_pcfg = None
        self._boosted_pcfg = None
        self._cached_focus_predicates = None
        # Merge atoms learned during training into discovered_variables so
        # the search-time DSL exposes them as typed constants. Test-time
        # variable discovery (super()._maybe_discover_variables) continues
        # to run additively on top.
        for category in ("objects", "properties", "actions"):
            seen = list(self.discovered_variables.get(category, []))
            for name in self.macro_library.atoms.get(category, []):
                if name not in seen:
                    seen.append(name)
            self.discovered_variables[category] = seen
        self._rebuild_dsl_from_variables()

    def _rebuild_dsl_from_variables(self):
        _get_dsl = get_vlp_dsl_symbolic if self.symbolic else get_vlp_dsl
        semantics, primitive_types = _get_dsl(self.prompter, self.discovered_variables, seed=self.seed)
        semantics, primitive_types = self._add_variables_to_dsl(
            semantics, primitive_types, self.discovered_variables
        )
        # Augment with learned atoms, predicates, and macros.
        if getattr(self, "macro_library", None) is not None:
            self.macro_library.compile_into(semantics, primitive_types)
        self.dsl = dsl.DSL(semantics, primitive_types, None)
        # Discard cached PCFGs -- the DSL changed.
        self._base_pcfg = None
        self._boosted_pcfg = None
        self._cached_focus_predicates = None

    def _split_examples_for_intuition(self):
        positives = [(rep, lbl) for rep, lbl in self._vlp_examples() if lbl == 1]
        negatives = [(rep, lbl) for rep, lbl in self._vlp_examples() if lbl == 0]
        k = self.vlp_intuition_examples
        positives = positives[:k]
        negatives = negatives[:k]
        return positives, negatives

    def _query_intuitions(self):
        if self.task_idx in self._intuitions_by_task:
            return self._intuitions_by_task[self.task_idx]
        if self.prompter is None or self.symbolic:
            self._intuitions_by_task[self.task_idx] = {
                "focus_predicates": [], "focus_objects": [],
                "global_hints": [], "confidence": 0.0,
            }
            return self._intuitions_by_task[self.task_idx]

        from prompts.zendo.player import vlp_weighted_intuition_prompt

        positives, negatives = self._split_examples_for_intuition()
        if not positives or not negatives:
            self._intuitions_by_task[self.task_idx] = {
                "focus_predicates": [], "focus_objects": [],
                "global_hints": [], "confidence": 0.0,
            }
            return self._intuitions_by_task[self.task_idx]

        predicate_list = sorted(self.dsl.semantics.keys())
        variables = (
            self.discovered_variables.get("objects", [])
            + self.discovered_variables.get("properties", [])
            + self.discovered_variables.get("actions", [])
        )
        prompt = vlp_weighted_intuition_prompt.format(
            n_pos=len(positives),
            n_neg=len(negatives),
            predicate_list=", ".join(predicate_list),
            variable_list=", ".join(variables) or "(none discovered yet)",
        )
        paths = [p for p, _ in positives] + [p for p, _ in negatives]
        try:
            response = self._prompt_with_images(prompt_text=prompt, paths=paths,
                                                max_new_tokens=600)
        except Exception as e:
            print(f"[VLPWeighted] intuition query failed: {e}")
            self._intuitions_by_task[self.task_idx] = {
                "focus_predicates": [], "focus_objects": [],
                "global_hints": [], "confidence": 0.0,
            }
            return self._intuitions_by_task[self.task_idx]

        parsed = self._parse_intuition_response(response, predicate_list, variables)
        self._intuitions_by_task[self.task_idx] = parsed
        print(f"[VLPWeighted] task {self.task_idx} intuitions: {parsed}")
        return parsed

    @staticmethod
    def _parse_intuition_response(response, predicate_list, variable_list):
        empty = {"focus_predicates": [], "focus_objects": [],
                 "global_hints": [], "confidence": 0.0}
        if response is None:
            return empty
        text = str(response)
        for sentinel in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>", "</s>"):
            text = text.replace(sentinel, "")
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, flags=re.DOTALL)
        payload = match.group(1).strip() if match else text.strip()
        try:
            data = json.loads(payload)
        except Exception:
            # Tolerant fallback: try to find the first {...} block.
            brace = re.search(r"\{.*\}", payload, flags=re.DOTALL)
            if not brace:
                return empty
            try:
                data = json.loads(brace.group(0))
            except Exception:
                return empty
        focus_preds = [p for p in data.get("focus_predicates", [])
                       if isinstance(p, str) and p in predicate_list]
        focus_objs = [o for o in data.get("focus_objects", [])
                      if isinstance(o, str) and o in variable_list]
        hints = [h for h in data.get("global_hints", []) if isinstance(h, str)]
        try:
            conf = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        return {
            "focus_predicates": focus_preds,
            "focus_objects": focus_objs,
            "global_hints": hints,
            "confidence": max(0.0, min(1.0, conf)),
        }

    def _get_or_build_base_pcfg(self):
        if self._base_pcfg is not None:
            return self._base_pcfg
        # Build the same way run_vlp_tasks does.
        type_request = Arrow(IMG, BOOL)
        self._base_pcfg = self.dsl.DSL_to_CFG(
            type_request,
            max_program_depth=10,
            min_variable_depth=1,
            upper_bound_type_size=10,
            n_gram=2,
        )
        return self._base_pcfg

    def _get_or_build_boosted_pcfg(self, focus_predicates):
        if (self._boosted_pcfg is not None
                and self._cached_focus_predicates == tuple(sorted(focus_predicates))):
            return self._boosted_pcfg
        if not focus_predicates:
            return None
        import copy as _copy
        base = self._get_or_build_base_pcfg()
        boosted = _copy.deepcopy(base)
        focus_set = set(focus_predicates)
        w = self.vlp_weight_factor
        for S in boosted.rules:
            for P, (args, weight) in list(boosted.rules[S].items()):
                head_name = getattr(P, "primitive", None)
                if head_name in focus_set:
                    boosted.rules[S][P] = (args, weight * w)
        boosted.normalise()
        boosted.sort()
        self._boosted_pcfg = boosted
        self._cached_focus_predicates = tuple(sorted(focus_predicates))
        return boosted

    def _search_programs_vlp(self, examples):
        from experiments.run_experiment import run_algorithm
        from experiment_helper import make_program_checker_with_accuracy

        intuitions = self._query_intuitions()
        focus_predicates = intuitions.get("focus_predicates", [])

        checker = make_program_checker_with_accuracy(self.dsl, examples)
        empty = [(None, 0.0, 0.0, 0, 0.0, 0.0, 0.0)]

        # Stage A: boosted PCFG with a small budget. Stage A is skipped if the
        # VLM gave no focus_predicates -- in that case we fall straight through
        # to Stage B.
        candidates = empty
        boosted = self._get_or_build_boosted_pcfg(focus_predicates)
        if boosted is not None:
            print(f"[VLPWeighted] Stage A (boosted) on {focus_predicates}")
            try:
                stage_a = self._run_two_stage_pass(
                    checker, boosted, examples, max_seconds=self.vlp_weighted_budget,
                )
            except Exception as e:
                print(f"[VLPWeighted] Stage A failed, falling back: {e}")
                stage_a = empty
            if stage_a and stage_a[0][0] is not None and stage_a[0][5] >= 1.0 - 1e-9:
                return stage_a
            candidates = stage_a

        # Stage B: full unboosted PCFG, threshold-relaxing search like the base
        # VLPZendoPlayer does.
        print("[VLPWeighted] Stage B (full PCFG)")
        base_pcfg = self._get_or_build_base_pcfg()
        try:
            stage_b = self._run_two_stage_pass(
                checker, base_pcfg, examples, max_seconds=None,
            )
        except Exception as e:
            print(f"[VLPWeighted] Stage B failed: {e}")
            stage_b = empty

        if stage_b and stage_b[0][0] is not None:
            if (not candidates or candidates[0][0] is None
                    or stage_b[0][5] > candidates[0][5]):
                return stage_b
        return candidates if candidates and candidates[0][0] is not None else stage_b

    def _run_two_stage_pass(self, checker, pcfg, examples, max_seconds):
        from experiments.run_experiment import run_algorithm
        import experiments.run_experiment as _rx

        # Temporarily override the search timeout if a max_seconds is given.
        saved_timeout = getattr(_rx, "timeout", None)
        if max_seconds is not None:
            _rx.timeout = float(max_seconds)
        try:
            best = [(None, 0.0, 0.0, 0, 0.0, 0.0, 0.0)]
            for t in range(len(examples)):
                required_accuracy = 1 - (t / len(examples))
                data = run_algorithm(
                    checker, pcfg, 0, required_accuracy, self.already_guessed,
                )
                if data and data[0][0] is not None:
                    best = data
                    break
            return best
        finally:
            if max_seconds is not None and saved_timeout is not None:
                _rx.timeout = saved_timeout

