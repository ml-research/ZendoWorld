import zendoworld.serialization as legacy_pickle
from zendoworld.paths import RULES_FILE
import ast
import json
from pathlib import Path
import re
import shutil
import uuid
from collections import Counter
from zendoworld.datasets.create_prolog import dsl_to_prolog
from zendoworld.datasets.pieces2tensor import prolog_strings_to_tensor
from zendoworld.datasets.tensor2piece import tensor_to_prolog_strings
from zendoworld.rendering.render import render_artifact_path, render_scene
from zendoworld.synthesis.program import Program, strip_trailing_var0
import torch
from zendoworld.prompts.zendo import *

from zendoworld.game.conversions import extract_dsl_from_hypothesis
from zendoworld.game.player import call_prolog_subprocess_with_retries, normalize_rule

# ── Semantic equivalences ─────────────────────────────────────────────────────
# Rules that are logically identical but differ syntactically after normalize_rule.

_SEMANTIC_EQUIVALENT_PAIRS = {
    frozenset({"(ALL_1 IS_UPRIGHT)", "(EXCLUSIVELY IS_UPRIGHT)"}),
    # Both orderings included: normalize_rule sorts ODD_2 args (DOORSTOP < WEDGE).
    frozenset({"(ODD_2 IS_DOORSTOP IS_WEDGE)", "(ODD_1 IS_DOORSTOP)"}),
    frozenset({"(ODD_2 IS_WEDGE IS_DOORSTOP)", "(ODD_1 IS_DOORSTOP)"}),
}

# Any AND-tree of (AT_LEAST_1 1 IS_X) nodes covering this set ≡ the named atom.
_AND_EXPANSION_ATOMS = {
    frozenset({"IS_RED", "IS_BLUE", "IS_YELLOW"}): "ALL_THREE_COLORS",
    frozenset({"IS_BLOCK", "IS_WEDGE", "IS_PYRAMID"}): "ALL_THREE_SHAPES",
}


def _parse_and_expansion(s: str):
    ops = re.findall(r'\((\w+)', s)
    if not all(op in {"AND", "AT_LEAST_1"} for op in ops):
        return None
    preds = re.findall(r'AT_LEAST_1 1 (IS_\w+)', s)
    if len(preds) != len(set(preds)):
        return None
    return frozenset(preds)


def _canonical_atom(s: str):
    pred_set = _parse_and_expansion(s)
    if pred_set is not None:
        return _AND_EXPANSION_ATOMS.get(pred_set, s)
    return s


def _are_semantically_equivalent(s1: str, s2: str) -> bool:
    if frozenset({s1, s2}) in _SEMANTIC_EQUIVALENT_PAIRS:
        return True
    return _canonical_atom(s1) == _canonical_atom(s2)


class ZendoStateGameMaster:
    def __init__(self, true_program: Program, task_idx, dataset, paths, zendo_dsl, cfg, images=True, ask_for_counter=False, use_images=False, prompter=None, vlp=False, seed=1, symbolic=False):
        self.true_program = true_program
        self.dsl = zendo_dsl
        self.cfg = cfg
        self.seed = seed
        self.symbolic = symbolic
        self.remaining_examples = []
        self.paths = paths
        self.counter = 0
        self.task_idx = task_idx
        self.token_NONE = 8
        self.use_images = use_images
        self.ask_counter = ask_for_counter
        self.prompter = prompter
        self.program_cache = {}
        self.counter_description = None
        self.directions = ["left", "right", "front", "back", "top", "bottom"]
        self.vlp = vlp
        self.lexicons = {
            "color_lexicon": ["red", "blue", "yellow"],
            "shape_lexicon": ["block", "wedge", "pyramid"],
            "orientation_lexicon": ["upright", "upside_down", "flat", "cheesecake"],
        }
        self.create_images = images
        self._flat_dataset = None
        self._used_flat_example_paths = {str(path) for path in paths}
        print("Initializing GameMaster with dataset of size:", len(paths))
        for i, (tensor, label) in enumerate(dataset):
            try:
                pred = self.true_program.eval(dsl=self.dsl, environment=(tensor, None), i=i)(tensor)
                if pred != label:
                    print(
                        f"Label mismatch at index {i}: expected {label}, but got {pred} from true_program."
                    )
                label = bool(label)
            except Exception as e:
                raise ValueError(f"Failed to evaluate example {i}: {e}")
            self.remaining_examples.append(((tensor, label), self.paths[i]))
        print("GameMaster initialized with", len(self.remaining_examples), "examples.")

    def format_for_player(self, example):
        """Translate a ground-truth example ``((tensor, label), path)`` into the
        view that should be handed to a player. The ground-truth example itself
        is still kept by callers (e.g. in ``state.examples``) so saving the
        ``.pt`` archives preserves the tensor encoding."""
        if example is None or not self.use_images:
            return example
        (io_pair, path) = example
        if not (isinstance(io_pair, tuple) and len(io_pair) == 2):
            return example
        tensor_or_input, label = io_pair
        if self.vlp and self.symbolic:
            if tensor_or_input is not None:
                try:
                    scene_desc = str(tensor_to_prolog_strings([tensor_or_input])[0])
                except Exception:
                    scene_desc = str(path)
            else:
                scene_desc = str(path)
            # "counter_generated" sentinel keeps classify_label able to tag GM
            # counterexamples that have no real file path.
            path_for_state = str(path) if (path and str(path) not in ("", "None")) else "counter_generated"
            return ((scene_desc, bool(label)), path_for_state)
        if self.vlp:
            if path in (None, ""):
                return example
            path_str = str(path)
            return ((path_str, bool(label)), path_str)
        if self.use_images:
            if path in (None, ""):
                return example
            return ((None, bool(label)), str(path))
        return example

    def initial_examples(self):
        positives = [(ex, path) for ex, path in self.remaining_examples if ex[1] is True]
        negatives = [(ex, path) for ex, path in self.remaining_examples if ex[1] is False]

        if not positives or not negatives:
            raise ValueError("Not enough positive and negative examples to start.", self.remaining_examples)

        pos_example = positives[0]
        neg_example = negatives[0]

        def safe_remove(example):
            (target, target_path) = example
            for i, ((tensor, label), path) in enumerate(self.remaining_examples):
                if tensor is not None:
                    if torch.equal(tensor, target[0]) and label == target[1]:
                        del self.remaining_examples[i]
                        return
                else:
                    if path == target_path:
                        del self.remaining_examples[i]
                        return

        safe_remove(pos_example)
        safe_remove(neg_example)
        return [pos_example, neg_example]

    def initial_example(self):
        positives = [(ex, path) for ex, path in self.remaining_examples if ex[1] is True]

        if not positives:
            print(self.remaining_examples)
            raise ValueError("Not enough positive examples to start.")

        pos_example = positives[0]

        def safe_remove(example):
            (target, target_path) = example
            for i, ((tensor, label), path) in enumerate(self.remaining_examples):
                if tensor is not None:
                    if torch.equal(tensor, target[0]) and label == target[1]:
                        del self.remaining_examples[i]
                        return
                else:
                    if path == target_path:
                        del self.remaining_examples[i]
                        return

        safe_remove(pos_example)
        return pos_example

    def test_scenes(self):
        positives = [(ex, path) for ex, path in self.remaining_examples if ex[1] is True]
        negatives = [(ex, path) for ex, path in self.remaining_examples if ex[1] is False]

        if not positives or not negatives:
            print(self.remaining_examples)
            raise ValueError("Not enough positive and negative examples to start.")

        pos_examples = positives[:4]
        neg_examples = negatives[:4]

        def safe_remove(example):
            (target, target_path) = example
            for i, ((tensor, label), path) in enumerate(self.remaining_examples):
                if tensor is not None:
                    if torch.equal(tensor, target[0]) and label == target[1]:
                        del self.remaining_examples[i]
                        return
                else:
                    if path == target_path:
                        del self.remaining_examples[i]
                        return
        for pos_example in pos_examples:
            safe_remove(pos_example)
        for neg_example in neg_examples:
            safe_remove(neg_example)
        return pos_examples + neg_examples

    def get_next_example(self):
        print("Getting next example from remaining examples.", len(self.remaining_examples))
        if self.remaining_examples:
            return self.remaining_examples.pop(0)
        return self._get_next_flat_example()

    def _get_next_flat_example(self):
        """Draw an unused flat-dataset scene and label it with the true rule."""
        flat_dataset = self._load_flat_dataset()
        for entry in flat_dataset:
            if not isinstance(entry, (tuple, list)) or len(entry) != 3:
                continue
            tensor, _stored_label, img_path = entry
            path_key = str(img_path)
            if path_key in self._used_flat_example_paths:
                continue
            if self.use_images and (not img_path or not Path(img_path).exists()):
                print(f"Skipping flat example with missing image: {img_path}")
                self._used_flat_example_paths.add(path_key)
                continue
            try:
                label = bool(self.label_input(tensor))
            except Exception as e:
                print(f"ERROR: Skipping flat example due to evaluation error: {e}")
                self._used_flat_example_paths.add(path_key)
                continue
            self._used_flat_example_paths.add(path_key)
            print(f"Using flat-dataset example {img_path} with computed label {label}.")
            return ((tensor, label), img_path)
        print("No unused evaluable example remains in the flat dataset.")
        return None

    def label_input(self, tensor):
        try:
            strip_trailing_var0(self.true_program)
            program = self.true_program.eval(
                dsl=self.dsl,
                environment=(tensor, None),
                i=0
            )
            return program(tensor)
        except Exception as e:
            raise ValueError(f"Failed to evaluate input: {e}")

    def check_guess(self, guess):
        guess_converted = None
        if not isinstance(guess, Program):
            if self.prompter is not None:
                if guess in self.program_cache:
                    program = self.program_cache[guess]
                    guess_converted = program
                else:
                    program = extract_dsl_from_hypothesis(guess, self.prompter, self.cfg, self.seed)
                    if program is not None:
                        self.program_cache[guess] = program
                        guess_converted = program
                if guess_converted is None:
                    print(f"Failed to convert guess to program for hypothesis '{guess}'")
                    prompt = (
                        evaluate_rule_prompt.format(
                            rule1=guess,
                            rule2=self.true_program,
                        )
                    )
                    outputs = self.prompter.prompt_with_text(
                        prompt,
                        seed=self.seed,
                    )
                    response = outputs.strip()

                    parsed = extract_json(response)

                    if parsed is None:
                        print(f"Failed to parse LLM response: {response}")
                        self.counter_description = None
                        return False, guess

                    if parsed.get("equivalent") is True:
                        self.counter_description = None
                        return True, guess

                    counter = parsed.get("counterexample")

                    if counter is None:
                        print(f"Missing counterexample: {response}")
                        self.counter_description = None
                        return False, guess

                    print(f"LLM produced counterexample:\n{counter}")
                    self.counter_description = counter

                    return False, guess
                norm_true_program = normalize_rule(self.true_program)
                norm_guess = normalize_rule(guess_converted)
                is_correct = (str(norm_guess) == str(norm_true_program)
                              or _are_semantically_equivalent(str(norm_guess), str(norm_true_program)))
                return is_correct, guess_converted
            else:
                mode = input(f"Player guessed program: {guess}\n correct is {self.true_program}\n is it correct? (y/n): ").strip()
                if mode.lower() == 'y':
                    return True, guess
                else:
                    return False, guess

        norm_true_program = normalize_rule(self.true_program)
        norm_guess = normalize_rule(guess)
        is_correct = (str(norm_guess) == str(norm_true_program)
                      or _are_semantically_equivalent(str(norm_guess), str(norm_true_program)))
        return is_correct, guess

    def disprove_guess(self, guess):
        if isinstance(guess, Program):
            for i, ((tensor, _), _) in enumerate(self.remaining_examples):
                try:
                    strip_trailing_var0(guess)
                    strip_trailing_var0(self.true_program)
                    true_val = self.true_program.eval(dsl=self.dsl, environment=(tensor, None), i=i)
                    true_label = true_val(tensor)

                    guess_val = guess.eval(dsl=self.dsl, environment=(tensor, None), i=i)
                    guess_label = guess_val(tensor)

                    if guess_label and not true_label:
                        return self.remaining_examples.pop(i)

                    if not guess_label and true_label:
                        return self.remaining_examples.pop(i)

                except Exception as e:
                    print(f"ERROR: Skipping example due to evaluation error: {e}")
                    continue

            print("Guess could not be disproven with remaining examples.")
            result = self._search_flat_dataset(guess)
            if result is not None:
                return result
            return self.disprove_guess_via_prolog(guess)
        else:
            if self.prompter is not None and self.counter_description is not None:
                prompt = (
                    propose_counter_prompt.format(
                        description=self.counter_description
                    )
                )
                outputs = self.prompter.prompt_with_text(
                    prompt,
                    seed=self.seed,
                )
                cur_output = outputs
                xs = extract_labeled_structures(cur_output)
                print("LLM proposed counterexample:", xs)
                if len(xs) > 0:
                    structure = prolog_strings_to_tensor(xs)[0]
                    try:
                        strip_trailing_var0(self.true_program)
                        program = self.true_program.eval(
                            dsl=self.dsl,
                            environment=(structure, None),
                            i=0
                        )
                        label = program(structure)
                        if self.create_images:
                            prolog_strings = tensor_to_prolog_strings([structure])
                            path = Path(str(self.task_idx)) / Path(str(self.seed)) / Path("counter") / str(self.counter)
                            full_input_path = render_artifact_path(path)
                            true_input_rendered = render_scene(prolog_strings[0], path)
                            if true_input_rendered is not None:
                                self.counter += 1
                                self.counter_description = None
                                permanent_path = self._append_counterexample_to_flat_dataset(structure, label, full_input_path)
                                return ((structure, label), permanent_path)
                            else:
                                self.counter_description = None
                                return self.get_next_example()
                        self.counter_description = None
                        return ((structure, label), "")
                    except Exception as e:
                        print(f"Failed to evaluate counterexample: {e}")
                        self.counter_description = None
                        return self.get_next_example()
                else:
                    self.counter_description = None
                    return self.get_next_example()

            print(f"Player guessed program: {guess}, correct is {self.true_program} \n give counter example")
            if self.ask_counter:
                structure, label = self.ask_for_counter(guess)
                if self.create_images:
                    prolog_strings = tensor_to_prolog_strings([structure])
                    path = Path(str(self.task_idx)) / Path(str(self.seed)) / Path("counter") / str(self.counter)
                    full_input_path = render_artifact_path(path)
                    true_input_rendered = render_scene(prolog_strings[0], path)
                    if true_input_rendered is not None:
                        self.counter += 1
                        permanent_path = self._append_counterexample_to_flat_dataset(structure, label, full_input_path)
                        return ((structure, label), permanent_path)
                    else:
                        return self.get_next_example()
                return ((structure, label), "")
            else:
                return self.get_next_example()

    def _load_flat_dataset(self):
        # Atomic replacement by writers means readers see either the old or
        # new complete dataset. Reload so one game can observe additions made
        # by another worker instead of retaining a stale per-game snapshot.
        flat_path = Path("dataset_flat") / "dataset.pt"
        if flat_path.exists():
            self._flat_dataset = torch.load(flat_path, weights_only=False, pickle_module=legacy_pickle)
            print(f"Loaded flat dataset with {len(self._flat_dataset)} entries.")
        else:
            self._flat_dataset = []
            print("No flat dataset found at dataset_flat/dataset.pt")
        return self._flat_dataset

    def _append_counterexample_to_flat_dataset(self, tensor, label: bool, rendered_img_path: Path) -> Path:
        """Persist a uniquely named image and append to the shared flat dataset."""
        import fcntl
        import os

        img_dir = Path("dataset") / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        permanent_path = img_dir / f"{uuid.uuid4().hex}.png"
        shutil.copy2(rendered_img_path, permanent_path)

        flat_path = Path("dataset_flat") / "dataset.pt"
        flat_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = flat_path.with_suffix(flat_path.suffix + ".lock")
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if flat_path.exists():
                flat_dataset = torch.load(flat_path, weights_only=False, pickle_module=legacy_pickle)
            else:
                flat_dataset = []
            flat_dataset.append((tensor, label, permanent_path))
            tmp_path = flat_path.with_name(
                f".{flat_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                torch.save(flat_dataset, tmp_path)
                os.replace(tmp_path, flat_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
            self._flat_dataset = flat_dataset

        print(f"Appended counterexample to flat dataset ({len(flat_dataset)} total). Image: {permanent_path}")
        return permanent_path

    def _search_flat_dataset(self, guess):
        flat_dataset = self._load_flat_dataset()
        for tensor, _label, img_path in flat_dataset:
            try:
                strip_trailing_var0(guess)
                strip_trailing_var0(self.true_program)
                true_label = self.true_program.eval(dsl=self.dsl, environment=(tensor, None), i=0)(tensor)
                guess_label = guess.eval(dsl=self.dsl, environment=(tensor, None), i=0)(tensor)
                if bool(true_label) != bool(guess_label):
                    label = bool(true_label)
                    return ((tensor, label), img_path)
            except Exception as e:
                print(f"ERROR: Skipping flat dataset entry due to evaluation error: {e}")
                continue
        print("No counterexample found in flat dataset.")
        return None

    def disprove_guess_via_prolog(self, guess_program, max_attempts=2):
        true_prolog = dsl_to_prolog(self.true_program)
        guess_prolog = dsl_to_prolog(guess_program)

        strip_trailing_var0(guess_program)
        strip_trailing_var0(self.true_program)

        # Primary: Prolog XOR discriminator finds a structure where the two rules disagree.
        discrim_query = f"generate_discriminating_structure([{true_prolog}], [{guess_prolog}], Structure)"
        print(f"Trying Prolog XOR discriminator: true vs guess")
        for attempt in range(max_attempts):
            scene = call_prolog_subprocess_with_retries(1, discrim_query, str(RULES_FILE))
            if scene is None:
                continue
            scene = scene[0]
            if scene is None:
                continue

            try:
                tensor_input = prolog_strings_to_tensor([scene])[0]
            except Exception as e:
                print(f"Failed to convert discriminating scene to tensor: {e}")
                continue

            try:
                out_true = self.true_program.eval(
                    dsl=self.dsl, environment=(tensor_input, None), i=0
                )(tensor_input)
            except Exception as e:
                print(f"Evaluation error on discriminating scene: {e}")
                continue

            label = bool(out_true)

            if self.symbolic:
                return ((tensor_input, label), "")
            if self.create_images:
                path = Path(str(self.task_idx)) / Path(str(self.seed)) / Path("counter") / str(self.counter)
                full_input_path = render_artifact_path(path)
                rendered = render_scene(scene, path)
                if rendered is None:
                    print(f"Render failed on discriminating attempt {attempt + 1}, generating new scene...")
                    continue
                self.counter += 1
                permanent_path = self._append_counterexample_to_flat_dataset(tensor_input, label, full_input_path)
                return ((tensor_input, label), permanent_path)
            else:
                return ((tensor_input, label), "")

        print("XOR discriminator exhausted, falling back to per-rule strategies...")

        strategies = [
            (
                "accepted by guess, rejected by true_program",
                f"generate_valid_structure([{guess_prolog}], Structure)",
                lambda out_true, out_guess: out_guess and not out_true,
                False,
            ),
            (
                "accepted by true_program, rejected by guess",
                f"generate_valid_structure([{true_prolog}], Structure)",
                lambda out_true, out_guess: out_true and not out_guess,
                True,
            ),
            (
                "rejected by guess, accepted by true_program",
                f"generate_invalid_structure([{guess_prolog}], Structure)",
                lambda out_true, out_guess: out_true and not out_guess,
                True,
            ),
        ]

        for desc, query, is_counter, label in strategies:
            print(f"Try to find example {desc}")
            for attempt in range(max_attempts):
                scene = call_prolog_subprocess_with_retries(1, query, str(RULES_FILE))
                if scene is None:
                    continue
                scene = scene[0]
                if scene is None:
                    continue

                try:
                    tensor_input = prolog_strings_to_tensor([scene])[0]
                except Exception as e:
                    print(f"Failed to convert scene to tensor: {e}")
                    continue

                try:
                    out_true = self.true_program.eval(
                        dsl=self.dsl, environment=(tensor_input, None), i=0
                    )(tensor_input)
                    out_guess = guess_program.eval(
                        dsl=self.dsl, environment=(tensor_input, None), i=0
                    )(tensor_input)
                except Exception as e:
                    print(f"Evaluation error on attempt {attempt + 1}: {e}")
                    continue

                if not is_counter(out_true, out_guess):
                    continue

                if self.symbolic:
                    return ((tensor_input, label), "")
                if self.create_images:
                    path = Path(str(self.task_idx)) / Path(str(self.seed)) / Path("counter") / str(self.counter)
                    full_input_path = render_artifact_path(path)
                    rendered = render_scene(scene, path)
                    if rendered is None:
                        print(f"Render failed on attempt {attempt + 1}, generating new scene...")
                        continue
                    self.counter += 1
                    permanent_path = self._append_counterexample_to_flat_dataset(tensor_input, label, full_input_path)
                    return ((tensor_input, label), permanent_path)
                else:
                    return ((tensor_input, label), "")
        if self.ask_counter:
            structure, label = self.ask_for_counter(guess_program)
            if structure[0][1] == 3:
                print("Player provided padded structure, ignoring.")
                return None
            if self.create_images:
                prolog_strings = tensor_to_prolog_strings([structure])
                path = Path(str(self.task_idx)) / Path(str(self.seed)) / Path("counter") / str(self.counter)
                full_input_path = render_artifact_path(path)
                true_input_rendered = render_scene(prolog_strings[0], path)
                if true_input_rendered is not None:
                    self.counter += 1
                    permanent_path = self._append_counterexample_to_flat_dataset(structure, label, full_input_path)
                    return ((structure, label), permanent_path)
                else:
                    return self.get_next_example()
            return ((structure, label), "")
        else:
            # Mirror disprove_guess()'s natural-language-guess fallback (line ~386):
            # rather than giving up (which the caller reads as "guess unrefuted,
            # treat as correct"), hand back the next example from the remaining
            # pool. It's not guaranteed to actually discriminate the guess from
            # the true rule, but it keeps the game going on an unattended run
            # instead of either silently crediting a wrong guess as a win or
            # blocking on ask_for_counter's interactive input.
            return self.get_next_example()

    def ask_for_counter(self, guessed):
        if str(guessed) == "(AT_LEAST_INTERACTION 1 (ON_TOP_OF IS_YELLOW IS_WEDGE))":
            structure = torch.tensor([[ 0,  2,  0,  0,  8,  8,  8,  8,  8,  1,  8, -1, -1, -1, -1],
                [ 1,  1,  1,  2,  8,  8,  8,  8,  0,  8,  8, -1, -1, -1, -1],
                [ 7,  3,  3,  4,  7,  7,  7,  7,  7,  7,  7, -1, -1, -1, -1],
                [ 7,  3,  3,  4,  7,  7,  7,  7,  7,  7,  7, -1, -1, -1, -1],
                [ 7,  3,  3,  4,  7,  7,  7,  7,  7,  7,  7, -1, -1, -1, -1],
                [ 7,  3,  3,  4,  7,  7,  7,  7,  7,  7,  7, -1, -1, -1, -1],
                [ 7,  3,  3,  4,  7,  7,  7,  7,  7,  7,  7, -1, -1, -1, -1]], dtype=torch.long)
            label = False
            print("Auto counterexample for known guess:", structure, label)
            return structure, label
        if str(guessed) == "(ODD_INTERACTION (ON_TOP_OF IS_YELLOW IS_WEDGE))":
            structure = torch.tensor([[ 0,  2,  0,  0,  8,  8,  8,  8,  8,  1,  8, -1, -1, -1, -1],
                [ 1,  1,  1,  3,  8,  8,  8,  8,  0,  8,  8, -1, -1, -1, -1],
                [ 2,  2,  2,  2,  8,  8,  8,  8,  8,  3,  8, -1, -1, -1, -1],
                [ 3,  0,  1,  3,  8,  8,  8,  8,  2,  8,  8, -1, -1, -1, -1],
                [ 7,  3,  3,  4,  7,  7,  7,  7,  7,  7,  7, -1, -1, -1, -1],
                [ 7,  3,  3,  4,  7,  7,  7,  7,  7,  7,  7, -1, -1, -1, -1],
                [ 7,  3,  3,  4,  7,  7,  7,  7,  7,  7,  7, -1, -1, -1, -1]], dtype=torch.long)
            label = True
            print("Auto counterexample for known guess:", structure, label)
            return structure, label
        if str(guessed) == "(ODD_INTERACTION (ON_TOP_OF IS_YELLOW IS_CHEESECAKE))":
            structure = torch.tensor([[ 0,  2,  0,  0,  8,  8,  8,  8,  8,  1,  8, -1, -1, -1, -1],
                [ 1,  0,  1,  3,  8,  8,  8,  8,  0,  8,  8, -1, -1, -1, -1],
                [ 2,  2,  2,  0,  8,  8,  8,  8,  8,  3,  8, -1, -1, -1, -1],
                [ 3,  1,  1,  3,  8,  8,  8,  8,  2,  8,  8, -1, -1, -1, -1],
                [ 7,  3,  3,  4,  7,  7,  7,  7,  7,  7,  7, -1, -1, -1, -1],
                [ 7,  3,  3,  4,  7,  7,  7,  7,  7,  7,  7, -1, -1, -1, -1],
                [ 7,  3,  3,  4,  7,  7,  7,  7,  7,  7,  7, -1, -1, -1, -1]], dtype=torch.long)
            label = True
            print("Auto counterexample for known guess:", structure, label)
            return structure, label
        if str(guessed) == "(AT_LEAST_2 3 IS_VERTICAL IS_WEDGE)":
            structure = torch.tensor([[ 0,  2,  1,  0,  1,  8,  8,  8,  8,  8,  8, -1, -1, -1, -1],
                [ 1,  0,  1,  1,  2,  0,  8,  8,  0,  8,  8, -1, -1, -1, -1],
                [ 2,  2,  1,  0,  8,  1,  8,  8,  8,  3,  8, -1, -1, -1, -1],
                [ 7,  3,  3,  4,  7,  7,  7,  7,  7,  7,  7, -1, -1, -1, -1],
                [ 7,  3,  3,  4,  7,  7,  7,  7,  7,  7,  7, -1, -1, -1, -1],
                [ 7,  3,  3,  4,  7,  7,  7,  7,  7,  7,  7, -1, -1, -1, -1],
                [ 7,  3,  3,  4,  7,  7,  7,  7,  7,  7,  7, -1, -1, -1, -1]], dtype=torch.long)
            label = False
            print("Auto counterexample for known guess:", structure, label)
            return structure, label
        print(f"Player guessed program: {guessed}, correct is {self.true_program}")
        print("\a", end="", flush=True)
        print("\nNow enter a new input piece-by-piece (max 8 pieces). Leave blank to finish early.")
        pieces = []
        i = 0
        while i < 8:
            raw = input(f"---Piece {i} [format: color,shape,orientation]: ").strip()
            if not raw:
                break
            try:
                color_str, shape_str, orient_str = map(str.strip, raw.split(","))
                color = self.lexicons['color_lexicon'].index(color_str)
                shape = self.lexicons['shape_lexicon'].index(shape_str)
                if orient_str == "doorstop":
                    orientation = 2
                else:
                    orientation = self.lexicons['orientation_lexicon'].index(orient_str)

                touching = [self.token_NONE] * 6
                for d, dir_name in enumerate(self.directions):
                    val = input(f"  ↳ touching on {dir_name} (target ID or blank): ").strip()
                    if val.isdigit():
                        touching[d] = int(val)

                pointing = input("  ↳ pointing at (target ID or blank): ").strip()
                pointing_val = int(pointing) if pointing.isdigit() else self.token_NONE

                # Piece ID 7 collides with the padding sentinel that valid_pieces()
                # (src/zendoworld/synthesis/dsl/zendo.py, src/zendoworld/synthesis/dsl/zendo_extended.py) checks for everywhere, so an
                # 8th piece would be silently treated as non-existent during rule
                # evaluation. Touching/pointing/on_top_of are resolved by tensor
                # position (see src/zendoworld/synthesis/dsl/zendo.py's structure_tensor[idx] lookups and
                # tensor2piece.py's position-keyed id_map), not by this ID column, so
                # giving the 8th piece (position 7) a different, unused ID here is
                # safe and invisible to how you reference it from other pieces --
                # still just enter 7 as the target ID for this piece.
                self_id = 9 if i == 7 else i
                piece = torch.tensor([self_id, color, shape, orientation] + touching + [pointing_val] + [-1]*4, dtype=torch.int64)
                print(f"Piece {i} created: {piece}")
                pieces.append(piece)
                i += 1
            except Exception as e:
                print("Invalid input. Please try again.", e)
                continue

        if any(len(p) != 15 for p in pieces):
            print("One or more entered pieces have incorrect length.")
            return None

        while len(pieces) < 7:
            pad = torch.tensor([7, 3, 3, 4] + [7]*7 + [-1]*4, dtype=torch.int64)
            pieces.append(pad)

        structure = torch.stack(pieces)
        label = input("Is this a positive example? (y/n): ").strip().lower() == 'y'
        print("Structure created:", structure, "Label:", label)
        return structure, label

def extract_labeled_structures(text):
    results = []

    pattern = re.compile(
        r'\[\s*(?:".*?"\s*,?\s*)*\]',
        re.DOTALL
    )

    matches = pattern.findall(text)

    for match in matches:
        try:
            parsed = ast.literal_eval(match)
        except Exception:
            continue

        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            results.append(parsed)

    return results

def extract_json(response: str):
    try:
        return json.loads(response)
    except:
        pass

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return None
