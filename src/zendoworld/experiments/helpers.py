from zendoworld.synthesis.dsl import zendo
from zendoworld.synthesis.cons_list import tuple2constlist
from zendoworld.synthesis.predictions.io_encodings import ZendoFixedSizeEncoding
from zendoworld.datasets.create_programs import convert_prolog_to_dsl
import torch
from zendoworld.synthesis.type_system import BOOL, IMG, INT, STRING, Arrow, Type
from zendoworld.synthesis.type_system import List as TypeList
import zendoworld.synthesis.type_system as type_system
from zendoworld.synthesis.predictions.models import RulesPredictor, BigramsPredictor
from zendoworld.synthesis.grammar.pcfg import PCFG
from typing import Callable, List, Tuple
from zendoworld.synthesis.grammar.dsl import DSL
from zendoworld.synthesis.program import BasicPrimitive, Function, Program, Variable

def make_program_checker(dsl: DSL, examples) -> Callable[[Program, bool], bool]:
    def checker(prog: Program, use_cached_evaluator: bool) -> bool:
        if len(examples) <=1:
            exit(1)
        if use_cached_evaluator:
            for i, example in enumerate(examples):
                input, output = example
                print(f"Checking example {i}: {input} -> {output}")
                print(len(input))
                out = prog.eval(
                    dsl=dsl,
                    environment=(input, None),
                    i=i
                )
                if output != out:
                    return False
            return True
        else:
            if len(examples) <=1:
                exit(1)
            for example in examples:
                input, output = example
                out = prog.eval_naive(
                    dsl=dsl,
                    environment=(input, None),
                    i=i
                )
                if output != out:
                    return False
            return True
    return checker

def make_program_checker_with_accuracy(
    dsl: DSL, examples
) -> Callable[[Program, bool], float]:
    def checker(prog: Program, use_cached_evaluator: bool) -> float:
        n_examples = len(examples)
        n_correct = 0
        if use_cached_evaluator:
            for i, example in enumerate(examples):
                input, output = example
                out = prog.eval(dsl, (input, None), i)
                if output == out:
                    n_correct += 1
        else:
            for example in examples:
                input, output = example
                out = prog.eval_naive(dsl, (input, None))
                if output == out:
                    n_correct += 1

        return n_correct / n_examples

    return checker

def _format_structure_tensor(t) -> str:
    """One-line-per-piece human-readable summary of an uncertainty-DSL input
    tensor of shape [max_objects, NUM_FEATURES=11, MAX_CLASSES=9].

    For each piece, prints presence and the argmax class + probability for
    color / shape / orientation / pointing / each touching direction. Absent
    pieces (presence < 0.05) collapse to a single line.
    """
    import torch.nn.functional as _F

    # Slot indices kept local so this helper doesn't import from DSL.
    COLOR_SLOT, SHAPE_SLOT, ORIENT_SLOT = 0, 1, 2
    TOUCHING_SLOTS = [3, 4, 5, 6, 7, 8]
    POINTING_SLOT = 9
    PRESENCE_SLOT = 10
    NUM_COLORS, NUM_SHAPES, NUM_ORIENT, NUM_TARGET = 4, 4, 5, 9
    COLOR_NAMES = ["red", "blue", "yellow", "extra"]
    SHAPE_NAMES = ["block", "wedge", "pyramid", "extra"]
    ORIENT_NAMES = ["upright", "ud", "flat", "cheese", "extra"]
    TARGET_NAMES = ["t0", "t1", "t2", "t3", "t4", "t5", "t6", "self", "NONE"]

    if not hasattr(t, "dim"):
        return repr(t)
    if t.dim() != 3:
        return f"    <non-3D structure tensor shape={tuple(t.shape)}>"

    def _top(slot, n):
        probs = _F.softmax(t[i, slot, :n], dim=-1)
        k = int(probs.argmax().item())
        return k, probs[k].item()

    lines = []
    for i in range(t.size(0)):
        pres = _F.softmax(t[i, PRESENCE_SLOT, :2], dim=-1)[1].item()
        if pres < 0.05:
            lines.append(f"    piece {i}: ABSENT (pres={pres:.2f})")
            continue
        c, cp = _top(COLOR_SLOT, NUM_COLORS)
        s, sp = _top(SHAPE_SLOT, NUM_SHAPES)
        o, op = _top(ORIENT_SLOT, NUM_ORIENT)
        pt, pp = _top(POINTING_SLOT, NUM_TARGET)
        touch_bits = []
        for d, slot in enumerate(TOUCHING_SLOTS):
            tk, tp = _top(slot, NUM_TARGET)
            touch_bits.append(f"t{d}={TARGET_NAMES[tk]}({tp:.2f})")
        lines.append(
            f"    piece {i}: pres={pres:.2f}  "
            f"color={COLOR_NAMES[c]}({cp:.2f})  "
            f"shape={SHAPE_NAMES[s]}({sp:.2f})  "
            f"orient={ORIENT_NAMES[o]}({op:.2f})  "
            f"point={TARGET_NAMES[pt]}({pp:.2f})  "
            f"touch=[{' '.join(touch_bits)}]"
        )
    return "\n".join(lines)


def make_program_checker_with_uncertainty(
    dsl: DSL,
    examples,
    threshold: float = 0.5,
    margin: float = 0.1,
) -> Callable[[Program, bool], Tuple[float, List[int], List[int]]]:
    """Program checker for the probabilistic (uncertainty) DSL.

    Each program evaluates to a score in [0, 1] (e.g. the rule-satisfaction
    probability produced by DSL/zendo_uncertainty.py). A score >= ``threshold``
    counts as a positive prediction.

    The returned checker, given a program, returns:
      - ``accuracy``: fraction of examples whose binary prediction matches the
        labeled output.
      - ``uncertain_indices``: example indices where the program's score is
        within ``margin`` of ``threshold`` — borderline calls that are good
        candidates for re-querying perception.
      - ``disagreement_indices``: example indices where the thresholded
        prediction does NOT match the labeled output. These are the program's
        outright mistakes on this example set.
    """
    def checker(prog: Program, use_cached_evaluator: bool) -> Tuple[float, List[int], List[int]]:
        n_examples = len(examples)
        if n_examples == 0:
            return 0.0, [], []
        n_correct = 0
        uncertain_indices: List[int] = []
        disagreement_indices: List[int] = []
        for i, example in enumerate(examples):
            input, output = example
            try:
                if use_cached_evaluator:
                    raw = prog.eval(dsl=dsl, environment=(input, None), i=i)
                else:
                    raw = prog.eval_naive(dsl=dsl, environment=(input, None))
            except Exception as e:
                # print(f"make_program_checker_with_uncertainty: eval failed on example {i}: {e}")
                raw = 0.0
            try:
                score = float(raw) if raw is not None else 0.0
            except (TypeError, ValueError):
                print(f"make_program_checker_with_uncertainty: could not convert output to float on example {i}: {raw}")
                score = 0.0
            prediction = score >= threshold
            if prediction == bool(output):
                n_correct += 1
            else:
                disagreement_indices.append(i)
            if abs(score - threshold) <= margin:
                uncertain_indices.append(i)
        return n_correct / n_examples, uncertain_indices, disagreement_indices
    return checker


def bias_pcfg_toward_depth_range(
    pcfg,
    target_min: int = 3,
    target_max: int = 5,
    strength: float = 4.0,
):
    """Reweight ``pcfg`` so enumeration prefers programs whose tree depth
    falls in ``[target_min, target_max]``.

    Each non-terminal in DeepSynth's CFG (see ``src/zendoworld/synthesis/grammar/dsl.py``) is encoded
    as ``(type, context, depth)`` — the third component is the tree depth
    at which this non-terminal sits. The enumerator orders programs by the
    product of production weights along their derivation, so making
    "keep going" productions cheap at shallow depths and "stop" productions
    cheap past ``target_max`` shifts the early candidates into the desired
    depth band.

    Per non-terminal at depth ``d``:
      * ``d < target_min`` → weight ∝ ``1 + strength * n_args`` (push deeper).
      * ``target_min ≤ d ≤ target_max`` → uniform weight.
      * ``d > target_max`` → terminal productions get ``1 + strength``, the
        rest stay at ``1`` (let recursion wind down).

    Weights are renormalised per LHS and the PCFG is re-sorted. ``strength=0``
    recovers the uniform PCFG. Mutates ``pcfg`` in place and returns it.
    """
    for S in pcfg.rules:
        try:
            depth = S[2]
        except (TypeError, IndexError):
            depth = 0
        productions = pcfg.rules[S]
        raw = {}
        for P, (args, _w) in productions.items():
            n_args = len(args) if args is not None else 0
            if depth < target_min:
                raw[P] = 1.0 + strength * n_args
            elif depth <= target_max:
                raw[P] = 1.0
            else:
                raw[P] = (1.0 + strength) if n_args == 0 else 1.0
        total = sum(raw.values())
        if total <= 0:
            continue
        for P in list(productions.keys()):
            args, _w = productions[P]
            productions[P] = (args, raw[P] / total)
    pcfg.normalise()
    pcfg.sort()
    return pcfg


def run_algorithm_with_uncertainty(
    is_correct_program: Callable[[Program, bool], Tuple[float, List[int], List[int]]],
    pcfg: PCFG,
    algo_index: int = 0,
    accuracy: float = 1.0,
    incorrect_rules=None,
    amount: int = 2,
    timeout: float = 300.0,
    total_number_programs: int = 100_000_000,
    use_cached_evaluator: bool = False,
) -> List[Tuple[Program, float, float, int, float, float, float, List[int], List[int]]]:
    """Enumerate programs and return the top ``amount`` by accuracy, expecting
    the checker returned by :func:`make_program_checker_with_uncertainty`.

    Search stops when ``timeout`` seconds have elapsed (search + evaluation
    combined) or when ``total_number_programs`` candidates have been examined.
    Best programs are still selected by accuracy as before.

    Each candidate carries:
      - ``uncertain_indices``: examples where the program's score was within
        ``margin`` of the threshold (borderline; re-query perception here).
      - ``disagreement_indices``: examples whose thresholded prediction did
        not match the labeled output (the program's outright mistakes).

    Returns: list of tuples
        (program, search_time, evaluation_time, nb_programs,
         cumulative_probability, program_accuracy, probability,
         uncertain_indices, disagreement_indices)
    """
    import time as _time
    import concurrent.futures as _futures
    # Lazy-import the algorithm helpers to keep this module light for callers
    # that don't run synthesis (and to avoid pulling ray on import).
    from zendoworld.synthesis.search_runner import (
        canonicalize_program,
        has_invalid_interaction,
        list_algorithms,
        normalize_program_structure,
        reconstruct,
    )
    from zendoworld.synthesis.program_as_list import reconstruct_from_compressed

    if incorrect_rules is None:
        incorrect_rules = []

    algorithm, _, param = list_algorithms[algo_index]
    n_candidates = amount
    search_time = 0.0
    evaluation_time = 0.0
    gen = algorithm(pcfg, **param)
    seen_programs: set = set()
    nb_programs = 0
    cumulative_probability = 0.0
    probability = 0.0
    program_candidates: List[Tuple[float, Tuple]] = []

    while (search_time + evaluation_time < timeout
           and nb_programs < total_number_programs):
        search_time -= _time.perf_counter()
        try:
            with _futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(next, gen)
                try:
                    program = future.result(timeout=5)
                except _futures.TimeoutError:
                    print(f"Generator timed out at program #{nb_programs}")
                    search_time += _time.perf_counter()
                    break
        except StopIteration:
            search_time += _time.perf_counter()
            break
        except Exception:
            search_time += _time.perf_counter()
            break

        search_time += _time.perf_counter()

        if program is None:
            break

        nb_programs += 1
        if algorithm in reconstruct:
            target_type = pcfg.start[0]
            program_r = reconstruct_from_compressed(program, target_type)
            probability = pcfg.probability_program(pcfg.start, program_r)
        else:
            probability = pcfg.probability_program(pcfg.start, program)
            program_r = program
        cumulative_probability += probability

        norm = normalize_program_structure(program_r)
        if isinstance(program_r, Function):
            head = program_r.function
            args = [normalize_program_structure(arg) for arg in program_r.arguments]
            if isinstance(head, BasicPrimitive) and (
                ((head.primitive == "MORE_THAN" or head.primitive == "gt?")
                 and args[0].typeless_eq(args[1]))
                or (head.primitive in {"even?", "odd?"}
                    and isinstance(args[0], BasicPrimitive)
                    and args[0].is_a_constant)
                or (head.primitive == "gt?"
                    and isinstance(args[0], BasicPrimitive) and args[0].is_a_constant
                    and isinstance(args[1], BasicPrimitive) and args[1].is_a_constant)
                or (head.primitive == "EITHER_OR" and args[0].typeless_eq(args[1]))
            ):
                continue
        canonical = canonicalize_program(norm)
        if str(canonical) in seen_programs or str(canonical) != str(norm):
            continue
        if str(canonical) in incorrect_rules:
            continue
        if has_invalid_interaction(canonical):
            continue
        seen_programs.add(str(canonical))

        evaluation_time -= _time.perf_counter()
        program_accuracy, uncertain_indices, disagreement_indices = is_correct_program(
            program_r, use_cached_evaluator
        )
        evaluation_time += _time.perf_counter()

        payload = (
            program_r,
            search_time,
            evaluation_time,
            nb_programs,
            cumulative_probability,
            program_accuracy,
            probability,
            uncertain_indices,
            disagreement_indices,
        )

        if len(program_candidates) < n_candidates:
            program_candidates.append((program_accuracy, payload))
        else:
            min_acc = min(acc for acc, _ in program_candidates)
            if program_accuracy > min_acc:
                worst = min(program_candidates, key=lambda x: x[0])
                program_candidates.remove(worst)
                program_candidates.append((program_accuracy, payload))
            elif program_accuracy == min_acc:
                # Tie-break on uncertainty: among candidates tied at min_acc,
                # find the one with the most examples within the threshold
                # margin (payload[7] = uncertain_indices). If the new program
                # has strictly fewer such borderline examples, swap it in.
                new_uncertain = len(uncertain_indices)
                tied = [
                    (acc, p) for acc, p in program_candidates if acc == min_acc
                ]
                most_uncertain = max(tied, key=lambda x: len(x[1][7]))
                if len(most_uncertain[1][7]) > new_uncertain:
                    program_candidates.remove(most_uncertain)
                    program_candidates.append((program_accuracy, payload))

        if (len(program_candidates) == n_candidates
                and all(acc >= accuracy for acc, _ in program_candidates)):
            print(f"\tFound {len(program_candidates)} high-accuracy programs, stopping search.")
            break

    # Sort by accuracy desc; tie-break by fewer uncertain examples first.
    program_candidates.sort(key=lambda x: (x[0], -len(x[1][7])), reverse=True)
    return [payload for _, payload in program_candidates]


def make_program_checker_with_constants(dsl: DSL, examples, constants) -> Callable[[Program, bool], Tuple[bool, Program]]:
    def checker(prog: Program, use_cached_evaluator: bool) -> Tuple[bool, Program]:
        programs = prog.make_all_constant_variations(constants)
        for fixed_prog in programs:
            failed = False
            if use_cached_evaluator:
                for i, example in enumerate(examples):
                    input, output = example
                    out = prog.eval(
                        dsl=dsl,
                        environment=(input, None),
                        i=i
                    )
                    if output != out:
                        failed = True
                        break
            else:
                for example in examples:
                    input, output = example
                    out = prog.eval_naive(
                        dsl=dsl,
                        environment=(input, None)
                    )
                    if output != out:
                        failed = True
                        break
            if not failed:
                return True, fixed_prog
        return False, None
    return checker

def task_set2dataset(tasks, model, dsl: DSL) -> List[Tuple[str, PCFG, Callable[[Program, bool], bool]]]:
    dataset = []
    batch_IOs = []
    batch_types = []
    for task in tasks:
        if len(task) == 3:
            name, examples, constants = task
        else:
            name, examples = task
            constants = None
        ex = [([i[0]], o) for i, o in examples]
        batch_IOs.append(ex)
        if isinstance(model, BigramsPredictor):
            batch_types.append(__get_type_request(examples))
    try:
        with torch.no_grad():
            grammars = model(batch_IOs)
    except AssertionError as e:
        print("src/zendoworld/experiments/helpers.py: task_set2dataset: An error occured while generating grammars:\n\t", e)
        return []
    if isinstance(model, RulesPredictor):
        grammars = model.reconstruct_grammars(grammars)
    if isinstance(model, BigramsPredictor):
        grammars = model.reconstruct_grammars(
            grammars, batch_types, tensors=False)
        grammars = [g.normalise() for g in grammars]
    for i, grammar in enumerate(grammars):
        name = tasks[i][0]
        examples = tasks[i][1]
        constants = None if len(tasks[i]) < 3 else tasks[i][2]
        dataset.append(
            (name, grammar, make_program_checker_with_constants(dsl, examples, constants) if constants else make_program_checker(dsl, examples)))
    return dataset

def merge_grammars(pcfg_old, cfg_ext):
    """Extend ``pcfg_old`` to ``cfg_ext``: shared productions keep learned weights, new ones stay uniform."""
    pcfg_new = cfg_ext.CFG_to_Uniform_PCFG()

    for S in pcfg_new.rules:
        if S not in pcfg_old.rules:
            continue

        for P in pcfg_new.rules[S]:
            if P in pcfg_old.rules[S]:
                args_old, w_old = pcfg_old.rules[S][P]
                pcfg_new.rules[S][P] = (args_old, w_old)

    pcfg_new.normalise()
    pcfg_new.sort()
    return pcfg_new

def task_set2zendodataset(tasks, model, dsl: DSL, cfg, use_model=True, uniform=True):
    dataset = []
    batch_IOs = []
    batch_types = []
    type_request = Arrow(TypeList(zendo.PIECE), BOOL)
    if use_model:
        print("Using model for Zendo dataset conversion.")
        for task in tasks:
            if len(task) == 3:
                name, examples, constants = task
            else:
                name, examples = task
                constants = None
            ex = examples
            batch_IOs.append(ex)
            if isinstance(model, BigramsPredictor):
                batch_types.append(type_request)

        try:
            with torch.no_grad():
                grammars = model(batch_IOs)
        except AssertionError as e:
            print("src/zendoworld/experiments/helpers.py: task_set2dataset: An error occurred while generating grammars:\n\t", e)
            return []

        if isinstance(model, RulesPredictor):
            grammars = model.reconstruct_grammars(grammars)
        if isinstance(model, BigramsPredictor):
            grammars = model.reconstruct_grammars(
                grammars, [model.cfg_dictionary.keys().__iter__().__next__()] * len(batch_IOs))
            grammars = [g.normalise() for g in grammars]
        grammars = [merge_grammars(g_old, cfg) for g_old in grammars]
        for i, grammar in enumerate(grammars):
            name = tasks[i][0]
            program = None
            if name != "":
                program = convert_prolog_to_dsl(name, cfg)
            examples = tasks[i][1]
            constants = None if len(tasks[i]) < 3 else tasks[i][2]

            checker_fn = (
                make_program_checker_with_constants(dsl, examples, constants)
                if constants else
                make_program_checker_with_accuracy(dsl, examples)
            )

            dataset.append((program, grammar, checker_fn))
    else:
        print("Using DSL to construct Zendo dataset without model.")
        type_request = Arrow(TypeList(zendo.PIECE), BOOL)
        cfg = dsl.DSL_to_CFG(type_request, max_program_depth=7)
        if not uniform:
            grammar = cfg.CFG_to_Random_PCFG()
        else:
            grammar = cfg.CFG_to_Uniform_PCFG()
        for task in tasks:
            if len(task) == 3:
                name, examples, constants = task
            else:
                name, examples = task
                constants = None
            checker_fn = (
                make_program_checker_with_constants(dsl, examples, constants)
                if constants else
                make_program_checker_with_accuracy(dsl, examples)
            )
            ex = examples
            batch_IOs.append(ex)
            program = None
            if name != "":
                program = convert_prolog_to_dsl(name, cfg)
            dataset.append((program, grammar, checker_fn))

    return dataset


def task_set2zendodataset_vlp(tasks, model, dsl: DSL, cfg, zendo_cfg, uniform=True):
    dataset = []
    batch_IOs = []
    batch_types = []

    print("Using DSL to construct Zendo dataset without model.")
    type_request = Arrow(IMG, BOOL)
    cfg = dsl.DSL_to_CFG(type_request, max_program_depth=7)
    if not uniform:
        grammar = cfg.CFG_to_Random_PCFG()
    else:
        grammar = cfg.CFG_to_Uniform_PCFG()
    for task in tasks:
        if len(task) == 3:
            name, examples, constants = task
        else:
            name, examples = task
            constants = None
        checker_fn = (
            make_program_checker_with_constants(dsl, examples, constants)
            if constants else
            make_program_checker_with_accuracy(dsl, examples)
        )
        ex = examples
        batch_IOs.append(ex)
        program = None
        if name != "":
            program = convert_prolog_to_dsl(name, zendo_cfg)
        dataset.append((program, grammar, checker_fn))

    return dataset


def to_cons_list(tensor):
    cons = None
    for i in reversed(range(tensor.shape[0])):
        cons = (tensor[i], cons)
    return cons


def task_set2uniform_dataset(tasks, dsl: DSL, max_program_depth: int = 4) -> List[Tuple[str, PCFG, Callable[[Program, bool], bool]]]:
    dataset = []
    for task in tasks:
        if len(task) == 3:
            name, examples, constants = task
        else:
            name, examples = task
            constants = None
        type_req = __get_type_request(examples)
        grammar = dsl.DSL_to_CFG(type_req, max_program_depth=max_program_depth).CFG_to_Uniform_PCFG()
        dataset.append(
            (name, grammar, make_program_checker_with_constants(dsl, examples, constants) if constants else make_program_checker(dsl, examples)))
    return dataset

def filter_examples(examples, nb_arguments_max, max_list_size, lexicon, verbose=False):
    filtered_examples = []
    one_output_is_nonempty = False
    for i, o in examples:

        if len(i) - 1 > nb_arguments_max:
            if verbose:
                print("\ttoo many arguments:", len(i) - 1, ">", nb_arguments_max)
            continue
        li = [x for x in i if hasattr(x, "__len__")]
        nli = [x for x in i if not hasattr(x, "__len__")]
        if any(len(x) > max_list_size for x in li):
            if verbose:
                print("\tinput iterable too long:", max(len(x) for x in li), ">", max_list_size)
            continue
        if any(any(el not in lexicon for el in x) for x in li):
            if verbose:
                print("\tinput iterable not in lexicon:", [
                    [el for el in x if el not in lexicon] for x in li])
            continue
        if hasattr(o, "__len__") and len(o) > max_list_size:
            if verbose:
                print("\toutput iterable too long:", len(o), ">", max_list_size)
            continue
        if hasattr(o, "__len__") and any(x not in lexicon for x in o):
            if verbose:
                print("\toutput iterable not in lexicon:",
                    [el for el in o if el not in lexicon])
            continue
        if any(x not in lexicon and x is not None for x in nli):
            if verbose:
                print("\tinput not in lexicon:", [x for x in nli if x not in lexicon])
            continue
        if not hasattr(o, "__len__") and o not in lexicon:
            if verbose:
                print("\toutput not in lexicon:", o)
            continue
        if not hasattr(o, "__len__") or len(o) > 0:
            one_output_is_nonempty = True
        filtered_examples.append((i, o))
    if one_output_is_nonempty:
        return filtered_examples
    return []


def __get_type__(el) -> Type:
    if isinstance(el, int):
        return INT
    elif isinstance(el, str):
        return STRING
    return type_system.List(INT)


def __get_type_request(examples):
    print("src/zendoworld/experiments/helpers.py: __get_type_request called with examples:", examples)
    input, output = examples
    type_req = __get_type__(output)
    for el in input[:-1][::-1]:
        type_req = Arrow(__get_type__(el), type_req)
    return type_req
