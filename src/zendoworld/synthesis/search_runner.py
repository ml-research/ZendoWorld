# This file contains code derived from:
# - https://github.com/nathanael-fijalkow/DeepSynth (MIT License)
# Original authors: Nathanaël Fijalkow
from zendoworld.synthesis.cons_list import cons_list2list
import typing
import tqdm
from zendoworld.synthesis.grammar.pcfg import PCFG
import logging
from zendoworld.synthesis.program import BasicPrimitive, Function, Lambda, New, Program, Variable, strip_trailing_var0
import time
import math
from typing import Callable, List, Tuple

from zendoworld.synthesis.algorithms.heap_search import heap_search
from zendoworld.synthesis.algorithms.threshold_search import threshold_search

from zendoworld.synthesis.program_as_list import reconstruct_from_compressed

logging_levels = {0: logging.INFO, 1: logging.DEBUG}


verbosity = 0
logging.basicConfig(format='%(message)s', level=logging_levels[verbosity])
total_number_programs = 100_000_000
# Set False to disable bottom-up cached evaluation for heap search.
use_heap_search_cached_eval = True

list_algorithms = [
    (heap_search, 'Heap Search', {}),
]
# Algorithms whose programs need to be reconstructed before evaluation.
reconstruct = {threshold_search}

def canonicalize_program(prog: Program) -> Program:
    if not isinstance(prog, Function):
        return prog

    fn = prog.function
    args = [canonicalize_program(arg) for arg in prog.arguments]

    # Collapse commutative-binary forms applied to identical args into their unary equivalents.
    if isinstance(fn, BasicPrimitive):
        name = fn.primitive
        if name in {"ODD_2", "EVEN_2", "ZERO_2"} and len(args) == 2:
            if args[0].typeless_eq(args[1]):
                if name == "ODD_2":
                    return Function(BasicPrimitive("ODD_1"), [args[0]])
                elif name == "EVEN_2":
                    return Function(BasicPrimitive("EVEN_1"), [args[0]])
                elif name == "ZERO_2":
                    return Function(BasicPrimitive("ZERO_1"), [args[0]])

        elif name in {"EXACTLY_2", "AT_LEAST_2"} and len(args) == 3:
            if args[1].typeless_eq(args[2]):
                if name == "AT_LEAST_2":
                    return Function(BasicPrimitive("AT_LEAST_1"), [args[0], args[1]])
                elif name == "EXACTLY_2":
                    return Function(BasicPrimitive("EXACTLY_1"), [args[0], args[1]])

    return Function(fn, args)

GROUND_PREDS = {"IS_GROUNDED", "IS_UNGROUNDED"}
INVALID_HEADS = {"EXCLUSIVELY", "ZERO_1", "ZERO_2"}

def contains_ground_predicate(prog: Program) -> bool:

    if isinstance(prog, BasicPrimitive):
        return prog.primitive in GROUND_PREDS

    if isinstance(prog, Function):

        if isinstance(prog.function, BasicPrimitive):
            if prog.function.primitive in GROUND_PREDS:
                return True

        return any(contains_ground_predicate(a) for a in prog.arguments)

    if isinstance(prog, Lambda):
        return contains_ground_predicate(prog.body)

    if isinstance(prog, New):
        return contains_ground_predicate(prog.body)

    return False


def has_invalid_interaction(prog: Program) -> bool:

    if isinstance(prog, Function):

        if isinstance(prog.function, BasicPrimitive):
            head = prog.function.primitive

            # Interaction rules and INVALID_HEADS may not contain ground/ungrounded preds.
            if head.endswith("_INTERACTION") or head in INVALID_HEADS:
                for arg in prog.arguments:
                    if contains_ground_predicate(arg):
                        return True

        return any(has_invalid_interaction(a) for a in prog.arguments)

    if isinstance(prog, Lambda):
        return has_invalid_interaction(prog.body)

    if isinstance(prog, New):
        return has_invalid_interaction(prog.body)

    return False

COMMUTATIVE_FUNCS = {"OR", "AND", "TOUCHING", "ODD_2", "EVEN_2", "EITHER_OR", "SAME_AMOUNT", "ZERO_2",
                     "and", "or", "eq?"}
# Maps function name -> N: only the last N arguments are commutative.
PARTIAL_COMMUTATIVE_FUNCS = {
    "EXACTLY_2": 2,
    "AT_LEAST_2": 2,
    "exists_properties": 2,
    "exists_object_with_properties": 2,
}
def normalize_program_structure(prog: Program) -> Program:
    if isinstance(prog, Function):
        head = prog.function
        args = [normalize_program_structure(arg) for arg in prog.arguments]

        if isinstance(head, BasicPrimitive):
            if head.primitive in COMMUTATIVE_FUNCS:
                args = sorted(args, key=lambda x: str(x))
            elif head.primitive in PARTIAL_COMMUTATIVE_FUNCS:
                n = PARTIAL_COMMUTATIVE_FUNCS[head.primitive]
                fixed = args[:-n]
                commutative = sorted(args[-n:], key=lambda x: str(x))
                args = fixed + commutative
        return Function(head, args)

    elif isinstance(prog, Lambda):
        return Lambda(normalize_program_structure(prog.body))

    elif isinstance(prog, New):
        return New(normalize_program_structure(prog.body))

    return prog


def run_algorithm(is_correct_program: Callable[[Program, bool], bool], pcfg: PCFG, algo_index: int, accuracy=1, incorrect_rules=None, amount=2, timeout=300, return_best_if_none=False) -> List[Tuple[Program, float, float, int, float, float, float]]:
    """Return up to ``amount`` best candidates within a cooperative wall-time budget.

    An in-flight generator/evaluator call may exceed the deadline. Run calls in
    this thread so a timed-out search cannot leave background threads mutating
    the shared grammar while the next search begins.
    """
    if amount < 1:
        raise ValueError("amount must be at least 1")
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("timeout must be finite and nonnegative")
    if not 0 <= accuracy <= 1:
        raise ValueError("accuracy must be between 0 and 1")
    incorrect_rules = set(incorrect_rules or ())
    deadline = time.monotonic() + timeout
    if timeout == 0:
        return []
    algorithm, name_algo, param = list_algorithms[algo_index]
    n_candidates = amount
    search_time = 0
    evaluation_time = 0
    gen = algorithm(pcfg, **param)
    seen_programs = set()
    if name_algo == "SQRT":
        _ = next(gen)
    nb_programs = 0
    cumulative_probability = 0
    cached_eval = use_heap_search_cached_eval and algorithm == heap_search
    probability = 0
    program_candidates = []
    try:
        while time.monotonic() < deadline and nb_programs < total_number_programs:
            started = time.perf_counter()
            try:
                program = next(gen)
            except StopIteration:
                break
            finally:
                search_time += time.perf_counter() - started

            if time.monotonic() >= deadline:
                break

            if program is None:
                logging.debug(
                    "Output the last program after {}".format(nb_programs))
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
            # Canonicalize from a var0-stripped COPY, matching src/zendoworld/game/player.py's
            # normalize_rule() -- which is what actually populates and re-checks
            # incorrect_rules. Without stripping, a candidate carrying that
            # artifact canonicalizes to a different string here than in
            # incorrect_rules, so this filter fails to recognize it as
            # already-rejected and lets it through, only for the caller's own
            # (correctly-stripped) re-check to discard it. Using a copy (rather
            # than stripping program_r in place, as strip_trailing_var0 mutates
            # its argument) matters: program_r is the object actually passed to
            # is_correct_program below and stored as the candidate -- mutating
            # it here would silently evaluate/return a truncated program
            # instead of the one heap_search actually generated.
            if isinstance(program_r, Function) and program_r.arguments:
                program_for_canon = strip_trailing_var0(Function(program_r.function, list(program_r.arguments)))
            else:
                program_for_canon = program_r
            norm = normalize_program_structure(program_for_canon)
            if isinstance(program_r, Function):
                head = program_r.function
                args = [normalize_program_structure(arg) for arg in program_r.arguments]
                # Skip programs whose result is trivially constant or a tautology.
                if isinstance(head, BasicPrimitive) and (
                    ((head.primitive == "MORE_THAN" or head.primitive == "gt?") and len(args) >= 2 and args[0].typeless_eq(args[1]))
                    or (head.primitive in {"even?", "odd?"} and len(args) >= 1 and isinstance(args[0], BasicPrimitive) and args[0].is_a_constant)
                    or (head.primitive == "gt?" and len(args) >= 2 and isinstance(args[0], BasicPrimitive) and args[0].is_a_constant
                        and isinstance(args[1], BasicPrimitive) and args[1].is_a_constant)
                or (head.primitive == "EITHER_OR" and len(args) >= 2 and args[0].typeless_eq(args[1]))
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
            if time.monotonic() >= deadline:
                break
            evaluation_time -= time.perf_counter()
            program_accuracy = is_correct_program(program_r, cached_eval)
            evaluation_time += time.perf_counter()

            if nb_programs % 100_000 == 0:
                logging.debug('tested {} programs'.format(nb_programs))

            if len(program_candidates) < n_candidates:
                program_candidates.append((program_accuracy, (
                    program_r,
                    search_time,
                    evaluation_time,
                    nb_programs,
                    cumulative_probability,
                    program_accuracy,
                    probability
                )))
            else:
                # Replace the least accurate candidate if this one beats it.
                min_acc = min([candidate[0] for candidate in program_candidates])
                if program_accuracy > min_acc:
                    program_candidates.remove(min(program_candidates, key=lambda x: x[0]))
                    program_candidates.append((program_accuracy, (
                        program_r,
                        search_time,
                        evaluation_time,
                        nb_programs,
                        cumulative_probability,
                        program_accuracy,
                        probability
                    )))

            if len(program_candidates) == n_candidates:
                if all(acc >= accuracy for acc, _ in program_candidates):
                    print("\tFound {} high-accuracy programs, stopping search.".format(len(program_candidates)))
                    break
    finally:
        close = getattr(gen, "close", None)
        if close is not None:
            close()

    program_candidates.sort(key=lambda x: x[0], reverse=True)
    qualifying = [candidate[1] for candidate in program_candidates if candidate[0] >= accuracy]
    if qualifying:
        return qualifying
    if return_best_if_none:
        return [candidate[1] for candidate in program_candidates]
    return []

def insert_prefix(prefix, prog):
    try:
        head, tail = prog
        return (head, insert_prefix(prefix, tail))
    except:
        return prefix


def reconstruct_from_list(program_as_list, target_type):
    if len(program_as_list) == 1:
        return program_as_list.pop()
    else:
        P = program_as_list.pop()
        if isinstance(P, (New, BasicPrimitive)):
            list_arguments = P.type.ends_with(target_type)
            arguments = [None] * len(list_arguments)
            for i in range(len(list_arguments)):
                arguments[len(list_arguments) - i - 1] = reconstruct_from_list(
                    program_as_list, list_arguments[len(
                        list_arguments) - i - 1]
                )
            return Function(P, arguments)
        if isinstance(P, Variable):
            return P
        assert False


def insert_prefix_toprog(prefix, prog, target_type):
    prefix = cons_list2list(prefix)
    return reconstruct_from_list([prog] + prefix, target_type)

def gather_data(dataset: typing.List[Tuple[str, PCFG, Callable]], algo_index: int, accuracy=1, incorrect_rules=None, amount=2, timeout=300, return_best_if_none=False) -> typing.List[Tuple[str, List[Tuple[Program, float, float, int, float, float, float]]]]:
    algorithm, _, _ = list_algorithms[algo_index]
    logging.info('\n## Running: %s' % algorithm.__name__)
    output = []
    successes = 0
    pbar = tqdm.tqdm(total=len(dataset))
    pbar.set_postfix_str(f"{successes} solved")
    try:
        for task_name, pcfg, is_correct_program in dataset:
            data = run_algorithm(
                is_correct_program, pcfg, algo_index, accuracy, incorrect_rules,
                amount, timeout=timeout, return_best_if_none=return_best_if_none,
            )
            if not data:
                print("\tsolution=", task_name)
                print("\ttype request=", pcfg.type_request())
                data = [(None, 0.0, 0.0, 0, 0.0, 0.0, 0.0)]
            if isinstance(task_name, Program):
                try:
                    prob = pcfg.probability_program(pcfg.start, task_name)
                    if data == [(None, 0.0, 0.0, 0, 0.0, 0.0, 0.0)]:
                        print("\tsolution probability=", prob)
                except KeyError as e:
                    print("Failed to compute probability of:", task_name)
                    print("Error:", e)
            successes += int(any(d[0] is not None and d[5] >= accuracy for d in data))
            output.append((task_name, data))
            pbar.update(1)
            pbar.set_postfix_str(f"{successes} solved")
    finally:
        pbar.close()
    return output

def gather_data_vlp(dataset, algo_index, accuracy=1, incorrect_rules=None, amount=2, timeout=300):
    """Compatibility entry point sharing gather_data's search and timeout handling."""
    return gather_data(dataset, algo_index, accuracy, incorrect_rules, amount, timeout)
