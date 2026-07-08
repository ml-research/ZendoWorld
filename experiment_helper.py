from DSL import zendo
from cons_list import tuple2constlist
from Predictions.IOencodings import ZendoFixedSizeEncoding
from data.create_programs import convert_prolog_to_dsl
import torch
from type_system import BOOL, IMG, INT, STRING, Arrow, Type
from type_system import List as TypeList
import type_system
from Predictions.models import RulesPredictor, BigramsPredictor
from grammar.pcfg import PCFG
from typing import Callable, List, Tuple
from grammar.dsl import DSL
from program import BasicPrimitive, Function, Program, Variable

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
        print("experiment_helper.py: task_set2dataset: An error occured while generating grammars:\n\t", e)
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
            print("experiment_helper.py: task_set2dataset: An error occurred while generating grammars:\n\t", e)
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
    print("experiment_helper.py: __get_type_request called with examples:", examples)
    input, output = examples
    type_req = __get_type__(output)
    for el in input[:-1][::-1]:
        type_req = Arrow(__get_type__(el), type_req)
    return type_req
