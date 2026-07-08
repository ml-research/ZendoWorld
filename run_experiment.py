"""Unified Zendo experiment runner.

Usage:
    python run_experiment.py --player zendo --seeds 1 2 3
    python run_experiment.py --player vlp --threads 4 --seeds 1
    python run_experiment.py --player scientist --threads 2 --seeds 1 2 3
    python run_experiment.py --player vlp --threads 4 --tasks 0 2 5 6

Append ``_symbolic`` to the player name to run with symbolic inputs.
"""

import argparse
import atexit
import gc
import os
import pickle
import json
import shutil
import time
import traceback
from datetime import datetime
from models.prompters import get_prompter
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import compute_log

try:
    from rtpt import RTPT as _RTPT
except ImportError:
    _RTPT = None


# ── Shared helpers ────────────────────────────────────────────────────────────


def load_zendo_dataset(pkl_path="data/combined_dataset_short.pkl"):
    with open(pkl_path, "rb") as f:
        tasks = pickle.load(f)
    return tasks


def classify_label(src_path: Path) -> str:
    s = src_path.as_posix().lower()
    return "gm" if ("zendo-synthetic-data" in s or "counter" in s) else "player"


def copy_and_rename(paths_list, examples_tensor, images_out_dir):
    new_paths = []
    for idx, src in enumerate(paths_list):
        _, example_label = (
            examples_tensor[idx]
            if idx < len(examples_tensor)
            else ("unknown", "unknown")
        )
        src_path = Path(src)
        label = classify_label(src_path)
        ext = (src_path.suffix or ".png").lower()
        dest_path = images_out_dir / f"{label}_{example_label}_{idx}{ext}"
        try:
            shutil.copy2(src_path, dest_path)
        except Exception as e:
            print(f"Warning: failed to copy {src_path} -> {dest_path}: {e}")
        new_paths.append(str(dest_path))
    return new_paths


def save_state(state, output_dir, task_idx, seed, player=None):
    iteration_dir = output_dir / f"seed_{seed}"
    iteration_dir.mkdir(parents=True, exist_ok=True)

    state_path = iteration_dir / f"task_{task_idx}_state.json"
    example_path = iteration_dir / f"examples_{task_idx}.pt"

    state_dict, examples_tensor, bramley_examples_tensor = state.to_dict()

    images_out_dir = iteration_dir / f"task_{task_idx}_images"
    images_out_dir.mkdir(parents=True, exist_ok=True)
    state_dict["paths"] = copy_and_rename(
        state_dict.get("paths", []), examples_tensor, images_out_dir
    )

    if player is not None and hasattr(player, "discovered_variables"):
        state_dict["discovered_variables"] = player.discovered_variables

    with open(state_path, "w") as f:
        json.dump(state_dict, f, indent=2)
    torch.save(examples_tensor, example_path)


def already_done(output_dir, task_idx, seed):
    state_path = output_dir / f"seed_{seed}" / f"task_{task_idx}_state.json"
    return state_path.exists()


def _release_prompter(prompter):
    """Free the prompter's GPU memory before the next seed's prompter loads.

    Without this, the next ``prompter = get_prompter(...)`` allocates a fresh
    ~62 GiB of gemma weights *before* Python's GC has freed the previous
    instance, briefly doubling VRAM use and triggering CUDA OOM. Cloud
    prompters (GPT) don't have remove_from_gpu — skip silently.
    """
    if hasattr(prompter, "remove_from_gpu"):
        prompter.remove_from_gpu()


# ── Per-player setup & run logic ──────────────────────────────────────────────

def _rtpt_step(args):
    rtpt = getattr(args, "rtpt", None)
    if rtpt is not None:
        rtpt.step()


def run_zendo_tasks(task_indices, seeds, args):
    from DSL import zendo, zendo_extended
    from data.create_programs import convert_prolog_to_dsl
    from grammar import dsl
    from model_loader import __build_generic_zendo_model
    from vision_model.zendo_classification.zendo_detection.model import (
        ZendoImageToVectorModel,
    )
    from zendo.game import play_game_state
    from zendo.game_master import ZendoStateGameMaster
    from zendo.player import ZendoPlayer, FullGPTZendoPlayer

    use_fullgpt = args.player == "fullgpt"
    player_tag = "fullgpt-qwen" if use_fullgpt else "zendo"

    base_cfg = {
        "max_objects": 7,
        "token_dim": 384,
        "color_lexicon": ["red", "blue", "yellow"],
        "shape_lexicon": ["block", "wedge", "pyramid"],
        "orientation_lexicon": ["upright", "upside_down", "flat", "cheesecake"],
        "dropout": 0.23,
        "layers": 4,
        "pointing_mult_layer": True,
        "touching_mult_layer": True,
        "bbox_mult_layer": True,
        "color_mult_layer": False,
        "shape_mult_layer": False,
        "orientation_mult_layer": False,
        "presence_mult_layer": False,
    }
    visionmodel = ZendoImageToVectorModel(
        base_cfg,
        num_output_tokens=base_cfg["max_objects"],
        token_dim=base_cfg["token_dim"],
        max_objects=base_cfg["max_objects"],
        num_colors=len(base_cfg["color_lexicon"]) + 1,
        num_shapes=len(base_cfg["shape_lexicon"]) + 1,
        num_orientations=len(base_cfg["orientation_lexicon"]) + 1,
    )
    ckpt_path = Path("zendo_model.pt")
    visionmodel.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=True)
    )
    visionmodel.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visionmodel.to(device)

    zendo_dsl = dsl.DSL(zendo.semantics, zendo.primitive_types, None)
    zendo_dsl_extended = dsl.DSL(
        zendo_extended.semantics, zendo_extended.primitive_types, None
    )

    cfg, model = __build_generic_zendo_model(
        dsl=zendo_dsl,
        max_program_depth=5,
        size_max=11,
        size_hidden=64,
        embedding_output_dimension=78,
        number_layers_RNN=1,
        autoload=False,
        name="bigramsPredictor_no_overlap.weights",
    )
    cfg_extended, _ = __build_generic_zendo_model(
        dsl=zendo_dsl_extended,
        max_program_depth=5,
        size_max=11,
        size_hidden=64,
        embedding_output_dimension=78,
        number_layers_RNN=1,
        autoload=False,
        name="model_weights/bigramsPredictor_variable.weights",
    )

    tasks = load_zendo_dataset()
    default_dir = f"gamestates/gamestates_{player_tag}"
    output_dir = Path(args.output_dir or default_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        prompter = get_prompter(args.prompter_model, "zendo", seed, reasoning=False, sampling=False)
        try:
            for task_idx in task_indices:
                name, examples, images = tasks[task_idx]
                print(f"\n[{player_tag}] Starting task {task_idx} (seed {seed}): {name}")
                with compute_log.use_log(
                    compute_log.seed_log_path(player_tag, seed),
                    args=vars(args),
                    scope={"player": player_tag, "seed": seed},
                ):
                    if already_done(output_dir, task_idx, seed):
                        print(f"Skipping task {task_idx} seed {seed} — already exists.")
                        _rtpt_step(args)
                        continue
                    try:
                        program = convert_prolog_to_dsl(name, cfg_extended)
                        gm = ZendoStateGameMaster(
                            true_program=program,
                            task_idx=task_idx,
                            dataset=examples.copy(),
                            paths=images.copy(),
                            zendo_dsl=zendo_dsl_extended,
                            cfg=cfg_extended,
                            images=True,
                            use_images=True,
                            ask_for_counter=False,
                            prompter=prompter,
                            seed=seed,
                        )
                        PlayerClass = FullGPTZendoPlayer if use_fullgpt else ZendoPlayer
                        player_kwargs = dict(
                            player_id=0,
                            task_idx=task_idx,
                            cfg=cfg,
                            dsl=zendo_dsl,
                            model=None,
                            bar=5e-9,
                            prefer_valid=False,
                            min_examples=4,
                            images=True,
                            gs_threshold=-1,
                            vision_model=visionmodel,
                            genai_client=None,
                            use_dsl=False,
                            seed=seed,
                        )
                        if use_fullgpt:
                            player_kwargs["prompter"] = prompter
                        player = PlayerClass(**player_kwargs)
                        cache_path = f"cached_states/zendo_cache_{player_tag}_task_{task_idx}_seed_{seed}.pkl"
                        state = play_game_state(gm, [player], cached=False, path=cache_path)
                        save_state(state, output_dir, task_idx, seed)
                    except Exception as e:
                        print(f"Task {task_idx} seed {seed} failed: {type(e).__name__}: {e!r}")
                        traceback.print_exc()
                    _rtpt_step(args)
        finally:
            _release_prompter(prompter)
            del prompter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def run_zendo_symbolic_tasks(task_indices, seeds, args):
    from DSL import zendo, zendo_extended
    from data.create_programs import convert_prolog_to_dsl
    from grammar import dsl
    from model_loader import __build_generic_zendo_model
    from vision_model.zendo_classification.zendo_detection.model import (
        ZendoImageToVectorModel,
    )
    from zendo.game import play_game_state
    from zendo.game_master import ZendoStateGameMaster
    from zendo.player import ZendoPlayer, FullGPTZendoPlayer

    use_fullgpt = args.player == "fullgpt_symbolic"
    player_tag = "fullgpt_symbolic" if use_fullgpt else "zendo_symbolic"

    zendo_dsl = dsl.DSL(zendo.semantics, zendo.primitive_types, None)
    zendo_dsl_extended = dsl.DSL(
        zendo_extended.semantics, zendo_extended.primitive_types, None
    )

    cfg, model = __build_generic_zendo_model(
        dsl=zendo_dsl,
        max_program_depth=5,
        size_max=11,
        size_hidden=64,
        embedding_output_dimension=78,
        number_layers_RNN=1,
        autoload=False,
        name="bigramsPredictor_no_overlap.weights",
    )
    cfg_extended, _ = __build_generic_zendo_model(
        dsl=zendo_dsl_extended,
        max_program_depth=5,
        size_max=11,
        size_hidden=64,
        embedding_output_dimension=78,
        number_layers_RNN=1,
        autoload=False,
        name="model_weights/bigramsPredictor_variable.weights",
    )

    tasks = load_zendo_dataset()
    default_dir = f"gamestates/gamestates_{player_tag}"
    output_dir = Path(args.output_dir or default_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        prompter = get_prompter(args.prompter_model, "zendo", seed, reasoning=False, sampling=False)
        try:
            for task_idx in task_indices:
                name, examples, images = tasks[task_idx]
                print(f"\n[{player_tag}] Starting task {task_idx} (seed {seed}): {name}")
                with compute_log.use_log(
                    compute_log.seed_log_path(player_tag, seed),
                    args=vars(args),
                    scope={"player": player_tag, "seed": seed},
                ):
                    if already_done(output_dir, task_idx, seed):
                        print(f"Skipping task {task_idx} seed {seed} — already exists.")
                        _rtpt_step(args)
                        continue
                    try:
                        program = convert_prolog_to_dsl(name, cfg_extended)
                        gm = ZendoStateGameMaster(
                            true_program=program,
                            task_idx=task_idx,
                            dataset=examples.copy(),
                            paths=images.copy(),
                            zendo_dsl=zendo_dsl_extended,
                            cfg=cfg_extended,
                            images=False,
                            use_images=False,
                            ask_for_counter=False,
                            prompter=prompter,
                            seed=seed,
                        )
                        PlayerClass = FullGPTZendoPlayer if use_fullgpt else ZendoPlayer
                        player_kwargs = dict(
                            player_id=0,
                            task_idx=task_idx,
                            cfg=cfg,
                            dsl=zendo_dsl,
                            model=None,
                            bar=5e-9,
                            prefer_valid=False,
                            min_examples=4,
                            images=False,
                            gs_threshold=-1,
                            vision_model=None,
                            genai_client=None,
                            use_dsl=False,
                            seed=seed,
                        )
                        if use_fullgpt:
                            player_kwargs["prompter"] = prompter
                        player = PlayerClass(**player_kwargs)
                        cache_path = f"cached_states/zendo_cache_{player_tag}_task_{task_idx}_seed_{seed}.pkl"
                        state = play_game_state(gm, [player], cached=False, path=cache_path)
                        save_state(state, output_dir, task_idx, seed)
                    except Exception as e:
                        print(f"Task {task_idx} seed {seed} failed: {type(e).__name__}: {e!r}")
                        traceback.print_exc()
                    _rtpt_step(args)
        finally:
            _release_prompter(prompter)
            del prompter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

_VLP_ABLATION_VARIABLES = {
    "objects": ["block", "pyramid", "wedge"],
    "properties": [
        "red", "blue", "yellow", "vertical",
        "upright", "upside_down", "flat", "cheesecake",
        "doorstop", "grounded", "ungrounded",
    ],
    "actions": ["on_top_of", "pointing_at", "touching"],
}


def run_vlp_tasks(task_indices, seeds, args, initial_variables=None, symbolic=False, uncertainty=False):
    from DSL import zendo, zendo_extended
    from DSL.vlp_dsl_symbolic import get_dsl as get_dsl_symbolic
    from DSL.vlp_dsl import get_dsl as get_dsl_visual
    if symbolic and uncertainty:
        raise ValueError("symbolic and uncertainty modes are mutually exclusive for the VLP player")
    if symbolic:
        get_dsl = get_dsl_symbolic
    else:
        get_dsl = get_dsl_visual
    from data.create_programs import convert_prolog_to_dsl
    from grammar import dsl
    from model_loader import __build_generic_zendo_model
    from models.prompters import get_prompter
    from type_system import BOOL, IMG, Arrow
    from zendo.game import play_game_state
    from zendo.game_master import ZendoStateGameMaster
    from zendo.player import VLPZendoPlayer
    PlayerCls = VLPZendoPlayer

    org_zendo_dsl = dsl.DSL(zendo.semantics, zendo.primitive_types, None)
    zendo_dsl_extended = dsl.DSL(
        zendo_extended.semantics, zendo_extended.primitive_types, None, 8
    )

    zendo_cfg, model = __build_generic_zendo_model(
        dsl=org_zendo_dsl,
        max_program_depth=7,
        size_max=11,
        size_hidden=64,
        embedding_output_dimension=78,
        number_layers_RNN=1,
        autoload=False,
        name="model_weights/bigramsPredictor_variable.weights",
    )
    cfg_extended, _ = __build_generic_zendo_model(
        dsl=zendo_dsl_extended,
        max_program_depth=5,
        size_max=11,
        size_hidden=64,
        embedding_output_dimension=78,
        number_layers_RNN=1,
        autoload=False,
        name="model_weights/bigramsPredictor_variable.weights",
    )

    tasks = load_zendo_dataset()
    is_ablation = initial_variables is not None
    if uncertainty:
        player_tag = "vlp_uncertainty"
    elif symbolic:
        player_tag = "vlp_symbolic"
    elif is_ablation:
        player_tag = "vlp_ablation"
    else:
        player_tag = "vlp_kimi"
    output_dir = Path(args.output_dir or f"gamestates/gamestates_{player_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    type_request = Arrow(IMG, BOOL)

    for seed in seeds:
        prompter = get_prompter(args.prompter_model, "zendo", seed, reasoning=False, sampling=False)
        try:
            semantics, primitive_types = get_dsl(prompter, {}, seed=seed)
            vlp_dsl = dsl.DSL(semantics, primitive_types, None)
            cfg = vlp_dsl.DSL_to_CFG(
                type_request,
                max_program_depth=10,
                min_variable_depth=1,
                upper_bound_type_size=10,
                n_gram=2,
            )
            extra_player_kwargs = {}
            if uncertainty:
                extra_player_kwargs.update(
                    max_objects=5,
                    max_properties=7,
                    max_actions=5,
                )
            for task_idx in task_indices:
                name, examples, images = tasks[task_idx]
                print(f"\n[{player_tag}] Starting task {task_idx} (seed {seed}): {name}")
                with compute_log.use_log(
                    compute_log.seed_log_path(player_tag, seed),
                    args=vars(args),
                    scope={"player": player_tag, "seed": seed},
                ):
                    if already_done(output_dir, task_idx, seed):
                        print(f"Skipping task {task_idx} seed {seed} — already exists.")
                        _rtpt_step(args)
                        continue
                    try:
                        player = PlayerCls(
                            player_id=0,
                            task_idx=task_idx,
                            cfg=cfg,
                            zendo_cfg=cfg_extended,
                            dsl=vlp_dsl,
                            model=None,
                            bar=5e-9,
                            prefer_valid=False,
                            min_examples=4,
                            images=True,
                            gs_threshold=-1,
                            vision_model=None,
                            genai_client=None,
                            prompter=prompter,
                            discovery_examples=4,
                            n_objects=3,
                            n_properties=5,
                            n_actions=3,
                            n_sceneries=0,
                            seed=seed,
                            initial_variables=initial_variables,
                            symbolic=symbolic,
                            **extra_player_kwargs,
                        )
                        program = convert_prolog_to_dsl(name, cfg_extended)
                        gm = ZendoStateGameMaster(
                            true_program=program,
                            task_idx=task_idx,
                            dataset=examples.copy(),
                            paths=images.copy(),
                            zendo_dsl=zendo_dsl_extended,
                            cfg=cfg_extended,
                            images=True,
                            use_images=True,
                            ask_for_counter=False,
                            prompter=prompter,
                            vlp=True,
                            seed=seed,
                            symbolic=symbolic,
                        )
                        cache_path = f"cached_states/zendo_cache_{player_tag}_task_{task_idx}_seed_{seed}.pkl"
                        state = play_game_state(gm, [player], cached=False, path=cache_path)
                        save_state(state, output_dir, task_idx, seed, player=player)
                    except Exception as e:
                        print(f"Task {task_idx} seed {seed} failed: {type(e).__name__}: {e!r}")
                        traceback.print_exc()
                    _rtpt_step(args)
        finally:
            _release_prompter(prompter)
            del prompter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_vlp_ablation_tasks(task_indices, seeds, args):
    run_vlp_tasks(task_indices, seeds, args, initial_variables=_VLP_ABLATION_VARIABLES)


def run_vlp_symbolic_tasks(task_indices, seeds, args):
    run_vlp_tasks(task_indices, seeds, args, symbolic=True)


def run_vlp_uncertainty_tasks(task_indices, seeds, args):
    run_vlp_tasks(task_indices, seeds, args, uncertainty=True)


def run_random_zendo_symbolic_tasks(task_indices, seeds, args):
    from DSL import zendo, zendo_extended
    from data.create_programs import convert_prolog_to_dsl
    from grammar import dsl
    from model_loader import __build_generic_zendo_model
    from zendo.game import play_game_state
    from zendo.game_master import ZendoStateGameMaster
    from zendo.player import RandomZendoPlayer

    player_tag = "random_zendo_symbolic"

    zendo_dsl = dsl.DSL(zendo.semantics, zendo.primitive_types, None)
    zendo_dsl_extended = dsl.DSL(
        zendo_extended.semantics, zendo_extended.primitive_types, None
    )

    cfg, model = __build_generic_zendo_model(
        dsl=zendo_dsl,
        max_program_depth=5,
        size_max=11,
        size_hidden=64,
        embedding_output_dimension=78,
        number_layers_RNN=1,
        autoload=False,
        name="bigramsPredictor_no_overlap.weights",
    )
    cfg_extended, _ = __build_generic_zendo_model(
        dsl=zendo_dsl_extended,
        max_program_depth=5,
        size_max=11,
        size_hidden=64,
        embedding_output_dimension=78,
        number_layers_RNN=1,
        autoload=False,
        name="model_weights/bigramsPredictor_variable.weights",
    )

    tasks = load_zendo_dataset()
    output_dir = Path(args.output_dir or f"gamestates/gamestates_{player_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        prompter = get_prompter(args.prompter_model, "zendo", seed, reasoning=False, sampling=False)
        try:
            for task_idx in task_indices:
                name, examples, images = tasks[task_idx]
                print(f"\n[{player_tag}] Starting task {task_idx} (seed {seed}): {name}")
                with compute_log.use_log(
                    compute_log.seed_log_path(player_tag, seed),
                    args=vars(args),
                    scope={"player": player_tag, "seed": seed},
                ):
                    if already_done(output_dir, task_idx, seed):
                        print(f"Skipping task {task_idx} seed {seed} — already exists.")
                        _rtpt_step(args)
                        continue
                    try:
                        program = convert_prolog_to_dsl(name, cfg_extended)
                        gm = ZendoStateGameMaster(
                            true_program=program,
                            task_idx=task_idx,
                            dataset=examples.copy(),
                            paths=images.copy(),
                            zendo_dsl=zendo_dsl_extended,
                            cfg=cfg_extended,
                            images=False,
                            use_images=False,
                            ask_for_counter=False,
                            prompter=prompter,
                            seed=seed,
                        )
                        player = RandomZendoPlayer(
                            player_id=0,
                            task_idx=task_idx,
                            cfg=cfg,
                            dsl=zendo_dsl,
                            model=None,
                            bar=5e-9,
                            prefer_valid=False,
                            min_examples=4,
                            images=False,
                            gs_threshold=-1,
                            vision_model=None,
                            genai_client=None,
                            use_dsl=False,
                            seed=seed,
                        )
                        cache_path = f"cached_states/zendo_cache_{player_tag}_task_{task_idx}_seed_{seed}.pkl"
                        state = play_game_state(gm, [player], cached=False, path=cache_path)
                        save_state(state, output_dir, task_idx, seed)
                    except Exception as e:
                        print(f"Task {task_idx} seed {seed} failed: {type(e).__name__}: {e!r}")
                        traceback.print_exc()
                    _rtpt_step(args)
        finally:
            _release_prompter(prompter)
            del prompter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_scientist_tasks(task_indices, seeds, args):
    from DSL import zendo_extended
    from data.create_programs import convert_prolog_to_dsl
    from grammar import dsl
    from model_loader import __build_generic_zendo_model
    from zendo_conf import AdvZendoConfig
    from zendo.game import play_game_state
    from zendo.game_master import ZendoStateGameMaster
    from zendo.scientist_player import LLMScientistPlayer

    zendo_dsl_extended = dsl.DSL(
        zendo_extended.semantics, zendo_extended.primitive_types, None, constants_range=7
    )

    cfg, _ = __build_generic_zendo_model(
        dsl=zendo_dsl_extended,
        max_program_depth=5,
        size_max=11,
        size_hidden=64,
        embedding_output_dimension=78,
        number_layers_RNN=1,
        autoload=False,
        name="model_weights/bigramsPredictor_variable.weights",
    )

    tasks = load_zendo_dataset()
    output_dir = Path(args.output_dir or "gamestates/gamestates_scientist")
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        prompter = get_prompter(args.prompter_model, "zendo", seed, reasoning=False, sampling=False)
        try:
            for task_idx in task_indices:
                name, examples, images = tasks[task_idx]
                print(f"\n[scientist] Starting task {task_idx} (seed {seed}): {name}")
                with compute_log.use_log(
                    compute_log.seed_log_path("scientist", seed),
                    args=vars(args),
                    scope={"player": "scientist", "seed": seed},
                ):
                    if already_done(output_dir, task_idx, seed):
                        print(f"Skipping task {task_idx} seed {seed} — already exists.")
                        _rtpt_step(args)
                        continue
                    try:
                        program = convert_prolog_to_dsl(name, cfg)
                        gm = ZendoStateGameMaster(
                            true_program=program,
                            task_idx=task_idx,
                            dataset=examples.copy(),
                            paths=images.copy(),
                            zendo_dsl=zendo_dsl_extended,
                            cfg=cfg,
                            images=True,
                            use_images=True,
                            ask_for_counter=False,
                            prompter=prompter,
                            seed=seed,
                        )
                        player = LLMScientistPlayer(
                            player_id=0,
                            task_idx=task_idx,
                            cfg=cfg,
                            dsl=zendo_dsl_extended,
                            zendo_config=AdvZendoConfig,
                            min_examples=4,
                            prompter=prompter,
                            use_dsl=False,
                            use_paths=True,
                            seed=seed,
                        )
                        cache_path = f"zendo_cache_scientist_task_{task_idx}_seed_{seed}.pkl"
                        state = play_game_state(
                            gm, [player], cached=False, path=cache_path
                        )
                        save_state(state, output_dir, task_idx, seed)
                    except Exception as e:
                        print(f"Task {task_idx} seed {seed} failed: {type(e).__name__}: {e!r}")
                        traceback.print_exc()
                    _rtpt_step(args)
        finally:
            _release_prompter(prompter)
            del prompter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def run_scientist_symbolic_tasks(task_indices, seeds, args):
    from DSL import zendo_extended
    from data.create_programs import convert_prolog_to_dsl
    from grammar import dsl
    from model_loader import __build_generic_zendo_model
    from zendo_conf import AdvZendoConfig
    from zendo.game import play_game_state
    from zendo.game_master import ZendoStateGameMaster
    from zendo.scientist_player import LLMScientistPlayer

    zendo_dsl_extended = dsl.DSL(
        zendo_extended.semantics, zendo_extended.primitive_types, None, constants_range=7
    )

    cfg, _ = __build_generic_zendo_model(
        dsl=zendo_dsl_extended,
        max_program_depth=5,
        size_max=11,
        size_hidden=64,
        embedding_output_dimension=78,
        number_layers_RNN=1,
        autoload=False,
        name="model_weights/bigramsPredictor_variable.weights",
    )

    tasks = load_zendo_dataset()
    output_dir = Path(args.output_dir or "gamestates/gamestates_scientist_symbolic")
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        prompter = get_prompter(args.prompter_model, "zendo", seed, reasoning=False, sampling=False)
        try:
            for task_idx in task_indices:
                name, examples, images = tasks[task_idx]
                print(f"\n[scientist] Starting task {task_idx} (seed {seed}): {name}")
                with compute_log.use_log(
                    compute_log.seed_log_path("scientist_symbolic", seed),
                    args=vars(args),
                    scope={"player": "scientist_symbolic", "seed": seed},
                ):
                    if already_done(output_dir, task_idx, seed):
                        print(f"Skipping task {task_idx} seed {seed} — already exists.")
                        _rtpt_step(args)
                        continue
                    try:
                        program = convert_prolog_to_dsl(name, cfg)
                        gm = ZendoStateGameMaster(
                            true_program=program,
                            task_idx=task_idx,
                            dataset=examples.copy(),
                            paths=images.copy(),
                            zendo_dsl=zendo_dsl_extended,
                            cfg=cfg,
                            images=False,
                            use_images=False,
                            ask_for_counter=False,
                            prompter=prompter,
                            seed=seed,
                        )
                        player = LLMScientistPlayer(
                            player_id=0,
                            task_idx=task_idx,
                            cfg=cfg,
                            dsl=zendo_dsl_extended,
                            zendo_config=AdvZendoConfig,
                            min_examples=4,
                            prompter=prompter,
                            use_dsl=False,
                            use_paths=False,
                            images=False,
                            seed=seed,
                        )
                        cache_path = f"zendo_cache_scientist_symbolic_task_{task_idx}_seed_{seed}.pkl"
                        state = play_game_state(
                            gm, [player], cached=False, path=cache_path
                        )
                        save_state(state, output_dir, task_idx, seed)
                    except Exception as e:
                        print(f"Task {task_idx} seed {seed} failed: {type(e).__name__}: {e!r}")
                        traceback.print_exc()
                    _rtpt_step(args)
        finally:
            _release_prompter(prompter)
            del prompter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ── Worker entry point for multiprocessing ────────────────────────────────────

PLAYER_RUNNERS = {
    "zendo": run_zendo_tasks,
    "fullgpt": run_zendo_tasks,
    "vlp": run_vlp_tasks,
    "vlp_ablation": run_vlp_ablation_tasks,
    "vlp_symbolic": run_vlp_symbolic_tasks,
    "vlp_uncertainty": run_vlp_uncertainty_tasks,
    "scientist": run_scientist_tasks,
    "zendo_symbolic": run_zendo_symbolic_tasks,
    "fullgpt_symbolic": run_zendo_symbolic_tasks,
    "scientist_symbolic": run_scientist_symbolic_tasks,
    "random_zendo_symbolic": run_random_zendo_symbolic_tasks,
}


def worker(player_type, task_indices, seeds, args):
    runner = PLAYER_RUNNERS[player_type]
    runner(task_indices, seeds, args)
    return task_indices


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified Zendo experiment runner")
    parser.add_argument(
        "--player",
        type=str,
        required=True,
        choices=["zendo", "fullgpt", "vlp", "vlp_ablation", "vlp_symbolic", "vlp_uncertainty", "scientist", "zendo_symbolic", "fullgpt_symbolic", "scientist_symbolic", "random_zendo_symbolic"],
        help="Player type to run",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Seeds to run (default: 1, 2, 3)",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        default=None,
        help="Task indices to run (default: all tasks)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: player-specific)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/combined_dataset_short.pkl",
        help="Path to the dataset pickle file",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="SK",
        help="Initials shown in the RTPT process title so others on the "
             "server can see whose experiment it is and when it will finish.",
    )
    parser.add_argument(
        "--prompter-model",
        type=str,
        default="gpt-5-mini",
        help=(
            "LLM/VLM backend to use for prompts. Supported values include "
            "'gpt-5-mini', 'gpt-5', 'gpt-4o', 'gpt-5-chat-latest', "
            "'Qwen2.5-VL-7B-Instruct', 'Qwen3-VL-30B-A3B-Instruct', "
            "'Qwen3-VL-30B-A3B-Thinking', 'gemma-3-4b-it', 'gemma-3-12b-it', "
            "'gemma-3-27b-it' (see models/prompters.py for the full list)."
        ),
    )
    args = parser.parse_args()

    # Set up compute logging. If ZENDO_COMPUTE_LOG is already set (e.g. user
    # exported it), respect that path; otherwise pick a timestamped default
    # under logs/ and export so worker subprocesses inherit it.
    if not os.environ.get("ZENDO_COMPUTE_LOG"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log = os.path.abspath(
            f"logs/compute_{args.player}_{args.prompter_model}_{stamp}.jsonl"
        )
        os.environ["ZENDO_COMPUTE_LOG"] = default_log
    log_dir = os.path.dirname(os.environ["ZENDO_COMPUTE_LOG"])
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    print(f"Compute log: {os.environ['ZENDO_COMPUTE_LOG']}")
    compute_log.start(args=vars(args))
    atexit.register(compute_log.end)

    tasks = load_zendo_dataset(args.dataset)
    num_tasks = len(tasks)
    print(f"Loaded {num_tasks} tasks from {args.dataset}")

    if args.tasks is not None:
        task_indices = [t for t in args.tasks if 0 <= t < num_tasks]
    else:
        task_indices = list(range(num_tasks))

    n_iterations = len(task_indices) * len(args.seeds)
    print(
        f"Running player={args.player}, tasks={len(task_indices)}, "
        f"seeds={args.seeds}, threads={args.threads}"
    )

    if _RTPT is not None and n_iterations > 0:
        args.rtpt = _RTPT(name_initials=args.name, experiment_name=args.player,
                          max_iterations=n_iterations)
        args.rtpt.start()
    else:
        args.rtpt = None
        if _RTPT is None:
            print("[rtpt] package not found — install rtpt for process-title ETA")

    if args.threads <= 1:
        runner = PLAYER_RUNNERS[args.player]
        runner(task_indices, args.seeds, args)
    else:
        n = args.threads
        chunks = [[] for _ in range(n)]
        for i, task_idx in enumerate(task_indices):
            chunks[i % n].append(task_idx)
        chunks = [c for c in chunks if c]

        print(f"Splitting {len(task_indices)} tasks into {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"  Worker {i}: tasks {chunk}")

        with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            futures = {
                executor.submit(worker, args.player, chunk, args.seeds, args): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    completed_tasks = future.result()
                    print(f"Worker {worker_id} finished tasks {completed_tasks}")
                except Exception as e:
                    print(f"Worker {worker_id} crashed: {e}")
                    traceback.print_exc()

    print("\nAll done.")


if __name__ == "__main__":
    main()
