from zendoworld.paths import VISION_WEIGHTS
from zendoworld.serialization import load_pickle
"""Unified Zendo experiment runner.

Usage:
    python -m zendoworld.experiments.runner --player zendo --seeds 1 2 3
    python -m zendoworld.experiments.runner --player vlp --threads 4 --seeds 1
    python -m zendoworld.experiments.runner --player scientist --threads 2 --seeds 1 2 3
    python -m zendoworld.experiments.runner --player vlp --threads 4 --tasks 0 2 5 6

Append ``_symbolic`` to the player name to run with symbolic inputs.
For VLP and Scientist, add ``--scene-parser symbolic`` for deterministic scene
parsing, leaving the remaining model calls unchanged.
"""

import argparse
import atexit
import fcntl
import gc
import multiprocessing
import os
import pickle
import json
import hashlib
import re
import shutil
import time
import traceback
import tempfile
from datetime import datetime
from zendoworld.backends.prompters import get_prompter
from zendoworld.backends.errors import OpenAIQuotaExhausted
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import zendoworld.experiments.compute_log as compute_log
from zendoworld.game.input_mode import uses_image_paths


def _acquire_work_locks(namespace, task_indices, seeds):
    """Prevent independent launches from running the same seed/task pair."""
    lock_dir = Path("outputs/logs") / "run_locks" / namespace
    lock_dir.mkdir(parents=True, exist_ok=True)
    held = []
    try:
        for seed in seeds:
            for task_idx in task_indices:
                path = lock_dir / f"seed_{seed}_task_{task_idx}.lock"
                handle = open(path, "a+")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    raise RuntimeError(
                        f"Seed {seed}, task {task_idx} is already running in "
                        f"another {namespace!r} experiment. Use disjoint "
                        "--seeds/--tasks or a different --output-dir."
                    )
                handle.seek(0)
                handle.truncate()
                handle.write(f"pid={os.getpid()}\n")
                handle.flush()
                held.append(handle)
    except Exception:
        for handle in held:
            handle.close()
        raise
    return held

try:
    from rtpt import RTPT as _RTPT
except ImportError:
    _RTPT = None


# ── Shared helpers ────────────────────────────────────────────────────────────


def load_zendo_dataset(pkl_path="data/combined_dataset_short.pkl"):
    with open(pkl_path, "rb") as f:
        tasks = load_pickle(f)
    return tasks


def classify_label(src_path: Path) -> str:
    # Only player-proposed scenes are rendered under outputs/rendering/.
    # Everything else — dataset images (data/images/...) and GM counterexamples
    # (which carry "counter" in the path) — belongs to the game master.
    s = src_path.as_posix().lower()
    return "player" if any(p in s for p in ("outputs/rendering", "generation/output")) and "counter" not in s else "gm"

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
        shutil.copy2(src_path, dest_path)
        new_paths.append(str(dest_path))
    return new_paths


def save_state(state, output_dir, task_idx, seed, player=None, symbolic=False):
    iteration_dir = output_dir / f"seed_{seed}"
    iteration_dir.mkdir(parents=True, exist_ok=True)

    state_path = iteration_dir / f"task_{task_idx}_state.json"
    example_path = iteration_dir / f"examples_{task_idx}.pt"

    # Invalidate any incomplete earlier result before touching its assets.
    state_path.unlink(missing_ok=True)
    state_dict, examples_tensor, bramley_examples_tensor = state.to_dict()

    if symbolic:
        from zendoworld.datasets.tensor2piece import tensor_to_prolog_strings
        state_dict["input_mode"] = "symbolic"
        state_dict["scene_descriptions"] = tensor_to_prolog_strings(
            [tensor for tensor, _label in examples_tensor]
        )
    else:
        images_out_dir = iteration_dir / f"task_{task_idx}_images"
        images_out_dir.mkdir(parents=True, exist_ok=True)
        state_dict["paths"] = copy_and_rename(
            state_dict.get("paths", []), examples_tensor, images_out_dir
        )

    if player is not None and hasattr(player, "discovered_variables"):
        state_dict["discovered_variables"] = player.discovered_variables
    if hasattr(state, "resume_provenance"):
        state_dict["resume_provenance"] = state.resume_provenance

    # The JSON is the completion marker; publish it only after all assets.
    with tempfile.TemporaryDirectory(dir=iteration_dir) as staging:
        staged = Path(staging)
        torch.save(examples_tensor, staged / "examples.pt")
        (staged / "state.json").write_text(json.dumps(state_dict, indent=2))
        os.replace(staged / "examples.pt", example_path)
        os.replace(staged / "state.json", state_path)
    (iteration_dir / f"task_{task_idx}_error.json").unlink(missing_ok=True)


def already_done(output_dir, task_idx, seed, symbolic=False):
    iteration_dir = output_dir / f"seed_{seed}"
    state_path = iteration_dir / f"task_{task_idx}_state.json"
    try:
        state = json.loads(state_path.read_text())
        examples = iteration_dir / f"examples_{task_idx}.pt"
        return (
            isinstance(state, dict)
            and isinstance(state.get("paths"), list)
            and examples.is_file() and examples.stat().st_size > 0
            and (
                (symbolic or state.get("input_mode") == "symbolic")
                or all(Path(path).is_file() for path in state["paths"])
            )
        )
    except (OSError, ValueError, TypeError):
        return False


def save_task_error(output_dir, task_idx, seed, error):
    directory = output_dir / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(task=task_idx, seed=seed, error=repr(error),
                   traceback=traceback.format_exc())
    with tempfile.TemporaryDirectory(dir=directory) as staging:
        temporary = Path(staging) / "error.json"
        temporary.write_text(json.dumps(payload, indent=2))
        os.replace(temporary, directory / f"task_{task_idx}_error.json")


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

def _args_dict(args):
    """Return vars(args) with non-JSON-serializable fields removed."""
    redacted = {"openai_api_key"}
    return {
        k: v for k, v in vars(args).items()
        if k != "rtpt" and k not in redacted
    }


def _configure_openai_api_key(args):
    """Select this launch's OpenAI key without passing it to worker args."""
    direct_key = getattr(args, "openai_api_key", None)
    key_file = getattr(args, "openai_api_key_file", None)
    if direct_key:
        os.environ["OPENAI_API_KEY"] = direct_key.strip()
        source = "--openai-api-key"
    elif key_file:
        try:
            key = Path(key_file).expanduser().read_text().strip()
        except OSError as exc:
            raise ValueError(f"Could not read --openai-api-key-file {key_file!r}: {exc}") from exc
        if not key:
            raise ValueError(f"--openai-api-key-file {key_file!r} is empty")
        os.environ["OPENAI_API_KEY"] = key
        source = f"--openai-api-key-file {key_file}"
    else:
        source = "OPENAI_API_KEY/environment or legacy fallback"

    # Never pickle the literal key into worker arguments or serialize it into
    # compute logs. Spawned workers inherit the selected environment instead.
    args.openai_api_key = None
    print(f"OpenAI API key source: {source}")


def _validate_parallel_backend(args):
    """Reject worker configurations that would duplicate a local VLM."""
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    if args.task_concurrency < 1:
        raise ValueError("--task-concurrency must be at least 1")
    if args.use_vllm and args.prompter_model not in ("Qwen3.8-27B", "gemma-4-31B-it"):
        raise ValueError(
            "--use-vllm currently supports Qwen3.5 and Gemma 4 models only; "
            f"got {args.prompter_model!r}"
        )

    is_remote_api = args.prompter_model.startswith(("gpt-", "gemini"))
    shares_server_model = args.use_vllm and args.prompter_model in ("Qwen3.8-27B", "gemma-4-31B-it")
    if args.threads > 1 and not (is_remote_api or shares_server_model):
        raise ValueError(
            f"--threads {args.threads} starts {args.threads} processes and would "
            f"load {args.prompter_model!r} once per process. Use --threads 1 "
            "with --task-concurrency for a shared in-process model, or run a "
            "Gemma vLLM server and pass --use-vllm."
        )


def _rtpt_step(args):
    rtpt = getattr(args, "rtpt", None)
    if rtpt is not None:
        rtpt.step()


def run_zendo_tasks(task_indices, seeds, args):
    from zendoworld.synthesis.dsl import zendo, zendo_extended
    from zendoworld.datasets.create_programs import convert_prolog_to_dsl
    from zendoworld.synthesis.grammar import dsl
    from zendoworld.synthesis.predictions.loader import __build_generic_zendo_model
    from zendoworld.vision.zendo_classification.zendo_detection.model import (
        ZendoImageToVectorModel,
    )
    from zendoworld.game.game import play_game_state
    from zendoworld.game.game_master import ZendoStateGameMaster
    from zendoworld.game.player import ZendoPlayer, FullGPTZendoPlayer

    use_fullgpt = args.player == "fullgpt"
    model_tag = args.prompter_model.replace("/", "_")
    if args.prompter_model.startswith("gpt-"):
        model_tag += f"_{args.reasoning_effort}"
    player_tag = f"fullgpt_{model_tag}" if use_fullgpt else "zendo"

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
    ckpt_path = VISION_WEIGHTS
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
        name="weights/synthesis/bigramsPredictor_variable.weights",
    )

    tasks = load_zendo_dataset(args.dataset)
    default_dir = f"outputs/gamestates/gamestates_{player_tag}"
    output_dir = Path(args.output_dir or default_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        prompter = get_prompter(args.prompter_model, "zendo", seed, reasoning=False, sampling=False,
                                use_vllm=args.use_vllm,
                                reasoning_effort=args.reasoning_effort)
        try:
            for task_idx in task_indices:
                name, examples, images = tasks[task_idx]
                print(f"\n[{player_tag}] Starting task {task_idx} (seed {seed}): {name}")
                with compute_log.use_log(
                    compute_log.seed_log_path(
                        player_tag, seed, task_idx=task_idx
                    ),
                    args=_args_dict(args),
                    scope={"player": player_tag, "seed": seed},
                ):
                    if already_done(output_dir, task_idx, seed, symbolic=not uses_image_paths(args.player)):
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
                            images=uses_image_paths(args.player),
                            use_images=uses_image_paths(args.player),
                            ask_for_counter=args.ask_for_counter,
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
                            images=uses_image_paths(args.player),
                            gs_threshold=-1,
                            vision_model=visionmodel,
                            genai_client=None,
                            use_dsl=False,
                            seed=seed,
                        )
                        if use_fullgpt:
                            player_kwargs["prompter"] = prompter
                        player = PlayerClass(**player_kwargs)
                        cache_path = f"outputs/cached_states/zendo_cache_{player_tag}_task_{task_idx}_seed_{seed}.pkl"
                        state = play_game_state(gm, [player], cached=False, path=cache_path)
                        save_state(state, output_dir, task_idx, seed, symbolic=not uses_image_paths(args.player))
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
    from zendoworld.synthesis.dsl import zendo, zendo_extended
    from zendoworld.datasets.create_programs import convert_prolog_to_dsl
    from zendoworld.synthesis.grammar import dsl
    from zendoworld.synthesis.predictions.loader import __build_generic_zendo_model
    from zendoworld.vision.zendo_classification.zendo_detection.model import (
        ZendoImageToVectorModel,
    )
    from zendoworld.game.game import play_game_state
    from zendoworld.game.game_master import ZendoStateGameMaster
    from zendoworld.game.player import ZendoPlayer, FullGPTZendoPlayer

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
        name="weights/synthesis/bigramsPredictor_variable.weights",
    )

    tasks = load_zendo_dataset(args.dataset)
    default_dir = f"outputs/gamestates/gamestates_{player_tag}"
    output_dir = Path(args.output_dir or default_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        prompter = get_prompter(args.prompter_model, "zendo", seed, reasoning=False, sampling=False,
                                use_vllm=args.use_vllm,
                                reasoning_effort=args.reasoning_effort)
        try:
            for task_idx in task_indices:
                name, examples, images = tasks[task_idx]
                print(f"\n[{player_tag}] Starting task {task_idx} (seed {seed}): {name}")
                with compute_log.use_log(
                    compute_log.seed_log_path(player_tag, seed, task_idx=task_idx),
                    args=_args_dict(args),
                    scope={"player": player_tag, "seed": seed},
                ):
                    if already_done(output_dir, task_idx, seed, symbolic=not uses_image_paths(args.player)):
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
                            images=uses_image_paths(args.player),
                            use_images=uses_image_paths(args.player),
                            ask_for_counter=args.ask_for_counter,
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
                            images=uses_image_paths(args.player),
                            gs_threshold=-1,
                            vision_model=None,
                            genai_client=None,
                            use_dsl=False,
                            seed=seed,
                        )
                        if use_fullgpt:
                            player_kwargs["prompter"] = prompter
                        player = PlayerClass(**player_kwargs)
                        cache_path = f"outputs/cached_states/zendo_cache_{player_tag}_task_{task_idx}_seed_{seed}.pkl"
                        state = play_game_state(gm, [player], cached=False, path=cache_path)
                        save_state(state, output_dir, task_idx, seed, symbolic=not uses_image_paths(args.player))
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
    from zendoworld.synthesis.dsl import zendo, zendo_extended
    from zendoworld.synthesis.dsl.vlp_dsl_symbolic import get_dsl as get_dsl_symbolic
    from zendoworld.synthesis.dsl.vlp_dsl import get_dsl as get_dsl_visual
    symbolic = not uses_image_paths(args.player)
    if symbolic and uncertainty:
        raise ValueError("symbolic and uncertainty modes are mutually exclusive for the VLP player")
    if symbolic:
        get_dsl = get_dsl_symbolic
    else:
        get_dsl = get_dsl_visual
    from zendoworld.datasets.create_programs import convert_prolog_to_dsl
    from zendoworld.synthesis.grammar import dsl
    from zendoworld.synthesis.predictions.loader import __build_generic_zendo_model
    from zendoworld.backends.prompters import get_prompter
    from zendoworld.synthesis.type_system import BOOL, IMG, Arrow
    from zendoworld.game.game import play_game_state
    from zendoworld.game.game_master import ZendoStateGameMaster
    from zendoworld.game.player import VLPZendoPlayer
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
        name="weights/synthesis/bigramsPredictor_variable.weights",
    )
    cfg_extended, _ = __build_generic_zendo_model(
        dsl=zendo_dsl_extended,
        max_program_depth=5,
        size_max=11,
        size_hidden=64,
        embedding_output_dimension=78,
        number_layers_RNN=1,
        autoload=False,
        name="weights/synthesis/bigramsPredictor_variable.weights",
    )

    tasks = load_zendo_dataset(args.dataset)
    is_ablation = initial_variables is not None
    if uncertainty:
        player_tag = "vlp_uncertainty"
    elif symbolic:
        player_tag = "vlp_symbolic"
    elif is_ablation:
        player_tag = "vlp_ablation"
    else:
        player_tag = "vlp_gemini"
    if symbolic and getattr(args, "scene_parser", "neural") == "symbolic":
        player_tag += "_parser"
    output_dir = Path(args.output_dir or f"outputs/gamestates/gamestates_{player_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    type_request = Arrow(IMG, BOOL)

    failures = []
    for seed in seeds:
        prompter = get_prompter(args.prompter_model, "zendo", seed, reasoning=False, sampling=False,
                                use_vllm=args.use_vllm,
                                reasoning_effort=args.reasoning_effort)
        try:
            parser_kwargs = {"scene_parser": getattr(args, "scene_parser", "neural")} if symbolic else {}
            semantics, primitive_types = get_dsl(prompter, {}, seed=seed, **parser_kwargs)
            vlp_dsl = dsl.DSL(semantics, primitive_types, None)
            cfg = vlp_dsl.DSL_to_CFG(
                type_request,
                max_program_depth=10,
                min_variable_depth=1,
                upper_bound_type_size=10,
                n_gram=2,
            )
            extra_player_kwargs = {}
            if symbolic:
                extra_player_kwargs["scene_parser"] = getattr(args, "scene_parser", "neural")
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
                    compute_log.seed_log_path(player_tag, seed, task_idx=task_idx),
                    args=_args_dict(args),
                    scope={"player": player_tag, "seed": seed},
                ):
                    if already_done(output_dir, task_idx, seed, symbolic=not uses_image_paths(args.player)):
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
                            images=uses_image_paths(args.player),
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
                            images=uses_image_paths(args.player),
                            use_images=True,  # Enables the VLP adapter; symbolic selects text inside it.
                            ask_for_counter=args.ask_for_counter,
                            prompter=prompter,
                            vlp=True,
                            seed=seed,
                            symbolic=symbolic,
                        )
                        if getattr(args, "rewind_timeouts_from", None):
                            from zendoworld.game.states import GameCache
                            from zendoworld.game.vlp_resume import rewind_timeout_cache, continue_symbolic_vlp
                            source = Path(args.rewind_timeouts_from) / f"seed_{seed}" / f"task_{task_idx}_state.json"
                            source_cache = Path(args.rewind_cache_dir) / f"zendo_cache_{player_tag}_task_{task_idx}_seed_{seed}.pkl"
                            rewound, provenance = rewind_timeout_cache(
                                GameCache.from_file(source_cache), json.loads(source.read_text())
                            )
                            provenance.update(source_result=str(source.resolve()),
                                              source_cache=str(source_cache.resolve()))
                            rewound.state.resume_provenance = provenance
                            cache_dir = output_dir / "resume_caches"
                            cache_dir.mkdir(parents=True, exist_ok=True)
                            cache_path = cache_dir / f"task_{task_idx}_seed_{seed}.pkl"
                            print(f"Resuming task {task_idx}, seed {seed}: {provenance}")
                            state = continue_symbolic_vlp(rewound, gm, player, cache_path)
                        else:
                            cache_path = f"outputs/cached_states/zendo_cache_{player_tag}_task_{task_idx}_seed_{seed}.pkl"
                            state = play_game_state(gm, [player], cached=False, path=cache_path)
                        save_state(state, output_dir, task_idx, seed, player=player, symbolic=symbolic)
                    except Exception as e:
                        print(f"Task {task_idx} seed {seed} failed: {type(e).__name__}: {e!r}")
                        traceback.print_exc()
                        failures.append(f"Task {task_idx} seed {seed}: {type(e).__name__}: {e}")
                        try:
                            save_task_error(output_dir, task_idx, seed, e)
                        except Exception as save_error:
                            print(f"Could not write task error record: {save_error}")
                    _rtpt_step(args)
        finally:
            _release_prompter(prompter)
            del prompter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return failures


def run_vlp_ablation_tasks(task_indices, seeds, args):
    return run_vlp_tasks(task_indices, seeds, args, initial_variables=_VLP_ABLATION_VARIABLES)


def run_vlp_symbolic_tasks(task_indices, seeds, args):
    return run_vlp_tasks(task_indices, seeds, args, symbolic=True)


def run_vlp_uncertainty_tasks(task_indices, seeds, args):
    return run_vlp_tasks(task_indices, seeds, args, uncertainty=True)


def run_random_zendo_symbolic_tasks(task_indices, seeds, args):
    from zendoworld.synthesis.dsl import zendo, zendo_extended
    from zendoworld.datasets.create_programs import convert_prolog_to_dsl
    from zendoworld.synthesis.grammar import dsl
    from zendoworld.synthesis.predictions.loader import __build_generic_zendo_model
    from zendoworld.game.game import play_game_state
    from zendoworld.game.game_master import ZendoStateGameMaster
    from zendoworld.game.player import RandomZendoPlayer

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
        name="weights/synthesis/bigramsPredictor_variable.weights",
    )

    tasks = load_zendo_dataset(args.dataset)
    output_dir = Path(args.output_dir or f"outputs/gamestates/gamestates_{player_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        prompter = get_prompter(args.prompter_model, "zendo", seed, reasoning=False, sampling=False,
                                use_vllm=args.use_vllm,
                                reasoning_effort=args.reasoning_effort)
        try:
            for task_idx in task_indices:
                name, examples, images = tasks[task_idx]
                print(f"\n[{player_tag}] Starting task {task_idx} (seed {seed}): {name}")
                with compute_log.use_log(
                    compute_log.seed_log_path(player_tag, seed, task_idx=task_idx),
                    args=_args_dict(args),
                    scope={"player": player_tag, "seed": seed},
                ):
                    if already_done(output_dir, task_idx, seed, symbolic=not uses_image_paths(args.player)):
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
                            images=uses_image_paths(args.player),
                            use_images=uses_image_paths(args.player),
                            ask_for_counter=args.ask_for_counter,
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
                            images=uses_image_paths(args.player),
                            gs_threshold=-1,
                            vision_model=None,
                            genai_client=None,
                            use_dsl=False,
                            seed=seed,
                        )
                        cache_path = f"outputs/cached_states/zendo_cache_{player_tag}_task_{task_idx}_seed_{seed}.pkl"
                        state = play_game_state(gm, [player], cached=False, path=cache_path)
                        save_state(state, output_dir, task_idx, seed, symbolic=not uses_image_paths(args.player))
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


def run_random_zendo_tasks(task_indices, seeds, args):
    """Visual counterpart of run_random_zendo_symbolic_tasks.

    Same setup as run_zendo_tasks (rendered images, vision model recovering the
    scene from pixels), but the player proposes uniformly random structures
    instead of searching, so its EIG is comparable to the other visual runs.
    """
    from zendoworld.synthesis.dsl import zendo, zendo_extended
    from zendoworld.datasets.create_programs import convert_prolog_to_dsl
    from zendoworld.synthesis.grammar import dsl
    from zendoworld.synthesis.predictions.loader import __build_generic_zendo_model
    from zendoworld.vision.zendo_classification.zendo_detection.model import (
        ZendoImageToVectorModel,
    )
    from zendoworld.game.game import play_game_state
    from zendoworld.game.game_master import ZendoStateGameMaster
    from zendoworld.game.player import RandomZendoPlayer

    player_tag = "random_zendo"

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
    ckpt_path = VISION_WEIGHTS
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
        name="weights/synthesis/bigramsPredictor_variable.weights",
    )

    tasks = load_zendo_dataset(args.dataset)
    output_dir = Path(args.output_dir or f"outputs/gamestates/gamestates_{player_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        prompter = get_prompter(args.prompter_model, "zendo", seed, reasoning=False, sampling=False,
                                use_vllm=args.use_vllm,
                                reasoning_effort=args.reasoning_effort)
        try:
            for task_idx in task_indices:
                name, examples, images = tasks[task_idx]
                print(f"\n[{player_tag}] Starting task {task_idx} (seed {seed}): {name}")
                with compute_log.use_log(
                    compute_log.seed_log_path(player_tag, seed, task_idx=task_idx),
                    args=_args_dict(args),
                    scope={"player": player_tag, "seed": seed},
                ):
                    if already_done(output_dir, task_idx, seed, symbolic=not uses_image_paths(args.player)):
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
                            images=uses_image_paths(args.player),
                            use_images=uses_image_paths(args.player),
                            ask_for_counter=args.ask_for_counter,
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
                            images=uses_image_paths(args.player),
                            gs_threshold=-1,
                            vision_model=visionmodel,
                            genai_client=None,
                            use_dsl=False,
                            seed=seed,
                        )
                        cache_path = f"outputs/cached_states/zendo_cache_{player_tag}_task_{task_idx}_seed_{seed}.pkl"
                        state = play_game_state(gm, [player], cached=False, path=cache_path)
                        save_state(state, output_dir, task_idx, seed, symbolic=not uses_image_paths(args.player))
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
    from zendoworld.synthesis.dsl import zendo_extended
    from zendoworld.datasets.create_programs import convert_prolog_to_dsl
    from zendoworld.synthesis.grammar import dsl
    from zendoworld.synthesis.predictions.loader import __build_generic_zendo_model
    from zendoworld.game.config import AdvZendoConfig
    from zendoworld.game.game import play_game_state
    from zendoworld.game.game_master import ZendoStateGameMaster
    from zendoworld.game.scientist_player import LLMScientistPlayer

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
        name="weights/synthesis/bigramsPredictor_variable.weights",
    )

    tasks = load_zendo_dataset(args.dataset)
    output_dir = Path(args.output_dir or "outputs/gamestates/gamestates_scientist_qwen")
    output_dir.mkdir(parents=True, exist_ok=True)

    def run_one_task(task_idx, seed, prompter):
        name, examples, images = tasks[task_idx]
        seed_dir = output_dir / f"seed_{seed}"
        error_path = seed_dir / f"task_{task_idx}_error.log"
        print(f"\n[scientist] Starting task {task_idx} (seed {seed}): {name}")
        with compute_log.use_log(
            compute_log.seed_log_path("scientist", seed, task_idx=task_idx),
            args=_args_dict(args),
            scope={"player": "scientist", "seed": seed},
        ):
            if already_done(output_dir, task_idx, seed, symbolic=not uses_image_paths(args.player)):
                print(f"Skipping task {task_idx} seed {seed} — already exists.")
                error_path.unlink(missing_ok=True)
                _rtpt_step(args)
                return None
            try:
                program = convert_prolog_to_dsl(name, cfg)
                gm = ZendoStateGameMaster(
                    true_program=program,
                    task_idx=task_idx,
                    dataset=examples.copy(),
                    paths=images.copy(),
                    zendo_dsl=zendo_dsl_extended,
                    cfg=cfg,
                    images=uses_image_paths(args.player),
                    use_images=uses_image_paths(args.player),
                    ask_for_counter=args.ask_for_counter,
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
                    use_paths=uses_image_paths(args.player),
                    images=uses_image_paths(args.player),
                    seed=seed,
                )
                cache_path = f"zendo_cache_scientist_task_{task_idx}_seed_{seed}.pkl"
                state = play_game_state(
                    gm, [player], cached=False, path=cache_path
                )
                save_state(state, output_dir, task_idx, seed, symbolic=not uses_image_paths(args.player))
                error_path.unlink(missing_ok=True)
            except Exception as e:
                failure = f"Task {task_idx} seed {seed} failed: {type(e).__name__}: {e!r}"
                trace = traceback.format_exc()
                print(failure, flush=True)
                print(trace, flush=True)
                seed_dir.mkdir(parents=True, exist_ok=True)
                error_path.write_text(f"{failure}\n\n{trace}")
                _rtpt_step(args)
                return failure
            _rtpt_step(args)
            return None

    failures = []
    for seed in seeds:
        prompter = get_prompter(args.prompter_model, "zendo", seed, reasoning=False, sampling=False,
                                use_vllm=args.use_vllm,
                                reasoning_effort=args.reasoning_effort)
        try:
            concurrency = max(1, int(getattr(args, "task_concurrency", 1)))
            if concurrency <= 1:
                for task_idx in task_indices:
                    failure = run_one_task(task_idx, seed, prompter)
                    if failure is not None:
                        failures.append(failure)
            else:
                # Multiple tasks in flight, ONE shared prompter/model instance
                # (loaded once — no extra GPU memory per worker). The
                # prompter's own lock (src/zendoworld/backends/gemma3/prompter.py,
                # src/zendoworld/backends/qwen3/prompter.py) serializes the actual generate()
                # calls, so this doesn't run two GPU forward passes at once;
                # it fills the GPU-idle gaps between calls (Prolog subprocess
                # calls, particle-filter math, Blender rendering) with
                # another task's work instead of leaving the GPU idle.
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [executor.submit(run_one_task, task_idx, seed, prompter)
                               for task_idx in task_indices]
                    for future in as_completed(futures):
                        failure = future.result()
                        if failure is not None:
                            failures.append(failure)
        finally:
            _release_prompter(prompter)
            del prompter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return failures

def run_scientist_symbolic_tasks(task_indices, seeds, args):
    from zendoworld.synthesis.dsl import zendo_extended
    from zendoworld.datasets.create_programs import convert_prolog_to_dsl
    from zendoworld.synthesis.grammar import dsl
    from zendoworld.synthesis.predictions.loader import __build_generic_zendo_model
    from zendoworld.game.config import AdvZendoConfig
    from zendoworld.game.game import play_game_state
    from zendoworld.game.game_master import ZendoStateGameMaster
    from zendoworld.game.scientist_player import LLMScientistPlayer

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
        name="weights/synthesis/bigramsPredictor_variable.weights",
    )

    tasks = load_zendo_dataset(args.dataset)
    player_tag = "scientist_symbolic"
    if getattr(args, "scene_parser", "neural") == "symbolic":
        player_tag += "_parser"
    output_dir = Path(args.output_dir or f"outputs/gamestates/gamestates_{player_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        prompter = get_prompter(args.prompter_model, "zendo", seed, reasoning=False, sampling=False,
                                use_vllm=args.use_vllm,
                                reasoning_effort=args.reasoning_effort)
        try:
            for task_idx in task_indices:
                name, examples, images = tasks[task_idx]
                print(f"\n[scientist] Starting task {task_idx} (seed {seed}): {name}")
                with compute_log.use_log(
                    compute_log.seed_log_path(player_tag, seed, task_idx=task_idx),
                    args=_args_dict(args),
                    scope={"player": player_tag, "seed": seed},
                ):
                    if already_done(output_dir, task_idx, seed, symbolic=not uses_image_paths(args.player)):
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
                            images=uses_image_paths(args.player),
                            use_images=uses_image_paths(args.player),
                            ask_for_counter=args.ask_for_counter,
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
                            use_paths=uses_image_paths(args.player),
                            images=uses_image_paths(args.player),
                            seed=seed,
                            scene_parser=getattr(args, "scene_parser", "neural"),
                        )
                        cache_path = f"zendo_cache_{player_tag}_task_{task_idx}_seed_{seed}.pkl"
                        state = play_game_state(
                            gm, [player], cached=False, path=cache_path
                        )
                        save_state(state, output_dir, task_idx, seed, symbolic=not uses_image_paths(args.player))
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
    "random_zendo": run_random_zendo_tasks,
    "random_zendo_symbolic": run_random_zendo_symbolic_tasks,
}


def worker(player_type, work_items, args, gpu_id=None, worker_id=None):
    if gpu_id is not None:
        # Restrict this worker to a single physical GPU *before* torch/CUDA is
        # touched, so parallel workers spread across the available GPUs
        # instead of all loading a model copy onto the same one.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # The parent RTPT title is not inherited after multiprocessing "spawn".
    # Give each GPU-owning worker its own title so process-based monitoring
    # (including Grafana/NVML dashboards) attributes compute to this run.
    n_iterations = len(work_items)
    if _RTPT is not None and n_iterations > 0:
        worker_label = player_type
        if worker_id is not None:
            worker_label += f"_w{worker_id}"
        args.rtpt = _RTPT(
            name_initials=args.name,
            experiment_name=worker_label,
            max_iterations=n_iterations,
        )
        args.rtpt.start()
    runner = PLAYER_RUNNERS[player_type]
    # Runners accept separate task and seed lists and evaluate their Cartesian
    # product. Group this worker's explicit (task, seed) assignments by seed so
    # it runs exactly those pairs, without recreating work assigned elsewhere.
    tasks_by_seed = {}
    for task_idx, seed in work_items:
        tasks_by_seed.setdefault(seed, []).append(task_idx)
    failures = []
    for seed, seed_tasks in tasks_by_seed.items():
        reported_failures = runner(seed_tasks, [seed], args)
        if reported_failures:
            failures.extend(reported_failures)
    if failures:
        raise RuntimeError("\n".join(failures))
    return work_items

def work_item_worker(player_type, work_items, args, gpu_id=None):
    """Run assigned pairs and propagate the runner's failures."""
    return worker(player_type, work_items, args, gpu_id=gpu_id)


def _split_task_seed_work(task_indices, seeds, worker_count):
    """Distribute the task × seed Cartesian product round-robin."""
    work_items = [
        (task_idx, seed)
        for seed in seeds
        for task_idx in task_indices
    ]
    chunks = [[] for _ in range(min(worker_count, len(work_items)))]
    for i, work_item in enumerate(work_items):
        chunks[i % len(chunks)].append(work_item)
    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified Zendo experiment runner")
    parser.add_argument(
        "--scene-parser", choices=["neural", "symbolic"], default="neural",
        help="Use deterministic scene parsing for vlp_symbolic/scientist_symbolic. "
             "Default neural preserves the existing pipeline; other model calls are unchanged.",
    )
    parser.add_argument(
        "--player",
        type=str,
        required=True,
        choices=["zendo", "fullgpt", "vlp", "vlp_ablation", "vlp_symbolic", "vlp_uncertainty", "scientist", "zendo_symbolic", "fullgpt_symbolic", "scientist_symbolic", "random_zendo", "random_zendo_symbolic"],
        help="Player type to run",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of parallel task/seed worker PROCESSES (default: 1). Each process loads "
             "its own full copy of a local model — safe for API backbones "
             "(GPT/Gemini) and for Gemma when run with --use-vllm (each "
             "process is then just a lightweight HTTP client against the "
             "shared vLLM server pool, no local model copy) — but do NOT use "
             "this for other local backbones (Qwen) or for Gemma WITHOUT "
             "--use-vllm; use --task-concurrency instead there.",
    )
    parser.add_argument(
        "--task-concurrency",
        type=int,
        default=1,
        help="Number of tasks to run concurrently WITHIN one process, "
             "sharing a single loaded model/prompter (currently wired up "
             "for --player scientist only). Increases GPU utilization by "
             "overlapping one task's CPU-only work (Prolog subprocess "
             "calls, particle-filter math, Blender rendering) with another "
             "task's model.generate() call, instead of leaving the GPU idle "
             "between calls — the prompter's internal lock still serializes "
             "the actual GPU calls one at a time, so this does not increase "
             "GPU memory usage. Start at 2; going much higher mostly adds "
             "CPU/Blender contention without more GPU-idle time to reclaim.",
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
        "--run-namespace",
        type=str,
        default=None,
        help="Namespace below outputs/rendering for render files. The default "
             "is derived from the player, model, reasoning effort, and output directory.",
    )
    openai_key_group = parser.add_mutually_exclusive_group()
    openai_key_group.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key for this launch. Prefer --openai-api-key-file so "
             "the secret is not stored in shell history.",
    )
    openai_key_group.add_argument(
        "--openai-api-key-file",
        default=None,
        help="Path to a file containing the OpenAI API key for this launch.",
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
        "--ask-for-counter",
        action="store_true",
        help="When disprove_guess_via_prolog can't find a counterexample to a "
             "wrong guess automatically, prompt on stdin to enter one by hand "
             "(piece-by-piece) instead of silently crediting the player with a "
             "win. Requires an attended/foreground run.",
    )
    parser.add_argument(
        "--prompter-model",
        type=str,
        default="gpt-5-mini",
        help=(
            "LLM/VLM backend to use for prompts. Supported values include "
            "'gpt-5.6-luna', 'gpt-5-mini', 'gpt-5', 'gpt-4o', "
            "'gpt-5-chat-latest', "
            "'Qwen2.5-VL-7B-Instruct', 'Qwen3-VL-30B-A3B-Instruct', "
            "'Qwen3-VL-30B-A3B-Thinking', 'gemma-3-4b-it', 'gemma-3-12b-it', "
            "'gemma-3-27b-it' (see src/zendoworld/backends/prompters.py for the full list)."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default="low",
        help="Reasoning effort for GPT API models (default: low).",
    )
    parser.add_argument(
        "--memory-fallback-reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default=None,
        help="Read responses from this reasoning effort's cache when the "
             "primary cache has no usable response. New responses are still "
             "written only to the primary reasoning-effort cache.",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Call an already-running Qwen3.5 or Gemma vLLM "
             "server pool (see the "
             "ZENDO_VLLM_URLS env var) over its OpenAI-compatible API instead "
             "of loading the model into this process. Unlike the local-"
             "transformers backend, this has no per-process GPU memory cost "
             "and no local call-serializing lock, so --threads is safe to use "
             "with Gemma when this flag is set -- many concurrent processes "
             "can share the same server pool, with vLLM's own continuous "
             "batching handling concurrent requests.",
    )
    parser.add_argument(
        "--rewind-timeouts-from", type=str,
        help="Source result directory for symbolic VLP games to rewind to their first trailing 300s timeout. Requires a separate --output-dir.",
    )
    parser.add_argument(
        "--rewind-cache-dir", type=str, default="outputs/cached_states",
        help="Directory containing the original symbolic VLP caches.",
    )
    args = parser.parse_args()
    if args.rewind_timeouts_from:
        if args.player != "vlp_symbolic" or args.scene_parser != "symbolic":
            parser.error("--rewind-timeouts-from requires --player vlp_symbolic --scene-parser symbolic")
        if not args.output_dir or Path(args.output_dir).resolve() == Path(args.rewind_timeouts_from).resolve():
            parser.error("Rewinding requires an --output-dir different from the source directory")
    if args.scene_parser == "symbolic" and args.player not in ("vlp_symbolic", "scientist_symbolic"):
        parser.error("--scene-parser symbolic requires --player vlp_symbolic or scientist_symbolic")
    if args.memory_fallback_reasoning_effort:
        os.environ["ZENDO_MEMORY_FALLBACK_REASONING_EFFORT"] = (
            args.memory_fallback_reasoning_effort
        )
    else:
        os.environ.pop("ZENDO_MEMORY_FALLBACK_REASONING_EFFORT", None)
    _configure_openai_api_key(args)
    _validate_parallel_backend(args)

    namespace_identity = "|".join(
        [
            args.player,
            args.prompter_model,
            args.reasoning_effort,
            os.path.abspath(args.output_dir) if args.output_dir else "default-output",
        ]
    )
    if args.scene_parser == "symbolic":
        namespace_identity += "|scene-parser=symbolic"
    namespace_label = re.sub(
        r"[^A-Za-z0-9_.-]+", "_", args.run_namespace or args.player
    ).strip("._-") or "run"
    if args.run_namespace is None:
        namespace_label += "_" + hashlib.sha256(namespace_identity.encode()).hexdigest()[:12]
    os.environ["ZENDO_RUN_NAMESPACE"] = namespace_label
    args.run_namespace = namespace_label
    print(f"Render namespace: outputs/rendering/{namespace_label}")

    # Set up compute logging. If ZENDO_COMPUTE_LOG is already set (e.g. user
    # exported it), respect that path; otherwise pick a timestamped default
    # under outputs/logs/ and export so worker subprocesses inherit it.
    if not os.environ.get("ZENDO_COMPUTE_LOG"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log = os.path.abspath(
            f"outputs/logs/compute_{args.player}_{args.prompter_model}_{stamp}.jsonl"
        )
        os.environ["ZENDO_COMPUTE_LOG"] = default_log
    log_dir = os.path.dirname(os.environ["ZENDO_COMPUTE_LOG"])
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    print(f"Compute log: {os.environ['ZENDO_COMPUTE_LOG']}")
    fatal_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    os.environ["ZENDO_FATAL_ERROR_FILE"] = os.path.abspath(
        f"outputs/logs/fatal_api_{args.player}_{fatal_stamp}_{os.getpid()}.json"
    )
    compute_log.start(args=_args_dict(args))
    atexit.register(compute_log.end)

    tasks = load_zendo_dataset(args.dataset)
    num_tasks = len(tasks)
    print(f"Loaded {num_tasks} tasks from {args.dataset}")

    if args.tasks is not None:
        task_indices = [t for t in args.tasks if 0 <= t < num_tasks]
    else:
        task_indices = list(range(num_tasks))

    if not task_indices:
        raise ValueError("No valid tasks selected")
    task_indices = list(dict.fromkeys(task_indices))
    args.seeds = list(dict.fromkeys(args.seeds))
    # Retain output-based locks used by current local launches, as well as
    # upstream namespace locks, so either version detects overlapping work.
    output_namespace = hashlib.sha256(str(Path(
        args.output_dir or f"outputs/gamestates/gamestates_{args.player}"
    ).resolve()).encode()).hexdigest()[:20]
    work_locks = _acquire_work_locks(output_namespace, task_indices, args.seeds)
    if args.run_namespace != output_namespace:
        work_locks.extend(_acquire_work_locks(args.run_namespace, task_indices, args.seeds))
    atexit.register(lambda: [handle.close() for handle in work_locks])
    Path("outputs/cached_states").mkdir(parents=True, exist_ok=True)

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
        failures = runner(task_indices, args.seeds, args)
        if failures:
            raise RuntimeError("\n".join(failures))
    else:
        chunks = _split_task_seed_work(task_indices, args.seeds, args.threads)
        print(
            f"Splitting {n_iterations} task/seed pairs into "
            f"{len(chunks)} chunks:"
        )
        for i, chunk in enumerate(chunks):
            print(f"  Worker {i}: {chunk}")

        # Spread workers round-robin across the GPUs named in
        # CUDA_VISIBLE_DEVICES (e.g. "2,3") instead of letting every worker
        # default to the first one -- each worker pins itself to a single
        # entry from this list before touching CUDA (see worker()).
        visible = [g.strip() for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
        gpu_for_worker = (lambda i: visible[i % len(visible)]) if visible else (lambda i: None)
        if visible:
            print(f"Pinning workers round-robin across GPUs: {visible}")

        # "spawn" avoids forking a process that has already imported torch/CUDA --
        # forked children can inherit a broken CUDA context and die silently
        # (BrokenProcessPool with no traceback) on their first GPU touch.
        mp_context = multiprocessing.get_context("spawn")
        # Some container/Python launch combinations execute this file with
        # ``__name__ == "__main__"`` while bootstrapping a spawned worker.
        # Mark the child environment explicitly so it imports worker() without
        # recursively running main() and contending with the parent's locks.
        child_marker = "ZENDO_PROCESS_POOL_CHILD"
        previous_child_marker = os.environ.get(child_marker)
        os.environ[child_marker] = "1"
        executor = ProcessPoolExecutor(max_workers=len(chunks), mp_context=mp_context)
        # RTPT is process-local and not reliably pickleable with spawn.
        worker_args = argparse.Namespace(**_args_dict(args))
        worker_args.rtpt = None
        abort_pool = False
        pool_errors = []
        try:
            try:
                futures = {
                    executor.submit(
                        worker,
                        args.player,
                        chunk,
                        worker_args,
                        gpu_for_worker(i),
                        i,
                    ): i
                    for i, chunk in enumerate(chunks)
                }
            finally:
                if previous_child_marker is None:
                    os.environ.pop(child_marker, None)
                else:
                    os.environ[child_marker] = previous_child_marker
            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    completed_work = future.result()
                    print(f"Worker {worker_id} finished pairs {completed_work}")
                    if args.rtpt is not None:
                        for _ in completed_work:
                            args.rtpt.step()
                except OpenAIQuotaExhausted as e:
                    abort_pool = True
                    print(f"Fatal OpenAI quota error from worker {worker_id}: {e}")
                    for pending in futures:
                        pending.cancel()
                    raise
                except Exception as e:
                    print(f"Worker {worker_id} crashed: {e}")
                    traceback.print_exc()
                    pool_errors.append(f"Worker {worker_id}: {e}")
        finally:
            # Grab the worker handles *before* shutdown() -- it clears
            # executor._processes to None as part of its own teardown, so
            # reading it afterward is unreliable.
            worker_procs = list((getattr(executor, "_processes", None) or {}).values())
            if abort_pool:
                for proc in worker_procs:
                    if proc.is_alive():
                        proc.terminate()
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True, cancel_futures=True)
            # When a pool "breaks" (any one worker dying marks the whole pool
            # broken), concurrent.futures abandons the other, still-alive
            # workers instead of terminating them -- they leak indefinitely,
            # each still holding its loaded model in GPU memory. Force-kill
            # any stragglers as a safety net so a failed run can't orphan
            # processes on a shared GPU.
            for proc in worker_procs:
                if proc.is_alive():
                    print(f"Force-killing straggler worker process pid={proc.pid}")
                    proc.kill()
                    proc.join(timeout=5)
        if pool_errors:
            raise RuntimeError(
                f"{len(pool_errors)} worker(s) reported failures:\n"
                + "\n".join(pool_errors)
            )

    missing_outputs = []
    if args.output_dir is not None:
        missing_outputs = [
            (seed, task_idx)
            for seed in args.seeds
            for task_idx in task_indices
            if not already_done(Path(args.output_dir), task_idx, seed, symbolic=not uses_image_paths(args.player))
        ]
    if missing_outputs:
        raise RuntimeError(
            f"Experiment finished with {len(missing_outputs)} missing output(s): "
            f"{missing_outputs}"
        )

    print("\nAll done.")


if __name__ == "__main__" and os.environ.get("ZENDO_PROCESS_POOL_CHILD") != "1":
    main()
