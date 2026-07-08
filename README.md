# zendo-model
This repository holds all necessary code to run a vision based zendo player, including a trained object detection model.

## Dataset
To test the Zendo player or train new predictors, either use the available datasets:
```bash
data/combined_dataset_short.pkl
```
or download the data from https://huggingface.co/datasets/ss567uhg/zendo-synthetic-data and use the create_dataset scripts in the data directory.

## Setup
Create a conda environment:
```bash
conda env create -f environment.yml
conda activate zendo-model
```
> [!IMPORTANT]  
> SWI-Prolog must be installed and properly set up on your system for the generation process to work. Without it, the scripts will be unable to execute the project's Prolog logic.

### Install Blender
```bash
cd /opt
wget https://download.blender.org/release/Blender4.4/blender-4.4.0-linux-x64.tar.xz

tar -xf blender-4.4.0-linux-x64.tar.xz

export PATH=/opt/blender-4.4.0-linux-x64:$PATH

/opt/blender-4.4.0-linux-x64/4.4/python/bin/python3.11 -m pip install pyyaml torch
```

## API Key
To run the VLM-based agents you need to add your Open-AI API or Gemini API key into ./models/gpt/open-ai-key-vlp or ./models/gemini/key.txt

## Test
You can now let any agent play the game:
```bash
export PYTHONHASHSEED=0
python run_experiment.py --player <agent> --prompter-model <model>
```

## Repository layout
Where to look when working on a given piece of the system.

### Top-level entry points
- `run_experiment.py` — main experiment driver; sets up a game master + the chosen player and runs Zendo tasks.
- `experiment_helper.py` — helpers used by `run_experiment.py`: program checkers (with/without accuracy), grammar merging, dataset adapters (`task_set2zendodataset`, …).
- `call_vision_model.py` — runs the trained Zendo vision model on a single image and decodes its outputs into a piece tensor.
- `call_generate_prolog.py` — thin CLI wrapper around `rules.rules.generate_prolog_structure` (used as a subprocess).
- `create_programs_from_string.py` — parses DSL rule strings (e.g. `AT_LEAST_1 1 IS_RED`, S-expressions, AND/OR composites) into `program.Function` ASTs.
- `create_predictions.py` — script that walks JSON/JSONL prediction records.
- `model_loader.py` — builds the **program-synthesis** predictors (`BigramsPredictor`, `RulesPredictor`) over the Zendo DSL/CFG and optionally loads saved weights (e.g. `variable+simple+bigrams_zendo.weights`).
- `compute_log.py` — append-only JSONL compute log for paper-ready reporting (render events, token counts). Worker subprocesses append via `ZENDO_COMPUTE_LOG`; includes a `--summarize` mode.
- `utils.py` — text-side helpers: parse LLM listed outputs, extract DSL from hypothesis strings, convert tensors ↔ `ZendoStructure`/`ZendoPiece`.
- `grammar_splitter.py` — splits a PCFG into shards for parallel search.
- `zendo_model.pt` — trained weights for the vision model invoked from `call_vision_model.py`.

### Game logic — `zendo/`
- `game.py` — `play_game_state` outer loop and difficulty heuristic.
- `game_master.py` — `ZendoStateGameMaster`: maintains the true Prolog rule, renders scenes, judges proposed examples, applies semantic-equivalence canonicalisation.
- `player.py` — base/`Random`/`VLP`/`FullGPT` players and Prolog-subprocess helpers (`call_prolog_subprocess_with_retries`, `normalize_rule`).
- `scientist_player.py` — `LLMScientistPlayer`, derived from the *Doing Experiments and Revising Rules* repo (Top Piriyakulkij); uses a particle filter (`scipy.optimize`, `scipy.stats`).
- `states.py` — `GameState`, `Turn`, `step`, and `GameCache` (pickled to `cached_states/`; the directory is auto-created).
- `utils.py` — systematic resampling for particle filters.

### Program synthesis core
- `program.py` — `Program`, `Function`, `Variable`, `BasicPrimitive`, `New`, `Lambda` with deterministic hashing (requires `PYTHONHASHSEED=0`).
- `program_as_list.py` — stack-based evaluation of compressed program lists.
- `type_system.py` — `Type`, `PolymorphicType`, `PrimitiveType`, `Arrow`, `List`, `BOOL/INT/STRING/IMG/OBJECT/…`.
- `cons_list.py` — cons-list environment used during program evaluation.
- `grammar/` (derived from DeepSynth) — `cfg.py`, `pcfg.py`, `pcfg_logprob.py`, `dsl.py`. PCFG sampling uses `vose`; `pcfg.py` also has an optional `sbsur` path.
- `Algorithms/` (derived from DeepSynth) — search algorithms: `heap_search.py`, `threshold_search.py`, `parallel.py` (multiprocessing), `ray_parallel.py` (Ray).
- `Predictions/` (derived from DeepSynth) — neural guide for the search: `models.py` (`RulesPredictor`, `BigramsPredictor`, `NNDictRulesPredictor`), `embeddings.py`, `IOencodings.py`, `dataset_sampler.py`.

### DSLs — `DSL/`
- `zendo.py`, `zendo_extended.py` — the symbolic Zendo DSL (predicates over pieces: colour/shape/orientation indices, touch/on-top/point relations, AND/OR rules, interaction predicates).
- `vlp_dsl.py`, `vlp_dsl_symbolic.py` — visual / symbolic variants used by VLP players.
- `minimal_DSL.py` — reduced DSL built on top of `zendo.py` primitives.

### Rule generation — `rules/`
- `rules.py` (uses `pyswip` / SWI-Prolog) — `generate_prolog_structure`, placeholder/template machinery for Prolog rules. Invoked as a subprocess via `call_generate_prolog.py`.

### Scene generation — `generation/`
Mixed: `render.py` runs in the conda env and `subprocess`-launches Blender, which executes the rest with its bundled Python. `bpy` / `mathutils` / `bpy_extras` only resolve inside Blender.
- `render.py` — conda-side launcher (`blender --background --python render_single.py ...`), loads the resulting tensor.
- `render_single.py` — entry point Blender runs.
- `structure.py`, `zendo_objects.py` — scene primitives (`Pyramid`, `Block`, `Wedge`, …).
- `generate.py` — DSL/Prolog → scene graph; uses `world_to_camera_view`.
- `encoding.py` — converts a scene description into a tensor with the project's lexicons.
- `utils.py` — Blender arg parsing.

### Data — `data/`
- `create_dataset.py`, `create_test_dataset.py` — assemble pickled datasets.
- `create_programs.py` — Prolog query ↔ DSL conversion.
- `create_prolog.py` — DSL `Function` → Prolog string.
- `pieces2tensor.py`, `tensor2piece.py` — piece-string ↔ tensor with the project's `COLOR_MAP` / `SHAPE_MAP` / `ORIENTATION_MAP`.
- `combined_dataset.pkl`, `combined_dataset_short.pkl`, `training_tasks*.pkl` — pre-built datasets.
- `dataset_flat/` — flat-file dataset assets.

### Players / VLMs — `models/`
One subfolder per backend, each exposing a `main.py` with a `*Prompter` class:
- `gpt/` (`GPTPrompter`), `gemini/` (`google.generativeai`), `qwen/` (`Qwen2_5Prompter`), `qwen3/` (`Qwen3Prompter`), `gemma3/` (`Gemma3Prompter`), `internvl/` (`InternVLPrompter`), `kimi/` (`KimiPrompter`).
- `prompters.py` — `get_prompter(model, dataset, seed, …)` dispatch (imports each backend lazily).
- `prompts/zendo/` — prompt templates (`player`, `perception`, `rule_conversion`).

### Vision model — `vision_model/zendo_classification/`
Training + evaluation pipeline for the image→structure model used at game time.
- `zendo_detection/` — model, dataset loaders, encoder, training loop (`ZendoImageToVectorModel`, `ZendoLightweightModel`, `ZendoSimpleModel`, `ZendoYOLODataset`, …).
- `train_model.py`, `run_training.py` — train from scratch.
- `finetune_model.py`, `run_finetuning.py` — fine-tune existing weights.
- `evaluate_model_tensors.py`, `test_model_tensors.py`, `complete_evaluation.py`, `analyze_experiments.py`, `data_analysis.py`, `plot_loss.py` — evaluation / analysis.

### Configuration
- `zendo_conf.py` — `AdvZendoConfig`: attribute names, allowed values for colours/shapes/orientations/groundedness/touching/pointing/stacking and natural-language specs used in prompts.
- `conf/` — Hydra-style config (`config.yaml`, `agent/`, `dataset/`).

### Experiment runners — `experiments/`
- `run_experiment.py` (DeepSynth-derived) — `gather_data`, `list_algorithms`, `run_algorithm`, Ray-based ML-guided synthesis loop.
- `run_experiment_zendo.py` — Zendo-specific entry point.

### Analysis & evaluation — `analysis/`
EIG computation (`compute_eig.py`), evaluation ( `evaluate_players.py`, `evaluate_by_difficulty.py`).

### Runtime artefacts (created on demand, gitignored)
- `cached_states/` — pickled `GameCache` per task/seed (auto-created by `zendo/states.py`).
- `gamestates/`, `logs/`, `prompt_logs/` — per-run logging output.

### Environment
- `environment-complete.yml` — full conda + pip lockfile.
- `Dockerfile` — container image.

