# ZendoWorld

Visual and symbolic Zendo agents, scene rendering, and program synthesis. The synthesis implementation includes code derived from DeepSynth.

## Setup

Create the experiment environment and install the package from the repository root:

```bash
conda env create -n zendo -f environments/complete.yml
conda activate zendo
python -m pip install -e . --no-deps
```

SWI-Prolog and Blender must be installed for scene generation. Blender uses its own Python interpreter; install the rendering dependencies (`pyyaml`, `torch`, and `pillow`) into that interpreter as well. If Blender needs an additional Python package directory, set `BLENDER_PYTHONPATH` to that directory.

For starting Docker, reusing an existing image, creating a container, and activating Conda, see [the Docker testing instructions](docker/README.md).

The Dockerfile builds the experiment environment:

```bash
docker build -f docker/Dockerfile -t zendoworld .
```

Mount or copy the repository into the container and install the package there with `python -m pip install -e . --no-deps` in the `zendo` environment. Run vLLM in a separate server environment; the experiment image does not install vLLM.

## Run experiments

Run these commands from the repository root:

```bash
export PYTHONHASHSEED=0
zendoworld run --player vlp --prompter-model gpt-5-mini --threads 4 --seeds 1
```

The module command is equivalent:

```bash
python -m zendoworld run --player vlp --prompter-model gpt-5-mini --threads 4 --seeds 1
```

Use `zendoworld run --help` for player choices, symbolic inputs, task selection, and concurrency options. `--threads` controls worker processes; `--task-concurrency` controls concurrent tasks within a process for the scientist player. Remote API models and supported vLLM servers can share requests across workers; local model instances require care with GPU memory.

API keys can be supplied through `OPENAI_API_KEY` or `GOOGLE_API_KEY`. Existing file-based credentials live under `configs/credentials/gpt/` and `configs/credentials/gemini/`, which are ignored by Git. GPT uses `open-ai-key-vlp`; Gemini uses `key_my.txt`. The OpenAI key can also be provided using `--openai-api-key-file`.

The default dataset is `data/combined_dataset_short.pkl`. Supply another dataset with `--dataset`. Additional data is available from [the synthetic Zendo dataset](https://huggingface.co/datasets/ss567uhg/zendo-synthetic-data).

## Shared vLLM server

Run a configured vLLM server separately from the experiment environment. For example, in your existing vLLM environment:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve google/gemma-4-31B-it \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --enforce-eager
```

Then run the workers in the experiment environment:

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONHASHSEED=0 \
ZENDO_VLLM_URLS=http://127.0.0.1:8000 \
zendoworld run \
  --player fullgpt \
  --prompter-model gemma-4-31B-it \
  --use-vllm \
  --threads 4 \
  --seeds 73 \
  --dataset data/combined_dataset_short.pkl
```

GPU indices must match those visible inside your container. The example reserves the third visible GPU for Blender.

## Repository layout

```text
src/zendoworld/
  synthesis/      Programs, types, grammars, DSLs, search algorithms, predictors
  game/           Game state, game master, players, parsing, conversions
  backends/       GPT, Gemini, local VLMs, and the vLLM client
  prompts/        Prompt templates
  rendering/      Scene generation, Blender launcher, bundled assets and config
  rules/          Prolog logic and rule generation
  datasets/       Dataset creation and representation conversion
  vision/         Image-to-structure model, training, and evaluation
  experiments/    Game experiment runner, helpers, compute logging
  analysis/       EIG computation, player evaluation, and reporting
  cli/            Command dispatch and Prolog subprocess entry point
configs/          Experiment configuration and player plot styles
                  Local credentials are stored in configs/credentials/
environments/     Experiment and vision environment definitions
docker/           Container build definition
tests/            Regression and integration tests
data/             Existing dataset files
weights/vision/   Trained image-to-structure weights
outputs/          Generated renders, logs, caches, figures, and backend memory
```

The package uses lowercase names throughout. `backends/*/prompter.py` implements each model client. The game runner is `experiments/runner.py`; the synthesis search loop is `synthesis/search_runner.py`.

Python code imports through `zendoworld`, for example:

```python
from zendoworld.synthesis.program import Program
from zendoworld.game.input_mode import uses_image_paths
```

The package resolves bundled Prolog and Blender resources from their installed locations. Vision weights and local credentials are resolved from the checkout; set `ZENDOWORLD_HOME` to your project directory when using an installed wheel outside the checkout. Dataset arguments and explicit result paths remain under your control. Existing pickle datasets and game caches are read with a compatibility loader that maps their old module names to the new package.

## Analysis and verification

```bash
zendoworld compute-log --summarize outputs/logs/compute_run.jsonl --out outputs/summary.json
zendoworld compute-eig --help
PYTHONHASHSEED=0 python -m pytest -q
```

Other tools can be run by module, for example `python -m zendoworld.analysis.evaluate_by_difficulty`. Runtime files are ignored under `outputs/`; source code stays under `src/`.

## Website demo

The standalone GitHub Pages demo is in `demo-src/`, and the published website files are in `docs/`. See [local preview and development instructions](demo-src/README.md). To preview the website from the repository root, run `python -m http.server 8080 --bind 127.0.0.1 --directory docs` and open http://localhost:8080/. No Docker or model backend is required.
