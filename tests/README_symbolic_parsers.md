Run the input/output tests from the repository root in the experiment Python environment:

```bash
PYTHONHASHSEED=0 python -m pytest tests/test_symbolic_parsers.py -q
```

Select the parsers with the existing symbolic runners:

```bash
python -m zendoworld run --player vlp_symbolic --scene-parser symbolic --seeds 1 --tasks 0
python -m zendoworld run --player scientist_symbolic --scene-parser symbolic --seeds 1 --tasks 0
```

Add your usual model and dataset arguments. Omitting `--scene-parser` (or using
`--scene-parser neural`) retains the previous pipeline. The option applies only
to these two symbolic runners; visual runs are unchanged.

VLP parses `item(...)` descriptions into vocabulary-filtered object/property
lists and interaction triples. Discovery and the other model calls remain neural.
Scientist parses descriptions into `ZendoStructure.pieces`, resolving relation IDs
to piece indices. Its runner keeps tensors internally and converts them to the
same scene-description format used in its prompts before parsing. The existing
description format encodes one relation per item; the parser cannot recover
additional relations omitted by that description. Hypothesis generation and
translation, scene proposals, and the rest of the pipeline are unchanged.

The Scientist default already uses `tensor_to_zendo_structure`, not an LLM, for
this conversion. The new option selects the description parser in its place.

Parser runs default to `outputs/gamestates/gamestates_vlp_symbolic_parser` and
`outputs/gamestates/gamestates_scientist_symbolic_parser`, with separate cache/log tags.
An explicit `--output-dir` still takes precedence.
# Rewind the GPT-mini symbolic timeout runs

The final caches can reconstruct these games because all turns after the first
300-second proposal failure only added one GM fallback observation each.
`--rewind-timeouts-from` validates the cache against the corresponding published
result, removes that suffix, restores the GM fallback queue and player state,
and retries the first failed turn. It rejects mixed successful/failed suffixes,
unfinished discovery, inconsistent histories, and non-300-second failures.

Run in the project's normal Python environment, from the repository root:

```bash
PYTHONHASHSEED=0 python -m zendoworld run \
  --player vlp_symbolic --scene-parser symbolic \
  --prompter-model gpt-5-mini --reasoning-effort low \
  --openai-api-key-file vlp-openai \
  --dataset data/combined_dataset_28.pkl \
  --seeds 1 2 3 --tasks 22 23 26 27 --threads 1 \
  --rewind-timeouts-from outputs/gamestates/gamestates_vlp_symbolic \
  --rewind-cache-dir outputs/cached_states \
  --output-dir outputs/gamestates/gamestates_vlp_gpt_mini_symbolic_resumed
```

Original results and caches remain untouched. New caches go under the output
directory's `resume_caches/`; results record `resume_provenance`. The output
contains only the 12 continued games, not a copy of the unaffected games.
Completed results are skipped on another invocation. If a continuation crashes,
another invocation starts from the original rewind point, not its partial cache.

| Task | Seed 1 restart | Seed 2 restart | Seed 3 restart |
|---|---:|---:|---:|
| 22 | 10 | 7 | 7 |
| 23 | 10 | 7 | 7 |
| 26 | 10 | 7 | 10 |
| 27 | 11 | 8 | 10 |

Earlier observations and guesses retain their original labels and durations.
Only the continuation uses the corrected parser. Random-generator state was not
saved, so this is a continuation of the recovered history, not an exact replay.
The restored search has up to three independent 300-second attempts per search.

Lightweight rewind and search checks (no PyTorch needed):

```bash
PYTHONHASHSEED=0 python3 -m unittest tests.test_vlp_resume tests.test_search_regressions
```
