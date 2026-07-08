from typing import Dict, Tuple
import gc
import json
import os

# Set the allocator config before importing torch so it takes effect on the
# first CUDA init. expandable_segments reduces fragmentation across the long
# sequence of generate() calls in an experiment run.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    import compute_log
except ImportError:
    compute_log = None


_MODEL_NAME_MAP = {
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "gemma-4-31B-it": "google/gemma-4-31B-it",
    "gemma-4-E2B-it": "google/gemma-4-E2B-it",
}


class Gemma3Prompter:
    """HuggingFace Transformers prompter for Google Gemma 3 IT (multimodal) models.

    Mirrors the interface of GPTPrompter / Qwen3Prompter so it can be used as a
    drop-in replacement via models.prompters.get_prompter."""

    def __init__(
        self,
        model="gemma-4-E2B-it",
        dataset=None,
        seed=0,
        sampling=False,
    ):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.seed = seed
        self.sampling = sampling

        if model not in _MODEL_NAME_MAP:
            raise ValueError(
                f"Model {model} not supported. Choose from {list(_MODEL_NAME_MAP)}."
            )
        self.model_name = _MODEL_NAME_MAP[model]

        self.model_loaded = False
        self.memory: Dict[Tuple[str, str], str] = {}
        self.produced_tokens = 0

        # Same dataset shorthand handling as the other prompters.
        if dataset and "bongard-op" in dataset:
            dataset = "bongard-op"

        suffix = "_no_sampling" if not sampling else "_sampling"
        self.memory_file = (
            f"models/gemma3/memory/{dataset}/vlm_memory_{model}{suffix}_{seed}.json"
        )
        self._load_memory()

    # ── Memory management ────────────────────────────────────────────────────
    def _load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                data = json.load(f)
                self.memory = {eval(k): v for k, v in data.items()}
        else:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)

    def _save_memory(self):
        memory_dict = {str(k): v for k, v in self.memory.items()}
        with open(self.memory_file, "w") as f:
            f.write(json.dumps(memory_dict, indent=4))

    # ── Model lifecycle ──────────────────────────────────────────────────────
    def _load_model(self):
        print(f"Loading Gemma model: {self.model_name}")
        # Shard VLM weights across every CUDA device visible to this process.
        # We deliberately do NOT hardcode host GPU indices because docker
        # remaps them: `--gpus device=0,1,3` makes those visible inside the
        # container as 0,1,2. Control which host GPUs are exposed via the
        # docker --gpus flag (or NVIDIA_VISIBLE_DEVICES); the code uses
        # whatever lands inside the container. Blender (generation/render.py)
        # pins to the last visible device by default and shares that GPU
        # with gemma — its ~1-2 GiB easily fits alongside ~21 GiB of weights.
        # For gemma-4-31B-it (~62 GiB bf16) across 3 visible GPUs, the
        # 36 GiB/GPU cap leaves ~15 GiB headroom per GPU.
        # Override per run with ZENDO_VLM_MAX_MEMORY using *container-local*
        # indices (e.g. '{"0":"36GiB","1":"36GiB","2":"36GiB"}').
        _DEFAULT_PER_GPU = "36GiB"

        max_memory_override = os.environ.get("ZENDO_VLM_MAX_MEMORY")
        if max_memory_override:
            import json as _json
            override = {int(k): v for k, v in _json.loads(max_memory_override).items()}
            max_memory = {}
            for i in range(torch.cuda.device_count()):
                max_memory[i] = override.get(i, "0GiB")
        elif torch.cuda.is_available() and torch.cuda.device_count() >= 1:
            max_memory = {
                i: _DEFAULT_PER_GPU for i in range(torch.cuda.device_count())
            }
        else:
            max_memory = None

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            attn_implementation="sdpa",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, padding_side="left"
        )
        self.model_loaded = True

    def remove_from_gpu(self):
        if self.model_loaded:
            print("Removing Gemma3 model from GPU...")
            del self.model
            del self.processor
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            self.model_loaded = False

    def get_produced_tokens(self):
        return self.produced_tokens

    def reset_produced_tokens(self):
        self.produced_tokens = 0

    # ── Image preprocessing ──────────────────────────────────────────────────
    @staticmethod
    def _preprocess_images(paths, max_size=896):
        images = []
        for path in paths:
            img = Image.open(path).convert("RGB")
            if max_size is not None:
                img.thumbnail((max_size, max_size))
            images.append(img)
        return images

    # ── Generation ───────────────────────────────────────────────────────────
    def _generate(self, messages, max_new_tokens):
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        # Use nucleus sampling for proposal diversity. Greedy decoding produced
        # near-identical outputs across turns even when the input set of
        # example images changed, suppressing the player's ability to explore
        # the hypothesis space. The `sampling` constructor arg is now ignored
        # in favour of always sampling — run-level reproducibility is provided
        # by the torch RNG seed set once in __init__, not by per-call resets.
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        generated_trimmed = generated[0][input_len:].clone()
        n_new = int(generated_trimmed.shape[0])
        self.produced_tokens += n_new
        if compute_log is not None:
            compute_log.record_tokens(n_new, model=self.model_name)

        output_text = self.processor.decode(
            generated_trimmed, skip_special_tokens=True
        )

        # Release activations / KV cache / input tensors back to the CUDA
        # allocator so other processes (e.g. Blender Cycles on a different
        # GPU, or batched VLM calls later) see free memory instead of seeing
        # PyTorch's growing cached pool.
        del inputs, generated, generated_trimmed
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_text.strip()

    def prompt_with_text(
        self,
        prompt_text,
        use_memory=True,
        max_new_tokens=1500,
        overwrite_memory=False,
        seed=None,
        **_ignored,
    ):
        # `seed` is accepted for interface compatibility with other prompters
        # but is intentionally NOT applied per-call: resetting the RNG to the
        # same value before each generate call produced near-identical samples
        # turn after turn. Reproducibility is preserved by the one-time seed
        # set in __init__.

        key = (prompt_text, "")
        if use_memory and not overwrite_memory and key in self.memory:
            return self.memory[key]

        if not self.model_loaded:
            self._load_model()

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ]
        output_text = self._generate(messages, max_new_tokens=max_new_tokens)

        if use_memory:
            self.memory[key] = output_text
            self._save_memory()

        return output_text

    def prompt_with_images(
        self,
        prompt_text,
        paths,
        url=False,
        use_memory=True,
        max_new_tokens=1500,
        overwrite_memory=False,
        seed=None,
        **_ignored,
    ):
        # See prompt_with_text: per-call seed reset deliberately omitted to
        # preserve sampling diversity across turns.

        path_string = ",".join(str(p) for p in paths)
        key = (prompt_text, path_string)
        if use_memory and not overwrite_memory and key in self.memory:
            return self.memory[key]

        if not self.model_loaded:
            self._load_model()

        # Gemma 3 expects images as PIL via the chat template "image" entry.
        images = self._preprocess_images(paths)
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]
        output_text = self._generate(messages, max_new_tokens=max_new_tokens)

        # Release PIL refs so they don't linger in the cached memory pool
        # alongside the next call's allocations.
        del images, content, messages
        gc.collect()

        if use_memory:
            self.memory[key] = output_text
            self._save_memory()

        return output_text


if __name__ == "__main__":
    prompter = Gemma3Prompter(model="gemma-4-31B-it", dataset="zendo", seed=42, sampling=False)
    print(prompter.prompt_with_text("Hello, briefly introduce yourself.", max_new_tokens=100))
