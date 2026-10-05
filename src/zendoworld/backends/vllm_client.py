"""OpenAI-compatible client for the local multimodal vLLM servers."""

import ast
from contextlib import ExitStack
import fcntl
import tempfile
import warnings
import base64
import json
import os
import random
import threading
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image

try:
    import zendoworld.experiments.compute_log as compute_log
except ImportError:
    compute_log = None


MODEL_NAMES = {
    "Qwen3.8-27B": "Qwen/Qwen3.8-27B",
    "gemma-4-31B-it": "google/gemma-4-31B-it",
}


class VLLMPrompter:
    def __init__(self, model, dataset=None, seed=0, sampling=True):
        from openai import OpenAI

        if model not in MODEL_NAMES:
            raise ValueError(f"Unsupported vLLM model: {model!r}")
        self.model_name = MODEL_NAMES[model]
        self.seed = seed
        self.sampling = sampling
        urls = os.environ.get("ZENDO_VLLM_URLS", "http://localhost:8000")
        self._clients = [
            OpenAI(base_url=f"{url.strip().rstrip('/')}/v1", api_key="not-needed")
            for url in urls.split(",")
            if url.strip()
        ]
        if not self._clients:
            raise ValueError("ZENDO_VLLM_URLS must contain at least one URL")
        self.memory: Dict[Tuple[str, str], str] = {}
        self.produced_tokens = 0
        self._memory_lock = threading.Lock()
        dataset_name = "bongard-op" if dataset and "bongard-op" in dataset else dataset
        self.memory_file = (
            Path("outputs/backends/vllm_memory") / str(dataset_name)
            / f"vlm_memory_{model}_sampling_{seed}.json"
        )
        self._load_memory()

    def _read_memory(self):
        try:
            with self.memory_file.open() as handle:
                return {ast.literal_eval(key): value for key, value in json.load(handle).items()}
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, SyntaxError, TypeError, AttributeError) as exc:
            warnings.warn(f"Ignoring invalid vLLM cache {self.memory_file}: {exc}")
            return {}

    def _load_memory(self):
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.memory = self._read_memory()

    def _save_memory(self, updates=None):
        with self._memory_lock:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            # Coordinate with both existing local workers (*.lock) and
            # upstream workers (*.json.lock), without changing cache paths.
            lock_paths = (self.memory_file.with_suffix(".lock"),
                          self.memory_file.with_suffix(self.memory_file.suffix + ".lock"))
            with ExitStack() as locks:
                for path in lock_paths:
                    handle = locks.enter_context(path.open("a+"))
                    fcntl.flock(handle, fcntl.LOCK_EX)
                merged = self._read_memory()
                merged.update(self.memory if updates is None else updates)
                with tempfile.TemporaryDirectory(dir=self.memory_file.parent) as staging:
                    temporary = Path(staging) / "memory.json"
                    with temporary.open("w") as handle:
                        json.dump({str(key): value for key, value in merged.items()}, handle, indent=2)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, self.memory_file)
                self.memory = merged

    @staticmethod
    def _encode_image(path):
        image = Image.open(path).convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def _generate(self, content, max_new_tokens, seed=None):
        response = random.choice(self._clients).chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_new_tokens,
            seed=self.seed if seed is None else seed,
            temperature=0.7,
            top_p=0.80,
            extra_body={
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            presence_penalty=1.5,
        )
        output = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        self.produced_tokens += tokens or 0
        if compute_log is not None:
            compute_log.record_tokens(tokens or 0, model=self.model_name)
        return output

    def _prompt(self, content, key, max_new_tokens, use_memory, overwrite_memory, seed=None):
        if use_memory and not overwrite_memory and key in self.memory:
            return self.memory[key]
        output = self._generate(content, max_new_tokens, seed=seed)
        if use_memory:
            self._save_memory({key: output})
        return output

    def prompt_with_text(self, prompt_text, use_memory=True, max_new_tokens=1500,
                         overwrite_memory=False, seed=None, **_ignored):
        call_seed = self.seed if seed is None else seed
        return self._prompt(
            [{"type": "text", "text": prompt_text}],
            (prompt_text, f"seed={call_seed}"), max_new_tokens, use_memory,
            overwrite_memory, seed=call_seed,
        )

    def prompt_with_images(self, prompt_text, paths, use_memory=True,
                           max_new_tokens=1500, overwrite_memory=False, seed=None, **_ignored):
        call_seed = self.seed if seed is None else seed
        content = [{"type": "text", "text": prompt_text}]
        content.extend({"type": "image_url", "image_url": {"url": self._encode_image(path)}} for path in paths)
        return self._prompt(
            content, (prompt_text, ",".join(map(str, paths)) + f"|seed={call_seed}"),
            max_new_tokens, use_memory, overwrite_memory, seed=call_seed,
        )

    def prompt_with_parts(self, parts, use_memory=True, max_new_tokens=1500,
                          overwrite_memory=False, seed=None, **_ignored):
        content = []
        cache = []
        for part in parts:
            if part["type"] == "text":
                content.append({"type": "text", "text": part["text"]})
                cache.append("<text>" + part["text"])
            elif part["type"] == "image":
                path = str(part["path"])
                content.append({"type": "image_url", "image_url": {"url": self._encode_image(path)}})
                cache.append("<image>" + path)
        key = ("<ordered-parts>\x1f" + "\x1f".join(cache), f"seed={seed if seed is not None else self.seed}")
        return self._prompt(
            content, key, max_new_tokens, use_memory, overwrite_memory, seed=seed,
        )

    def remove_from_gpu(self):
        pass

    def get_produced_tokens(self):
        return self.produced_tokens

    def reset_produced_tokens(self):
        self.produced_tokens = 0
