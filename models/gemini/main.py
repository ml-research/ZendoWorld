import os
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
from PIL import Image
from google import genai
from google.genai import types

try:
    import compute_log
except ImportError:
    compute_log = None

QUOTA_RETRY_FILE = Path("logs/gemini_retry_after.txt")
QUOTA_EXIT_CODE = 75  # EX_TEMPFAIL: signals the restart-loop script to sleep and retry


def _extract_retry_seconds(error_text: str):
    """Pull the suggested retry delay out of a Gemini RESOURCE_EXHAUSTED error."""
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", error_text)
    if m:
        return float(m.group(1))
    m = re.search(r"retry in\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?", error_text)
    if m and any(m.groups()):
        hours, minutes, seconds = (float(g) if g else 0.0 for g in m.groups())
        return hours * 3600 + minutes * 60 + seconds
    return None


def _handle_quota_error(e: Exception):
    """On quota exhaustion, record the retry delay and kill the process outright.

    Callers throughout this codebase wrap prompter calls in broad
    `except Exception` blocks, so a normal raised exception gets swallowed
    long before it reaches the process boundary and a restart-loop script
    could see it. Exiting the process directly is the only way to guarantee
    the caller doesn't silently limp along issuing doomed requests for hours."""
    error_text = str(e)
    if "RESOURCE_EXHAUSTED" not in error_text and "429" not in error_text:
        return False
    retry_seconds = _extract_retry_seconds(error_text)
    QUOTA_RETRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    QUOTA_RETRY_FILE.write_text(str((retry_seconds or 3600) + 60))
    print(f"Gemini quota exhausted, retry after ~{retry_seconds}s. Exiting process.", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(QUOTA_EXIT_CODE)


class GeminiPrompter:
    def __init__(self, model="gemini-3.5-flash", dataset="default", seed=42):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            key_file = "models/gemini/key.txt"
            if not os.path.exists(key_file):
                raise RuntimeError(
                    "No Google API key found. Set the GOOGLE_API_KEY environment variable "
                    "or create models/gemini/key.txt."
                )
            with open(key_file, "r") as f:
                api_key = f.read().strip()

        self.client = genai.Client(api_key=api_key)

        self.model_name = model
        self.seed = seed
        self.execution_counter = 0
        self.produced_tokens = 0
        self.generated_response = {}

        print(f"USING MODEL: {model}")

        self.memory: Dict[Tuple, str] = {}
        self.memory_file = f"models/gemini/memory/{dataset}/vlm_memory_{model}_{seed}.json"
        self._load_memory()

        self.token_file = f"models/gemini/tokens/{dataset}/vlm_tokens_{model}_{seed}.json"
        self._load_tokens()

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                data = json.load(f)
                self.memory = {eval(k): v for k, v in data.items()}

    def _load_tokens(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if os.path.exists(self.token_file):
            with open(self.token_file, "r") as f:
                self.token_usage = json.load(f)
            self.token_usage[self.timestamp] = 0
        else:
            os.makedirs(os.path.dirname(self.token_file), exist_ok=True)
            self.token_usage = {self.timestamp: 0}

    def _save_memory(self):
        memory_dict = {str(k): v for k, v in self.memory.items()}
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, "w") as f:
            json.dump(memory_dict, f, indent=4)

    def get_produced_tokens(self):
        return self.produced_tokens

    def reset_produced_tokens(self):
        self.produced_tokens = 0

    def prompt_with_text(
        self,
        prompt_text: str,
        system_prompt=None,
        seed=None,
        use_memory=True,
        max_new_tokens=2000,
        do_sample=True,
        overwrite_memory=False,
    ):
        if use_memory:
            key = (prompt_text, seed)
            if key in self.memory:
                cached = self.memory[key]
                if not overwrite_memory and cached != "":
                    return cached
                elif cached == "":
                    print("Cached response is empty, re-generating.")
                else:
                    print("Overwriting memory for this prompt...")
            else:
                print("NO MEMORY FOR THIS PROMPT.")

        if seed is None:
            seed = self.seed

        response_text = ""
        try:
            config = types.GenerateContentConfig(
                max_output_tokens=max_new_tokens,
                system_instruction=system_prompt,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            )
            result = self.client.models.generate_content(
                model=self.model_name, contents=prompt_text, config=config
            )
            response_text = result.text.strip() if result.text else ""
            output_tokens = result.usage_metadata.candidates_token_count or 0
            self.token_usage[self.timestamp] += output_tokens
            with open(self.token_file, "w") as f:
                json.dump(self.token_usage, f, indent=4)
            if compute_log is not None:
                compute_log.record_tokens(output_tokens, model=self.model_name)
        except Exception as e:
            if _handle_quota_error(e):
                raise
            print(f"Error in prompt_with_text: {e}")

        if use_memory and response_text != "":
            self.memory[key] = response_text
            self._save_memory()

        return response_text

    def prompt_with_images(
        self,
        prompt_text: str,
        paths: [str],
        system_prompt=None,
        seed=None,
        use_memory=True,
        max_new_tokens=2000,
        overwrite_memory=False,
        **kwargs,
    ):
        if use_memory:
            path_string = ",".join(paths)
            key = (prompt_text, path_string)
            if key in self.memory and not overwrite_memory:
                if type(self.memory[key]) == dict:
                    if self.memory[key]["response"] != "":
                        if key not in self.generated_response:
                            self.generated_response[key] = True
                            self.produced_tokens += self.memory[key]["num_tokens"]
                        return self.memory[key]["response"]
                    else:
                        print("Memorized response is empty, re-generating.")

        if seed is None:
            seed = self.seed

        print(f"Processing prompt with {len(paths)} images.")

        response_text = ""
        output_tokens = 0
        try:
            images = [Image.open(p) for p in paths]
            config = types.GenerateContentConfig(
                max_output_tokens=max_new_tokens,
                system_instruction=system_prompt,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            )
            result = self.client.models.generate_content(
                model=self.model_name,
                contents=images + [prompt_text],
                config=config,
            )
            response_text = result.text.strip() if result.text else ""
            output_tokens = result.usage_metadata.candidates_token_count or 0
            print(f"Output tokens: {output_tokens}")
            self.token_usage[self.timestamp] += output_tokens
            with open(self.token_file, "w") as f:
                json.dump(self.token_usage, f, indent=4)
        except Exception as e:
            if _handle_quota_error(e):
                raise
            print(f"Error: {e}")

        self.execution_counter += 1

        if use_memory:
            self.memory[key] = {"response": response_text, "num_tokens": output_tokens}
            self._save_memory()

        self.produced_tokens += output_tokens
        if compute_log is not None:
            compute_log.record_tokens(output_tokens, model=self.model_name)

        return response_text


if __name__ == "__main__":
    prompter = GeminiPrompter(model="gemini-2.5-flash", dataset="default", seed=42)
    image_path = "data/CLEVR-Hans3/train/images/CLEVR_Hans_classid_0_000000.png"
    response = prompter.prompt_with_images(
        "Describe the image in detail.", [image_path]
    )
    print("Response:", response)
