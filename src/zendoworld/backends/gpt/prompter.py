from zendoworld.paths import CREDENTIALS_DIR
import base64
import fcntl
import os
import json
import tempfile
from pathlib import Path
from openai import OpenAI
from datetime import datetime
from typing import Dict, Tuple
from PIL import Image
from io import BytesIO

try:
    import zendoworld.experiments.compute_log as compute_log
except ImportError:
    compute_log = None


from zendoworld.backends.errors import OpenAIQuotaExhausted


class GPTPrompter:
    def __init__(
        self,
        model="gpt-4o",
        dataset="default",
        seed=42,
        reasoning=True,
        reasoning_effort=None,
    ):
        # Load the API key from the environment variable OPENAI_API_KEY,
        # falling back to the key file for local development.
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            key_file = str(CREDENTIALS_DIR / "gpt" / "open-ai-key-vlp")
            if not os.path.exists(key_file):
                raise RuntimeError(
                    "No OpenAI API key found. Set the OPENAI_API_KEY environment variable "
                    "or create configs/credentials/gpt/open-ai-key-vlp."
                )
            with open(key_file, "r") as f:
                api_key = f.read().strip()

        self.api_key = api_key
        # A quota error is permanent until billing changes. Disable the SDK's
        # generic 429 retries so it is surfaced after the first failed request.
        self.client = OpenAI(api_key=api_key, max_retries=0)
        if model == "gpt-4o":
            model = "gpt-4o-2024-08-06"
        if model == "o1":
            model = "o1-2024-12-17"
        if model == "gpt-5-mini":
            model = "gpt-5-mini-2025-08-07"
        else:
            model = model

        self.model = model
        self.seed = seed
        self.system_fingerprint = None
        self.execution_counter = 0
        self.produced_tokens = 0
        self.generated_response = {}
        self.reasoning = reasoning
        self.reasoning_effort = reasoning_effort or (
            "high" if reasoning else "low"
        )
        valid_efforts = {"none", "low", "medium", "high", "xhigh", "max"}
        if self.reasoning_effort not in valid_efforts:
            raise ValueError(
                f"Unsupported reasoning effort {self.reasoning_effort!r}; "
                f"choose one of {sorted(valid_efforts)}"
            )

        print(f"USING MODEL: {model}")
        print(f"USING REASONING EFFORT: {self.reasoning_effort}")

        self.memory: Dict[Tuple[str, str], str] = (
            {}
        )  # Dictionary to store (prompt, image_path) -> response
        effort_suffix = (
            f"_{self.reasoning_effort}" if reasoning_effort is not None else ""
        )
        self.memory_file = (
            f"outputs/backends/gpt/memory/{dataset}/"
            f"vlm_memory_{model}{effort_suffix}_{seed}.json"
        )
        self.fallback_memory = {}
        fallback_effort = os.environ.get("ZENDO_MEMORY_FALLBACK_REASONING_EFFORT")
        if fallback_effort and fallback_effort != self.reasoning_effort:
            fallback_file = (
                f"outputs/backends/gpt/memory/{dataset}/"
                f"vlm_memory_{model}_{fallback_effort}_{seed}.json"
            )
            self.fallback_memory = self._read_memory_file(fallback_file)
            print(
                f"Loaded {len(self.fallback_memory)} usable {fallback_effort} "
                "memory entries as a read-only fallback."
            )
        self._load_memory()

        self.token_file = (
            f"outputs/backends/gpt/tokens/{dataset}/"
            f"vlm_tokens_{model}{effort_suffix}_{seed}.json"
        )
        self._load_tokens()

    @staticmethod
    def _quota_error_details(error):
        """Return the API error payload when this is exhausted quota."""
        body = getattr(error, "body", None)
        if not isinstance(body, dict):
            body = {}
        nested = body.get("error")
        if isinstance(nested, dict):
            body = nested

        code = str(body.get("code") or getattr(error, "code", "") or "").lower()
        error_type = str(body.get("type") or "").lower()
        message = str(body.get("message") or error).lower()
        quota_codes = {
            "insufficient_quota",
            "billing_hard_limit_reached",
            "billing_limit_reached",
            "billing_not_active",
        }
        message_matches = (
            "exceeded your current quota",
            "insufficient quota",
            "billing hard limit",
            "billing limit",
            "out of credits",
        )
        if (
            code in quota_codes
            or error_type == "insufficient_quota"
            or any(text in message for text in message_matches)
        ):
            return {
                "code": code or error_type or "insufficient_quota",
                "message": str(body.get("message") or error),
                "status_code": getattr(error, "status_code", None),
                "request_id": getattr(error, "request_id", None),
            }
        return None

    @staticmethod
    def _fatal_error_path():
        return os.environ.get("ZENDO_FATAL_ERROR_FILE")

    def _check_fatal_error(self):
        path = self._fatal_error_path()
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    details = json.load(f)
                message = details.get("message", "OpenAI API quota exhausted")
            except (OSError, json.JSONDecodeError):
                message = "OpenAI API quota exhausted in another worker"
            raise OpenAIQuotaExhausted(message)

    def _raise_if_quota_exhausted(self, error):
        details = self._quota_error_details(error)
        if details is None:
            return

        path = self._fatal_error_path()
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                pass
            else:
                with os.fdopen(fd, "w") as f:
                    json.dump(details, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
        raise OpenAIQuotaExhausted(details["message"])

    @staticmethod
    def _cached_response(value):
        """Return a non-empty cached response and its recorded token count."""
        if isinstance(value, dict):
            response = value.get("response")
            tokens = value.get("num_tokens", 0)
        else:
            response = value
            tokens = 0
        if not isinstance(response, str) or not response.strip():
            return None
        return response, tokens if isinstance(tokens, int) else 0

    @classmethod
    def _read_memory_file(cls, path):
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            data = json.load(f)
        return {
            eval(key): value
            for key, value in data.items()
            if cls._cached_response(value) is not None
        }

    def _lookup_memory(self, key):
        for source, memory in (("primary", self.memory), ("fallback", self.fallback_memory)):
            cached = self._cached_response(memory.get(key))
            if cached is not None:
                return (*cached, source)
        return None

    def _load_memory(self):
        """Load only usable responses from the writable primary memory."""
        self.memory = self._read_memory_file(self.memory_file)

    def _load_tokens(self):
        """Loads token usage from a file if it exists."""
        # Include PID and microseconds so parallel workers never share a
        # counter key even when they start during the same second.
        self.timestamp = (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            + f" pid={os.getpid()}"
        )

        if os.path.exists(self.token_file):
            with open(self.token_file, "r") as f:
                self.token_usage = json.load(f)
            # Add current timestamp
            self.token_usage[self.timestamp] = 0

        else:
            # create directories if not exist
            os.makedirs(os.path.dirname(self.token_file), exist_ok=True)
            self.token_usage = {self.timestamp: 0}

    @staticmethod
    def _merge_json_file(path, values):
        """Merge a dict into a JSON file without losing concurrent updates."""
        parent = os.path.dirname(path)
        os.makedirs(parent, exist_ok=True)
        lock_path = path + ".lock"
        with open(lock_path, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                try:
                    with open(path, "r") as f:
                        merged = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    merged = {}
                merged.update(values)

                fd, tmp_path = tempfile.mkstemp(
                    prefix=f".{Path(path).name}.", suffix=".tmp", dir=parent
                )
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(merged, f, indent=4)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_path, path)
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                return merged
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _save_memory(self):
        """Saves memory to a file, converting tuple keys to strings for JSON storage."""
        memory_dict = {str(k): v for k, v in self.memory.items()}
        merged = self._merge_json_file(self.memory_file, memory_dict)
        self.memory = {eval(k): v for k, v in merged.items()}

    def _save_tokens(self):
        # Only this process owns its PID-qualified session key. Merging just
        # that counter avoids overwriting counters held by sibling workers.
        merged = self._merge_json_file(
            self.token_file, {self.timestamp: self.token_usage[self.timestamp]}
        )
        self.token_usage = merged

    def get_produced_tokens(self):
        return self.produced_tokens

    def reset_produced_tokens(self):
        self.produced_tokens = 0

    def _reasoning_output_budget(self, requested):
        """Leave enough room for both reasoning tokens and the visible answer."""
        configured = int(os.environ.get("ZENDO_GPT_MAX_OUTPUT_TOKENS", "8192"))
        return max(int(requested), configured)

    @staticmethod
    def _incomplete_reason(response):
        details = getattr(response, "incomplete_details", None)
        if details is None:
            return None
        if isinstance(details, dict):
            return details.get("reason")
        return getattr(details, "reason", None)

    def _create_reasoning_response(self, input_content, max_new_tokens):
        """Retry once with twice the budget when reasoning exhausts the limit."""
        budget = self._reasoning_output_budget(max_new_tokens)
        total_output_tokens = 0
        for attempt in range(2):
            response = self.client.responses.create(
                model=self.model,
                input=input_content,
                reasoning={"effort": self.reasoning_effort},
                max_output_tokens=budget,
            )
            usage = getattr(response, "usage", None)
            total_output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
            reason = self._incomplete_reason(response)
            response_text = (getattr(response, "output_text", "") or "").strip()
            should_retry = reason == "max_output_tokens" or not response_text
            if not should_retry:
                return response, total_output_tokens
            if attempt == 0:
                next_budget = budget * 2
                cause = reason or "empty output_text"
                print(
                    f"Response returned {cause} with max_output_tokens={budget}; "
                    f"retrying once with {next_budget}."
                )
                budget = next_budget
        return response, total_output_tokens

    def prompt(self, prompt_text, system_prompt=None, seed=None, temp=None):
        """Generate a response to a prompt using the OpenAI API."""

        self._check_fatal_error()

        if seed is None:
            seed = self.seed

        # if system_prompt is None:
        #     system_prompt = "You are a helpful assistant that can describe images provided by the user in extreme detail. You are able to recognize abstract concepts in images like humans do. You are helping a scientist discover relevant patterns in images."

        # Call the completion endpoint
        token_kwarg = {"max_completion_tokens": 2000} if "gpt-5" in self.model else {"max_tokens": 2000}
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": system_prompt,
                            }
                        ],
                    },
                    {"role": "user", "content": prompt_text},
                ],
                model=self.model,
                seed=seed,
                **token_kwarg,
                temperature=temp,
            )
        except Exception as e:
            self._raise_if_quota_exhausted(e)
            raise
        # Get system fingerprint
        self.system_fingerprint = response.system_fingerprint
        self.model = response.model

        response_text = response.choices[0].message.content.strip()

        # self._log_prompt(prompt_text, response_text)

        return response_text

    def _encode_image(self, image_path, resize=False):

        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"No such file: {image_path}")

        # Always go through PIL to ensure consistent format
        with Image.open(p) as img:
            # # Convert to RGB for JPEG to avoid errors with PNG/PNG+alpha
            # if format.upper() == "JPEG" and img.mode != "RGB":
            #     img = img.convert("RGB")

            if resize:
                size = (224, 224)
                # High quality downscale
                img = img.resize(size, Image.Resampling.LANCZOS)

            buf = BytesIO()
            save_kwargs = {}
            # if format.upper() == "JPEG":
            #     # Reasonable defaults for JPEG
            #     save_kwargs.update({"quality": 90, "optimize": True})
            img.save(buf, format="PNG", **save_kwargs)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return b64

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
            cached = None if overwrite_memory else self._lookup_memory(key)
            if cached is not None:
                response, _tokens, source = cached
                if source == "fallback":
                    print("Retrieving response from fallback reasoning memory.")
                return response
            if overwrite_memory:
                print("Overwriting memory for this prompt...")
            else:
                print("NO MEMORY FOR THIS PROMPT.")

        if seed is None:
            seed = self.seed

        try:
            self._check_fatal_error()
            if "gpt-5" in self.model and not self.model == "gpt-5-chat-latest":
                # Use the responses API for gpt-5 reasoning models (same as prompt_with_images)
                input_content = [{"role": "user", "content": prompt_text}]
                response_obj, output_tokens = self._create_reasoning_response(
                    input_content, max_new_tokens
                )
                response_text = getattr(response_obj, "output_text", "").strip()
                if compute_log is not None:
                    compute_log.record_tokens(output_tokens, model=self.model)
                self.token_usage[self.timestamp] += output_tokens
                self._save_tokens()
            else:
                token_kwarg = {"max_tokens": max_new_tokens}
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": prompt_text},
                    ],
                    model=self.model,
                    seed=seed,
                    **token_kwarg,
                )
                response_text = response.choices[0].message.content
                response_text = response_text.strip() if response_text else ""
        except Exception as e:
            self._raise_if_quota_exhausted(e)
            print(f"Error in prompt_with_text: {e}")
            response_text = ""

        if use_memory and response_text != "":
            # Only cache non-empty responses to avoid poisoning retries
            self.memory[key] = response_text
            self._save_memory()

        return response_text

    def prompt_with_images(
        self,
        prompt_text: str,
        paths: [str],
        system_prompt=None,
        seed=None,
        url=False,
        use_memory=True,
        max_new_tokens=2000,
        overwrite_memory=False,
        thinking=False,
    ):

        # # overwrite_memory = True
        if use_memory:
            path_string = ",".join(paths)
            key = (prompt_text, path_string)
            cached = None if overwrite_memory else self._lookup_memory(key)
            if cached is not None:
                response, tokens, source = cached
                if key not in self.generated_response:
                    self.generated_response[key] = True
                    self.produced_tokens += tokens
                if source == "fallback":
                    print("Retrieving response from fallback reasoning memory.")
                return response

        if seed is None:
            seed = self.seed

        print(f"Processing prompt with {len(paths)} images.")

        # encode images
        resize = len(paths) > 1
        encoded_images = [self._encode_image(path, resize=resize) for path in paths]

        # build input content
        input_content = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt_text}],
            }
        ]

        for image in encoded_images:
            input_content[-1]["content"].append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image}",
                }
            )

        try:
            self._check_fatal_error()

            if "gpt-5" in self.model and self.model != "gpt-5-chat-latest":
                print(f"Using reasoning effort: {self.reasoning_effort}")
                response, output_tokens = self._create_reasoning_response(
                    input_content, max_new_tokens
                )
            else:
                response = self.client.responses.create(
                    model=self.model,
                    input=input_content,
                    max_output_tokens=max_new_tokens,
                    temperature=0.2,
                )

            # print(response)
            if response == "":
                print("EMPTY RESPONSE!")

            # extract model output
            response_text = getattr(response, "output_text", "")
            if not ("gpt-5" in self.model and self.model != "gpt-5-chat-latest"):
                output_tokens = response.usage.output_tokens
            # reasoning_tokens = (
            #     response.usage.output_tokens_details.reasoning_tokens
            #     if hasattr(response.usage.output_tokens_details, "reasoning_tokens")
            #     else 0
            # )

            response_text = response_text.strip()

            print("Output tokens details:", output_tokens)
            self.token_usage[self.timestamp] += output_tokens
            self._save_tokens()
            # print(f"Reasoning tokens: {reasoning_tokens}")

            # get number of tokens in response
            # output_tokens = getattr(response, "output_tokens", 0)
            all_tokens = output_tokens

        except Exception as e:
            self._raise_if_quota_exhausted(e)
            print("Error:", e)
            response_text = ""
            all_tokens = 0

        self.execution_counter += 1

        if use_memory and response_text != "":
            # Failed or token-exhausted responses must be retried, not cached.
            self.memory[key] = {"response": response_text, "num_tokens": all_tokens}
            self._save_memory()

        self.produced_tokens += all_tokens
        if compute_log is not None:
            compute_log.record_tokens(all_tokens, model=self.model)

        return response_text

    def prompt_with_parts(
        self,
        parts,
        system_prompt=None,
        seed=None,
        use_memory=True,
        max_new_tokens=2000,
        overwrite_memory=False,
    ):
        """Send an ordered sequence of text/image parts as a single
        multimodal message, preserving order -- so an image sits directly
        after the text label for its own group instead of being described
        only by a single upfront count. ``parts`` is a list of
        {"type": "text", "text": ...} / {"type": "image", "path": ...} dicts.
        """
        # Cache key must reflect the actual order of text/image parts, not
        # just their combined contents, since re-ordering changes the prompt.
        cache_key_text = "\x1f".join(
            p["text"] if p["type"] == "text" else f"<image:{p['path']}>"
            for p in parts
        )
        key = (cache_key_text, "")
        if use_memory:
            cached = None if overwrite_memory else self._lookup_memory(key)
            if cached is not None:
                response, tokens, source = cached
                if key not in self.generated_response:
                    self.generated_response[key] = True
                    self.produced_tokens += tokens
                if source == "fallback":
                    print("Retrieving response from fallback reasoning memory.")
                return response

        if seed is None:
            seed = self.seed

        n_images = sum(1 for p in parts if p["type"] == "image")
        print(f"Processing interleaved prompt with {n_images} images.")
        resize = n_images > 1

        content = []
        for p in parts:
            if p["type"] == "text":
                content.append({"type": "input_text", "text": p["text"]})
            else:
                encoded = self._encode_image(p["path"], resize=resize)
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encoded}",
                })
        input_content = [{"role": "user", "content": content}]

        try:
            self._check_fatal_error()
            if "gpt-5" in self.model and self.model != "gpt-5-chat-latest":
                print(f"Using reasoning effort: {self.reasoning_effort}")
                response, output_tokens = self._create_reasoning_response(
                    input_content, max_new_tokens
                )
            else:
                response = self.client.responses.create(
                    model=self.model,
                    input=input_content,
                    max_output_tokens=max_new_tokens,
                    temperature=0.2,
                )

            response_text = getattr(response, "output_text", "").strip()
            if not ("gpt-5" in self.model and self.model != "gpt-5-chat-latest"):
                output_tokens = response.usage.output_tokens
            self.token_usage[self.timestamp] += output_tokens
            self._save_tokens()
            all_tokens = output_tokens
        except Exception as e:
            self._raise_if_quota_exhausted(e)
            print("Error:", e)
            response_text = ""
            all_tokens = 0

        self.execution_counter += 1
        if use_memory and response_text != "":
            # Failed or token-exhausted responses must be retried, not cached.
            self.memory[key] = {"response": response_text, "num_tokens": all_tokens}
            self._save_memory()

        self.produced_tokens += all_tokens
        if compute_log is not None:
            compute_log.record_tokens(all_tokens, model=self.model)

        return response_text


if __name__ == "__main__":
    prompter = GPTPrompter(model="gpt-5-mini", dataset="cocologic", seed=42)
    prompt = "Describe the image in detail."
    image_path = "data/CLEVR-Hans3/train/images/CLEVR_Hans_classid_0_000000.png"

    response = prompter.prompt_with_images(
        prompt,
        [image_path, "data/CLEVR-Hans3/train/images/CLEVR_Hans_classid_0_000001.png"],
    )
    print("Response:", response)
