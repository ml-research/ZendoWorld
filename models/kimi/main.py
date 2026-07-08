import gc
import json
import os
import re

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

try:
    import compute_log
except ImportError:
    compute_log = None


class KimiPrompter:
    """Local Kimi-VL prompter modeled on Qwen3Prompter.

    Two intentional deviations from models/qwen3/main.py, motivated by the
    Zendo regression analysis:
      1. skip_special_tokens=True on decode — leaving it False on qwen3 leaked
         <|im_end|> into NL rule guesses and broke downstream conversion.
      2. No 224x224 thumbnail in the multi-image branch — the qwen3 resize
         collapsed FullGPT perception. Kimi-VL has its own per-image token
         budget driven by the processor config; lower it there if VRAM is
         tight rather than reintroducing a blanket thumbnail here.
    """

    def __init__(
        self,
        model="Kimi-VL-A3B-Thinking-2506",
        dataset=None,
        seed=0,
        sampling=False,
    ):
        # Seed once at construction; never re-seed per call. Per-call resets
        # collapsed sampling diversity across turns when the same prompter
        # was reused (see qwen3/gemma3 notes).
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.seed = seed

        if model == "Kimi-VL-A3B-Thinking-2506":
            self.model_name = "moonshotai/Kimi-VL-A3B-Thinking-2506"
            self.is_thinking = True
        elif model == "Kimi-VL-A3B-Instruct":
            self.model_name = "moonshotai/Kimi-VL-A3B-Instruct"
            self.is_thinking = False
        else:
            raise ValueError(
                f"Model {model} not supported. Use 'Kimi-VL-A3B-Thinking-2506' "
                "or 'Kimi-VL-A3B-Instruct'."
            )

        self.model_loaded = False
        self.memory = {}
        self.generated_response = {}
        self.produced_tokens = 0
        self.sampling = sampling

        if dataset and "bongard-op" in dataset:
            dataset = "bongard-op"

        if not self.sampling:
            print("Using greedy decoding (no sampling) for VLM.")
            tag = model + "_no_sampling"
        else:
            tag = model + "_sampling"

        self.memory_file = (
            f"models/kimi/memory/{dataset}/vlm_memory_{tag}_{seed}.json"
        )
        self._load_memory()

    def _load_model(self):
        # Kimi-VL is shipped as a CausalLM with custom code; trust_remote_code
        # is required for both the model and the processor.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model_loaded = True

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

    def remove_from_gpu(self):
        if self.model_loaded:
            print("Removing model from GPU...")
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

    def _strip_thinking(self, text):
        if not self.is_thinking:
            return text
        return re.sub(r".*?</think>", "", text, flags=re.DOTALL).strip()

    def _decode_and_count(self, generated_ids, input_ids):
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        n_new = sum(int(t.shape[0]) for t in trimmed)
        self.produced_tokens += n_new
        if compute_log is not None:
            compute_log.record_tokens(n_new, model=self.model_name)
        # skip_special_tokens=True is deliberate; see class docstring.
        decoded = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return (decoded[0] if decoded else ""), n_new

    def prompt_with_text(
        self,
        prompt_text,
        use_memory=True,
        max_new_tokens=1500,
        overwrite_memory=False,
        seed=None,
    ):
        if use_memory:
            key = (prompt_text, "")
            if not overwrite_memory and key in self.memory:
                return self.memory[key]

        if not self.model_loaded:
            self._load_model()

        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
        ]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.sampling,
            )

        raw, _ = self._decode_and_count(generated_ids, inputs.input_ids)
        response = self._strip_thinking(raw)

        if use_memory:
            self.memory[key] = response
            self._save_memory()
        return response

    def prompt_with_images(
        self,
        prompt_text,
        paths,
        url=False,
        use_memory=True,
        max_new_tokens=1500,
        overwrite_memory=False,
        sampling=False,
        thinking=False,
        seed=None,
    ):
        if use_memory:
            key = (prompt_text, ",".join(paths))
            if not overwrite_memory and key in self.memory:
                cached = self.memory[key]
                if isinstance(cached, dict):
                    if key not in self.generated_response:
                        self.generated_response[key] = True
                        self.produced_tokens += cached.get("num_tokens", 0)
                    return cached.get("response", "")
                return cached

        if not self.model_loaded:
            self._load_model()

        print(f"Prompting model with {len(paths)} images...")
        images = [Image.open(p).convert("RGB") for p in paths]

        # Image entries point to the same PIL objects we hand to the
        # processor; Kimi-VL's chat template uses the per-entry placeholder,
        # the actual pixel pass is via images=... below.
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": p} for p in paths],
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            images=images,
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.sampling,
            )

        raw, n_new = self._decode_and_count(generated_ids, inputs.input_ids)
        response = self._strip_thinking(raw)

        del inputs, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if use_memory:
            self.memory[key] = {"response": response, "num_tokens": n_new}
            self._save_memory()
        return response


if __name__ == "__main__":
    prompter = KimiPrompter(
        model="Kimi-VL-A3B-Thinking-2506",
        dataset="clevr",
        seed=42,
        sampling=False,
    )
    response = prompter.prompt_with_images(
        "Describe the image in detail.",
        ["data/clevr/all_cubes_10/CLEVR_Hans_classid_0_000000.png"],
        max_new_tokens=500,
    )
    print(response)
