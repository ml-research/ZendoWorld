

def get_prompter(
    model,
    dataset,
    seed,
    reasoning=False,
    sampling=True,
    use_vllm=False,
    reasoning_effort=None,
):
    if use_vllm and model in ["Qwen3.8-27B", "gemma-4-31B-it"]:
        from zendoworld.backends.vllm_client import VLLMPrompter

        return VLLMPrompter(model=model, dataset=dataset, seed=seed, sampling=sampling)
    if model in ["InternVL3-8B", "InternVL3-14B", "InternVL3-78B"]:
        from zendoworld.backends.internvl.prompter import InternVLPrompter

        return InternVLPrompter(
            model=model, dataset=dataset, seed=seed, sampling=sampling
        )
    elif model in ["Qwen2.5-VL-7B-Instruct"]:
        from zendoworld.backends.qwen.prompter import Qwen2_5Prompter

        return Qwen2_5Prompter(
            model=model, dataset=dataset, seed=seed, sampling=sampling
        )
    elif model in ["Qwen3-VL-30B-A3B-Instruct", "Qwen3-VL-30B-A3B-Thinking", "Qwen3.5-27B"]:
        from zendoworld.backends.qwen3.prompter import Qwen3Prompter

        return Qwen3Prompter(model=model, dataset=dataset, seed=seed, sampling=sampling)
    elif model in ["Kimi-VL-A3B-Thinking-2506", "Kimi-VL-A3B-Instruct"]:
        from zendoworld.backends.kimi.prompter import KimiPrompter

        return KimiPrompter(model=model, dataset=dataset, seed=seed, sampling=sampling)

    elif model in ["Kimi-VL-A3B-Thinking-2506", "Kimi-VL-A3B-Instruct", "Kimi-K2.6"]:
        from zendoworld.backends.kimi.prompter import KimiPrompter

        return KimiPrompter(model=model, dataset=dataset, seed=seed, sampling=sampling)

    elif model.startswith("gemma"):
        # Matched by prefix (not an enumerated tuple of exact IDs) so this
        # covers every entry in gemma3/main.py's _MODEL_NAME_MAP -- Gemma 3
        # (gemma-3-4b-it, -12b-it, -27b-it) and Gemma 4 (gemma-4-31B-it,
        # gemma-4-E2B-it) alike -- without this dispatch needing to be kept
        # in sync with that map by hand every time a new variant is added.
        if use_vllm:
            from zendoworld.backends.gemma3.prompter import Gemma3VLLMPrompter

            return Gemma3VLLMPrompter(model=model, dataset=dataset, seed=seed, sampling=sampling)

        from zendoworld.backends.gemma3.prompter import Gemma3Prompter

        return Gemma3Prompter(model=model, dataset=dataset, seed=seed, sampling=sampling)
    elif model.startswith("gpt-"):
        from zendoworld.backends.gpt.prompter import GPTPrompter

        return GPTPrompter(
            model=model,
            dataset=dataset,
            seed=seed,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
        )
    elif model.startswith("gemini"):
        from zendoworld.backends.gemini.prompter import GeminiPrompter

        return GeminiPrompter(model=model, dataset=dataset, seed=seed)
    else:
        raise ValueError(f"Model {model} not supported.")
