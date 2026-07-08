from models.gpt.main import GPTPrompter


def get_prompter(model, dataset, seed, reasoning=False, sampling=True):
    if model in ["InternVL3-8B", "InternVL3-14B", "InternVL3-78B"]:
        from models.internvl.main import InternVLPrompter

        return InternVLPrompter(
            model=model, dataset=dataset, seed=seed, sampling=sampling
        )
    elif model in ["Qwen2.5-VL-7B-Instruct"]:
        from models.qwen.main import Qwen2_5Prompter

        return Qwen2_5Prompter(
            model=model, dataset=dataset, seed=seed, sampling=sampling
        )
    elif model in ["Qwen3-VL-30B-A3B-Instruct", "Qwen3-VL-30B-A3B-Thinking", "Qwen3.5-27B"]:
        from models.qwen3.main import Qwen3Prompter

        return Qwen3Prompter(model=model, dataset=dataset, seed=seed, sampling=sampling)
    elif model in ["Kimi-VL-A3B-Thinking-2506", "Kimi-VL-A3B-Instruct"]:

        from kimi.main_transformers import KimiPrompter

        return KimiPrompter(model=model, dataset=dataset, seed=seed, sampling=sampling)

    elif model in ["Kimi-VL-A3B-Thinking-2506", "Kimi-VL-A3B-Instruct", "Kimi-K2.6"]:
        from models.kimi.main import KimiPrompter

        return KimiPrompter(model=model, dataset=dataset, seed=seed, sampling=sampling)

    elif model == "Molmo-7B" or model == "Molmo-72B":
        from molmo.main import MolmoPrompter

        return MolmoPrompter(model=model, dataset=dataset, seed=seed, sampling=sampling)
    elif "Ovis2.5" in model:
        from ovis.main import OvisPrompter

        return OvisPrompter(model=model, dataset=dataset, seed=seed, sampling=sampling)
    elif model in ("gemma-3-4b-it", "gemma-4-31B-it", "gemma-4-E2B-it"):
        from models.gemma3.main import Gemma3Prompter

        return Gemma3Prompter(model=model, dataset=dataset, seed=seed, sampling=sampling)
    elif (
        model == "gpt-5-mini"
        or model == "gpt-4o"
        or model == "gpt-5"
        or model == "gpt-5-chat-latest"
    ):

        return GPTPrompter(model=model, dataset=dataset, seed=seed, reasoning=reasoning)
    elif model.startswith("gemini"):
        from models.gemini.main import GeminiPrompter

        return GeminiPrompter(model=model, dataset=dataset, seed=seed)
    elif model.startswith("claude"):
        from models.claude.main import ClaudePrompter

        return ClaudePrompter(model=model, dataset=dataset, seed=seed)
    else:
        raise ValueError(f"Model {model} not supported.")
