"""Registry of every VLM we might evaluate on the perspective-taking dataset.

This file only declares what models exist and how to reach them - it never loads a model or calls
an API itself. That logic lives in run_inference.py. `backend` is "transformers" for models we run
locally via HuggingFace transformers, or "openai" for models called through the OpenAI API.

Entries marked "unverified" have not been confirmed to exist on HuggingFace and will fail with a
clear error at run time until the real repo id is known.
"""

MODEL_REGISTRY = {
    "qwen3-vl-8b-instruct": {
        "backend": "transformers",
        "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
        "display_name": "Qwen3-VL 8B Instruct",
        "mode": "instruct",
        "param_count": "8B",
    },
    "qwen3-vl-8b-thinking": {
        "backend": "transformers",
        "repo_id": "Qwen/Qwen3-VL-8B-Thinking",
        "display_name": "Qwen3-VL 8B Thinking",
        "mode": "thinking",
        "param_count": "8B",
    },
    "qwen3-vl-32b-instruct": {
        "backend": "transformers",
        "repo_id": "Qwen/Qwen3-VL-32B-Instruct",
        "display_name": "Qwen3-VL 32B Instruct",
        "mode": "instruct",
        "param_count": "32B",
    },
    "qwen3-vl-32b-thinking": {
        "backend": "transformers",
        "repo_id": "Qwen/Qwen3-VL-32B-Thinking",
        "display_name": "Qwen3-VL 32B Thinking",
        "mode": "thinking",
        "param_count": "32B",
    },
    "gemma-4-26b-a4b-it": {
        "backend": "transformers",
        "repo_id": "google/gemma-4-26B-A4B-it",
        "display_name": "Gemma 4 26B-A4B IT (unverified repo id)",
        "mode": "instruct",
        "param_count": "26B-A4B",
    },
    "gemma-4-31b-it": {
        "backend": "transformers",
        "repo_id": "google/gemma-4-31B-it",
        "display_name": "Gemma 4 31B IT (unverified repo id)",
        "mode": "instruct",
        "param_count": "31B",
    },
    "gpt-5.5": {
        "backend": "openai",
        "repo_id": "gpt-5.5",
        "display_name": "GPT-5.5",
        "mode": None,
        "param_count": None,
    },
    "gpt-5.5-mini": {
        "backend": "openai",
        "repo_id": "gpt-5.5-mini",
        "display_name": "GPT-5.5 Mini",
        "mode": None,
        "param_count": None,
    },
    "gpt-6-astra": {
        "backend": "openai",
        "repo_id": "gpt-6-astra",
        "display_name": "GPT-6 Astra",
        "mode": None,
        "param_count": None,
    },
    "gpt-5.6-sol": {
        "backend": "openai",
        "repo_id": "gpt-5.6-sol",
        "display_name": "GPT-5.6 Sol",
        "mode": None,
        "param_count": None,
    },
    "gpt-5.6-terra": {
        "backend": "openai",
        "repo_id": "gpt-5.6-terra",
        "display_name": "GPT-5.6 Terra",
        "mode": None,
        "param_count": None,
    },
    "gpt-5.6-luna": {
        "backend": "openai",
        "repo_id": "gpt-5.6-luna",
        "display_name": "GPT-5.6 Luna",
        "mode": None,
        "param_count": None,
    },
}
