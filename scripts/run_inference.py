"""Runs one VLM over the whole perspective-taking dataset and logs a timestamped run.

For every image in data/, sends the image plus the perspective-taking prompt (prompts/
perspective_taking_prompt.txt) to the chosen model and records its answer. Local models
(backend "transformers" in models.py) are loaded with HuggingFace transformers and run on the
GPU in bf16. API models (backend "openai") are called through the OpenAI API, reading the key
from an external .env file that is never copied into this repo.

Everything is written under runs/<timestamp>_<model_key>/ as it happens:
  - metadata.json   run-level info, updated once at the start and once at the end
  - results.json    list of per-image records, rewritten after every image
  - report.md       human-readable report, appended to after every image

Usage:
    python scripts/run_inference.py --model qwen3-vl-8b-instruct [--limit 5]
    python scripts/run_inference.py --model gpt-5.5 [--openai-env-file PATH]
"""

import argparse
import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

from models import MODEL_REGISTRY

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
METADATA_PATH = DATA_DIR / "metadata.jsonl"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "perspective_taking_prompt.txt"
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_OPENAI_ENV_FILE = (
    "/home/michal/Documents/asg-server-save/01-Research/00-MultiviewRobustCP/"
    "synth-data-gen/02-Scripts/09-GSAM/.env"
)


def load_dataset_rows(data_dir: Path, limit: int | None) -> list[dict]:
    with open(data_dir / "metadata.jsonl") as f:
        rows = [json.loads(line) for line in f]
    return rows[:limit] if limit else rows


def build_prompt(row: dict) -> str:
    template = PROMPT_PATH.read_text()
    return template.format(
        question=row["question"],
        target_type=row["target_type"],
        option_a=row["option_a"],
        option_b=row["option_b"],
        option_c=row["option_c"],
    )


def parse_choice(raw_response: str) -> str | None:
    for line in reversed(raw_response.strip().splitlines()):
        line = line.strip()
        if line.lower().startswith("answer:"):
            letter = line.split(":", 1)[1].strip().strip(".").upper()
            if letter in ("A", "B", "C"):
                return letter
    return None


class TransformersRunner:
    def __init__(self, repo_id: str, device: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        HfApi().model_info(repo_id)  # fails fast with a clear error if the repo doesn't exist

        self.processor = AutoProcessor.from_pretrained(repo_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            repo_id, dtype=torch.bfloat16, device_map=device
        )

    def answer(self, image_path: Path, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=512)
        new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


class OpenAIRunner:
    def __init__(self, model_name: str, env_file: str):
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(env_file)
        self.model_name = model_name
        self.client = OpenAI()

    def answer(self, image_path: Path, prompt: str) -> str:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        data_url = f"data:image/png;base64,{image_b64}"

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        return response.choices[0].message.content.strip()


def write_report_header(report_path: Path, run_id: str, config: dict, start_time: str) -> None:
    report_path.write_text(
        f"# Run: {run_id}\n\n"
        f"- Model: {config['display_name']} (`{config['repo_id']}`)\n"
        f"- Backend: {config['backend']}\n"
        f"- Started: {start_time}\n\n"
        f"---\n\n",
        encoding="utf-8",
    )


def append_report_entry(report_path: Path, row: dict, record: dict) -> None:
    options = {"A": row["option_a"], "B": row["option_b"], "C": row["option_c"]}
    ground_truth = row["ground_truth_choice"]

    lines = [f"## {row['id']}", "", f"![{row['file_name']}](../../data/images/{row['file_name']})", ""]
    lines.append(f"**Question:** {row['question']}")
    lines.append("")
    for letter, text in options.items():
        marker = " (ground truth)" if letter == ground_truth else ""
        lines.append(f"- {letter}) {text}{marker}")
    lines.append("")

    if record["error"]:
        lines.append(f"**Error:** {record['error']}")
    else:
        lines.append(f"**Model answer:** {record['parsed_choice'] or 'UNPARSED'}")
        lines.append("")
        lines.append("**Raw response:**")
        lines.append("")
        lines.append("```")
        lines.append(record["raw_response"])
        lines.append("```")
    lines.append("\n---\n")

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run(model_key: str, limit: int | None, device: str, openai_env_file: str, data_dir: Path) -> None:
    if model_key not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model '{model_key}'. Options: {', '.join(MODEL_REGISTRY)}")
    config = MODEL_REGISTRY[model_key]

    rows = load_dataset_rows(data_dir, limit)

    start_time = datetime.now(timezone.utc).isoformat()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{model_key}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True)
    metadata_path = run_dir / "metadata.json"
    results_path = run_dir / "results.json"
    report_path = run_dir / "report.md"

    metadata = {
        "run_id": run_id,
        "model_key": model_key,
        "model_repo": config["repo_id"],
        "display_name": config["display_name"],
        "backend": config["backend"],
        "mode": config["mode"],
        "device": device if config["backend"] == "transformers" else "api",
        "prompt_file": str(PROMPT_PATH.relative_to(PROJECT_ROOT)),
        "data_dir": str(data_dir),
        "num_images": len(rows),
        "start_time": start_time,
        "end_time": None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    write_report_header(report_path, run_id, config, start_time)

    if config["backend"] == "transformers":
        runner = TransformersRunner(config["repo_id"], device)
    else:
        runner = OpenAIRunner(config["repo_id"], openai_env_file)

    results = []
    for row in rows:
        image_path = data_dir / "images" / row["file_name"]
        prompt = build_prompt(row)

        error = None
        raw_response = None
        try:
            raw_response = runner.answer(image_path, prompt)
        except Exception as exc:
            error = str(exc)

        record = {
            "id": row["id"],
            "file_name": row["file_name"],
            "target_type": row["target_type"],
            "question": row["question"],
            "option_a": row["option_a"],
            "option_b": row["option_b"],
            "option_c": row["option_c"],
            "ground_truth_choice": row["ground_truth_choice"],
            "label": row["label"],
            "model_key": model_key,
            "model_repo": config["repo_id"],
            "backend": config["backend"],
            "prompt": prompt,
            "raw_response": raw_response,
            "parsed_choice": parse_choice(raw_response) if raw_response else None,
            "error": error,
        }
        results.append(record)

        results_path.write_text(json.dumps(results, indent=2))
        append_report_entry(report_path, row, record)

        status = record["parsed_choice"] or record["error"] or "UNPARSED"
        print(f"[{len(results)}/{len(rows)}] {row['id']} -> {status}")

    metadata["end_time"] = datetime.now(timezone.utc).isoformat()
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Done. Run written to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY), help="Model registry key")
    parser.add_argument("--limit", type=int, default=None, help="Only run on the first N images")
    parser.add_argument("--device", default="cuda", help="Device for transformers backend")
    parser.add_argument("--openai-env-file", default=DEFAULT_OPENAI_ENV_FILE, help=".env file with OPENAI_API_KEY")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Dataset folder with images/ and metadata.jsonl")
    args = parser.parse_args()
    run(args.model, args.limit, args.device, args.openai_env_file, args.data_dir)
