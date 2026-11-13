# vlm-perspective-taking

Evaluates how well vision-language models can judge, from a single third-person image, whether a
visible robot can see a particular object — a test of visual perspective-taking.

## Layout

- `prompts/perspective_taking_prompt.txt` — the prompt template sent to every model.
- `data/` — the downloaded dataset (images + `metadata.jsonl`), created by `download_dataset.py`,
  not committed to git.
- `scripts/models.py` — registry of every model we might evaluate (repo id / API id, backend,
  mode). No download or API logic lives here.
- `scripts/download_dataset.py` — pulls `kieranpattison/visual-perspective-taking` from
  HuggingFace into `data/`.
- `scripts/run_inference.py` — runs one model over the whole dataset and writes a timestamped run
  folder under `runs/`.
- `runs/<timestamp>_<model-key>/` — one folder per run, containing `metadata.json` (run info),
  `results.json` (per-image records), and `report.md` (human-readable report with the image, the
  three options, and the model's answer). Both `results.json` and `report.md` are updated after
  every image, so a run in progress can be watched live and a partial run is still readable if it
  gets interrupted.

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Downloading the dataset

```bash
python scripts/download_dataset.py
```

Later dataset iterations live under `iterations/iteration_N/` in the HF repo. Fetch one with
`python scripts/download_dataset.py --iteration 3` (saved to `data_iteration_3/`), then run models
on it with `--data-dir data_iteration_3`.

## Running a model

Local models (downloaded and run on this machine's GPU):

```bash
python scripts/run_inference.py --model qwen3-vl-8b-instruct
```

Add `--limit 5` to smoke-test on a handful of images first. See `scripts/models.py` for every
registered model key — some (e.g. the 32B Qwen3-VL variants, the Gemma entries) aren't downloaded
by default and will be pulled from HuggingFace the first time they're used.

API-backed models (OpenAI):

```bash
python scripts/run_inference.py --model gpt-5.5
```

These read `OPENAI_API_KEY` from an external `.env` file — by default the one already on this
machine at
`/home/michal/Documents/asg-server-save/01-Research/00-MultiviewRobustCP/synth-data-gen/02-Scripts/09-GSAM/.env`.
Override the path with `--openai-env-file` if needed. The key itself is never read into this repo
or printed anywhere.
