"""Downloads the visual perspective-taking dataset from HuggingFace into data/.

The source dataset is kieranpattison/visual-perspective-taking (gated, but our account has
access). Each row asks whether a robot visible in an image can see a named target object, with
label one of yes / no / cannot_tell. This script copies every image into data/images/ and writes
a local data/metadata.jsonl with the same rows, so the rest of the pipeline never has to touch the
HuggingFace cache directly.

Usage:
    python scripts/download_dataset.py [--force]
"""

import argparse
import collections
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

DATASET_REPO = "kieranpattison/visual-perspective-taking"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
METADATA_PATH = DATA_DIR / "metadata.jsonl"


def download(force: bool) -> None:
    if METADATA_PATH.exists() and not force:
        print(f"{METADATA_PATH} already exists, skipping download (use --force to redo).")
        return

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    files = [f for f in list_repo_files(DATASET_REPO, repo_type="dataset") if f.startswith("train/")]
    metadata_file = next(f for f in files if f.endswith("metadata.jsonl"))
    image_files = [f for f in files if f != metadata_file]

    metadata_local = hf_hub_download(DATASET_REPO, metadata_file, repo_type="dataset")
    shutil.copy(metadata_local, METADATA_PATH)

    for image_file in image_files:
        local_path = hf_hub_download(DATASET_REPO, image_file, repo_type="dataset")
        shutil.copy(local_path, IMAGES_DIR / Path(image_file).name)

    with open(METADATA_PATH) as f:
        rows = [json.loads(line) for line in f]

    label_counts = collections.Counter(row["label"] for row in rows)
    print(f"Downloaded {len(rows)} images to {IMAGES_DIR}")
    print(f"Label counts: {dict(label_counts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if data/ already exists")
    args = parser.parse_args()
    download(force=args.force)
