import json
import logging
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def format_record(item: Dict) -> Dict:
    """
    Keeps EXACT same schema you already use.
    """

    return {
        "task_id": item["task_id"],
        "prompt": item["prompt"],
        "test_setup_code": item.get("test_setup_code", ""),
        "test_list": item.get("test_list", []),
        "challenge_test_list": item.get("challenge_test_list", []),
    }


# -----------------------------
# Write JSONL
# -----------------------------
def write_jsonl(dataset_split, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset_split:
            record = format_record(item)
            f.write(json.dumps(record) + "\n")


# -----------------------------
# Main pipeline
# -----------------------------
def main():

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    logger.info("Loading MBPP dataset (official splits)...")

    # IMPORTANT: keeps original benchmark splits
    dataset = load_dataset("mbpp", "sanitized")

    splits = {
        "train": "train.jsonl",
        "validation": "val.jsonl",
        "test": "test.jsonl",
    }

    for split_name, filename in splits.items():

        logger.info(f"Processing {split_name}...")

        split_data = dataset[split_name]

        output_path = data_dir / filename

        write_jsonl(split_data, output_path)

        logger.info(
            f"Saved {len(split_data)} rows → {output_path}"
        )

    logger.info("All splits successfully exported.")


if __name__ == "__main__":
    main()