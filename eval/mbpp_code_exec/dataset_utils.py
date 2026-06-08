import json
import logging
from pathlib import Path
from typing import List, TypedDict

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class MBPPTask(TypedDict):
    task_id: int
    prompt: str
    test_setup_code: str
    test_list: List[str]
    challenge_test_list: List[str]


DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def download_and_cache_mbpp(
    num_tasks: int = 50,
    output_filename: str = "mbpp_50.json"
):
    """Download MBPP from HuggingFace and cache locally."""

    output_path = DATA_DIR / output_filename

    logger.info("Downloading MBPP dataset from HuggingFace...")

    dataset = load_dataset("mbpp", "sanitized", split="test")
    dataset_subset = dataset.select(range(min(num_tasks, len(dataset))))

    processed_tasks = []

    for item in dataset_subset:
        try:
            task: MBPPTask = {
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "test_setup_code": item.get("test_setup_code", ""),
                "test_list": item["test_list"],
                "challenge_test_list": item.get(
                    "challenge_test_list", []
                ),
            }

            processed_tasks.append(task)

        except KeyError as e:
            logger.warning(f"Skipping task due to missing key: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_tasks, f, indent=4)

    logger.info(f"Saved {len(processed_tasks)} tasks to {output_path}")


def load_local_mbpp(
    filename: str = "mbpp_50.json"
) -> List[MBPPTask]:
    """Load cached MBPP dataset."""

    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset cache not found at {file_path}"
        )

    logger.info(f"Loading dataset from {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        tasks: List[MBPPTask] = json.load(f)

    logger.info(f"Loaded {len(tasks)} tasks")

    return tasks


if __name__ == "__main__":

    download_and_cache_mbpp()

    tasks = load_local_mbpp()

    print(f"\nLoaded {len(tasks)} tasks")

    if tasks:
        sample = tasks[0]

        print(f"\nTask ID: {sample['task_id']}")
        print(f"Prompt: {sample['prompt']}")
        print(f"First test: {sample['test_list'][0]}")