import json
import random
from pathlib import Path
from typing import List, Dict


# -----------------------------
# Load JSON
# -----------------------------
def load_json(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Reproducible split
# -----------------------------
def split_data(
    data: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
):

    rng = random.Random(seed)
    data = data.copy()
    rng.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    return train, val, test


# -----------------------------
# Convert to JSONL
# -----------------------------
def to_jsonl(data: List[Dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            record = {
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "test_setup_code": item.get("test_setup_code", ""),
                "test_list": item.get("test_list", []),
                "challenge_test_list": item.get("challenge_test_list", [])
            }

            f.write(json.dumps(record) + "\n")


# -----------------------------
# Main
# -----------------------------
def main():
    base_dir = Path(__file__).parent / "data"
    input_file = base_dir / "mbpp_50.json"

    data = load_json(input_file)

    train, val, test = split_data(data, seed=42)

    # Save JSONL
    to_jsonl(train, base_dir / "train.jsonl")
    to_jsonl(val, base_dir / "val.jsonl")
    to_jsonl(test, base_dir / "test.jsonl")

    print("✅ Done converting dataset")
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")


if __name__ == "__main__":
    main()