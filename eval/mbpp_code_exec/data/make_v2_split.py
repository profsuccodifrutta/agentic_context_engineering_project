#!/usr/bin/env python3
"""Build the MBPP v2 split: 120 train / 85 val / 215 test.

docs/background/EXPERIMENT_MBPP.md sec 4, step 1. The v1 split validates on 43 rows, which carries roughly
+/-15pp of noise -- and that estimate is what selects the checkpoint each arm carries into the
test set. In the superseded runs both memory arms tied at val 0.372 on those 43 rows and were
sent to the test set with different playbooks on that basis. Moving 42 rows from test to val
costs power (half-width +/-5.8pp -> +/-6.3pp) and buys a validation estimate at ~+/-10pp.

Train is untouched. The 42 rows are drawn by a seeded shuffle rather than a positional slice,
because both files are ordered by task_id and a contiguous slice would move one block of the
id range instead of a spread.

Deterministic: same inputs, same seed, same outputs. Run once, before any run under the new
design, and never regenerate afterwards -- a resplit silently invalidates every finished arm.

    uv run python eval/mbpp_code_exec/data/make_v2_split.py
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
MOVE = 42
SEED = 0


def read(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write(name, rows):
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def main():
    val, test = read("val.jsonl"), read("test.jsonl")

    order = list(range(len(test)))
    random.Random(SEED).shuffle(order)
    moved = set(order[:MOVE])

    # Emit in the original file order, not shuffled order: train order is deterministic
    # everywhere else in this harness and the eval path should match.
    val_v2 = val + [row for i, row in enumerate(test) if i in moved]
    test_v2 = [row for i, row in enumerate(test) if i not in moved]

    write("val_v2.jsonl", val_v2)
    write("test_v2.jsonl", test_v2)

    config = {
        "mbpp_code_exec": {
            "train_data": "./eval/mbpp_code_exec/data/train.jsonl",
            "val_data": "./eval/mbpp_code_exec/data/val_v2.jsonl",
            "test_data": "./eval/mbpp_code_exec/data/test_v2.jsonl",
        }
    }
    with open(os.path.join(HERE, "task_config_v2.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")

    train = read("train.jsonl")
    ids = {"train": {r["task_id"] for r in train},
           "val": {r["task_id"] for r in val_v2},
           "test": {r["task_id"] for r in test_v2}}
    print(f"train {len(train)}  val {len(val_v2)}  test {len(test_v2)}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = ids[a] & ids[b]
        print(f"  {a}/{b} task_id overlap: {len(overlap)}")
    print(f"  moved {MOVE} rows test -> val, seed {SEED}")


if __name__ == "__main__":
    main()
