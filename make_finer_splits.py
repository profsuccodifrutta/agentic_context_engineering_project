"""Build the `finer` split set: 250 train / 150 val / 441 test, plus a 50/20/20 probe.

Deterministic and seeded. Run once, commit the output, never regenerate -- the splits are part
of the pre-registration (docs/EXPERIMENT_FINER.md §4) and a re-draw would silently change what
every completed run was measured on.

Three rules, each with a reason:

* **The test split is not touched.** `finer_test_subset_006_seed42.jsonl` is the ACE authors'
  own subset and is the only thing making a number here comparable to a published one.
* **Train and val are downsampled, and only they are.** 1000 training rows is ~4x the formula
  runs for no gain: at 250 rows, 101.8 of the 137 test tags already have 3 or more training
  examples, covering 83.6% of test tag instances, and 250 is also formula's step count, so
  curator call volume and playbook growth are directly comparable between the two experiments.
  Going to 400 buys 6.8pp more instance coverage for 60% more compute.
* **Nothing leaks.** 3 of the 1275 test sentences appear in the training pool and 129 of the 500
  val rows share a sentence with it. Both are filtered here rather than disclosed later.

Probe splits are drawn from the train and val pools only, never from test. A probe is not a
result, but there is no reason to spend the pristine set on one.

    uv run python make_finer_splits.py [--check]

`--check` re-derives everything and reports without writing, so the audit can be re-run on a
checkout without touching the files.
"""

import argparse
import json
import os
import random
import re
from collections import Counter

DATA = os.path.join("eval", "finance", "data")
POOL_TRAIN = os.path.join(DATA, "finer_train_batched_1000_samples.jsonl")
POOL_VAL = os.path.join(DATA, "finer_val_batched_500_samples.jsonl")
TEST = os.path.join(DATA, "finer_test_subset_006_seed42.jsonl")

OUT_TRAIN = os.path.join(DATA, "finer_train_250.jsonl")
OUT_VAL = os.path.join(DATA, "finer_val_150.jsonl")
OUT_PROBE_TRAIN = os.path.join(DATA, "finer_probe_train.jsonl")
OUT_PROBE_VAL = os.path.join(DATA, "finer_probe_val.jsonl")
OUT_PROBE_TEST = os.path.join(DATA, "finer_probe_test.jsonl")

SEED = 42
N_TRAIN, N_VAL = 250, 150
N_PROBE_TRAIN, N_PROBE_VAL, N_PROBE_TEST = 50, 20, 20

_SENTENCE = re.compile(r'in sentence:\s*"(.*?)\?"', re.S)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def sentences(row):
    return set(_SENTENCE.findall(row.get("context", "")))


def tags(row):
    return [t.strip() for t in row.get("target", "").split(",") if t.strip()]


def sample_in_file_order(pool, n, rng):
    """Pick n rows at random, then emit them in their original file order.

    Order matters: training is a single pass in file order with no shuffle, so the sequence the
    curator sees is fixed by this function. Emitting in file order keeps that sequence a
    property of the source data rather than of the sampler's internal state.
    """
    picked = set(rng.sample(range(len(pool)), n))
    return [row for i, row in enumerate(pool) if i in picked]


def build():
    train_pool, val_pool, test = read(POOL_TRAIN), read(POOL_VAL), read(TEST)
    test_sentences = set().union(*(sentences(r) for r in test))

    clean_train = [r for r in train_pool if not (sentences(r) & test_sentences)]
    train_sentences = set().union(*(sentences(r) for r in train_pool))
    clean_val = [r for r in val_pool
                 if not (sentences(r) & train_sentences) and not (sentences(r) & test_sentences)]

    rng = random.Random(SEED)
    train = sample_in_file_order(clean_train, N_TRAIN, rng)
    val = sample_in_file_order(clean_val, N_VAL, rng)

    # Probe train is the first 50 rows of the real training split, deliberately: the probe then
    # sees a prefix of the exact stream the real run will see, which is what makes its measured
    # citation density and playbook growth predict the real run's. That containment is the one
    # overlap the audit below exempts.
    probe_train = train[:N_PROBE_TRAIN]

    # Probe val and probe test come out of the clean val pool, disjoint from the real val split
    # by sentence rather than by row: val rows drawn from one filing share sentences, so
    # excluding by identity alone leaks. It matters because probe results pin hyperparameters
    # that the real run then uses, and the real val split selects its checkpoint.
    val_sentences = set().union(*(sentences(r) for r in val))
    spare = [r for r in clean_val if not (sentences(r) & val_sentences)]
    probe_val = spare[:N_PROBE_VAL]
    probe_test = [r for r in spare[N_PROBE_VAL:]
                  if not (sentences(r) & set().union(*(sentences(p) for p in probe_val)))
                  ][:N_PROBE_TEST]

    return {
        "pools": (train_pool, val_pool, test),
        OUT_TRAIN: train, OUT_VAL: val,
        OUT_PROBE_TRAIN: probe_train, OUT_PROBE_VAL: probe_val, OUT_PROBE_TEST: probe_test,
    }


def audit(built):
    train_pool, val_pool, test = built["pools"]
    splits = {name: rows for name, rows in built.items() if name != "pools"}

    print(f"seed {SEED}; pools train={len(train_pool)} val={len(val_pool)} test={len(test)} (test untouched)\n")
    print(f"{'file':<46}{'rows':>6}{'tags':>7}{'distinct':>10}")
    for name, rows in splits.items():
        flat = [t for r in rows for t in tags(r)]
        print(f"{os.path.basename(name):<46}{len(rows):>6}{len(flat):>7}{len(set(flat)):>10}")
    print(f"{os.path.basename(TEST) + '  (unchanged)':<46}{len(test):>6}"
          f"{len([t for r in test for t in tags(r)]):>7}{len({t for r in test for t in tags(r)}):>10}")

    print("\nsentence overlap (must be 0 everywhere except the one exempt pair):")
    named = list(splits.items()) + [(TEST, test)]
    sets = {name: set().union(*(sentences(r) for r in rows)) if rows else set() for name, rows in named}
    # The probe's training rows are by construction the first 50 of the real training split.
    exempt = {frozenset((OUT_TRAIN, OUT_PROBE_TRAIN))}
    problems = []
    for i, (a, _) in enumerate(named):
        for b, _ in named[i + 1:]:
            shared = len(sets[a] & sets[b])
            if frozenset((a, b)) in exempt:
                marker = "   (exempt: probe train is a prefix of train)"
            elif shared:
                marker = "   <-- LEAK"
                problems.append((a, b, shared))
            else:
                marker = ""
            print(f"  {os.path.basename(a):<40} x {os.path.basename(b):<40} {shared:>4}{marker}")

    train_tags = Counter(t for r in splits[OUT_TRAIN] for t in tags(r))
    test_tags = Counter(t for r in test for t in tags(r))
    backed = sum(v for t, v in test_tags.items() if train_tags[t] >= 3)
    print(f"\ncoverage: {sum(1 for t in test_tags if train_tags[t] >= 3)} of {len(test_tags)} test tags have "
          f">=3 training examples, backing {backed / sum(test_tags.values()):.1%} of test tag instances")
    distinct = [len(set(tags(r))) for r in test]
    print(f"test rows carry {sum(distinct) / len(distinct):.2f} distinct tags on average "
          f"({sum(1 for d in distinct if d >= 2) / len(distinct):.1%} carry 2 or more)")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="re-derive and audit without writing anything")
    args = parser.parse_args()

    built = build()
    problems = audit(built)

    if args.check:
        print("\n--check: nothing written")
        return 1 if problems else 0

    if problems:
        print("\nrefusing to write: the splits above share sentences, so a run on them would "
              "train on its own test rows")
        return 1
    for name, rows in built.items():
        if name == "pools":
            continue
        write(name, rows)
        print(f"wrote {name}")
    print("\nCommit these. They are part of the pre-registration; do not regenerate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
