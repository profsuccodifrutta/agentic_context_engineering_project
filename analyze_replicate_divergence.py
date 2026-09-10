"""Are two runs of the same arm actually two replicates? And how much does the arm move between them?

Recomputes the replicate figures in `docs/FINER_RESULTS.md` section 8. No LLM, no network.

    uv run python analyze_replicate_divergence.py results/finer/final/seed*/text/ace_run_*

**Why the prediction-file hash is not a sufficient check, which is the finding this exists for.**
`--seed` is inert at temperature 0, so runs diverge only when the serving state changes, and how
far that change reaches is not the same in every pair. On `finer`, seeds 1 and 2 produced distinct
prediction files while training byte-identically: same curator operations, same validation, same
saved playbook, differing only in the final test generations. They pass a hash check on the
predictions and are still one training draw.

So this compares runs **role by role**, keyed on the prompt, and prints the playbook hash beside
the prediction hash. Two runs sharing a `best_playbook.txt` are one training draw however their
test files compare, and a standard deviation computed over them is a floor on the replicate spread.
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import statistics

import ace.native_deps  # noqa: F401  -- must precede sentence-transformers, see CLAUDE.md
from analysis_common import (
    arm_of,
    collect_runs,
    prediction_rows,
    read_config,
    replicate_label,
)
from compare_arms import METRICS, PREDICTION_FILES
from replay_curator_sanitizer import replay

#: The live-endpoint curator ceiling measured with `check_curator_ceiling.py` on `gemma4:e2b`,
#: the model every `finer` role used. A prompt above it returns nothing at all rather than
#: erroring, so the run still looks healthy. It is a per-model, per-endpoint reading and is
#: applied only where it was measured; see `ceiling_for`.
CURATOR_CEILINGS = {"gemma4:e2b": 26860}
ROLES = ("generator_test", "generator", "reflector", "curator")


def digest(path):
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()[:10]


def role_of(name):
    for role in ROLES:
        if name.startswith(role):
            return role
    return "other"


def calls(run_dir):
    """{role: {(prompt hash, nth occurrence): response}} over every saved call.

    Keyed on the prompt because filenames carry timestamps, and two runs of one configuration
    issue the same prompts in the same order. **Keyed on the occurrence as well**, because a
    prompt is not unique: the same text really is sent more than once in a run (an empty-response
    retry, a reflector round over a repeated sample), and a plain prompt-hash key silently
    collapsed those into one entry, under-counting the calls compared and hiding any call it
    dropped from the comparison entirely.
    """
    out = collections.defaultdict(dict)
    seen = collections.Counter()
    for path in sorted(glob.glob(os.path.join(run_dir, "detailed_llm_logs", "*.json"))):
        try:
            with open(path, encoding="utf-8") as handle:
                blob = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        prompt = blob.get("prompt")
        if prompt is None:
            continue
        role = role_of(os.path.basename(path))
        key = hashlib.md5(prompt.encode("utf-8", "replace")).hexdigest()
        seen[(role, key)] += 1
        out[role][(key, seen[(role, key)])] = blob.get("response") or ""
    return out


def curator_operations(run_dir):
    """Rows of curator_operations_diff.jsonl, and the same rows with timestamps removed."""
    path = os.path.join(run_dir, "curator_operations_diff.jsonl")
    if not os.path.isfile(path):
        return None, None
    raw, stripped = [], []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw.append(line.rstrip("\n"))
            stripped.append(json.dumps({k: v for k, v in row.items() if k != "timestamp"},
                                       sort_keys=True))
    return raw, stripped


def per_row_scores(run_dir, task):
    """{row index: score}, using the scorer that task's own results are reported under.

    Dispatches through `compare_arms.METRICS` rather than re-deriving either comparison, so this
    file cannot disagree with the accuracies `compare_arms.py` prints. `finer` is per-tag partial
    credit; every other task is the binary `is_correct` the runner already wrote.
    """
    scorer, _binary = METRICS["finer_per_tag" if task == "finer" else "exact"]
    rows = prediction_rows(run_dir)
    if rows is None:
        return None
    return {row["index"]: scorer(row) for row in rows}


def last_curator_prompt(run_dir):
    """Prompt length of the run's LAST curator call, in the server's own token count.

    The last step, not the largest. The curator prompt grows with the memory, so the two are
    usually close but they are different quantities and section 5 quotes the last step.
    """
    latest, tokens = None, None
    for path in glob.glob(os.path.join(run_dir, "detailed_llm_logs", "curator_*.json")):
        try:
            with open(path, encoding="utf-8") as handle:
                blob = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        # Ordered by the recorded time, never by filename: the step is embedded as
        # `curator_train_e_1_s_<step>_...`, so a lexicographic sort ends at step 9 and reports a
        # prompt from early training as the run's last.
        stamp = blob.get("datetime") or blob.get("timestamp")
        count = blob.get("prompt_num_tokens")
        if stamp and isinstance(count, int) and (latest is None or stamp > latest):
            latest, tokens = stamp, count
    return tokens


def ceiling_for(config):
    """The measured curator ceiling for this run's model, or None if none was measured.

    `26860` is a reading taken against one endpoint with `gemma4:e2b`. Printing a share of it for
    a run on another model would dress a guess as a measurement.
    """
    return CURATOR_CEILINGS.get(config.get("curator_model"))


def absorption(run_dir):
    """(live rules, ADDs the curator applied) for the grow-and-refine share.

    The applied count, not the logged one. `curator_operations_diff.jsonl` is written *before*
    `sanitize_curator_operations` runs, deliberately, so that `check_learning_signal.py` [6] reads
    the model rather than the fix. Exact re-adds are dropped after that log, so counting its rows
    overstates what reached the playbook. `replay_curator_sanitizer.replay` is the tool that
    already replays the sanitiser offline, and its `kept` total is the number to divide by.
    """
    live = None
    playbook = os.path.join(run_dir, "best_playbook.txt")
    if os.path.isfile(playbook):
        with open(playbook, encoding="utf-8") as handle:
            live = sum(1 for line in handle if " :: " in line)
    if not os.path.isfile(os.path.join(run_dir, "curator_operations_diff.jsonl")):
        return live, None, None
    totals, _examples = replay(run_dir)
    return live, totals.get("adds"), totals.get("kept")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="run directories, or parents of ace_run_*")
    args = parser.parse_args()

    by_arm = collections.defaultdict(list)
    for run_dir in collect_runs(args.paths):
        by_arm[arm_of(run_dir)].append(run_dir)

    for arm, runs in sorted(by_arm.items()):
        print("=" * 100)
        print(f"ARM {arm}   {len(runs)} run(s)")

        print("\n  IDENTITY -- hash the playbook, not only the predictions")
        print(f"    {'run':<44}{'predictions':>13}{'best_playbook':>16}{'accuracy':>13}")
        fingerprints = {}
        for run_dir in runs:
            scores = per_row_scores(run_dir, read_config(run_dir).get("task_name", ""))
            accuracy = statistics.mean(scores.values()) if scores else float("nan")
            predictions_hash = None
            for name in PREDICTION_FILES:
                path = os.path.join(run_dir, name)
                if os.path.isfile(path):
                    predictions_hash = digest(path)
                    break
            playbook_hash = digest(os.path.join(run_dir, "best_playbook.txt"))
            fingerprints[run_dir] = (predictions_hash, playbook_hash, scores)
            print(f"    {os.path.basename(run_dir):<44}{str(predictions_hash):>13}"
                  f"{str(playbook_hash):>16}{accuracy:>13.4f}")

        shared_playbooks = collections.defaultdict(list)
        for run_dir, (_p, playbook_hash, _s) in fingerprints.items():
            if playbook_hash:
                shared_playbooks[playbook_hash].append(run_dir)
        duplicated = [group for group in shared_playbooks.values() if len(group) > 1]
        if duplicated:
            for group in duplicated:
                names = ", ".join(os.path.basename(d) for d in group)
                print(f"    !! {len(group)} runs share one trained playbook: {names}")
            print("    The within-arm sd over these runs is a FLOOR on the replicate spread: it "
                  "spans fewer training draws than runs.")
        else:
            print("    All runs have distinct playbooks: the sd spans one training draw per run.")

        if len(runs) > 1:
            pairs = [(a, b) for i, a in enumerate(runs) for b in runs[i + 1:]]
            print("\n  WHERE THE RUNS DIVERGE (shared calls, keyed on prompt and occurrence)")
            for first, second in pairs:
                left, right = calls(first), calls(second)
                print(f"    {replicate_label(first)} ({os.path.basename(first)})  vs  "
                      f"{replicate_label(second)} ({os.path.basename(second)})")
                for role in ROLES:
                    shared = set(left.get(role, {})) & set(right.get(role, {}))
                    if not shared:
                        continue
                    differ = sum(1 for key in shared if left[role][key] != right[role][key])
                    only = len(set(left.get(role, {})) ^ set(right.get(role, {})))
                    extra = f", {only} call(s) in one run only" if only else ""
                    print(f"      {role:<18}{len(shared):>6} shared,{differ:>6} responses differ"
                          f"{extra}")
                raw_a, stripped_a = curator_operations(first)
                raw_b, stripped_b = curator_operations(second)
                if raw_a is not None and raw_b is not None:
                    same = stripped_a == stripped_b
                    note = "  (so they differ ONLY in timestamp)" if same and raw_a != raw_b else ""
                    print(f"      {'curator ops':<18}{len(raw_a):>6} rows, "
                          f"identical ignoring timestamp: {same}{note}")

            print("\n  THE FLOOR -- test rows whose score moves between two runs")
            print(f"    {'pair':<28}{'rows moved':>12}{'of':>6}")
            for first, second in pairs:
                left = fingerprints[first][2] or {}
                right = fingerprints[second][2] or {}
                common = set(left) & set(right)
                moved = sum(1 for index in common if left[index] != right[index])
                label = f"{replicate_label(first)} vs {replicate_label(second)}"
                print(f"    {label:<28}{moved:>12}{len(common):>6}")
            accuracies = [statistics.mean(fingerprints[d][2].values())
                          for d in runs if fingerprints[d][2]]
            if len(accuracies) > 1:
                spread = (statistics.pstdev(accuracies) if len(accuracies) < 3
                          else statistics.stdev(accuracies))
                print(f"    accuracy across runs: mean {statistics.mean(accuracies):.4f}, "
                      f"sd {spread:.4f}")

        print("\n  GROW-AND-REFINE, AND THE CURATOR'S CONTEXT")
        print(f"    {'run':<44}{'live':>7}{'ADDs kept':>11}{'absorbed':>10}"
              f"{'curator @ last step':>21}{'of ceiling':>12}")
        for run_dir in runs:
            live, logged, kept = absorption(run_dir)
            last = last_curator_prompt(run_dir)
            ceiling = ceiling_for(read_config(run_dir))
            share = f"{100.0 * (1 - live / kept):.1f}%" if (live and kept and kept >= live) else "n/a"
            of_ceiling = f"{100.0 * last / ceiling:.0f}%" if (last and ceiling) else "n/a"
            print(f"    {os.path.basename(run_dir):<44}{str(live):>7}{str(kept):>11}{share:>10}"
                  f"{str(last):>21}{of_ceiling:>12}")
            if logged and kept and logged != kept:
                print(f"      ({logged} ADDs logged, {logged - kept} dropped by the sanitiser "
                      f"before they reached the playbook)")


if __name__ == "__main__":
    main()
