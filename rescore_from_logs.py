#!/usr/bin/env python3
"""Re-score a finished run's final test from its saved generator logs.

Every generator call is written to detailed_llm_logs/ with the full prompt and the full
response, so a scoring-side defect can be corrected after the fact without re-running the
model. No LLM call, no network: same bytes in, deterministic re-extraction and re-execution.

Why this exists: the MBPP arms were scored by an extract_answer() that stringified a
one-element JSON array instead of unwrapping it, so `["def f(): ..."]` reached the sandbox
with literal brackets and failed on an undefined name. It hit 22 of 257 final-test answers
in one arm and 0 in the other two -- an 8.6% penalty charged to a single arm by the harness
rather than by the model. Re-running three arms to recover that costs hours; this costs
minutes.

    uv run python rescore_from_logs.py results/arm_text
    uv run python rescore_from_logs.py --root results

Writes rescored_test_predictions.jsonl and rescored_results.json into each run directory
and leaves the original artifacts untouched.

MBPP needs Docker running (the sandbox is constructed by the data processor).
"""

import argparse
import ast
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import llm  # noqa: F401  -- import for its console-encoding guard, before anything prints
from utils import PREDICTION_FIELDS, extract_answer, get_original_item, write_predictions_jsonl

TEST_CALL_PREFIX = "generator_test_eval_"

#: Sentinels extract_answer() returns when it could not recover an answer at all. Counted
#: separately from a wrong answer: they are harness failures, not model reasoning failures.
NO_ANSWER_SENTINELS = (
    "No final answer found",
    "raise ValueError('No valid code extracted from LLM')",
)

TASK_DEFAULT_CONFIGS = {
    "mbpp_code_exec": "./eval/mbpp_code_exec/data/task_config.json",
    "formula": "./eval/finance/data/sample_config.json",
    "finer": "./eval/finance/data/sample_config.json",
}


def expand_run_dirs(path):
    """Resolve a path to the run directories under it.

    Accepts a run directory or a parent holding several. A parent that contains no run at
    all returns empty rather than silently reporting success on nothing.
    """
    if os.path.isdir(os.path.join(path, "detailed_llm_logs")):
        return [path]
    if not os.path.isdir(path):
        return []
    return sorted(
        os.path.join(path, name) for name in sorted(os.listdir(path))
        if os.path.isdir(os.path.join(path, name, "detailed_llm_logs"))
    )


def select_final_test_calls(log_dir):
    """Return {index: log filename} for the final test, plus an integrity note.

    Validation reuses the ``test_eval_<i>`` call id, so ids below len(val_set) appear once
    per validation checkpoint as well. The final test is the last pass in every mode, so
    keeping the latest timestamp per index selects it -- and because that assumption is
    load-bearing it is checked, not trusted: every selected call must be later than every
    unselected one.
    """
    by_index = collections.defaultdict(list)
    for name in os.listdir(log_dir):
        if not name.startswith(TEST_CALL_PREFIX) or not name.endswith(".json"):
            continue
        rest = name[len(TEST_CALL_PREFIX):]
        head, _, tail = rest.partition("_")
        if not head.isdigit():
            continue
        by_index[int(head)].append((tail.rsplit(".json", 1)[0], name))

    selected, unselected = {}, []
    for index, entries in by_index.items():
        entries.sort()
        selected[index] = entries[-1]
        unselected.extend(entries[:-1])

    note = None
    if unselected and selected:
        earliest_selected = min(ts for ts, _ in selected.values())
        latest_unselected = max(ts for ts, _ in unselected)
        if earliest_selected < latest_unselected:
            note = ("earlier test_eval calls overlap the final pass in time; the last-per-index "
                    "selection may be mixing a validation pass into the test set")
    return {i: name for i, (_ts, name) in selected.items()}, note


def score_one(args):
    index, answer, sample, data_processor = args
    try:
        is_correct = bool(data_processor.answer_is_correct(
            answer, sample["target"], get_original_item(sample)))
        err = None
    except Exception as exc:  # a sandbox failure is a scoring failure, not a correct answer
        is_correct, err = False, f"{type(exc).__name__}: {exc}"
    return {
        "index": index,
        "final_answer": answer,
        "target": sample["target"],
        "is_correct": is_correct,
        "success": err is None,
        "error": err,
    }


def load_task(run_dir, config_path_override):
    """Build the data processor and test samples this run was scored against."""
    with open(os.path.join(run_dir, "run_config.json"), encoding="utf-8") as f:
        run_config = json.load(f)
    task_name = run_config.get("task_name") or run_config.get("config", {}).get("task_name")

    config_path = config_path_override or TASK_DEFAULT_CONFIGS.get(task_name)
    if not config_path or not os.path.exists(config_path):
        raise SystemExit(f"no task config for task {task_name!r}; pass --config_path")
    with open(config_path, encoding="utf-8") as f:
        task_config = json.load(f)[task_name]

    if task_name == "mbpp_code_exec":
        from eval.mbpp_code_exec.data_processor import DataProcessor, load_data
    else:
        from eval.finance.data_processor import DataProcessor, load_data

    processor = DataProcessor(task_name=task_name)
    samples = processor.process_task_data(load_data(task_config["test_data"]))
    return run_config, task_name, processor, samples


def rescore(run_dir, config_path_override, workers):
    run_config, task_name, processor, samples = load_task(run_dir, config_path_override)
    log_dir = os.path.join(run_dir, "detailed_llm_logs")
    calls, note = select_final_test_calls(log_dir)

    if note:
        print(f"    WARN  {note}")
    if len(calls) != len(samples):
        print(f"    WARN  {len(calls)} logged test calls vs {len(samples)} test samples; "
              f"scoring the intersection")

    answers, stats = {}, collections.Counter()
    for index in sorted(calls):
        if index >= len(samples):
            continue
        with open(os.path.join(log_dir, calls[index]), encoding="utf-8") as f:
            response = json.load(f).get("response") or ""
        answer = extract_answer(response) if response.strip() else NO_ANSWER_SENTINELS[1]
        answers[index] = answer
        if answer in NO_ANSWER_SENTINELS:
            stats["no_answer"] += 1
        elif task_name == "mbpp_code_exec":
            try:
                ast.parse(answer)
                stats["runnable"] += 1
            except SyntaxError:
                stats["unparseable"] += 1

    work = [(i, a, samples[i], processor) for i, a in answers.items()]
    rows = []
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(score_one, w) for w in work]
            for done in as_completed(futures):
                rows.append(done.result())
    else:
        rows = [score_one(w) for w in work]
    rows.sort(key=lambda r: r["index"])

    correct = sum(1 for r in rows if r["is_correct"])
    total = len(rows)
    results = {
        "run_dir": run_dir,
        "task_name": task_name,
        "memory_kind": run_config.get("memory_kind"),
        "seed": run_config.get("seed"),
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "no_answer": stats["no_answer"],
        "unparseable": stats.get("unparseable", 0),
        "scoring_errors": sum(1 for r in rows if not r["success"]),
    }

    write_predictions_jsonl(os.path.join(run_dir, "rescored_test_predictions.jsonl"), rows)
    with open(os.path.join(run_dir, "rescored_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def read_original(run_dir):
    for name in ("final_test_results.json", "test_test_results.json"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f).get("test_results", {})
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="*", help="Run directories, or parents holding them")
    parser.add_argument("--root", help="Rescore every run found under this directory")
    parser.add_argument("--config_path", help="Task config JSON (defaults per task name)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel scoring workers. Keep at 1 for MBPP unless the "
                             "sandbox limits were raised -- containers are memory-capped")
    args = parser.parse_args()

    targets = []
    for path in list(args.run_dirs) + ([args.root] if args.root else []):
        found = expand_run_dirs(path)
        if not found:
            print(f"no run directory under {path!r} (looked for detailed_llm_logs/)")
        targets.extend(found)
    if not targets:
        parser.error("nothing to rescore")

    summaries = []
    for run_dir in targets:
        print(f"\n=== {os.path.basename(run_dir)} ===")
        before = read_original(run_dir)
        after = rescore(run_dir, args.config_path, args.workers)
        summaries.append((run_dir, before, after))
        print(f"    before  {before.get('correct')}/{before.get('total')} "
              f"acc={before.get('accuracy', 0):.4f}  no_answer={before.get('no_answer')}")
        print(f"    after   {after['correct']}/{after['total']} "
              f"acc={after['accuracy']:.4f}  no_answer={after['no_answer']}"
              + (f"  unparseable={after['unparseable']}" if after["unparseable"] else ""))

    print(f"\n{'run':<46}{'before':>10}{'after':>10}{'delta':>9}")
    for run_dir, before, after in summaries:
        b = before.get("accuracy")
        delta = f"{after['accuracy'] - b:+.4f}" if isinstance(b, (int, float)) else "n/a"
        print(f"{os.path.basename(run_dir)[:44]:<46}"
              f"{(f'{b:.4f}' if isinstance(b, (int, float)) else 'n/a'):>10}"
              f"{after['accuracy']:>10.4f}{delta:>9}")
    print("\nWrote rescored_test_predictions.jsonl per run "
          f"(fields: {', '.join(PREDICTION_FIELDS)}). Compare with compare_arms.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
