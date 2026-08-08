#!/usr/bin/env python3
"""
Report whether a run's learning signal is actually alive, before spending a full arm on it.

Run 20260802_220602 finished cleanly, cost 3.2 GPU-hours, and produced a memory that fixed
17 test problems and broke 19. Nothing in the harness flagged it, because every artifact it
writes looks healthy: the playbook grows, bullets get cited, accuracy is plausible. The three
things that were dead are only visible if you go looking for them.

This looks for them. Point it at a run directory (ideally a short probe run) and it prints:

  1. harmful tags   -- the reflector emitted 'harmful' zero times in 120 reflections, including
                       on 15 bullets inside attempts that had just failed their assertions.
                       Without it, memory can only accumulate and never self-correct.
  2. refines        -- the curator emitted the field zero times in 24 calls. It is the graph
                       arm's only real capability, so a graph run under that curator is a
                       guaranteed null regardless of what the graph does.
  3. headroom       -- training accuracy was 92%, i.e. 18 learnable failures in 120 steps.
  4. bullet usage   -- generator citations, which drive every counter downstream.
  5. output integrity -- whether the model can hold the JSON schemas at all. This is the one
                       that matters when swapping in a smaller generator to create headroom:
                       a model that fails by emitting unparseable output produces failures
                       ACE cannot learn from either, which is the same dead end reached from
                       the opposite direction.
  6. curator hygiene -- whether the curator's ADDs contain anything new. On the formula text
                       arm 43 of 75 pasted a rendered playbook line into the rule text and 23
                       were verbatim re-adds; checks 1-5 all passed while a third of curation
                       was adding second copies of rules with zeroed counters.

    uv run python check_learning_signal.py results/ace_run_..._offline

Reads only files on disk. No network, no Docker, no LLM.
"""
import argparse
import ast
import collections
import contextlib
import io
import json
import os
import re
import sys

from playbook_utils import (
    extract_json_from_text,
    format_playbook_line,
    sanitize_curator_operations,
)
from utils import extract_answer


def _load_logs(run_dir):
    log_dir = os.path.join(run_dir, "detailed_llm_logs")
    if not os.path.isdir(log_dir):
        return []
    out = []
    for name in os.listdir(log_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(log_dir, name)
        try:
            with open(path, encoding="utf-8") as fh:
                out.append(json.load(fh))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _task_name(run_dir):
    """Task this run was executed on, or None. Selects the [5] answer-validity gate."""
    path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return cfg.get("task_name") or cfg.get("config", {}).get("task_name")


def _answer_is_usable(answer, task_name):
    """Whether an extracted answer is something the environment can actually score.

    Task-specific, because "the model failed to produce a usable answer" has a different
    shape per task and the distinction is the whole point of check [5]: a failure the
    environment cannot even run teaches the reflector nothing.
    """
    if task_name in ("formula", "finer"):
        from eval.finance.data_processor import normalize_numeric
        if task_name == "finer":
            return bool(answer.strip())
        return normalize_numeric(answer) is not None
    try:
        ast.parse(answer)
        return True
    except SyntaxError:
        return False


def analyse(run_dir):
    logs = _load_logs(run_dir)
    task_name = _task_name(run_dir)
    problems = []

    # ---- 1. reflector tags ---------------------------------------------------------
    # Counted from the raw response rather than the applied tags, so a model that emits
    # 'harmful' and a model whose 'harmful' is being filtered out look different here.
    tags_by_outcome = collections.defaultdict(collections.Counter)
    reflections = 0
    for entry in logs:
        if entry.get("role") != "reflector":
            continue
        reflections += 1
        # ACE names the correct-answer reflection differently from the failure rounds.
        outcome = "correct" if "reflect_on_correct" in entry.get("call_id", "") else "failed"
        for match in re.finditer(r'"tag"\s*:\s*"(\w+)"', entry.get("response") or ""):
            tags_by_outcome[outcome][match.group(1).lower()] += 1

    total_tags = sum(sum(c.values()) for c in tags_by_outcome.values())
    harmful = sum(c.get("harmful", 0) for c in tags_by_outcome.values())
    helpful_on_failure = tags_by_outcome["failed"].get("helpful", 0)

    print(f"\n[1] reflector tags   ({reflections} reflections, {total_tags} tags)")
    for outcome in ("correct", "failed"):
        counts = dict(tags_by_outcome[outcome])
        print(f"    on {outcome:<8} attempts: {counts or '{}'}")
    if reflections == 0:
        print("    no reflector calls -- eval_only run, nothing to check")
    elif harmful == 0:
        problems.append("reflector never emitted 'harmful': memory can only accumulate")
        print("    FAIL  'harmful' never emitted")
    else:
        print(f"    ok    'harmful' emitted {harmful}x")
    if helpful_on_failure:
        print(f"    note  {helpful_on_failure} 'helpful' tag(s) on failed attempts "
              f"(demoted to neutral at apply time)")

    # ---- 2. curator refines --------------------------------------------------------
    curator_calls = 0
    refines_present = 0
    refines_nonnull = 0
    for entry in logs:
        if entry.get("role") != "curator":
            continue
        curator_calls += 1
        found = re.findall(r'"refines"\s*:\s*("(?:[^"]*)"|null)', entry.get("response") or "")
        if found:
            refines_present += 1
            refines_nonnull += sum(1 for v in found if v != "null")

    print(f"\n[2] curator refines  ({curator_calls} curator calls)")
    if curator_calls == 0:
        print("    no curator calls -- eval_only run, nothing to check")
    elif refines_present == 0:
        problems.append("curator never emitted 'refines': the graph arm has nothing to act on")
        print("    FAIL  field absent from every response")
    else:
        print(f"    ok    field present in {refines_present}/{curator_calls} responses, "
              f"{refines_nonnull} non-null (an actual supersession)")
        if refines_nonnull == 0:
            print("    note  always null. Not necessarily wrong, but the graph arm will not "
                  "evict anything, so it reduces to text + edges.")

    # ---- 3. headroom ----------------------------------------------------------------
    # One row per training step, not per log line: ace.py logs bullet usage twice for a
    # correct sample (once after the initial generation, once again in the correct branch
    # with the reflection attached) and only once for an incorrect one. Counting raw rows
    # therefore double-counts every success -- it read 0.537 on a run whose real pre-train
    # accuracy was 0.367. Keeping the first row per step reproduces train_results.json's
    # pre_train_accuracy exactly, which is the number ACE itself reports.
    usage_rows = _read_jsonl(os.path.join(run_dir, "bullet_usage_log.jsonl"))
    usage = {}
    for row in usage_rows:
        usage.setdefault((row.get("epoch"), row.get("step")), row)
    usage = list(usage.values())

    print(f"\n[3] headroom         ({len(usage)} training steps, "
          f"{len(usage_rows)} log rows)")
    if usage:
        correct = sum(1 for r in usage if r.get("is_correct"))
        acc = correct / len(usage)
        failures = len(usage) - correct
        print(f"    train accuracy {acc:.3f} ({correct}/{len(usage)}), {failures} failures to learn from")
        # Inclusive: the gemma run that produced a directionless memory sat at exactly
        # 0.850, which a strict '>' would wave through.
        if acc >= 0.85:
            problems.append(f"train accuracy {acc:.3f} leaves almost nothing to learn from "
                            f"({failures} failures across {len(usage)} steps)")
            print("    FAIL  at or above 0.85 -- too few failures for the loop to have material")
        elif acc < 0.20:
            # A floor as well as a ceiling. The probe run at 0.000 passed the old
            # ceiling-only check while being completely broken -- the generator was
            # emitting code wrapped in a stray quote pair, so every sample failed on a
            # syntax error. Uniform failure is not headroom; there is no signal in it.
            problems.append(
                f"train accuracy {acc:.3f} is too low to be headroom -- at this rate the "
                f"failures are probably systematic (check [5]), not the model reasoning badly"
            )
            print("    FAIL  below 0.20 -- suspiciously uniform, look for a systematic fault")
        else:
            print("    ok    inside the 0.20-0.85 band")
    else:
        print("    no bullet_usage_log.jsonl -- eval_only run, nothing to check")

    # ---- 4. bullet citations --------------------------------------------------------
    if usage:
        cited = sum(1 for r in usage if r.get("bullet_ids_used"))
        distinct = {b for r in usage for b in (r.get("bullet_ids_used") or [])}
        print("\n[4] bullet citations")
        print(f"    {cited}/{len(usage)} generations cited a bullet, {len(distinct)} distinct bullets cited")
        if cited == 0:
            problems.append("generator never cited a bullet: every counter downstream is dead")
            print("    FAIL  no citations at all")
        else:
            print("    ok")

    # ---- 5. output integrity --------------------------------------------------------
    # A weaker generator only creates useful headroom if its extra failures are *reasoning*
    # failures. Failures where the model could not hold the output schema teach the reflector
    # nothing and break the curator outright.
    print("\n[5] output integrity")
    counts = collections.Counter()
    for entry in logs:
        role = entry.get("role")
        response = entry.get("response") or ""
        if role not in ("generator", "reflector", "curator"):
            continue
        counts[f"{role}_total"] += 1
        if role == "generator":
            # extract_answer() prints the raw response on every parse failure, which would
            # bury this report under hundreds of lines from a badly-behaved model.
            with contextlib.redirect_stdout(io.StringIO()):
                answer = extract_answer(response)
            if answer.startswith("raise ValueError("):
                counts["generator_bad"] += 1
                counts["generator_unrunnable"] += 1
            elif not _answer_is_usable(answer, task_name):
                # Extracting cleanly is not enough. Code that does not parse fails every
                # assertion on a syntax error, and a numeric answer the scorer cannot read
                # is marked wrong whatever the reasoning behind it -- neither teaches the
                # reflector anything about the problem. This is what a 0.000 train accuracy
                # looks like from here.
                counts["generator_unrunnable"] += 1
        else:
            parsed = extract_json_from_text(response)
            key = "bullet_tags" if role == "reflector" else "operations"
            if not isinstance(parsed, dict) or key not in parsed:
                counts[f"{role}_bad"] += 1

    for role, limit in (("generator", 0.15), ("reflector", 0.20), ("curator", 0.20)):
        total = counts[f"{role}_total"]
        if not total:
            continue
        bad = counts[f"{role}_bad"]
        rate = bad / total
        status = "FAIL" if rate > limit else "ok  "
        print(f"    {status}  {role:<10} {bad}/{total} unparseable ({rate:.1%}, limit {limit:.0%})")
        if rate > limit:
            problems.append(
                f"{role} produced unparseable output on {rate:.1%} of calls -- this model "
                f"cannot hold the schema, so its failures are not learnable"
            )

    gen_total = counts["generator_total"]
    if gen_total:
        unrunnable = counts["generator_unrunnable"]
        rate = unrunnable / gen_total
        status = "FAIL" if rate > 0.15 else "ok  "
        if task_name in ("formula", "finer"):
            what, why = "are not a scorable answer", "are unreadable to the scorer"
        else:
            what, why = "do not parse as Python", "fail on a syntax error"
        print(f"    {status}  {'generator':<10} {unrunnable}/{gen_total} {what} "
              f"({rate:.1%}, limit 15%)")
        if rate > 0.15:
            problems.append(
                f"{rate:.1%} of generated answers {why} rather than on their logic, "
                f"so they are not learnable"
            )

    # ---- 6. curator hygiene ----------------------------------------------------------
    # Every check above can pass while the curator is doing no work at all. On the formula
    # text arm 43 of 75 ADDs arrived with a rendered playbook line pasted into `content`
    # ("[calc-00013] helpful=5 harmful=0 :: ...", once nested three deep) and 23 were verbatim
    # re-adds of a rule already in the playbook -- so roughly a third of all curation added
    # nothing but a second copy with zeroed counters. It scales with playbook size (0 of 221
    # ADDs across every MBPP run, 5/15 on formula probe 3), which is why nothing caught it
    # on the short runs.
    #
    # Measured from curator_operations_diff.jsonl, which records what the model emitted
    # *before* Curator.curate sanitises it, so this stays a read on the model after the fix.
    print("\n[6] curator hygiene")
    diff_rows = _read_jsonl(os.path.join(run_dir, "curator_operations_diff.jsonl"))
    adds = [r for r in diff_rows if r.get("operation_type") == "ADD"]
    if not adds:
        print("    no ADD operations logged -- eval_only run, nothing to check")
    else:
        # Replayed per curator call, against a playbook rebuilt from the ADDs that survived.
        # The memory is ADD-only, so that reconstruction is what the curator was shown --
        # except in the graph arm, where a superseded rule is withdrawn from its view and may
        # legitimately be re-added, so the duplicate count there is an upper bound.
        batches = collections.OrderedDict()
        for row in adds:
            batches.setdefault(row.get("call_id"), []).append(row)

        playbook_lines = []
        totals = collections.Counter()
        for rows in batches.values():
            ops = [{"type": "ADD", "section": r.get("section", "others"),
                    "content": r.get("content", "")} for r in rows]
            kept, report = sanitize_curator_operations(ops, "\n".join(playbook_lines))
            totals.update(report)
            for op in kept:
                content = " ".join((op.get("content") or "").split())
                playbook_lines.append(
                    format_playbook_line(f"add-{len(playbook_lines) + 1:05d}", 0, 0, content))

        n = totals["adds"]
        dropped = totals["dropped"]
        # Reported, not failed on: Curator.curate strips these, so a prefixed ADD still
        # lands as a correct rule. It is the rate at which the *model* does it, kept visible
        # because it scales with playbook size and predicts nothing good.
        prefixed = totals["stripped"]
        marker = "note " if prefixed / n > 0.10 else "     "
        print(f"    {marker} {prefixed}/{n} ADDs carried a pasted rendered-bullet prefix "
              f"({prefixed / n:.1%}, stripped at run time)")
        print(f"    {dropped}/{n} ADDs added nothing new "
              f"({totals['deduped']} duplicate, {totals['emptied']} empty after stripping)")
        print(f"    {totals['kept']}/{n} ADDs were new knowledge")
        if dropped / n > 0.50:
            problems.append(
                f"{dropped / n:.1%} of curator ADDs were duplicates of rules already in the "
                f"playbook -- the curator has stopped producing new knowledge and the memory "
                f"arm is paying full price for it"
            )
            print("    FAIL  above 50% -- curation is not adding knowledge")
        else:
            print("    ok    below the 50% limit")

    failure_log = os.path.join(run_dir, "detailed_llm_logs", "curator_failures.txt")
    if os.path.exists(failure_log):
        with open(failure_log, encoding="utf-8", errors="replace") as fh:
            skipped = fh.read().count("CURATOR FAILURE LOG")
        print(f"    note  curator_failures.txt records {skipped} skipped curation(s)")

    return problems


def _expand_run_dirs(path):
    """Resolve a path to the run directories under it.

    --save_path nests the run one level down (results/mbpp/probes/p3/ace_run_.../), so the
    obvious invocation points at the parent. Without this, every check found nothing,
    reported 'nothing to check' for each, and the script printed a green verdict for a
    directory containing no run at all -- the exact false pass it exists to prevent.
    """
    if os.path.isdir(os.path.join(path, "detailed_llm_logs")):
        return [path]
    nested = sorted(
        os.path.join(path, name) for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name, "detailed_llm_logs"))
    )
    return nested


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+",
                    help="run directories, or a parent containing them")
    args = ap.parse_args()

    resolved = []
    for path in args.run_dirs:
        if not os.path.isdir(path):
            print(f"not a directory: {path}")
            return 2
        found = _expand_run_dirs(path)
        if not found:
            print(f"no run directory found under {path} -- expected a detailed_llm_logs/ "
                  f"folder either there or one level down. Refusing to report a verdict on "
                  f"a directory with nothing in it.")
            return 2
        resolved.extend(found)

    all_problems = []
    for run_dir in resolved:
        print("=" * 78)
        print(os.path.basename(os.path.normpath(run_dir)))
        print("=" * 78)
        for problem in analyse(run_dir):
            all_problems.append(f"{os.path.basename(os.path.normpath(run_dir))}: {problem}")
        print()

    print("=" * 78)
    if all_problems:
        print("SIGNAL IS DEAD -- a full arm would burn compute on a foregone conclusion:")
        for problem in all_problems:
            print(f"  - {problem}")
        return 1
    print("Learning signal is alive. Safe to spend a full arm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
