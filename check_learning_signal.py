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
import glob
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


def _finer_pre_train_per_tag(run_dir):
    """Per-tag accuracy of the pre-train generations, or None if the file is absent.

    Same rows check [3] reads out of bullet_usage_log, scored on the metric the run's own
    train_results.json reports and ace.py:1012 selects on.
    """
    path = os.path.join(run_dir, "pre_train_post_train_results.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    from eval.finance.data_processor import finer_tag_counts
    got = total = 0
    for row in rows:
        answer = (row.get("pre_train_result") or {}).get("final_answer")
        if answer is None:
            continue
        count, tags, _ = finer_tag_counts(answer, row.get("target") or "")
        got += count
        total += tags
    return got / total if total else None


def _answer_is_usable(answer, task_name):
    """Whether an extracted answer is something the environment can actually score.

    Task-specific, because "the model failed to produce a usable answer" has a different
    shape per task and the distinction is the whole point of check [5]: a failure the
    environment cannot even run teaches the reflector nothing.
    """
    if task_name in ("formula", "finer"):
        from eval.finance.data_processor import normalize_finer_answer, normalize_numeric
        if task_name == "finer":
            # One finer row is four sub-questions, so a usable answer is four non-empty tags
            # after the same normalisation the scorer applies. "Non-empty" would pass anything
            # and report 0/N on every run, which is not a check. A short or long list is a
            # format failure: it scores partially at best, and the position it goes wrong at
            # is not the position the model got wrong.
            parts = [p.strip() for p in normalize_finer_answer(answer).split(",")]
            return len(parts) == 4 and all(parts)
        return normalize_numeric(answer) is not None
    try:
        ast.parse(answer)
        return True
    except SyntaxError:
        return False


def _run_config(run_dir):
    path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


#: EXPERIMENT_FINER.md section 7, gate added 2026-08-30: at least this many distinct rendered
#: rule sets over the first `GENERATIONS_READ` test generations. Fixed counts, not a share of
#: however many prompts happen to parse.
DISTINCT_SET_FLOOR = 20
GENERATIONS_READ = 25

# EXPERIMENT_FINER.md section 7, gate added 2026-09-02. `finer` only: the curator is asked for
# one `IF <condition> THEN <tag>` line per rule, filed in one section. A live rule counts as a
# mapping if everything after the `::` separator is that and nothing else.
MAPPING_SHARE_FLOOR = 0.50
OFF_SECTION_CEILING = 0.10
FINER_SECTION = "ctx"
_LIVE_RULE = re.compile(r"^\[([a-z]+)-\d+\] helpful=\d+ harmful=\d+ :: (.*)$", re.M)
# Six characters, not ten: the shortest tags in the vocabulary are `Goodwill` and `Revenues`,
# and a ten-character floor scored a compliant rule naming either of them as non-compliant.
# The end anchor is what rejects prose -- a directive is never one capitalised word.
_MAPPING = re.compile(r"^IF .+ THEN [A-Z][A-Za-z0-9]{5,}\.?$")

#: A rendered rule line, as `playbook_utils.format_playbook_line` writes it.
_RENDERED_RULE = re.compile(r"^\[([\w-]+)\] helpful=", re.M)


def _test_generation_prompts(run_dir):
    """(index, prompt) for the final test generations, in index order, one per index.

    **Last wins per index, and that is the whole point of this function.** Validation
    evaluations write under the same `generator_test_eval_<index>_*` prefix, so a plain
    lexicographic walk mixes an earlier checkpoint's generations into the population
    (`..._0_`, `..._0_`, `..._10_`, ...). The filenames carry a timestamp, so the final test is
    always the last file for a given index. `analyze_walk_share.py` carries the same correction.

    Both readers below go through here so the correction cannot be fixed in one and missed in
    the other, which is exactly how the bug this replaces came about.
    """
    by_index = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "detailed_llm_logs",
                                              "generator_test_eval_*.json"))):
        match = re.match(r"generator_test_eval_(\d+)_", os.path.basename(path))
        if not match:
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                prompt = json.load(fh).get("prompt") or ""
        except (OSError, ValueError):
            continue
        if isinstance(prompt, str):
            by_index[int(match.group(1))] = prompt
    return [(index, by_index[index]) for index in sorted(by_index)]


def _test_rendered_sets(run_dir, limit=25):
    """The rule ids retrieval put in front of the model, per test generation, in index order.

    Read off the saved prompts because `bullet_usage_log.jsonl` covers training generations only.
    A generation whose prompt carries no rules does not consume a slot.
    """
    sets = []
    for _, prompt in _test_generation_prompts(run_dir):
        ids = _RENDERED_RULE.findall(prompt)
        if ids:
            sets.append(frozenset(ids))
        if len(sets) >= limit:
            break
    return sets


def _rendered_slice_shares(run_dir, limit=25):
    """Per test generation, the share of rendered rules in the requested IF..THEN <tag> form.

    One value per generation rather than one pooled ratio over all rendered slots, so every
    generation weighs the same whatever its slice size -- the gate asks what a generator call
    reads, and a call with a big slice is not more than one call. The distinct-rule share is
    returned beside it for reporting only.

    Why the rendered slice rather than the stored playbook: docs/EXPERIMENT_FINER.md section 9,
    2026-09-03, which carries the measurement.
    """
    shares, seen = [], {}
    for _, prompt in _test_generation_prompts(run_dir):
        rules = _LIVE_RULE.findall(prompt)
        if not rules:
            continue
        shares.append(sum(1 for _, text in rules if _MAPPING.match(text.strip())) / len(rules))
        for section, text in rules:
            seen[(section, text)] = bool(_MAPPING.match(text.strip()))
        if len(shares) >= limit:
            break
    return shares, seen


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
        if task_name == "finer":
            # `is_correct` on finer is all-four-exact, which is roughly the fourth power of
            # the per-tag number the run itself reports and selects checkpoints on. Judging
            # headroom on it fails a healthy run: the first probes read 0.100 and 0.060
            # all-four against 0.355 and 0.270 per tag. Band on per-tag, print both.
            per_tag = _finer_pre_train_per_tag(run_dir)
            if per_tag is not None:
                acc = per_tag
                print(f"    per-tag {acc:.3f} (all-four above; the band applies to per-tag)")
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

    print("\n[7] retrieval actually retrieves  (EXPERIMENT_FINER.md section 7, gate added 2026-08-30)")
    # THE GATE THAT WAS MISSING WHEN THE 2026-08-30 DEFECT SHIPPED. `all-mpnet-base-v2` truncates
    # at 384 tokens, and on `finer` the first ~1,500 characters of every question are the same
    # 140-tag taxonomy, so every question encoded identically (pairwise cosine exactly 1.0000),
    # `score_against_query` returned one ranking, and both seed-0 arms rendered ONE rule set
    # across all 441 test generations. Every other gate in section 7 passed. Retrieval was inert
    # and no artifact said so.
    #
    # The rendered set is recovered from the saved test prompts rather than from
    # `bullet_usage_log.jsonl`, which records training generations only.
    config = _run_config(run_dir)
    retrieval = (config.get("retrieval") or "none").lower()
    if retrieval == "none":
        print(f"    skip  retrieval={retrieval!r} -- the whole memory is injected, so there is "
              f"nothing to vary")
    else:
        # Section 7 fixes the bar as a count, not a share: at least DISTINCT_SET_FLOOR distinct
        # sets over the first GENERATIONS_READ test generations. A share would let the gate
        # lower its own bar whenever fewer prompts parse, which is the failure mode a validity
        # check must not have -- 3 readable prompts and 3 distinct sets would pass a 0.8 share.
        rendered = _test_rendered_sets(run_dir, limit=GENERATIONS_READ)
        distinct = len({frozenset(ids) for ids in rendered})
        if len(rendered) < GENERATIONS_READ:
            problems.append(
                f"only {len(rendered)} of the first {GENERATIONS_READ} test generations had a "
                f"readable rendered playbook, so the section 7 retrieval gate cannot be read; "
                f"it needs {DISTINCT_SET_FLOOR} distinct sets out of {GENERATIONS_READ}"
            )
            print(f"    FAIL  only {len(rendered)}/{GENERATIONS_READ} test prompts carried a "
                  f"readable playbook block -- the gate cannot be evaluated")
        else:
            sizes = sorted({len(ids) for ids in rendered})
            print(f"    {distinct} distinct rendered rule sets over the first "
                  f"{GENERATIONS_READ} test generations (sizes {sizes[0]} to {sizes[-1]})")
            if distinct < DISTINCT_SET_FLOOR:
                problems.append(
                    f"retrieval rendered only {distinct} distinct rule sets over "
                    f"{GENERATIONS_READ} test generations (section 7 requires "
                    f"{DISTINCT_SET_FLOOR}) -- the query is not discriminating between "
                    f"questions, so the arm is injecting a fixed slice rather than retrieving, "
                    f"and no accuracy from it is about retrieval"
                )
                print(f"    FAIL  below {DISTINCT_SET_FLOOR} -- retrieval is inert, see "
                      f"EXPERIMENT_FINER.md section 9, 2026-08-30")
            else:
                print(f"    ok    at or above {DISTINCT_SET_FLOOR} of {GENERATIONS_READ}")

    if _task_name(run_dir) == "finer":
        print("\n[8] the curator writes the requested form  "
              "(EXPERIMENT_FINER.md section 7, gate added 2026-09-02)")
        # Two clauses, and they are read on two different populations on purpose.
        #
        # Clause 1 is read on what retrieval RENDERED, because the generator never sees the
        # stored playbook -- it sees a budgeted slice, and the rules that drift out of the
        # requested form win the ranking. Clause 2 is read on the STORED playbook, because
        # arm-to-arm composition is a fact about what the curator wrote.
        #
        # The measurements behind both, and why the population for clause 1 moved on 2026-09-03,
        # are in docs/EXPERIMENT_FINER.md section 9. They are not restated here.
        #
        # Neither clause applies to the control, and the skip comes before any file is read.
        # `--memory none` injects nothing and `ace/ace.py:176` refuses retrieval for it, so there
        # is no rendered slice and no curator to have written anything. Same skip as [7], and for
        # the same reason: a gate that cannot be evaluated must say so rather than fail an arm for
        # a property it does not have.
        if retrieval == "none":
            print(f"    skip  retrieval={retrieval!r} -- no memory is retrieved, so there is no "
                  f"rendered slice and no curator output to read")
        else:
            pb = os.path.join(run_dir, "final_playbook.txt")
            pb_text = open(pb, encoding="utf-8", errors="replace").read() if os.path.exists(pb) else ""
            rules = _LIVE_RULE.findall(pb_text)
            shares, seen = _rendered_slice_shares(run_dir)

            # No fallback to the stored playbook. Section 7 claims this gate can only make a run
            # harder to pass, and falling back to a population that scores several times higher would
            # break that claim exactly when the evidence is missing.
            if not shares:
                problems.append(
                    "no test generation carried a readable rendered playbook, so the section 7 "
                    "curator-form gate cannot be read -- it is not evaluated against the stored "
                    "playbook, which scores far higher and would pass a run that has no evidence"
                )
                print("    FAIL  no test prompt carried a readable rendered playbook")
            else:
                m_share = sum(shares) / len(shares)
                distinct_share = sum(seen.values()) / len(seen)
                print(f"    {m_share:.1%} of rendered rules are a single IF..THEN <tag> line "
                      f"(mean over {len(shares)} test generations, floor {MAPPING_SHARE_FLOOR:.0%})")
                print(f"    {sum(seen.values())}/{len(seen)} distinct rendered rules "
                      f"({distinct_share:.1%}, reported, not gated)")
                if m_share < MAPPING_SHARE_FLOOR:
                    problems.append(
                        f"only {m_share:.1%} of the rules retrieval rendered are the requested "
                        f"IF..THEN <tag> mapping (section 7 requires {MAPPING_SHARE_FLOOR:.0%}) -- "
                        f"most of what the generator reads is not the thing the design is about"
                    )
                    print(f"    FAIL  below {MAPPING_SHARE_FLOOR:.0%}")

            if not rules:
                problems.append("no live rules readable from final_playbook.txt, so the section 7 "
                                "composition clause cannot be read")
                print("    FAIL  final_playbook.txt carried no readable live rules")
            else:
                stored = sum(1 for _, text in rules if _MAPPING.match(text.strip()))
                off = sum(1 for section, _ in rules if section != FINER_SECTION)
                o_share = off / len(rules)
                print(f"    {stored}/{len(rules)} stored rules are in the requested form "
                      f"({stored/len(rules):.1%}, reported, not gated)")
                print(f"    {off}/{len(rules)} stored rules sit outside '{FINER_SECTION}' "
                      f"({o_share:.1%}, ceiling {OFF_SECTION_CEILING:.0%})")
                if o_share > OFF_SECTION_CEILING:
                    problems.append(
                        f"{o_share:.1%} of stored rules sit outside '{FINER_SECTION}' (section 7 "
                        f"allows {OFF_SECTION_CEILING:.0%}) -- compare against the sibling arms "
                        f"before reading any accuracy, because this differs between arms under an "
                        f"identical curator prompt"
                    )
                    print(f"    FAIL  above {OFF_SECTION_CEILING:.0%}")

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
