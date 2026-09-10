"""Recompute every figure in `docs/FINER_RESULTS.md` section 6, offline, from finished runs.

Section 6 is the substantive finding of the `finer` experiment, this script re-derives them. 

No LLM calls, no network. Reads finished run directories only.

    uv run python analyze_finer_coverage.py results/finer/final/seed*/*/ace_run_* \
        results/finer/repaired_v2/seed0/none/ace_run_*

**What "covered" means, because it is a judgement call and it moves every number below.** Every
`finer` rule is meant to be a mapping (`IF <condition> THEN <ExactTagName>`), so a playbook is
read as the set of vocabulary tags its rules name as a target; `analysis_common.mapping_targets`
is that reading and its docstring records what the earlier version got wrong. Two restrictions on
top:

* A tag counts for an arm only when **every one of that arm's replicate playbooks** names it.
* The set is intersected with the tags that actually appear as a gold answer. Two printed tags
  (`DefinedContributionPlanCostRecognized`, `IncomeLossFromEquityMethodInvestments`) are never
  correct on this test split, so a rule about one of them can neither help nor hurt and counting
  it would inflate the coverage figure against a denominator it cannot affect.

**On the self-check.** The script asserts that the two piles, weighted by size, reproduce the gap
`compare_arms.py` publishes for this arm, obtained by calling `pooled_arm_scores` here rather than
by averaging the same rows a second time. Checking against a locally recomputed mean would let the
two drift together and pass on a wrong pair. It catches a decomposition scored on a *different
scale* from the result it decomposes, which is a real defect this file has had. It is blind to how
`covered` is defined: the identity holds for any partition of the questions, so a wrong coverage
rule moves both piles and still balances. Nothing here validates the partition itself.
"""

import argparse
import collections
import json
import os
import sys

from analysis_common import (
    RULE_LINE,
    arm_of,
    collect_runs,
    covered_tags,
    mapping_targets,
    prediction_rows,
)
from check_learning_signal import _RENDERED_RULE, _test_generation_prompts
from compare_arms import METRICS, load_predictions, pooled_arm_scores
from eval.finance.data_processor import finer_tag_counts

DEFAULT_TRAIN = os.path.join("eval", "finance", "data", "finer_train_200.jsonl")
DEFAULT_TEST = os.path.join("eval", "finance", "data", "finer_test_subset_006_seed42.jsonl")

#: The printed vocabulary sits after this marker as a comma-separated list. Parsed as a list
#: rather than pattern-matched out of the prose, which over-collects: a `[A-Z][A-Za-z]{5,}` scan
#: of the same prompt head returns 160 tokens against the 139 tags actually printed.
VOCAB_MARKER = "Here is a list of US GAAP tags options:"

EXPOSURE_BANDS = ((0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 10**9, "11+"))


def band(count):
    for low, high, label in EXPOSURE_BANDS:
        if low <= count <= high:
            return label
    raise AssertionError(count)


def tag_counts(path):
    """Occurrences of each gold tag in a split file, lowercased to match `finer_tag_counts`."""
    counts = collections.Counter()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            for tag in json.loads(line)["target"].split(","):
                tag = tag.strip().strip("\"'[]").lower()
                if tag:
                    counts[tag] += 1
    return counts


def printed_vocabulary(run_dir):
    """The tag list every prompt carries, read off a saved generator prompt.

    Taken from the prompt rather than from a constant so that a change to the printed list shows
    up here instead of being silently assumed away.
    """
    prompts = _test_generation_prompts(run_dir)
    if not prompts:
        return None
    prompt = prompts[-1][1]
    if VOCAB_MARKER not in prompt:
        return None
    listing = prompt.split(VOCAB_MARKER, 1)[1].split("What is best tag")[0]
    # The list runs straight into the instruction sentence ("...OfPossibleLoss. Answer the
    # following 4 independent questions ... saperated by a comma (,)."), which both corrupts the
    # final tag and contributes two comma-separated fragments of prose. Cutting each entry at its
    # first period recovers the last tag; a trailing strip would discard it, because that entry
    # does not end in a period.
    tags = set()
    for part in listing.split(","):
        candidate = part.split(".", 1)[0].strip()
        if candidate.isalnum():
            tags.add(candidate.lower())
    return tags


def predictions(run_dir):
    """{row index: (predicted answer, gold answer)}, or None.

    File precedence is `compare_arms.PREDICTION_FILES` through `analysis_common.prediction_rows`,
    so a rescored run is preferred here exactly as it is there.
    """
    rows = prediction_rows(run_dir)
    if rows is None:
        return None
    return {row["index"]: (row.get("final_answer") or "", row.get("target") or "")
            for row in rows}


def pooled_gap(control_dirs, arm_dirs):
    """The arm's pooled gap over the control, as `compare_arms.py` computes it, or (None, why).

    The point of asserting against this rather than against a locally recomputed mean is that the
    published +8.84pp comes from `pooled_arm_scores`: each row scored as its mean across an arm's
    replicates, then averaged. If the decomposition is checked only against a number this file
    derives itself, the two can drift together and the check passes on a wrong pair.

    `pooled_arm_scores` refuses unequal replicate counts, and the control is legitimately run once
    on this task. Repeating it is exact only when its runs are identical, which is the same
    licence `docs/FINER_RESULTS.md` uses when it passes the control three times, so that is
    checked here rather than assumed.
    """
    scorer, _binary = METRICS["finer_per_tag"]
    control = [load_predictions(d, scorer=scorer)[0] for d in control_dirs]
    arm = [load_predictions(d, scorer=scorer)[0] for d in arm_dirs]
    if any(rows is None for rows in control + arm):
        return None, "a run has no predictions file"
    if len(control) != len(arm):
        if any(rows != control[0] for rows in control[1:]):
            return None, (f"{len(control)} control run(s) against {len(arm)} in the arm, and the "
                          "control runs differ, so neither padding nor pooling is defensible")
        control = control * len(arm)
    scores, _replicates, shared, refusal = pooled_arm_scores(
        {"none": [{"preds": rows} for rows in control],
         "arm": [{"preds": rows} for rows in arm]})
    if scores is None:
        return None, refusal or "nothing to pool"
    return (sum(scores["arm"][i] for i in shared) / len(shared)
            - sum(scores["none"][i] for i in shared) / len(shared)), None


def graded(runs):
    """Yield (gold_tag, predicted_tag, matched) per tag question, over every replicate.

    Scoring goes through `finer_tag_counts`, the same comparison `_evaluate_finer_accuracy` and
    `compare_arms.py --metric finer_per_tag` use. Nothing else may score tags here.
    """
    for rows in runs:
        for _index, (predicted, gold) in rows.items():
            _correct, _total, per_position = finer_tag_counts(predicted, gold)
            for _slot, got, want, matched in per_position:
                if want:
                    yield want, got, bool(matched)


def rate(hits, total):
    return hits / total if total else 0.0


def delivery(run_dir, vocabulary):
    """Per question, was a rule mapping to the gold tag among the rules actually rendered?

    Uses this run's own playbook, not the cross-replicate intersection, because the question is
    about what *this* run delivered. Rendered ids come from the saved test prompts through
    `check_learning_signal`'s reader and id pattern, since `bullet_usage_log.jsonl` covers
    training only and the id shape is not what the harness emits.
    """
    playbook = os.path.join(run_dir, "best_playbook.txt")
    rows = predictions(run_dir)
    if not os.path.isfile(playbook) or not rows:
        return None

    rule_tags = {}
    with open(playbook, encoding="utf-8") as handle:
        for line in handle:
            match = RULE_LINE.match(line.rstrip())
            if not match:
                continue
            tags = mapping_targets(match.group(2), vocabulary)
            if tags:
                rule_tags[match.group(1)] = tags
    if not rule_tags:
        return None
    covered = set().union(*rule_tags.values())

    prompts = dict(_test_generation_prompts(run_dir))
    hits = total = 0
    for index, (predicted, goldrow) in rows.items():
        prompt = prompts.get(index)
        if prompt is None:
            continue
        rendered = set(_RENDERED_RULE.findall(prompt))
        shown = set().union(*[rule_tags[r] for r in rendered if r in rule_tags]) if rendered else set()
        _correct, _t, per_position = finer_tag_counts(predicted, goldrow)
        for _slot, _got, want, _ok in per_position:
            if want and want in covered:
                total += 1
                hits += want in shown
    return (hits, total) if total else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="run directories, or parents of ace_run_*")
    parser.add_argument("--train", default=DEFAULT_TRAIN, help="training split, for exposure")
    parser.add_argument("--test", default=DEFAULT_TEST, help="test split, for the gold tag set")
    parser.add_argument("--tolerance", type=float, default=0.0005,
                        help="allowed |weighted piles - overall gap| before the self-check fails")
    args = parser.parse_args()

    by_arm = collections.defaultdict(list)
    for run_dir in collect_runs(args.paths):
        by_arm[arm_of(run_dir)].append(run_dir)

    if "none" not in by_arm:
        sys.exit("no control arm (memory_kind == none) among the given runs; it is the baseline "
                 "every pile is scored against")
    memory_arms = [arm for arm in ("text", "graph", "cocite") if arm in by_arm]
    if not memory_arms:
        sys.exit("no memory arm among the given runs")

    vocabulary = None
    for run_dir in sum(by_arm.values(), []):
        vocabulary = printed_vocabulary(run_dir)
        if vocabulary:
            break
    if not vocabulary:
        sys.exit("no saved generator prompt found; the printed vocabulary cannot be recovered")

    exposure = tag_counts(args.train)
    gold = tag_counts(args.test)
    control_runs = [predictions(d) for d in by_arm["none"]]
    if any(rows is None for rows in control_runs):
        sys.exit("a control run has no predictions file")

    print(f"printed vocabulary: {len(vocabulary)} tags")
    print(f"test gold tags:     {len(gold)} distinct over {sum(gold.values())} tag questions")
    print(f"training split:     {args.train}, {sum(exposure.values())} tag occurrences over "
          f"{len(exposure)} distinct tags")
    unseen = [tag for tag in gold if exposure.get(tag, 0) == 0]
    print(f"test tags never in training: {len(unseen)} "
          f"({sum(gold[t] for t in unseen)} questions, "
          f"{100.0 * sum(gold[t] for t in unseen) / sum(gold.values()):.1f}%)")
    print(f"test tags with 3+ training examples: "
          f"{sum(1 for tag in gold if exposure.get(tag, 0) >= 3)} of {len(gold)}")
    print(f"control: {len(control_runs)} run(s); " +
          "; ".join(f"{arm}: {len(by_arm[arm])} run(s)" for arm in memory_arms))

    control_graded = list(graded(control_runs))
    overall_control = rate(sum(1 for _w, _g, ok in control_graded if ok), len(control_graded))

    print("\n" + "=" * 88)
    print("ROWS BY NUMBER OF TAGS CORRECT (mean over replicates, out of the test row count)")
    print(f"{'arm':<10}" + "".join(f"{k}/4".rjust(9) for k in range(5)))
    for arm, runs in ([("none", control_runs)]
                      + [(a, [predictions(d) for d in by_arm[a]]) for a in memory_arms]):
        buckets = collections.Counter()
        for rows in runs:
            for _index, (predicted, goldrow) in rows.items():
                correct, _total, _per = finer_tag_counts(predicted, goldrow)
                buckets[correct] += 1
        print(f"{arm:<10}" + "".join(f"{buckets[k] / len(runs):9.1f}" for k in range(5)))

    for arm in memory_arms:
        runs = [predictions(d) for d in by_arm[arm]]
        if any(rows is None for rows in runs):
            print(f"\n{arm}: a run has no predictions file, skipped")
            continue
        playbooks = [os.path.join(d, "best_playbook.txt") for d in by_arm[arm]]
        playbooks = [p for p in playbooks if os.path.isfile(p)]
        if not playbooks:
            print(f"\n{arm}: no best_playbook.txt, skipped")
            continue
        # Covered only where every replicate agrees, and only among tags that are ever a gold
        # answer: a rule about a tag no question asks for cannot move this test's accuracy.
        covered = set.intersection(*[covered_tags(p, vocabulary) for p in playbooks]) & set(gold)
        arm_graded = list(graded(runs))
        overall_arm = rate(sum(1 for _w, _g, ok in arm_graded if ok), len(arm_graded))

        print("\n" + "=" * 88)
        print(f"{arm.upper()}  --  coverage decomposition "
              f"({len(covered)} of {len(gold)} gold tags covered in all {len(playbooks)} replicates)")
        print(f"{'pile':<12}{'questions':>11}{'no memory':>12}{'with memory':>13}{'delta':>10}")

        weighted = 0.0
        questions_total = 0
        for want_covered, label in ((True, "covered"), (False, "uncovered")):
            def score(rows, want=want_covered):
                hits = seen = 0
                for goldtag, _got, ok in rows:
                    if (goldtag in covered) != want:
                        continue
                    seen += 1
                    hits += ok
                return rate(hits, seen), seen

            before, _n_before = score(control_graded)
            after, _n_after = score(arm_graded)
            # The pile is a property of the test split, not of how many runs were passed:
            # `graded` yields one entry per question *per replicate*, so counting its rows would
            # multiply the pile by the replicate count and skew the weighting between arms whose
            # replicate counts differ.
            pile = sum(count for tag, count in gold.items() if (tag in covered) == want_covered)
            print(f"{label:<12}{pile:>11}{before:>12.4f}{after:>13.4f}{after - before:>+10.4f}")
            weighted += pile * (after - before)
            questions_total += pile

        weighted /= questions_total
        gap = overall_arm - overall_control
        print(f"{'':<12}{questions_total:>11}{overall_control:>12.4f}{overall_arm:>13.4f}"
              f"{gap:>+10.4f}   <- overall")

        published, why = pooled_gap(by_arm["none"], by_arm[arm])
        if published is None:
            print(f"  self-check: SKIPPED, cannot build the published statistic ({why})")
        else:
            status = "OK" if abs(weighted - published) <= args.tolerance else "MISMATCH"
            print(f"  self-check: piles weighted by size = {weighted:+.4f} against the pooled "
                  f"`compare_arms` gap {published:+.4f} [{status}]")
            print("  (the piles are checked against the number the results file publishes, not "
                  "against a\n   mean this script recomputes, so the two cannot drift together. "
                  "It is a scale check: it\n   cannot tell whether `covered` is defined well, "
                  "since the identity holds for any partition.)")
            if status == "MISMATCH":
                sys.exit(1)

        def shortlist(rows):
            wrong = pulled = 0
            for goldtag, got, ok in rows:
                if goldtag in covered or ok:
                    continue
                wrong += 1
                pulled += bool(got) and got in covered
            return pulled, wrong

        pulled_c, wrong_c = shortlist(control_graded)
        pulled_a, wrong_a = shortlist(arm_graded)
        print("  on the UNCOVERED pile, when the model is wrong it names a COVERED tag:")
        print(f"    control {pulled_c:>5}/{wrong_c:<5} = {100.0 * rate(pulled_c, wrong_c):5.1f}%"
              f"   |   {arm} {pulled_a:>5}/{wrong_a:<5} = {100.0 * rate(pulled_a, wrong_a):5.1f}%")

        print("\n  COVERAGE TRACKS EXPOSURE (test tags, not questions)")
        print(f"    {'examples':<10}{'tags':>6}{'covered':>9}{'share':>8}")
        tally = collections.defaultdict(lambda: [0, 0])
        for tag in gold:
            cell = tally[band(exposure.get(tag, 0))]
            cell[0] += 1
            cell[1] += tag in covered
        for _low, _high, label in EXPOSURE_BANDS:
            total, hits = tally[label]
            if total:
                print(f"    {label:<10}{total:>6}{hits:>9}{100.0 * hits / total:>7.0f}%")

        print("\n  HOLDING EXPOSURE FIXED (tag questions)")
        print(f"    {'examples':<10}{'rule?':<7}{'questions':>10}{'no memory':>11}"
              f"{'with memory':>13}{'delta':>10}")
        cells = collections.defaultdict(lambda: [0, 0, 0, 0])
        for rows, base in ((control_graded, 0), (arm_graded, 2)):
            for goldtag, _got, ok in rows:
                cell = cells[(band(exposure.get(goldtag, 0)), goldtag in covered)]
                cell[base] += 1
                cell[base + 1] += ok
        for _low, _high, label in EXPOSURE_BANDS:
            for has_rule in (True, False):
                cell = cells.get((label, has_rule))
                if not cell or not cell[0]:
                    continue
                before = rate(cell[1], cell[0])
                after = rate(cell[3], cell[2])
                print(f"    {label:<10}{'yes' if has_rule else 'no':<7}"
                      f"{cell[0] // len(control_runs):>10}"
                      f"{before:>11.3f}{after:>13.3f}{after - before:>+10.4f}")

        print("\n  RETRIEVAL DELIVERY: of the questions a playbook could answer, how often was "
              "the matching rule actually rendered for that question?")
        for run_dir in by_arm[arm]:
            delivered = delivery(run_dir, vocabulary)
            if delivered is None:
                print(f"    {os.path.basename(run_dir):<44} no readable rendered playbooks")
            else:
                hits, total = delivered
                print(f"    {os.path.basename(run_dir):<44} {hits:>5}/{total:<5} = "
                      f"{100.0 * rate(hits, total):5.1f}%")


if __name__ == "__main__":
    main()
