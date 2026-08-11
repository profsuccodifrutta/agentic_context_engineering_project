#!/usr/bin/env python3
"""Compare finished ACE runs on their per-problem test predictions.

Accuracy alone cannot separate "the memory did nothing" from "the memory fixed 17 problems
and broke 19". Those are different findings and the second one is the interesting one, but
you only see it by pairing the arms problem by problem. That is what this does:

  * per-arm accuracy with a Wilson 95% interval (not normal-approximation -- at n=200 and
    p near 0.3 the normal interval is visibly wrong at the edges)
  * pairwise McNemar exact test on aligned problem indices, reporting the discordant
    counts b and c, not only p
  * on a task whose rows come in correlated blocks, a cluster sign-flip test and cluster
    bootstrap alongside it -- see cluster_paired_stats()
  * when several runs share an arm, mean and sd across seeds, plus a seed-blocked view

Reads <prefix>_test_predictions.jsonl, preferring rescored_ over final_ over test_.

    uv run python compare_arms.py results/mbpp/arms_qwen_seed0/none \
        results/mbpp/arms_qwen_seed0/text results/mbpp/arms_qwen_seed0/graph
    uv run python compare_arms.py --root results/formula/seed0_exploratory/text \
        --root results/formula/seed0_exploratory/graph

Refuses to compare runs that differ on anything the fairness rules pin, unless --force.
"""

import argparse
import contextlib
import io
import itertools
import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict

import llm  # noqa: F401  -- console-encoding guard
from rescore_from_logs import select_final_test_calls
from utils import extract_answer, repair_json_escapes

#: Pinned by the fairness rules in CLAUDE.md. Two arms differing on any of these are not a
#: controlled comparison, whatever the accuracies say. --memory is the independent variable
#: and --seed is deliberately varied, so neither appears here.
PINNED_FIELDS = (
    "generator_model", "reflector_model", "curator_model", "num_ctx", "think",
    "max_num_rounds", "curator_frequency", "eval_steps", "test_workers",
    "use_bulletpoint_analyzer", "dedup", "dedup_threshold",
    "retrieval_budget", "retrieval_seeds", "task_name",
)

PREDICTION_FILES = (
    "rescored_test_predictions.jsonl",
    "final_test_predictions.jsonl",
    "test_test_predictions.jsonl",
)


def expand_run_dirs(path):
    if os.path.isdir(os.path.join(path, "detailed_llm_logs")):
        return [path]
    if not os.path.isdir(path):
        return []
    return sorted(
        os.path.join(path, name) for name in sorted(os.listdir(path))
        if os.path.isdir(os.path.join(path, name, "detailed_llm_logs"))
    )


def wilson_interval(correct, total, z=1.959963985):
    """Wilson score interval for a binomial proportion."""
    if not total:
        return (0.0, 0.0)
    p = correct / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b, c):
    """Two-sided exact (binomial) McNemar p-value for discordant counts b and c.

    The exact form rather than the chi-square approximation: with b+c often under 25 here,
    the approximation is not trustworthy and the correction choice would become a
    researcher degree of freedom.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


#: Tasks whose test rows come in correlated blocks, and how to recover the block label.
#: `formula` is 50 formulas x 4 instances by construction -- verified exactly 4 per formula in
#: formula_test.jsonl and exactly 5 per formula in the 250-row train split. MBPP is absent on
#: purpose: its problems are independent, which is the documented reason nothing transfers there.
CLUSTER_SOURCES = {
    "formula": ("./eval/finance/data/task_config.json", "test_data",
                r"[Uu]se formula (.+?) to answer"),
}


def load_clusters(task_name, path_override=None):
    """Map test-row index -> block label, or None when the task has no block structure.

    Resolved from the task name rather than the run, because run_config.json does not record
    which data file was evaluated. That is fine while the splits are fixed, and --cluster_file
    exists for when they are not.
    """
    source = CLUSTER_SOURCES.get(task_name)
    if source is None and path_override is None:
        return None

    if path_override:
        data_path, pattern = path_override, (source[2] if source else None)
        if pattern is None:
            return None
    else:
        config_path, key, pattern = source
        if not os.path.exists(config_path):
            return None
        with open(config_path, encoding="utf-8") as fh:
            data_path = json.load(fh).get(task_name, {}).get(key)
        if not data_path or not os.path.exists(data_path):
            return None

    labels, expr = {}, re.compile(pattern)
    with open(data_path, encoding="utf-8") as fh:
        index = 0
        for line in fh:
            line = line.strip()
            if not line:
                continue
            match = expr.search(json.loads(line).get("context", ""))
            if match:
                labels[index] = match.group(1).strip()
            index += 1
    return labels or None


def cluster_paired_stats(shared, left_preds, right_preds, clusters, draws=10000, seed=0):
    """Paired comparison that treats the block, not the row, as the unit of resampling.

    The formula test set is 50 formulas x 4 instances, and rows inside a formula are strongly
    dependent: a playbook carrying the wrong convention for "Sharpe Ratio" gets that formula's
    instances wrong together. McNemar and the Wilson intervals both assume iid trials, so at
    200 rows they report the precision of 200 independent draws when the design has closer to
    50 -- which makes their p-values anti-conservative, i.e. real gaps are called significant
    more often than they should be.

    Two statistics, both blocked on the formula:

      * a **sign-flip permutation test**. Under the null the arm label is exchangeable, and
        because the design is paired, exchanging it flips the sign of a whole block's
        left-vs-right difference. Sampling those flips gives the null distribution of the
        overall accuracy difference without assuming anything about the within-block
        correlation.
      * a **cluster bootstrap** CI, resampling formula ids with replacement rather than rows.

    Returns None when any shared index has no block label, rather than quietly dropping rows:
    a partial clustering would understate the correlation it exists to account for.

    Scores may be booleans (one run per arm) or real numbers in [0, 1] (a row's correctness
    averaged over several replicates -- see `pooled_arm_scores`). Everything below is arithmetic
    on block sums, so the two cases run through identical code; `float(True) == 1.0` keeps the
    binary path bit-identical to the pre-pooling version, which `selftest` asserts.
    """
    blocks = defaultdict(lambda: [0, 0.0, 0.0])  # label -> [rows, left score, right score]
    for index in shared:
        label = clusters.get(index)
        if label is None:
            return None
        block = blocks[label]
        block[0] += 1
        block[1] += float(left_preds[index])
        block[2] += float(right_preds[index])

    labels = sorted(blocks)
    diffs = [blocks[label][2] - blocks[label][1] for label in labels]
    total_rows = sum(blocks[label][0] for label in labels)
    observed = sum(diffs) / total_rows

    rng = random.Random(seed)

    # Sign-flip null. The +1s are the standard Monte Carlo correction: without them a p-value
    # of exactly 0 is reportable, which no finite resampling can justify.
    at_least_as_extreme = 0
    for _ in range(draws):
        stat = sum(d if rng.random() < 0.5 else -d for d in diffs) / total_rows
        if abs(stat) >= abs(observed) - 1e-12:
            at_least_as_extreme += 1
    p_value = (at_least_as_extreme + 1) / (draws + 1)

    boot = []
    for _ in range(draws):
        picked = [blocks[rng.choice(labels)] for _ in labels]
        rows = sum(b[0] for b in picked)
        boot.append((sum(b[2] for b in picked) - sum(b[1] for b in picked)) / rows)
    boot.sort()
    lo = boot[max(0, int(0.025 * len(boot)) - 1)]
    hi = boot[min(len(boot) - 1, int(0.975 * len(boot)))]

    return {"clusters": len(labels), "rows": total_rows, "diff": observed,
            "ci": (lo, hi), "p": p_value,
            "block_sizes": sorted({blocks[label][0] for label in labels})}


# Arms in the order the experiment reads them, so pooled output does not reorder run to run.
ARM_ORDER = ("none", "text", "graph")


def _arm_sort_key(arm):
    return (ARM_ORDER.index(arm) if arm in ARM_ORDER else len(ARM_ORDER), arm)


def pooled_arm_scores(by_arm):
    """Average each row's correctness across an arm's replicates, for the seed-blocked test.

    EXPERIMENT_FORMULA.md §5 pre-registers the comparison as blocked on the formula *and*
    seed-blocked across the replicates. Blocking on the formula is what `cluster_paired_stats`
    does; this is the other half, and it exists because the two error sources behave differently:

      * **run noise** -- the same configuration re-run gives a different answer, because the
        serving stack is not bit-reproducible and the ACE loop amplifies that (§4: `none` moves
        1 test row between replicates, `text` moves 37). Averaging replicates shrinks it.
      * **formula-sampling variance** -- there are only 50 formulas, and every replicate sees the
        same 50. Averaging does nothing to it, and nothing here pretends otherwise.

    So pooling removes one of two components. The block count is unchanged -- 50 formulas before
    and after -- and the row count must never be reported as replicates x rows.

    Returns `(scores, replicates, shared, refusal)`. `scores` maps arm -> {index: mean score in
    [0, 1]}. `refusal` is a human-readable reason when pooling would be misleading, in which case
    `scores` is None; `scores` is also None with no refusal when there is simply nothing to pool.
    """
    counts = {arm: len(group) for arm, group in by_arm.items()}
    if len(set(counts.values())) != 1:
        detail = ", ".join(f"{arm}={n}" for arm, n in sorted(counts.items(), key=lambda kv: _arm_sort_key(kv[0])))
        return None, 0, [], (
            f"unequal replicate counts ({detail}). Pooling would weight the arms differently, "
            f"which is a silent bias rather than a rounding detail -- so this is refused until "
            f"every arm has the same number of runs. The per-replicate tests below still apply.")

    replicates = next(iter(counts.values()))
    if replicates < 2 or len(by_arm) < 2:
        return None, replicates, [], None  # nothing to pool; stay quiet

    per_run = [set(r["preds"]) for group in by_arm.values() for r in group]
    shared = sorted(set.intersection(*per_run))
    if not shared:
        return None, replicates, [], "no problem index is shared by every run"

    scores = {
        arm: {i: sum(float(r["preds"][i]) for r in group) / replicates for i in shared}
        for arm, group in by_arm.items()
    }
    return scores, replicates, shared, None


def report_pooled(by_arm, clusters):
    """Print the seed-blocked comparison: replicates averaged first, then blocked on formula."""
    scores, replicates, shared, refusal = pooled_arm_scores(by_arm)
    if scores is None and not refusal:
        return
    # ASCII only in printed output: on Windows stdout falls back to the ANSI codepage whenever
    # it is not a terminal, so a section sign here comes out mangled in any piped log.
    print("\n--- pooled across replicates (seed-blocked, EXPERIMENT_FORMULA.md sec 5) ---")
    if refusal:
        print(f"  skipped: {refusal}")
        return

    print(f"  each row scored as its mean over {replicates} replicates, then blocked on the "
          f"formula.\n  this averages out run-to-run noise; it does NOT add data -- the block "
          f"count is\n  unchanged and the unit of resampling is still the formula.")

    print(f"\n  {'arm':<10}{'replicates':>12}{'pooled acc':>13}")
    for arm in sorted(scores, key=_arm_sort_key):
        pooled_acc = sum(scores[arm].values()) / len(shared)
        print(f"  {arm:<10}{replicates:>12}{pooled_acc:>13.4f}")

    if not clusters:
        print("\n  no block labels for this task, so the clustered test is not run; the pooled "
              "accuracies\n  above are descriptive only.")
        return

    print("\n  McNemar is not defined on averaged scores (b and c count discordant *binary* "
          "pairs),\n  so only the clustered statistics appear here -- which is the figure sec 5 "
          "says governs anyway.")
    for left, right in itertools.combinations(sorted(scores, key=_arm_sort_key), 2):
        stats = cluster_paired_stats(shared, scores[left], scores[right], clusters)
        print(f"\n  {left}  ->  {right}")
        if stats is None:
            print("    clustered: skipped -- some test rows have no block label")
            continue
        lo, hi = stats["ci"]
        print(f"    {stats['clusters']} blocks, {stats['rows']} rows scored over {replicates} "
              f"replicates each")
        print(f"    diff={stats['diff']:+.4f}  bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"sign-flip p={stats['p']:.4f}{'  *' if stats['p'] < 0.05 else ''}")


def selftest(draws=20000):
    """Check cluster_paired_stats against the two cases where the answer is known in advance.

    Statistical code that is only ever run on real data cannot be wrong loudly. These two
    limits can:

      * singleton blocks -- every block contributes d in {-1, 0, +1}, concordant pairs
        contribute 0 and stay 0 under a flip, so the null is a sign flip of the b+c
        discordant +/-1s. That *is* McNemar's binomial, so the two p-values must agree to
        within Monte Carlo error.
      * perfectly correlated blocks of k -- duplicating every row k times adds no
        information, so the point estimate must not move and the interval must widen by
        exactly sqrt(k), the design effect at correlation 1.
    """
    failures = []
    rng = random.Random(7)

    for trial in range(5):
        n = 200
        left = {i: rng.random() < 0.60 for i in range(n)}
        right = {i: (left[i] if rng.random() < 0.75 else rng.random() < 0.55) for i in range(n)}
        shared = sorted(left)
        b = sum(1 for i in shared if not left[i] and right[i])
        c = sum(1 for i in shared if left[i] and not right[i])
        stats = cluster_paired_stats(shared, left, right, {i: f"row-{i}" for i in range(n)},
                                     draws=draws, seed=trial)
        exact = mcnemar_exact(b, c)
        ok = abs(stats["p"] - exact) < 0.02 and abs(stats["diff"] - (b - c) / n) < 1e-12
        print(f"  {'ok  ' if ok else 'FAIL'}  singleton blocks reduce to McNemar: "
              f"b={b} c={c} exact={exact:.4f} sign-flip={stats['p']:.4f}")
        if not ok:
            failures.append("singleton blocks do not reproduce McNemar")

    size = 4
    left_b, right_b, labels, flat = {}, {}, {}, {}
    index = 0
    for block in range(50):
        left_correct, right_correct = rng.random() < 0.60, rng.random() < 0.50
        for _ in range(size):
            left_b[index], right_b[index] = left_correct, right_correct
            labels[index], flat[index] = f"block-{block}", f"row-{index}"
            index += 1

    shared = sorted(left_b)
    blocked = cluster_paired_stats(shared, left_b, right_b, labels, draws=draws, seed=1)
    unblocked = cluster_paired_stats(shared, left_b, right_b, flat, draws=draws, seed=1)
    ratio = ((blocked["ci"][1] - blocked["ci"][0])
             / (unblocked["ci"][1] - unblocked["ci"][0]))
    same_estimate = abs(blocked["diff"] - unblocked["diff"]) < 1e-12
    print(f"  {'ok  ' if same_estimate else 'FAIL'}  duplicated rows do not move the estimate: "
          f"{blocked['diff']:+.4f} vs {unblocked['diff']:+.4f}")
    if not same_estimate:
        failures.append("blocking moved the point estimate")
    widened = abs(ratio - math.sqrt(size)) < 0.15
    print(f"  {'ok  ' if widened else 'FAIL'}  interval widens by the design effect: "
          f"x{ratio:.2f} (expected x{math.sqrt(size):.2f})")
    if not widened:
        failures.append(f"interval widened by {ratio:.2f}, expected {math.sqrt(size):.2f}")

    partial = dict(labels)
    partial.pop(shared[0])
    refused = cluster_paired_stats(shared, left_b, right_b, partial, draws=100) is None
    print(f"  {'ok  ' if refused else 'FAIL'}  a partial clustering is refused, not silently "
          f"dropped")
    if not refused:
        failures.append("a partial clustering was reported on")

    # --- pooling across replicates -------------------------------------------------------
    # Two more limits where the answer is known before running it.

    # (a) Pooling identical replicates must change nothing. Averaging R copies of the same run
    #     reproduces the run, so the statistics must come back bit-identical -- any drift is the
    #     pooling path corrupting its input rather than averaging it.
    single = cluster_paired_stats(shared, left_b, right_b, labels, draws=draws, seed=3)
    by_arm_same = {
        "left": [{"preds": dict(left_b)} for _ in range(3)],
        "right": [{"preds": dict(right_b)} for _ in range(3)],
    }
    pooled_same, reps_same, shared_same, refusal_same = pooled_arm_scores(by_arm_same)
    identical = False
    if pooled_same is not None and refusal_same is None and reps_same == 3:
        again = cluster_paired_stats(shared_same, pooled_same["left"], pooled_same["right"],
                                     labels, draws=draws, seed=3)
        identical = (abs(again["diff"] - single["diff"]) < 1e-12
                     and abs(again["ci"][0] - single["ci"][0]) < 1e-12
                     and abs(again["ci"][1] - single["ci"][1]) < 1e-12)
    print(f"  {'ok  ' if identical else 'FAIL'}  pooling identical replicates reproduces the "
          f"single-run result")
    if not identical:
        failures.append("pooling identical replicates moved the result")

    # (b) Pooling noisy replicates must *narrow* the interval. This is the entire property the
    #     pooling exists to buy: run noise averages down, formula-sampling variance does not.
    #     Built so both components are present -- block-level true rates vary (formula variance)
    #     and rows are Bernoulli draws around them (run noise).
    reps, block_count, size = 3, 50, 4
    truth = [(0.35 + 0.30 * rng.random(), 0.35 + 0.30 * rng.random()) for _ in range(block_count)]
    noisy_labels, replicate_runs = {}, [({}, {}) for _ in range(reps)]
    index = 0
    for block, (p_left, p_right) in enumerate(truth):
        for _ in range(size):
            noisy_labels[index] = f"block-{block}"
            for left_map, right_map in replicate_runs:
                left_map[index] = rng.random() < p_left
                right_map[index] = rng.random() < p_right
            index += 1

    noisy_shared = sorted(noisy_labels)
    widths = []
    for rep, (left_map, right_map) in enumerate(replicate_runs):
        one = cluster_paired_stats(noisy_shared, left_map, right_map, noisy_labels,
                                   draws=draws, seed=10 + rep)
        widths.append(one["ci"][1] - one["ci"][0])
    mean_single_width = statistics.mean(widths)

    by_arm_noisy = {
        "left": [{"preds": left_map} for left_map, _ in replicate_runs],
        "right": [{"preds": right_map} for _, right_map in replicate_runs],
    }
    pooled_noisy, _, pooled_shared, _ = pooled_arm_scores(by_arm_noisy)
    pooled_stats = cluster_paired_stats(pooled_shared, pooled_noisy["left"],
                                        pooled_noisy["right"], noisy_labels, draws=draws, seed=10)
    pooled_width = pooled_stats["ci"][1] - pooled_stats["ci"][0]
    narrower = pooled_width < mean_single_width
    print(f"  {'ok  ' if narrower else 'FAIL'}  pooling noisy replicates narrows the interval: "
          f"{pooled_width:.4f} vs {mean_single_width:.4f} mean single-run")
    if not narrower:
        failures.append(f"pooled interval {pooled_width:.4f} did not beat single-run "
                        f"{mean_single_width:.4f}")

    # And the guard: arms with different replicate counts must be refused, not silently pooled.
    lopsided = {"left": by_arm_noisy["left"], "right": by_arm_noisy["right"][:2]}
    _, _, _, refusal = pooled_arm_scores(lopsided)
    print(f"  {'ok  ' if refusal else 'FAIL'}  unequal replicate counts are refused, not pooled")
    if not refusal:
        failures.append("unequal replicate counts were pooled anyway")

    return failures


def load_predictions(run_dir):
    for name in PREDICTION_FILES:
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            continue
        rows = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    rows[row["index"]] = bool(row["is_correct"])
        return rows, name
    return None, None


def load_identity(run_dir):
    path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    inner = cfg.get("config", {})
    ident = {"memory_kind": cfg.get("memory_kind") or inner.get("memory"),
             "seed": cfg.get("seed", inner.get("seed")),
             "git_commit": cfg.get("git_commit")}
    for field in PINNED_FIELDS:
        ident[field] = cfg.get(field, inner.get(field))
    # Deliberately outside PINNED_FIELDS: arms selecting different checkpoints is a legitimate
    # outcome of the method, and failing the comparison on it would enforce the wrong rule.
    ident["checkpoint_choice"] = load_checkpoint_choice(run_dir)
    return ident


def load_checkpoint_choice(run_dir):
    """Which playbook the final test actually ran on, and what it beat.

    Reporting only. Arms are free to select different steps -- `ace/ace.py` applies one rule
    to all of them and a different outcome is a property of the memory, not a confound -- but
    the margin is often a handful of validation rows, so the choice is itself noisy and has to
    be visible next to the accuracies rather than dug out of train_results.json afterwards.
    """
    path = os.path.join(run_dir, "train_results.json")
    if not os.path.exists(path):
        return None  # eval_only: no training, no checkpoints to choose between
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return None

    checkpoints = []
    for entry in payload.get("results") or []:
        val = (entry or {}).get("val_result") or {}
        if entry.get("step") is not None and val.get("accuracy") is not None:
            checkpoints.append({"step": entry["step"], "val_accuracy": val["accuracy"]})
    if not checkpoints:
        return None

    # Mirrors ace/ace.py's `acc > best_accuracy`: strictly greater, so a tie keeps the
    # earlier -- and therefore smaller -- playbook. Same rule in every arm.
    best = checkpoints[0]
    for candidate in checkpoints[1:]:
        if candidate["val_accuracy"] > best["val_accuracy"]:
            best = candidate
    return {"selected_step": best["step"], "selected_val_accuracy": best["val_accuracy"],
            "checkpoints": checkpoints}


def report_checkpoint_choices(runs):
    """Print the selected checkpoint per run. Never an audit -- see load_checkpoint_choice."""
    rows = [(r, r["identity"].get("checkpoint_choice")) for r in runs]
    if not any(choice for _, choice in rows):
        return

    print("\n--- checkpoint selected for the final test ---")
    for r, choice in rows:
        if not choice:
            print(f"  {r['name'][:42]:<44} no training checkpoints (eval_only)")
            continue
        others = "  ".join(f"step {c['step']}={c['val_accuracy']:.4f}"
                           for c in choice["checkpoints"])
        print(f"  {r['name'][:42]:<44} step {choice['selected_step']} "
              f"(val {choice['selected_val_accuracy']:.4f})   of: {others}")
    print("\n  Arms may select different steps; that is the pre-registered rule applied "
          "identically,\n  not a fairness problem. Report the step, and treat a narrow "
          "validation margin as noise\n  that contributes to the across-seed sd below.")


def audit(runs, force):
    """Fail loudly when the arms were not run under the same conditions."""
    problems = []
    for field in PINNED_FIELDS:
        values = {json.dumps(r["identity"].get(field), sort_keys=True) for r in runs}
        if len(values) > 1:
            problems.append(f"{field}: {sorted(values)}")
    commits = {r["identity"].get("git_commit") for r in runs}
    if len(commits) > 1:
        problems.append(f"git_commit: {sorted(str(c) for c in commits)}")

    print("\n--- fairness audit ---")
    if not problems:
        print("  ok    all pinned settings agree across runs")
        return True
    for p in problems:
        print(f"  DIFF  {p}")
    if force:
        print("  --force given: continuing anyway. Any difference below is confounded.")
        return True
    print("\nThese runs are not a controlled comparison. Re-run them matched, or pass "
          "--force if you know why they differ.")
    return False


def extraction_paths(run_dir, indices=None):
    """Classify how each final-test answer was recovered from the model's response.

    `indices` bounds the call ids to the ones actually scored. Validation reuses the
    ``test_eval_<i>`` prefix, so a run whose validation split is larger than its test split
    leaves ids above the test size with no test call to supersede them: on formula
    (val 300, test 200) the last-per-index selection returns 300 calls, of which 100 are
    validation. Counting those mixes a different question set into the arm's totals -- and
    only into the *training* arms, since an eval_only control never runs validation, which
    is exactly the direction that would manufacture a spurious asymmetry here.

    Every repair in `extract_answer` is guarded -- it runs only after a strict parse has
    failed and cannot alter a well-formed response -- but "cannot alter a good response" is
    not the same as "cannot bias a comparison". If one arm needs the fallbacks far more than
    another, the arms were scored through different amounts of machinery and the contrast is
    confounded no matter what the p-value says. That is not hypothetical: on MBPP a single
    unhandled wrapper form hit 28 of one arm's 257 test answers and 0 in the other two,
    which was four times the between-arm difference being reported.

    A difference here is not automatically a harness bug -- the memory is upstream of how
    the model writes, so an arm really can change the output shape. It is reported rather
    than enforced for exactly that reason, and either reading needs the number.
    """
    log_dir = os.path.join(run_dir, "detailed_llm_logs")
    if not os.path.isdir(log_dir):
        return None
    calls, _ = select_final_test_calls(log_dir)
    if indices is not None:
        calls = {i: n for i, n in calls.items() if i in indices}
    counts = Counter()
    for name in calls.values():
        with open(os.path.join(log_dir, name), encoding="utf-8") as f:
            response = (json.load(f).get("response") or "").strip()
        with contextlib.redirect_stdout(io.StringIO()):
            answer = extract_answer(response)
        if answer.startswith("raise ValueError(") or answer == "No final answer found":
            counts["no answer"] += 1
            continue
        body = response
        if "```json" in body:
            body = body.split("```json")[1].split("```")[0].strip()
        start, end = body.find("{"), body.rfind("}")
        if start != -1 and end != -1:
            body = body[start:end + 1]
        try:
            json.loads(body, strict=False)
            counts["strict"] += 1
            continue
        except json.JSONDecodeError:
            pass
        try:
            json.loads(repair_json_escapes(body), strict=False)
            counts["escape repair"] += 1
        except json.JSONDecodeError:
            counts["regex fallback"] += 1
    return counts


PATH_ORDER = ("strict", "escape repair", "regex fallback", "no answer")


def report_extraction_paths(runs):
    rows = []
    for r in runs:
        counts = extraction_paths(r["dir"], set(r["preds"]))
        if counts:
            rows.append((r, counts))
    if not rows:
        return

    print("\n--- how each arm's answers were extracted ---")
    print(f"{'run':<38}{'arm':>7}" + "".join(f"{p:>16}" for p in PATH_ORDER))
    shares = {}
    for r, counts in rows:
        total = sum(counts.values())
        cells = "".join(f"{counts[p]:>9} {counts[p]/total:>5.1%}" for p in PATH_ORDER)
        print(f"{r['name'][:36]:<38}{str(r['identity'].get('memory_kind')):>7}{cells}")
        shares[r["name"]] = 1.0 - counts["strict"] / total

    spread = max(shares.values()) - min(shares.values())
    print(f"\n  share of answers NOT read by a strict parse: "
          f"{min(shares.values()):.1%} to {max(shares.values()):.1%} (spread {spread:.1%})")
    if spread > 0.05:
        print("  WARN  the arms lean on the extraction fallbacks by materially different "
              "amounts.\n        Report this alongside the accuracies: it is either a real "
              "effect of the memory\n        on output shape, or a scoring asymmetry, and "
              "the two are not distinguishable here.")
    else:
        print("  ok    the arms were scored through comparable amounts of extraction machinery")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="*", help="Run directories, or parents holding them")
    parser.add_argument("--root", action="append", default=[],
                        help="Add every run under this directory (repeatable)")
    parser.add_argument("--force", action="store_true",
                        help="Compare even when the fairness audit finds differences")
    parser.add_argument("--cluster_file",
                        help="JSONL of the evaluated test split, for tasks whose rows come in "
                             "correlated blocks. Auto-resolved for formula; use this when the "
                             "split is not the one task_config.json points at.")
    parser.add_argument("--no_cluster", action="store_true",
                        help="Report McNemar alone, even on a task with block structure")
    parser.add_argument("--selftest", action="store_true",
                        help="Check the clustered statistics against their known limiting "
                             "cases and exit. No runs needed.")
    args = parser.parse_args()

    if args.selftest:
        print("\n--- cluster_paired_stats self-test ---")
        failures = selftest()
        print()
        if failures:
            for problem in failures:
                print(f"  - {problem}")
            return 1
        print("clustered statistics behave as they must in both limits")
        return 0

    targets = []
    for path in list(args.run_dirs) + args.root:
        found = expand_run_dirs(path)
        if not found:
            print(f"no run directory under {path!r}")
        targets.extend(found)
    if len(targets) < 1:
        parser.error("nothing to compare")

    runs, missing = [], []
    for run_dir in targets:
        preds, source = load_predictions(run_dir)
        if preds is None:
            missing.append(run_dir)
            continue
        runs.append({"dir": run_dir, "name": os.path.basename(run_dir), "preds": preds,
                     "source": source, "identity": load_identity(run_dir)})

    for run_dir in missing:
        print(f"no predictions in {run_dir}; run rescore_from_logs.py on it first "
              f"(older runs saved only aggregates)")
    if not runs:
        return 2

    print(f"\n{'run':<44}{'arm':>7}{'seed':>6}{'n':>6}{'correct':>9}{'acc':>8}"
          f"{'95% CI':>18}")
    for r in runs:
        correct = sum(r["preds"].values())
        n = len(r["preds"])
        lo, hi = wilson_interval(correct, n)
        r["accuracy"] = correct / n if n else 0.0
        print(f"{r['name'][:42]:<44}{str(r['identity'].get('memory_kind')):>7}"
              f"{str(r['identity'].get('seed')):>6}{n:>6}{correct:>9}{r['accuracy']:>8.4f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}")

    ok = audit(runs, args.force)
    if not ok:
        return 2

    report_extraction_paths(runs)
    report_checkpoint_choices(runs)

    by_arm = defaultdict(list)
    for r in runs:
        by_arm[str(r["identity"].get("memory_kind"))].append(r)

    if any(len(v) > 1 for v in by_arm.values()):
        print("\n--- across seeds ---")
        print(f"{'arm':<10}{'runs':>6}{'mean':>9}{'sd':>9}{'min':>9}{'max':>9}")
        for arm, group in sorted(by_arm.items()):
            accs = [g["accuracy"] for g in group]
            sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
            print(f"{arm:<10}{len(accs):>6}{statistics.mean(accs):>9.4f}{sd:>9.4f}"
                  f"{min(accs):>9.4f}{max(accs):>9.4f}")
        print("\n  A between-arm gap smaller than the within-arm sd is seed noise, "
              "not an effect of the memory.")

    task_names = {r["identity"].get("task_name") for r in runs}
    clusters = None
    if not args.no_cluster and len(task_names) == 1:
        clusters = load_clusters(next(iter(task_names)), args.cluster_file)

    report_pooled(by_arm, clusters)

    print("\n--- paired McNemar (exact, two-sided) ---")
    print("  b = fixed by the second run, c = broken by it; p tests whether b and c differ")
    if clusters:
        print("  clustered lines below block on the test set's own structure; where the two "
              "disagree,\n  trust the clustered one -- McNemar assumes rows are independent "
              "and here they are not")
    for left, right in itertools.combinations(runs, 2):
        shared = sorted(set(left["preds"]) & set(right["preds"]))
        if not shared:
            print(f"\n  {left['name']} vs {right['name']}: no shared problem indices")
            continue
        b = sum(1 for i in shared if not left["preds"][i] and right["preds"][i])
        c = sum(1 for i in shared if left["preds"][i] and not right["preds"][i])
        both = sum(1 for i in shared if left["preds"][i] and right["preds"][i])
        neither = len(shared) - b - c - both
        p = mcnemar_exact(b, c)
        print(f"\n  {left['name'][:38]}  ->  {right['name'][:38]}")
        print(f"    n={len(shared)}  both correct={both}  both wrong={neither}")
        print(f"    b (fixed)={b}  c (broken)={c}  net={b - c:+d}  p={p:.4f}"
              f"{'  *' if p < 0.05 else ''}")
        if b + c < 10:
            print(f"    note: only {b + c} discordant pairs -- this test has almost no power")

        if clusters:
            stats = cluster_paired_stats(shared, left["preds"], right["preds"], clusters)
            if stats is None:
                print("    clustered: skipped -- some test rows have no block label")
            else:
                lo, hi = stats["ci"]
                sizes = stats["block_sizes"]
                shape = (f"{sizes[0]} rows each" if len(sizes) == 1
                         else f"{sizes[0]}-{sizes[-1]} rows each")
                print(f"    clustered by block ({stats['clusters']} blocks, {shape}):")
                print(f"      diff={stats['diff']:+.4f}  bootstrap 95% CI "
                      f"[{lo:+.4f}, {hi:+.4f}]  sign-flip p={stats['p']:.4f}"
                      f"{'  *' if stats['p'] < 0.05 else ''}")

    print("\nNo multiple-comparison correction is applied. With 3 arms there are 3 pairwise "
          "tests;\nif you report all of them, say so and treat a single p<0.05 accordingly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
