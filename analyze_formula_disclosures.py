"""Recompute the three disclosure figures in `docs/COCITE_RESULTS.md` that had no tool.

Sections 6 and 8 of that file carry three claims produced by one-off analysis. They are
disclosures and sensitivity checks rather than headline results, they get a script like everything else anyway.

    uv run python analyze_formula_disclosures.py results/formula/cocite_experiment/seed*/*/ace_run_*

**Circularity (section 8).** A pair can only be co-cited if it was co-rendered, and co-rendering
is driven by the edges, so the evidence for an edge is partly produced by the edge. The bound on
that is how much of the pair space gets a chance *without* any exploration mechanism: question
variety alone. Reported as the share of live pairs ever co-rendered.

**Are co-citation communities just duplicate families? (section 6).** If they were, the curator
could consolidate them and the graph would be storing something a de-duplication pass already
handles. Checked by comparing the embedding cosine of connected rules against the cosine of all
live pairs, and by counting how many co-cited pairs sit above the run's own `--dedup_threshold`.
Uses the same encoder the memory uses, through the same backend.

**The m=6 multiplicity sensitivity (section 3.1).** The natural family is the three comparisons
this experiment pre-registered. Section 3.1 also reports a deliberately pessimistic family that
folds in the earlier `related_to` experiment's three, where Holm breaks and Westfall-Young does
not. That table cannot come from `compare_arms.py`, which labels arms by `memory_kind`: the two
experiments' treatment arms collide under one label and would be pooled as six replicates of one
arm rather than compared as two experiments. Pass the earlier set with `--family6`. It is kept out
of `compare_arms.py` deliberately, because the grouping spans two code paths and the results file
itself calls it a worst case rather than the natural family.

**Not here: extraction asymmetry.** `compare_arms.py` already reports it ("share of answers NOT
read by a strict parse", 43.5% to 51.5% on this set), and its version is the careful one: it
bounds the population to the indices actually scored, which matters on `formula` because
validation writes 300 calls under the same `test_eval_<i>` prefix the 200 test calls use.
"""

import argparse
import collections
import itertools
import os
import random
import statistics

import ace.native_deps  # noqa: F401  -- must precede sentence-transformers, see CLAUDE.md
from ace.memory.embeddings import EmbeddingBackend
from analysis_common import (
    cocite_adjacency,
    cocite_gates,
    collect_runs,
    components,
    graph_snapshot,
    read_config,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="run directories, or parents of ace_run_*")
    parser.add_argument("--max_pairs", type=int, default=200000,
                        help="cap on all-pairs cosine sampling, for large playbooks")
    parser.add_argument("--family6", nargs="+", metavar="RUN_DIR", default=None,
                        help="the earlier related_to experiment's run dirs, for the "
                             "m=6 multiplicity sensitivity")
    args = parser.parse_args()

    runs = collect_runs(args.paths)

    backend = EmbeddingBackend()

    print("=" * 100)
    print("CIRCULARITY -- share of the live pair space ever co-rendered, with no exploration "
          "mechanism at all")
    print("  Counts a pair whose co-render weight is at least 1. That weight INCLUDES what a")
    print("  refinement inherited from the rule it superseded (graph.py::_inherit_cocitation),")
    print("  and the store keeps no marker separating the two, so this is an upper bound on the")
    print("  pairs genuinely shown together rather than a count of them.")
    print(f"  {'run':<50}{'live':>6}{'pairs':>9}{'co-rendered':>13}{'share':>8}")
    for run_dir in runs:
        snapshot = graph_snapshot(os.path.join(run_dir, "best_playbook.graph.json"))
        if not snapshot:
            continue
        live, _text, pairs, _thresholds = snapshot
        possible = len(live) * (len(live) - 1) // 2
        seen = sum(1 for _a, _b, _c, corendered in pairs if corendered >= 1)
        if not possible:
            continue
        print(f"  {os.path.basename(run_dir):<50}{len(live):>6}{possible:>9}{seen:>13}"
              f"{100.0 * seen / possible:>7.1f}%")

    print("\n" + "=" * 100)
    print("ARE CO-CITATION COMMUNITIES DUPLICATE FAMILIES? If they were, de-duplication would "
          "already handle them")
    for run_dir in runs:
        snapshot = graph_snapshot(os.path.join(run_dir, "best_playbook.graph.json"))
        if not snapshot:
            continue
        live, text, pairs, thresholds = snapshot
        config = read_config(run_dir)
        floor, gate = cocite_gates(thresholds)
        dedup_threshold = config.get("dedup_threshold", 0.90)
        edges = [(a, b) for a, neighbours in cocite_adjacency(pairs, floor, gate).items()
                 for b in neighbours if a < b]
        if not edges:
            continue

        ordered = sorted(live)
        index = {rule_id: position for position, rule_id in enumerate(ordered)}
        vectors = backend.encode([text[rule_id] for rule_id in ordered])

        def cosine(a, b):
            return float(vectors[index[a]] @ vectors[index[b]])

        edge_cosines = [cosine(a, b) for a, b in edges]
        within = [cosine(a, b)
                  for group in components(edges)
                  for a, b in itertools.combinations(sorted(group), 2)]
        all_pairs = list(itertools.combinations(ordered, 2))
        if len(all_pairs) > args.max_pairs:
            # A random sample, never `[:max_pairs]`. The ids are sorted, so a prefix is every
            # pair whose first member sorts early: a slice of one corner of the matrix rather
            # than a sample of it, and the baseline it produces is not the playbook's mean.
            all_pairs = random.Random(0).sample(all_pairs, args.max_pairs)
        baseline = [cosine(a, b) for a, b in all_pairs]
        above = sum(1 for value in edge_cosines if value >= dedup_threshold)

        print(f"  {os.path.basename(run_dir)}")
        print(f"    {len(edges)} co-citation edges over {len(live)} live rules, "
              f"{len(components(edges))} components")
        print(f"    mean cosine: edges {statistics.mean(edge_cosines):.3f}, "
              f"within-community {statistics.mean(within) if within else float('nan'):.3f}, "
              f"all live pairs {statistics.mean(baseline):.3f}")
        print(f"    co-cited pairs at or above the run's --dedup_threshold "
              f"({dedup_threshold}): {above} of {len(edges)}"
              + ("   (none: these are not duplicate families)" if above == 0 else ""))

    if args.family6:
        report_family6(runs, args.family6)


def report_family6(this_set, earlier_set):
    """Holm and Westfall-Young over both experiments' comparisons, resampled together.

    The statistics come from `compare_arms.report_multiplicity`, the same function that prints
    the m=3 block, so the two tables cannot drift apart. All this adds is the ability to build
    one family out of two experiments, which `compare_arms.py` cannot do on its own because it
    labels arms by `memory_kind` and the two treatment arms collide under one label.
    """
    from compare_arms import (
        block_diffs,
        load_clusters,
        load_identity,
        load_predictions,
        pooled_arm_scores,
        report_multiplicity,
    )

    clusters = load_clusters("formula")
    if clusters is None:
        print()
        print("  m=6: no block labels for formula, so the clustered vectors cannot be built")
        return

    families = []
    for label, group in (("this set", this_set), ("earlier set", earlier_set)):
        by_arm = collections.defaultdict(list)
        for run_dir in group:
            identity = load_identity(run_dir)
            rows, _name = load_predictions(run_dir)
            if rows:
                by_arm[str(identity.get("memory_kind"))].append({"preds": rows})
        scores, _replicates, shared, refusal = pooled_arm_scores(by_arm)
        if scores is None:
            print()
            print(f"  m=6: cannot pool the {label}: {refusal or 'nothing to pool'}")
            return
        families.append((label, scores, shared))

    comparisons = []
    for label, scores, shared in families:
        for left, right in itertools.combinations(sorted(scores), 2):
            vector, rows = block_diffs(shared, scores[left], scores[right], clusters)
            if vector and rows:
                comparisons.append((f"{left} -> {right} ({label})", vector, rows))
    if len(comparisons) < 2:
        print()
        print("  m=6: fewer than two usable comparisons")
        return

    print()
    print("=" * 100)
    print(f"MULTIPLICITY, m={len(comparisons)}: the pessimistic family spanning both experiments.")
    print("  Section 3.1 calls this a worst case rather than the natural family, because the two")
    print("  sets ran on different code paths. It is reported so the conclusion can be checked")
    print("  against a family wider than the one this experiment declared.")
    print("  Arm pairs are named in alphabetical order, so `graph -> none` is the comparison the")
    print("  results file writes as `none -> graph`. The tests are two-sided, so the p is the same.")
    report_multiplicity(comparisons)


if __name__ == "__main__":
    main()
