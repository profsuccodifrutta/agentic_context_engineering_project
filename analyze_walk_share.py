#!/usr/bin/env python3
"""Recompute `walk_only_share` for the generations that actually produced a run's accuracy.

`walk_only_share` answers the falsification question in `EXPERIMENT_FORMULA.md` §7 and
`EXPERIMENT_MBPP.md` §8: **is the graph doing anything, or is it an annotation?** It is a
counterfactual *inside the graph arm* -- of the rules this arm rendered, how many are present
because the walk followed `related_to`, and would not have been selected by relevance ranking
alone under the same budget. It is not a comparison against the text arm: the two arms diverge
from step 1 and hold different rules under different ids, so no cross-arm set difference is
even defined.

**The number recorded during a run is not that quantity.** `GraphMemory._retrieval_calls`,
`_retrieved_total` and `_walk_only_total` are initialised in `__init__` (graph.py:155-157) and
never reset, so `structural_stats()["walk_only_share"]` is a running total over every retrieval
the memory has ever served. Two consequences:

  * it is **diluted downward** -- the share can only be non-zero once the playbook outgrows
    `--retrieval_budget`, and every earlier call adds rules to the denominator and nothing to
    the numerator, so a run's late behaviour is averaged against its own beginning;
  * it **excludes the final test** -- `structural_stats()` is recorded at eval checkpoints only
    (ace.py:987, ace.py:1288), and neither site runs afterwards, so the very generations whose
    accuracy gets reported contribute nothing to the diagnostic that is supposed to license
    attributing that accuracy to structure.

The bias is one-directional and therefore safe for a floor: a run that clears the threshold on
the diluted figure clears it on the real one too. It is still the wrong quantity to calibrate a
threshold against, which is what this script fixes.

    uv run python analyze_walk_share.py results/formula/preregistered/seed0/graph
    uv run python analyze_walk_share.py results/mbpp/probes_v2/graph --trajectory

Reads the saved graph sidecar and the generator logs. No LLM, no network, and it never writes
to the run directory. It reuses `GraphMemory`'s own selection path rather than reimplementing
it, so it cannot drift away from what the run did.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

from ace.memory.graph import CO_CITED, RELATED_TO, GraphMemory
from ace.memory.retrieval import fit_to_budget, flat_order, graph_order, score_against_query

#: Where the question sits in each prompt set's generator template. The raw `question` string
#: is what ACE passes as the retrieval query (`ace/ace.py:230`), and the template embeds it
#: verbatim, so slicing between the headings recovers it exactly.
QUESTION_MARKERS = (
    ("QUESTION (PROMPT AND TESTS)\n", "\n\nCONTEXT (SETUP CODE)"),   # code prompts
    ("**Question:**\n", "\n\n**Context:**"),                          # finance prompts
)


def extract_question(prompt):
    for opener, closer in QUESTION_MARKERS:
        start = prompt.find(opener)
        if start < 0:
            continue
        start += len(opener)
        end = prompt.find(closer, start)
        if end > start:
            return prompt[start:end]
    return None


def test_questions(run_dir):
    """The distinct test-set queries, one per problem, in index order.

    A run that evaluates the test set more than once writes several
    `generator_test_eval_<index>_*` files per problem. De-duplicating by index matters: leaving
    the repeats in would weight whichever problems were re-evaluated more heavily than the rest,
    and the share would stop describing the test set.
    """
    by_index = {}
    paths = glob.glob(os.path.join(run_dir, "detailed_llm_logs", "generator_test_eval_*.json"))
    for path in paths:
        match = re.match(r"generator_test_eval_(\d+)_", os.path.basename(path))
        if not match or match.group(1) in by_index:
            continue
        with open(path, encoding="utf-8") as fh:
            question = extract_question(json.load(fh).get("prompt", ""))
        if question:
            by_index[match.group(1)] = question
    return [by_index[key] for key in sorted(by_index, key=int)]


def selection(memory, query):
    """(graph-ordered selection, relevance-ordered selection) for one query.

    Mirrors `GraphMemory.render_for_generator` (graph.py:173-208) by calling the same helpers on
    the same object, so the two cannot diverge.
    """
    active = memory.active_nodes()
    if not active:
        return set(), set()

    scores = score_against_query(
        query, [(n, memory.graph.nodes[n]["content"]) for n in active],
        memory.backend, memory._embeddings)

    def select(order):
        return fit_to_budget(
            order,
            memory._render_cost,
            lambda keep: memory._render(with_legend=True, only=set(keep)),
            memory.retrieval_budget,
        )

    live = set(active)
    # Which relation the walk follows is a property of the run, not of this script: under
    # --cocite no similarity edges are written at all, so hardcoding RELATED_TO here walks an
    # empty edge set and reports 0.0000 for every cocite run. Mirrors graph.py:228.
    association = CO_CITED if memory.cocite else RELATED_TO
    walked = select(graph_order(
        scores,
        lambda node_id: [n for n in memory._neighbours(node_id, association) if n in live],
        memory.retrieval_seeds,
    ))
    return set(walked), set(select(flat_order(scores)))


def measure(graph_path, questions, config):
    """Pooled and per-call walk-only share over `questions` at one checkpoint."""
    memory = GraphMemory.load(
        graph_path,
        retrieval=config.get("retrieval", "graph"),
        retrieval_budget=config.get("retrieval_budget", 2500),
        retrieval_seeds=config.get("retrieval_seeds", 5),
    )

    rendered_total = walk_only_total = 0
    per_call = []
    for question in questions:
        walked, flat = selection(memory, question)
        if not walked:
            continue
        walk_only = len(walked - flat)
        rendered_total += len(walked)
        walk_only_total += walk_only
        per_call.append(walk_only / len(walked))

    return {
        "calls": len(per_call),
        "active_rules": len(memory.active_nodes()),
        "rendered_mean": rendered_total / len(per_call) if per_call else 0.0,
        # Pooled: share of everything read. This is the definition the run uses.
        "pooled": walk_only_total / rendered_total if rendered_total else 0.0,
        # Macro: mean of the per-generation shares. Reported alongside because "the proportion
        # of rules the graph adds" reads as a per-generation quantity, and the two only
        # coincide when every call renders the same number of rules.
        "per_call_mean": statistics.mean(per_call) if per_call else 0.0,
        "per_call_zero": sum(1 for share in per_call if share == 0.0),
    }


def run_dirs_under(path):
    if os.path.isdir(os.path.join(path, "detailed_llm_logs")):
        return [path]
    return sorted(glob.glob(os.path.join(path, "ace_run_*")))


def recorded_share(run_dir):
    """What the run itself recorded, for comparison. Cumulative, see the module docstring."""
    try:
        with open(os.path.join(run_dir, "train_results.json"), encoding="utf-8") as fh:
            results = json.load(fh)["results"]
    except (OSError, KeyError):
        return None
    stats = [r["structural_stats"] for r in results if r.get("structural_stats")]
    return stats[-1]["walk_only_share"] if stats else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", help="graph-arm run directories")
    parser.add_argument("--trajectory", action="store_true",
                        help="also measure every intermediate_playbooks/ checkpoint")
    args = parser.parse_args()

    resolved = []
    for path in args.run_dirs:
        found = run_dirs_under(path)
        if not found:
            print(f"no run with detailed_llm_logs/ under {path}")
            return 2
        resolved.extend(found)

    for run_dir in resolved:
        with open(os.path.join(run_dir, "run_config.json"), encoding="utf-8") as fh:
            config = json.load(fh)
        label = os.path.basename(os.path.normpath(run_dir))
        print("=" * 96)
        print(f"{label}   memory={config.get('memory_kind')} retrieval={config.get('retrieval')} "
              f"budget={config.get('retrieval_budget')} seeds={config.get('retrieval_seeds')}")

        if config.get("memory_kind") != "graph":
            print("  not a graph arm -- walk_only_share is undefined here")
            continue

        questions = test_questions(run_dir)
        if not questions:
            print("  no generator_test_eval_* logs with a recognisable question")
            continue

        graph_path = os.path.join(run_dir, "best_playbook.txt")
        result = measure(graph_path, questions, config)
        recorded = recorded_share(run_dir)

        print(f"  FINAL TEST   {result['calls']} generations, {result['active_rules']} live rules, "
              f"{result['rendered_mean']:.1f} rendered per call")
        print(f"    walk_only_share (pooled)        {result['pooled']:.4f}")
        print(f"    walk_only_share (per-call mean) {result['per_call_mean']:.4f}")
        print(f"    generations where the walk added nothing: "
              f"{result['per_call_zero']}/{result['calls']}")
        if recorded is not None:
            print(f"    recorded during the run (cumulative over training, excludes this test): "
                  f"{recorded:.4f}")

        if args.trajectory:
            print("  TRAJECTORY (same test questions, earlier checkpoints)")
            checkpoints = sorted(
                glob.glob(os.path.join(run_dir, "intermediate_playbooks", "*_playbook.txt")),
                key=lambda p: int(m.group(1)) if (m := re.search(r"step_(\d+)", p)) else 10**9)
            for checkpoint in checkpoints:
                point = measure(checkpoint, questions, config)
                print(f"    {os.path.basename(checkpoint):<44} "
                      f"{point['active_rules']:>4} live  {point['rendered_mean']:>5.1f} rendered  "
                      f"pooled {point['pooled']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
