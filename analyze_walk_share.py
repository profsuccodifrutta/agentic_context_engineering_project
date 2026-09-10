#!/usr/bin/env python3
"""Recompute `walk_only_share` for the generations that actually produced a run's accuracy.

`walk_only_share` answers the falsification question in `docs/background/EXPERIMENT_FORMULA.md` §7 and
`docs/background/EXPERIMENT_MBPP.md` §8: **is the graph doing anything, or is it an annotation?** It is a
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
from analysis_common import run_dirs_under
from playbook_utils import extract_json_from_text

#: Where the question sits in each prompt set's generator template. The template embeds the
#: question verbatim, so slicing between the headings recovers it exactly. **Recovering it is
#: only half the job**: since 2026-08-30 ACE scores rules against `ACE._retrieval_query`, which
#: is the question for most tasks but, on `finer`, a list of the row's sub-questions with the
#: question template stripped from each (2026-09-03). `query_transform` below re-applies the same
#: transform before anything is scored.
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


def query_transform(config):
    """The same query transform the run applied before scoring rules, keyed off the task.

    **This must mirror `ACE._retrieval_query`, or the share is measured with a query the run
    never used.** `extract_question` recovers the question from the saved prompt; on `finer` that
    question carries two substrings shared by every row -- the 140-tag taxonomy that fills the
    encoder's window (2026-08-30) and the `What is best tag for entity` template that decides
    which rules win the ranking (2026-09-03). Scoring the recovered question here would reproduce
    both defects inside the measurement even on a run that had neither, and `walk_only_share`
    would describe a selection the generator never saw.

    On `finer` the transform returns a **list**, one entry per sub-question, which
    `score_against_query` scores as each rule's best match against any entry. A task that needs
    no transform gets the identity, which is what this script did before.
    """
    if (config.get("task_name") or "").lower() == "finer":
        from eval.finance.data_processor import finer_retrieval_query
        return finer_retrieval_query
    return lambda question: question


def cited_ids(response):
    """The rule ids the generator said it used, or an empty set when the response is unreadable.

    `extract_json_from_text` already retries once with unrecognised backslash escapes doubled,
    which matters here: the formula generator writes its working in LaTeX inside a string field
    and 45% of responses fail a strict parse. An unreadable response contributes nothing rather
    than counting as "cited nothing", so it cannot depress the rate by itself -- the caller
    intersects with the rendered set, and an empty set drops out of both numerator and
    denominator only if the generation is skipped, which is what `measure` does.
    """
    parsed = extract_json_from_text(response or "")
    if not isinstance(parsed, dict):
        return None
    ids = parsed.get("bullet_ids")
    return set(ids) if isinstance(ids, list) else None


def test_generations(run_dir, transform=None):
    """The distinct test-set generations as (query, cited ids), one per problem, in index order.

    A run that evaluates the test set more than once writes several
    `generator_test_eval_<index>_*` files per problem. De-duplicating by index matters: leaving
    the repeats in would weight whichever problems were re-evaluated more heavily than the rest,
    and the share would stop describing the test set.

    **The last file per index wins, not the first.** Validation evaluations write under this same
    prefix, so on an offline run with `--eval_steps` set, the low indices carry one file per
    validation checkpoint ahead of the final-test one. Keeping the first took the earliest
    validation pass for every index below the validation set size and the final test for the rest,
    which silently mixed two evaluations into one "final test" figure. Filenames carry
    `YYYYMMDD_HHMMSS_mmm`, so sorting is chronological and the final test is always last.
    """
    by_index = {}
    paths = sorted(glob.glob(os.path.join(run_dir, "detailed_llm_logs",
                                          "generator_test_eval_*.json")))
    for path in paths:
        match = re.match(r"generator_test_eval_(\d+)_", os.path.basename(path))
        if not match:
            continue
        with open(path, encoding="utf-8") as fh:
            call = json.load(fh)
        question = extract_question(call.get("prompt", ""))
        if question:
            query = transform(question) if transform else question
            by_index[match.group(1)] = (query, cited_ids(call.get("response", "")))
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



def rendered_report(run_dir):
    """Distinct rendered rule sets, distinct rules ever rendered, and rules per call.

    Recovered from the saved final-test prompts through `check_learning_signal`'s reader and id
    pattern, so validation generations cannot leak in and an invented rule id is not dropped.
    Works for every retrieval arm, which the walk replay below cannot: it needs a graph.
    """
    from check_learning_signal import _RENDERED_RULE, _test_generation_prompts

    sets, union, per_call = [], set(), []
    for _index, prompt in _test_generation_prompts(run_dir):
        ids = set(_RENDERED_RULE.findall(prompt))
        if not ids:
            continue
        sets.append(frozenset(ids))
        union |= ids
        per_call.append(len(ids))
    if not sets:
        print("  RENDERED  no test prompt carries a readable rendered playbook")
        return
    print(f"  RENDERED  over {len(sets)} test generations with a readable playbook")
    print(f"    rules rendered per call         {statistics.mean(per_call):.1f}")
    print(f"    distinct rendered rule sets     {len(set(sets))} of {len(sets)}"
          f"   (one set for all of them is the 2026-08-30 defect)")
    print(f"    distinct rules ever rendered    {len(union)}   (union over questions, not per call)")


def measure(graph_path, generations, config, overrides=None):
    """Walk-only share, and whether the rules the walk added were then used.

    The share says how much of what the generator reads is there because of the edges. It is a
    necessary condition and never a sufficient one: a random rewiring at matched density scores
    2.3x to 3.6x higher than the real similarity graph, so the share rewards pointing *away*
    from relevance and cannot tell a useful neighbour from a scrambled one.

    The citation rates are the missing half. `walk_only_cite_rate` is the share of walk-added
    rules the generator went on to name in `bullet_ids`; `relevance_cite_rate` is the same for
    the rules relevance would have selected anyway. Their ratio is the number to read: near 1
    means the walk surfaces rules as useful as relevance does, and near 0 means it is spending
    a shared token budget on rules the generator ignores.
    """
    settings = {
        "retrieval": config.get("retrieval", "graph"),
        "retrieval_budget": config.get("retrieval_budget", 2500),
        "retrieval_seeds": config.get("retrieval_seeds", 5),
    }
    # An explicit keyword beats the thresholds saved with the graph (graph.py from_dict), so the
    # co-citation gate can be re-scored offline against a finished checkpoint with no re-run.
    settings.update({key: value for key, value in (overrides or {}).items() if value is not None})
    memory = GraphMemory.load(graph_path, **settings)

    rendered_total = walk_only_total = 0
    per_call = []
    # Citation accounting runs only over generations whose response parsed, so an unreadable
    # response cannot masquerade as "cited nothing" and depress the rate.
    walk_only_shown = walk_only_cited = relevance_shown = relevance_cited = 0
    scored_calls = 0
    # Section 5 of the results file quotes four set statistics that only exist here, because
    # only this path knows what retrieval actually selected per question. Counting them beside
    # the share costs nothing and means the table has a tool behind it.
    rendered_sets, rendered_union = set(), set()
    walk_added_sets, walk_added_union = set(), set()
    for question, cited in generations:
        walked, flat = selection(memory, question)
        if not walked:
            continue
        walk_only = walked - flat
        rendered_total += len(walked)
        walk_only_total += len(walk_only)
        per_call.append(len(walk_only) / len(walked))
        rendered_sets.add(frozenset(walked))
        rendered_union |= walked
        walk_added_sets.add(frozenset(walk_only))
        walk_added_union |= walk_only
        if cited is None:
            continue
        scored_calls += 1
        relevance = walked & flat
        walk_only_shown += len(walk_only)
        walk_only_cited += len(walk_only & cited)
        relevance_shown += len(relevance)
        relevance_cited += len(relevance & cited)

    walk_rate = walk_only_cited / walk_only_shown if walk_only_shown else 0.0
    relevance_rate = relevance_cited / relevance_shown if relevance_shown else 0.0
    return {
        "calls": len(per_call),
        "active_rules": len(memory.active_nodes()),
        "rendered_mean": rendered_total / len(per_call) if per_call else 0.0,
        "scored_calls": scored_calls,
        "walk_only_shown": walk_only_shown,
        "walk_only_cited": walk_only_cited,
        "relevance_shown": relevance_shown,
        "relevance_cited": relevance_cited,
        "walk_only_cite_rate": walk_rate,
        "relevance_cite_rate": relevance_rate,
        "cite_rate_ratio": walk_rate / relevance_rate if relevance_rate else 0.0,
        # Pooled: share of everything read. This is the definition the run uses.
        "pooled": walk_only_total / rendered_total if rendered_total else 0.0,
        # Macro: mean of the per-generation shares. Reported alongside because "the proportion
        # of rules the graph adds" reads as a per-generation quantity, and the two only
        # coincide when every call renders the same number of rules.
        "per_call_mean": statistics.mean(per_call) if per_call else 0.0,
        "per_call_zero": sum(1 for share in per_call if share == 0.0),
        # Distinct rendered sets is the section 7 validity gate at full length: the defect-
        # invalidated 2026-08-30 arms rendered ONE set across all 441 generations.
        "rendered_sets": len(rendered_sets),
        "rendered_union": len(rendered_union),
        # Distinct walk-added sets separates "query-dependent walk" from "a fixed extra block of
        # text on every call", which is the degeneration the growing-share gate guards against.
        "walk_added_sets": len(walk_added_sets),
        "walk_added_union": len(walk_added_union),
    }




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
    parser.add_argument("--checkpoint", metavar="STEP",
                        help="measure intermediate_playbooks/*step_<STEP>_playbook.txt instead of "
                             "best_playbook.txt")
    # Overrides re-score a finished checkpoint under different settings. They change nothing about
    # the run; unset, every one of them is inert and the output reproduces the run's own numbers.
    for flag, kind, helptext in (
            ("--retrieval-budget", int, "token budget for the whole rendering"),
            ("--retrieval-seeds", int, "number of relevance seeds the walk starts from"),
            ("--cocite-min-rate", float, "w_cocited / w_corendered gate"),
            ("--cocite-min-opportunity", float, "minimum w_corendered for a pair to be eligible"),
            ("--cocite-half-life", float, "steps; 0 disables decay")):
        parser.add_argument(flag, type=kind, default=None, help=helptext)
    args = parser.parse_args()

    overrides = {
        "retrieval_budget": args.retrieval_budget,
        "retrieval_seeds": args.retrieval_seeds,
        "cocite_min_rate": args.cocite_min_rate,
        "cocite_min_opportunity": args.cocite_min_opportunity,
        "cocite_half_life": args.cocite_half_life,
    }
    active = {k: v for k, v in overrides.items() if v is not None}
    if active:
        print(f"OVERRIDES {active}  (re-scoring a finished checkpoint, no run is affected)")

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

        # What retrieval put in front of the model is readable straight off the saved prompts,
        # for any arm, with no replay and no memory object. That is the whole of section 5's
        # retrieval table, and reading it here is what lets a `text` arm have a column at all:
        # it has no walk, but it does have a rendered slice per question.
        rendered_report(run_dir)

        if config.get("memory_kind") != "graph":
            print("  not a graph arm -- walk_only_share is undefined here, see the table above")
            continue

        generations = test_generations(run_dir, transform=query_transform(config))
        if not generations:
            print("  no generator_test_eval_* logs with a recognisable question")
            continue

        graph_path = os.path.join(run_dir, "best_playbook.txt")
        if args.checkpoint:
            matches = glob.glob(os.path.join(run_dir, "intermediate_playbooks",
                                             f"*step_{args.checkpoint}_playbook.txt"))
            if not matches:
                print(f"  no intermediate checkpoint at step {args.checkpoint}")
                continue
            graph_path = matches[0]
            print(f"  checkpoint {os.path.basename(graph_path)}")
        result = measure(graph_path, generations, config, overrides)
        recorded = recorded_share(run_dir)

        print(f"  FINAL TEST   {result['calls']} generations, {result['active_rules']} live rules, "
              f"{result['rendered_mean']:.1f} rendered per call")
        print(f"    rules rendered per call         {result['rendered_mean']:.1f} "
              f"of {result['active_rules']} live")
        print(f"    distinct rendered rule sets     {result['rendered_sets']} of {result['calls']}"
              f"   (a constant set is the 2026-08-30 defect)")
        print(f"    distinct rules ever rendered    {result['rendered_union']} of "
              f"{result['active_rules']} "
              f"({100.0 * result['rendered_union'] / result['active_rules']:.1f}% of the memory, "
              f"as a union over questions)")
        print(f"    distinct walk-added rule sets   {result['walk_added_sets']} of {result['calls']}")
        print(f"    rules the walk ever added       {result['walk_added_union']} of "
              f"{result['active_rules']} "
              f"({100.0 * result['walk_added_union'] / result['active_rules']:.1f}%)")
        print(f"    walk_only_share (pooled)        {result['pooled']:.4f}")
        print(f"    walk_only_share (per-call mean) {result['per_call_mean']:.4f}")
        print(f"    generations where the walk added nothing: "
              f"{result['per_call_zero']}/{result['calls']}")
        if result["scored_calls"]:
            print(f"    were the added rules used? ({result['scored_calls']} responses parsed)")
            print(f"      walk-added rules cited      {result['walk_only_cited']:>5}/"
                  f"{result['walk_only_shown']:<6} {result['walk_only_cite_rate']:.4f}")
            print(f"      relevance rules cited       {result['relevance_cited']:>5}/"
                  f"{result['relevance_shown']:<6} {result['relevance_cite_rate']:.4f}")
            print(f"      ratio                       {result['cite_rate_ratio']:.3f}"
                  "   (near 1: the walk is as useful as relevance; near 0: it renders noise)")
        if recorded is not None:
            print(f"    recorded during the run (cumulative over training, excludes this test): "
                  f"{recorded:.4f}")

        if args.trajectory:
            print("  TRAJECTORY (same test questions, earlier checkpoints)")
            checkpoints = sorted(
                glob.glob(os.path.join(run_dir, "intermediate_playbooks", "*_playbook.txt")),
                key=lambda p: int(m.group(1)) if (m := re.search(r"step_(\d+)", p)) else 10**9)
            for checkpoint in checkpoints:
                point = measure(checkpoint, generations, config, overrides)
                print(f"    {os.path.basename(checkpoint):<44} "
                      f"{point['active_rules']:>4} live  {point['rendered_mean']:>5.1f} rendered  "
                      f"pooled {point['pooled']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
