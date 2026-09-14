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
`_retrieved_total` and `_walk_only_total` are initialised in `__init__` (graph.py:279-281) and
never reset, so `structural_stats()["walk_only_share"]` is a running total over every retrieval
the memory has ever served. Two consequences:

  * it is **diluted downward** -- the share can only be non-zero once the playbook outgrows
    `--retrieval_budget`, and every earlier call adds rules to the denominator and nothing to
    the numerator, so a run's late behaviour is averaged against its own beginning;
  * it **excludes the final test** -- `structural_stats()` is recorded at eval checkpoints only
    (ace.py:1054, ace.py:1355), and neither site runs afterwards, so the very generations whose
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

from ace.memory.graph import DEFAULT_COCITE_BETA, DEFAULT_COCITE_K, GraphMemory
from ace.memory.retrieval import allocate, seed_groups
from analysis_common import run_dirs_under
from playbook_utils import extract_json_from_text, rendered_bullet_ids

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

    Delegates to `GraphMemory.select_for_query`, which is the code the run itself uses. This
    function used to reimplement the selection and it drifted twice: once by walking
    `related_to` on runs that only ever wrote `co_cited`, which reported 0.0000 for every
    cocite run, and again when per-sub-query seeding and the best-first walk landed. A replay
    that does not go through the run's own path measures a configuration nothing ever ran.
    """
    selected = memory.select_for_query(query)
    if selected is None:
        return set(), set()
    walked, relevance = selected
    return set(walked), set(relevance)


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




def v2_replay(graph_path, generations, config, limit=None):
    """Stage A metrics 5 to 7 of docs/EXPERIMENT_COCITE_V2.md, on a finished checkpoint.

    Each metric re-renders the run's own saved questions against the same stored memory under
    two configurations that differ in exactly the setting being measured, with everything else
    at the shipped v2 values. Every selection and every seed comes from `GraphMemory` itself,
    never from a copy of its logic here. Returns the names of blocking metrics that failed or
    could not be computed; only metric 6 is blocking.

    **Outcome weighting is off in every configuration here, and that is forced, not chosen.**
    A checkpoint written before the reward counter existed reloads with reward 0.0 on every
    pair, so a weighted `rank` predicate would build no edges at all and every walk metric would
    read as zero. These metrics measure the walk and the rendering, which weighting does not
    touch; weighting only reorders nominations, and that is metric 4's question.
    """
    # Every question is re-rendered under several configurations, each with its own budget fit,
    # so a full 441-question pass takes minutes. `--limit` samples evenly across the run rather
    # than truncating, so a spot check still spans the whole test set.
    if limit and limit < len(generations):
        stride = len(generations) / limit
        generations = [generations[int(i * stride)] for i in range(limit)]

    shipped = dict(retrieval=config.get("retrieval", "graph"),
                   retrieval_budget=config.get("retrieval_budget", 2500),
                   retrieval_seeds=config.get("retrieval_seeds", 5),
                   cocite=True, cocite_predicate="rank", cocite_beta=DEFAULT_COCITE_BETA,
                   cocite_k=DEFAULT_COCITE_K,
                   cocite_outcome_weighted=False, cocite_walk="best_first",
                   render_relations=False, subquery_seeding=True)

    def load(**change):
        return GraphMemory.load(graph_path, **{**shipped, **change})

    v2 = load()
    breadth = load(cocite_walk="breadth")
    with_lines = load(render_relations=True)

    def chosen(memory, query):
        picked = memory.select_for_query(query)
        return set() if picked is None else set(picked[0])

    calls = differ = lines_n = quiet_n = 0
    covered = multi = 0
    for question, _cited in generations:
        best = chosen(v2, question)
        wide = chosen(breadth, question)
        if not best or not wide:
            continue
        calls += 1
        differ += len(best - wide) / len(best)
        quiet_n += len(best)
        lines_n += len(chosen(with_lines, question))

        # Metric 7: every sub-question has at least one of its own seeds in what the memory
        # actually rendered. Asking only whether a seed was *allocated* is true by construction;
        # a seed can still fail to reach the generator when fit_to_budget skips a rule that does
        # not fit, and that is the case this measures.
        if not isinstance(question, str) and len([q for q in question if q]) > 1:
            multi += 1
            groups = v2.seed_groups_for_query(question)
            if groups and all(set(group) & best for group in groups):
                covered += 1

    print("  STAGE A metrics 5 to 7 (docs/EXPERIMENT_COCITE_V2.md section 9)")
    print("    shipped v2 configuration, outcome weighting forced off (see docstring)")
    failures = []
    if not calls:
        print("    [5][6][7] NOT COMPUTABLE: no generation produced a selection in both "
              "configurations -- metric 6 is blocking, so this fails")
        return ["6 relation lines"]

    print(f"    [5] best-first differs from breadth first on {100.0 * differ / calls:.1f}% "
          f"of the rendered slice   (bar >= 10%, descriptive)")
    grew = (quiet_n - lines_n) / lines_n if lines_n else None
    if grew is None:
        failures.append("6 relation lines")
        print("    [6] NOT COMPUTABLE: the configuration with relation lines rendered nothing "
              "-- blocking, so this fails")
    else:
        ok = grew >= 0.30
        if not ok:
            failures.append("6 relation lines")
        print(f"    [6] dropping relation lines renders {quiet_n / calls:.1f} rules against "
              f"{lines_n / calls:.1f}, {100.0 * grew:+.1f}%   (bar >= +30%, BLOCKING)  "
              f"{'PASS' if ok else 'FAIL'}")
    if multi:
        print(f"    [7] every sub-question has a seed in the rendered slice in "
              f"{100.0 * covered / multi:.1f}% of {multi} calls   (bar >= 95%, descriptive)")
    else:
        print("    [7] one sub-query per question on this task, so seeding is unchanged "
              "by construction")
    return failures


def text_v2_replay(playbook_path, generations, config, limit=None):
    """Stage A metrics 7 and 8 of docs/EXPERIMENT_COCITE_V2.md on a `text` arm.

    The text arm has no seeds, so a sub-question counts as represented when at least one of its
    own top `ceil(seeds / sub-questions)` rules is in the rendered slice. Metric 8 reads that
    under the global ranking the committed runs used, which is the imbalance section 2.5 repairs;
    metric 7 reads it under round-robin, the allocation the v2 text arm ships with. Both come
    from `TextPlaybookMemory.render_for_generator`, the path the arm itself runs, and neither is
    blocking. Returns no failures for that reason, but refuses to report on a task whose
    questions are a single query, where the metric is undefined.
    """
    from ace.memory.embeddings import EmbeddingBackend
    from ace.memory.text import TextPlaybookMemory

    if limit and limit < len(generations):
        stride = len(generations) / limit
        generations = [generations[int(i * stride)] for i in range(limit)]
    multi = [(question, cited) for question, cited in generations
             if not isinstance(question, str) and len([q for q in question if q]) > 1]
    print("  STAGE A metrics 7 and 8, text arm (docs/EXPERIMENT_COCITE_V2.md section 9)")
    if not multi:
        print("    [7][8] one sub-query per question on this task, so allocation cannot starve a "
              "sub-question and neither metric applies")
        return []

    with open(playbook_path, encoding="utf-8") as handle:
        playbook = handle.read()
    backend = EmbeddingBackend()
    common = dict(backend=backend, retrieval="flat",
                  retrieval_budget=config.get("retrieval_budget", 2500),
                  retrieval_seeds=config.get("retrieval_seeds", 5))
    memories = {label: TextPlaybookMemory(playbook, subquery_seeding=balanced, **common)
                for label, balanced in (("global", False), ("round-robin", True))}

    scorer = memories["round-robin"]
    covered = {label: 0 for label in memories}
    for question, _cited in multi:
        # Which rules are a sub-question's own comes from the same function that allocates the
        # graph arm's seeds, scored through the text memory's own scoring call.
        per_query = scorer._score_query(question)
        scores, _order = allocate(per_query, subquery_seeding=True)
        wanted = [set(group) for group in
                  seed_groups(per_query, scores, scorer.retrieval_seeds, subquery_seeding=True)]
        for label, memory in memories.items():
            rendered = set(rendered_bullet_ids(memory.render_for_generator(question)))
            if all(want & rendered for want in wanted):
                covered[label] += 1

    share = {label: 100.0 * count / len(multi) for label, count in covered.items()}
    print(f"    [8] under the global ranking the committed runs used, every sub-question is "
          f"represented in {share['global']:.1f}% of {len(multi)} calls   (descriptive)")
    print(f"    [7] under round-robin, the v2 allocation, in {share['round-robin']:.1f}%   "
          f"(bar >= 95%, descriptive)")
    return []


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
            ("--cocite-half-life", float, "steps; 0 disables decay"),
            ("--cocite-beta", float, "denominator offset under --cocite-predicate rank"),
            ("--cocite-k", int, "partners each rule nominates under rank")):
        parser.add_argument(flag, type=kind, default=None, help=helptext)
    parser.add_argument("--cocite-predicate", choices=("gate", "rank"), default=None,
                        help="re-score the checkpoint under the other edge predicate")
    parser.add_argument("--cocite-walk", choices=("breadth", "best_first"), default=None,
                        help="re-score the checkpoint under the other walk")
    parser.add_argument("--subquery-seeding", dest="subquery_seeding", action="store_true",
                        default=None, help="give each sub-question its own retrieval slots")
    parser.add_argument("--no-render-relations", dest="render_relations",
                        action="store_false", default=None,
                        help="re-score with relation lines suppressed")
    parser.add_argument("--limit", type=int, default=None,
                        help="sample this many generations evenly, for a faster --v2-replay")
    parser.add_argument("--v2-replay", action="store_true",
                        help="Stage A metrics 5 to 8 of docs/EXPERIMENT_COCITE_V2.md: compare "
                             "this checkpoint under the v2 configuration against the shipped "
                             "one, on the run's own saved questions. Read-only, no LLM.")
    args = parser.parse_args()

    overrides = {
        "retrieval_budget": args.retrieval_budget,
        "retrieval_seeds": args.retrieval_seeds,
        "cocite_min_rate": args.cocite_min_rate,
        "cocite_min_opportunity": args.cocite_min_opportunity,
        "cocite_half_life": args.cocite_half_life,
        "cocite_beta": args.cocite_beta,
        "cocite_k": args.cocite_k,
        "cocite_predicate": args.cocite_predicate,
        "cocite_walk": args.cocite_walk,
        "subquery_seeding": args.subquery_seeding,
        "render_relations": args.render_relations,
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

    blocking_failures = []
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
            if args.v2_replay and config.get("memory_kind") == "text":
                generations = test_generations(run_dir, transform=query_transform(config))
                for name in text_v2_replay(os.path.join(run_dir, "best_playbook.txt"),
                                           generations, config, args.limit):
                    blocking_failures.append((label, name))
                continue
            print("  not a graph arm -- walk_only_share is undefined here, see the table above")
            continue

        generations = test_generations(run_dir, transform=query_transform(config))
        if not generations:
            print("  no generator_test_eval_* logs with a recognisable question")
            continue

        graph_path = os.path.join(run_dir, "best_playbook.txt")
        if args.v2_replay:
            for name in v2_replay(graph_path, generations, config, args.limit):
                blocking_failures.append((label, name))
            continue
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

    if blocking_failures:
        print()
        print(f"{len(blocking_failures)} BLOCKING Stage A metric(s) failed or could not be "
              f"computed. Under section 9 no run is authorised:")
        for label, name in blocking_failures:
            print(f"  {label}: metric {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
