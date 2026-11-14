"""Recompute the co-citation structure figures in `docs/FINER_RESULTS.md` section 5, offline.

Reads the `.graph.json` sidecars a graph run writes at every checkpoint, so it needs no LLM, no
network and no retrieval replay. What the *walk* did with these edges is a different question and
belongs to `analyze_walk_share.py`, which replays retrieval against the same checkpoints.

    uv run python analyze_cocite_structure.py results/finer/final/seed0/cocite/ace_run_*
    uv run python analyze_cocite_structure.py --selftest

Three things it reports by default, each because a results document quotes it:

**The trajectory.** Section 7 of the pre-registration gates on the co-citation component *growing
with the playbook*. That gate is read here: live rules, edges, and the connected set at each
checkpoint.

**Membership survival.** How much of the connected set persists from one reported checkpoint to
the next, at the section 5 intervals rather than at every checkpoint the run wrote. A small core
that keeps its members is a fixed block; a small core that turns over is not, and the distinction
is what section 5 turns on.

**The pre-registered fallback.** When a structural gate fails, `EXPERIMENT_FINER.md` section 9
says to pick the rate from {0.05, 0.10, 0.15, 0.20} maximising connected nodes subject to
`cocite_max_degree <= 0.25 * connected_nodes`. Only structural quantities are read, never
accuracy. The table is printed whether or not a candidate clears, because "no rate was selected"
is itself the pre-registered outcome and has to be on the record.

`--extra` adds three diagnostics no results document quotes: why members leave the connected set,
the relation-line token bill at each candidate rate, and citation concentration. They belong to
the design work in `TODOS.md`, not to a reported result, so they are off by default.
"""

import argparse
import collections
import glob
import json
import os
import random
import re
import statistics

from analysis_common import (
    cocite_adjacency,
    cocite_gates,
    components,
    graph_snapshot,
    run_dirs_under,
)

FALLBACK_RATES = (0.05, 0.10, 0.15, 0.20)
FALLBACK_DEGREE_SHARE = 0.25
#: A run writes a checkpoint every 10 steps. Quoting all of them is noise, so the default is
#: every 50th plus the last one the run wrote. That is dynamic on purpose: hardcoding
#: `(50, 100, 150, 200)` reported `finer` correctly and silently dropped `formula`'s step 250.
REPORTED_STRIDE = 50
#: A rendered id costs roughly this many tokens once the ", " separator is included. Approximate
#: by design: the point is the order of magnitude against a 2500-token budget, and `tiktoken`
#: undercounts this server by ~10% anyway.
TOKENS_PER_RENDERED_ID = 5.5
CHECKPOINT = re.compile(r"epoch_\d+_step_(\d+)_playbook\.graph\.json$")


def reported_steps(available):
    """Which checkpoints to print: every `REPORTED_STRIDE`th, plus the last one written.

    `available` is the step numbers a run actually has, so a 250-step run reports 250 and a
    200-step run reports 200 without either being named in the source.
    """
    steps = sorted(step for step in available if step is not None)
    if not steps:
        return []
    chosen = {step for step in steps if step % REPORTED_STRIDE == 0}
    chosen.add(steps[-1])
    return sorted(chosen)


def degrees(graph):
    return sorted((len(v) for v in graph.values()), reverse=True)


def checkpoints(run_dir):
    found = []
    for path in glob.glob(os.path.join(run_dir, "intermediate_playbooks", "*_playbook.graph.json")):
        match = CHECKPOINT.search(os.path.basename(path))
        if match:
            found.append((int(match.group(1)), path))
    found.sort()
    final = os.path.join(run_dir, "best_playbook.graph.json")
    if os.path.isfile(final):
        found.append((None, final))
    return found


def citation_concentration(run_dir):
    """Distinct rules ever cited during training, and the share the top ten carry."""
    path = os.path.join(run_dir, "bullet_usage_log.jsonl")
    if not os.path.isfile(path):
        return None
    counts = collections.Counter()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for rule_id in row.get("bullet_ids_used") or row.get("bullet_ids") or []:
                counts[rule_id] += 1
    if not counts:
        return None
    total = sum(counts.values())
    return len(counts), total, sum(c for _i, c in counts.most_common(10)) / total


def relation_line_cost(live, graph, rendered, trials, rng):
    """Mean neighbour ids listed when `rendered` rules are shown, over random draws."""
    ordered = sorted(live)
    if len(ordered) <= rendered:
        return sum(len(graph.get(r, ())) for r in ordered)
    return statistics.mean(
        sum(len(graph.get(r, ())) for r in rng.sample(ordered, rendered))
        for _ in range(trials)
    )


def selftest():
    """Check the union-find and the gate against cases whose answer is known in advance."""
    assert components([]) == [], "no edges is no components"
    assert components([("a", "b")]) == [{"a", "b"}], "one edge is one pair"

    # A chain must merge into one component however the edges are ordered, which is the property
    # a naive "group by first endpoint" gets wrong and union-find exists for.
    chain = [("a", "b"), ("c", "d"), ("b", "c")]
    for order in ([0, 1, 2], [2, 1, 0], [1, 0, 2], [1, 2, 0]):
        found = components([chain[i] for i in order])
        assert found == [{"a", "b", "c", "d"}], f"chain must merge, got {found} for {order}"

    two = components([("a", "b"), ("c", "d")])
    assert sorted(map(sorted, two)) == [["a", "b"], ["c", "d"]], "disjoint edges stay disjoint"

    # A star and a triangle: same node count, different shape, both one component.
    assert components([("h", "x"), ("h", "y"), ("h", "z")]) == [{"h", "x", "y", "z"}]
    assert components([("p", "q"), ("q", "r"), ("r", "p")]) == [{"p", "q", "r"}]

    # Self-loops and repeats must not create phantom members.
    assert components([("a", "b"), ("a", "b"), ("a", "a")]) == [{"a", "b"}]

    # The gate: a pair under the opportunity floor is not an edge however high its rate, and a
    # pair over the floor is an edge only at or above the rate.
    pairs = [
        ("a", "b", 3.0, 6.0),    # rate 0.50 over a co-render count of 6: an edge
        ("c", "d", 2.0, 2.0),    # rate 1.00 but only 2 opportunities: below the floor
        ("e", "f", 1.0, 20.0),   # 20 opportunities but rate 0.05: below the gate
    ]
    graph = cocite_adjacency(pairs, 3, 0.10)
    assert set(graph) == {"a", "b"}, f"opportunity floor and rate gate, got {sorted(graph)}"
    assert not cocite_adjacency(pairs, 3, 0.99), "no pair here clears a rate of 0.99"
    assert degrees(cocite_adjacency([("a", "b", 3.0, 3.0), ("a", "c", 3.0, 3.0)], 3, 0.1)) == [2, 1, 1]

    print("selftest: all checks passed")


def report(run_dir, args, rng):
    marks = checkpoints(run_dir)
    if not marks:
        print(f"{run_dir}: no .graph.json sidecar, not a graph arm")
        return

    _live, _text, _pairs, thresholds = graph_snapshot(marks[-1][1])
    floor, gate = cocite_gates(thresholds, args.min_opportunity, args.min_rate)

    print("=" * 96)
    print(f"{run_dir}")
    print(f"  gate as read: co-rendered >= {floor:g} and co-cited/co-rendered >= {gate:g}"
          + ("   (OVERRIDDEN)" if (args.min_opportunity or args.min_rate) else ""))

    print("\n  TRAJECTORY -- does the connected set grow with the playbook?")
    print(f"    {'step':<8}{'live':>7}{'edges':>8}{'connected':>11}{'share':>8}"
          f"{'max deg':>9}{'median deg':>12}")
    snapshots = {}
    wanted = set(reported_steps(step for step, _p in marks))
    for step, path in marks:
        live, _text, pairs, _thr = graph_snapshot(path)
        graph = cocite_adjacency(pairs, floor, gate)
        deg = degrees(graph)
        snapshots[step] = (live, pairs, set(graph))
        if not (args.all_steps or step is None or step in wanted):
            continue
        label = "final" if step is None else str(step)
        print(f"    {label:<8}{len(live):>7}{sum(deg) // 2:>8}{len(graph):>11}"
              f"{100.0 * len(graph) / len(live):>7.1f}%"
              f"{(deg[0] if deg else 0):>9}{(statistics.median(deg) if deg else 0):>12.0f}")

    print("\n  MEMBERSHIP SURVIVAL -- is the small core a fixed block, or does it turn over?")
    print(f"    {'interval':<20}{'kept':>8}{'of':>5}{'joined':>9}")
    reported = [step for step in reported_steps(snapshots) if step in snapshots]
    for before, after in zip(reported, reported[1:]):
        was = snapshots[before][2]
        now = snapshots[after][2]
        print(f"    {f'step {before} -> {after}':<20}{len(was & now):>8}{len(was):>5}"
              f"{len(now - was):>9}")

    live, _tested_text, pairs, _tested_thresholds = graph_snapshot(marks[-1][1])
    print(f"\n  PRE-REGISTERED FALLBACK at the tested checkpoint ({len(live)} live rules)")
    print(f"    rule: maximise connected nodes subject to max degree <= "
          f"{FALLBACK_DEGREE_SHARE:g} x connected nodes")
    print(f"    {'rate':<8}{'edges':>8}{'connected':>11}{'max deg':>9}{'bar':>8}{'verdict':>10}")
    selected = []
    for candidate in FALLBACK_RATES:
        graph = cocite_adjacency(pairs, floor, candidate)
        deg = degrees(graph)
        top = deg[0] if deg else 0
        bar = FALLBACK_DEGREE_SHARE * len(graph)
        clears = bool(graph) and top <= bar
        print(f"    {candidate:<8.2f}{sum(deg) // 2:>8}{len(graph):>11}{top:>9}{bar:>8.1f}"
              f"{('clears' if clears else 'fails'):>10}")
        if clears:
            selected.append((len(graph), candidate))
    if selected:
        print(f"    -> selected rate {max(selected)[1]:g} ({max(selected)[0]} connected nodes)")
    else:
        print("    -> NO candidate clears; no rate is selected and no re-run is licensed, "
              "which is the outcome the pre-registration names.")

    if not args.extra:
        return

    print("\n  [--extra] WHY MEMBERS LEAVE -- superseded, or still live and no longer connected?")
    print(f"    {'interval':<16}{'left':>6}{'superseded':>12}{'lost edges':>12}{'joined':>8}"
          f"   of the lost edges, how many had a grown denominator")
    ordered = [step for step, _p in marks]
    for before, after in zip(ordered, ordered[1:]):
        _bl, before_pairs, before_conn = snapshots[before]
        after_live, after_pairs, after_conn = snapshots[after]
        left = before_conn - after_conn
        superseded = {r for r in left if r not in after_live}
        lookup = {(a, b): (c, r) for a, b, c, r in after_pairs}
        grew = held = 0
        for a, b, cocited, corendered in before_pairs:
            if a not in after_live or b not in after_live:
                continue
            new = lookup.get((a, b))
            if new is None:
                continue
            was_edge = corendered >= floor and cocited / corendered >= gate
            still = new[1] >= floor and new[0] / new[1] >= gate
            if was_edge and not still:
                grew += new[1] > corendered
                held += new[1] <= corendered
        label = f"{before if before is not None else 'final'} -> " \
                f"{after if after is not None else 'final'}"
        print(f"    {label:<16}{len(left):>6}{len(superseded):>12}{len(left) - len(superseded):>12}"
              f"{len(after_conn - before_conn):>8}   {grew} grew, {held} unchanged")

    print(f"\n  [--extra] RELATION-LINE BILL at {args.rendered} rendered rules")
    print(f"    {'rate':<8}{'connected':>11}{'ids listed':>12}{'approx tokens':>15}")
    for candidate in dict.fromkeys((gate,) + FALLBACK_RATES):
        graph = cocite_adjacency(pairs, floor, candidate)
        ids = relation_line_cost(live, graph, args.rendered, args.trials, rng)
        marker = "  <- shipped" if candidate == gate else ""
        print(f"    {candidate:<8.2f}{len(graph):>11}{ids:>12.0f}"
              f"{ids * TOKENS_PER_RENDERED_ID:>15.0f}{marker}")

    concentration = citation_concentration(run_dir)
    if concentration:
        distinct, total, top_share = concentration
        print(f"\n  [--extra] CITATION CONCENTRATION over training generations: {distinct} "
              f"distinct rules cited, {total} citations, top ten carry {100.0 * top_share:.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="graph run directories, or parents of ace_run_*")
    parser.add_argument("--selftest", action="store_true",
                        help="check the union-find and gate logic; needs no runs")
    parser.add_argument("--extra", action="store_true",
                        help="also print diagnostics no results document quotes")
    parser.add_argument("--all_steps", action="store_true",
                        help="print every checkpoint, not only those section 5 reports")
    parser.add_argument("--min_opportunity", type=float, default=None,
                        help="override the co-render floor the run was built with")
    parser.add_argument("--min_rate", type=float, default=None,
                        help="override the rate gate the run was built with")
    parser.add_argument("--rendered", type=int, default=50,
                        help="rules per call, for the --extra relation-line bill")
    parser.add_argument("--trials", type=int, default=500, help="draws for that estimate")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.paths:
        parser.error("give at least one run directory, or --selftest")

    rng = random.Random(0)
    for path in args.paths:
        for run_dir in run_dirs_under(path):
            report(run_dir, args, rng)


if __name__ == "__main__":
    main()
