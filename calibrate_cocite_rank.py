"""Stage A metrics 1 to 4 and 9 of `docs/EXPERIMENT_COCITE_V2.md`, on finished runs.

    uv run python calibrate_cocite_rank.py results/finer/final/seed*/cocite
    uv run python calibrate_cocite_rank.py results/formula/cocite_experiment/seed*/cocite
    uv run python calibrate_cocite_rank.py --selftest

  [1] does the predicate reach the memory?   connected >= 40% of live rules        BLOCKING
  [2] is there a hub?                        max degree <= 25% of connected        BLOCKING
  [3] do the edges predict co-citation?      held-out lift > 1.0x                  BLOCKING
  [4] is outcome weighting a no-op?          top-k picks differ on >= 10% of rules  descriptive
  [9] does binary reward collapse formula?   severance < 25% at depth >= 3         BLOCKING

**Nothing here re-implements the graph.** Every edge, weight and statistic comes from
`GraphMemory` itself (`record_generation`, `_cocite_edge_pairs`, `_cocite_weight`,
`structural_stats`), every reward from `ace._generation_reward` fed by the task's own
`answer_is_correct`, and `beta` and `k` are the constants `ace.memory.graph` exports. A copy of
that logic here is exactly what drifts, and an earlier version of this script kept one.

**What is measured is the outcome-weighted graph the v2 arm ships, rebuilt by replay.** No
committed run recorded a reward, so the weighted store cannot be read off a sidecar. Instead each
run's checkpoint is reloaded, its pair store emptied, and every logged pre-train generation up to
the checkpoint the final test used is replayed through `record_generation` with the reward the
run would have banked. That store holds less evidence than the run's own: the post-curate
generation of each step is not in `bullet_usage_log.jsonl`, and a replay performs no inheritance.
So the unweighted graph is reported twice beside the verdict -- off the sidecar, which is the full
evidence, and off the same replay -- which separates the effect of weighting from the effect of
the smaller evidence base.

Read-only. No LLM, no network, and nothing is written to a run directory.
"""

import argparse
import itertools
import json
import os

from ace.ace import _generation_reward
from ace.memory.graph import DEFAULT_COCITE_BETA, DEFAULT_COCITE_K, CocitePair, GraphMemory
from analysis_common import arm_of, collect_runs, read_config
from compare_arms import load_checkpoint_choice

#: Fixed in docs/EXPERIMENT_COCITE_V2.md section 9 before any of this ran. Printed beside every
#: measurement so a reader never has to take the verdict on trust.
BARS = {
    "connected_share": 0.40,
    "max_degree_share": 0.25,
    "held_out_lift": 1.0,
    "weighting_effect": 0.10,
    "severance": 0.25,
}

#: The v2 edge predicate, as the treatment command sets it. `rank` names the treatment; `beta`
#: and `k` are the runtime's own constants, so this script cannot score a different predicate
#: from the one a run builds.
V2_PREDICATE = dict(cocite=True, cocite_predicate="rank", cocite_beta=DEFAULT_COCITE_BETA,
                    cocite_k=DEFAULT_COCITE_K)


class NoEncoder:
    """A replay never embeds anything; any call to encode is a bug in the replay."""

    nli_available = False

    def encode(self, texts):
        raise AssertionError("the Stage A replay must not encode text")


def generation_rewards(run_dir, task_name):
    """(epoch, step) -> the reward the run banks for that step's pre-train generation, or None.

    Only `finer` needs this. The log records the all-four-exact boolean, which is not what the
    arm banks there, so each saved pre-train answer is re-graded through the task's own
    `answer_is_correct` and the resulting feedback handed to `ace._generation_reward`, exactly as
    the loop does. Every other task banks the logged boolean, so None means "use it".
    """
    if (task_name or "").lower() != "finer":
        return None
    path = os.path.join(run_dir, "pre_train_post_train_results.json")
    if not os.path.isfile(path):
        # Never fall back to the logged boolean: on `finer` that is the binary reward the arm
        # does not use, and a metric scored on it would describe a different graph.
        raise FileNotFoundError(f"{run_dir}: finer needs pre_train_post_train_results.json to "
                                f"recompute the fractional reward, and it is missing")
    from eval.finance.data_processor import DataProcessor

    processor = DataProcessor(task_name="finer")
    with open(path, encoding="utf-8") as handle:
        records = json.load(handle)
    rewards = {}
    for record in records:
        feedback = {}
        correct = processor.answer_is_correct(record["pre_train_result"]["final_answer"],
                                              record["target"], feedback=feedback)
        rewards[(record["epoch"], record["step"])] = _generation_reward(correct, feedback)
    return rewards


def logged_generations(run_dir, task_name, up_to_step=None):
    """Each logged pre-train generation as (step, rendered ids, cited ids, reward), in order.

    Rows with fewer than two rendered rules carry no pair; that includes the reflection-round
    rows, which log an empty rendered set and which the run never records either.
    """
    path = os.path.join(run_dir, "bullet_usage_log.jsonl")
    if not os.path.isfile(path):
        return []
    fractional = generation_rewards(run_dir, task_name)
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rendered = list(row.get("bullet_ids_rendered") or [])
            if len(rendered) < 2:
                continue
            step = row.get("step")
            if up_to_step is not None and step is not None and step > up_to_step:
                continue
            if fractional is None:
                reward = _generation_reward(bool(row.get("is_correct")), {})
            else:
                reward = fractional.get((row.get("epoch"), step))
                if reward is None:
                    raise ValueError(f"{run_dir}: no pre-train answer for epoch "
                                     f"{row.get('epoch')} step {step}, so its reward cannot be "
                                     f"computed")
            rows.append((step or 0, rendered, list(row.get("bullet_ids_used") or []), reward))
    return rows


def checkpoint_memory(run_dir, weighted):
    """The run's checkpoint under the v2 predicate, with its sidecar pair store intact."""
    return GraphMemory.load(os.path.join(run_dir, "best_playbook.txt"), backend=NoEncoder(),
                            cocite_outcome_weighted=weighted, **V2_PREDICATE)


def replay(run_dir, rows, weighted):
    """The checkpoint with its pair store rebuilt from `rows` by the run's own recorder."""
    memory = checkpoint_memory(run_dir, weighted)
    memory._cocite_pairs = {}
    memory._cocite_adjacency = None
    for step, rendered, cited, reward in rows:
        memory.record_generation(rendered, cited, step=step, reward=reward)
    return memory


def reach(memory):
    """(connected, live, max degree) from the memory's own structural statistics."""
    stats = memory.structural_stats()
    return stats["cocite_connected_nodes"], stats["active_node_count"], stats["cocite_max_degree"]


def held_out_lift(run_dir, rows):
    """Lift of the shipped edges over base rate on generations they were not built from.

    Edges come from `_cocite_edge_pairs` on a store replayed from the first half of the logged
    generations, outcome-weighted; they are scored on the second half. Returns None when there is
    nothing to score, which the caller treats as a failure of a blocking metric.
    """
    if len(rows) < 4:
        return None
    half = len(rows) // 2
    memory = replay(run_dir, rows[:half], weighted=True)
    edges = memory._cocite_edge_pairs()
    if not edges:
        return None
    on_edge_seen = on_edge_hit = seen = hit = 0
    for _step, rendered, cited, _reward in rows[half:]:
        cited = set(cited)
        for pair in itertools.combinations(sorted(set(rendered)), 2):
            together = pair[0] in cited and pair[1] in cited
            seen += 1
            hit += together
            if pair in edges:
                on_edge_seen += 1
                on_edge_hit += together
    if not on_edge_seen or not hit:
        return None
    return (on_edge_hit / on_edge_seen) / (hit / seen), len(edges), on_edge_seen


def severance_verdict(memory):
    """Share of live pairs co-cited at least three times whose banked reward is zero."""
    live = set(memory.active_nodes())
    rates = {}
    for depth in (1, 3, 5):
        selected = [entry for (a, b), entry in memory._cocite_pairs.items()
                    if a in live and b in live and entry.cocited >= depth]
        if selected:
            zero = sum(1 for entry in selected if entry.reward == 0.0)
            rates[depth] = (zero, len(selected), zero / len(selected))
    deep = rates.get(3)
    if deep is None:
        print("  [9] binary severance: NOT COMPUTABLE (no pair co-cited three or more times) -- a "
              "blocking metric that cannot be computed fails")
        return ("9 severance", None, True)
    line = "  ".join(f"depth>={d}: {z}/{n} = {r:.1%}" for d, (z, n, r) in sorted(rates.items()))
    ok = deep[2] < BARS["severance"]
    print(f"  [9] binary reward severs {line}   (bar < {BARS['severance']:.0%} at depth >= 3)  "
          f"{'PASS' if ok else 'FAIL'}")
    return ("9 severance", ok, True)


def report(run_dir):
    label = os.path.relpath(run_dir).replace(os.sep, "/")
    config = read_config(run_dir)
    task = config.get("task_name")
    if config.get("memory_kind") != "graph":
        print(f"{label}: {arm_of(run_dir, config)} arm, no graph to score")
        return []
    uncomputable = [("1 reach", None, True), ("2 hub", None, True), ("3 lift", None, True)]
    if not os.path.isfile(os.path.join(run_dir, "best_playbook.graph.json")):
        print(f"{label}: graph arm with no best_playbook.graph.json -- every metric is "
              f"uncomputable, which fails rather than passes")
        return uncomputable

    choice = load_checkpoint_choice(run_dir)
    step = choice["selected_step"] if choice else None
    rows = logged_generations(run_dir, task, up_to_step=step)

    print("=" * 98)
    print(f"{label}   task={task}  checkpoint step={step}  generations replayed={len(rows)}")
    if not rows:
        print("  no logged generations to replay -- every metric is uncomputable, which fails")
        return uncomputable

    weighted = replay(run_dir, rows, weighted=True)
    unweighted = replay(run_dir, rows, weighted=False)
    full_connected, full_live, full_top = reach(checkpoint_memory(run_dir, weighted=False))
    replay_connected, replay_live, replay_top = reach(unweighted)
    print(f"  reference, unweighted: sidecar (full evidence) connected {full_connected}/{full_live} "
          f"max degree {full_top};  replay connected {replay_connected}/{replay_live} "
          f"max degree {replay_top}")

    connected, live, top = reach(weighted)
    verdicts = []
    share = connected / live if live else 0.0
    ok = connected > 0 and share >= BARS["connected_share"]
    verdicts.append(("1 reach", ok, True))
    print(f"  [1] connected {connected} of {live} live rules = {share:.1%}   "
          f"(outcome-weighted replay, bar >= {BARS['connected_share']:.0%})  "
          f"{'PASS' if ok else 'FAIL'}")

    degree_share = top / connected if connected else 0.0
    ok = connected > 0 and degree_share <= BARS["max_degree_share"]
    verdicts.append(("2 hub", ok, True))
    print(f"  [2] max degree {top} of {connected} connected = {degree_share:.1%}   "
          f"(outcome-weighted replay, bar <= {BARS['max_degree_share']:.0%})  "
          f"{'PASS' if ok else 'FAIL'}")

    lift = held_out_lift(run_dir, rows)
    if lift is None:
        verdicts.append(("3 lift", None, True))
        print("  [3] held-out lift: NOT COMPUTABLE (too few generations, no edges, or no held-out "
              "co-citation) -- a blocking metric that cannot be computed fails")
    else:
        ratio, built, scored = lift
        ok = ratio > BARS["held_out_lift"]
        verdicts.append(("3 lift", ok, True))
        print(f"  [3] held-out lift {ratio:.2f}x over base rate, {built} edges built on the first "
              f"half, {scored} co-render events scored on the second   "
              f"(bar > {BARS['held_out_lift']:.1f}x)  {'PASS' if ok else 'FAIL'}")

    # A rule's own outbound picks, never its neighbourhood: a neighbourhood also holds the rules
    # that picked it, so a change there can be someone else's choice rather than this rule's.
    heavy, light = weighted._cocite_nominations(), unweighted._cocite_nominations()
    nodes = set(heavy) | set(light)
    moved = sum(1 for node in nodes if set(heavy.get(node, ())) != set(light.get(node, ())))
    effect = moved / len(nodes) if nodes else 0.0
    verdicts.append(("4 weighting", effect >= BARS["weighting_effect"], False))
    print(f"  [4] outcome weighting changes the top-{DEFAULT_COCITE_K} picks of {moved} of {len(nodes)} rules "
          f"= {effect:.1%}   (bar >= {BARS['weighting_effect']:.0%}, descriptive)")

    if (task or "").lower() == "formula":
        verdicts.append(severance_verdict(weighted))
    else:
        print("  [9] not formula, so the reward is fractional here and severance is not the risk")
    return verdicts


def selftest():
    """The replay glue, on a memory small enough to check by hand.

    The graph logic itself is checked by `check_finance_wiring.py` [1o]; what can go wrong here is
    the plumbing around it, so that is what is tested.
    """
    memory = GraphMemory(backend=NoEncoder(), cocite_outcome_weighted=True, **V2_PREDICATE)
    for node in ("a", "b", "c", "d"):
        memory._add_node(node, "others", node, step=0)

    rows = [(1, ["a", "b", "c"], ["a", "b"], 1.0),
            (2, ["a", "b", "c"], ["a", "b"], 0.0),
            (3, ["a", "c", "d"], ["a", "c"], 0.5),
            (4, ["b", "d"], ["b", "d"], 1.0)]
    for step, rendered, cited, reward in rows:
        memory.record_generation(rendered, cited, step=step, reward=reward)
    entry = memory._cocite_pairs[("a", "b")]
    assert (entry.cocited, entry.corendered, entry.reward) == (2.0, 2.0, 1.0), entry
    assert memory._cocite_pairs[("a", "c")] == CocitePair(1.0, 3.0, 3, 0.5)

    connected, live, _top = reach(memory)
    assert (connected, live) == (4, 4), (connected, live)

    # The reward is the loop's own: a fractional score passes through, a bool becomes 0 or 1.
    assert _generation_reward(False, {"partial_credit": 0.75}) == 0.75
    assert _generation_reward(True, {}) == 1.0 and _generation_reward(False, {}) == 0.0

    # Severance reads the banked reward on the store, not a recomputation of it.
    memory._cocite_pairs[("a", "d")] = CocitePair(4.0, 4.0, 4, 0.0)
    assert severance_verdict(memory) == ("9 severance", False, True)

    # The replay reproduces the recorder exactly: replaying the same rows onto an empty store of
    # the same memory yields the same entries.
    replayed = GraphMemory(backend=NoEncoder(), cocite_outcome_weighted=True, **V2_PREDICATE)
    for node in ("a", "b", "c", "d"):
        replayed._add_node(node, "others", node, step=0)
    for step, rendered, cited, reward in rows:
        replayed.record_generation(rendered, cited, step=step, reward=reward)
    assert replayed._cocite_pairs[("a", "b")] == entry

    print("calibrate_cocite_rank selftest passed")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="run directories, or parents of ace_run_*")
    parser.add_argument("--selftest", action="store_true",
                        help="check the replay glue on a tiny memory; needs no runs")
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if not args.paths:
        parser.error("give at least one run directory, or --selftest")

    runs = collect_runs(args.paths)
    if not runs:
        print("no runs with a run_config.json under those paths")
        return 2

    blocking_failures = []
    for run_dir in runs:
        for name, ok, blocking in report(run_dir):
            # `is not True`: a blocking metric that could not be computed is a failure.
            if blocking and ok is not True:
                reason = "not computable" if ok is None else "failed"
                blocking_failures.append((os.path.basename(run_dir), name, reason))

    print()
    print("=" * 98)
    if blocking_failures:
        print(f"{len(blocking_failures)} BLOCKING metric(s) failed. Under section 9 of the "
              f"pre-registration no run is authorised:")
        for run, name, reason in blocking_failures:
            print(f"  {run}: metric {name} {reason}")
        return 1
    print("every blocking metric passed on every run scored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
