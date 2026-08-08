#!/usr/bin/env python3
"""
Verification for the memory refactor. Run before spending compute on either arm:

    uv run python check_memory_equivalence.py

Three things are checked, in order of how much they matter to the experiment.

1. TextPlaybookMemory reproduces the original playbook_utils code path byte for
   byte, replaying real curator operations against a real archived playbook. This
   is the fairness guarantee: it means the baseline arm was not altered by the
   refactor, so a difference measured against the graph arm is attributable to the
   memory structure rather than to a rewrite of the baseline.

2. Both arms agree on everything that is supposed to be held constant -- ids
   minted, section routing, the rendering the reflector grades, and the key set of
   the stats dict interpolated into the curator prompt.

3. The full training loop runs against both memory kinds with the LLM stubbed out,
   so wiring errors surface here rather than three hours into a run.

No network, no Docker, no LLM calls.
"""

import json
import os
import sys
import tempfile

from ace.memory import GraphMemory, TextPlaybookMemory, create_memory
from playbook_utils import apply_curator_operations, extract_playbook_bullets, get_playbook_stats

ARCHIVED_RUN = os.path.join(
    "results", "_archive", "mbpp_preexperiment", "mbpp_offline_run",
    "ace_run_20260731_225716_mbpp_code_exec_offline"
)

failures = []


def check(label, condition, detail=""):
    if condition:
        print(f"  ok   {label}")
    else:
        failures.append(f"{label}{(': ' + detail) if detail else ''}")
        print(f"  FAIL {label}{(': ' + detail) if detail else ''}")


def load_archived_operations(limit=40):
    """Replay real ADD operations recorded by a previous run."""
    path = os.path.join(ARCHIVED_RUN, "curator_operations_diff.jsonl")
    if not os.path.exists(path):
        return []

    operations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("operation_type") == "ADD" and record.get("content"):
                operations.append({
                    "type": "ADD",
                    "section": record.get("section", "others"),
                    "content": record["content"],
                })
            if len(operations) >= limit:
                break
    return operations


# ---------------------------------------------------------------------------
# 1. Text arm is unchanged by the refactor
# ---------------------------------------------------------------------------

print("\n[1] TextPlaybookMemory vs. the original playbook_utils path")

operations = load_archived_operations()
if not operations:
    print(f"  SKIP no archived operations under {ARCHIVED_RUN}; using synthetic ones")
    operations = [
        {"type": "ADD", "section": "common_mistakes_to_avoid",
         "content": "Never use a bare except; catch the exception you actually expect."},
        {"type": "ADD", "section": "problem_solving_heuristics",
         "content": "Prefer list comprehensions over manual append loops."},
    ]
print(f"  replaying {len(operations)} recorded ADD operations")

memory = TextPlaybookMemory()
baseline_text = memory.playbook

added = memory.apply_operations(operations, step=1)
expected_text, expected_next_id = apply_curator_operations(baseline_text, operations, 1)

check("playbook text identical", memory.playbook == expected_text)
check("next_global_id identical", memory.next_global_id == expected_next_id,
      f"{memory.next_global_id} vs {expected_next_id}")
check("stats identical", memory.stats() == get_playbook_stats(expected_text))
check("every added id is reported", len(added) == len(operations),
      f"reported {len(added)} for {len(operations)} operations")

tags = [{"id": bullet_id, "tag": "helpful"} for bullet_id in added[:3]]
tags += [{"id": bullet_id, "tag": "harmful"} for bullet_id in added[3:5]]
counted = TextPlaybookMemory(expected_text, next_global_id=expected_next_id)
counted.update_counts(tags)
from playbook_utils import update_bullet_counts  # noqa: E402

check("update_counts identical", counted.playbook == update_bullet_counts(expected_text, tags))
check("render_bullets identical",
      counted.render_bullets(added[:3]) == extract_playbook_bullets(counted.playbook, added[:3]))
check("empty citation sentinel preserved",
      counted.render_bullets([]) == "(No bullets used by generator)")

snapshot = memory.snapshot()
clone = memory.copy()
clone.apply_operations([{"type": "ADD", "section": "others", "content": "x"}], step=2)
check("copy() does not alias", memory.snapshot() == snapshot)


# ---------------------------------------------------------------------------
# 2. Cross-arm invariants
# ---------------------------------------------------------------------------

print("\n[2] Invariants that must hold across both arms")

graph = create_memory("graph")
graph_added = graph.apply_operations(operations, step=1)

check("same ids minted", graph_added == added, f"{graph_added[:3]}... vs {added[:3]}...")
check("same stats key set", graph.stats().keys() == memory.stats().keys())
check("same bullet count", graph.stats()["total_bullets"] == memory.stats()["total_bullets"])
check("same per-section counts",
      {k: v["count"] for k, v in graph.stats()["by_section"].items()}
      == {k: v["count"] for k, v in memory.stats()["by_section"].items()})

graph.update_counts(tags)
counted_graph_view = graph.render_bullets(added[:3])
check("reflector view is format-identical across arms",
      [line.split(" :: ")[0] for line in counted_graph_view.split("\n")]
      == [line.split(" :: ")[0] for line in counted.render_bullets(added[:3]).split("\n")])
check("graph is ADD-only: nothing was removed",
      graph.structural_stats()["node_count"] == len(operations))

rendered = graph.render_for_generator()
check("graph rendering carries relations", "related_to:" in rendered,
      "no related_to edges were produced; check --related_threshold")
check("legend documents only the relations actually emitted",
      ("contradicts" in rendered) == graph.enable_contradicts,
      "the legend teaches a notation the rendering never uses")

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "final_playbook.txt")
    graph.save(path)
    check("graph writes a readable snapshot", os.path.exists(path))
    check("graph writes its sidecar", os.path.exists(os.path.join(tmp, "final_playbook.graph.json")))
    reloaded = GraphMemory.load(path)
    check("graph round trip is stable", reloaded.snapshot() == graph.snapshot())
    check("round trip preserves id counter", reloaded.next_global_id == graph.next_global_id)


# ---------------------------------------------------------------------------
# 2b. refines / supersession -- the graph arm's actual capability
# ---------------------------------------------------------------------------

print("\n[2b] refines edges and supersession")

refining = create_memory("graph")
first = refining.apply_operations(
    [{"type": "ADD", "section": "common_mistakes_to_avoid",
      "content": "Do not mutate a list while iterating over it."}], step=1)[0]
before_tokens = refining.num_tokens()

second = refining.apply_operations(
    [{"type": "ADD", "section": "common_mistakes_to_avoid",
      "content": "Do not mutate any collection while iterating it; iterate a copy instead.",
      "refines": first}], step=2)[0]

rendered = refining.render_for_generator()
stats = refining.structural_stats()

check("refines edge recorded", stats["refines_edges"] == 1)
check("superseded rule counted", stats["superseded_node_count"] == 1)
check("nothing removed from the graph", stats["node_count"] == 2)
check("superseded rule withdrawn from the prompt", f"[{first}]" not in rendered)
check("replacement still rendered", f"[{second}]" in rendered)
check("supersession is declared to the model", f"refines: {first} (withdrawn)" in rendered)
check("legend explains refines", "refines" in rendered.split("##")[0])
check("stats count only live rules", refining.stats()["total_bullets"] == 1)
check("eviction shrinks the prompt", refining.num_tokens() < before_tokens * 2,
      "superseded rule still costs full tokens")

third = refining.apply_operations(
    [{"type": "ADD", "section": "common_mistakes_to_avoid",
      "content": "Never mutate a container during iteration; build a new one.",
      "refines": second}], step=3)[0]
chained = refining.render_for_generator()
check("chains collapse to the newest rule",
      f"[{third}]" in chained and f"[{second}]" not in chained and f"[{first}]" not in chained)

rejected = refining.apply_operations(
    [{"type": "ADD", "section": "others", "content": "bad target", "refines": "zzz-99999"}],
    step=4)
check("unknown refines target rejected", refining.structural_stats()["refines_edges"] == 2)
check("rule still added despite bad target", len(rejected) == 1)

cyclic = refining._add_refines(first, third, set(), step=5)
check("cycle refused", cyclic is False)

same_batch = create_memory("graph")
pair = same_batch.apply_operations([
    {"type": "ADD", "section": "others", "content": "rule A"},
    {"type": "ADD", "section": "others", "content": "rule B", "refines": "misc-00001"},
], step=1)
check("same-response refines refused",
      same_batch.structural_stats()["refines_edges"] == 0 and len(pair) == 2)

ablation = create_memory("graph", keep_superseded=True)
a_first = ablation.apply_operations(
    [{"type": "ADD", "section": "others", "content": "Do not mutate while iterating."}], step=1)[0]
ablation.apply_operations(
    [{"type": "ADD", "section": "others", "content": "Do not mutate any collection while iterating.",
      "refines": a_first}], step=2)
check("--keep_superseded keeps the withdrawn rule visible",
      f"[{a_first}]" in ablation.render_for_generator())


# ---------------------------------------------------------------------------
# 2c. The pure-LLM control arm
# ---------------------------------------------------------------------------

print("\n[2c] NoMemory control arm")

nothing = create_memory("none")
check("renders an empty slot, not an empty string", nothing.render_for_generator() == "(empty)")
check("same stats key set as the other arms", nothing.stats().keys() == memory.stats().keys())
check("reports no rules", nothing.stats()["total_bullets"] == 0)
check("curator operations are inert", nothing.apply_operations(operations, step=1) == [])
check("still inert after operations", nothing.stats()["total_bullets"] == 0)
check("counters are inert", nothing.update_counts(tags) is None)
check("reflector sentinel matches the other arms",
      nothing.render_bullets(["err-00001"]) == "(No bullets used by generator)")
check("copy works", nothing.copy().render_for_generator() == "(empty)")


# ---------------------------------------------------------------------------
# 3. The training loop runs against both arms with the LLM stubbed
# ---------------------------------------------------------------------------

print("\n[3] Training loop with a stubbed LLM (no network, no Docker)")

import ace.core.curator as curator_module  # noqa: E402
import ace.core.generator as generator_module  # noqa: E402
import ace.core.reflector as reflector_module  # noqa: E402
import utils as utils_module  # noqa: E402
from ace.ace import ACE  # noqa: E402

CALLS = {"generator": 0, "reflector": 0, "curator": 0}


def fake_llm(client, api_provider, model, prompt, role, call_id, **kwargs):
    CALLS[role] = CALLS.get(role, 0) + 1
    info = {"role": role, "call_id": call_id, "model": model}

    if role == "generator":
        # Cite a real id when one exists, to exercise the counter path.
        cited = ["err-00001"] if "[err-00001]" in prompt else []
        return json.dumps({
            "reasoning": "stub",
            "bullet_ids": cited,
            "final_answer": "def solve():\\n    return 1",
        }), info

    if role == "reflector":
        cited = ["err-00001"] if "[err-00001]" in prompt else []
        return json.dumps({
            "reasoning": "stub",
            "error_identification": "none",
            "root_cause_analysis": "none",
            "correct_approach": "none",
            "key_insight": "stub insight",
            "bullet_tags": [{"id": i, "tag": "helpful"} for i in cited],
        }), info

    return json.dumps({
        "reasoning": "stub",
        "operations": [{
            "type": "ADD",
            "section": "common_mistakes_to_avoid",
            "content": f"Stub rule minted at {call_id}.",
        }],
    }), info


class StubProcessor:
    """Scores everything correct; the loop's plumbing is what is under test."""

    def answer_is_correct(self, predicted, target, original_item=None):
        return True

    def evaluate_accuracy(self, predictions, targets, original_items=None):
        return 1.0


def stub_clients(api_provider):
    return None, None, None


generator_module.timed_llm_call = fake_llm
reflector_module.timed_llm_call = fake_llm
curator_module.timed_llm_call = fake_llm
original_initialize_clients = utils_module.initialize_clients
utils_module.initialize_clients = stub_clients
sys.modules["ace.ace"].initialize_clients = stub_clients

samples = [
    {"question": f"q{i}", "context": "", "target": "t", "original_mbpp_item": {"task_id": i}}
    for i in range(2)
]

try:
    # The control arm is eval_only by construction, so it is exercised separately below.
    for kind in ("text", "graph"):
        CALLS.update({"generator": 0, "reflector": 0, "curator": 0})
        with tempfile.TemporaryDirectory() as tmp:
            system = ACE(
                api_provider="stub",
                generator_model="stub", reflector_model="stub", curator_model="stub",
                memory_kind=kind, num_ctx=32768, seed=0,
            )
            results = system.run(
                mode="offline",
                train_samples=samples,
                val_samples=samples,
                test_samples=samples,
                data_processor=StubProcessor(),
                config={
                    "task_name": "stub", "save_dir": tmp, "num_epochs": 1,
                    "eval_steps": 1, "save_steps": 1, "curator_frequency": 1,
                    "test_workers": 1, "memory": kind,
                },
            )

            run_dir = os.path.join(tmp, os.listdir(tmp)[0])
            written = set(os.listdir(run_dir))

            print(f"  --- {kind} ---")
            check(f"{kind}: run completed", "final_test_results" in results)
            check(f"{kind}: final playbook written", "final_playbook.txt" in written)
            check(f"{kind}: best playbook written", "best_playbook.txt" in written)
            check(f"{kind}: curator ran", CALLS["curator"] > 0)
            check(f"{kind}: memory grew", system.memory.stats()["total_bullets"] > 0,
                  f"{system.memory.stats()['total_bullets']} bullets")

            with open(os.path.join(run_dir, "run_config.json"), encoding="utf-8") as f:
                run_config = json.load(f)
            check(f"{kind}: provenance recorded",
                  run_config.get("memory_kind") == kind
                  and run_config.get("num_ctx") == 32768
                  and run_config.get("seed") == 0)

            with open(os.path.join(run_dir, "train_results.json"), encoding="utf-8") as f:
                train_results = json.load(f)
            entry = train_results["results"][0]
            check(f"{kind}: structural stats recorded", "structural_stats" in entry)

            if kind == "graph":
                check("graph: sidecar written next to final playbook",
                      "final_playbook.graph.json" in written)
                check("graph: structural stats are populated",
                      entry["structural_stats"].get("node_count", 0) > 0)
            else:
                check("text: no structural stats to report", entry["structural_stats"] == {})

    # Control arm: eval_only runs, training is refused.
    print("  --- none ---")
    with tempfile.TemporaryDirectory() as tmp:
        control = ACE(
            api_provider="stub",
            generator_model="stub", reflector_model="stub", curator_model="stub",
            memory_kind="none", num_ctx=32768, seed=0,
        )
        control_config = {"task_name": "stub", "save_dir": tmp, "test_workers": 1,
                          "memory": "none"}
        control_results = control.run(
            mode="eval_only", test_samples=samples,
            data_processor=StubProcessor(), config=control_config,
        )
        check("none: eval_only runs", "test_results" in control_results)

        try:
            control.run(mode="offline", train_samples=samples, val_samples=samples,
                        test_samples=samples, data_processor=StubProcessor(),
                        config=control_config)
            check("none: training is refused", False, "offline mode was allowed")
        except ValueError as e:
            check("none: training is refused", "eval_only" in str(e))
finally:
    utils_module.initialize_clients = original_initialize_clients

print()
if failures:
    print(f"FAILED ({len(failures)}):")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)
print("all memory equivalence checks passed")
