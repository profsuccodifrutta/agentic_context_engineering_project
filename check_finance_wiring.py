#!/usr/bin/env python3
"""End-to-end check of the finance (formula) port, with the LLM stubbed.

Everything except the model: the real probe splits, the real DataProcessor, the real ACE
loop, the real memories, the real prompts. The generator, reflector and curator are replaced
by functions that return fixed JSON, so the whole offline loop runs in seconds with no
network, no Docker and no tokens spent.

What this is guarding:

  * the finance data processor's signatures line up with what utils/ACE call them with
    (upstream shipped 2-arg answer_is_correct against a 3-arg call site)
  * the finance prompt set is actually selected, and the generator's four fields land under
    the right headings -- the upstream template orders them differently from our call site,
    and bound wrongly it fails silently rather than raising
  * environment feedback reaches the reflector with expected-vs-got in it
  * all three memory arms complete a run and write the artifacts the analysis tools read

    uv run python check_finance_wiring.py
"""

import contextlib  # noqa: I001  -- native_deps must precede anything that pulls in torch
import io
import json
import os
import sys
import tempfile

import ace.native_deps  # noqa: F401  -- must precede sentence-transformers imports

import ace.core.curator as curator_module
import ace.core.generator as generator_module
import ace.core.reflector as reflector_module
import utils as utils_module
from ace.ace import ACE
from ace.prompts import FINANCE_PROMPTS
from eval.finance.data_processor import DataProcessor, load_data

FAILURES = []
SEEN = {"generator_prompts": [], "reflector_prompts": [], "curator_prompts": []}


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{(' -- ' + detail) if detail and not ok else ''}")
    if not ok:
        FAILURES.append(label)


def fake_llm(client, api_provider, model, prompt, role, call_id, **kwargs):
    """Return schema-valid output, and record the prompt so it can be inspected."""
    SEEN[f"{role}_prompts"].append(prompt)
    info = {"role": role, "call_id": call_id, "model": model}

    if role == "generator":
        cited = ["calc-00001"] if "[calc-00001]" in prompt else []
        # Deliberately messy: exercises normalize_numeric on the way through, and is wrong
        # against every target so the failure path (reflect -> curate) actually runs.
        return json.dumps({"reasoning": "stub", "bullet_ids": cited,
                           "final_answer": "$1,234.50"}), info

    if role == "reflector":
        cited = ["calc-00001"] if "[calc-00001]" in prompt else []
        return json.dumps({
            "reasoning": "stub", "error_identification": "wrong formula",
            "root_cause_analysis": "stub", "correct_approach": "stub",
            "key_insight": "stub insight",
            "bullet_tags": [{"id": i, "tag": "harmful"} for i in cited],
        }), info

    return json.dumps({
        "reasoning": "stub",
        "operations": [{"type": "ADD", "section": "formulas_and_calculations",
                        "content": f"Stub formula rule minted at {call_id}.",
                        "refines": None}],
    }), info


def stub_clients(api_provider):
    return None, None, None


def main():
    generator_module.timed_llm_call = fake_llm
    reflector_module.timed_llm_call = fake_llm
    curator_module.timed_llm_call = fake_llm
    utils_module.initialize_clients = stub_clients
    sys.modules["ace.ace"].initialize_clients = stub_clients

    print("\n[1] data processor")
    processor = DataProcessor(task_name="formula")
    with open("./eval/finance/data/probe_config.json", encoding="utf-8") as fh:
        cfg = json.load(fh)["formula"]
    train = processor.process_task_data(load_data(cfg["train_data"]))
    val = processor.process_task_data(load_data(cfg["val_data"]))
    test = processor.process_task_data(load_data(cfg["test_data"]))
    check("probe splits load", len(train) == 50 and len(val) == 20 and len(test) == 20,
          f"{len(train)}/{len(val)}/{len(test)}")
    check("questions parsed (formula definition stripped, as upstream does)",
          all("Formula:" not in s["question"] for s in train))

    feedback = {}
    correct = utils_module.check_answer(processor, "0.01", train[0]["target"], train[0],
                                        feedback=feedback)
    check("answer_is_correct accepts the 4-arg call utils makes", correct is False)
    check("feedback carries expected and got",
          feedback.get("expected") == train[0]["target"] and feedback.get("got") == "0.01",
          str(feedback))
    rendered = utils_module.format_environment_feedback(False, feedback)
    check("environment feedback shows expected vs got",
          "Expected answer:" in rendered and "Predicted answer:" in rendered, rendered[:120])
    check("evaluate_accuracy accepts the 3-arg call utils makes",
          processor.evaluate_accuracy(["1.0", "2.0"], ["1.0", "9.0"], [train[0], train[1]]) == 0.5)

    print("\n[1b] answer extraction on the envelopes this task actually produces")
    # Every case below was taken from a real formula probe response. A numeric task breaks
    # the extractor in ways a code task never did: the model writes its working in LaTeX,
    # which is not valid JSON escaping, and writes the answer as a bare number because it
    # is one -- and both regex fallbacks demanded a quoted string.
    cases = [
        ('clean envelope',
         '{"reasoning": "r", "bullet_ids": [], "final_answer": "0.05"}', "0.05"),
        ('answer as a JSON number',
         '{"reasoning": "r", "bullet_ids": [], "final_answer": 10.72}', "10.72"),
        ('LaTeX in reasoning breaks strict JSON',
         '{"reasoning": "$$Y = \\frac{D}{P}$$ and \\sigma_d", "bullet_ids": [], '
         '"final_answer": "0.05"}', "0.05"),
        ('LaTeX in reasoning AND a bare number',
         '{"reasoning": "\\approx \\sigma", "bullet_ids": [], "final_answer": 229.27}',
         "229.27"),
        ('bare number with a stray closing quote',
         '{"reasoning": "r", "bullet_ids": [], "final_answer": 229.27"\n}', "229.27"),
        ('negative and exponent forms',
         '{"reasoning": "\\frac", "final_answer": -1.5e6}', "-1500000.0"),
        ('negative bare number in a broken envelope',
         '{"reasoning": "\\sigma unterminated, "final_answer": -15000000.0}', "-15000000.0"),
        # An even run of backslashes must be consumed whole. Doubling by lookahead alone
        # skips the pair, resumes on its second character, and turns "\\approx" into a run
        # of three -- a valid \\ followed by an invalid \a, still unparseable.
        ('a doubled backslash before a LaTeX macro',
         '{"reasoning": "0.08 / 0.05 \\\\approx 1.6", "final_answer": 1.6}', "1.6"),
    ]
    for label, payload, expected in cases:
        with contextlib.redirect_stdout(io.StringIO()):
            got = utils_module.extract_answer(payload)
        # Compared as numbers where both sides are numeric: a value that survived as JSON
        # comes back as a float ("-1500000.0"), one rescued by regex as written text.
        ok = str(got).strip() == expected
        if not ok:
            try:
                ok = float(str(got).strip()) == float(expected)
            except ValueError:
                ok = False
        check(f"{label}", ok, f"got {got!r}, want {expected!r}")

    # The fallback must not invent an answer where the model emitted none.
    with contextlib.redirect_stdout(io.StringIO()):
        got = utils_module.extract_answer('{"reasoning": "ran out of room mid-sentence and')
    check("no answer stays no answer",
          got in ("No final answer found", "raise ValueError('No valid code extracted from LLM')"),
          repr(got))

    # The reflector and curator read their own envelopes through a different function, and
    # it had the same blind spot: on the formula probe 24 of 50 schema-perfect reflector
    # responses were rejected whole because the model wrote its working in LaTeX. The tags
    # survived only because Reflector falls back to a bracket scan; a curator response has
    # no such fallback and would have lost the whole operation list.
    from playbook_utils import extract_json_from_text, parse_playbook_line
    latex_reflection = (
        '```json\n{"reasoning": "$R_p = 11\\%$, $\\sigma_d = 5\\%$, so $0.08 / 0.05 '
        '\\\\approx 1.6$", "key_insight": "\\text{ratios need decimals}", '
        '"bullet_tags": [{"id": "calc-00002", "tag": "neutral"}]}\n```'
    )
    parsed = extract_json_from_text(latex_reflection)
    check("LaTeX in a reflector envelope still yields bullet_tags",
          isinstance(parsed, dict) and parsed.get("bullet_tags") == [
              {"id": "calc-00002", "tag": "neutral"}], repr(parsed)[:120])
    clean = '{"operations": [{"type": "ADD", "section": "s", "content": "c", "refines": null}]}'
    check("a well-formed envelope is read identically",
          extract_json_from_text(clean) == json.loads(clean))

    print("\n[1c] reflector prompt stays inside the served context")
    # The generator can spend its whole max_tokens budget on reasoning, and that trace goes
    # into the reflector prompt verbatim. On the formula probe one reached 36,268 chars and
    # the reflector returned empty fifteen times: num_ctx is pinned to 32768 but Ollama
    # clamps it to the model's own context, and gemma4:e2b's wall sits near 12.4k tokens.
    from ace.core.reflector import MAX_TRACE_TOKENS, _truncate_trace
    runaway = "The model reconsiders the same step. " * 4000
    trimmed = _truncate_trace(runaway)
    check("a runaway trace is capped",
          utils_module.count_tokens(trimmed) <= MAX_TRACE_TOKENS,
          f"{utils_module.count_tokens(trimmed)} tokens")
    check("truncation keeps head and tail",
          trimmed.startswith(runaway[:80]) and trimmed.endswith(runaway[-80:]))
    check("truncation is signposted to the reflector", "omitted" in trimmed)
    for label, trace in (("empty", ""), ("short", "step 1: add. step 2: done.")):
        check(f"a {label} trace is passed through unchanged", _truncate_trace(trace) == trace)
    # Whole-prompt budget: cap plus every other field must clear the wall with room to answer.
    whole = FINANCE_PROMPTS.reflector.format(
        train[0]["question"], runaway, "1.0", "2.0", "expected 2.0 got 1.0", "x " * 3000)
    check("worst-case reflector prompt clears the 12.4k-token wall",
          utils_module.count_tokens(
              FINANCE_PROMPTS.reflector.format(train[0]["question"], trimmed, "1.0", "2.0",
                                               "expected 2.0 got 1.0", "x " * 3000)) < 10000,
          f"untruncated would be {utils_module.count_tokens(whole)} tokens")

    print("\n[1d] curator hygiene sanitiser")
    # The curator is shown "[calc-00013] helpful=5 harmful=0 :: <rule>" and told not to repeat
    # the id. gemma4:e2b pastes it into the rule text anyway (43 of 75 ADDs on the formula text
    # arm, 34 of 65 on the graph arm, 0 of 221 across every MBPP run) and re-adds rules verbatim
    # (23 of 75). Both cases are enforcement of the prompt's own instructions, in shared code,
    # identical in all three arms -- so it has to be exact about what it touches.
    from playbook_utils import sanitize_curator_operations, strip_rendered_prefix

    prefix_cases = [
        ("full counters", "[calc-00013] helpful=5 harmful=0 :: Convert percentages first.",
         "Convert percentages first.", 1),
        ("nested three deep",
         "[err-00017] helpful=4 harmful=0 :: [err-00011] helpful=3 harmful=0 :: "
         "[err-00002] helpful=1 harmful=0 :: Convert percentages first.",
         "Convert percentages first.", 3),
        ("helpful only, no harmful", "[prob-00066] helpful=1 :: Convert percentages first.",
         "Convert percentages first.", 1),
        ("bare id, no counters", "[calc-00003] Convert percentages first.",
         "Convert percentages first.", 1),
        ("a rule that legitimately opens with a bracket",
         "[Important] Convert percentages first.", "[Important] Convert percentages first.", 0),
        ("a bracketed id of the wrong shape is left alone",
         "[calc-13] Convert percentages first.", "[calc-13] Convert percentages first.", 0),
        ("an ordinary rule passes through byte-identical",
         "Convert percentages first.", "Convert percentages first.", 0),
    ]
    for label, raw, expected, expected_n in prefix_cases:
        cleaned, removed = strip_rendered_prefix(raw)
        check(f"prefix: {label}", cleaned == expected and removed == expected_n,
              f"got {cleaned!r} after {removed} strip(s)")

    playbook = ("## FORMULAS AND CALCULATIONS\n"
                "[calc-00001] helpful=5 harmful=0 :: Convert percentages first.\n")
    ops = [
        {"type": "ADD", "section": "s", "content": "[calc-00001] helpful=5 harmful=0 :: "
                                                   "Convert percentages first."},
        {"type": "ADD", "section": "s", "content": "  convert   percentages first "},
        {"type": "ADD", "section": "s", "content": "Convert percentages first, unless scaled."},
        {"type": "ADD", "section": "s", "content": "Round to the nearest hundredth."},
        {"type": "ADD", "section": "s", "content": "Round to the nearest hundredth"},
        {"type": "ADD", "section": "s", "content": "[calc-00007] helpful=0 harmful=0 :: "},
        {"type": "UPDATE", "bullet_id": "calc-00001", "content": "untouched"},
    ]
    kept, report = sanitize_curator_operations(ops, playbook)
    kept_adds = [op["content"] for op in kept if op.get("type") == "ADD"]
    check("dedup: an exact re-add of a visible rule is dropped",
          "Convert percentages first." not in kept_adds, str(kept_adds))
    check("dedup: whitespace/case/trailing-punctuation variants count as the same rule",
          not any(k.strip().casefold().startswith("convert   percentages") for k in kept_adds),
          str(kept_adds))
    check("dedup: a near-duplicate survives (that is what refines is for)",
          "Convert percentages first, unless scaled." in kept_adds, str(kept_adds))
    check("dedup: a duplicate within one batch is dropped",
          kept_adds.count("Round to the nearest hundredth.") == 1
          and "Round to the nearest hundredth" not in kept_adds, str(kept_adds))
    check("content that is nothing but a pasted id is dropped", report["emptied"] == 1,
          str(report))
    check("non-ADD operations pass through untouched",
          any(op.get("type") == "UPDATE" for op in kept), str(kept))
    check("report accounts for every ADD",
          report["adds"] == 6 and report["kept"] + report["dropped"] == 6, str(report))

    # Inertness: the sanitiser must be a no-op on operations that were already clean, or it
    # would change what the MBPP arms would have accumulated and break their comparability.
    clean_ops = [{"type": "ADD", "section": "s", "content": "Prefer list comprehensions."},
                 {"type": "ADD", "section": "s", "content": "Avoid mutating while iterating."}]
    kept_clean, clean_report = sanitize_curator_operations(clean_ops, playbook)
    check("clean operations are returned unchanged and unwrapped",
          kept_clean == clean_ops and clean_report["stripped"] == 0
          and clean_report["dropped"] == 0, str(clean_report))

    print("\n[1e] grow-and-refine, and that both arms make the same decision")
    # The paper's third key innovation (arXiv:2510.04618 section 3.2) was disabled in every run
    # here, because BulletpointAnalyzer merges bullets and the graph arm could not mirror the
    # removal. That left the flat arm as ACE-minus-a-component and made `refines` a
    # re-implementation of what its competitor had been stripped of (FORMULA_RESULTS.md 6.3).
    # It is now shared: one decision rule, two representations. These cases guard the decision
    # and, more importantly, that the two arms reach the *same* one -- if they ever diverge,
    # the comparison is measuring two treatments at once.
    import math as _math

    import numpy as np

    from ace.memory import create_memory
    from ace.memory.refine import duplicate_groups
    from ace.memory.retrieval import score_against_query

    class StubBackend:
        """Hand-placed unit vectors, so these cases test the mechanism, not the encoder."""

        def __init__(self, angles):
            self.angles = angles

        def encode(self, texts):
            return np.array([[_math.cos(self.angles[t]), _math.sin(self.angles[t])]
                             for t in texts], dtype=np.float32)

    near = _math.acos(0.95)      # 0.95 similarity to the rule at angle 0
    step = _math.acos(0.93)      # 0.93 between neighbours, 0.73 across two hops
    far = _math.acos(0.40)
    rules = {"first rule": 0.0, "near duplicate of the first": near,
             "unrelated rule": far, "chain B": step, "chain C": 2 * step}
    stub = StubBackend(rules)

    items = [("calc-00001", "first rule"), ("calc-00002", "near duplicate of the first"),
             ("calc-00003", "unrelated rule")]
    groups = duplicate_groups(items, stub, 0.90)
    check("dedup: a near-duplicate is absorbed by the earlier rule",
          groups == {"calc-00001": ["calc-00002"]}, str(groups))
    check("dedup: an unrelated rule survives", "calc-00003" not in
          {v for vs in groups.values() for v in vs}, str(groups))
    check("dedup: below threshold nothing is absorbed",
          duplicate_groups(items, stub, 0.99) == {}, str(duplicate_groups(items, stub, 0.99)))

    chain = [("calc-00001", "first rule"), ("calc-00002", "chain B"), ("calc-00003", "chain C")]
    chained = duplicate_groups(chain, stub, 0.90)
    check("dedup: similarity is measured against the survivor, not transitively",
          chained == {"calc-00001": ["calc-00002"]},
          f"{chained} -- C is 0.73 from the survivor and must not be chained in")

    # The invariant that matters: same rules in, same rules visible out, in both arms.
    ops = [{"type": "ADD", "section": "formulas_and_calculations", "content": text}
           for text in ("first rule", "near duplicate of the first", "unrelated rule")]
    tags = [{"id": "calc-00002", "tag": "helpful"}, {"id": "calc-00002", "tag": "helpful"}]

    visible, counters, reports = {}, {}, {}
    for kind in ("text", "graph"):
        memory = create_memory(kind, None, **({"backend": stub} if kind == "graph" else {}))
        memory.apply_operations(ops, step=1)
        for tag in tags:
            memory.update_counts([tag])
        reports[kind] = memory.refine(stub, 0.90)
        rendered = [entry for entry in
                    (parse_playbook_line(line) for line in memory.snapshot().split("\n"))
                    if entry]
        visible[kind] = [entry["id"] for entry in rendered]
        counters[kind] = {e["id"]: (e["helpful"], e["harmful"]) for e in rendered}
        if kind == "graph":
            graph_memory = memory

    check("both arms end with the same visible rules",
          visible["text"] == visible["graph"], f"{visible['text']} vs {visible['graph']}")
    check("both arms end with the same counters",
          counters["text"] == counters["graph"], f"{counters['text']} vs {counters['graph']}")
    check("both arms absorbed the same number of rules",
          reports["text"]["absorbed"] == reports["graph"]["absorbed"] == 1, str(reports))
    check("the absorbed rule's counters move to the survivor, not into the bin",
          counters["text"].get("calc-00001") == (2, 0), str(counters["text"]))
    check("the absorbed rule is gone from the flat playbook",
          "calc-00002" not in visible["text"], str(visible["text"]))

    stats = graph_memory.structural_stats()
    check("the graph keeps the absorbed rule as a node",
          stats["node_count"] == 3 and stats["active_node_count"] == 2, str(stats))
    check("de-duplication edges are counted apart from curator-declared ones",
          stats["refines_edges_dedup"] == 1 and stats["refines_edges_curator"] == 0, str(stats))

    # Idempotence. Grow-and-refine runs after every curation, so a pass over an already-refined
    # memory must be a no-op -- otherwise the schedule would keep eroding the playbook.
    again = graph_memory.refine(stub, 0.90)
    check("graph: refining twice absorbs nothing further", again["absorbed"] == 0, str(again))

    # And the real encoder must agree with the hand-built vectors on an obvious case, or the
    # stub above is testing a mechanism that never fires in a run.
    from ace.memory.embeddings import EmbeddingBackend
    real = duplicate_groups(
        [("calc-00001", "Always convert percentage inputs to decimals before calculating."),
         ("calc-00002", "Always convert percentage inputs into decimals before you calculate."),
         ("calc-00003", "Round the final answer to two decimal places.")],
        EmbeddingBackend(), 0.90)
    check("the real encoder absorbs a genuine restatement and leaves a distinct rule alone",
          real == {"calc-00001": ["calc-00002"]}, str(real))

    print("\n[1f] retrieval, and the case that separates the two arms")
    # Paper section 3.1 lists fine-grained retrieval as a property of the itemized design, and
    # neither arm had it: render_for_generator(query=...) raised in both, so `related_to` was
    # written by cosine similarity and then read only for display text. That left the only
    # functional graph mechanism (`refines`) reproducible by a flat list with a supersedes
    # field, and the only mechanism needing a graph doing nothing. Retrieval is what makes the
    # comparison a test of structure -- so the case below is the experiment in miniature.
    from ace.memory.graph import GraphMemory
    from ace.memory.retrieval import fit_to_budget, flat_order, graph_order

    # Vectors chosen so relevance and connectivity disagree. q is the question.
    #   A  identical to q                       -> the seed
    #   B  0.70 to q, 0.70 to A                 -> relevant and connected
    #   C  0.10 to q, 0.78 to B                 -> reachable in two hops, never relevant
    #   D  0.20 to q, unconnected to everything -> relevant enough to beat C, unreachable
    coords = {
        "q": [1.0, 0.0, 0.0],
        "rule A": [1.0, 0.0, 0.0],
        "rule B": [0.70, 0.7141, 0.0],
        "rule C": [0.10, 0.9950, 0.0],
        "rule D": [0.20, 0.0, 0.9798],
    }

    class VectorBackend:
        def encode(self, texts):
            out = np.array([coords[t] for t in texts], dtype=np.float32)
            return out / np.linalg.norm(out, axis=1, keepdims=True)

    vectors = VectorBackend()
    graph = GraphMemory(backend=vectors, related_threshold=0.60, retrieval="graph",
                        retrieval_seeds=1, retrieval_budget=10000)
    graph.apply_operations([{"type": "ADD", "section": "formulas_and_calculations",
                             "content": name} for name in
                            ("rule A", "rule B", "rule C", "rule D")], step=1)

    ids = {graph.graph.nodes[n]["content"]: n for n in graph.graph.nodes()}
    scores = score_against_query("q", [(n, graph.graph.nodes[n]["content"])
                                       for n in graph.active_nodes()], vectors, graph._embeddings)
    live = set(graph.active_nodes())
    walk = graph_order(scores, lambda n: [m for m in graph._neighbours(n, "related_to")
                                          if m in live], seeds=1)
    relevance = flat_order(scores)
    label = {node: content for content, node in ids.items()}

    check("edges exist only where similarity put them",
          sorted(label[n] for n in graph._neighbours(ids["rule B"], "related_to"))
          == ["rule A", "rule C"],
          str([label[n] for n in graph._neighbours(ids["rule B"], "related_to")]))
    check("relevance alone ranks the unconnected rule above the far neighbour",
          [label[n] for n in relevance] == ["rule A", "rule B", "rule D", "rule C"],
          str([label[n] for n in relevance]))
    check("the walk surfaces the far neighbour and demotes the unconnected rule",
          [label[n] for n in walk] == ["rule A", "rule B", "rule C", "rule D"],
          str([label[n] for n in walk]))
    check("every rule still appears in the walk order, so a sparse graph can fill its budget",
          set(walk) == set(relevance))

    # Budgets. The ceiling covers the whole rendering -- legend and relation lines included --
    # because budgeting only the rule text would hand the graph arm its annotations for free.
    for budget in (60, 120, 400):
        graph.retrieval_budget = budget
        rendered = graph.render_for_generator("q")
        check(f"graph rendering fits a {budget}-token budget",
              utils_module.count_tokens(rendered) <= budget or
              len([ln for ln in rendered.split("\n") if ln.startswith("[")]) == 1,
              f"{utils_module.count_tokens(rendered)} tokens")

    graph.retrieval_budget = 10000

    def relation_lines(text):
        # The legend always names related_to, so match the indented per-rule lines only.
        return [ln.strip() for ln in text.split("\n") if ln.startswith("    related_to:")]

    check("relations are rendered when both ends are selected",
          relation_lines(graph.render_for_generator("q")))

    # EXPERIMENT_FORMULA.md 7 turns on this number, so it has to be recorded, not asserted.
    # Budget set to exactly what the walk's top three cost, so relevance (A, B, D) and the
    # walk (A, B, C) must diverge on the third rule.
    graph.retrieval_budget = utils_module.count_tokens(graph._render(
        with_legend=True, only={ids["rule A"], ids["rule B"], ids["rule C"]}))
    before = graph.structural_stats()["retrieval_calls"]
    for _ in range(3):
        graph.render_for_generator("q")
    walk_stats = graph.structural_stats()
    check("the run records how much the walk added over relevance alone",
          walk_stats["retrieval_calls"] - before == 3 and walk_stats["walk_only_share"] > 0,
          f"calls={walk_stats['retrieval_calls'] - before} "
          f"walk_only_share={walk_stats['walk_only_share']:.2f} "
          f"retrieved_mean={walk_stats['retrieved_rules_mean']:.1f}")
    graph.retrieval_budget = 10000
    check("a relation pointing at an unselected rule is not rendered",
          relation_lines(graph._render(with_legend=True, only={ids["rule A"]})) == [],
          "a dangling reference tells the model to consider a rule it cannot read")

    flat_memory = create_memory("text", None, backend=vectors, retrieval="flat",
                                retrieval_budget=400)
    flat_memory.apply_operations([{"type": "ADD", "section": "formulas_and_calculations",
                                   "content": name} for name in
                                  ("rule A", "rule B", "rule C", "rule D")], step=1)
    flat_rendered = flat_memory.render_for_generator("q")
    check("flat retrieval fits the same budget",
          utils_module.count_tokens(flat_rendered) <= 400,
          f"{utils_module.count_tokens(flat_rendered)} tokens")
    check("flat retrieval keeps playbook order, so relevance never reorders what is read",
          [ln for ln in flat_rendered.split("\n") if ln.startswith("[")] ==
          [ln for ln in flat_memory.snapshot().split("\n")
           if ln.startswith("[") and any(f":: {c}" in ln for c in
                                         ("rule A", "rule B", "rule C", "rule D"))])

    check("retrieval off renders exactly what it always did",
          create_memory("text", flat_memory.snapshot()).render_for_generator("q")
          == flat_memory.snapshot())
    try:
        create_memory("text", None, retrieval="graph")
        check("a flat playbook refuses graph retrieval", False, "no error raised")
    except ValueError:
        check("a flat playbook refuses graph retrieval", True)

    check("fit_to_budget always renders at least one rule, however far over budget",
          fit_to_budget(["only"], lambda item, picked: 10 ** 6,
                        lambda keep: "over budget " * 20, 10) == ["only"])

    print("\n[2] full offline loop, all three arms")
    results = {}
    for kind in ("text", "graph"):
        for key in SEEN:
            SEEN[key].clear()
        with tempfile.TemporaryDirectory() as tmp:
            ace_system = ACE(api_provider="stub", generator_model="m", reflector_model="m",
                             curator_model="m", memory_kind=kind, num_ctx=32768, seed=0,
                             task_name="formula")
            check(f"{kind}: finance prompts selected", ace_system.prompts is FINANCE_PROMPTS)
            out = ace_system.run(
                mode="offline", train_samples=train[:6], val_samples=val[:2],
                test_samples=test[:2], data_processor=processor,
                config={"task_name": "formula", "mode": "offline", "save_dir": tmp,
                        "num_epochs": 1, "max_num_rounds": 1, "curator_frequency": 2,
                        "eval_steps": 6, "save_steps": 10, "test_workers": 1,
                        "skip_initial_test": True, "memory": kind, "seed": 0,
                        "num_ctx": 32768},
            )
            results[kind] = out
            run_dir = next(os.path.join(tmp, d) for d in os.listdir(tmp))
            for artifact in ("run_config.json", "final_playbook.txt",
                             "final_test_predictions.jsonl", "train_results.json"):
                check(f"{kind}: wrote {artifact}",
                      os.path.exists(os.path.join(run_dir, artifact)))
            with open(os.path.join(run_dir, "run_config.json"), encoding="utf-8") as fh:
                run_config = json.load(fh)
            check(f"{kind}: run_config records the prompt set",
                  run_config.get("prompt_set") == "finance", str(run_config.get("prompt_set")))
            rows = [json.loads(x) for x in
                    open(os.path.join(run_dir, "final_test_predictions.jsonl"),
                         encoding="utf-8").read().splitlines() if x.strip()]
            check(f"{kind}: per-problem predictions written", len(rows) == 2, str(len(rows)))
            check(f"{kind}: prediction rows carry is_correct",
                  all("is_correct" in r and "index" in r for r in rows))

        gen_prompt = SEEN["generator_prompts"][0]
        check(f"{kind}: generator prompt is the finance one",
              "You are an analysis expert" in gen_prompt)
        question_body = gen_prompt.split("**Question:**", 1)[-1].lstrip()
        check(f"{kind}: the question is under the Question heading",
              question_body.startswith(train[0]["question"][:40]),
              question_body[:80])
        refl_prompt = SEEN["reflector_prompts"][0]
        check(f"{kind}: reflector saw expected-vs-got feedback",
              "Expected answer:" in refl_prompt)
        check(f"{kind}: reflector prompt defines the tags",
              "causal impact on THIS attempt" in refl_prompt)
        cur_prompt = SEEN["curator_prompts"][0]
        check(f"{kind}: curator prompt requires refines", '"refines": null' in cur_prompt)

    print("\n[2b] the loop with grow-and-refine and retrieval switched on")
    # [2] runs the pre-2026-08-06 configuration. This runs the one the next experiment uses,
    # because retrieval makes the memory a function of the question and that changes the shape
    # of the eval path: evaluate_test_set can no longer render once before the worker fan-out.
    budget = 300
    for kind, retrieval in (("text", "flat"), ("graph", "graph")):
        for key in SEEN:
            SEEN[key].clear()
        with tempfile.TemporaryDirectory() as tmp:
            ace_system = ACE(api_provider="stub", generator_model="m", reflector_model="m",
                             curator_model="m", memory_kind=kind, num_ctx=32768, seed=0,
                             task_name="formula", dedup=True, dedup_threshold=0.90,
                             retrieval=retrieval, retrieval_budget=budget, retrieval_seeds=3)
            ace_system.run(
                mode="offline", train_samples=train[:6], val_samples=val[:2],
                test_samples=test[:2], data_processor=processor,
                config={"task_name": "formula", "mode": "offline", "save_dir": tmp,
                        "num_epochs": 1, "max_num_rounds": 1, "curator_frequency": 2,
                        "eval_steps": 6, "save_steps": 10, "test_workers": 1,
                        "skip_initial_test": True, "memory": kind, "seed": 0,
                        "num_ctx": 32768},
            )
            run_dir = next(os.path.join(tmp, d) for d in os.listdir(tmp))
            with open(os.path.join(run_dir, "run_config.json"), encoding="utf-8") as fh:
                run_config = json.load(fh)
            check(f"{kind}/{retrieval}: run_config records the treatment",
                  run_config.get("retrieval") == retrieval
                  and run_config.get("retrieval_budget") == budget
                  and run_config.get("dedup") is True, str(run_config.get("retrieval")))
            check(f"{kind}/{retrieval}: per-problem predictions still written",
                  os.path.exists(os.path.join(run_dir, "final_test_predictions.jsonl")))

        rendered = ace_system.memory.render_for_generator(train[0]["question"])
        check(f"{kind}/{retrieval}: the generator's memory stays inside the budget",
              utils_module.count_tokens(rendered) <= budget,
              f"{utils_module.count_tokens(rendered)} tokens")
        # Against the un-retrieved *generator* rendering, not snapshot(): the graph arm's
        # legend is part of what the generator reads and is charged to the same budget, so
        # comparing with the legend-less snapshot would flag the legend as growth.
        check(f"{kind}/{retrieval}: retrieval never shows more than injecting everything did",
              utils_module.count_tokens(rendered)
              <= utils_module.count_tokens(ace_system.memory.render_for_generator(None)),
              "retrieved rendering is larger than the whole memory")
        # The eval path renders per sample now, so a fixed string would silently mean every
        # test question saw the memory selected for the *first* one.
        render_fn = ace_system._render_for_sample()
        check(f"{kind}/{retrieval}: the eval path renders per sample",
              render_fn is not None and callable(render_fn))

    print("\n[3] cross-arm invariants")
    check("both arms completed", set(results) == {"text", "graph"})
    check("both arms produced a final test result",
          all("final_test_results" in r for r in results.values()))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("finance wiring is sound")
    return 0


if __name__ == "__main__":
    sys.exit(main())
