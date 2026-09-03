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
import re
import sys
import tempfile

import ace.native_deps  # noqa: F401  -- must precede sentence-transformers imports

import ace.core.curator as curator_module
import ace.core.generator as generator_module
import ace.core.reflector as reflector_module
import compare_arms
import utils as utils_module
from ace.ace import ACE
from ace.prompts import FINANCE_PROMPTS
from eval.finance.data_processor import (_FINER_SUBQUESTION_STEM, DataProcessor,
                                         finer_retrieval_query, finer_tag_counts, load_data)

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
        # against every target so the failure path (reflect -> curate) actually runs. A finer
        # prompt gets a four-tag list instead, for the same reason -- wrong, but the right
        # shape, so the positional comparison and its per-tag feedback both run.
        answer = ("StubTagA,StubTagB,StubTagC,StubTagD"
                  if "US GAAP tags" in prompt else "$1,234.50")
        return json.dumps({"reasoning": "stub", "bullet_ids": cited,
                           "final_answer": answer}), info

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
        # The id shape used to be pinned to what the harness emits (\w+-\d{5}), and that leaked:
        # the model invents ids rather than copying the one it was shown. 82 rules across the
        # nine pre-registered formula playbooks kept a prefix under these forms. This case
        # asserted the opposite until 2026-08-16; the expectation was what was wrong.
        ("an invented id with a suffix",
         "[calc-00047_new] helpful=15 harmful=0 :: Convert percentages first.",
         "Convert percentages first.", 1),
        ("an invented id with too few digits",
         "[calc-13] Convert percentages first.", "Convert percentages first.", 1),
        ("an invented id with no digits at all",
         "[ggm-r] Convert percentages first.", "Convert percentages first.", 1),
        # The negative that constrains how far the bare-bracket tier may be broadened: one
        # text-arm rule is an entire formula wrapped in brackets, and stripping it deletes the
        # rule. Spaces in the tag are what disqualify it, so the tier requires an id shape.
        ("a whole rule wrapped in brackets is never stripped",
         "[IRP Theorem: To calculate the forward rate F, use interest rate parity.]",
         "[IRP Theorem: To calculate the forward rate F, use interest rate parity.]", 0),
        ("a bracketed clause with no hyphen is left alone",
         "[t 1] Convert percentages first.", "[t 1] Convert percentages first.", 0),
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
    # re-implementation of what its competitor had been stripped of (docs/background/EXPERIMENT_FORMULA.md sec 10).
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
        # Used by [1g] only: the refinement of A, so the stub backend can embed it. Placed
        # here because the fixture is shared and the coordinates have to stay in one table.
        "rule A, clarified": [1.0, 0.0, 0.0],
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

    # docs/background/EXPERIMENT_FORMULA.md 7 turns on this number, so it has to be recorded, not asserted.
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

    print("\n[1g] co-citation edges")
    # related_to is cosine between two rules' texts while retrieval scores rules by cosine
    # between the question and each rule -- nearly the same signal, so the walk lands on rules
    # relevance would have picked anyway. Measured on the pre-registered runs, a random
    # rewiring at matched density surfaces 2.3-3.6x MORE novel rules than the real graph, so
    # walk_only_share never established that those edges carried information.
    #
    # Co-citation is the relation a flat playbook provably cannot hold: related_to is a
    # function of the two rules' content, so a list plus an embedding index reproduces it,
    # while "the generator named these together" exists only in accumulated history. The case
    # below is that claim in miniature -- same fixture as [1f], but the association comes from
    # usage rather than similarity, and it reverses the ranking in the opposite direction.
    cocite_graph = GraphMemory(backend=vectors, related_threshold=0.60, retrieval="graph",
                               retrieval_seeds=1, retrieval_budget=10000,
                               cocite=True, cocite_min_opportunity=3, cocite_min_rate=0.30)
    cocite_graph.apply_operations([{"type": "ADD", "section": "formulas_and_calculations",
                                    "content": name} for name in
                                   ("rule A", "rule B", "rule C", "rule D")], step=1)
    co_ids = {cocite_graph.graph.nodes[n]["content"]: n for n in cocite_graph.graph.nodes()}
    co_label = {node: content for content, node in co_ids.items()}

    check("similarity edges are not written when co-citation replaces them",
          cocite_graph._neighbours(co_ids["rule B"], "related_to") == [],
          "emitting both would make a graph+cocite arm differ from graph in two places")

    # D is the rule relevance ranks third and similarity never connects to anything. Cite it
    # with A often enough to clear the gate, and nothing else together.
    all_four = [co_ids[n] for n in ("rule A", "rule B", "rule C", "rule D")]
    for step in range(1, 5):
        cocite_graph.record_generation(all_four, [co_ids["rule A"], co_ids["rule D"]], step=step)

    check("a co-citation edge forms only where the model cited both",
          [co_label[n] for n in cocite_graph._neighbours(co_ids["rule A"], "co_cited")]
          == ["rule D"],
          str([co_label[n] for n in cocite_graph._neighbours(co_ids["rule A"], "co_cited")]))
    check("a pair rendered together but never cited together stays unlinked",
          cocite_graph._neighbours(co_ids["rule B"], "co_cited") == [])
    check("citations for rules that were not rendered are ignored",
          (lambda before: (cocite_graph.record_generation([co_ids["rule A"]], ["ghost-00001"], 9)
                           or len(cocite_graph._cocite_pairs) == before))(
              len(cocite_graph._cocite_pairs)),
          "an invented id would create an edge from evidence that never existed")

    co_scores = score_against_query(
        "q", [(n, cocite_graph.graph.nodes[n]["content"]) for n in cocite_graph.active_nodes()],
        vectors, cocite_graph._embeddings)
    co_live = set(cocite_graph.active_nodes())
    co_walk = graph_order(co_scores,
                          lambda n: [m for m in cocite_graph._neighbours(n, "co_cited")
                                     if m in co_live], seeds=1)
    check("the co-citation walk promotes the co-used rule above the merely similar one",
          [co_label[n] for n in co_walk][:2] == ["rule A", "rule D"],
          str([co_label[n] for n in co_walk]))
    check("co_cited relations are rendered, and related_to is not",
          "co_cited:" in cocite_graph._render(with_legend=True)
          and "related_to" not in cocite_graph._render(with_legend=True))

    # Edge inheritance. refine() already merges an absorbed rule's helpful/harmful counters
    # into its survivor; this is the same policy for the state a flat list cannot hold. Without
    # it every clarification re-enters the graph orphaned and throws away the behavioural
    # evidence the run paid for.
    weights_before = list(cocite_graph._cocite_pairs[
        cocite_graph._cocite_key(co_ids["rule A"], co_ids["rule D"])])
    cocite_graph.apply_operations([{"type": "ADD", "section": "formulas_and_calculations",
                                    "content": "rule A, clarified",
                                    "refines": co_ids["rule A"]}], step=10)
    heir = next(n for n in cocite_graph.graph.nodes()
                if cocite_graph.graph.nodes[n]["content"] == "rule A, clarified")
    inherited = cocite_graph._cocite_pairs.get(
        cocite_graph._cocite_key(heir, co_ids["rule D"]))
    check("a refinement inherits the rule it supersedes' co-citation history",
          inherited is not None and inherited[:2] == weights_before[:2],
          f"heir={inherited} ancestor={weights_before}")
    check("the heir is connected where its ancestor was",
          co_ids["rule D"] in cocite_graph._neighbours(heir, "co_cited"))

    # Inheritance copies the ancestor's pairs onto the heir without deleting the originals,
    # so the raw store keeps pairs whose nodes no longer render. The walk and the relation
    # lines both drop those already; the reported numbers must too, or cocite_max_degree --
    # the hub gate this arm is judged on -- counts edges the generator can never follow, and
    # cocite_pair_coverage divides a store-wide numerator by a live-node denominator.
    co_stats = cocite_graph.structural_stats()
    co_live_now = set(cocite_graph.active_nodes())
    check("co-citation stats count live nodes only",
          len(cocite_graph._cocite_pairs) > co_stats["cocite_pairs_tracked"]
          and co_stats["cocite_pairs_tracked"]
          == sum(1 for a, b in cocite_graph._cocite_pairs
                 if a in co_live_now and b in co_live_now),
          f"store={len(cocite_graph._cocite_pairs)} reported={co_stats['cocite_pairs_tracked']}")
    check("co-citation pair coverage cannot exceed 1.0",
          0.0 <= co_stats["cocite_pair_coverage"] <= 1.0,
          str(co_stats["cocite_pair_coverage"]))
    check("no reported co-citation edge touches a withdrawn rule",
          co_stats["cocite_edges"]
          == sum(1 for a, b in cocite_graph._cocite_edge_pairs()
                 if a in co_live_now and b in co_live_now)
          and co_stats["cocite_max_degree"] <= max(len(co_live_now) - 1, 0),
          f"edges={co_stats['cocite_edges']} raw={len(cocite_graph._cocite_edge_pairs())} "
          f"max_degree={co_stats['cocite_max_degree']} live={len(co_live_now)}")

    # The store is the source of truth and edges are a view over it, so persistence has one
    # thing to round-trip rather than two to keep in sync.
    with tempfile.TemporaryDirectory() as tmp:
        saved = os.path.join(tmp, "cocite_playbook.txt")
        cocite_graph.save(saved)
        reloaded = GraphMemory.load(saved, backend=vectors)
        check("co-citation survives save and reload",
              reloaded.cocite
              and reloaded._cocite_pairs == cocite_graph._cocite_pairs
              and reloaded._cocite_edge_pairs() == cocite_graph._cocite_edge_pairs())

    clone = cocite_graph.copy()
    clone.record_generation(all_four, all_four, step=99)
    check("copy() does not alias the pair store",
          clone._cocite_pairs != cocite_graph._cocite_pairs,
          "entries are mutated in place by decay, so a shallow copy would let best track current")

    off = GraphMemory(backend=vectors, retrieval="graph", retrieval_budget=10000)
    off.apply_operations([{"type": "ADD", "section": "formulas_and_calculations",
                           "content": name} for name in ("rule A", "rule B")], step=1)
    off.record_generation(list(off.graph.nodes()), list(off.graph.nodes()), step=1)
    check("record_generation is inert when co-citation is off",
          off._cocite_pairs == {} and off.structural_stats()["cocite_edges"] == 0,
          "every arm receives the call; only the representation differs")

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

    print("\n[1h] curator-declared refines can be turned off without touching de-duplication")
    # `refines` is unary, so it is not the graph capability under test, and an experiment may
    # want the arms to differ in pairwise representation and nothing else (docs/background/EXPERIMENT_MBPP.md
    # sec 3). The flag has to skip *only* the curator's declaration: de-duplication writes its
    # own refines edges with origin="dedup" and that mechanism is symmetric across arms, so
    # disabling it here would silently remove grow-and-refine from the graph arm alone. A
    # regression in either direction invalidates every run made under the flag.
    from ace.memory import GraphMemory

    two_rules = [
        {"type": "ADD", "section": "general", "content": "Always validate inputs before use."},
        {"type": "ADD", "section": "general", "content": "Validate every input before using it."},
    ]

    for enabled, want_curator, want_active in ((True, 1, 1), (False, 0, 2)):
        mem = GraphMemory(curator_refines=enabled)
        first = mem.apply_operations(two_rules[:1], step=1)
        superseding = dict(two_rules[1], refines=first[0])
        mem.apply_operations([superseding], step=2)
        stats = mem.structural_stats()
        check(f"curator_refines={enabled}: {want_curator} curator refines edge(s)",
              stats["refines_edges_curator"] == want_curator,
              f"got {stats['refines_edges_curator']}")
        check(f"curator_refines={enabled}: {want_active} rule(s) still rendered",
              stats["active_node_count"] == want_active,
              f"got {stats['active_node_count']}")

    # The flag off must leave grow-and-refine intact, or the graph arm loses a mechanism the
    # text arm keeps and the arms differ in two places rather than none.
    dedup_mem = GraphMemory(curator_refines=False, backend=StubBackend({
        two_rules[0]["content"]: 0.0, two_rules[1]["content"]: near}))
    dedup_mem.apply_operations(two_rules, step=1)
    dedup_mem.refine(dedup_mem.backend, 0.90)
    dedup_stats = dedup_mem.structural_stats()
    check("curator_refines=False: de-duplication still writes its own refines edge",
          dedup_stats["refines_edges_dedup"] == 1 and dedup_stats["refines_edges_curator"] == 0,
          f"curator={dedup_stats['refines_edges_curator']} dedup={dedup_stats['refines_edges_dedup']}")

    print("\n[1i] finer: data processor and prompt binding")
    # `finer` rows carry no "Instruction:"/"Input:" markers, so parse_instruction_and_input
    # falls through to its `return "", all_context` branch: `context` is EMPTY and `question`
    # carries the whole ~1750-token payload. That is the shape the generator template has to
    # bind, and an empty context is exactly the case where a positional bug hides -- swapping
    # two fields where one of them is "" produces a prompt that still looks plausible.
    finer_processor = DataProcessor(task_name="finer")
    with open("./eval/finance/data/probe_config.json", encoding="utf-8") as fh:
        finer_cfg = json.load(fh)["finer"]
    f_train = finer_processor.process_task_data(load_data(finer_cfg["train_data"]))
    f_val = finer_processor.process_task_data(load_data(finer_cfg["val_data"]))
    f_test = finer_processor.process_task_data(load_data(finer_cfg["test_data"]))
    check("finer probe splits load",
          len(f_train) == 50 and len(f_val) == 20 and len(f_test) == 20,
          f"{len(f_train)}/{len(f_val)}/{len(f_test)}")
    check("the tag list is in the question, and context is empty",
          all(s["context"] == "" and "US GAAP tags" in s["question"] for s in f_train))
    check("every target is four comma-separated tags",
          all(len(s["target"].split(",")) == 4 for s in f_train))

    f_target = f_train[0]["target"]
    three_of_four = ",".join(f_target.split(",")[:3] + ["WrongTag"])
    f_feedback = {}
    f_correct = utils_module.check_answer(finer_processor, three_of_four, f_target,
                                          f_train[0], feedback=f_feedback)
    check("answer_is_correct is the all-four conjunction, so 3 of 4 is False",
          f_correct is False)
    check("feedback names which tag went wrong, not just that the row did",
          f_feedback.get("tags_correct") == "3 of 4" and "tag 4: WRONG" in f_feedback.get("per_tag", ""),
          str(f_feedback.get("per_tag"))[:110])
    f_rendered = utils_module.format_environment_feedback(False, f_feedback)
    check("environment feedback renders the per-tag detail",
          "per_tag:" in f_rendered and "tags_correct:" in f_rendered, f_rendered[:110])

    print("\n[1j] finer: extraction of the comma-separated list")
    # The envelopes a model can plausibly wrap a four-item list in. The last two are the live
    # risks: unwrap_quoted_source and the one-element-array unwrap are the code paths that
    # could mangle a list, and a four-element array is what a model asked for four tags is
    # most likely to invent. Anything that comes back must still score 4 of 4.
    tags4 = "AmortizationOfIntangibleAssets,DebtInstrumentFaceAmount,InterestExpense,RepaymentsOfDebt"
    envelopes = [
        ("plain", json.dumps({"reasoning": "r", "final_answer": tags4})),
        ("spaced", json.dumps({"reasoning": "r", "final_answer": tags4.replace(",", ", ")})),
        ("one-element array", json.dumps({"reasoning": "r", "final_answer": [tags4]})),
        ("double-stringified", json.dumps({"reasoning": "r", "final_answer": json.dumps(tags4)})),
        ("four-element array",
         json.dumps({"reasoning": "r", "final_answer": tags4.split(",")})),
        ("fenced", "```json\n" + json.dumps({"reasoning": "r", "final_answer": tags4}) + "\n```"),
    ]
    for label, raw in envelopes:
        extracted = utils_module.extract_answer(raw)
        count, total, _ = finer_tag_counts(str(extracted), tags4)
        check(f"{label}: extracts to all four tags", count == 4 and total == 4,
              f"{count}/{total} from {str(extracted)[:70]!r}")

    print("\n[1k] finer: the two pre-registered metrics, and that they agree where they must")
    # docs/EXPERIMENT_FINER.md sec 5 pre-registers per-tag as primary and all-four as the
    # authors' secondary. They are two readings of the same comparison and they must never
    # drift apart: freeze the disagreement that is intended and the agreement that is required.
    check("evaluate_accuracy is per-tag partial credit",
          abs(finer_processor.evaluate_accuracy([three_of_four], [f_target], [f_train[0]]) - 0.75) < 1e-9,
          str(finer_processor.evaluate_accuracy([three_of_four], [f_target], [f_train[0]])))
    check("answer_is_correct and evaluate_accuracy disagree on 3 of 4, on purpose",
          f_correct is False
          and finer_processor.evaluate_accuracy([three_of_four], [f_target], [f_train[0]]) > 0)
    check("compare_arms scores a row exactly as the processor does",
          abs(compare_arms._score_finer_per_tag(
              {"final_answer": three_of_four, "target": f_target}) - 0.75) < 1e-9)
    check("a perfect row scores 1.0 under both",
          compare_arms._score_finer_per_tag({"final_answer": f_target, "target": f_target}) == 1.0
          and utils_module.check_answer(finer_processor, f_target, f_target, f_train[0]) is True)

    print("\n[1l] finer: the comparison is positional")
    # pred[i] against gold[i]. The right four tags in the wrong order score near zero, which
    # turns a formatting slip into a total loss on the row. That is the authors' metric and is
    # not repaired here; it is frozen so the pre-registration's claim about it stays true.
    reversed_tags = ",".join(reversed(f_target.split(",")))
    r_count, r_total, _ = finer_tag_counts(reversed_tags, f_target)
    check("reordering the same four tags does not score as correct",
          r_count < r_total, f"{r_count}/{r_total}")
    short_count, short_total, _ = finer_tag_counts(",".join(f_target.split(",")[:2]), f_target)
    check("a short answer is padded, never truncated to look perfect",
          short_total == 4 and short_count == 2, f"{short_count}/{short_total}")

    print("\n[1m] finer: partial credit reaches the reflector, and only the reflector")
    # `answer_is_correct` on finer is all-four-exact, so a 3-of-4 row is reported as a total
    # failure and `_filter_bullet_tags` withdraws the `helpful` tag those three tags earned.
    # On the seed-0 arm that demoted 164 tags, 74 of them on rows scoring 0.5 or 0.75
    # (docs/EXPERIMENT_FINER.md section 10.3). The demotion rule is this fork's own, not the
    # authors' (EXPERIMENT_FORMULA.md section 8 item 2), so respecting partial credit repairs
    # our instrument rather than editing ACE.
    pc_gold = "TagA,TagB,TagC,TagD"
    for pred, want in ((pc_gold, 1.0), ("TagA,TagB,TagC,TagX", 0.75), ("TagW,TagX,TagY,TagZ", 0.0)):
        pc_fb = {}
        finer_processor.answer_is_correct(pred, pc_gold, None, feedback=pc_fb)
        check(f"finer: partial_credit is {want} for {pred.split(chr(44))[3]}",
              abs(pc_fb.get("partial_credit", -1) - want) < 1e-9, str(pc_fb.get("partial_credit")))
    # It is read by the loop, never shown: adding it would change the reflector prompt of a
    # task whose only intended prompt change is to the curator.
    pc_fb = {}
    finer_processor.answer_is_correct("TagA,TagB,TagC,TagX", pc_gold, None, feedback=pc_fb)
    pc_rendered = utils_module.format_environment_feedback(False, pc_fb)
    check("finer: partial_credit is not rendered into the reflector prompt",
          "partial_credit" not in pc_rendered)
    check("finer: tags_correct still is", "tags_correct" in pc_rendered)
    # The demotion itself: a majority-right row keeps `helpful`, a mostly-wrong one does not,
    # and a plain bool behaves exactly as before for every task that scores a row as one answer.
    pc_tags = [{"id": "b1", "tag": "helpful"}]
    for verdict, expected in ((1.0, "helpful"), (0.75, "helpful"), (0.5, "helpful"),
                              (0.25, "neutral"), (0.0, "neutral"),
                              (True, "helpful"), (False, "neutral"), (None, "helpful")):
        got = reflector_module.Reflector._filter_bullet_tags(pc_tags, {"b1"}, verdict)[0]["tag"]
        check(f"finer: attempt_is_correct={verdict!r} keeps the tag as {expected}",
              got == expected, got)

    print("\n[1n] finer: the retrieval query varies per row, and no other task changed")
    # THE CHECK THAT WOULD HAVE CAUGHT THE 2026-08-30 DEFECT. `all-mpnet-base-v2` truncates at
    # 384 tokens (~1,500 characters) and the first ~1,500 characters of every `finer` question
    # are the identical 140-tag list, so every row encoded to the same vector -- pairwise cosine
    # exactly 1.0000 -- and `score_against_query` returned one ranking for every question. The
    # seed-0 arms rendered ONE distinct rule set across all 441 test generations, in both arms,
    # while every gate in EXPERIMENT_FINER.md section 7 passed. Retrieval was inert and nothing
    # said so. See docs/EXPERIMENT_FINER.md section 9, 2026-08-30.
    #
    # **The assertion is on the truncation window, not on total length.** A quarter to a third of
    # `finer` queries are longer than the window (median 1,169 characters, p90 2,156, max 10,196
    # on `finer_val_150`), and that is fine: truncation now discards *later sub-questions* rather
    # than a shared preamble, so what survives still identifies the row. What must never be true
    # again is that the surviving prefix is shared, so that is what is checked -- on the whole
    # split, not on a sample small enough to pass by luck.
    ENCODER_WINDOW_CHARS = 1500  # 384 tokens for all-mpnet-base-v2, measured at ~4 chars/token

    finer_rows = load_data("./eval/finance/data/finer_val_150.jsonl")
    finer_ctx = [r["context"] for r in finer_rows]
    queries = [finer_retrieval_query(c) for c in finer_ctx]
    # Since 2026-09-03 the query is a LIST, one entry per sub-question with the template stripped.
    # Under the old single string a third of rows had sub-questions 3 and 4 truncated away
    # entirely; per entry that is down to a handful, reported below rather than asserted, because
    # the gated property is that no shared prefix survives.
    check("[1n] the query is a list of sub-questions", all(isinstance(q, list) and q for q in queries),
          f"{sum(1 for q in queries if isinstance(q, list))}/{len(queries)} rows returned a list")
    check("[1n] the marker is found in every row",
          all(q != [c] for q, c in zip(queries, finer_ctx)),
          f"{sum(q != [c] for q, c in zip(queries, finer_ctx))}/{len(finer_ctx)} rows carry it")
    joined = ["\n".join(q) for q in queries]
    check("[1n] the query differs per row", len(set(joined)) == len(joined),
          f"{len(set(joined))} distinct of {len(joined)}")
    check("[1n] the full question does NOT differ inside the encoder window",
          len({c[:ENCODER_WINDOW_CHARS] for c in finer_ctx}) == 1,
          "this is the defect: every row shares its first 1,500 characters")
    windows = [tuple(p[:ENCODER_WINDOW_CHARS] for p in q) for q in queries]
    check("[1n] the query DOES differ inside the encoder window, on every row",
          len(set(windows)) == len(windows),
          f"{len(set(windows))} distinct truncated windows of {len(windows)} rows")
    # Overflow is reported, not gated. The gated property is the one above and the one below --
    # distinct windows, no shared template -- exactly as the 2026-08-30 entry argued: "Length is
    # not the property to check; a shared surviving prefix is." Splitting takes overflow from a
    # third of whole rows to a handful of individual sub-questions, and a truncated sub-question
    # still carries its own entity and sentence.
    overflow = sum(1 for q in queries for part in q if len(part) > ENCODER_WINDOW_CHARS)
    print(f"    note  {overflow} of {sum(len(q) for q in queries)} sub-questions exceed the "
          f"{ENCODER_WINDOW_CHARS}-character window (was 32.4% of whole rows before splitting)")
    # The template is what the drifted rules match, so it must be gone from every entry.
    check("[1n] the question template is stripped from every sub-question",
          not any(_FINER_SUBQUESTION_STEM in part for q in queries for part in q),
          "a shared substring in every query is the 2026-08-30 defect one level down")

    # And the end-to-end consequence, through the real processor and a real memory. The rules are
    # built from the task's own tag vocabulary so relevance has something to discriminate on; a
    # fixture of unrelated toy rules would pass this at a rate that says nothing.
    finer_proc = DataProcessor(task_name="finer")
    finer_processed = finer_proc.process_task_data(finer_rows[:25])
    check("[1n] the processor emits retrieval_query on finer",
          all(item.get("retrieval_query") for item in finer_processed))
    vocab = sorted({t for t in re.findall(r"\b(?:[A-Z][a-z0-9]+){2,}[A-Za-z0-9]*\b", finer_ctx[0])
                    if len(t) > 14})[:60]
    retr_mem = create_memory("text", "", retrieval="flat", retrieval_budget=900,
                             retrieval_seeds=5, backend=EmbeddingBackend())
    retr_mem.apply_operations([{"type": "ADD", "section": "context_clues_and_indicators",
                                "content": f"IF the text reports this concept THEN {tag}"}
                               for tag in vocab], step=1)
    rendered_sets = [frozenset(re.findall(r"^\[([\w-]+)\] helpful=",
                                          retr_mem.render_for_generator(ACE._retrieval_query(item)),
                                          re.M))
                     for item in finer_processed]
    distinct = len(set(rendered_sets))
    # The bar is section 7's own: at least 20 distinct sets over 25 generations.
    check("[1n] finer renders a different rule set per question", distinct >= 20,
          f"{distinct} distinct sets over {len(finer_processed)} questions, section 7 requires 20 "
          f"(the defect produced exactly 1 over 441)")

    # formula must be untouched: no key, so _retrieval_query falls through to the question.
    formula_proc = DataProcessor(task_name="formula")
    formula_processed = formula_proc.process_task_data(
        load_data("./eval/finance/data/formula_probe_test.jsonl")[:5])
    check("[1n] formula emits no retrieval_query",
          not any(item.get("retrieval_query") for item in formula_processed))
    check("[1n] formula still scores against its own question",
          all(ACE._retrieval_query(item) == item["question"] for item in formula_processed))
    # And the scores themselves, not just the string: a single query must produce exactly the
    # dict the pre-2026-09-03 one-vector path produced. The list branch changed `encode([query])`
    # into `encode(queries)` plus a max, so this is the assertion that the max over one element
    # is the element and that nothing about a one-query task moved. Float-exact on purpose --
    # "close enough" here would hide a reordering of the ranking.
    from ace.memory.retrieval import score_against_query as _score
    formula_rules = [(f"calc-{i:05d}", c) for i, c in enumerate(
        ["Compute the forward rate from spot and interest differentials",
         "Depreciation is cost minus salvage over useful life",
         "Interest coverage is EBIT divided by interest expense"])]
    backend = EmbeddingBackend()
    formula_question = formula_processed[0]["question"]
    one = _score(formula_question, formula_rules, backend)
    listed = _score([formula_question], formula_rules, backend)
    check("[1n] formula scoring is identical for a string and a one-entry list",
          one == listed, f"{one} vs {listed}")

    print("\n[2c] finer: the full offline loop, both arms")
    # The finer counterpart of [2b]. Everything above is a unit; this is the one that catches
    # signature drift end to end, and it runs on the real probe splits rather than a fixture.
    for kind, retrieval in (("text", "flat"), ("graph", "graph")):
        for key in SEEN:
            SEEN[key].clear()
        with tempfile.TemporaryDirectory() as tmp:
            ace_system = ACE(api_provider="stub", generator_model="m", reflector_model="m",
                             curator_model="m", memory_kind=kind, num_ctx=32768, seed=0,
                             task_name="finer", dedup=True, dedup_threshold=0.90,
                             retrieval=retrieval, retrieval_budget=300, retrieval_seeds=3,
                             memory_kwargs=({"cocite": True, "curator_refines": False}
                                            if kind == "graph" else None))
            ace_system.run(
                mode="offline", train_samples=f_train[:4], val_samples=f_val[:2],
                test_samples=f_test[:2], data_processor=finer_processor,
                config={"task_name": "finer", "mode": "offline", "save_dir": tmp,
                        "num_epochs": 1, "max_num_rounds": 1, "curator_frequency": 2,
                        "eval_steps": 4, "save_steps": 10, "test_workers": 1,
                        "skip_initial_test": True, "memory": kind, "seed": 0,
                        "num_ctx": 32768},
            )
            run_dir = next(os.path.join(tmp, d) for d in os.listdir(tmp))
            preds_path = os.path.join(run_dir, "final_test_predictions.jsonl")
            check(f"finer/{kind}: per-problem predictions written", os.path.exists(preds_path))
            rows = [json.loads(x) for x in
                    open(preds_path, encoding="utf-8").read().splitlines() if x.strip()]
            # Both metrics are recomputed post hoc from these two fields. If either stops being
            # persisted the primary metric becomes unrecoverable after the run.
            check(f"finer/{kind}: predictions carry final_answer and target",
                  all(r.get("final_answer") is not None and r.get("target") is not None
                      for r in rows), str(rows[:1])[:110])
            check(f"finer/{kind}: per-tag score is recoverable from the saved predictions",
                  all(0.0 <= compare_arms._score_finer_per_tag(r) <= 1.0 for r in rows))
        gen_prompt = SEEN["generator_prompts"][0]
        check(f"finer/{kind}: generator prompt is the finance one",
              "You are an analysis expert" in gen_prompt)
        question_body = gen_prompt.split("**Question:**", 1)[-1].lstrip()
        check(f"finer/{kind}: the tag list lands under the Question heading, not the Playbook",
              question_body.startswith(f_train[0]["question"][:40]), question_body[:70])
        check(f"finer/{kind}: the playbook slot never holds the tag list",
              "US GAAP tags" not in gen_prompt.split("**Question:**", 1)[0])
        check(f"finer/{kind}: reflector saw which tag was wrong",
              "per_tag:" in SEEN["reflector_prompts"][0])

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
