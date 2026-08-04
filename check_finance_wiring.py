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
    from playbook_utils import extract_json_from_text
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
