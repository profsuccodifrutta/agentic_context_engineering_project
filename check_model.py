#!/usr/bin/env python3
"""
Verify a candidate generator model can actually drive the ACE loop, before probing with it.

Swapping in a smaller generator to create headroom only works if its extra failures are
*reasoning* failures. Three ways that goes wrong silently, each checked here:

  1. The model cannot hold the output schema. Every prompt in this repo demands a bare JSON
     object; a model that wraps it in prose produces failures the reflector cannot diagnose
     and a curator that returns no operations. That is the same dead end as too little
     headroom, reached from the other side.
  2. The model is a reasoning model and keeps thinking. Qwen3 and R1 think by default.
     --think off sends Ollama's switch, but an Ollama build predating the flag ignores it
     without erroring, and the run then spends its output budget on hidden reasoning.
  3. The model does not exist on the endpoint, or answers under a different name.

    uv run python check_model.py --model qwen3.5:2b --think off

Talks to the LLM endpoint only. No Docker, no dataset. Takes about a minute.
"""
import argparse
import json
import sys

from llm import timed_llm_call
from utils import extract_answer, initialize_clients

# The real generator prompt's shape, minus the playbook: JSON out, code inside a field.
GEN_PROMPT = """You must output exactly one valid JSON object with exactly these fields:

{
  "reasoning": "<string>",
  "bullet_ids": [],
  "final_answer": "<the Python function, as a string>"
}

Output only the JSON object. Do not output markdown or triple backticks. Do not output any
text before or after the JSON.

Write a Python function that returns the number of vowels in a string.

Your solution should pass the following tests:
assert count_vowels("hello") == 2
assert count_vowels("xyz") == 0
"""

# A harder problem, to read off roughly where the model sits. One sample is not a benchmark;
# it only separates "can code at all" from "cannot".
HARD_PROMPT = GEN_PROMPT.replace(
    'Write a Python function that returns the number of vowels in a string.\n\n'
    'Your solution should pass the following tests:\n'
    'assert count_vowels("hello") == 2\n'
    'assert count_vowels("xyz") == 0\n',
    'Write a Python function that finds the length of the longest palindromic subsequence '
    'of a string.\n\nYour solution should pass the following tests:\n'
    'assert lps("bbbab") == 4\nassert lps("cbbd") == 2\n'
)


def run(client, model, prompt, call_id, think, num_ctx, seed, max_tokens,
        mechanism="reasoning_effort"):
    try:
        text, info = timed_llm_call(
            client, "ollama", model, prompt, role="generator", call_id=call_id,
            max_tokens=max_tokens, num_ctx=num_ctx, seed=seed, think=think,
            think_mechanism=mechanism,
            # Fail fast: this is a check. Retrying would mask the behaviour being measured.
            retries_on_timeout=1, retries_on_empty=1,
        )
        return text, info
    except Exception as e:
        print(f"    call raised: {type(e).__name__}: {str(e)[:200]}")
        return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="candidate model, e.g. qwen3.5:2b")
    ap.add_argument("--think", choices=["default", "on", "off"], default="off",
                    help="matches the run.py flag; 'off' for a reasoning model")
    ap.add_argument("--num_ctx", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--think_mechanism", choices=["think", "reasoning_effort", "both", "all"],
                    default="all",
                    help="how to ask the server to stop thinking. 'all' tries each in turn and "
                         "reports which one the server honours")
    args = ap.parse_args()

    think = {"default": None, "on": True, "off": False}[args.think]
    client, _, _ = initialize_clients("ollama")
    failures, notes = [], []

    print(f"\nmodel={args.model}  think={args.think}  num_ctx={args.num_ctx}  seed={args.seed}")

    # ---- 1/2. does the model answer, and does any mechanism actually stop the thinking --
    # Tried in turn rather than assumed: `think` in extra_body was measured NOT to work
    # against this endpoint, and a server that does not recognise a key ignores it silently.
    mechanisms = (["think", "reasoning_effort", "both"] if args.think_mechanism == "all"
                  else [args.think_mechanism])
    if think is None:
        mechanisms = mechanisms[:1]

    print("\n[1/2] model responds, and reasoning is suppressed")
    text = None
    working_mechanism = None
    for mechanism in mechanisms:
        attempt_text, attempt_info = run(client, args.model, GEN_PROMPT,
                                         f"check_model_easy_{mechanism}", think,
                                         args.num_ctx, args.seed, args.max_tokens, mechanism)
        if attempt_text is None:
            print(f"    {mechanism:<16} no response")
            continue
        chars = attempt_info.get("reasoning_chars", 0)
        tokens = attempt_info.get("response_num_tokens")
        secs = attempt_info.get("call_time", 0)
        verdict = "reasoning suppressed" if not chars else f"{chars} chars of reasoning"
        print(f"    {mechanism:<16} {tokens} tok  {secs:.1f}s  -> {verdict}")
        if text is None:
            text = attempt_text
        if think is not None and not chars and working_mechanism is None:
            working_mechanism = mechanism
            text = attempt_text
            break

    if text is None:
        print("    FAIL  no response from any mechanism. Check the model name against the "
              "endpoint's /v1/models.")
        print("\nNOT USABLE.")
        return 1

    if think is False and working_mechanism is None:
        failures.append(
            f"no mechanism stopped the thinking ({', '.join(mechanisms)} all tried) -- this "
            f"model reasons unconditionally on this server. It will spend its output budget "
            f"before answering, and return nothing at all when the budget runs out. Use a "
            f"non-reasoning model."
        )
        print("    FAIL  none of the mechanisms worked")
    elif working_mechanism:
        print(f"    ok    '{working_mechanism}' is honoured by this server")
        if working_mechanism != "reasoning_effort":
            notes.append(f"llm.py defaults to think_mechanism='reasoning_effort'; this server "
                         f"honours '{working_mechanism}' instead, so change the default")

    # ---- 3. the JSON schema holds ------------------------------------------------------
    print("\n[3] output schema")
    try:
        parsed = json.loads(text.strip())
        schema_ok = isinstance(parsed, dict) and "final_answer" in parsed
    except json.JSONDecodeError:
        parsed, schema_ok = None, False

    if schema_ok:
        print("    ok    response is a bare JSON object with final_answer")
    else:
        # extract_answer's salvage path is what the real loop relies on, so a model that
        # needs it is degraded but not disqualified.
        salvaged = extract_answer(text)
        if salvaged.startswith("raise ValueError("):
            failures.append(
                "response is neither valid JSON nor salvageable by extract_answer -- this "
                "model's failures would be format failures, which teach the reflector nothing"
            )
            print("    FAIL  unparseable even by the fallback extractor")
            print(f"    raw: {text[:200]!r}")
        else:
            notes.append("response was not bare JSON but extract_answer salvaged it; expect a "
                         "raised no_answer rate. Check [5] output integrity on the probe run.")
            print("    warn  not bare JSON, but the fallback extractor recovered it")

    # ---- 4. rough capability read -------------------------------------------------------
    print("\n[4] rough capability (one easy + one hard problem, not a benchmark)")
    easy_code = extract_answer(text)
    print(f"    easy: {'wrote code' if 'def ' in easy_code else 'NO FUNCTION'} "
          f"({len(easy_code)} chars)")
    hard_text, hard_info = run(client, args.model, HARD_PROMPT, "check_model_hard",
                               think, args.num_ctx, args.seed, args.max_tokens,
                               working_mechanism or mechanisms[0])
    if hard_text is None:
        # A reasoning model that answers the easy problem and returns nothing on the hard one
        # is the specific failure this whole check exists to catch: the extra difficulty went
        # into hidden reasoning until the budget ran out, so the harder problems -- exactly
        # the ones meant to create headroom -- become no_answer instead of wrong answers.
        failures.append(
            "returned nothing on the harder problem while answering the easy one -- its "
            "failures on hard problems would be empty responses, which teach the reflector "
            "nothing"
        )
        print("    hard: NO RESPONSE")
    else:
        hard_code = extract_answer(hard_text)
        print(f"    hard: {'wrote code' if 'def ' in hard_code else 'NO FUNCTION'} "
              f"({len(hard_code)} chars, {hard_info.get('response_num_tokens')} tokens)")
    print("    Neither is scored here -- the probe run measures accuracy. Aim for a "
          "baseline near 0.60; below ~0.40 the failures get too chaotic to learn from.")

    # ---- verdict --------------------------------------------------------------------------
    print("\n" + "=" * 72)
    for note in notes:
        print(f"note: {note}")
    if failures:
        print("NOT USABLE as a generator:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"{args.model} can drive the loop. Run the probe next:")
    print("  uv run python -m eval.mbpp_code_exec.run --mode offline --save_path results/mbpp/probes/smoke \\")
    print(f"    --memory text --generator_model {args.model} --think {args.think} \\")
    print(f"    --num_ctx {args.num_ctx} --seed {args.seed} --skip_initial_test --eval_steps 30 \\")
    print("    --config_path ./eval/mbpp_code_exec/data/probe_config.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
