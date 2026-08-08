#!/usr/bin/env python3
"""
Find the playbook size at which the curator stops working, before a full arm reaches it.

`--retrieval_budget` bounds what the *generator* reads. The curator reads
`Memory.render_for_curator()`, which is the whole memory with no query and no budget --
deliberately, because it cannot avoid re-adding a rule it cannot see, nor name the rule a
new one supersedes. So the curator prompt grows with the playbook, linearly, for the whole
run. On the 50-step formula probe it went 2,290 -> 6,890 tokens; a 250-step arm projects to
roughly 26,000.

Whether that is fatal depends on the window the server actually serves, which is NOT the
`--num_ctx` asked for: Ollama clamps to the model's own trained context, silently. The
failure is silent too -- the model is left too little output budget, spends it on hidden
reasoning, and returns empty. A run degrades from some step onward and still writes
complete-looking artifacts.

This sends real curator prompts at increasing playbook sizes and reports where the role
stops returning usable operations. It answers one question: does the curator survive to the
end of a 250-step arm, or does it need a smaller memory, a shorter run, or a bigger window.

    uv run python check_curator_ceiling.py --model gemma4:e2b

Talks to the LLM endpoint only. No Docker, no dataset, no training. A few minutes.
"""
import argparse
import os
import sys

import tiktoken

from ace.prompts import get_prompts
from llm import timed_llm_call
from playbook_utils import extract_json_from_text
from utils import initialize_clients

ENC = tiktoken.get_encoding("cl100k_base")

DEFAULT_PLAYBOOK = os.path.join(
    "results", "formula", "probes", "gemma_text",
    "ace_run_20260808_170721_formula_offline", "final_playbook.txt",
)

# Stands in for the reflection and question the curator is given alongside the playbook. Held
# fixed at every size so the only thing varying across the ladder is the playbook.
REFLECTION = (
    "The attempt computed the Sharpe ratio using the arithmetic mean of the returns without "
    "subtracting the risk-free rate, and reported 1.42 where the ground truth is 0.87. The "
    "error is in the numerator: excess return, not raw return. The playbook rule on "
    "annualising volatility was applied correctly and is not implicated."
)
QUESTION_CONTEXT = (
    "Question: Compute the Sharpe ratio for a portfolio with mean annual return 11.2%, "
    "risk-free rate 3.1%, and annualised volatility 9.3%."
)

STATS = (
    '{\n  "total_bullets": 0,\n  "high_performing": 0,\n  "problematic": 0,\n'
    '  "unused": 0,\n  "by_section": {}\n}'
)


def load_bullets(path):
    """Real rules from a finished run, so the prompt is representative rather than synthetic."""
    if not os.path.exists(path):
        return None
    lines = [ln.rstrip("\n") for ln in open(path, encoding="utf-8") if ln.strip()]
    return [ln for ln in lines if ln.lstrip().startswith("[")] or lines


def grow_playbook(bullets, target_tokens):
    """Tile real rules, renumbering ids, until the rendering reaches target_tokens."""
    out, i = [], 0
    while len(ENC.encode("\n".join(out))) < target_tokens:
        src = bullets[i % len(bullets)]
        # Renumber so no two rules share an id -- a duplicate id is its own failure mode and
        # would confound what is being measured here.
        if src.lstrip().startswith("["):
            src = "[ceil-%05d]%s" % (i, src[src.index("]") + 1:])
        out.append(src)
        i += 1
        if i > 20000:  # pathological input; stop rather than spin
            break
    return "\n".join(out)


def call(client, model, prompt, call_id, think, num_ctx, seed, max_tokens):
    try:
        return timed_llm_call(
            client, "ollama", model, prompt, role="curator", call_id=call_id,
            max_tokens=max_tokens, num_ctx=num_ctx, seed=seed, think=think,
            # Fail fast: retrying would mask the behaviour being measured, and the empty-retry
            # path escalates temperature above 0.0, which changes what is being tested.
            retries_on_timeout=1, retries_on_empty=1,
        )
    except Exception as e:
        print(f"    call raised: {type(e).__name__}: {str(e)[:200]}")
        return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemma4:e2b", help="the curator model under test")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[6000, 10000, 14000, 18000, 22000, 26000, 30000],
                    help="playbook sizes in tokens to test, ascending")
    ap.add_argument("--think", choices=["default", "on", "off"], default="default",
                    help="matches --think_curator; 'default' is what the runs use")
    ap.add_argument("--num_ctx", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--task_name", default="formula")
    ap.add_argument("--playbook", default=DEFAULT_PLAYBOOK,
                    help="a finished run's final_playbook.txt, tiled to reach each size")
    ap.add_argument("--projected", type=int, default=26000,
                    help="curator prompt tokens projected at the end of a full arm; the "
                         "verdict is whether the ceiling clears this")
    args = ap.parse_args()

    bullets = load_bullets(args.playbook)
    if not bullets:
        print(f"no playbook at {args.playbook}\n"
              f"pass --playbook <a finished run's final_playbook.txt>")
        return 2

    think = {"default": None, "on": True, "off": False}[args.think]
    client, _, _ = initialize_clients("ollama")
    template = get_prompts(args.task_name).curator

    print(f"\nmodel={args.model}  think={args.think}  num_ctx={args.num_ctx}  seed={args.seed}")
    print(f"playbook source: {args.playbook} ({len(bullets)} real rules, tiled)\n")
    print(f"{'playbook':>9s} {'prompt':>7s} {'out':>6s} {'reason':>7s} {'secs':>6s}  verdict")
    print("-" * 72)

    last_ok = 0
    first_bad = None
    for size in sorted(args.sizes):
        playbook = grow_playbook(bullets, size)
        prompt = template.format(
            current_step=125, total_samples=250, token_budget=80000,
            playbook_stats=STATS, recent_reflection=REFLECTION,
            current_playbook=playbook, question_context=QUESTION_CONTEXT,
        )
        prompt_tokens = len(ENC.encode(prompt))
        text, info = call(client, args.model, prompt, f"ceiling_{size}", think,
                          args.num_ctx, args.seed, args.max_tokens)

        if text is None:
            verdict = "CALL FAILED"
        elif text.startswith("INCORRECT_DUE_TO_EMPTY_RESPONSE") or not text.strip():
            verdict = "EMPTY RESPONSE"
        else:
            parsed = extract_json_from_text(text, "operations")
            if not parsed or not parsed.get("operations"):
                verdict = "no usable operations"
            else:
                ops = parsed["operations"]
                # `refines` is required-but-nullable in the schema and is the graph arm's one
                # capability, so a curator that keeps answering but drops the field has still
                # failed for this experiment's purposes.
                has_refines = sum(1 for o in ops if "refines" in o)
                verdict = f"ok, {len(ops)} op(s), refines present in {has_refines}"
                last_ok = max(last_ok, prompt_tokens)

        if verdict.startswith("ok"):
            pass
        elif first_bad is None:
            first_bad = prompt_tokens

        out = (info or {}).get("response_num_tokens")
        reasoning = (info or {}).get("reasoning_chars")
        secs = (info or {}).get("total_time")
        print(f"{size:9d} {prompt_tokens:7d} {str(out):>6s} {str(reasoning):>7s} "
              f"{(f'{secs:.1f}' if isinstance(secs, (int, float)) else '-'):>6s}  {verdict}")

    print("-" * 72)
    if last_ok == 0:
        print("The curator failed at every size tested, including the smallest. That is not a\n"
              "context-window result -- check the model name and the endpoint first.")
        return 1

    print(f"largest prompt that produced usable operations: {last_ok:,} tokens")
    if first_bad is not None:
        print(f"first prompt that did not:                     {first_bad:,} tokens")
        print("The served window is between those two, whatever --num_ctx asked for.")
    else:
        print("No size tested failed. The ceiling is above the top of the ladder -- extend\n"
              "--sizes upward if you want to find it.")

    print()
    if last_ok >= args.projected:
        print(f"CLEARS the projected {args.projected:,} tokens at the end of a 250-step arm.")
        print("The curator survives the full run at the current dedup threshold.")
        return 0
    print(f"DOES NOT CLEAR the projected {args.projected:,} tokens at the end of a 250-step arm.")
    print("The back part of every training arm would curate past this point, silently.")
    print("Options, in the order they cost least: lower --dedup_threshold (measured to buy\n"
          "only 10-20%, so probably not enough alone), raise curator_frequency so fewer\n"
          "rules accumulate, shorten the run, or budget render_for_curator() in code.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
