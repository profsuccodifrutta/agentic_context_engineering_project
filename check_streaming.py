#!/usr/bin/env python3
"""
Verify that streaming is a safe swap before spending a full training run on it.

Streaming was added to llm.py to stop long generations being killed by the tunnel's
read timeout (HTTP 524). That only helps if three things hold, and each of them is a
silent failure if it does not:

  1. Streamed calls return content at all.
  2. Streamed calls still report token usage. Usage is not recoverable from a stream
     unless the server honours `stream_options.include_usage`, and every cost and
     fairness number in aggregate_run_costs.py is computed from it. A server that
     ignores the flag produces a complete-looking run with no cost data.
  3. Streaming does not change what the model generates. It is meant to be a transport
     change only -- same model, same seed, same temperature, same num_ctx -- so the
     text must come back byte-identical. If it does not, streamed and non-streamed arms
     are not comparable and the arm already run non-streamed would have to be redone.

Then it pushes a deliberately long generation through, which is the failure this whole
change exists to fix.

    uv run python check_streaming.py
    uv run python check_streaming.py --model gemma4:e2b --num_ctx 32768

No Docker and no dataset needed -- this talks to the LLM endpoint only.
"""
import argparse
import sys
import time

from llm import timed_llm_call
from utils import initialize_clients

SHORT_PROMPT = (
    "Write a Python function that returns the nth Fibonacci number. "
    "Respond with code only."
)

# Sized to run well past the tunnel's read timeout. The 524s appeared on generator
# calls emitting 7-8k tokens, so this needs to be in that range to be a real test.
LONG_PROMPT = (
    "Write 40 small, distinct Python utility functions. Number them 1 to 40. "
    "Give each one a docstring, a type-annotated signature, and a worked usage "
    "example in a comment. Do not stop early and do not summarise -- write all 40."
)


def call(client, model, prompt, call_id, stream, num_ctx, seed, max_tokens, include_usage=True):
    """Run one call and return (text, call_info, wall_seconds), or (None, None, secs) on failure."""
    started = time.time()
    try:
        text, info = timed_llm_call(
            client,
            "ollama",
            model,
            prompt,
            role="generator",
            call_id=call_id,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
            seed=seed,
            stream=stream,
            stream_include_usage=include_usage,
            # Fail fast: this is a check, not a run. Retrying here would hide the
            # very error we are trying to observe.
            retries_on_timeout=1,
            retries_on_empty=1,
        )
        return text, info, time.time() - started
    except Exception as e:
        print(f"    call raised: {type(e).__name__}: {str(e)[:200]}")
        return None, None, time.time() - started


def describe(label, text, info, secs):
    if text is None:
        print(f"  {label:<14} FAILED after {secs:.1f}s")
        return
    print(f"  {label:<14} {len(text):>6} chars  "
          f"prompt_tok={info.get('prompt_num_tokens')}  "
          f"resp_tok={info.get('response_num_tokens')}  "
          f"{secs:.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemma4:e2b")
    ap.add_argument("--num_ctx", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=0)
    # Matches eval/mbpp_code_exec/run.py's default. The 4096 in the timed_llm_call and ACE
    # signatures is never what a real run uses, and testing at 4096 caps the long generation
    # before it can run long enough to exercise the timeout this check exists for.
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--skip_long", action="store_true",
                    help="skip the long-generation test (it is the slow one)")
    args = ap.parse_args()

    client, _, _ = initialize_clients("ollama")
    common = dict(num_ctx=args.num_ctx, seed=args.seed, max_tokens=args.max_tokens)
    failures = []

    print(f"\nmodel={args.model}  num_ctx={args.num_ctx}  seed={args.seed}  "
          f"max_tokens={args.max_tokens}")

    # ---- 1/2/3: streamed vs non-streamed on an identical request -------------------
    print("\n[1] streamed vs non-streamed, identical request")
    s_text, s_info, s_secs = call(client, args.model, SHORT_PROMPT, "check_stream_on",
                                  stream=True, **common)
    describe("streamed", s_text, s_info, s_secs)
    n_text, n_info, n_secs = call(client, args.model, SHORT_PROMPT, "check_stream_off",
                                  stream=False, **common)
    describe("non-streamed", n_text, n_info, n_secs)

    if s_text is None:
        failures.append("streaming call failed outright -- streaming is not usable")
    else:
        if s_info.get("response_num_tokens") is None or s_info.get("prompt_num_tokens") is None:
            failures.append(
                "streamed call returned no token usage -- the server is ignoring "
                "stream_options.include_usage, so aggregate_run_costs.py would have "
                "nothing to sum. Retry with --no-usage-flag handling or run non-streamed."
            )
        if n_text is None:
            print("    non-streamed call failed, so equivalence could not be checked")
        elif s_text == n_text:
            print("    -> IDENTICAL output. Streaming is transport-only; arms stay comparable.")
        else:
            failures.append(
                f"streamed and non-streamed output DIFFER "
                f"({len(s_text)} vs {len(n_text)} chars) at the same seed and temperature. "
                f"Streaming would be a sampling confound, not just a transport change."
            )

    # ---- 4: the actual bug being fixed ---------------------------------------------
    if not args.skip_long:
        print("\n[2] long generation (the case that was returning HTTP 524)")
        l_text, l_info, l_secs = call(client, args.model, LONG_PROMPT, "check_stream_long",
                                      stream=True, **common)
        describe("streamed long", l_text, l_info, l_secs)
        if l_text is None:
            failures.append("long streamed generation failed -- the 524 fix did not take")
        else:
            if l_info.get("response_num_tokens") == args.max_tokens:
                print(f"    stopped at exactly max_tokens={args.max_tokens}, so the cap is "
                      f"enforced and this is as long as a call can get.")
            if l_secs < 100:
                print(f"    completed in {l_secs:.1f}s, under the ~100s timeout, so this run "
                      f"did not actually exercise the failure. Not a pass or a fail.")
            else:
                print(f"    -> survived {l_secs:.1f}s, past the timeout that was killing calls.")

    # ---- verdict --------------------------------------------------------------------
    print("\n" + "=" * 72)
    if failures:
        print("NOT SAFE TO RUN:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("Streaming verified: content returned, usage preserved, output unchanged.")
    print("Safe to start the text and graph arms.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
