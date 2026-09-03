#!/usr/bin/env python3
"""Check the per-task prompt sets bind their fields correctly.

Two silent failure modes this guards against:

  * The MBPP prompts must be byte-identical to what they were before prompt selection
    existed, or the completed MBPP arms stop being reproducible.
  * The finance generator prompt orders its four fields playbook -> reflection -> question
    -> context, while `Generator.generate` supplies question, context, playbook, reflection.
    Bound positionally, that puts the question under the "Playbook" heading in every call
    and nothing errors. A run would complete and score near zero for no visible reason.

No network, no Docker, no model.

    uv run python check_prompts.py
"""

import re
import sys

import ace.native_deps  # noqa: F401  -- must precede sentence-transformers imports
from ace.prompts import CODE_PROMPTS, FINANCE_PROMPTS, FINER_PROMPTS, get_prompts
from ace.prompts.generator import GENERATOR_PROMPT as HISTORICAL_CODE_GENERATOR

Q, C, P, R = "<<QUESTION>>", "<<CONTEXT>>", "<<PLAYBOOK>>", "<<REFLECTION>>"


def check(ok, label, failures):
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}")
    if not ok:
        failures.append(label)


def main():
    failures = []
    print("\n[1] selection")
    check(get_prompts(None) is CODE_PROMPTS, "no task name -> code prompts", failures)
    check(get_prompts("unknown_task") is CODE_PROMPTS, "unknown task -> code prompts", failures)
    check(get_prompts("mbpp_code_exec") is CODE_PROMPTS, "mbpp -> code prompts", failures)
    check(get_prompts("formula") is FINANCE_PROMPTS, "formula -> finance prompts", failures)
    # finer moved off FINANCE_PROMPTS on 2026-08-28: same generator and reflector objects,
    # its own curator. The identity checks below are what keep formula a replication.
    check(get_prompts("finer") is FINER_PROMPTS, "finer -> finer prompts", failures)
    check(FINER_PROMPTS.generator is FINANCE_PROMPTS.generator,
          "finer generator is the finance generator, same object", failures)
    check(FINER_PROMPTS.reflector is FINANCE_PROMPTS.reflector
          and FINER_PROMPTS.reflector_no_gt is FINANCE_PROMPTS.reflector_no_gt,
          "finer reflector is the finance reflector, same object", failures)
    check(FINER_PROMPTS.generator_binding == FINANCE_PROMPTS.generator_binding,
          "finer binds its generator fields by name, like finance", failures)

    print("\n[2] mbpp prompt is unchanged by the refactor")
    check(CODE_PROMPTS.format_generator(Q, C, P, R)
          == HISTORICAL_CODE_GENERATOR.format(Q, C, P, R),
          "generator prompt byte-identical to the positional call", failures)

    print("\n[3] finance generator field binding")
    rendered = FINANCE_PROMPTS.format_generator(Q, C, P, R)
    for heading, token in (("**Playbook:**", P), ("**Reflection:**", R),
                           ("**Question:**", Q), ("**Context:**", C)):
        body = rendered.split(heading, 1)[-1].lstrip()
        check(body.startswith(token), f"{heading} is followed by {token}", failures)
    for token in (Q, C, P, R):
        check(rendered.count(token) == 1, f"{token} substituted exactly once", failures)

    print("\n[4] finance reflector / curator render completely")
    reflector = FINANCE_PROMPTS.reflector.format(
        "QUEST", "TRACE", "PRED", "GTRUTH", "ENVFB", "BULLETS")
    reflector_no_gt = FINANCE_PROMPTS.reflector_no_gt.format(
        "QUEST", "TRACE", "PRED", "ENVFB", "BULLETS")
    curator_kwargs = dict(token_budget=1, current_step=2, total_samples=3,
                          playbook_stats="STATS", recent_reflection="REFL",
                          current_playbook="PBOOK", question_context="QCTX")
    curator = FINANCE_PROMPTS.curator.format(**curator_kwargs)
    curator_no_gt = FINANCE_PROMPTS.curator_no_gt.format(**curator_kwargs)

    for name, text in (("reflector", reflector), ("reflector_no_gt", reflector_no_gt),
                       ("curator", curator), ("curator_no_gt", curator_no_gt)):
        leftover = re.findall(r"\{[a-z_]+\}", text)
        check(not leftover, f"{name}: no unfilled placeholder {leftover or ''}", failures)

    body = reflector.split("**Ground Truth Answer:**", 1)[-1].lstrip()
    check(body.startswith("GTRUTH"), "reflector: ground truth in its own slot", failures)
    body = reflector.split("**Environment Feedback:**", 1)[-1].lstrip()
    check(body.startswith("ENVFB"), "reflector: environment feedback in its own slot", failures)
    for token in ("STATS", "REFL", "PBOOK", "QCTX"):
        check(curator.count(token) == 1, f"curator: {token} substituted exactly once", failures)

    print("\n[6] the finer curator delta, and that formula did not move with it")
    for label, finer_text, finance_text in (
            ("curator", FINER_PROMPTS.curator, FINANCE_PROMPTS.curator),
            ("curator_no_gt", FINER_PROMPTS.curator_no_gt, FINANCE_PROMPTS.curator_no_gt)):
        check("IF <what the text says> THEN <ExactTagName>" in finer_text,
              f"finer {label}: asks for a conditional mapping", failures)
        check("The THEN clause is the tag name alone" in finer_text
              and "NEVER use words like prioritize, prefer, or over" in finer_text,
              f"finer {label}: the THEN clause is restricted to a tag name", failures)
        check('- section: must be exactly "context_clues_and_indicators"' in finer_text,
              f"finer {label}: the ADD section is pinned at the schema line", failures)
        check("the section to add the new bullet to" not in finer_text
              and "the section to add the bullet to" not in finer_text,
              f"finer {label}: neither section line offers a choice", failures)
        check("IF the text reports cash paid to acquire a business THEN" in finer_text,
              f"finer {label}: the example is a mapping, not a calculation", failures)
        check("IF <what the text says>" not in finance_text
              and "context_clues_and_indicators" not in finance_text
              and "The THEN clause is the tag name alone" not in finance_text,
              f"formula {label}: unchanged by the finer delta", failures)
        check("the section to add the new bullet to" in finance_text,
              f"formula {label}: keeps the authors' free section choice", failures)
        check("[New calculation method...]" in finance_text,
              f"formula {label}: keeps the authors' example", failures)

    print("\n[5] the two deliberate deltas from the authors' text")
    for name, text in (("curator", curator), ("curator_no_gt", curator_no_gt)):
        check('"refines": null' in text and "refines:" in text,
              f"{name}: refines documented and shown in the example", failures)
    for name, text in (("reflector", reflector), ("reflector_no_gt", reflector_no_gt)):
        check("causal impact on THIS attempt" in text,
              f"{name}: tags defined by causal impact", failures)
        check('"helpful" is not available' in text,
              f"{name}: helpful withheld on a failed attempt", failures)

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("prompt sets bind correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
