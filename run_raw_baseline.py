#!/usr/bin/env python3
"""Score the generator model on a task with no ACE scaffolding at all.

**This is a reference point, not a fourth arm, and it must never be reported as one.** The
`none` arm is the control the experiment is designed around: it carries the *identical* generator
prompt and fills the playbook slot with `(empty)`, so it differs from the memory arms in memory
content and in nothing else (`ace/memory/none.py`). That is what makes `none -> text` and
`none -> graph` clean comparisons.

This script deliberately breaks that property. It strips the ACE prompt entirely -- no playbook
section, no reflection section, no bullet_ids, no instruction block, no "show your reasoning
step-by-step" -- and asks the model the task question directly. So it differs from every arm in
*prompt structure*, which is exactly the confound the `none` arm exists to avoid, and it cannot
enter the paired statistics as a peer.

What it is good for is a decomposition the arms alone cannot express:

    raw  -> none            the scaffolding effect: what the ACE prompt itself is worth
    none -> text / graph    the memory effect: what this experiment measures

It also answers a specific objection. The ACE paper quotes a "Base LLM" number, and our control
is not that: it is the same model under a playbook-shaped prompt that happens to be empty. Any
comparison to a published base-LLM figure has to be made against this script's number, not
against the `none` arm's.

Two things held identical to the arms on purpose, so the only difference is the prompt: the
decoding settings (model, `--num_ctx`, `--seed`, temperature, `--max_tokens`, thinking off) and
the scoring path (`utils.extract_answer`, then the task's own `answer_is_correct`). The answer is
still requested as a one-field JSON object, because an unparseable answer would measure the
extractor rather than the model -- see EXPERIMENT_FORMULA.md. That single format instruction is
the irreducible minimum of scaffolding and is disclosed as such.

    uv run python run_raw_baseline.py --task_name formula --seed 0 \
        --save_path results/formula/raw_baseline/seed0

Writes `raw_test_predictions.jsonl` in the same schema `compare_arms.py` reads, plus
`final_test_results.json` and a `run_config.json` marking `memory_kind: "raw"`.
"""
import argparse
import json
import os
import sys
from datetime import datetime

from eval.finance.data_processor import DataProcessor, load_data
from llm import timed_llm_call
from utils import check_answer, extract_answer, initialize_clients, write_predictions_jsonl

#: The whole prompt. Everything that is not the task question is the format instruction, and the
#: format instruction is one sentence. Keep it that way: every clause added here is scaffolding
#: that moves this away from "what the model does unaided".
RAW_PROMPT = """{question}{context}

Respond with only a JSON object in exactly this format, and nothing else:
{{"final_answer": "[your answer]"}}
"""


def build_prompt(sample):
    context = sample.get("context") or ""
    return RAW_PROMPT.format(
        question=sample.get("question", ""),
        context=("\n\n" + context) if context.strip() else "",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task_name", type=str, default="formula", choices=["formula", "finer"])
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--config_path", type=str,
                        default="./eval/finance/data/task_config_v2.json")
    parser.add_argument("--generator_model", type=str, default="gemma4:e2b")
    parser.add_argument("--api_provider", type=str, default="groq")
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--num_ctx", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--think", type=str, default="off", choices=["on", "off"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only the first N rows. For smoke-testing, never for a result.")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config_path, encoding="utf-8") as fh:
        test_path = json.load(fh)[args.task_name]["test_data"]

    processor = DataProcessor(task_name=args.task_name)
    samples = processor.process_task_data(load_data(test_path))
    if args.limit:
        samples = samples[:args.limit]

    run_dir = os.path.join(
        args.save_path,
        f"raw_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.task_name}_raw")
    log_dir = os.path.join(run_dir, "detailed_llm_logs")
    os.makedirs(log_dir, exist_ok=True)

    client, _reflector, _curator = initialize_clients(args.api_provider)

    print(f"raw baseline: {len(samples)} rows, model {args.generator_model}, seed {args.seed}")
    print("  no playbook, no reflection, no bullet_ids -- reference point only, not an arm")

    results = []
    for index, sample in enumerate(samples):
        response, _call_info = timed_llm_call(
            client, args.api_provider, args.generator_model, build_prompt(sample),
            role="generator", call_id=f"raw_test_eval_{index}",
            max_tokens=args.max_tokens, log_dir=log_dir, use_json_mode=False,
            num_ctx=args.num_ctx, seed=args.seed, think=(args.think == "on"),
        )
        answer = extract_answer(response)
        target = sample.get("target", "")
        correct = check_answer(processor, answer, target, sample)
        results.append({
            "index": index, "final_answer": answer, "target": target,
            "is_correct": bool(correct), "success": True,
        })
        if (index + 1) % 25 == 0:
            done = sum(1 for r in results if r["is_correct"])
            print(f"  {index + 1}/{len(samples)}  accuracy {done / len(results):.3f}")

    correct = sum(1 for r in results if r["is_correct"])
    final = {"accuracy": correct / len(results) if results else 0.0,
             "correct": correct, "total": len(results),
             "no_answer": sum(1 for r in results
                              if r["final_answer"] == "No final answer found")}

    write_predictions_jsonl(os.path.join(run_dir, "raw_test_predictions.jsonl"), results)
    with open(os.path.join(run_dir, "final_test_results.json"), "w", encoding="utf-8") as fh:
        json.dump(final, fh, indent=2)
    # memory_kind "raw" rather than "none": the fairness audits key off this field, and this run
    # is deliberately not comparable to the arms.
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump({"memory_kind": "raw", "retrieval": "none", "prompt_set": "raw_minimal",
                   "task_name": args.task_name, "test_data": test_path,
                   "generator_model": args.generator_model, "seed": args.seed,
                   "num_ctx": args.num_ctx, "max_tokens": args.max_tokens,
                   "think_generator": args.think,
                   "note": "reference point, not an experimental arm -- see module docstring"},
                  fh, indent=2)

    print(f"\naccuracy {final['accuracy']:.4f} ({correct}/{len(results)}), "
          f"no answer {final['no_answer']}")
    print(f"written to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
