# Relational vs linear memory for self-improving agents

A fork of [ACE (Agentic Context Engineering)](https://arxiv.org/abs/2510.04618) built to answer
one question:

> **Does representing an agent's accumulated rules as a typed relation graph produce better task
> performance than the same rules held as a flat list?**

ACE lets an LLM improve without touching its weights: a **Generator** answers, a **Reflector**
grades which accumulated rules helped or hurt, and a **Curator** writes new rules into a
playbook that is injected into the next prompt. This fork keeps that loop exactly as the
authors wrote it and swaps out only *how the rules are stored*.

**This is a research repository** It exists to run a controlled experiment and
to make that experiment auditable. Most of the design decisions are constrained by
fairness between arms.

## The three arms

| arm | `--memory` | `--retrieval` | mode | what it answers |
|---|---|---|---|---|
| control | `none` | `none` | `eval_only` | does memory help at all? |
| baseline | `text` | `flat` | `offline` | the ACE baseline |
| treatment | `graph` | `graph` | `offline` | does *structure* help? |

Both memory arms accumulate the same rules from the same curator, de-duplicate them by the same
rule, and render them into the prompt under the same token budget. The single difference is how
the rules are selected for the prompt: the baseline takes the highest-scoring by similarity to
the question; the treatment takes the same seeds and then walks `related_to` edges outward, so
it can surface a rule that similarity alone would never reach.

## Status


## Setup

```bash
uv sync                       # package manager is uv; Python 3.12 is pinned
```

The LLM backend is **hardcoded to a personal Ollama tunnel** in `utils.py::initialize_clients()`
(`https://ollama.caduceus.lol/v1`). `--api_provider` is accepted but ignored for routing. This
is the expected development setup, change that function to point elsewhere.

Models used throughout: `gemma4:e2b` for all 3 roles of the agent loop.
`eval/mbpp_code_exec` additionally needs Docker running, since it executes generated code in a
sandbox; `eval/finance` does not.

## Running

```bash
# the control -- --dedup and --eval_steps are inert here but must match the other arms,
# because compare_arms.py pins them; --retrieval none is forced by ace/ace.py:176
uv run python -m eval.finance.run --task_name formula --mode eval_only \
  --save_path results/formula/preregistered/seed0/none \
  --memory none --retrieval none --dedup --dedup_threshold 0.88 \
  --config_path ./eval/finance/data/task_config_v2.json \
  --generator_model gemma4:e2b --reflector_model gemma4:e2b --curator_model gemma4:e2b \
  --think_generator off --num_ctx 32768 --seed 0 --eval_steps 125

# the baseline
uv run python -m eval.finance.run --task_name formula --mode offline \
  --save_path results/formula/preregistered/seed0/text \
  --memory text --retrieval flat --dedup --dedup_threshold 0.88 \
  --config_path ./eval/finance/data/task_config_v2.json \
  --generator_model gemma4:e2b --reflector_model gemma4:e2b --curator_model gemma4:e2b \
  --think_generator off --num_ctx 32768 --seed 0 --eval_steps 125 --skip_initial_test

# the treatment
uv run python -m eval.finance.run --task_name formula --mode offline \
  --save_path results/formula/preregistered/seed0/graph \
  --memory graph --retrieval graph --dedup --dedup_threshold 0.88 \
  --config_path ./eval/finance/data/task_config_v2.json \
  --generator_model gemma4:e2b --reflector_model gemma4:e2b --curator_model gemma4:e2b \
  --think_generator off --num_ctx 32768 --seed 0 --eval_steps 125 --skip_initial_test
```

Repeat for `--seed 1` and `--seed 2`, all nine pinned to one commit. Why `0.88` rather than the
`0.90` default, and why one model in all three roles: `EXPERIMENT_FORMULA.md` §9, entry
2026-08-08.

`eval/mbpp_code_exec/run.py` takes the same flags, deliberately.

**Before spending compute**, run the checks.

```bash
uv run python check_memory_equivalence.py   # the text arm is byte-identical to upstream
uv run python check_prompts.py              # prompt fields bind to the right headings
uv run python check_finance_wiring.py       # the whole loop, with the LLM stubbed
uv run python compare_arms.py --selftest    # the statistics, against their closed-form limits
```

```bash
uv run python check_model.py --model <name> --think off   # can it hold the schema, is it thinking?
uv run python check_curator_ceiling.py --model <name>     # how big a playbook before it breaks?
```


## Analysis

```bash
uv run python check_learning_signal.py <run_dir>      # is the loop actually learning?
uv run python compare_arms.py <run_dir> ...           # paired, cluster-robust cross-arm tests
uv run python aggregate_run_costs.py --root results   # tokens, latency, and a fairness audit
uv run python replay_curator_sanitizer.py --root results
```

`compare_arms.py` is where every cross-arm claim comes from. Accuracy alone cannot tell "the
memory did nothing" apart from "the memory fixed 31 problems and broke 31", on MBPP it was the
second. It pairs the arms problem by problem, and on `formula` it also **blocks on the formula**,
because the 200-row test set is 50 formulas × 4 instances and treating those as independent
overstates the precision. That correction removed the only significant result on the books.

## Credit

Upstream method and prompts: Qizheng Zhang et al., *Agentic Context Engineering: Evolving
Contexts for Self-Improving Language Models* (arXiv:2510.04618), and the
[`ace-agent/ace`](https://github.com/ace-agent/ace) implementation. The finance prompts in
`ace/prompts/formula.py` are theirs verbatim apart from two documented deltas, both applied
identically in every arm. Numbers quoted from the paper are the authors', not this fork's;
nothing here reproduces them, and the reasons are in the results documents above.
