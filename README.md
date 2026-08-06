# Relational vs linear memory for self-improving agents

A fork of [ACE (Agentic Context Engineering)](https://arxiv.org/abs/2510.04618) built to answer
one question:

> **Does representing an agent's accumulated rules as a typed relation graph produce better task
> performance than the same rules held as a flat list?**

ACE lets an LLM improve without touching its weights: a **Generator** answers, a **Reflector**
grades which accumulated rules helped or hurt, and a **Curator** writes new rules into a
playbook that is injected into the next prompt. This fork keeps that loop exactly as the
authors wrote it and swaps out only *how the rules are stored*.

**This is a research repository, not a library.** It exists to run a controlled experiment and
to make that experiment auditable. Most of the design decisions here are constrained by
fairness between arms rather than by what would be convenient.

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

Honest summary, and the reason to read the docs before the code:

- **MBPP** — a null result. `none` 0.3346, `text` 0.3619, `graph` 0.3346 at n=257, all paired
  p > 0.4, at 8× the tokens. MBPP's problems are independent, so nothing transfers between them
  and the curator can only write style advice. Structurally unsuited to the question.
  See [`MBPP_RESULTS.md`](MBPP_RESULTS.md).
- **formula** — the better instrument, and where the work is. One exploratory seed per arm is
  complete and is reported in [`FORMULA_RESULTS.md`](FORMULA_RESULTS.md). It is *not* a result:
  the baseline was missing one of ACE's three components and the graph arm had no functional
  use of its graph. Both are fixed; the pre-registered nine runs have not been run yet.

## Documentation

| file | what it is |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | the working guide — architecture, fairness rules, and every gotcha that has cost a run |
| [`EXPERIMENT_FORMULA.md`](EXPERIMENT_FORMULA.md) | the pre-registration, its amendment, and a dated deviations log |
| [`FORMULA_RESULTS.md`](FORMULA_RESULTS.md) | exploratory seed-0 results and what they exposed |
| [`MBPP_RESULTS.md`](MBPP_RESULTS.md) | the MBPP null, with its own criticism section |
| [`EXTENDING_ACE.md`](EXTENDING_ACE.md) | adding a task or a memory kind |

## Setup

```bash
uv sync                       # package manager is uv; Python 3.12 is pinned
```

The LLM backend is **hardcoded to a personal Ollama tunnel** in `utils.py::initialize_clients()`
(`https://ollama.caduceus.lol/v1`). `--api_provider` is accepted but ignored for routing. This
is the expected development setup, not a bug — change that function to point elsewhere.

Models used throughout: `qwen3.5:2b` as generator, `gemma4:e2b` as reflector and curator.
`eval/mbpp_code_exec` additionally needs Docker running, since it executes generated code in a
sandbox; `eval/finance` does not.

## Running

```bash
# the control
uv run python -m eval.finance.run --task_name formula --mode eval_only \
  --save_path results/formula_none --memory none --retrieval none \
  --config_path ./eval/finance/data/task_config_v2.json \
  --generator_model qwen3.5:2b --think_generator off --num_ctx 32768 --seed 0

# the baseline
uv run python -m eval.finance.run --task_name formula --mode offline \
  --save_path results/formula_text --memory text --retrieval flat --dedup \
  --config_path ./eval/finance/data/task_config_v2.json \
  --generator_model qwen3.5:2b --think_generator off --num_ctx 32768 --seed 0 \
  --skip_initial_test

# the treatment
uv run python -m eval.finance.run --task_name formula --mode offline \
  --save_path results/formula_graph --memory graph --retrieval graph --dedup \
  --config_path ./eval/finance/data/task_config_v2.json \
  --generator_model qwen3.5:2b --think_generator off --num_ctx 32768 --seed 0 \
  --skip_initial_test
```

`eval/mbpp_code_exec/run.py` takes the same flags, deliberately.

**Before spending compute**, run the checks. There is no test suite; these stand in for one and
none of them touch the network:

```bash
uv run python check_memory_equivalence.py   # the text arm is byte-identical to upstream
uv run python check_prompts.py              # prompt fields bind to the right headings
uv run python check_finance_wiring.py       # the whole loop, with the LLM stubbed
uv run python compare_arms.py --selftest    # the statistics, against their closed-form limits
```

## Analysis

```bash
uv run python check_learning_signal.py <run_dir>      # is the loop actually learning?
uv run python compare_arms.py <run_dir> ...           # paired, cluster-robust cross-arm tests
uv run python aggregate_run_costs.py --root results   # tokens, latency, and a fairness audit
uv run python replay_curator_sanitizer.py --root results
```

`compare_arms.py` is where every cross-arm claim comes from. Accuracy alone cannot tell "the
memory did nothing" apart from "the memory fixed 31 problems and broke 31" — on MBPP it was the
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
