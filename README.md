# Relational vs linear memory for self-improving agents

A fork of [ACE (Agentic Context Engineering)](https://arxiv.org/abs/2510.04618) built to answer
one question:

> **Does representing an agent's accumulated rules as a relation graph produce better task
> performance than the same rules held as a flat list?**

ACE lets an LLM improve without touching its weights: a **Generator** answers, a **Reflector**
grades which accumulated rules helped or hurt, and a **Curator** writes new rules into a playbook
that is injected into the next prompt. This fork keeps that loop exactly as the authors wrote it
and swaps out only *how the rules are stored*.

**This is a research repository.** It exists to run controlled experiments and to make them
auditable. Most design decisions here are constrained by fairness between arms.

## What has to be true for a graph to matter

A graph earns its keep only if it holds something a flat list cannot. That is a sharper test than
it sounds, and most candidate relations fail it:

| relation | pairwise? | recoverable from the rules' own text? | needs a graph |
|---|---|---|---|
| `refines` — "this rule supersedes that one" | no, unary | n/a | **no** — a flat list with a `supersedes` field |
| `related_to` — cosine between two rules' texts | yes | **yes** | **no** — a flat list plus an embedding index |
| `co_cited` — the generator named both in one answer | yes | **no**, it exists only in accumulated history | **yes** |

**`co_cited` is the capability under test.** Per generation the memory records which rules were
rendered and which the generator actually cited; an edge exists when a pair was co-rendered at
least *n* times and co-cited on at least *r* of those occasions. The stored quantity is a rate, so
the relation ranks pairs *chosen together* rather than pairs *shown often*. No flat representation
holds that, it is a fact about the pair, not about either item.

**The edge is used, not just displayed.** Both memory arms score every live rule against the
question and render the highest-scoring under one shared token budget. The graph arm takes the
same top-*k* as seeds and then walks `co_cited` **breadth-first** outward, so it can surface a rule
that relevance ranking alone would never reach. Structure is charged to that same budget so the
graph arm shows *fewer* rules for the same tokens, and the edges have to pay for themselves.

`--memory graph --retrieval flat` is the ablation separating "edges choose what is read" from
"edges are shown alongside what is read".

## The three arms

| arm | flags | mode | what it answers |
|---|---|---|---|
| control | `--memory none --retrieval none` | `eval_only` | the accumulated content, by having none |
| baseline | `--memory text --retrieval flat` | `offline` | the ACE baseline |
| treatment | `--memory graph --retrieval graph --cocite` | `offline` | does *structure* help? |

Both memory arms accumulate the same rules from the same curator, de-duplicate them by the same
rule, receive the same citation stream, and render under the same token budget. The single
difference is that one can represent facts about **pairs** and the other only facts about
**items**.

**The control is not a bare LLM.** It carries the identical generator prompt and fills the playbook
slot with `(empty)` (`ace/memory/none.py`). Removing the section would make it differ from the other
arms in prompt structure *and* in memory content, when only the second is the intended comparison.
So `none → text` isolates the accumulated content with the prompt held fixed. The bare-LLM number,
for anyone who asks for one, comes from `run_raw_baseline.py`, and it is a reference point rather
than a fourth arm.

One further difference is disclosed rather than removed: the curator prompt is held identical in
all three arms, including a `refines` field only a graph arm acts on, so the arms also differ in
eviction capability. The baseline is honestly "ACE with an extended curator schema whose extra
field it ignores".

## Status

| experiment | state |
|---|---|
| **`formula` co-citation** | ✅ **complete**, nine runs, three arms × three replicates. **The result of this repository.** Memory helps: `none → cocite` **+10.83pp** and `none → text` **+8.67pp**, both significant after Holm correction over the three pre-registered comparisons. Relational structure does not beat linear: `text → cocite` **+2.17pp**, CI [−0.020, +0.060], **p=0.3344, null**. |
| **`mbpp_code_exec` co-citation** | 📋 **pre-registered, not started.** The contrast case: MBPP's problems are independent, so nothing transfers between them and a null there is a claim about *when* memory helps rather than a failed replication. Its expected direction is declared null in advance. |

**Why the primary null is worth reporting.** A null only means something if the treatment actually
happened, so it was instrumented three ways before the accuracy was read. The edges predict
held-out co-citation at **3.6× to 11.6×** base rate. The walk placed **5% to 44%** of what the
generator read where relevance alone would not have put it, and the generator went on to cite those
walk-added rules at **1.3× to 2.3×** the rate it cited the relevance-selected ones. Across an
**eightfold spread in that dose**, accuracy did not move at all (0.7250 / 0.7300 / 0.7350).

Fifty formulas give a ~4pp detection floor and effects below it are
invisible here by construction.

### Background

Two earlier run sets are motivation, not results, and no claim rests on them:

- **The three-arm `formula` experiment** used `related_to` as the graph's relation and came out
  null. Its lasting contribution is *mechanical*, not statistical: `related_to` is cosine between
  two rules' texts while retrieval scores rules by cosine between the question and each rule, the
  same encoder in the same space, so the walk landed where relevance was already looking. A random
  rewiring at matched edge count surfaced **2.3× to 3.6× more** novel rules than the real graph.
  That is what motivated co-citation.
- **The earlier MBPP runs** (`docs/MBPP_RESULTS.md`) are superseded: the graph arm predates
  `--retrieval`, so the treatment never fired, and the baseline predates `--dedup`.

## Documents

`docs/` is the research record. Pre-registrations were written before their runs; results files
report what happened.

| document | what it is |
|---|---|
| [`docs/EXPERIMENT_COCITE.md`](docs/EXPERIMENT_COCITE.md) | **Pre-registration for the co-citation experiment**, with its amendment log. §0 states plainly that the experiment is post-hoc relative to the background runs. |
| [`docs/COCITE_RESULTS.md`](docs/COCITE_RESULTS.md) | **Results and final analysis.** The headline of this repository. |
| [`docs/EXPERIMENT_MBPP.md`](docs/EXPERIMENT_MBPP.md) | Pre-registration for the MBPP experiment, whose expected direction is declared **null**. |
| [`docs/EXPERIMENT_FORMULA.md`](docs/EXPERIMENT_FORMULA.md) | Background. Pre-registration for the three-arm `related_to` experiment. |
| [`docs/FORMULA_RESULTS_PREREGISTERED.md`](docs/FORMULA_RESULTS_PREREGISTERED.md) | Background. Results for those nine runs. |
| [`docs/MBPP_RESULTS.md`](docs/MBPP_RESULTS.md) | **Superseded.** Read it for the §2/§5/§6 mechanism write-up, never for its numbers. |

Each experiment stands alone and is analysed alone. They share a harness and a set of fairness
rules, which makes them *methodologically* comparable; no conclusion in one is conditioned on what
another finds.

## Setup

```bash
uv sync                       # package manager is uv; Python 3.12 
```

The LLM backend is **hardcoded to a personal Ollama tunnel** in `utils.py::initialize_clients()`
(`https://ollama.caduceus.lol/v1`). `--api_provider` is accepted but ignored for routing. This is
the expected development setup, change that function to point elsewhere.

`gemma4:e2b` fills all three roles. `eval/mbpp_code_exec` additionally needs Docker running, since
it executes generated code in a sandbox; `eval/finance` does not.

## Running

```bash
# control -- --dedup and --eval_steps are inert here but must match the other arms, because
# compare_arms.py pins them; --retrieval none is forced by ace/ace.py:176
uv run python -m eval.finance.run --task_name formula --mode eval_only \
  --save_path results/formula/cocite_experiment/seed0/none \
  --memory none --retrieval none --dedup --dedup_threshold 0.88 \
  --config_path ./eval/finance/data/task_config_v2.json \
  --generator_model gemma4:e2b --reflector_model gemma4:e2b --curator_model gemma4:e2b \
  --think_generator off --num_ctx 32768 --seed 0 --eval_steps 125

# baseline
uv run python -m eval.finance.run --task_name formula --mode offline \
  --save_path results/formula/cocite_experiment/seed0/text \
  --memory text --retrieval flat --dedup --dedup_threshold 0.88 \
  --config_path ./eval/finance/data/task_config_v2.json \
  --generator_model gemma4:e2b --reflector_model gemma4:e2b --curator_model gemma4:e2b \
  --think_generator off --num_ctx 32768 --seed 0 --eval_steps 125 --skip_initial_test

# treatment -- --cocite replaces related_to rather than joining it, and the runner's
# --cocite_min_rate default is 0.30, so the pinned 0.20 must be passed explicitly
uv run python -m eval.finance.run --task_name formula --mode offline \
  --save_path results/formula/cocite_experiment/seed0/cocite \
  --memory graph --retrieval graph --cocite \
  --cocite_min_opportunity 3 --cocite_min_rate 0.20 --cocite_half_life 0 \
  --dedup --dedup_threshold 0.88 \
  --config_path ./eval/finance/data/task_config_v2.json \
  --generator_model gemma4:e2b --reflector_model gemma4:e2b --curator_model gemma4:e2b \
  --think_generator off --num_ctx 32768 --seed 0 --eval_steps 125 --skip_initial_test
```

Repeat at `--seed 1` and `--seed 2`. `eval/mbpp_code_exec/run.py` takes the same flags by design.

**Thresholds are per benchmark and must be re-derived, never inherited.** `--dedup_threshold` is
`0.88` on formula and `0.86` on MBPP because financial-convention rules and Python style advice sit
at different similarities. The same applies to the co-citation gates, which were pinned from
formula's own held-out training lift: a gate set too tight silently fails to deliver the treatment
rather than erroring.

The rest of the pinned configuration is the runners' defaults: `--curator_frequency 1`,
`--max_num_rounds 3`, `--num_epochs 1`, `--test_workers 1`, `--retrieval_budget 2500`,
`--retrieval_seeds 5`, `--max_tokens 16384`.

**`--seed` does not make a run reproducible.** It reaches only Ollama's sampler, and at
temperature 0 the sampler is greedy and never reads it. Run-to-run divergence tracks the state the
Ollama server was in when it loaded the model, and the ACE loop amplifies one flipped token into a
different playbook. Never assume a run is reproducible, and always md5sum the prediction files before treating two runs as independent replicates; identical hashes have occurred.

**Before spending compute**, run the checks. None of these hit the network.

```bash
uv run python check_memory_equivalence.py   # the text arm is byte-identical to upstream
uv run python check_prompts.py              # prompt fields bind to the right headings
uv run python check_finance_wiring.py       # the whole loop, with the LLM stubbed
uv run python compare_arms.py --selftest    # the statistics, against their closed-form limits
```

These two do hit it, and are worth the minute:

```bash
uv run python check_model.py --model <name> --think off   # can it hold the schema, is it thinking?
uv run python check_curator_ceiling.py --model <name>     # how big a playbook before it breaks?
```

## Analysis

```bash
uv run python check_learning_signal.py <run_dir>      # is the loop actually learning?
uv run python compare_arms.py <run_dir> ...           # paired, cluster-robust cross-arm tests
uv run python analyze_walk_share.py <graph_run_dir>   # did the walk change what was read?
uv run python calibrate_cocite_gate.py <run_dir> ...  # do the edges predict held-out co-citation?
uv run python aggregate_run_costs.py --root results   # tokens, latency, and a fairness audit
```

`compare_arms.py` is where every cross-arm claim comes from. Accuracy alone cannot tell "the memory
did nothing" apart from "the memory fixed 15 problems and broke 15"  and in every replicate here
the two memories disagreed on 14% to 17% of the test set while the net difference collapsed to
zero. It pairs the arms problem by problem, and on `formula` it also **blocks on the formula**,
because the 200-row test set is 50 formulas × 4 instances and treating those as independent claims
the precision of 200 draws for a design that has about 50.

**No LLM grades any output, and none sits inside the write path either.** `formula` matches the
extracted answer numerically against the target and `mbpp_code_exec` executes it against fixed
assertions in a sandbox, so scoring is deterministic and cannot drift between arms or replicates.
An LLM judge would introduce grader variance that can correlate with output style, and memory
changes output style, so a judge could register a difference that is not a difference in
correctness. The same rule governs de-duplication, which compares rules by embedding rather than
asking a model which to keep (`ace/memory/refine.py`).

`analyze_walk_share.py` and `calibrate_cocite_gate.py` are what make a null reportable rather than
merely unexplained: the first measures whether the graph changed what the generator read, the
second whether the edges carry real information about unseen data.

`run_raw_baseline.py` scores the generator with no ACE prompt at all. It is a reference point for
readers who ask what the model does unaided, **never a fourth arm**, it differs in prompt
structure, which is the exact confound the `none` control exists to remove, so no statistic is
computed on it.

## Credit

Upstream method and prompts: Qizheng Zhang et al., *Agentic Context Engineering: Evolving Contexts
for Self-Improving Language Models* (arXiv:2510.04618), and the
[`ace-agent/ace`](https://github.com/ace-agent/ace) implementation. The finance prompts in
`ace/prompts/formula.py` are theirs verbatim apart from two documented deltas, both applied
identically in every arm. Numbers quoted from the paper are the authors', not this fork's; nothing
here reproduces them, and the reasons are in the results documents above.
