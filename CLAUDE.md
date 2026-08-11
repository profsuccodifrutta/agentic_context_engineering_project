# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ACE (Agentic Context Engineering) — LLMs self-improve by evolving accumulated rules through a
Generator / Reflector / Curator loop instead of fine-tuning weights. Fork of `ace-agent/ace`
(arXiv:2510.04618).

This fork's research question: **does relational memory beat linear memory for agent execution?**
Three arms:

| arm | `--memory` | `--retrieval` | what it answers |
|---|---|---|---|
| control | `none` | `none` | does memory help at all? |
| baseline | `text` | `flat` | the ACE baseline |
| treatment | `graph` | `graph` | does *structure* help? |

Everything else is held identical across arms — same prompts, same data order, same seed, same
context window, same token budget, same de-duplication. That is the point, and it constrains
most design decisions here.

**The graph's capability is that retrieval follows edges.** Both arms score every rule against
the question and take the highest-scoring under one shared token budget; the graph arm then
walks `related_to` outward from the same seeds, so it can render a rule that relevance alone
would never surface. That is the only place connectivity changes what a model reads — see
"Retrieval" below, and read it before touching either memory.

`refines` (the curator naming a rule its new rule supersedes) is **no longer the distinguishing
capability**, and treating it as one is what made the seed-0 runs uninformative. It is eviction,
which a flat list reproduces with a `supersedes` field and a render-time filter; the curator
declared it twice in 50 calls; and both arms now evict anyway through grow-and-refine.
`--keep_superseded` remains as an ablation.

- `ace/memory/` — the swap seam. `base.py` (Memory ABC), `text.py` (flat playbook, delegates to
  `playbook_utils` so the baseline is provably unchanged), `graph.py` (NetworkX),
  `refine.py` (shared de-duplication decision), `retrieval.py` (shared scoring and budgeting),
  `embeddings.py`
- `ace/core/` — `generator.py`, `reflector.py`, `curator.py`, `bulletpoint_analyzer.py`
- `ace/prompts/` — prompt templates, one per core module
- `eval/<task>/` — one dir per benchmark, each with its own `run.py` + `data_processor.py`
- `llm.py`, `utils.py`, `logger.py`, `playbook_utils.py` — shared infra at repo root
- `ace-appworld/` is an uninitialized submodule — ignore it

## Setup and running

Package manager is **uv** (`uv sync`). Python 3.12 pinned in `.python-version`. Evals run as modules
from the repo root:

```
uv run python -m eval.mbpp_code_exec.run --mode offline   --save_path results --memory text  --num_ctx 32768 --seed 0
uv run python -m eval.mbpp_code_exec.run --mode offline   --save_path results --memory graph --num_ctx 32768 --seed 0
uv run python -m eval.mbpp_code_exec.run --mode eval_only --save_path results --memory none  --num_ctx 32768 --seed 0
```

`--memory none` refuses `offline`/`online`: it accumulates nothing, so training would spend
generator, reflector and curator calls on a memory that provably cannot change.

**The control arm must still carry the memory flags, inert.** `dedup`, `dedup_threshold`,
`eval_steps`, `retrieval_budget` and `retrieval_seeds` are in `compare_arms.py::PINNED_FIELDS`,
so a control run that omits them differs from its siblings on five fields and the comparison is
refused without `--force`. Pass `--dedup --dedup_threshold <same> --eval_steps <same>` even
though nothing dedupes and `eval_only` never evaluates mid-run. `--retrieval` is the exception:
`ace/ace.py:176` **raises** on any retrieval mode with `--memory none`, so the control is
`--retrieval none` by force. `aggregate_run_costs.py` used to read that forced `none` as a
fairness violation on every correctly-configured three-arm set; the audit now exempts arms whose
`memory_kind` is `none`.

Modes: `offline`, `online`, `eval_only` (pass `--initial_playbook_path` for `eval_only`).
No `--num_samples` flag — to run a subset, point `--config_path` at a config referencing a truncated
JSONL.

The `formula` task in `eval/finance` is the second benchmark, and the one the ACE authors used:

```
uv run python -m eval.finance.run --task_name formula --mode offline \
  --save_path results/formula/preregistered/seed0/text \
  --memory text --generator_model qwen3.5:2b --think_generator off --num_ctx 32768 --seed 0 \
  --skip_initial_test
```

Its flags are identical to MBPP's by design, and unlike MBPP it needs no Docker.
**`formula` is the better instrument for the research question**: 50 formulas appear in every
split with ~5 training and ~4 test instances each, and the authors' preprocessing strips the
formula definition from the prompt, so what the playbook accumulates is directly reusable.
MBPP's problems are independent, so nothing transfers and the curator can only write style
advice — which is what it wrote. See `MBPP_RESULTS.md`, `EXPERIMENT_FORMULA.md`, and for results
**`FORMULA_RESULTS_PREREGISTERED.md`** (the pre-registered runs; seed 0 complete) rather than
`FORMULA_RESULTS.md`, which describes the superseded exploratory runs and is kept for the
mechanism write-up in its §6.

**`task_config_v2.json` is the current split set**
(250 train / **100 val, stratified 2 per formula** / 200 test) and is what the pre-registered
runs use; `task_config.json` is the old default (300 unbalanced val, 2–11 per formula) and is
kept only so the superseded seed-0 runs stay reproducible. Validation selects the checkpoint,
so an unbalanced val set biases which playbook gets called best. `probe_config.json` is the
50/20/20 probe.

**`results/` is gitignored, so its layout is only documented here** (reorganised 2026-08-08;
every run below moved, none were deleted). `--save_path` nests the run one level down, so each
leaf holds `ace_run_<timestamp>_<task>_<mode>/`:

```
results/formula/preregistered/seed{0,1,2}/{none,text,graph}   the nine pre-registered runs
       /formula/seed0_exploratory/{none,text,graph}           superseded, FORMULA_RESULTS.md
       /formula/probes/p{1,2,3}
       /mbpp/arms_qwen_seed0/{none,text,graph}                MBPP_RESULTS.md
       /mbpp/learning_signal_gemma/{none,text}                the pair "The learning signal" reads
       /mbpp/probes/p{1,2,3}
       /_archive/mbpp_preexperiment/                          Jun–Jul, no seed, no num_ctx, mixed models
```

Nothing in `_archive/` is comparable to anything — those runs predate `--seed` and `--num_ctx` —
but `check_memory_equivalence.py` replays real curator operations out of
`_archive/mbpp_preexperiment/mbpp_offline_run/ace_run_20260731_225716_*`, so it is a live
dependency, not dead weight. Moving it silently turns that check vacuous: it returns `[]` and
passes.

**`curator_frequency` defaults to 1 and `max_num_rounds` to 3** as of 2026-08-06. Both were
undisclosed departures from the paper (§4.2: batch size 1, five reflector rounds, five epochs)
and from the upstream finance runner's own defaults; the seed-0 runs used 5 and 1. `num_epochs`
stays at 1 — five multiplies every training-phase call by five — and is a disclosed deviation.
The two runners are changed together, because they are deliberately identical flag for flag.

**The pre-registered runs use `--dedup_threshold 0.88`, not the 0.90 default**, and all three
roles are `gemma4:e2b`. 0.88 was chosen at the probe stage, before any test number, because the
graph arm's projected curator prompt at step 250 landed within 3% of the measured ~31,000
ceiling at 0.90. An offline sweep replaying the probes' recorded ADDs — no LLM, see
`EXPERIMENT_FORMULA.md` §9 — put 0.88 at 34% absorption in both arms against gate 3's 50% limit,
where 0.85 already reached 49% in the graph arm. Going to one model for every role removes the
cross-model distillation in the seed-0 runs, which had a `qwen3.5:2b` generator writing into a
memory curated by `gemma4:e2b`.

There is no test suite. These no-network checks stand in for one; run them before spending
compute:

- `uv run python check_memory_equivalence.py` — proves the text arm is byte-identical to the
  pre-refactor code path, checks the cross-arm invariants, and runs the whole training loop against
  both arms with the LLM stubbed.
- `uv run python check_prompts.py` — proves the MBPP generator prompt is unchanged by prompt
  selection, and that the finance prompts bind their fields correctly. **The finance generator
  template orders its fields playbook → reflection → question → context while
  `Generator.generate` supplies question, context, playbook, reflection**; bound positionally it
  puts the question under the "Playbook" heading in every call, silently. `PromptSet.format_generator`
  is what prevents that, and this check is what proves it.
- `uv run python check_finance_wiring.py` — runs the full offline loop on the real probe splits for
  the text and graph arms with the LLM stubbed; covers processor signatures, prompt selection,
  environment feedback and the artifacts the analysis tools read.
- `uv run python check_learning_signal.py <run_dir>` — see "The learning signal" below.
- `uv run python replay_curator_sanitizer.py <run_dir> [--root results] [--show N]` — replays a
  finished run's `curator_operations_diff.jsonl` through
  `playbook_utils.sanitize_curator_operations` and reports what would have been stripped and
  dropped. Read-only, no LLM. This is where the curator-hygiene rates in the gotcha below come
  from, and running it over the MBPP runs is the check that the sanitiser is inert on clean input.

`uv run python check_curator_ceiling.py --model <name>` also hits the network. It sends real
curator prompts at increasing playbook sizes and reports the largest that still returns usable
operations — which is how the served window gets established, since `--num_ctx` is only a
request. Run it before committing to a longer run than the last one measured; see the `--num_ctx`
gotcha for what it found.

`uv run python check_streaming.py` is the companion check that *does* hit the network: it confirms
streamed calls return content, still report token usage, and produce byte-identical output to
non-streamed ones. Run it after any change to the transport in `llm.py` — a server that quietly
drops usage yields a complete-looking run with no cost data.

`uv run python calibrate_graph_thresholds.py <playbook.txt>` prints what each edge threshold actually
produces on a real playbook. Use it before trusting `--contradicts_threshold`.

After runs finish, `uv run python aggregate_run_costs.py <run_dir> [<run_dir> ...]` (or `--root
results`) sums `detailed_llm_logs/` into tokens, latency and cost per role and per loop phase, writes
`cost_summary.json` into each run dir, and prints a cross-arm table plus a fairness audit. Nothing
else in the harness aggregates cost, so "the graph arm cost X% more tokens for Y accuracy" is only
derivable here. The audit fails a run whose `num_ctx` varies call-to-call (auto-sizing — the arms got
different context windows), whose `num_ctx` was never recorded at all, that had the bulletpoint
analyzer on, or that differs from its sibling arms in seed, models, `test_workers`,
`curator_frequency`, `eval_steps` or `max_num_rounds`.

**`uv run python compare_arms.py <run_dir> ...` is where cross-arm claims come from.** Accuracy
alone cannot tell "the memory did nothing" apart from "the memory fixed 31 problems and broke 31" —
and on MBPP it was the second. It reads `*_test_predictions.jsonl`, prints Wilson intervals, and
runs a **paired McNemar exact test** per arm pair reporting the discordant counts `b` and `c`, not
just p. Given several runs of the same arm it also prints mean ± sd across seeds; **a between-arm
gap smaller than the within-arm sd is seed noise.** It refuses to compare runs that differ on
anything the fairness rules pin, unless `--force`.

**On `formula` it also blocks on the formula, and you should quote that number, not McNemar's.**
The test split is 50 formulas × 4 instances and the instances of one formula fail together, so
McNemar and Wilson — both of which assume iid trials — report the precision of 200 independent
draws for a design with ~50. `cluster_paired_stats` adds a **cluster sign-flip permutation test**
and a **cluster bootstrap CI**, resampling formulas rather than rows. It matters: the seed-0
none→text gap is p=0.0444 under McNemar and **p=0.1249** clustered, so the only significant
result in the run does not survive. MBPP is deliberately absent from `CLUSTER_SOURCES` — its
problems are independent, which is the same reason nothing transfers there.
`uv run python compare_arms.py --selftest` checks the clustered statistics against the two
limits where the answer is known in advance (singleton blocks must reproduce McNemar exact;
perfectly correlated blocks of *k* must widen the interval by exactly √*k*) and needs no runs.

`uv run python rescore_from_logs.py <run_dir>` re-scores a finished run's final test from its saved
generator logs — same responses, current extractor, re-executed. No LLM calls. Use it when a
scoring-side defect is found after the fact instead of re-running arms. It writes
`rescored_test_predictions.jsonl`, which `compare_arms.py` prefers over the live one. It is also the
only way to score runs from before per-problem predictions were persisted.

**`contradicts` edges are off by default (`--enable_contradicts`).** The NLI cross-encoder does not
work on this material: calibrated against a real 24-rule playbook it emitted 19 edges of which ~3
were genuine, scoring "Prioritize writing pure functions" against "Prioritize function composition"
at 1.000 confidence. NLI is trained on descriptive premise/hypothesis pairs; these rules are
prescriptive advice, and the model reads any two directives as competing. Scores are saturated (19
edges at 0.30, 14 at 0.95) so no threshold separates them. The graph arm currently ships with
`related_to` only. Re-enabling it needs a detector that understands prescriptive text — an LLM judge
at ADD time is the obvious candidate.

## Experimental fairness rules

Violating any of these silently invalidates a comparison:

- **Pin `--num_ctx` to the same value in both arms.** `llm.py` otherwise derives Ollama's context
  window from prompt length, and the graph rendering is longer than the flat one (+33% tokens with
  `related_to` only, +109% with contradicts on), so the arms would be decoded under different context
  windows. ACE warns at startup when it is unset.
- `--seed` defaults to 0 and is pinned across arms, but **it does not make a run reproducible and
  it does not vary the data** — see the gotcha below before designing anything around it.
- **`--think` must match across arms** (audited). It sends Ollama's reasoning switch;
  `--think_generator` / `--think_reflector` / `--think_curator` override it per role. **Set it
  per role, not globally.** The roles have opposite economics: the generator is ~80% of all calls,
  so thinking there is what a reasoning model blows its budget on, while the reflector and curator
  make only ~145 calls between them and those calls *are* the learning signal. `gemma4:e2b` is
  itself a reasoning model and was thinking in every run before this flag existed, so a blanket
  `--think off` would silently change the two components whose judgement was just repaired. The
  working configuration is `--think_generator off` alone.
  **The mechanism matters: use `reasoning_effort`, not `think`.** Ollama's own top-level `think`
  field belongs to `/api/chat` and is silently dropped on the OpenAI-compatible route we use.
  Measured on `qwen3.5:2b` against `ollama.caduceus.lol`, same vowel-counting prompt:

  | mechanism | tokens | time | reasoning |
  |---|---|---|---|
  | `think` | 5,716 | 52.8 s | 22,923 chars — ignored, and the harder prompt returned *nothing* |
  | `reasoning_effort` | 169 | 1.8 s | suppressed |

  `llm.py` defaults `think_mechanism` to `reasoning_effort` for that reason. It also records
  `reasoning_chars` on every call and warns once if `think=False` still returns reasoning, because
  a server that does not recognise a key ignores it rather than erroring — the failure is silent
  and looks like success. Verify any candidate with
  `uv run python check_model.py --model <name> --think off` before probing: one minute, no Docker,
  tries each mechanism in turn and also checks the model can hold the output schema. A reasoning
  model left thinking turns hard problems into `no_answer`, which is the same dead end as no
  headroom.
- **`--use_bulletpoint_analyzer` must be off; use `--dedup` instead, and pin it in every arm.**
  The analyzer is text-only and merges bullets with an LLM, so ACE raises rather than silently
  skipping it for the graph. `--dedup` is the symmetric replacement — one decision rule
  (`ace/memory/refine.py::duplicate_groups`) applied by every memory through `Memory.refine()`.
  See "Grow-and-refine" below. Turning it on in one arm only is the worst fairness violation
  available, because it changes what the memory *contains*; `--dedup` and `--dedup_threshold`
  are therefore audited by both `compare_arms.py` and `aggregate_run_costs.py`.
- Hold `test_workers`, `curator_frequency`, `eval_steps`, `max_tokens` and all three model names
  identical. Train order is already deterministic (file order, no shuffle).
- `stats()` must return the same key set for every memory kind — it is interpolated into the curator
  prompt, so an extra key there is a prompt difference disguised as a metric. Structure-specific
  numbers go in `structural_stats()`, which is recorded but never shown to a model.
- **The curator prompt is identical in all three arms**, including the `refines` field that only the
  graph arm acts on. That is deliberate: if the graph arm's curator were prompted differently, the
  arms would differ in two places and no result could be attributed to the memory. The text arm
  simply ignores the field. Describe the baseline honestly as "ACE with an extended curator schema
  whose extra field it ignores".
- `best_memory` must be assigned via `.copy()`, never `=`. The old code relied on strings being
  immutable; a graph assigned directly would alias and "best" would track "current".
- **Prompts are per task, and the two sets bind their generator fields differently.**
  `get_prompts(task_name)` returns a `PromptSet`; anything unrecognised gets the code prompts, so
  adding a task never changes an existing one. `ace/prompts/formula.py` holds the ACE authors'
  finance prompts verbatim apart from two documented deltas (`refines` in the curator, tag
  semantics in the reflector), both applied identically in every arm — see `EXPERIMENT_FORMULA.md`
  §8. Never format a generator template directly: the finance one orders its fields
  playbook → reflection → question → context and the code one is positional, so
  `PromptSet.format_generator` exists to keep that from silently scrambling every call.
  `run_config.json` records `prompt_set`; two runs on different sets are not comparable.

## Grow-and-refine

**The paper has three key innovations, and this repo shipped the third one turned off.** ACE
(§3) names the Reflector, incremental delta updates, and grow-and-refine (§3.2) — *"A
de-duplication step then prunes redundancy by comparing bullets via semantic embeddings"* —
which §4.7 calls a key enabler. `ace/core/bulletpoint_analyzer.py` is that mechanism, and the
fairness rules banned it because it merges bullets and the graph arm could not mirror the
removal.

That decision is what produced the seed-0 formula result. The `text` arm was **ACE minus one
of its three components**, so it degenerated exactly as an append-only memory does: 23 of 75
ADDs verbatim duplicates, 75 rules, 43% of the generator's window. And `refines` — the graph
arm's one capability — was a sparse, LLM-declared re-implementation of the mechanism its
competitor had been stripped of. Neither arm was the thing it was supposed to be. Full
measurement in `FORMULA_RESULTS.md` §6.3.

`--dedup` fixes it symmetrically rather than by restoring the analyzer:

- **One decision, two representations.** `ace/memory/refine.py::duplicate_groups` returns
  `{survivor: [absorbed]}` from the rendered rule order, which is section-then-insertion in
  both arms and therefore identical. `TextPlaybookMemory.refine` deletes the line, which is
  what the paper does; `GraphMemory.refine` adds `survivor --refines--> absorbed` with
  `origin="dedup"` so the existing supersession machinery withdraws it while the node, its
  counters and its edges stay in the graph. Both arms end up showing the same rules.
- **Earliest rule survives**, and absorbed counters are added to it. Keeping the later text
  would restart a track record the reflector had already built, and make the outcome depend on
  which draft the curator wrote more fluently.
- **Embeddings only, no LLM.** The analyzer asks a model to write a merged bullet; that puts a
  model's judgement inside the treatment, adds a call that could differ between arms, and
  reintroduces seed sensitivity into a step that should be deterministic.
- **Similarity is measured against the survivor, never transitively.** At threshold 0.90 a
  transitive closure chains unrelated rules together through a corridor of near-neighbours, and
  which chain it builds depends on iteration order.
- `structural_stats()` splits `refines_edges` into `refines_edges_curator` and
  `refines_edges_dedup`. Do not re-merge them: the curator-declared count is the graph arm's
  headline capability (2 in 50 calls on the seed-0 run) and a combined total would hide it.

Default is **off**, so a run reproduces pre-2026-08-06 behaviour byte for byte and
`check_memory_equivalence.py` still proves the text arm identical to the pre-refactor code
path. Covered by `check_finance_wiring.py` [1e], which asserts both arms reach the same visible
rule set and the same counters from the same operations — if that ever fails, the arms are
running two treatments at once.

## Retrieval — the mechanism that makes this a test of structure

**Until 2026-08-06 the graph arm had no functional use of its graph.** `refines` (eviction) is
reproducible by a flat list with a `supersedes` field and a render-time filter, so it needs no
graph; `related_to` — the only thing that does need one — was written by cosine similarity and
then read only for display text and `structural_stats()`, which no model ever sees.
`render_for_generator(query=...)` raised `NotImplementedError` in both arms. A win either way
would have been attributable to eviction, not to relational structure.

Paper §3.1 names *fine-grained retrieval, "so the Generator can focus on the most pertinent
knowledge"* as a property of the itemized design, so this is a replication gap, not an
extension. `--retrieval` closes it in **both** arms at once — building it in one only would
make it the confound instead of the treatment:

- `none` — inject the whole memory on every call. Reproduces every run before 2026-08-06.
- `flat` — score every live rule against the question, render the highest-scoring that fit.
  Available to both memories; this is the baseline arm's setting.
- `graph` — take the *same* top-`--retrieval_seeds` by the *same* score, then walk `related_to`
  outward breadth-first. Requires `--memory graph`.

Same encoder, same budget, same seed scoring, same prompt template, same emission order
(section then insertion, in both arms — relevance must never reorder what is read, or that is a
second treatment). The single difference is whether the neighbourhood follows edges.
`--memory graph --retrieval flat` is the ablation separating "edges choose what is read" from
"edges are shown alongside what is read".

**`--retrieval_budget` covers the entire generator-facing rendering, legend and relation lines
included.** Budgeting only rule text would hand the graph arm its annotations for free and
reinstate the +33% asymmetry retrieval exists to remove. Charging structure to the same budget
is the honest form of the question: the graph arm shows fewer rules for the same tokens, and
the structure has to pay for itself. It is also aimed at a measured problem — a 75-rule
playbook was 7,103 prompt tokens, 43% of the generator's served ~16,384-token window, spent
before it began to reason.

Two consequences worth knowing:

- **The eval path renders per sample now.** `utils.evaluate_test_set` gained `render_fn`; when
  it is `None` the old single-render-before-fan-out path runs untouched. Rendering happens on
  the main thread, since the encoder is shared and `test_workers` is pinned to 1 anyway.
- **De-duplication supersessions are not rendered** as `refines: … (withdrawn)` lines; only
  curator-declared ones are. The flat arm records nothing when it drops a duplicate, so
  rendering the graph's bookkeeping would charge it tokens for the same decision.

**Is the structure doing anything?** `structural_stats()` records `walk_only_share` — the
fraction of rendered rules the walk reached that relevance alone would not have selected under
the same budget. `EXPERIMENT_FORMULA.md` §7 turns on it: graph ≈ text with that near zero means
the arms were reading the same rules, which is a broken run rather than a negative result about
structure. It costs one extra budget fit per generation (token counting, no encoding).

`--retrieval_budget` and `--retrieval_seeds` are pinned by both fairness audits, and
`aggregate_run_costs.py` fails a set of arms where some retrieve and some do not. Covered by
`check_finance_wiring.py` [1f] — which includes the case that discriminates the arms: four
rules positioned so that relevance ranks an unconnected rule above a two-hop neighbour, and the
walk reverses them — and [2b], the full loop with both mechanisms on.

## The learning signal

Run `20260802_220602` (text arm, full offline) produced test 0.8444 vs the `none` arm's 0.8521 —
but paired per-problem it fixed 17 and broke 19 (McNemar p=0.87). Not an inert memory: a memory
with a large effect and no direction. Three defects underneath it, all now fixed, all shared by
every arm:

- **`harmful` was never emitted.** Across 120 reflections the tags were helpful×98, neutral×47,
  harmful×0 — including 15 `helpful` tags on attempts that had just failed their assertions. The
  reflector prompt now defines the tags by causal impact on *this* attempt and states that
  `helpful` is unavailable when the environment reports failure, and
  `Reflector._filter_bullet_tags` enforces it by demoting `helpful`→`neutral` when
  `attempt_is_correct=False`. It never manufactures `harmful` — blame stays a model judgement, and
  all-`neutral` is a legitimate answer. Prompt *and* code, because the prompt alone is what failed.
- **`refines` was never emitted** — 0 of 24 curator calls, not even as `null`, while 15 of the 25
  final rules sat in 6 near-duplicate clusters ("favor immutable data types" appeared three times).
  Since `refines` is the graph arm's only real capability, a graph run under that curator would
  have been a guaranteed null. The field is now **required-but-nullable** in both curator prompts,
  which is the standard fix for a model skipping an optional key. The validator already tolerated
  `null`, so nothing downstream changed.
- **The reflector never saw why the code failed.** `environment_feedback` was the hardcoded string
  `"Predicted answer does not match ground truth"`, while the sandbox already had
  `Test failed: <assertion> - <exception>` in stderr and threw it away. `answer_is_correct` now
  takes an optional `feedback` dict out-parameter; `utils.check_answer` fills it when the processor
  supports it (detected from the signature, so a processor with the upstream two-arg form still
  works) and `utils.format_environment_feedback` renders it, degrading to the old one-liner when
  nothing was captured. This is the reason accumulated rules came out as generic style advice
  rather than task strategy — the reflector was guessing at causes.

`uv run python check_learning_signal.py <run_dir>` re-checks all of this on any finished run and
exits non-zero if the signal is dead. Check **[6] curator hygiene** was added later, from the
seed-0 formula runs: checks [1]–[5] all passed on the text arm while a third of its curation was
adding second copies of rules it could already see (see the gotcha below). [6] fails a run in
which more than 50% of ADDs add nothing new. Run it on a short probe
(`--config_path ./eval/mbpp_code_exec/data/probe_config.json`, 25 train / 10 val / 10 test) before
committing to a full arm — the failed run above looked healthy in every artifact it wrote.

Do **not** tune the curator prompt toward MBPP to make the rules more specific. Fixing the
curator's *inputs* is a defect fix; steering its *instructions* toward the benchmark after seeing
the results is a researcher degree of freedom that would make any positive result unpublishable.
If the prompt is changed, change it once, before looking at test numbers, identically in all arms.

## Gotchas

- **`eval/mbpp_code_exec` and `eval/finance` (`formula`) are the two working benchmarks.**
  `finer` shares the finance processor and has the right signatures but has never been run here.
  `eval/mind2web*` and `ace/ace_batch.py` were deleted on 2026-08-06: the mind2web processors
  still had the upstream two-arg `answer_is_correct` against a three-arg call site, so they could
  not run, and `ACEBatch` was imported only by them. ~3,000 lines, no other references.
- **The LLM backend is hardcoded to a personal tunnel.** `utils.py::initialize_clients()` ignores
  `--api_provider` and always targets `https://ollama.caduceus.lol/v1` with an empty key and a
  `bypass: true` header. Expected dev setup, not a bug. `--api_provider` still selects
  `max_completion_tokens` vs `max_tokens` in `llm.py`.
- **Docker must be running for any MBPP run.** `DataProcessor.__init__` constructs a `DockerSandbox`
  and pulls `python:3.10-slim`, before any LLM call.
- **`ace/native_deps.py` must be called before importing sentence-transformers.** On this Windows
  install, importing it first kills the interpreter with an access violation inside pyarrow. Without
  the guard `import ace.core` crashes outright and no run starts.
- **A scoring-side defect can be arm-specific, and that invalidates a comparison.** The same
  double-stringification habit also appears bracketed: `"final_answer": "[\"def f(): ...\"]"`.
  It made **28 of the MBPP text arm's 257 test answers unreadable (10.9%) and none in the other
  two arms**; 12 of the 28 held code that passes, so the arm was scored 12 problems low. Measured
  across all parsed generations: text 68/599, graph 6/552, none 0/230. Cause unknown; the obvious hypothesis
  (the top-cited "use list comprehensions" rule) does not hold, at 86.8% citation in wrapped
  generations against 80.8% in normal ones. `unwrap_quoted_source` now handles `["…"]`/`['…']`,
  and `extract_answer` unwraps a genuine one-element array too. **Rescoring repairs the
  measurement, never the training** — those samples were scored as failures at run time and fed
  to the reflector and curator, so the text arm's playbook was built from a noisier signal than
  the graph arm's. Only re-running fixes that.
- **A numeric task breaks `extract_answer` in ways a code task never did.** Two defects, both
  found by the first `formula` probe, both fixed on the extraction side rather than by touching
  the authors' frozen prompt: (1) the generator writes its working in LaTeX inside `reasoning`
  — `\frac`, `\sigma`, `\approx` — which are not legal JSON escapes, so `json.loads` rejected
  the whole envelope with the answer field intact (22 of 29 unparseable probe responses);
  `repair_json_escapes` retries once with the unrecognised backslashes doubled. (2) Both regex
  fallbacks demanded `final_answer` be a *quoted string*, but a numeric answer is written bare
  (`"final_answer": 229.27`, sometimes with a stray closing quote), losing 15 of 29 with the
  right answer sitting in the text. Both paths run only after a strict parse fails, so they
  can never alter a well-formed response — verified: re-scoring the three MBPP arms with them
  in place returns 86/93/86, unchanged. Covered by `check_finance_wiring.py` [1b].
- **`--num_ctx 32768` is not what the server actually serves — measure it, don't assume it.**
  Ollama clamps `num_ctx` to the model's own trained context, silently, and the endpoint's
  answer has changed once already. On `gemma4:e2b` an earlier reflector measurement put the
  working ceiling at **~12,433 tokens**; re-measured 2026-08-08 with
  `check_curator_ceiling.py`, the served window is **32,768**. Pinning `--num_ctx` is still
  required — the clamp is deterministic, so both arms get the same window — but re-run the
  ladder after any endpoint change rather than trusting a number in this file.
  The measured curator ladder, `gemma4:e2b`, `think` default:

  | prompt (server tokens) | outcome |
  |---|---|
  | ~29,600 | ok, 873 output tokens, `refines` still emitted |
  | ~33,900 | `finish_reason: length` at 588 completion tokens, 2,870 chars of hidden reasoning, **empty response** |

  So the practical prompt ceiling is ~31,000: the window minus the ~1,650 tokens a curator call
  needs for its answer plus reasoning. Note the failure shape — over the wall the model does not
  error, it thinks until the budget is gone and returns nothing.
  **`tiktoken` undercounts the server by ~10%.** `utils.count_tokens` uses `cl100k_base`; the
  server's own count of the same prompt ran 1.10× higher (6,890 vs ~6,260 on a real curator
  call). Every token projection in this repo, and every budget derived from one, is that much
  optimistic. Compare against `prompt_num_tokens` in `detailed_llm_logs/`, which is the
  server's number, whenever the answer matters.
  **The generator's window is ~16,384 and the memory is spent out of it.** Measured on the
  `formula` final test, `qwen3.5:2b`: the `none` arm's longest generation is 15,997 output
  tokens on a 445-token prompt, the `text` arm's is 9,293 on a 7,153-token prompt — both land
  within ~60 tokens of 16,384, so that is the wall, not `--max_tokens 16384`. A 75-rule
  playbook is 7,103 prompt tokens at the median and therefore takes **43% of the generator's
  window away from generation**. That is inherent to the method rather than a defect, but it
  is an arm-asymmetric constraint by construction: the memory arm is not "the same model plus
  rules", it is "the same model plus rules and 6.7k fewer tokens to reason in", and it is a
  competing explanation for any accuracy loss the memory arm shows.
- **`--retrieval_budget` bounds the generator's view, not the curator's, and the curator's
  grows for the whole run.** Two renderings, one budgeted and one not: the generator gets
  `render_for_generator(query)` through `ACE._render` (`ace/ace.py:210`), capped at
  `--retrieval_budget`; the curator gets `render_for_curator()` (`ace/ace.py:769`), which is
  `return self.playbook` in `text.py:114` and an unfiltered `_render()` in `graph.py:223`.
  That asymmetry is deliberate and load-bearing — a curator that sees a retrieved slice cannot
  tell whether a rule already exists, nor name the rule a new one supersedes, which is exactly
  what `check_learning_signal.py` [2] and [6] measure. The cost is that the curator prompt
  grows linearly with the memory: measured on the 50-step probes, **+102 tokens/step (text) and
  +113 (graph)**, projecting to 27.3k and 30.0k server tokens by step 250 of a full arm against
  the ~31,000 ceiling above. Run `check_curator_ceiling.py` before assuming a longer run fits,
  and read the last step's `prompt_num_tokens` off a finished run rather than extrapolating.
- **Every role's JSON goes through a different reader, and both had the same blind spot.**
  The generator is parsed by `utils.extract_answer`; the reflector and curator by
  `playbook_utils.extract_json_from_text`. LaTeX in a string field (`\sigma`, `\%`,
  `\approx`) is not valid JSON escaping and rejected **24 of 50 schema-perfect reflector
  responses (48%)** on the formula probe. Both readers now retry once through
  `repair_json_escapes`, after a strict parse has already failed. Note the shape of the
  near-miss: the reflector has a bracket-depth scan fallback that recovered the tags anyway,
  so the damage was invisible in the results and only check [5] showed it — and the **curator
  has no fallback**, so one macro in its response drops a whole operation list silently.
  `repair_json_escapes` must consume valid escapes *whole*: doubling by lookahead alone skips
  a `\\` pair, rescans its second character, and turns `\\approx` into a run of three that is
  still invalid.
- **A runaway generator trace can take the reflector down with it.** The generator's reasoning
  goes into the reflector prompt verbatim, and it is the one field there with no natural bound.
  On the formula probe a 36,268-char trace made up 89.5% of a 40,522-char reflector prompt,
  overflowed the served context, and cost 15 retries plus that step's entire learning signal.
  `Reflector._truncate_trace` now caps it at `MAX_TRACE_TOKENS` (4000), keeping head and tail
  with a marker between. Never extend this to `bullets_used`: that is memory content, it renders
  longer in the graph arm, and truncating it would bias the comparison.
- **Runaway reasoning is a real failure mode on `formula`, not a bug.** `qwen3.5:2b` takes the
  authors' "show your reasoning step-by-step" literally: median response 723 tokens but p90
  8,128 and a tail that reaches `max_tokens`, occasionally emitting no `final_answer` at all
  (~6.5% on the probe). The tail dominates run cost. Do not "fix" it by editing the generator
  prompt — that is the authors' text and steering it toward the benchmark forfeits the
  replication claim.
- **Per-problem predictions must be persisted or no paired test is possible later.**
  `utils.evaluate_test_set` computed per-sample correctness and wrote only aggregates, which is
  why the first three arms needed `rescore_from_logs.py` to be compared at all. It now writes
  `<prefix>_test_predictions.jsonl`. Don't remove it: an unpaired comparison at n=200-257 cannot
  distinguish an inert memory from a directionless one.
- **Some models double-quote the code inside the JSON string.** The generator prompt asks for
  the program as a JSON string field, and `qwen3.5:2b` stringifies it twice, emitting
  `"final_answer": "\"def f():\n    ...\n\""`. After `json.loads` the value still carries literal
  quote characters at both ends, so the sandbox sees a bare string literal instead of a function
  definition and every assertion fails on a name that was never defined. This scores **zero while
  looking exactly like a reasoning problem** — the first probe came back 0.000 train accuracy.
  `utils.unwrap_quoted_source()` strips the wrapper, but only when the answer does not already
  parse and unwrapping makes it parse, so it can turn a syntax error into working code and can
  never alter code that was fine. Measured: 86 of 97 qwen answers were wrapped (10% → 78% parsing,
  test 0.10 → 0.30 rescored), against **0 of 790** for `gemma4:e2b` — which is why the extractor
  had never been exercised on this, and why the fix does not touch the gemma runs at all.
- **Emoji in diagnostics used to kill runs on Windows.** 37 print sites across 8 files carry
  emoji, and on Windows stdout falls back to the ANSI codepage whenever it is not a terminal
  (redirected, piped, captured by a task runner), so printing one raised `UnicodeEncodeError`.
  The worst offenders are inside `llm.py`'s empty-response handler, so it fired exactly when a
  call had already failed — the real error was replaced by an encoding traceback and the process
  died. `llm.py::_make_console_encoding_safe()` now runs at import and relaxes the streams' error
  policy to `replace` (encoding untouched, so output is byte-identical wherever it already
  worked). Every entry point imports `llm`, so this covers the repo. Don't remove it because the
  emoji "look fine in the terminal" — they do, right up until you pipe the run to a log.
- **`--initial_playbook_path` cannot restore a saved graph, and fails silently.** Both runners
  (`eval/finance/run.py:198`, `eval/mbpp_code_exec/run.py:178`) read the path as *text* and hand
  the string to `create_memory`, which for the graph arm calls `GraphMemory.from_playbook_text`
  (`ace/memory/__init__.py:46`) and re-derives every edge by similarity.
  `GraphMemory.load` (`ace/memory/graph.py:561`) does prefer the `.graph.json` sidecar that
  `save()` writes, but no runner reaches it. At `keep_superseded=False` the snapshot contains no
  superseded nodes, so a reload drops every `refines` edge — curator-declared and dedup alike —
  and rebuilds `related_to` over the reduced node set. An `eval_only` graph run, or a re-test of
  an `intermediate_playbooks/` checkpoint, therefore scores **a graph the training run never
  had**, with no warning. Closing it means having the runner detect the sidecar and call
  `GraphMemory.load`; that is a code change to weigh against whatever runs are in flight.
- **Arms may select different checkpoints, and that is not a fairness problem.** `ace/ace.py:1000`
  keeps the best-validation memory with a strict `acc > best_accuracy`, so ties go to the earlier
  and smaller playbook — one rule, applied identically in every arm. On the seed-0 MBPP set the
  graph arm's final test ran on its step-60 playbook and the text arm's on step 120. Forcing a
  common step would test an arm at a checkpoint its own validation rejected, chosen after seeing
  which arm that helps. `compare_arms.py` prints the selected step and the margin it beat; the
  margin is often a few validation rows, so treat it as noise feeding the across-seed sd. It is
  reporting only and deliberately **not** in `PINNED_FIELDS`.
- **Serving throughput on this endpoint is not stable across a long run, and the failure is
  silent.** During the seed-0 graph arm it fell from ~96 tok/s to ~7–10 tok/s inside one hour,
  stayed there twelve hours, then recovered — no error, no flag change, prompt sizes flat
  throughout. ~9 tok/s is CPU-inference speed for a 2B model, so the mechanism is Ollama
  re-deciding GPU/CPU layer placement when it reloads a model; `--num_ctx 32768` makes that
  reload fragile because the KV cache is allocated at load time, but it is not the trigger — the
  same 32k ran on GPU for hours in both arms. `ollama ps` prints the PROCESSOR split, and
  `OLLAMA_KEEP_ALIVE=-1` on the server keeps a run from being evicted mid-flight.
  **Consequence: never quote per-arm `llm_hours` without checking tok/s per hour first.**
  `aggregate_run_costs.py` prints first → last serving speed beside `LLM h` and notes any run
  whose hourly throughput varies more than 3× — a note rather than a fairness violation, because
  the tokens and the accuracy are unaffected and only the timing is contaminated. See
  `EXPERIMENT_FORMULA.md` §9, 2026-08-10.
- **`--seed` is inert, and the run-to-run variance it appears to control comes from somewhere
  else.** The flag reaches exactly one place, Ollama's sampler (`llm.py:215-216`), and at
  `temperature=0.0` the sampler is greedy argmax and never reads it. It does **not** split,
  shuffle, restratify or reorder data: the splits are fixed files in `task_config_v2.json` and
  there is no `random`/`shuffle`/`np.random` anywhere under `ace/`. (`--retrieval_seeds` is
  unrelated — walk start count.) Runs at different seeds still diverge, because GPU
  floating-point reduction order is not stable across batch composition and layer placement, so
  a near-tied argmax flips. Measured on the two `none` arms: **52/200 completions differ at
  temperature 0**, none at character 0, median 2,118-char shared prefix, splitting a median 78%
  through — and long responses diverge far more often than short ones (998 vs 413 median
  tokens). **The ACE loop then amplifies it ~37×**: `none` moves 1 test row between replicates,
  `text` moves 37 (18.5% of the test set), because one flipped token changes a reflection,
  then a curator operation, then the playbook for every remaining step. Consequences: keep
  pinning `--seed` (it costs nothing and covers the empty-retry escalation path at
  `llm.py:411-419`, which does leave temperature 0), never describe a run as reproducible, and
  never treat a between-arm gap smaller than the between-replicate movement as a result.
  Full measurement in `EXPERIMENT_FORMULA.md` §4; what it means for the design, §5.
- **`llm.py` is Ollama-specific.** It auto-sizes `num_ctx` via `extra_body` and has a bespoke
  empty-response retry path that escalates temperature above 0.0 after `empty_retry_escalate_after`
  failures — `call_info["empty_retries"]` records when a call stopped being greedy.
- **Calls stream by default (`stream=True`).** Non-streamed requests send nothing until generation
  finishes, so the tunnel's reverse proxy cannot tell a slow completion from a hung origin and kills
  it with an HTTP 524 — this killed a text-arm run at step 117/120 on a call emitting ~8k tokens.
  Streaming keeps bytes flowing so wall-clock generation length stops mattering. It is a transport
  change only: same model, seed, temperature and `num_ctx`, so output is byte-identical and arms run
  streamed and non-streamed stay comparable. Token counts are *not* recoverable from a stream, so
  `stream_options={"include_usage": True}` is sent alongside; if a server ignores it, `call_info`
  token counts come back `None`, `aggregate_run_costs.py` has nothing to sum, and `llm.py` prints a
  one-time warning. Run `uv run python check_streaming.py` against the live endpoint before
  trusting it — it verifies content, usage, and streamed/non-streamed output equivalence.
- **`--max_tokens` defaults to `16384` (`eval/mbpp_code_exec/run.py:35`), not the `4096` in the
  `ACE` and `timed_llm_call` signatures.** The backend does honour it — a call capped at exactly
  4096 when asked to — but 16384 is a very loose ceiling: at ~112 tok/s a maximal generation runs
  ~146s, which is why long generator calls could outlast a proxy read timeout. Real runs peak around
  8-9k tokens, so lowering it would be safe, but it changes generation and would invalidate any
  already-completed arm. Do not change it mid-experiment.
- **The curator pastes the playbook's rendering into the rule text, and re-adds rules it can
  already see.** Both curator prompts tell it not to (`"no need to include the bullet_id in
  the content"`, `"avoid redundancy"`); `gemma4:e2b` ignores both, at a rate that scales with
  playbook size — **0 of 263 ADDs across every MBPP run, 6/15 on formula probe 3, 43/75 on the
  formula text arm, 34/65 on the graph arm**, one of them nested three deep. A pasted prefix
  puts a snapshot of some *other* rule's counters into the generator prompt as rule text, where
  it never updates again; a verbatim re-add buys a fresh id and a permanent second copy.
  `playbook_utils.sanitize_curator_operations` strips the prefixes (three forms: full counters,
  `helpful=N` alone, bare `[id]`) and drops exact re-adds, and `Curator.curate` calls it.
  Three things about the placement are load-bearing: it is **in `Curator`, not
  `apply_curator_operations`**, so `check_memory_equivalence.py` keeps replaying raw operations
  and proving the text arm byte-identical; it runs **after** `log_curator_operation_diff`, so
  `curator_operations_diff.jsonl` keeps the model's raw behaviour and
  `check_learning_signal.py` [6] stays a read on the model rather than on the fix; and dedup is
  **exact match after normalisation only, never fuzzy or embedding similarity** — semantic
  near-duplicate detection is a graph-arm capability and importing it into the shared write path
  would put the arms' difference in two places. `uv run python replay_curator_sanitizer.py
  --root results` reports the rates on any finished run without touching it.

## Style and lint

Lint with `uv run ruff check .` (`ruff.toml`: line-length 120, rules `E`/`F`/`I`). ~150 pre-existing
findings, mostly from `from utils import *` in `ace/ace.py`. Don't mass-fix or
run `--fix` across the repo; clean up only lines you are already editing. The codebase is not
`ruff format`-ed — don't reformat whole files.

## Git workflow

Commit directly to `main` — solo research repo, no PR ceremony. `origin` is the fork
(`profsuccodifrutta/agentic_context_engineering_project`); `upstream` is `ace-agent/ace`.
