# Pre-registration: relational vs linear memory on `formula`

**Written 2026-08-04, before any `formula` run.** Nothing here may be changed after the first
test number is observed. Changes and their dates go in §9 with a reason, and the affected result
is reported as exploratory.

> **Condensed 2026-08-16, after all nine runs finished. No decision, threshold, test or
> commitment was altered**, §9's engineering narrative was reduced to a dated table and the superseded seed-0 results were moved
> in from the deleted `FORMULA_RESULTS.md` to §10, where the case that depends on them is made.
> Results are in `FORMULA_RESULTS_PREREGISTERED.md`.

## 1. Question

Does representing an agent's accumulated rules as a typed relation graph produce better task
performance than the same rules held as a flat list?

## 2. Why this task

**This experiment stands alone and is analysed alone.** It shares a harness, arms and fairness
rules with `EXPERIMENT_MBPP.md`, and those shared choices are cited where relevant so the two are
*methodologically* comparable. No conclusion here is conditioned on what any other benchmark
finds. Any cross-task reading belongs in a separate synthesis written after both are complete.

**A follow-up experiment exists and does not amend this one.** `EXPERIMENT_COCITE.md`
pre-registers a fourth arm that replaces `related_to` with a co-citation relation, designed
*after* the `text → graph` null here was analysed. It re-runs `text` and `graph` on a later commit
rather than reusing these runs, and its own §5 declares this document's `text → graph` result to
remain the answer to the three-arm question. Nothing below has been changed for it.

`formula` was chosen because it has the transfer structure the method needs in order to be
testable at all, an earlier attempt on a benchmark of mutually independent problems left the
curator nothing to accumulate but style advice:

- 50 distinct finance formulas, **all 50 present in train, val and test**
- 5 training instances per formula (subsampled), ~4 test instances per formula
- zero verbatim question overlap between splits
- the authors' preprocessing **strips the formula definition from the prompt**, so the model must
  recall it, making the formula itself accumulable playbook content
- per-formula conventions are genuinely error-prone: percent vs fraction, rounding, sign, scaling

It is also the task the ACE authors used, so this doubles as a replication check.

## 3. Arms

> **Amended 2026-08-06, after the seed-0 runs were observed.** See §10. The original design
> (§3a) is superseded; its runs are exploratory and excluded from the primary analysis.

Three, differing in exactly one flag pair, the memory and the retrieval it supports:

| arm | `--memory` | `--retrieval` | mode |
|---|---|---|---|
| control | `none` | `none` | `eval_only` |
| baseline | `text` | `flat` | `offline` |
| treatment | `graph` | `graph` | `offline` |

Everything else identical: `gemma4:e2b` in all three roles (see §9, 2026-08-08), generator
reasoning off, `num_ctx=32768`, `temperature=0.0`, `max_tokens=16384`, `max_num_rounds=3`,
`curator_frequency=1`, `num_epochs=1`, `eval_steps=125`, `test_workers=1`, `--dedup` at **0.88**,
`--retrieval_budget 2500`, `--retrieval_seeds 5`, bulletpoint analyzer off, `contradicts` off,
`related_threshold=0.60`, `keep_superseded=false`. Train order is file order, no shuffle.

`--seed 0`, `1`, `2` for each arm: **9 runs**.

**§3a, the superseded design.** The first three runs used `--retrieval none` (whole memory
injected every call), no `--dedup`, `max_num_rounds=1`, `curator_frequency=5`, a `qwen3.5:2b`
generator, and the unstratified 300-row validation split. Recorded so §10's results stay readable
against the configuration that produced them.

## 4. Data

Fixed now, not after seeing results.

| split | file | n |
|---|---|---|
| train | `formula_train_subset_250.jsonl` | 250 (5 per formula, stratified) |
| val | `formula_val_stratified_100.jsonl` | 100 (**2 per formula, stratified**) |
| test | `formula_test.jsonl` | **200** (4 per formula, stratified) |

Referenced by `task_config_v2.json`; `task_config.json` still points at the old validation split
so the §3a runs stay reproducible. Test is the authors' split, untouched, not looked at until §6.

**The test set is 50 formulas × 4 instances, and that is not 200 independent trials.** A playbook
carrying the wrong convention for one formula gets that formula's four rows wrong together, so the
effective sample size is nearer 50. §5 blocks on it.

**`--seed` does not vary the data, and is inert.** Measured 2026-08-11 on the completed `none`
runs. The splits are fixed files,
read in file order, byte-identical in all nine runs; there is no `random`/`shuffle`/`np.random`
under `ace/`. The flag reaches one place, Ollama's sampler (`llm.py:215-216`), and at
`temperature=0.0` the sampler is greedy argmax and never consults it. `eval/finance/run.py:142`
overstates this and should not be believed. (`--retrieval_seeds` is unrelated, walk start count.)

The runs diverge anyway: **52 of 200 `none` arm completions differ between seed 0 and seed 1** on
byte-identical prompts, at `temperature_used` 0.0 with zero `empty_retries`. None diverge at the
first character; the median pair shares 2,118 characters and splits 78% of the way through, and
differing responses are far longer than identical ones (998 vs 413 median tokens). A seeded
sampler would diverge at token 1. This is GPU floating-point reduction order flipping a near-tied
argmax. **The nondeterminism is real and originates in the serving stack; the seed is not its
source.** Consequence for §5: these are *replicates*, not seed variations.

## 5. Primary outcome and analysis

**Primary metric:** accuracy on the 200-row test set at the best-validation checkpoint, scored by
`DataProcessor.answer_is_correct` (exact float equality after `normalize_numeric`).

**Primary comparison:** graph+graph-retrieval vs text+flat-retrieval, paired by problem index,
**blocked on the formula**, seed-blocked across the three replicates, α = 0.05.

**Primary test:** the cluster sign-flip permutation test in
`compare_arms.py::cluster_paired_stats`, with its cluster bootstrap CI. McNemar's `b` and `c` are
reported alongside, never its p-value alone, where the two disagree **the clustered figure
governs**, because McNemar assumes the 200 rows are independent and §4 says they are not.

**Secondary:** text vs none and graph vs none, same test. Three pairwise tests, no correction;
the Bonferroni threshold (α = 0.0167) is quoted where it changes the reading.


**Decision rule.** Graph beats text only if the clustered test is significant *and* the mean gap
across replicates exceeds the within-arm standard deviation. A gap smaller than the replicate
spread is reported as noise regardless of p.

### What the three replicates buy, and what they do not

They estimate the **noise floor**: how far an arm moves when nothing meaningful changes. That is
precisely what the decision rule compares the gap against, and it is undefined at n = 1. They are
*not* draws from the task, §4 establishes the task is frozen so three consequences must be
honoured at analysis time:

- **They cannot narrow the confidence interval.** The bootstrap resamples *formulas*, and every
  replicate sees the same 50, so formula-sampling variance is common and does not shrink.
- **They may not be pooled as trials.** The across-replicate sd is a within-condition noise
  estimate, and **sd/√3 must never be used as a standard error**.
- **The floor is an underestimate**, because training order is held fixed (§9, 2026-08-11).

**Measured at two replicates (2026-08-11):** raw decode noise moves **1 row in 200** (`none`); the
same noise routed through the ACE loop moves **37** (`text`, 18.5% of the test set). One flipped
token changes a reflection, then a curator operation, then the playbook for every remaining step,
**the loop amplifies serving nondeterminism ~37× in rows moved.**

**Read against the effect under test, this is the governing fact of the study.** The seed-0
graph−text gap was +1.0pp and moved 44 rows; the text arm against *itself* at another replicate
moves 3.5pp and 37 rows. The effect is ~3.5× smaller than the instrument's noise. This design can
detect "memory vs no memory" and **cannot** detect a difference of this size between two memory
representations. Stated here so the null is read as a measured power limit, not as equivalence.

### How "seed-blocked" is computed

Fixed **2026-08-12, before the seed-2 runs and before the seed-1 graph arm's result was known** The procedure, `compare_arms.py::pooled_arm_scores`:

1. Require every arm to have the **same replicate count**; refuse otherwise, since unequal
   weighting biases one arm's mean.
2. Intersect the problem indices shared by every run.
3. Score each row as its **mean correctness across that arm's replicates**, a value in
   {0, ⅓, ⅔, 1} rather than a single run's 0/1.
4. Run the same cluster sign-flip permutation test and cluster bootstrap on those averaged rows.

It removes run noise (√R) and **not** formula-sampling variance. So the block count stays **50**
and the row count stays **200**, the pooled row count must never be reported as replicates ×
rows. `compare_arms.py --selftest` asserts both limits.

## 6. Procedure, in order

1. `check_prompts.py`, `check_finance_wiring.py`, `check_memory_equivalence.py` all pass.
2. Working tree committed.
3. **Probe**: both memory arms on `probe_config.json` (50 train / 20 val / 20 test,
   `--eval_steps 25`), then `check_learning_signal.py`. Gates, all of which must pass before
   spending the rest:
   - `harmful` emitted at least once
   - `refines` present in every curator response, at least one non-null
   - train accuracy inside 0.20–0.85
   - output integrity within limits
   - check [6] curator hygiene passes (≤50% of ADDs adding nothing new)
   - de-duplication fires without collapsing the memory: merges > 0 and < 50% of rules
   - the rendered memory is inside `--retrieval_budget` in **both** arms, medians within 10% of
     each other. An arm systematically rendering less than its sibling from the same budget is
     being starved, not treated
4. **Floor/ceiling gate** `none` arm at seed 0 on the real test set. Below 0.15 or above 0.85
   the design is dead. Decided on the *control* arm only.
5. **The nine runs.**
6. Analysis exactly as §5, via `compare_arms.py`; cost via `aggregate_run_costs.py`.

**Gate evidence, 2026-08-08 all gates passed, recorded before the first pre-registered run.**
Learning signal alive in both arms: `harmful` emitted 23× (text) and 22× (graph) against 0 before
the §9 fixes; curator `refines` non-null 23/50 and 16/50 against 2/50 in §3a; 0% unparseable at
every role; duplicate ADDs 14% and 20% against §3a's 31%. Rendered memory held the budget exactly,  medians 2,474 and 2,454 tokens, maxima 2,499 and 2,500 against 2,500, never exceeded so the
arms provably read equal token budgets. **The same 2,500 tokens buys 31 rules in the text arm and
21 in the graph arm**: the legend and relation lines are charged to the same budget, by design,
and that is the exchange rate the treatment has to beat. `walk_only_share` was 0.020 at step 25
and **0.080 at step 50**, rising as the graph densified (36 nodes/73 edges → 73/327), so §7's
falsification condition was met though thinly, the graph arm paying ~10 rules to gain 1.6.
Re-read mid-run at step 125 of the seed-0 graph arm it was **0.150**, and curator prompt growth
was 95.8 server tokens/step, projecting 26,256 at step 250 against the ~31,000 ceiling so the
0.88 decision cleared with ~15% margin.

## 7. What would falsify the hypothesis

Graph ≈ text on the primary comparison, **with the graph mechanism demonstrably doing something**:
the walk must reach rules that relevance alone would not have selected. That is a real negative
result about relational structure and will be reported as one.

Recorded per run as `walk_only_share` the share of rendered rules present in the graph arm's
selection and absent from the flat selection the same query and budget would produce. **If that
share is near zero the arms were reading the same rules and the comparison says nothing about
structure**, a broken run, reported as such.

`refines` is no longer the falsification hinge. §3a showed why: under the superseded curator
prompt it declared 2 supersessions in 50 calls, so the graph arm reduced to text-plus-annotations
and a null there was uninformative rather than negative. Eviction is now shared, both arms
de-duplicate.

## 8. Deviations from the ACE authors' setup

Recorded now, and to be repeated in the write-up.

1. **Models.** `gemma4:e2b` on a local Ollama endpoint, not DeepSeek-V3.1. Absolute accuracy is
   therefore not comparable to the published numbers; only the between-arm contrast is.
2. **Prompts** are the authors' finance prompts (`ace/prompts/formula.py`) with two changes, both
   applied identically in all three arms:
   - `refines`, required-but-nullable, added to the curator schema. It is the graph arm's
     capability; without it the treatment does not exist. Required-but-nullable because on MBPP an
     optional field was emitted 0 times in 24 calls.
   - Tag semantics added to the reflector: tags judged by causal impact on this attempt,
     `helpful` unavailable when the environment reports failure. Upstream defines none of the
     three tag values, and on this stack that produced `harmful` 0 times in 120 reflections, with
     15 `helpful` tags on attempts that had just failed.
3. **`normalize_numeric`** applied to predictions and targets before comparison ($, %, commas,
   quotes, scale words, a lone number inside a sentence). The authors relied on a model that obeys
   "no explanations or other text". Identical in all arms, including the control.
4. **Train subsampled to 250** of 500, stratified to exactly 5 instances per formula, to afford
   three replicates. Val and test untouched.
5. **`contradicts` edges disabled.** The NLI cross-encoder scores prescriptive advice as
   contradictory at saturation. The graph arm ships with `related_to` and `refines` only.
6. **`--think_generator off`.** Left on, the generator spends its output budget before answering.
   Reflector and curator keep model default.
7. **Grow-and-refine was disabled in §3a.** *(Added 2026-08-06; in force from the first run and
   should have been listed here from the start.)* The paper names three key innovations (§3): the
   Reflector, incremental delta updates, and grow-and-refine (§3.2), *"A de-duplication step then
   prunes redundancy by comparing bullets via semantic embeddings"*, which §4.7 calls a key
   enabler. `ace/core/bulletpoint_analyzer.py` is that mechanism, and the fairness rules required
   it off in every run because it merges bullets and the graph arm could not mirror the removal.

   **The `text` arm was therefore not the ACE baseline; it was ACE minus one of its three
   components**, and it degenerated exactly as that predicts (§10). It also means `refines` was a
   sparse, LLM-declared re-implementation of a mechanism the baseline had been stripped of.
   Describing the §3a text arm as "the ACE baseline" is not defensible; "append-only ACE,
   redundancy control removed" is. Fixed symmetrically from §3 onward: both memories get
   de-duplication under one shared decision rule, each in its own representation.
8. **Three training hyperparameters differed from the paper in §3a.** *(Added 2026-08-06.)*
   Paper §4.2: *"We adopt a batch size of 1... We set the maximum number of Reflector refinement
   rounds and the maximum number of epoch in offline adaptation to 5."*

   | parameter | paper | upstream `eval/finance/run.py` | §3a | §3 |
   |---|---|---|---|---|
   | `curator_frequency` | 1 | 1 | **5** | 1 |
   | `max_num_rounds` | 5 | 3 | **1** | 3 |
   | `num_epochs` | 5 | 1 | **1** | **1** |

   The upstream runner's defaults were overridden to match `eval/mbpp_code_exec/run.py` flag for
   flag so nothing differing between the two evals could confound a cross-eval comparison. That
   reasoning stands, but the gap to the paper should have been logged. Paper §4.6 reports these
   knobs move results only modestly. `num_epochs=5` is a 5× multiplier on every training-phase
   call and remains a disclosed deviation.

## 9. Deviations log

Every entry states whether it was decided before or after a test number was observed.

| date | change | timing | why it cannot flatter the result |
|---|---|---|---|
| 08-04 | `utils.extract_answer`: LaTeX-escape repair, and a numeric-literal fallback (both regex fallbacks demanded a *quoted* `final_answer`) | probe 1, before any arm | Runs only after a strict parse fails, so a well-formed response is unchanged. Re-scoring all three MBPP arms gives 86/93/86, identical. Probe usable answers 28 → 43 of 46. |
| 08-04 | `Reflector._truncate_trace`, `MAX_TRACE_TOKENS = 4000` | probe 2, before any arm | The generator trace is the one unbounded field in the reflector prompt; a 36,268-char trace was 89.5% of a 40,522-char prompt and cost 15 retries plus that step's signal. Identical in every arm, and deliberately **not** applied to `bullets_used`, which is memory content and renders longer in the graph arm. |
| 08-04 | Same escape repair in `playbook_utils.extract_json_from_text` (reflector/curator path) | probe 2, before any arm | **24 of 50 reflector responses (48%)** were schema-perfect but rejected. Effect on completed MBPP arms: none the recovered tags are identical to what the existing bracket-scan fallback produced in all 24+18+13 affected calls. The *curator* has no fallback, so one macro silently drops a whole operation list. |
| 08-04 | `eval_steps` 60 → 125 | before any arm | Budget, on call counts alone with no test number in view. 60 would fire four checkpoints × 300 val rows = 1,200 validation generations per arm, more than every other phase combined. Cost: coarser checkpoint selection, identical across arms by construction. |
| 08-06 | `sanitize_curator_operations`, called from `Curator.curate` | **after seed-0 test numbers** | Enforces in code what both prompts already say (*"no need to include the bullet_id"*, *"avoid redundancy"*)  the authors' instructions, not new ones, identically in all arms. Dedup is **exact match after normalisation only**; fuzzy matching would give the flat arm a graph capability. Placed in `Curator` so `check_memory_equivalence.py` still proves the text arm byte-identical, and logged upstream of it so check [6] stays a read on the model. |
| 08-06 | The §3 amendment | **after seed-0 test numbers** | See §10 in full. |
| 08-08 | `gemma4:e2b` in all three roles (supersedes §8.1) | probe stage, before any pre-registered run | §3a had a `qwen3.5:2b` generator writing into a memory curated by `gemma4:e2b`  cross-model distillation inside the treatment. |
| 08-08 | `--dedup_threshold` 0.90 → **0.88** | probe stage, before any pre-registered run | **Headroom, not accuracy.** Curator prompts grew +102 (text) and +113 (graph) tokens/step, projecting 27.3k/30.0k server tokens by step 250 against a ~31,000 ceiling, the treatment arm would have hit the wall mid-run. An offline sweep replaying the probes' recorded ADDs (no LLM, so no seed sensitivity) put 0.88 at 34% absorption in **both** arms; 0.85 already reached 49% in the graph arm against §6's 50% limit. |
| 08-08 | Served context re-measured: **32,768**, not ~12,433 | probe stage | `check_curator_ceiling.py` sends real curator prompts at increasing sizes. Usable at ~29,600 server tokens, empty at ~33,900 (`finish_reason: length`, 2,870 chars of hidden reasoning). Also established: `tiktoken` counts ~10% low against the server, so every projection here is that much optimistic. |
| 08-10 | **The nine runs will not share one `git_commit`** | between seed 0 and seed 1 | §6 asks for one SHA and that is no longer achievable: three commits since seed 0 added tooling and write-ups, and a history rewrite changed every unpushed SHA. What the rule protects is that the arms ran the same *code*, and that is verifiable: `git diff seed0-code-state main -- ace/ eval/ llm.py utils.py playbook_utils.py logger.py` is **empty**. Everything differing is documentation or post-hoc tooling, none of which the run path imports. **Consequence, stated in advance:** `compare_arms.py` will demand `--force`; quote that diff as the licence, and re-run it before the final analysis. |
| 08-10 | Endpoint throughput collapse during the seed-0 graph run — **no change to any arm** | mid-run | Serving fell ~96 → 7–10 tok/s for twelve hours, hitting all three roles by the same factor with prompt size flat, then recovered. 11.71 h of 12.60 h wall was inside LLM calls; graph machinery accounts for ~0.9 h. **Consequence: the graph arm's `llm_hours` must never be reported as a cost of relational memory** — tokens are the honest axis. `aggregate_run_costs.py` now notes any run whose hourly throughput varies >3×. Integrity checked before letting the run stand: 0 empty responses, 0 `empty_retries`, 0 calls missing usage, `no_answer: 0`. |
| 08-11 | Training order **not** shuffled per replicate | between seed 1 and seed 2 | The control that would make replicates meaningful by design. Deliberately not adopted: it would make seed 2 incomparable to seeds 0–1, it violates the 08-10 freeze, and **it could not rescue the primary comparison anyway** — order variation adds variance, making the noise floor larger. It strengthens a null and cannot convert one into a positive; adopting it here would look like an attempt to do exactly that. **Reporting consequence: the within-arm sd in §5 is a lower bound on true run-to-run variance.** |
| 08-12 | `pooled_arm_scores` — the seed-blocked statistic | **before seed 2 ran, and before the seed-1 graph result was known** | The primary comparison had never been implemented. See §5. |

**Freeze, in force from 2026-08-10 until all nine runs finished:** any change under `ace/`,
`eval/`, `llm.py`, `utils.py`, `playbook_utils.py` or `logger.py` invalidates the argument above
and means re-running seed 0. Analysis tools and documentation stayed editable.

## 10. Amendment, 2026-08-06 — after seed-0 results were observed

**Stated first because it is the thing a reader must weigh:** the seed-0 `none`, `text` and
`graph` runs were complete and analysed before this amendment was written. Amending a
pre-registration after seeing results is a researcher degree of freedom, and no justification
makes it not one. What follows is the case that these are defect fixes and replication repairs
rather than result-chasing.

### The runs that prompted it

Three runs, one seed, under §3a. **Exploratory: no conclusion is statistically licensed**, since
the decision rule needs a within-arm sd that does not exist at n = 1. They are on disk at
`results/formula/seed0_exploratory/` and excluded from the primary analysis.

| arm | accuracy (n=200) | vs `none`, clustered | sign-flip p |
|---|---|---|---|
| `none` | **0.6100** | — | — |
| `text` | **0.5350** | −0.0750, CI [−0.165, +0.010] | 0.1249 (McNemar 0.0444) |
| `graph` | **0.5450** | −0.0650, CI [−0.145, +0.010] | 0.1551 |

`text → graph` was +1.0pp, p = 0.9053: 24 fixed, 22 broken, 16 formulas
better and 16 worse. **Both memory arms landed below the no-memory control**, at 21.5× and 26.3×
the tokens, against the paper's Base 67.5 → ACE 85.5 on this benchmark. The one significant
result vanished under blocking, which is the fact that motivated §5's clustered tests.

`check_learning_signal.py` exited 0 on both training arms, so this was not a broken run. Three
findings underneath that healthy scorecard are why the experiment was redesigned rather than
replicated at more seeds:

1. **The curator was doing about a third less work than it appeared to.** It pasted the
   playbook's own rendering into the rule text (`[calc-00013] helpful=5 harmful=0 :: `, once
   nested three deep), **43/75 ADDs in `text`, 34/65 in `graph`**, so a snapshot of some *other*
   rule's counters entered the generator prompt as rule text and never updated again. It also
   re-added rules verbatim, which an ADD-only curator cannot notice: **23/75 and 9/65**. Now
   enforced in code (§9, 08-06). Not retroactively repairable: these runs *trained* on the
   corrupted memory.
2. **The graph arm's one capability fired twice.** `refines` was non-null in **2 of 50** curator
   calls, so 2 of 65 nodes were ever withdrawn. `related_to` edges were written by cosine
   similarity and read only for display text and `structural_stats()`, which no model sees, and
   `render_for_generator(query=...)` raised `NotImplementedError` in **both** arms. The graph arm
   reduced in practice to *text plus 170 annotation edges*, and the dead heat is what that
   predicts.
3. **The baseline was not the paper's method** §8.7. `text` was ACE minus grow-and-refine and
   degenerated as predicted: 31% duplicate ADDs, 75 rules, and a playbook consuming **43% of the
   generator's ~16,384-token window**. What was measured is not "linear memory doesn't help" but
   *"append-only memory without redundancy control doesn't help"*.

The runs are informative about the *harness*, not the hypothesis.

### What changed, and why each is not result-chasing

1. **Grow-and-refine restored, in both arms** (§8.7). **This would have been a defect whatever
   the results had been**, and the direction it corrects is *against* the hypothesis: it repairs
   the baseline the treatment must beat.
2. **Retrieval built, in both arms** (§3). Paper §3.1 lists fine-grained retrieval as a property
   of the design; neither arm had it, so the only functional graph mechanism was eviction, which a
   flat list reproduces, the experiment could not answer its own question in either direction.
   **This is the one change that is not purely a repair.** It is applied identically to both,
   decided before any number from it was observed, and §7's falsification criterion was tightened
   at the same time so a null remains reportable.
3. **`curator_frequency` 5 → 1, `max_num_rounds` 1 → 3** (§8.8). Undisclosed departures from the
   paper and from the upstream runner's own defaults.
4. **Validation stratified**, 300 unbalanced (2–11 per formula) → 100 balanced. Validation selects
   the checkpoint, and an unbalanced validation set biases which playbook is called best. 2 per
   formula is the only evenly-balanced size available: three formulas have just two instances.
5. **Cluster-robust inference** (§5). This change makes the analysis *less* likely to find an
   effect, and it already removed the only significant result on the books.

### What a sceptic should hold against this

- Change 2 is a new capability, not a repair, and was chosen after seeing that the graph arm had
  no functional mechanism. A reader is entitled to treat the next graph-vs-text number as a first
  look rather than a confirmation.
- The §3a runs cannot be salvaged. They trained on a corrupted memory under a baseline missing a
  component; rescoring repairs measurement, never training.
- Everything here was decided by one person with the seed-0 numbers in view. The mitigation is
  that all of it is in code and pinned by the fairness audits before the first new run, not that
  the process was blind.

### Standing

§3a is superseded and excluded from the primary analysis. The nine runs of §3 are the
pre-registered experiment, and this section was frozen as of the commit that started them.
