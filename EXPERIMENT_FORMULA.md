# Pre-registration: relational vs linear memory on `formula`

**Written 2026-08-04, before any `formula` run.** Nothing below may be changed after the
first test number is observed. If something has to change, the change and its date go in
§8 with a reason, and the affected result is reported as exploratory.

## 1. Question

Does representing an agent's accumulated rules as a typed relation graph produce better
task performance than the same rules held as a flat list?

## 2. Why this task

MBPP returned a null (`MBPP_RESULTS.md`) and, more importantly, is structurally unsuited:
its problems are independent, so nothing accumulated on one transfers to another and the
curator can only write style advice. `formula` has the transfer structure MBPP lacks:

- 50 distinct finance formulas, **all 50 present in train, val and test**
- 5 training instances per formula (subsampled), ~4 test instances per formula
- zero verbatim question overlap between splits
- the authors' preprocessing **strips the formula definition from the prompt**, so the
  model must recall it — making the formula itself accumulable playbook content
- per-formula conventions are genuinely error-prone: percent vs fraction, rounding, sign,
  unit scaling

It is also the task the ACE authors used, so this doubles as a replication check on the
implementation.

## 3. Arms

> **Amended 2026-08-06, after the seed-0 runs of §3a were observed.** See §10 for what changed
> and why, and `FORMULA_RESULTS.md` for the results that prompted it. The §3a design is
> superseded; its runs are reported separately as exploratory and excluded from the primary
> analysis.

Three, differing in exactly one flag pair — the memory and the retrieval it supports:

| arm | `--memory` | `--retrieval` | mode |
|---|---|---|---|
| control | `none` | `none` | `eval_only` |
| baseline | `text` | `flat` | `offline` |
| treatment | `graph` | `graph` | `offline` |

Everything else held identical: `qwen3.5:2b` generator with reasoning off, `gemma4:e2b`
reflector and curator at model default, `num_ctx=32768`, `temperature=0.0`,
`max_tokens=16384`, `max_num_rounds=3`, `curator_frequency=1`, `num_epochs=1`,
**`eval_steps=125`**, `test_workers=1`, `--dedup` on at `0.90`, `--retrieval_budget 2500`,
`--retrieval_seeds 5`, bulletpoint analyzer off, `contradicts` edges off,
`related_threshold=0.60`, `keep_superseded=false`. Train order is file order, no shuffle.

`--seed 0`, `--seed 1`, `--seed 2` for each arm: **9 runs**, all pinned to one commit.

### 3a. The superseded design

The first three runs used `--retrieval none` (whole memory injected on every call), no
`--dedup`, `max_num_rounds=1`, `curator_frequency=5`, and the unstratified 300-row validation
split. Kept here because the exploratory results in `FORMULA_RESULTS.md` were produced under
it and must be readable against the configuration that produced them.

## 4. Data

Fixed now, not after seeing results.

| split | file | n |
|---|---|---|
| train | `formula_train_subset_250.jsonl` | 250 (5 per formula, stratified) |
| val | `formula_val_stratified_100.jsonl` | 100 (**2 per formula, stratified**) |
| test | `formula_test.jsonl` | **200** (4 per formula, stratified) |

Referenced by `task_config_v2.json`; `task_config.json` is left pointing at the old validation
split so the §3a runs stay reproducible.

Test is the authors' split, untouched, and is not looked at until §6 runs.

**The test set is 50 formulas × 4 instances, and that is not 200 independent trials.** A
playbook carrying the wrong convention for one formula gets that formula's four rows wrong
together, so the effective sample size is nearer 50. §5 blocks on it.

### What the seed varies, and what it does not

**Measured 2026-08-11, on the completed `none` runs, after the question was asked and before
the answer was assumed.**

`--seed` does **not** resample, restratify or reorder the data. The three splits above are fixed
files referenced by `task_config_v2.json` and are byte-identical in all nine runs, read in file
order with no shuffle; there is no `random`, `shuffle` or `np.random` anywhere under `ace/`.
The flag is forwarded to exactly one place — Ollama's sampler, `llm.py:215-216` — and at
`temperature=0.0` the sampler is greedy argmax and never consumes it. The help text at
`eval/finance/run.py:142` overstates this: it claims the seed buys reproducibility at
temperature 0, which it cannot, because the RNG is not consulted. (Not to be confused with
`--retrieval_seeds`, which is how many rules the graph walk starts from and is unrelated.)

**The runs still differ, and the `none` arm measures by how much.** It carries no memory, so its
prompt for a given test row is identical across seeds and any difference in output is isolated
serving nondeterminism. Comparing `seed0/none` against `seed1/none` (configs differ only in
`seed`):

| | |
|---|---|
| prompts identical | 200 / 200 |
| responses identical | 148 / 200 — so **52 / 200 (26%) differ** |
| `temperature_used` | 0.0 on every call, both runs |
| `empty_retries` | 0 in both runs — the one path that consumes the seed (`llm.py:411-419`) never fired |

**The shape of the divergence is what identifies the cause.** None of the 52 diverge at the
first character; the median pair shares a **2,118-character** prefix and splits a median **78%**
of the way through the response, and differing responses are much longer than identical ones
(median 998 vs 413 tokens). A seeded sampler would diverge early and at token 1. This is
floating-point tie-breaking accumulating per token: GPU reduction order is not fixed across
batch composition and layer placement, addition is not associative in the last bits, and once
two candidate tokens are near-tied the argmax flips. Longer generations get more chances to hit
one. Consistent with the endpoint incident logged in §9, 2026-08-10.

**Conclusion: the seed is inert; the nondeterminism is real and originates in the serving stack.**
The replications are still load-bearing — see §5 — but they are replications, not seed
variations, and the label on them is wrong.

## 5. Primary outcome and analysis

**Primary metric:** accuracy on the 200-row test set, using the best-validation checkpoint,
scored by `DataProcessor.answer_is_correct` (exact float equality after
`normalize_numeric`).

**Primary comparison:** graph+graph-retrieval vs text+flat-retrieval, paired by problem index,
**blocked on the formula**, seed-blocked across the three seeds, α = 0.05. The two arms differ
only in whether the retrieved neighbourhood follows `related_to` edges; both carry the same
de-duplication, the same token budget and the same seed scoring.

**Primary test:** the cluster sign-flip permutation test in
`compare_arms.py::cluster_paired_stats`, reported with its cluster bootstrap CI. The exact
McNemar `b` and `c` are reported alongside, never the p-value alone — but where the two
disagree the clustered figure governs, because McNemar assumes the 200 rows are independent
and §4 says they are not. `compare_arms.py --selftest` checks the clustered statistics against
the two limits where the answer is known in closed form.

**Secondary:** text vs none and graph vs none, same test. Three pairwise tests are reported
together with no correction; any single p < 0.05 is interpreted accordingly.

**Power, stated in advance and corrected.** The pre-amendment version claimed ±7pp from
n = 200 under a Wilson interval. That assumed independent rows. With 50 blocks of 4 the
resolvable effect is roughly **±10pp per seed**, tightening across three seeds. The direction
of the error matters: the earlier figure made the design look more sensitive than it is, and
the one significant result of §3a (none→text, McNemar p = 0.0444) does not survive blocking
(sign-flip p = 0.1249). This design detects a large effect and cannot detect a small one.

**Decision rule.** Graph beats text only if the clustered test is significant *and* the mean
gap across the three seeds exceeds the within-arm standard deviation. A gap smaller than
the seed spread is reported as noise regardless of p.

**What the three seeds buy — and what they do not.** They estimate the **noise floor**: how far
an arm moves when nothing meaningful changes. That is precisely the quantity the decision rule
compares the gap against, and it is undefined at n = 1. They are *not* draws from the task —
§4 establishes the task is frozen — so three consequences follow, all of which must be honoured
at analysis time:

- **They cannot narrow the confidence interval.** The cluster bootstrap resamples *formulas*, and
  every seed sees the same 50 formulas, so formula-sampling variance is common across seeds and
  does not shrink. Three seeds buy the sd, never the precision.
- **The runs are not independent replications and may not be pooled as trials.**
  `compare_arms.py` prints mean ± sd across seeds; that sd is a within-condition noise estimate,
  and **sd/√3 must never be used as a standard error** for the effect.
- **The floor is an underestimate**, because training order is held fixed (see §9, 2026-08-11).

**The floor as measured at two replicates (2026-08-11):**

| arm | seed 0 → seed 1 | rows moved, paired |
|---|---|---|
| `none` | 0.6250 → 0.6200 (−0.5pp) | **1** (0 fixed, 1 broke) |
| `text` | 0.7050 → 0.6700 (−3.5pp) | **37** (15 fixed, 22 broke) |

Raw decode noise moves 1 row in 200. The same noise routed through the ACE loop moves 37 —
18.5% of the test set. One flipped token changes a reflection, which changes a curator
operation, which changes the playbook for every remaining step: **the loop amplifies serving
nondeterminism by roughly 37× in rows moved.**

**Read against the effect under test, this is the governing fact of the study.** The seed-0
graph−text gap was +1.0pp and moved 44 rows; the text arm compared against *itself* at another
replicate moves 3.5pp and 37 rows. The effect is ~3.5× smaller than the instrument's noise, and
the arm-to-arm difference is barely separable from the same-arm-across-replicates difference.
This design can detect "memory vs no memory" (+8 to +9pp) and **cannot** detect a difference of
this size between two memory representations. Stated here so the null in
`FORMULA_RESULTS_PREREGISTERED.md` is read as a power limit that was measured, not as evidence
of equivalence.

## 6. Procedure, in order

1. `check_prompts.py`, `check_finance_wiring.py`, `check_memory_equivalence.py` all pass.
2. Working tree committed; all 9 runs share one git SHA.
3. **Probe** — `--memory text --retrieval flat --dedup --mode offline --config_path
   ./eval/finance/data/probe_config.json` (50 train / 20 val / 20 test, 10 formulas × 5
   training instances, `--eval_steps 25`), and the same for `--memory graph --retrieval graph`.
   Then `check_learning_signal.py`. Gates, all of which must pass before spending the rest:
   - `harmful` emitted at least once
   - `refines` present in every curator response, at least one non-null
   - train accuracy inside 0.20–0.85
   - output integrity within limits
   - check [6] curator hygiene passes (≤50% of ADDs adding nothing new)
   - de-duplication fires and does not collapse the memory: merges > 0 and < 50% of rules
   - the rendered memory is inside `--retrieval_budget` in **both** arms, and the two medians
     are within 10% of each other. An arm systematically rendering less than its sibling from
     the same budget is being starved, not treated
4. **Floor/ceiling gate** — `none` arm at seed 0 on the real test set. If accuracy is below
   0.15 or above 0.85 the design is dead: no memory can move it, and the generator must be
   reconsidered before any further run. This decision is made on the *control* arm only.
5. **The nine runs.**
6. Analysis exactly as §5, via `compare_arms.py`; cost via `aggregate_run_costs.py`.

## 7. What would falsify the hypothesis

Graph ≈ text on the primary comparison, **with the graph mechanism demonstrably doing
something**: the walk must reach rules that relevance alone would not have selected. That is a
real negative result about relational structure, and it will be reported as one.

The measurement is recorded per run: the share of rendered rules reached only by the walk,
i.e. present in the graph arm's selection and absent from the flat selection the same query and
budget would have produced. If that share is near zero the arms were reading the same rules and
the comparison says nothing about structure — a broken run, reported as such.

`refines` alone is no longer the falsification hinge. The §3a runs showed why: the curator
declared 2 supersessions in 50 calls, so the graph arm reduced to text-plus-annotations, and a
null there was uninformative rather than negative. Eviction is also now shared — both arms
de-duplicate — so `refines` is no longer the thing that distinguishes them.

## 8. Deviations from the ACE authors' setup

Recorded now, and to be repeated in the write-up.

1. **Models.** `qwen3.5:2b` / `gemma4:e2b` on a local Ollama endpoint, not DeepSeek-V3.1.
   Absolute accuracy is therefore not comparable to the published numbers; only the
   between-arm contrast is.
2. **Prompts** are the authors' finance prompts (`ace/prompts/formula.py`) with two changes,
   both applied identically in all three arms:
   - `refines`, required-but-nullable, added to the curator schema. It is the graph arm's
     only capability; without it the treatment does not exist. Required-but-nullable
     because on MBPP an optional field was emitted 0 times in 24 calls.
   - Tag semantics added to the reflector: tags judged by causal impact on this attempt,
     `helpful` unavailable when the environment reports failure. Upstream defines none of
     the three tag values, and on this stack that produced `harmful` 0 times in 120
     reflections, with 15 `helpful` tags on attempts that had just failed.
3. **`normalize_numeric`** is applied to predictions and targets before comparison ($ , %
   commas, quotes, scale words, and a lone number inside a sentence). The authors relied on
   a model that obeys "no explanations or other text". Identical in all arms, including the
   control. The un-normalised numbers will be reported alongside.
4. **Train subsampled to 250** of 500, stratified to exactly 5 instances per formula, to
   afford three seeds. Val and test untouched.
5. **`contradicts` edges disabled.** The NLI cross-encoder scores prescriptive advice as
   contradictory at saturation. The graph arm ships with `related_to` and `refines` only.
6. **`--think_generator off`.** Both models are reasoning models; left on, the generator
   spends its output budget before answering. Reflector and curator keep model default.
7. **Grow-and-refine was disabled.** *(Added 2026-08-06; it was in force from the first run
   and should have been listed here from the start.)* The paper names three key innovations
   (§3): the Reflector, incremental delta updates, and grow-and-refine (§3.2) — *"A
   de-duplication step then prunes redundancy by comparing bullets via semantic embeddings."*
   §4.7 calls it a key enabler. `ace/core/bulletpoint_analyzer.py` is that mechanism, and
   `CLAUDE.md`'s fairness rules required it **off in every run**, because it merges bullets
   and the graph arm could not mirror the removal.

   The consequence is not a minor ablation. **The `text` arm was therefore not the ACE
   baseline; it was ACE minus one of its three components**, and it degenerated exactly as
   that predicts — 23 of 75 ADDs verbatim duplicates, 75 rules, 43% of the generator's window
   (`FORMULA_RESULTS.md` §5, §6.3). It also means `refines` was a sparse, LLM-declared
   re-implementation of a mechanism the baseline had been stripped of, rather than a
   capability the flat representation lacks. Describing the seed-0 text arm as "the ACE
   baseline" is not defensible; "append-only ACE, redundancy control removed" is.

   The fix is symmetric rather than restorative: both memories get de-duplication under one
   shared decision rule, each in its own representation, so the arms differ only in structure.
8. **Three training hyperparameters differ from the paper.** *(Added 2026-08-06; in force from
   the first run.)* Paper §4.2: *"We adopt a batch size of 1 (constructing a delta context from
   each sample). We set the maximum number of Reflector refinement rounds and the maximum
   number of epoch in offline adaptation to 5."*

   | parameter | paper | upstream `eval/finance/run.py` | these runs |
   |---|---|---|---|
   | `curator_frequency` | 1 (batch size 1) | 1 | **5** |
   | `max_num_rounds` | 5 | 3 | **1** |
   | `num_epochs` | 5 | 1 | **1** |

   The upstream finance runner's own defaults were overridden to match
   `eval/mbpp_code_exec/run.py` flag for flag, so that anything differing between the two
   evals could not confound a cross-eval comparison. That reasoning stands, but the gap to the
   paper should have been logged here. The paper's §4.6 reports these knobs move results only
   modestly (*"performance changes are modest, and ACE consistently remains above the
   corresponding baselines"*), so they are unlikely to explain a 25-point reversal alone.
   `curator_frequency=1` and `max_num_rounds=3` are adopted going forward; `num_epochs=5`
   is a 5× multiplier on every training-phase call and remains a disclosed deviation.

## 9. Deviations log

**2026-08-04, first probe run, before any arm.** Two extraction defects found by the probe
and fixed in `utils.extract_answer`. Both only run after the strict parse has already
failed, so neither can change how a well-formed response is read, and both are in shared
code that runs identically in all three arms.

- *Invalid JSON escapes.* The generator writes its working in LaTeX — `\frac`, `\sigma`,
  `\approx` — inside the `reasoning` field. Those are not legal JSON escapes, so
  `json.loads` rejected the whole envelope even though the answer field was intact:
  **22 of 29 unparseable probe responses**. `repair_json_escapes` retries once with the
  unrecognised backslashes escaped.
- *Unquoted numeric answer.* Both regex fallbacks required `final_answer` to be a quoted
  string. This task's answer is a number and the model writes it bare
  (`"final_answer": 229.27`, sometimes with a stray closing quote): **15 of 29** were
  discarded with the correct answer present in the text. A numeric-literal fallback was
  added, restricted to a number so it cannot swallow a fragment of a broken object.

Effect on the probe: usable answers rose from 28 to 43 of 46. Effect on the completed MBPP
runs: **none** — re-scoring all three arms gives 86 / 93 / 86 exactly as before, so
`MBPP_RESULTS.md` is unchanged. The probe was re-run after the fix; the pre-fix probe is
discarded.

**2026-08-04, second probe run, before any arm.** Reflector reasoning-trace cap added
(`ace/core/reflector.py::MAX_TRACE_TOKENS = 4000`). At training step 44 the reflector
returned empty fifteen consecutive times. Cause: the generator's reasoning trace is passed
into the reflector prompt verbatim and is the only field there with no bound; a 36,268-char
trace made up 89.5% of a 40,522-char prompt (14,675 tokens). **`--num_ctx 32768` is not what
is served** — Ollama clamps it to the model's own trained context, and on `gemma4:e2b` every
reflector call at or below 12,433 tokens succeeded while the 14,822-token one was left 147
tokens of output budget, spent them all on hidden reasoning, and produced nothing. The step's
learning signal was lost and fifteen retries were paid for it.

The cap keeps the trace's head and tail with a marker between; on the real failing call it
takes the prompt from 14,675 to 4,890 tokens. Applied identically in every arm, and
deliberately **not** applied to `bullets_used`, which is memory content, renders longer in
the graph arm, and would bias the comparison if truncated. It strictly increases the
information the reflector receives — the status quo was an empty response.

The 3 remaining unusable probe responses are genuine model failures, not harness defects:
the generator produced 3k–16k tokens of reasoning and never emitted `final_answer`. At
~6.5% that is inside check [5]'s limit and is reported rather than corrected. Note the cost
shape it implies — median response 723 tokens, p90 8,128 — the tail dominates the budget.

**2026-08-04, second probe run, before any arm.** The same LaTeX-escape defect found in the
generator path was still present in the *reflector and curator* path, and check [5] on probe 2
is what surfaced it: **24 of 50 reflector responses (48%) were rejected as unparseable** while
being schema-perfect, because the model writes `$\sigma_d = 5\%$` and `\approx` inside its
string fields. `playbook_utils.extract_json_from_text` now retries once through
`repair_json_escapes`, exactly as `extract_answer` already did, and only after a strict parse
has failed. `repair_json_escapes` itself was fixed at the same time: it doubled escapes by
lookahead, which skips a `\\` pair and then rescans its second character, turning `\\approx`
into an odd run of three that is still invalid — 10 of the 24 survived the first repair for
that reason.

Effect on probe 2: reflector unparseable **24/50 → 1/50**, and the remaining one is the empty
response from the context overflow that the trace cap above prevents. Effect on the completed
MBPP arms: **none**. The reflector already had a bracket-depth scan fallback for `bullet_tags`,
and the tags recovered by a full parse are **identical to the tags that fallback produced in
every one of the 24 + 18 + 13 affected calls across the probe and both MBPP arms**, so no run's
behaviour changes and `MBPP_RESULTS.md` stands. Re-extracting all 1,791 logged generator
responses under the old and new regex gives **0 changed answers**. What the fix buys is future
robustness — the *curator* has no scan fallback, so one LaTeX macro in its response silently
discards a whole operation list — and a check [5] that measures the model rather than the parser.
Covered by `check_finance_wiring.py` [1b].

**2026-08-04, before any formula run.** `eval_steps` 60 → 125. Reason: budget, decided on
call counts alone with no test number observed. Validation is 300 rows and 250 training
steps at `eval_steps=60` would fire four checkpoints, i.e. 1,200 validation generations per
training arm — more than every other phase combined, and roughly doubling the cost of the
nine runs. 125 gives two checkpoints, the same number MBPP used. The cost is coarser
checkpoint selection (2 candidates per arm instead of 4), which is a known and already
stated weakness of the MBPP design (`MBPP_RESULTS.md` §5.4) and is now identical across
arms by construction.

**2026-08-06, after the seed-0 `none`/`text`/`graph` runs — timing stated plainly.** Curator
hygiene enforced in code: `playbook_utils.sanitize_curator_operations`, called from
`Curator.curate`. This was found while diagnosing the seed-0 text arm's loss, i.e. **after
test numbers were observed**, which is why the boundary matters. It changes neither prompt.
Both curator prompts already say *"no need to include the bullet_id in the content like
'[ctx-00263] helpful=1 harmful=0 ::', the bullet_id will be added by the system"* and
*"avoid redundancy"*; `gemma4:e2b` ignores both. This enforces the authors' own instructions
in shared code, identically in all three arms — the same pattern as
`Reflector._filter_bullet_tags`. Steering the prompt's *content* toward the benchmark remains
out of scope and was not done.

Two behaviours, both measured with `replay_curator_sanitizer.py`:

- *Rendered-bullet prefixes pasted into the rule text.* The curator is shown the playbook as
  `[calc-00013] helpful=5 harmful=0 :: <rule>` and pastes that whole line into `content`, so
  a snapshot of some *other* rule's counters enters the generator prompt as rule text and
  never updates again. Three forms: full counters, `helpful=N` alone, and a bare `[id]` with
  no counters. One text-arm ADD nested three deep. Rates: **43/75 (57%) text arm, 34/65 (52%)
  graph arm, 6/15 (40%) formula probe 3, 0 of 263 ADDs across all twelve MBPP runs.** It
  scales with playbook size, which is why nothing short caught it.
- *Verbatim re-adds.* An ADD-only curator cannot notice that it is re-adding a rule already
  in front of it, so each duplicate buys a fresh id, freshly zeroed counters and a permanent
  second copy in the prompt. **23/75 (31%) text arm, 9/65 (14%) graph arm, 0/15 probe 3,
  0–12% in the MBPP arms** — and 59/63 (94%) in the exploratory `20260701_182251` MBPP run,
  which check [6] now fails.

Dedup is **exact match after normalisation only** (prefixes stripped, whitespace collapsed,
casefolded, trailing punctuation dropped). No fuzzy or embedding similarity: semantic
near-duplicate detection is a `GraphMemory` capability, and putting it in the shared write
path would give the flat playbook a second treatment and make any arm difference
unattributable. The 5 near-duplicates at difflib ≥0.80 in the text arm survive by design —
collapsing those is what `refines` is for, and that is the thing under test. In the graph
arm, `render_for_curator()` excludes superseded nodes, so a withdrawn rule is not in the
dedup set and may be re-added; that is correct, because the rule applied is "do not add what
is in front of you", identically in both arms, and the visible set differing *is* the
treatment.

Placed in `Curator.curate` rather than in `playbook_utils.apply_curator_operations` so that
`check_memory_equivalence.py`, which replays raw operations through that function directly,
keeps proving the text arm's storage layer byte-identical to the pre-refactor code path.
Logging is deliberately upstream of it: `curator_operations_diff.jsonl` still records what
the model emitted, so the new `check_learning_signal.py` check **[6] curator hygiene** stays
a read on the model after the fix. [6] reports the prefix rate and fails a run in which more
than 50% of ADDs added nothing new. Regression cases in `check_finance_wiring.py` [1d].

**Effect on completed runs: none, and none is recoverable.** The three seed-0 formula runs
were trained with the corruption present; rescoring repairs measurement, never training. They
stand as reported and are superseded by the Phase 3 runs rather than corrected.

**2026-08-10, between seed 0 and seed 1. The nine pre-registered runs will not share one
`git_commit`, and the fairness audit will say so.** §6 asks for one SHA across all runs. That
is not achievable now: seed 0 ran at `1acfe69`, and three commits since then added the
throughput profiler, the checkpoint reporter and the seed-0 write-up. Rewriting history on
2026-08-10 to strip a commit-message trailer changed every unpushed SHA as well, without
changing a byte of any tree.

What the rule is actually protecting is that the arms ran the same *code*, and that is
verifiable and holds:

```
git diff seed0-code-state main -- ace/ eval/ llm.py utils.py playbook_utils.py logger.py
```

is empty. Everything that differs between seed 0's commit and `main` is documentation
(`CLAUDE.md`, this file, the two results files) or post-hoc analysis tooling
(`compare_arms.py`, `aggregate_run_costs.py`), and nothing under `ace/` or `eval/` imports
either tool. The `seed0-code-state` tag keeps the commit named in the seed-0
`run_config.json` files resolvable now that it is off `main`.

**Consequence, stated in advance rather than discovered at analysis time:** the nine-run
`compare_arms.py` will report a `git_commit` disagreement and require `--force`. Run it with
`--force` and quote the `git diff` above as the justification. Re-run that diff before the
final analysis — if it is ever non-empty, the exemption does not apply and the affected runs
are not comparable.

**Before seed 1 and seed 2, freeze the run path.** Any further change under `ace/`, `eval/`,
`llm.py`, `utils.py`, `playbook_utils.py` or `logger.py` invalidates the argument above and
means re-running seed 0. Analysis tools and documentation stay editable.

**2026-08-11, between seed 1 and seed 2 — `--seed` does not vary training order, and that is a
limitation of this round rather than a defect to fix mid-flight.** §4 now records what the flag
does (nothing at temperature 0) and what the replications therefore measure (serving
nondeterminism, amplified ~37× by the loop). What they do *not* measure is curriculum
sensitivity: train order is file order in every run, so an accumulating memory always sees the
same 250 examples in the same sequence.

Shuffling training order per replicate is the control that would make the replications
meaningful **by design** rather than by accident, and it is the first change any round 2 should
make. It is **deliberately not adopted now**, for three reasons stated before the remaining runs
rather than after:

1. It would make seed 2 incomparable to seeds 0 and 1, and re-running those is not affordable.
2. It changes the run path, which §9's 2026-08-10 entry freezes until all nine runs finish.
3. **It would not rescue the primary comparison anyway.** Order variation adds a variance source
   on top of serving noise, so it makes the noise floor *larger*. It strengthens a null; it
   cannot convert one into a positive. Adopting it now would look like an attempt to do exactly
   that.

**The consequence for reporting:** the within-arm sd in §5 is a *lower bound* on this method's
true run-to-run variance, and must be described that way. A round 2 that shuffles order could
decompose the variance into serving nondeterminism versus curriculum order, which is a genuine
analysis — but it answers "how stable is ACE?", not "does structure beat linearity?".

**One test this round can still run, and should.** §4's amplification figure compares `none`
against `text` only, so what is established is that *memory* amplifies nondeterminism — **not**
that *structure* does. Once the seed-1 graph arm lands, the same paired-flip count on
`graph` seed 0 vs seed 1 answers whether the graph arm is more or less stable than the flat one
under identical conditions. Report it whichever way it comes out; it is descriptive and no
hypothesis was pre-registered on it.

## 10. Amendment, 2026-08-06 — after seed-0 results were observed

**Stated first because it is the thing a reader must weigh:** the seed-0 `none`, `text` and
`graph` runs were complete and had been analysed before this amendment was written. Amending a
pre-registration after seeing results is a researcher degree of freedom, and no amount of
justification makes it not one. What follows is the case that the changes are defect fixes and
replication repairs rather than result-chasing, plus the parts of that case a sceptic should
weigh against it.

### What the seed-0 runs showed

`none` 0.6100, `text` 0.5350, `graph` 0.5450 on the same 200-row test set. Both memory arms
below the no-memory control, at 21.5× and 26.3× the tokens; graph and text indistinguishable
(+1.0pp, sign-flip p = 0.91). The paper reports Base LLM 67.5 → ACE 85.5 on this benchmark
(Table 2). Full write-up in `FORMULA_RESULTS.md`.

### What changed, and why each is not result-chasing

1. **Grow-and-refine restored, in both arms** (§8.7). The paper names it one of three key
   innovations and §4.7 calls it a key enabler; it was disabled here from the first run. The
   baseline was therefore never the ACE baseline. **This would have been a defect whatever the
   results had been**, and the direction it corrects is *against* the hypothesis: it repairs
   the baseline the treatment must beat.
2. **Retrieval built, in both arms** (§3). Paper §3.1 lists fine-grained retrieval as a
   property of the design; neither arm had it. Without it the only functional graph mechanism
   was eviction, which a flat list reproduces, so the experiment could not answer its own
   question in either direction. **This is the one change that is not purely a repair** — it
   adds a capability neither arm had. It is applied identically to both, decided before any
   number from it was observed, and the falsification criterion (§7) was tightened at the same
   time so a null remains reportable.
3. **`curator_frequency` 5 → 1 and `max_num_rounds` 1 → 3** (§8.8). Undisclosed departures from
   the paper's §4.2 and from the upstream finance runner's own defaults. `num_epochs` stays at
   1 and remains disclosed, because 5 multiplies every training-phase call by five.
4. **Validation stratified**, 300 unbalanced (2–11 per formula) → 100 balanced (2 per formula).
   Validation selects the checkpoint, and an unbalanced validation set biases which playbook is
   called best. 2 per formula is the only evenly-balanced size available: three formulas have
   just two validation instances.
5. **Cluster-robust inference** (§5). The test set is 50 × 4 and was being analysed as 200
   independent rows. This change makes the analysis *less* likely to find an effect, and it
   already removed the only significant result on the books.

### What a sceptic should hold against this

- Change 2 is a new capability, not a repair, and was chosen after seeing that the graph arm
  had no functional mechanism. A reader is entitled to treat the next graph-vs-text number as
  a first look rather than a confirmation.
- The §3a runs cannot be salvaged. They trained on a corrupted memory (§9, 2026-08-06) and
  under a baseline missing a component; rescoring repairs measurement, never training.
- Everything here was decided by one person with the seed-0 numbers in view. The mitigation is
  that all of it is in code and pinned by the fairness audits before the first new run, not
  that the process was blind.

### Standing

§3a is superseded. Its runs are reported as exploratory in `FORMULA_RESULTS.md` and excluded
from the primary analysis. The nine runs of §3 are the pre-registered experiment, and this
section is frozen as of the commit that starts them.

---

**2026-08-06, before the amended §3 runs.** The four code changes the amendment depends on,
with what each was verified against:

- `--dedup` / `--dedup_threshold` (`ace/memory/refine.py`, `Memory.refine`). Off by default, so
  `check_memory_equivalence.py` still proves the text arm byte-identical to the pre-refactor
  path. `check_finance_wiring.py` [1e] asserts the cross-arm invariant that matters: the same
  operations produce the same visible rules and the same counters in both memories.
- `--retrieval` / `--retrieval_budget` / `--retrieval_seeds` (`ace/memory/retrieval.py`).
  `check_finance_wiring.py` [1f] includes the discriminating case — four rules positioned so
  relevance ranks an unconnected rule above a two-hop neighbour and the walk reverses them —
  and [2b] runs the full loop with both mechanisms on.
- Cluster-robust statistics (`compare_arms.py::cluster_paired_stats`), checked by
  `compare_arms.py --selftest` against the two limits with closed-form answers: singleton
  blocks must reproduce McNemar exact, and perfectly correlated blocks of *k* must leave the
  point estimate alone and widen the interval by exactly √*k*.
- `walk_only_share` in `GraphMemory.structural_stats()`, which is what §7 is stated in terms of.

`--num_ctx 32768` is still requested and still clamped by Ollama to roughly 16,384 on
`qwen3.5:2b`. That mattered before because whole-memory injection spent 43% of the generator's
window; with `--retrieval_budget 2500` it no longer differs between arms, so raising it is
worth doing but is no longer a fairness issue.

**2026-08-08, probe stage, before the first pre-registered run.** Both arms probed on
`probe_config.json` (50/20/20) with `gemma4:e2b` in all three roles. Every §6 gate passed;
three changes follow, all decided on probe data with no test number from the amended design
in view.

*Gate evidence.* Learning signal alive in both arms. `harmful` emitted 23× (text) and 22×
(graph) against 0 before the fixes; curator `refines` non-null 23/50 and 16/50 against 2/50 on
seed 0; 0% unparseable at every role; duplicate ADDs 14% and 20% against 31%. Rendered memory
held the budget exactly — medians 2,474 and 2,454 tokens, maxima 2,499 and 2,500 against 2,500,
never exceeded — so the arms provably read equal token budgets. The same 2,500 tokens buys
**31 rules in the text arm and 21 in the graph arm**: the legend and relation lines are charged
to the same budget, by design, and that is the exchange rate the treatment has to beat.
`walk_only_share` was 0.020 at step 25 and **0.080 at step 50**, rising as the graph densified
(36 nodes/73 edges → 73/327). §7's falsification condition is therefore met — the walk reaches
rules relevance alone would not select — but the channel is thin, and the graph arm is paying
about 10 rules to gain 1.6.

1. **Models: `gemma4:e2b` in all three roles.** The seed-0 runs used a `qwen3.5:2b` generator
   with `gemma4:e2b` reflector and curator, so the memory guiding one model was written by
   another. Supersedes §8.1 for the pre-registered runs. Absolute accuracy remains
   non-comparable to the published numbers either way.
2. **`--dedup_threshold` 0.90 → 0.88.** Not a tuning-for-accuracy change: it is a headroom
   change, forced by the measurement in (3). Curator prompts grew +102 tokens/step (text) and
   +113 (graph) over the probes, projecting to 27.3k and 30.0k server tokens by step 250
   against a measured practical ceiling of ~31,000 — i.e. the **treatment arm would have
   reached the wall inside the run**. An offline sweep replaying the probes' 114 and 91
   recorded ADDs through the real memory classes (no LLM, so no seed sensitivity) gave:

   | threshold | text absorbed | graph absorbed | graph render |
   |---|---|---|---|
   | 0.90 | 27% | 29% | — |
   | **0.88** | **34%** | **34%** | **−8%** |
   | 0.85 | 39% | **49%** | −21% |
   | 0.80 | 47% | **63%** | −34% |

   0.85 and below put the graph arm at or past §6's "dedup must not collapse the memory" limit
   of 50%. 0.88 buys ~11% margin while leaving both arms at the same absorption rate. Applied
   identically in every arm and pinned by both fairness audits.
3. **Served context re-measured: 32,768, not ~12,433.** `check_curator_ceiling.py` (added with
   this entry) sends real curator prompts at increasing playbook sizes. `gemma4:e2b` returned
   usable operations at ~29,600 server tokens and failed at ~33,900 — `finish_reason: length`,
   588 completion tokens, 2,870 chars of hidden reasoning, empty response. The endpoint's window
   had changed since the figure recorded in `CLAUDE.md`, in the favourable direction, and the
   failure mode over the wall is silent. Also established: `tiktoken` (`cl100k_base`, what
   `utils.count_tokens` uses) counts ~10% low against the server, so every projection in these
   documents is that much optimistic unless taken from `prompt_num_tokens`.

One tooling fix, no effect on any result: `aggregate_run_costs.py` flagged every correctly
configured three-arm set as a fairness violation, because `ace/ace.py:176` forces the control to
`--retrieval none` and the audit read that as "some arms retrieve and some do not". The audit now
exempts arms whose `memory_kind` is `none`.

**2026-08-10, during the seed-0 graph run — an endpoint incident, no change to any arm.** The
graph arm was 12.6 h in at step 168/250 while the text arm had run all 250 steps in 4.96 h.
The suspicion was that the graph structure is that expensive. It is not, and the record needs
to say so plainly because a 2.5× wall-clock ratio is exactly the kind of number that gets
quoted as a property of the memory.

*What the logs show.* 11.71 h of the 12.60 h wall was inside LLM calls; NetworkX, the BFS walk
and the encoder account for ~0.9 h across 167 steps. The median generator prompt was **flat at
3,280 → 3,337 server tokens across the whole run** — `--retrieval_budget 2500` held, nothing
grew. Serving throughput fell from ~96 tok/s to ~7–10 tok/s between the 01:00 and 02:00 buckets
on 2026-08-10, stayed there for twelve hours, and recovered to ~71 tok/s at ~15:00 after the
endpoint owner intervened. The drop hit generator, reflector and curator by the same factor and
was independent of prompt size. A uniform ~10× collapse with prompt size constant is the
endpoint, not the workload.

*Why `--num_ctx 32768` is not the explanation on its own,* though it was the first hypothesis:
the text arm logged `num_ctx=32768` on all 1,577 of its calls and was served at 84–101 tok/s
throughout, and the graph arm's own first 2.5 h ran at 96 → 63 tok/s at the same value. `num_ctx`
sets the KV-cache allocation Ollama makes **when a model is loaded** and does not grow during a
run, so it can only bite at a load event. The consistent reading is a mid-run evict-and-reload
under VRAM contention, at which point weights plus a 32k KV cache no longer fit and layers were
placed on CPU. `--num_ctx` was not changed: it is pinned across arms, the text arm had already
run at 32k, and §9's curator projection needs the window.

*Integrity, checked before deciding to let the run stand.* 0 empty responses, 0 `empty_retries`,
0 calls missing token usage, all 100 step-125 validation generations logged, `no_answer: 0`.
Tokens, accuracy and every artifact the analysis reads are unaffected.

*Consequences.* (1) **The graph arm's `llm_hours` and `wall_hours` must not be reported as a cost
of relational memory.** Token counts are the honest cost axis for this arm; use them.
(2) `aggregate_run_costs.py` now computes per-hour completion tok/s, prints first → last serving
speed beside `LLM h`, and raises a note — not a fairness violation, since the run's *outputs* are
unaffected — when a run's hourly throughput varies by more than 3×. Verified: the graph arm
raises it at 36.3×, the text arm and all three MBPP arms do not. (3) The honest caveat: if the
mechanism was GPU→CPU offload, the second half of the graph arm was decoded under different
kernels. Same model, seed, temperature and `num_ctx` — no pinned setting was violated — but not
bit-identical to its own first half or to the text arm.

*Two pre-registered numbers, read at step 125 while the evidence was fresh.* Graph curator prompt
growth is **95.8 server tokens/step**, projecting **26,256** at step 250 against the ~31,000
practical ceiling: it clears with ~15% margin, so the 0.88 threshold decision above was
sufficient. `walk_only_share` is **0.150** (2.88 of 19.2 rendered rules per call reached only by
the walk), up from 0.080 at probe step 50 — §7's falsification condition is met with room to
spare, and by a wider margin than the probe suggested.
