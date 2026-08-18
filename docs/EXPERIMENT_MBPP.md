# Pre-registration — MBPP, linear vs relational memory

**Status: written 2026-08-15, before any run under this design exists. Nothing here has been
executed.** The runs described in `MBPP_RESULTS.md` are **not** this experiment and do not
count toward it; §9 says why, item by item.

This mirrors `EXPERIMENT_FORMULA.md` deliberately. Same arms, same fairness rules, same
analysis, same decision rule — so that the two benchmarks differ in the *task* and in nothing
else. Where this document departs from the formula design, the departure is flagged and
justified in place.

---

## 1. Question

Does relational memory beat linear memory for agent execution, on a task whose problems share
**no transferable content**?

**This experiment stands alone and is analysed alone.** It shares a harness, a set of arms, a set
of fairness rules and a decision rule with `EXPERIMENT_FORMULA.md`, and those shared choices are
cited throughout so the two are *methodologically* comparable. What it does not share is
interpretation: no conclusion drawn here is conditioned on what the formula runs found, and none
of formula's conclusions depend on what happens here. The two are independent tasks tested against
the same hypothesis, and each result is reported on its own evidence.

Any cross-task reading — whether the two together say something about *when* accumulated memory
helps — belongs in a separate synthesis written after both are complete and conditioned on what
both actually show. It does not belong in either task's own analysis, and it must not be assumed
in advance by either.

## 2. Why this task — and what it can and cannot answer

**A null here is a result, not a failed replication.** State this plainly before running
anything, because the temptation afterwards will be to read it as a failure of the method rather
than as a property of the task.

Each MBPP-sanitized problem is a self-contained algorithmic exercise. Nothing learned on
problem 7 applies to problem 84 except style. A curator therefore has nothing to accumulate but
style advice — which is a claim about **this** benchmark, verifiable from this benchmark's own
artifacts (below), and not an inference from any other task.

### What the curator actually writes here, and why it cannot help

"Nothing transfers" understates it, and the probes of 2026-08-17 show the mechanism precisely
enough to state in advance. It is not that the accumulated rules are merely *inert*: the loop
has a positive tendency, and it runs somewhere useless.

Measured on the `text` probe's final playbook (30 train steps, 117 generations):

- **The rules are software-engineering advice, not task strategy.** Context managers for file
  handles, `copy.deepcopy()` for nested structures, structured logging, custom exception
  hierarchies, Abstract Base Classes for decoupling. MBPP asks for one self-contained function
  with no shared state, no I/O and no resources, so **none of it can ever apply**. The reflector
  is asked what would have helped on a failed attempt and, having no task-level answer
  available, reaches for generic good practice.
- **The counter signal collapses.** 21 of 28 rules end at `helpful=0 harmful=0` — never cited
  once. Three rules carry the entire citation signal. Grow-and-refine prunes by redundancy and
  the counters are what the curator reads to judge value, so with 75% of the playbook unrated
  there is nothing to prune *by* and nothing to promote.
- **The redundancy is paraphrastic, so embedding de-duplication does not catch it.** 14 of the
  20 rules in `COMMON MISTAKES TO AVOID` restate one piece of advice about mutable arguments and
  defensive copying. They sit at cosine 0.86–0.89 to each other — distinct sentences saying one
  thing, which is the hardest case for §3.2's mechanism and the reason the threshold has to be
  recalibrated here (§6 step 4) rather than inherited from anywhere. The superseded MBPP runs
  found the same shape independently — `MBPP_RESULTS.md` §5.5, 14 near-verbatim restatements of
  "favor immutable data types" among 24 rules — so this is a property of the task, not of one
  probe.
- **The one rule that is read is net-harmful.** `[code-00009] helpful=3 harmful=4 :: Favor list
  and dictionary comprehensions`. So the memory's realistic effect on this task is not zero in
  expectation — it is a small negative drawn from a budget the control does not spend.

The `graph` probe accumulates the same *content* with visibly better hygiene — 20 live rules
against 28, three withdrawn by curator `refines`, the purity/immutability family gathered under
`related_to`, and 8 of 20 rules cited against 3 of 28. **That is a difference in bookkeeping,
not in usefulness**, and it is the reason a positive `text → graph` result here would still not
license "structure helps": on this task there is no knowledge for structure to organise.

**The pre-registered expectation is therefore a null on all three comparisons.** The reason to
record it is not that a null would confirm anything — it is that recording the expected direction
in advance removes the degree of freedom that would otherwise be available afterwards: quietly
reframing MBPP as "exploratory" once it comes back null, or presenting a positive result as
predicted when it was not.

Either outcome is a result on its own terms. A null is evidence that accumulated memory does not
help on a task with no reusable content, with the ceiling in §7 as a competing explanation that
must be reported alongside it. A positive result is evidence that it helps even without reusable
content, which would mean the mechanism is something other than content transfer and would be the
more interesting finding. **Neither reading requires, or is permitted to borrow, the formula
result.**

## 3. Arms

Unchanged from `EXPERIMENT_FORMULA.md` §3. Three arms, one independent variable:

| arm | `--memory` | `--retrieval` | what it answers |
|---|---|---|---|
| control | `none` | `none` | does memory help at all? |
| baseline | `text` | `flat` | the ACE baseline |
| treatment | `graph` | `graph` | does *structure* help? |

The graph arm's capability under test is that **retrieval follows `related_to` edges**. Both
memory arms score every live rule against the question and render the highest-scoring under one
shared token budget; the graph arm then walks outward from the same seeds. `refines` is
eviction, reproducible by a flat list with a `supersedes` field, and is explicitly **not** the
distinguishing capability.

`contradicts` edges stay **off**. The NLI cross-encoder does not work on prescriptive text, so
"graph" here means `related_to` only — the same reduced treatment the formula runs used, kept
identical so the two benchmarks are comparable to each other. See §10 for the extension.

## 4. Data

| | |
|---|---|
| source | MBPP-sanitized, `eval/mbpp_code_exec/data/` |
| old splits | 120 train / **43 val** / 257 test (`task_config.json`, superseded runs only) |
| **splits in use** | 120 train / **85 val** / 215 test (`task_config_v2.json`, created 2026-08-15) |
| probe | 30 / 10 / 10 (`probe_config.json`) |

**The one data change to make before running: enlarge the validation split.** At 43 rows a
validation estimate carries roughly ±15pp of noise, and it is what selects the checkpoint each
arm carries into the test set. In the superseded runs the two memory arms tied at val 0.372 on
43 rows and were sent to the test set with different playbooks on that basis
(`MBPP_RESULTS.md` §5.4). That is model selection driven by noise.

Moving 42 rows from test to val costs power — the half-width goes from ±5.8pp to ±6.3pp at
n=215 — and buys a validation estimate at ±10pp instead of ±15pp. **Make this change once,
before any run, and never after seeing a test number.** This mirrors the formula design, where
`task_config_v2.json` exists for the same reason (there, to stratify val at 2 per formula).

Splits are fixed files read in order with no shuffle, identical in all nine runs.

### What `--seed` varies: nothing

Carried over from `EXPERIMENT_FORMULA.md` §4 and true of this harness generally, not of that
task. `--seed` reaches exactly one place, Ollama's sampler (`llm.py:215-216`), and at
`temperature=0.0` the sampler is greedy argmax and never consults it. It does not resample,
restratify, reorder or shuffle. Runs still diverge, because GPU floating-point reduction order
is not stable across batch composition and layer placement.

**So the three runs per arm are replicates, not seed variations**, and their purpose is to
measure the noise floor. On `formula` that floor was 0.7 rows for `none` and 37.3 rows for
`text` out of 200 — the loop amplifies serving nondeterminism ~37×. **That figure must be
re-measured here and not assumed**; the `none` arm measures it directly, since it carries no
memory and its prompt for a given row is identical across replicates.

## 5. Primary outcome and analysis

**Primary metric:** accuracy on the test split at the best-validation checkpoint, scored by
`DataProcessor.answer_is_correct` — sandbox execution of the extracted program against the
problem's assertions.

**Primary comparison:** `graph` vs `text`, paired by problem index, pooled across the three
replicates, α = 0.05.

**The tests, and why they differ from the formula analysis.** This is the one substantive
departure, and it goes in MBPP's favour:

| | `formula` | **MBPP** |
|---|---|---|
| row structure | 50 formulas × 4 correlated instances | **200+ independent problems** |
| unit of resampling | the formula | **the row** |
| McNemar exact | anti-conservative, reported but not governing | **valid, and reported** |
| primary test | cluster sign-flip + cluster bootstrap | **sign-flip + bootstrap over singleton blocks** |

MBPP is deliberately absent from `compare_arms.py::CLUSTER_SOURCES` because its problems really
are independent — which is the same fact that makes nothing transfer (§2). So McNemar's
independence assumption holds here and its p-value is not inflated.

**Pooling across replicates.** Each row is scored as its mean correctness across an arm's three
replicates — a value in {0, ⅓, ⅔, 1} — and those real-valued rows go through the sign-flip
permutation test and bootstrap CI **with every row as its own block**. McNemar is undefined on
averaged scores (`b` and `c` count discordant *binary* pairs), and the permutation test with
singleton blocks is exactly its permutation form: `compare_arms.py --selftest` asserts that
singleton blocks reproduce McNemar exact to within Monte Carlo error.

**This required a code change, made 2026-08-15, before any MBPP replicate existed.**
`report_pooled` previously printed pooled accuracies for a non-clustered task and ran no test at
all, so the comparison this document names could not have been computed. The change is a
fallback to singleton blocks; it cannot alter the formula results, which supply cluster labels
and take the other branch (verified: the formula output is unchanged). Timing is the point —
choosing a pooled statistic after seeing which answer it gives is the move this document exists
to prevent.

**What pooling buys and what it cannot.** It removes run noise (by roughly √R) and does nothing
to problem-sampling variance. **The pooled analysis is 215 rows, not 645.** Replicates buy the
sd, never the precision; `sd/√3` must never be used as a standard error.

**Secondary:** `text` vs `none` and `graph` vs `none`, same tests. Three pairwise comparisons
reported together with **no multiple-comparison correction**; the Bonferroni threshold
(α = 0.0167) is quoted wherever it changes the reading.

**Reported alongside, never as the conclusion:** Wilson intervals (marginal and unpaired) and
the discordant counts `b` and `c`. `b + c` is the number that separates "the memory did nothing"
from "the memory fixed 31 and broke 31" — on the superseded runs it was the latter, at exactly
b = c = 31, and accuracy alone could not have shown it.

**Power, stated in advance.** At n = 215 with independent rows the resolvable effect is roughly
**±6.3pp** per replicate. Pooling three replicates shrinks the run-noise component only. **This
design detects a large effect and cannot detect a small one**: any gap under ~3pp will be inside
the noise whatever its sign. Report a null as a power limit, never as equivalence.

**Decision rule.** Identical to the formula design: **graph beats text only if the pooled test
is significant *and* the mean gap across replicates exceeds the within-arm standard deviation.**
A gap smaller than the replicate spread is reported as noise regardless of p. Applied to the two
secondary comparisons as the same two-part standard.

## 6. Procedure, in order

Nothing in steps 3–5 may be revisited after step 6 begins.

1. ~~**Create `task_config_v2.json`**~~ — **done 2026-08-15**, see the amendment log.
2. ~~**Resolve the generator-model decision (§7)**~~ — **done 2026-08-15**: `gemma4:e2b` for all
   three roles, with the ceiling accepted and its reporting obligation pre-declared in §7.
3. **Run the probes** — `--memory text --retrieval flat` and `--memory graph --retrieval graph`
   on `probe_config.json`, at the provisional `--dedup_threshold 0.90` default. These serve
   three purposes at once: they exercise the wiring, they produce the recorded ADDs that step 4
   needs, and they give the train accuracy that §7's ceiling check reads.
4. **Choose `--dedup_threshold`** by replaying the probes' recorded ADDs offline (no LLM) with
   `calibrate_dedup_threshold.py`. **Do not reuse formula's 0.88 without re-measuring** — it was
   calibrated on financial-convention rules, and MBPP rules are style advice with a different
   similarity distribution. Two conditions, in order:
   1. absorption stays under **50%** in **both** memory arms — the guard against collapsing two
      genuinely different rules;
   2. of the thresholds satisfying (1), take the **highest** one that still prunes the
      benchmark's restatement family, and **read every merge it makes** (`--show-merges`) to
      confirm each is a restatement rather than two distinct rules.

   Condition (1) alone is not a selection rule and must not be used as one. It was written as if
   it were, and on this benchmark it selects nothing: absorption never approaches 50% anywhere
   in [0.80, 0.94], so taking "the largest" returns the top of the candidate range, where dedup
   prunes **zero** rules. A threshold that prunes nothing leaves the baseline as ACE minus one
   of its three key innovations — the precise defect `EXPERIMENT_FORMULA.md` §10 measures on the
   seed-0 runs, where an unpruned `text` arm was not the thing it was supposed to be. Condition
   (2) is what makes the rule bite in both directions, and inspecting the merges is what keeps
   it from becoming a free parameter.

   **This is calibration against the paper's specification, not tuning against the benchmark.**
   The decision is taken from playbook *contents* before any test number exists, it is applied
   identically in all three arms, and the evidence is recorded in the amendment log.
5. **Run `check_learning_signal.py` on the probes.** Checks [1]–[6] must pass before committing
   a full arm. Check [6] in particular: on formula it failed on the `text` arm at two of three
   replicates and that was only discovered afterwards. Check [3] is where §7's ceiling shows up.
   Then **pin every remaining flag** and record it here as an amendment.
6. **Run the nine arms.** Same commit for all nine if possible; if not, record the SHAs and
   verify with `git diff --stat <a> <b> -- ace/ eval/ llm.py utils.py playbook_utils.py` that
   **zero bytes of the run path differ**, which is what licenses `compare_arms.py --force`.
7. **Analyse**, in this order: `--selftest`, then `compare_arms.py --force`, then
   `aggregate_run_costs.py`, then `check_learning_signal.py` on all six memory runs.
8. **Write up**, quoting the pooled line as the headline and the per-replicate lines beneath it.

### The configuration to pin (all nine runs)

```
--config_path ./eval/mbpp_code_exec/data/task_config_v2.json
--num_ctx 32768 --seed {0,1,2} --think_generator off
--curator_frequency 1 --max_num_rounds 3 --num_epochs 1
--eval_steps 40            # 3 checkpoints over 120 train steps
--dedup --dedup_threshold 0.86   # from step 4, see the amendment log
--retrieval_budget 2500 --retrieval_seeds 5
--test_workers 1 --max_tokens 16384
```

The control arm carries `--dedup`, `--dedup_threshold` and `--eval_steps` **inert but present**,
or it differs from its siblings on fields `PINNED_FIELDS` audits. `--retrieval none` is forced
for the control (`ace/ace.py:176` raises otherwise) and the audit exempts it.

**Docker must be running** before any MBPP run: `DataProcessor.__init__` builds a `DockerSandbox`
and pulls `python:3.10-slim` before the first LLM call.

## 7. The one open decision: which generator

This is the pivotal parameter and it has no clean answer, so the rule is declared before the
probe rather than after.

| | `qwen3.5:2b` generator | `gemma4:e2b` all roles |
|---|---|---|
| no-memory test accuracy | **0.3346** | **0.8521** |
| headroom for memory to work in | ~67pp | **~15pp** |
| failures per 120 train steps to learn from | ~80 | **~18** |
| output-format defect | double-stringifies code (**now fixed** in `unwrap_quoted_source`) | 0 of 790 wrapped |
| cross-model distillation | yes — gemma reflector/curator | **no** |

Both figures are measured, on the same 257-row test set (`results/mbpp/arms_qwen_seed0/none`
and `results/mbpp/learning_signal_gemma/none`).

The tension is real: the single-model configuration is methodologically cleaner and is what the
formula runs use, but at 0.85 it leaves the reflector almost nothing to reflect on — and the ACE
loop learns from failures. A curator called on ~18 failures across 120 steps is being asked to
accumulate knowledge it never saw.

### DECIDED, 2026-08-15, before any run: `gemma4:e2b` for all three roles

Chosen for **cross-benchmark consistency**: identical models to the formula runs makes the
two-benchmark claim "same models, same harness, different task", and removes the cross-model
distillation deviation entirely. The decision was taken before any MBPP run under this design
existed, and it is recorded here rather than in the results write-up so it cannot be re-read as
a post-hoc justification.

**The cost of this choice is a ceiling, and it must be reported in the first paragraph of the
results, not in a criticism list.** The control scores 0.8521, so:

- The **maximum possible effect is 14.8pp**, against a detectable effect of ±6.3pp. The design
  can still see an effect ~2.3× smaller than the headroom, so it is not hopeless — but a null
  here carries a competing explanation ("no room") alongside the intended one ("nothing
  transfers"), and the write-up must present both rather than only the second.
- The reflector sees roughly **20 failures across 120 training steps**. The ACE loop learns from
  failures, so the learning signal is thin by construction.

**Pre-declared response if the probe (step 5) puts train accuracy above 0.85**, i.e. outside the
band `check_learning_signal.py` check [3] enforces: **do not silently proceed and do not retune
the band.** Record the number, and take one of two paths, choosing before seeing any test
accuracy — (a) run anyway and report the ceiling as the leading caveat, or (b) switch to the
`qwen3.5:2b` generator with `gemma4:e2b` reflector and curator, accepting the distillation
deviation, which is identical in all three arms and therefore cannot favour one. Whichever is
taken, amend this section with the reason.

For the record, the rejected alternative: `qwen3.5:2b` generator scores 0.3346 with ~67pp of
headroom and ~80 usable training failures, at the price of cross-model distillation and a
configuration that differs from the formula runs.

## 8. What would falsify the hypothesis, and what makes a run invalid

**The mechanism check, carried over from `EXPERIMENT_FORMULA.md` §7.** `structural_stats()`
records `walk_only_share` — the fraction of rendered rules the edge walk reached that relevance
alone would not have selected under the same budget. **`graph ≈ text` with `walk_only_share`
near zero is a broken run, not a negative result about structure**, because it means the arms
read the same rules. On formula it was 0.09–0.21 over the final-test generations. Report it per
replicate before reporting any accuracy comparison.

### The floor, fixed 2026-08-17 before any arm ran

"Near zero" needs a number here, because MBPP will run closer to it than formula did and the
threshold must not be chosen once the accuracies are visible.

**Measure it with `analyze_walk_share.py`, not from `structural_stats()`.** The counters at
`ace/memory/graph.py:155-157` are never reset, so the recorded `walk_only_share` is a
run-cumulative average over every retrieval since step 1, and `structural_stats()` is written at
eval checkpoints only (`ace/ace.py:987`, `:1288`) — so the final test, whose accuracy is the
reported result, contributes nothing to it. The script recomputes the share over the test
generations against the checkpoint the test actually ran on, reusing `GraphMemory`'s own
selection path. **The quantity this gate is defined on is the final-test pooled share.**

`walk_only_share` can only be non-zero while the budget **binds** — if the whole playbook fits
in `--retrieval_budget`, both arms render every live rule, `graph_order` and `flat_order` return
the same *set*, and the metric is 0 by construction rather than by failure. Projected from the
probes' growth (~40 tok/step `text`, ~33 `graph`, at `--dedup_threshold 0.86`), the 2,500-token
budget starts binding around **step 70 of 120**, and the final playbook is only ~1.6× the
budget. On formula it was 2.8×. **So this design has materially less retrieval pressure than
formula, and a lower reading here is expected, not anomalous.**

Pre-declared, on the **final-test pooled share** of each graph replicate:

**This gate is necessary, not sufficient, and must not be reported as validating the graph.**
Measured 2026-08-17 on the formula replicates: rewiring each graph at random, holding live nodes
and edge count fixed, scores **2.3–3.6× higher** than the real edges (0.4761/0.3333/0.5011 against
0.2066/0.0939/0.2050). The metric rewards pointing *away* from relevance, which arbitrary edges do
best, so it cannot tell a meaningful graph from a scrambled one. Clearing the floor establishes
only that the arms read different rules — never that the edges were the right ones.

| `walk_only_share` | reading |
|---|---|
| **≥ 0.05** | the arms read different rules, so the comparison is interpretable. Report the accuracy comparison normally, **without** claiming the metric validates the structure |
| **< 0.05** | report the run as **mechanism-inconclusive**, not as a negative result about structure. State the value, state that the budget did not bind hard enough to separate the arms' rule sets, and do not attribute any `text`/`graph` difference — in either direction — to relational retrieval |

**0.05 is calibrated against formula measured the same way**: on the final-test pooled share the
three formula replicates are **0.2066 / 0.0939 / 0.2050**, so the floor sits at roughly half the
weakest run that supported a published mechanism claim. It is deliberately low — the cost of
setting it too high is discarding a real mechanism, and the cost of setting it too low is a
caveat §2 already requires. Two calibration notes carried over from that measurement: a
substantial per-generation zero rate is **not** disqualifying (seed 1 got nothing from the walk
on 53 of 200 generations and still pooled to 0.0939), and pooled versus per-generation-mean makes
no practical difference (within 0.005 in every replicate), so either may be quoted provided the
choice is stated.

**A zero here does not predict that the two arms agree.** They diverge through three other
channels regardless: the graph arm's curator-declared `refines` withdraws rules the text arm
keeps (the text arm parses the field and drops it), its rendering carries a legend and
`related_to` / `refines … (withdrawn)` lines charged to the same budget, and any difference in
what the generator reads propagates into the next reflection, the next curator call and every
subsequent playbook. The probes demonstrate exactly this: `walk_only_share` was **0.000 across
all 95 retrieval calls** — the playbook never exceeded 1,441 tokens against the 2,500 budget —
and the arms still ended on different playbooks (28 live rules vs 20). What a zero costs is not
divergence but **attribution**: whatever difference appears is then eviction and annotation, and
those are the mechanisms this experiment explicitly says a flat list can reproduce (§3).

A run or a set is **invalid** and must not be reported as a result if any of these hold:

- `walk_only_share` is near zero in the graph arm (the treatment never fired);
- `check_learning_signal.py` fails on a memory arm and the failure is not disclosed;
- the fairness audit reports a difference other than `--memory`, `--retrieval`, `--seed` and
  `git_commit`-with-an-empty-code-diff;
- the arms selected checkpoints under different rules (different steps are fine — that is the
  pre-registered rule applied identically, and `compare_arms.py` prints the margin);
- the extraction-path audit shows a large between-arm spread. **This bit MBPP before**: the
  graph arm needed the regex fallback about twice as often as the others, an 8.9-point spread
  larger than any accuracy difference reported (`MBPP_RESULTS.md` §5.11). `compare_arms.py` now
  prints that table and warns above 5 points. A large spread does not invalidate an individual
  score — the fallbacks are guarded and run only after a strict parse fails — but it is a
  scoring-path confound and must be reported beside the accuracies, not omitted.

## 9. Why the existing MBPP runs do not count

`results/mbpp/arms_qwen_seed0/` was run 2026-08-03/04 at commit `82b0b65`. It is honestly
written up in `MBPP_RESULTS.md` and it is **not publishable as a test of this hypothesis**. The
reasons are structural, not presentational:

1. **The graph arm had no functional use of its graph.** `--retrieval` did not exist until
   2026-08-06; `render_for_generator(query=...)` raised `NotImplementedError` in both arms and
   `related_to` edges were built and then read only for display text and `structural_stats()`.
   The treatment under test — *retrieval follows edges* — **was not running**. This alone is
   disqualifying: the arms differed only in eviction, which a flat list reproduces.
2. **The baseline was ACE minus one of its three components.** Grow-and-refine (`--dedup`) did
   not exist, so the `text` arm was append-only and degenerated exactly as expected — 14 of 24
   rules near-verbatim restatements of one another.
3. **One run per arm.** No noise floor. We now know the loop amplifies serving nondeterminism
   ~37× and a memory arm moves ~37 rows in 200 against itself. Every gap reported there is
   inside that.
4. **The training signal was not equalised.** The `qwen3.5:2b` bracket-wrapping defect scored
   ~26 valid programs as failures in the `text` arm and ~4 in the graph arm, and those failures
   were fed to the reflector and curator. `rescore_from_logs.py` repaired the measurement; it
   cannot repair the playbook. **Fixed for future runs** — `unwrap_quoted_source` now handles
   the bracketed form in the live path, so this does not recur.
5. **Loop settings departed from the paper undisclosed.** `curator_frequency=5` and
   `max_num_rounds=1`, against the paper's batch size 1 and five reflector rounds; restored to
   1 and 3 on 2026-08-06.
6. **Scoring-path confound between arms**, 8.9pp spread, unmeasured at the time (§8).
7. **Checkpoint selected on 43 validation rows**, with the two memory arms tied (§4).
8. **The extraction fix was found after seeing the results.** Guarded so it can only raise a
   zero and measurably altered 0 answers outside the affected arm — but post-hoc.

Items 1, 2 and 5 mean those runs do not implement the design being tested at all. Item 3 means
that even if they did, no gap could be distinguished from noise. **`MBPP_RESULTS.md` is retained
for its mechanism write-up — §2 (the scoring defect), §5 and §6 (why MBPP is the wrong
instrument for accumulation) — and not for its numbers.**

## 10. Deviations from the ACE authors' setup

Same list as `EXPERIMENT_FORMULA.md` §8, restated so this document stands alone:

- **`num_epochs=1` against the paper's 5.** Disclosed. Five multiplies every training-phase
  call by five, and the budget does not carry it. This is the deviation most likely to
  under-train both memory arms relative to the published method.
- **2B models throughout**, against the paper's 671B. Roughly half of all curator ADDs on
  formula arrived with a pasted rendered-bullet prefix; `sanitize_curator_operations` strips
  them, but small-model curator competence is the known weak point.
- **`contradicts` edges off** (§3), so the treatment is weaker than the full design.
- **The curator prompt carries the `refines` field in all three arms**, including the two that
  ignore it, so the arms differ in one place rather than two. The baseline is described as
  "ACE with an extended curator schema whose extra field it ignores".

## 11. Budget

Estimated from the formula runs, scaled by training steps (120 here against 250 there) at the
restored `curator_frequency=1` / `max_num_rounds=3`:

| | per run | × runs | total |
|---|---|---|---|
| `none` | ~0.3 h | 3 | ~1 h |
| `text` | ~3 h | 3 | ~9 h |
| `graph` | ~3.5 h | 3 | ~10 h |
| | | | **~20 LLM-hours, ~25M tokens** |

An estimate, not a measurement. Two known ways it goes wrong: serving throughput on this
endpoint has collapsed mid-run more than once (36.3× on one formula arm) with no error and full
recovery afterwards, so **wall-clock is not predictable and per-arm hours must not be quoted as
a cost**; and the graph arm's curator prompt grows unbudgeted with the memory, which on formula
reached 83% of the ~31,000-token practical ceiling by step 250. At 120 steps here that is not a
risk, but read the last curator call's `prompt_num_tokens` rather than assuming.

---

## Amendment log

Append here, dated, before the runs. Anything added after step 6 begins must say so explicitly.

- *(2026-08-15)* Document created. `report_pooled` gained the singleton-block fallback (§5) so
  the pre-registered pooled statistic is computable on a non-clustered task; formula output
  verified unchanged.
- *(2026-08-15, step 1 — done)* `task_config_v2.json` created by
  `eval/mbpp_code_exec/data/make_v2_split.py`: **120 train / 85 val / 215 test**, 42 rows moved
  test → val by a seeded shuffle (seed 0), emitted in original file order. `task_id` overlap
  between every pair of splits is 0. Train untouched. Deterministic — do not regenerate.
- *(2026-08-15, step 2 — done)* Generator: **`gemma4:e2b` for all three roles**, chosen for
  cross-benchmark consistency with the formula runs. Rationale and the ceiling it accepts are
  in §7, along with the pre-declared response if the probe puts train accuracy above 0.85.
- *(2026-08-17, steps 3 and 5 — done)* Probes run, `check_learning_signal.py` read. **Check [3]
  puts train accuracy at 0.667 (`text`) and 0.767 (`graph`), inside the 0.20–0.85 band**, so
  §7's ceiling branch does not trigger: `gemma4:e2b` stands for all three roles, and §7's
  standing obligation to lead the results with the control's 0.8521 ceiling is unchanged.
  Checks [1], [2], [4] and [6] pass in both arms. **Check [5] fails on the `graph` arm** at
  15.1% of generations not parsing as Python against a 15% limit, and is disclosed here rather
  than waived: it is a small-sample artifact of the probe, not a property of the model. The
  probe test split is 10 problems evaluated three times and each train step re-generates five
  times, so two chronic offenders (`Split`, `extract_values`) dominate; the same measurement on
  a real 257-row test set gives 1.7%. 26 of the 31 failures are one dropped `]`, and the
  truncation is in the model's own *valid* JSON envelope, not in `extract_answer` — verified by
  strict-parsing the raw responses. Both arms fail identically on the probe test split (4/30
  each). **One thing to re-measure at seed 0 and report either way:** across the earlier gemma
  runs the rate was 0/257 in the `none` arm against 6/343 in `text` (Fisher p≈0.04), which would
  be a memory-induced scoring-path effect independent of the research question. The obvious
  mechanism only half-holds — comprehension usage is 5.4% without a playbook and 8.7% with, but
  only two of the six failures involve one. `compare_arms.py` already prints the extraction-path
  spread §8 requires.
- *(2026-08-17, step 4 — done)* **`--dedup_threshold 0.86`**, chosen under §6 step 4 before any
  test number existed. `calibrate_dedup_threshold.py` was written for this and committed;
  replaying both probes' recorded curator operations gives:

  | thr | `text` ADDs / absorbed | `graph` ADDs / absorbed |
  |---|---|---|
  | off | 30 / 0 | 29 / 0 |
  | 0.80 – **0.86** | 32 / 10 (31.2%) | 29 / 3 (10.3%) |
  | 0.88 | 30 / 4 (13.3%) | 29 / 1 (3.4%) |
  | 0.90 – 0.94 | 30 / 2 (6.7%) | 29 / 1 (3.4%) |

  Condition (1) is satisfied everywhere — absorption never approaches 50% — so it does not
  select, which is why condition (2) was added. Condition (2) resolves to 0.86: **0.80–0.86 is a
  single plateau collapsing an identical merge set**, so 0.86 is its top and carries the most
  margin against over-merging at no cost in pruning. At 0.90 dedup absorbs **zero** rules from
  the final playbooks while 14 of 20 rules in one section restate each other (§2), i.e. the
  paper's grow-and-refine would have been inert for the whole run.

  Every merge 0.86 makes was read before choosing it, and **all ten are genuine restatements —
  no false positive**. The pairs run 0.8634–0.9501; representative: `err-00018` "When dealing
  with mutable objects passed into functions, always treat them as state that must be explicitly
  managed" absorbing `err-00022` "…always ensure state changes are explicit and localized"
  (0.8700), and `err-00004`/`err-00012`, which share the clause "Favor immutable data structures
  (like tuples or strings)" verbatim and differ only in the tail (0.8634). The highest pair left
  alone is `err-00028`/`err-00030` at 0.8485 — also arguably one rule, so the residual error is
  in the conservative direction. Reproduce with
  `uv run python calibrate_dedup_threshold.py results/mbpp/probes_v2/text results/mbpp/probes_v2/graph --show-merges 0.86`.

  **Known cost, accepted:** pruning more shrinks the playbook, which delays the point at which
  `--retrieval_budget` binds and therefore lowers `walk_only_share` — the §8 floor exists partly
  to catch this. Fidelity to §3.2 wins over retrieval sensitivity, because an unpruned `text`
  arm is not the ACE baseline and no comparison against it would be fair. It also happens to
  bring the two arms' binding points together (~70 and ~68 of 120) rather than apart.
- *(2026-08-17)* §8 gained a **numeric `walk_only_share` floor of 0.05**, with the ≥/< reading
  pre-declared, and §2 gained the measured account of what the curator accumulates on this task.
  Both written from probe artifacts, before any arm ran.
- *(2026-08-17, qualification — same day, still before any arm ran)* §8 now states that the floor
  is **necessary but not sufficient**. A random-rewiring control on the formula replicates, at
  fixed live-node set and fixed edge count, scores 2.3–3.6× *higher* than the real graph. The
  metric rewards pointing away from relevance and arbitrary edges do that best, so clearing the
  floor shows only that the arms read different rules. The ≥/< thresholds are unchanged; the
  claim they license is narrower. `FORMULA_RESULTS_PREREGISTERED.md` §7 and §10 carry the same
  qualification.
- *(2026-08-17, correction — same day, still before any arm ran)* **The floor is defined on the
  final-test pooled share and must be read from `analyze_walk_share.py`, not from
  `structural_stats()`.** As first written it was calibrated against formula's recorded 0.11–0.18,
  which is the wrong quantity twice over: the counters at `ace/memory/graph.py:155-157` are never
  reset, so the recorded figure is a run-cumulative average diluted by every early step where the
  playbook still fitted the budget and the walk *could not* add anything; and `structural_stats()`
  is written at eval checkpoints only, so the final test contributes nothing to it. Re-measured on
  the test generations, formula is **0.2066 / 0.0939 / 0.2050**, and 0.05 is now half its weakest
  replicate on a like-for-like measurement rather than a third of a diluted one. The correction is
  **not** one-directional — seeds 0 and 2 moved up, seed 1 moved down, because the cumulative
  figure also covers a different question set (train, not test), which is why this is measured
  rather than adjusted. `FORMULA_RESULTS_PREREGISTERED.md` §7 is corrected in place; no formula
  conclusion changes, since the condition is a floor and every replicate clears it either way.
