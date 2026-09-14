# Pre-registration: MBPP, linear vs relational memory

**Status: no arm of this experiment has run.** Written 2026-08-15, rewritten 2026-08-24 when the
treatment relation changed from `related_to` to `co_cited`. The rewrite cut background about the
superseded MBPP runs and about a graph architecture this experiment no longer uses; every dated
decision it recorded is preserved in the amendment log at the end.

The runs in `MBPP_RESULTS.md` are **not** this experiment and do not count toward it. Their graph
arm predates `--retrieval`, so the treatment never fired, and their baseline predates `--dedup`, so
it was ACE with one of its three innovations missing. They are background.

---

## 1. Question and expected direction

Does relational memory beat linear memory for agent execution, on a task whose problems share
**no transferable content**?

**Expected direction: null.** This is declared in advance and is the reason for running it. MBPP's
problems are independent, so nothing an agent learns on one transfers to the next, and the curator
can only accumulate general style advice. A null here beside the `formula` co-citation result is a
claim about *when* accumulated memory helps, not a failed replication.

The informative outcome is therefore a **positive** one. It would be the first in this repository,
and it would have to be reported against the fact that the same treatment moved nothing on a task
where content does transfer.

**This experiment stands alone and is analysed alone.** It shares a harness, arms, fairness rules
and a decision rule with `../EXPERIMENT_COCITE.md`, which makes the two methodologically comparable.
No conclusion here is conditioned on what that experiment found. Any cross-task reading belongs in
a synthesis written after both are complete.

## 2. Why this task

`formula` and MBPP differ in exactly one property that matters to the hypothesis:

| | `formula` | MBPP |
|---|---|---|
| problem structure | 50 formulas, each appearing in train and test | independent problems, no repeats |
| what a playbook can hold | reusable per-formula procedure | general style advice only |
| test rows | 200 rows in 50 correlated blocks of 4 | independent rows |

That single difference is the contrast. Everything else is held identical on purpose: same arms,
same models, same budgets, same analysis machinery.

## 3. Arms

| arm | flags | mode | what it isolates |
|---|---|---|---|
| control | `--memory none --retrieval none` | `eval_only` | the accumulated content, by having none |
| baseline | `--memory text --retrieval flat` | `offline` | the ACE baseline |
| treatment | `--memory graph --retrieval graph --cocite` | `offline` | whether a memory that can represent **pairs** beats one that can only represent **items** |

**The control is not a bare LLM.** It carries the identical generator prompt and fills the playbook
slot with `(empty)` (`ace/memory/none.py`). Removing the section would make it differ from the other
arms in prompt structure *and* in memory content, when only the second is the intended comparison.
So `none` to `text` isolates the accumulated content, holding the prompt fixed. The bare-LLM number,
for anyone who asks for one, comes from `run_raw_baseline.py`, and it is a reference point rather
than a fourth arm.

**The capability under test is co-citation.** An edge connects two rules the generator itself named
together in `bullet_ids`, gated on a rate rather than a count: the pair must have been co-rendered
at least `--cocite_min_opportunity` times and co-cited on at least `--cocite_min_rate` of those
occasions. Retrieval takes the same top-k seeds the baseline would take, then walks those edges
breadth-first, so it can surface a rule that relevance ranking alone would never reach. The relation
exists only in accumulated history, and no flat representation holds it.

**`related_to` is not used.** It is cosine between two rules' texts, while retrieval scores rules by
cosine between the question and each rule: the same encoder in the same space, so the walk lands
where relevance was already looking. That is a property of how the relation is built rather than of
the task, so it would fail here for the same reason it failed on `formula`. See the amendment log.

**`refines` is disabled in every arm here** (`--no_curator_refines`). It is unary, "this rule is
dead", and a flat list reproduces it with a `supersedes` field plus a render-time filter, so it is
not the capability under test. Leaving it active would make the arms differ in eviction as well as
in pairwise representation. The curator prompt is unchanged and still carries the field; all three
arms now parse it and drop it.

De-duplication is unaffected and stays symmetric in both memory arms: `TextPlaybookMemory.refine`
deletes the absorbed line, `GraphMemory.refine` withdraws the node while keeping its counters, and
both arms end up showing the same rules.

The two mechanisms are not equivalent. De-duplication
compares rule *text* by embedding similarity, which prunes redundancy but not staleness; the
curator's declaration was the only mechanism that could retire a rule superseded by something
textually dissimilar, and now neither arm has it. That is symmetric across arms, and it leaves both
memories carrying the same *set* of pruning mechanisms as the published method, which has
grow-and-refine and nothing else: `refines` is this fork's addition to the curator schema, not the
authors' (`EXPERIMENT_FORMULA.md` §8). The implementation of grow-and-refine still differs from
theirs and is disclosed separately: `ace/core/bulletpoint_analyzer.py` merges a similar group into
one bullet with an LLM call, while `--dedup` keeps the earliest rule and drops the rest using
embeddings alone, because a model's judgement inside the treatment is a call that could differ
between arms. The removed field was also the minority contributor: on the `formula` co-citation runs de-duplication
accounted for 77% to 82% of all supersessions.

**`contradicts` edges stay off.** The NLI cross-encoder scores prescriptive advice as contradictory
at saturation, so it does not work on this material. The treatment is therefore weaker than the full
design, and this is disclosed rather than fixed.

## 4. Data

| | |
|---|---|
| source | MBPP-sanitized, `eval/mbpp_code_exec/data/` |
| splits in use | **120 train / 85 val / 215 test** (`task_config_v2.json`) |
| probe | 30 / 10 / 10 (`probe_config.json`) |
| superseded | 120 / 43 / 257 (`task_config.json`), background runs only |

The validation split was enlarged from 43 rows to 85 before any run of this experiment, because at
43 rows a validation estimate carries roughly 15pp of noise and it is what selects the checkpoint
each arm carries into the test set. In the superseded runs the two memory arms tied on validation
and were sent to test with different playbooks on that basis, which is model selection driven by
noise. The cost is stated rather than hidden: test half-width goes from 5.8pp to 6.3pp.

Splits are fixed files read in file order with no shuffle, identical in all nine runs.

## 5. Configuration

All nine runs:

```
--config_path ./eval/mbpp_code_exec/data/task_config_v2.json
--generator_model gemma4:e2b --reflector_model gemma4:e2b --curator_model gemma4:e2b
--num_ctx 32768 --seed {0,1,2} --think_generator off
--curator_frequency 1 --max_num_rounds 3 --num_epochs 1
--eval_steps 40                     # 3 checkpoints over 120 train steps
--dedup --dedup_threshold 0.86      # re-derived for this benchmark, see the amendment log
--no_curator_refines                # see §3; inert in the text and none arms
--retrieval_budget 2500 --retrieval_seeds 5
--test_workers 1 --max_tokens 16384
```

The treatment arm additionally carries:

```
--cocite --cocite_min_opportunity <TBD> --cocite_min_rate <TBD> --cocite_half_life 0
```

**Both gate values are still to be derived** (step 4 below). Formula's `3` and `0.20` must not be
copied: they were pinned from formula's own held-out training lift, and the gates depend on citation
density and playbook size, both of which differ here. The runner's own default for
`--cocite_min_rate` is `0.30`, so whatever is chosen has to be passed explicitly.

The control carries `--dedup`, `--dedup_threshold` and `--eval_steps` inert but present, or it
differs from its siblings on fields `PINNED_FIELDS` audits. `--retrieval none` is forced for it
(`ace/ace.py:176` raises otherwise) and the audit exempts it. The cocite gates are audited on graph
arms only, so the `text` and `none` arms need not carry them.

**Docker must be running** before any MBPP run: `DataProcessor.__init__` builds a `DockerSandbox`
and pulls `python:3.10-slim` before the first LLM call.

**The three runs per arm are replicates, not seed variations.** `--seed` reaches only Ollama's
sampler, and at temperature 0 the sampler is greedy and never reads it. Divergence tracks the state
the server was in when it loaded the model. `md5sum` the prediction files before treating two runs
as two replicates; on `formula`, two runs at different seeds came back byte-identical.

## 6. Analysis

**Primary metric:** accuracy on the test split at the best-validation checkpoint, scored by
`DataProcessor.answer_is_correct`, which executes the extracted program against the problem's
assertions in the sandbox.

**No LLM grades any output, in any arm.** Scoring is sandbox execution against fixed assertions, so
it is deterministic and cannot drift between arms or between replicates. This is a fairness
property rather than an implementation detail: an LLM judge introduces grader variance that can
correlate with output style, and memory changes output style, so a judge could register a
difference that is not a difference in correctness. The same rule governs the write path, where
`--dedup` compares rules by embedding rather than asking a model which to keep
(`ace/memory/refine.py`).

**Primary comparison:** `cocite` against `text`, paired by problem index, pooled across the three
replicates, alpha 0.05.

**The unit of resampling is the row, not a block.** This is the one substantive departure from the
formula analysis and it goes in MBPP's favour: MBPP's problems really are independent, which is the
same fact that makes nothing transfer. MBPP is deliberately absent from
`compare_arms.py::CLUSTER_SOURCES`, so `report_pooled` falls back to one block per row, and
McNemar's independence assumption holds here rather than being anti-conservative.

| statistic | role |
|---|---|
| sign-flip permutation over singleton blocks | **primary**, and the permutation form of McNemar on averaged scores |
| bootstrap 95% CI over rows | **primary** |
| McNemar exact | valid here, reported. Its `b` and `c` counts separate "the memory did nothing" from "the memory fixed 31 and broke 31", which on the superseded runs was exactly the case, at b = c = 31 |
| Wilson interval | orientation only, marginal and unpaired |

**Pooling across replicates.** Each row is scored as its mean correctness across an arm's three
runs, a value in {0, 1/3, 2/3, 1}, and those real-valued rows go through the same test with every
row its own block. Pooling removes run noise and does nothing to problem-sampling variance. **The
pooled analysis is 215 rows, not 645**, and `sd/sqrt(3)` must never be used as a standard error.

**Family and correction.** Three comparisons, declared here before any run: the primary, plus
`text` against `none` and `cocite` against `none`. Reported with **Holm** adjusted p-values, with
Westfall-Young alongside as the sensitivity check. Both are printed by `compare_arms.py`. This
matches `../EXPERIMENT_COCITE.md` §3.1.

**Power, stated in advance.** At n = 215 with independent rows the resolvable effect is roughly
**6.3pp** per replicate. Pooling shrinks the run-noise component only. This design detects a large
effect and cannot detect a small one: any gap under about 3pp is inside the noise whatever its sign.
A null is reported as a power limit, never as equivalence.

**Decision rule.** The treatment beats the baseline only if the pooled test is significant **and**
the mean gap across replicates exceeds the within-arm standard deviation. A gap smaller than the
replicate spread is reported as noise regardless of p. The same two-part standard applies to the
secondaries.

## 7. What makes a run invalid

**The treatment must be shown to have fired before any accuracy is compared.** Report these per
replicate, ahead of the accuracy table:

| gate | requirement |
|---|---|
| live co-citation edges | not near zero |
| `cocite_max_degree` | must not approach the live node count, which would be a hub rather than a graph |
| **`walk_only_share`** | **at or above 0.05**, on the final-test pooled share |
| held-out predictive lift of the shipped gate | above 1.0x base rate, measured on generations the gate was not built from |

`walk_only_share` is the fraction of rendered rules the walk reached that relevance alone would not
have selected under the same budget. **`cocite` accuracy equal to `text` with that share near zero
is a broken run, not a negative result about structure**, because it means the arms read the same
rules.

**Read it with `analyze_walk_share.py`, never from `structural_stats()`.** The counters at
`ace/memory/graph.py:155-157` are never reset, so the recorded value is a run-cumulative average
diluted by every early step where the playbook still fitted inside the budget, and
`structural_stats()` is written at eval checkpoints only, so the final test contributes nothing to
it. The script recomputes the share over the test generations against the checkpoint the test
actually ran on.

**The floor is necessary, not sufficient.** A random rewiring at matched live-node count and matched
edge count scores 2.3x to 3.6x *higher* than a real similarity graph, because the metric rewards
pointing away from relevance and arbitrary edges do that best. Clearing 0.05 shows the arms read
different rules and nothing more. Held-out citation lift is what validates the edges.

**A note on the floor's calibration, so it is not over-read.** 0.05 was fixed on 2026-08-17 against
formula's `related_to` arms, which ran 0.2066 / 0.0939 / 0.2050. The formula co-citation arms ran
0.0525 / 0.2094 / 0.4352, so a legitimate co-citation arm can land close to the floor. The threshold
is unchanged; the claim it licenses is narrow.

**Learning-signal gates.** `check_learning_signal.py` checks [1] to [6] must pass on the probes
before any full arm is committed. Check [6] in particular, which fails a run where more than 50% of
ADDs add nothing new: on formula it failed on the `text` arm in two of three replicates, and that
was only discovered afterwards. Check [3] is where the generator ceiling shows up.

**The generator ceiling, and the response fixed in advance.** If a probe puts train accuracy above
0.85, do not silently proceed and do not retune the band. Record the number and take one of two
paths, choosing before seeing any test accuracy: (a) run anyway and lead the results with the
ceiling as the first caveat, or (b) switch to a `qwen3.5:2b` generator with `gemma4:e2b` reflector
and curator, accepting the cross-model distillation deviation, which is identical in all three arms
and therefore cannot favour one. Amend §9 with the reason either way.

## 8. Procedure, in order

Nothing in steps 1 to 5 may be revisited once step 6 begins.

1. ~~Create `task_config_v2.json`~~ **done 2026-08-15.**
2. ~~Resolve the generator model~~ **done 2026-08-15:** `gemma4:e2b` in all three roles.
3. **Re-run the probe pair under the current treatment.** `--memory text --retrieval flat` and
   `--memory graph --retrieval graph --cocite`, on `probe_config.json`, at the provisional gates.
   This exercises the wiring, produces the recorded ADDs step 5 needs, produces the
   `bullet_usage_log.jsonl` step 4 needs, and gives the train accuracy §7 reads.
4. **Derive `--cocite_min_opportunity` and `--cocite_min_rate`** with `calibrate_cocite_gate.py`
   (retired; recover it with `git show b02e02d:calibrate_cocite_gate.py`),
   built on the first half of the probes' training generations and scored on the unseen second half,
   comparing candidates **at matched edge count** rather than at matched threshold. Never against a
   test row. Record the sweep in the amendment log before step 6.
5. **Re-confirm `--dedup_threshold 0.86`** on the new recorded ADDs with
   `calibrate_dedup_threshold.py`, and run `check_learning_signal.py` on both probes. The dedup
   decision is on rule text and independent of edge type, so 0.86 is expected to stand; confirming
   it is free while probing and must not be silently assumed.
6. **Run the nine arms.** One commit for all nine if possible. If not, record the SHAs and verify
   with `git diff --stat <a> <b> -- ace/ eval/ llm.py utils.py playbook_utils.py logger.py` that
   **zero bytes of the run path differ**, which is what licenses `compare_arms.py --force`.
7. **Analyse**, in this order: `compare_arms.py --selftest`, then `compare_arms.py`, then
   `analyze_walk_share.py` and `calibrate_cocite_gate.py` on the three treatment arms, then
   `aggregate_run_costs.py`, then `check_learning_signal.py` on all six memory runs.
8. **Write up**, leading with the §7 gates, then the pooled comparison, then the per-replicate lines
   beneath it.

## 9. Deviations from the ACE authors' setup

All identical across arms, so none can favour one:

- **`num_epochs 1`** against the paper's 5. The budget does not carry five. This is the deviation
  most likely to under-train both memory arms relative to the published method.
- **2B models throughout**, against the paper's 671B. Small-model curator competence is the known
  weak point: roughly half of all curator ADDs on formula arrived with a pasted rendered-bullet
  prefix, which `sanitize_curator_operations` strips.
- **`contradicts` edges off** (§3), so the treatment is weaker than the full design.
- **The curator prompt carries `refines` in all three arms**, including the two that ignore it.

## 10. Budget

Estimated from the formula runs, scaled by training steps (120 here against 250 there):

| | per run | runs | total |
|---|---|---|---|
| `none` | ~0.3 h | 3 | ~1 h |
| `text` | ~3 h | 3 | ~9 h |
| `cocite` | ~3.5 h | 3 | ~10 h |
| | | | **~20 LLM-hours, ~25M tokens** |

Plus the two probe runs before step 6.

## 11. Amendment log

Append here, dated. Anything added after step 6 begins must say so explicitly.

- *(2026-08-15)* Document created. `report_pooled` gained the singleton-block fallback so the pooled
  statistic named in §6 is computable on a non-clustered task; formula output verified unchanged.
  Timing is the point: choosing a pooled statistic after seeing which answer it gives is the move
  this document exists to prevent.
- *(2026-08-15, step 1 done)* `task_config_v2.json` created by `make_v2_split.py`: 120 / 85 / 215,
  with 42 rows moved from test to val by a seeded shuffle and emitted in original file order.
  `task_id` overlap between every pair of splits is 0. Train untouched. Deterministic, do not
  regenerate.
- *(2026-08-15, step 2 done)* Generator: `gemma4:e2b` in all three roles, for cross-benchmark
  consistency with the formula runs. The rejected alternative was a `qwen3.5:2b` generator at 0.3346
  with roughly 67pp of headroom, at the price of cross-model distillation.
- *(2026-08-17)* First probe pair run, under `related_to`. Train accuracy 0.667 (`text`) and 0.767
  (`graph`), inside the 0.20 to 0.85 band, so §7's ceiling branch did not trigger. Checks [1], [2],
  [4] and [6] passed in both arms. Check [5] failed on the graph arm at 15.1% of generations not
  parsing as Python against a 15% limit, diagnosed as a small-sample artifact of a 10-problem probe
  test split: the same measurement on a real 257-row test set gives 1.7%, and 26 of the 31 failures
  were one dropped bracket inside the model's own valid JSON envelope. Superseded by step 3, which
  re-probes under the current treatment.
- *(2026-08-17)* **`--dedup_threshold 0.86`**, chosen before any test number existed by replaying
  the probes' recorded ADDs offline at each candidate. The 50% absorption cap is a guard, not a
  selection rule: nothing in [0.80, 0.94] approaches it on this benchmark, so "take the largest
  under 50%" returns a threshold that prunes nothing and leaves the baseline as ACE minus
  grow-and-refine. 0.80 to 0.86 is a single plateau collapsing an identical merge set, so 0.86 is
  its top and carries the most margin against over-merging at no cost in pruning. At 0.90 dedup
  absorbs **zero** rules while 14 of 20 rules in one section restate each other. All ten merges 0.86
  makes were read individually and all ten are genuine restatements, with no false positive; the
  highest pair left alone sits at 0.8485 and is arguably one rule, so the residual error is in the
  conservative direction.
- *(2026-08-17)* §7 gained the numeric `walk_only_share` floor of 0.05, then the same-day
  qualification that the floor is necessary but not sufficient, then the correction that it is
  defined on the final-test pooled share read from `analyze_walk_share.py` rather than on the
  run-cumulative counter. All three before any arm ran.
- *(2026-08-24, before any arm ran)* **The treatment relation changes from `related_to` to
  `co_cited`.** The co-citation experiment on `formula` completed and established that `related_to`
  is a function of its own nodes: cosine between two rules' texts, against a retrieval score that is
  cosine between the question and each rule, the same encoder in the same space. A random rewiring
  at matched edge count surfaced 2.3x to 3.6x more novel rules than the real graph. That redundancy
  is a property of the construction rather than of the task, so it would reproduce here, and nine
  arms would be spent confirming something derivable rather than testing anything.

  Three things make this an instrument fix rather than the design chasing a wanted result.
  **Direction:** `co_cited` is the stronger treatment, verified predictive on held-out data and
  verified to change what the generator reads, so swapping it in makes a null *harder* to obtain,
  and §1's declared expected direction stays null either way. **Timing:** step 6 has not begun and
  no arm of this experiment exists. **Symmetry:** the change alters only what the treatment arm can
  store, touching no prompt, no split, no budget, and nothing in the baseline or the control.

  The cost is that §1's cross-task claim now runs to `../EXPERIMENT_COCITE.md` rather than
  `EXPERIMENT_FORMULA.md`, which is a gain: the matched pair becomes one treatment on two tasks that
  differ in whether content transfers between problems.
- *(2026-08-24, before any arm ran)* **The multiplicity treatment changes from "no correction" to Holm, with Westfall-Young alongside**, matching
  `../EXPERIMENT_COCITE.md` §3.1. `compare_arms.py` gained `holm()`, `westfall_young()` and
  `report_multiplicity()` for that experiment and prints the block for any family it is given, so
  this adopts an existing tested procedure rather than inventing one. Both are more conservative
  than no correction, so the change cannot manufacture a positive result.
- *(2026-08-24, before any arm ran)* **Curator-declared `refines` is disabled in all arms**
  (`--no_curator_refines`). It is unary and therefore not the capability under test, so leaving it
  active would make the arms differ in eviction as well as in pairwise representation.
  De-duplication is unaffected and remains symmetric. It was active in the graph arm on `formula`,
  where the asymmetry is disclosed and could only favour the treatment. Covered by
  `check_finance_wiring.py` [1h]; the flag defaults to `refines` active, so no earlier run changes.
- *(2026-08-24, before any arm ran)* **Step 3 is a re-probe, and it is blocking.** The 2026-08-17
  probe artifacts cannot calibrate a co-citation gate: they log `bullet_ids_used` but not
  `bullet_ids_rendered`, which was added with the co-citation work afterwards.
  `calibrate_cocite_gate.py:45` reads `bullet_ids_rendered` to build the co-rendered denominator, so
  on those artifacts the rendered set is empty, the cited set is empty, and the calibration returns
  nothing **without failing**. A mis-set gate does not error either: on formula,
  `--cocite_min_rate 0.30` produced 14 live edges and a final-test `walk_only_share` of 0.0689,
  below every arm it was meant to beat. §7's floor would only catch that after nine arms were spent.
