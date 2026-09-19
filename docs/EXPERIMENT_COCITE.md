# Experiment pre-registration: co-citation edges on `formula`

**Written 2026-08-18, before any run of this set.** Nothing below was chosen after seeing a test
number from it.

> **All nine runs are complete (2026-08-23). Results in `COCITE_RESULTS.md`.**
> The primary `text → cocite` is **null** (+2.17pp, CI [−0.020, +0.060], p=0.3344), so the §4a
> conditional **did not fire** and the three `graph` arms are not run. Both secondaries are
> significant. The treatment was verified delivered in all three replicates, and the null holds
> across an eightfold spread in `walk_only_share`. This document is kept as written; only §9
> step 6 and §10 record what happened.

---

## 0. Status: this experiment is post-hoc, and says so first

The three-arm formula experiment (`background/EXPERIMENT_FORMULA.md`, results in
`background/FORMULA_RESULTS_PREREGISTERED.md`) is complete. It found that **memory helps** and that
**structure was not shown to**:

| question | answer | figures |
|---|---|---|
| Does memory help at all? | **Yes for `graph`, marginal for `text`** | `none → graph` +8.00pp, p=0.0146; `none → text` +5.17pp, p=0.0922 |
| Is relational memory better than linear? | **Not shown** | `text → graph` +2.83pp, CI [−0.017, +0.078], p=0.2905 |

This experiment was designed on 2026-08-17, **after** that null was analysed and diagnosed.
Designing an arm after seeing a null is a researcher degree of freedom, and no justification
makes it not one. What follows is pre-registered before its own runs, which is the most that can
be claimed for it, and it is reported as a follow-up rather than as part of the pre-registered
result. 

## 1. Why co-citation, specifically

The `text → graph` null has a mechanical diagnosis, not a statistical one. `related_to` is cosine
similarity between two rules' **texts**, while retrieval scores rules by cosine between the
**question** and each rule. Those are nearly the same signal, so the walk landed on rules that
relevance ranking would have selected anyway. Measured: a **random rewiring** at matched live-node
and edge count surfaced **2.3–3.6× more** novel rules than the real graph
(0.4761/0.3333/0.5011 against 0.2066/0.0939/0.2050). The treatment fired; its edges were largely
redundant with the scoring that chose the seeds.

The test for whether a relation earns a graph is: **is the relation a function of the items, or an irreducible fact about the
pair?**

| relation | pairwise? | derivable from item content? | needs a graph? |
|---|---|---|---|
| `refines` | no, unary ("this rule is dead") | — | no: a flat list with a `supersedes` field |
| `related_to` | yes | **yes**, cosine of two texts | no: a flat list plus an embedding index |
| `co_cited` | yes | **no**, exists only in accumulated history | **yes** |

Both arms already receive the same citation stream; `helpful`/`harmful` are derived from it today.
A linear memory simply cannot hold the **pairwise** part. The experiment becomes: *same data, same
loop, same budget; one arm can represent per-item facts, the other can represent facts about
pairs.*

**Validated offline before implementation, on the three completed graph runs.** Edges built from
the first half of each run's training generations predict co-citation on the unseen second half at
**2.75× / 2.13× / 2.79×** over base rate. Random edges are 1.00× by construction.

## 2. The mechanism

Per unordered pair, three numbers: `w_cocited`, `w_corendered`, `last_step`.

**An edge exists iff `w_corendered ≥ --cocite_min_opportunity` and
`w_cocited / w_corendered ≥ --cocite_min_rate`.**

The stored quantity is a **rate, not a count** — `P(both cited | both rendered)`. A raw count
would rank whatever is rendered most often instead of what is chosen together, which is why the
denominator exists, and why `logger.py` now records `bullet_ids_rendered` beside
`bullet_ids_used`.

Pinned **from held-out training lift, before any test row**:
`--cocite_min_opportunity 3 --cocite_min_rate 0.20 --cocite_half_life 0`.

**The rate gate was 0.30 for the first seed-0 run and 0.30 does not deliver the treatment.**
That run built 14 live edges over 83 rules at the checkpoint its test used, and a final-test
`walk_only_share` of 0.0689, below all three pre-registered graph arms. The gate was re-pinned to
0.20, which §2 already named as the fallback, on the §7 mechanism diagnostic and before the run's
accuracy was read. Measured on that run's own pair store, same 200 test questions, same
checkpoint, one `GraphMemory` with the gate swept in memory:

| gate | live edges | rendered/call | `walk_only_share` (final test) | calls the walk added nothing to |
|---|---|---|---|---|
| 0.30 | 14 | 27.7 | 0.0689 | 93 / 200 |
| 0.25 | 26 | 26.5 | 0.1720 | 84 / 200 |
| **0.20** | **34** | **24.7** | **0.2695** | **58 / 200** |
| 0.15 | 45 | 22.6 | 0.4298 | 19 / 200 |
| 0.10 | 69 | 20.7 | 0.4301 | 3 / 200 |
| 0.05 | 130 | 20.5 | 0.3224 | 0 / 200 |

0.20 is the loosest gate that stays above the three graph arms (0.2066 / 0.0939 / 0.2050) and
below the random-rewiring band (0.4761 / 0.3333 / 0.5011). 0.15 and 0.10 land inside that band,
and walk share rewards pointing away from relevance, so a higher share is not a better graph.
`cocite_max_degree` never approaches the live node count at any gate in the sweep, so loosening
does not trade the §7 empty-graph condition for the §7 hub condition.

**Why 0.30 was too tight is diagnosable rather than bad luck.** The gates were validated on
held-out **lift** while the edge rule is **confidence**, `P(both cited | both rendered)`.
Confidence favours pairs whose members are each cited often, and frequent citation tracks high
relevance, so the surviving edges point into the set relevance already retrieves. Over the live
pairs at step 250 the rate distribution is median 0.000, p90 0.026, p99 0.297: 0.30 keeps only
the extreme tail, and 45 of the 58 pairs clearing it store-wide involve a rule later withdrawn by
supersession. A lift gate is the principled repair and is **not** adopted here: it is a change to
the treatment after seeing a run, so it needs its own held-out evidence first.

Design decisions, each measured rather than assumed:

- **All citations, not correct-only.** Filtering to correct generations halves the edge count for
  no lift gain and loses in two of three seeds. Co-citation captures "these rules address the same
  kind of problem", which is orthogonal to whether the model got it right.
- **Decay implemented, defaulted off.** Half-life 100 measured +0.05 lift for ~10% fewer edges,
  not enough to justify a hyperparameter. The machinery is lazy (applied on access, no global
  sweep), so shipping it disabled costs nothing.
- **Replaces `related_to` rather than joining it.** Similarity edges are not written under
  `--cocite`, so a `graph+cocite` arm differs from a `graph` arm in exactly one thing.
- **Co-citation relations are rendered and charged to `--retrieval_budget`**, like `related_to`.
  The treatment gets no free structure.
- **Edge inheritance on `refines`**, both origins, merged at full weight. The pairwise counterpart
  of something the flat arm already does for unary state: `refine()` adds an absorbed rule's
  counters to its survivor.
- **No exploration in v1.** 41–47% of the pair space already gets a chance with no exploration
  mechanism, because question variety does the work and the top-`--retrieval_seeds` are always
  relevance-chosen, so edges cannot suppress that floor. Circularity is real but bounded, and
  disclosed.

## 3. Why `text` must be re-run rather than reused

Between the pre-registered graph arm's commit `1acfe69` and this experiment's commit, the run-path
diff is ~800 insertions. **Exactly one change is behavioural for a non-cocite run**, established
by reading the diff:

- `ace/ace.py`, a hoisted `_render(question)` producing the identical string, plus
  `record_generation`, a no-op on the `Memory` ABC that early-returns in `GraphMemory` when
  `cocite=False`.
- `logger.py`, a write-only `bullet_ids_rendered` field, plus removal of dead key-mixer plumbing.
- `llm.py`, `utils.py`, dead-code removal only.
- `ace/memory/graph.py`, gated behind `--cocite`, plus a reporting-only stats fix.
- **`playbook_utils.py::_RENDERED_PREFIX`** — the 2026-08-16 two-tier fix, in
  `sanitize_curator_operations`, **a path every arm shares.**

Replaying the nine pre-registered runs' recorded curator operations through both regexes:

| run | curator ops | old strips | new strips | delta |
|---|---|---|---|---|
| seed0/graph | 452 | 213 | 258 | +45 |
| seed0/text | 404 | 215 | 233 | +18 |
| seed1/graph | 380 | 134 | 147 | +13 |
| seed1/text | 695 | 412 | 451 | +39 |
| seed2/graph | 448 | 290 | 331 | +41 |
| seed2/text | 623 | 396 | 469 | +73 |

3–12% of all curator operations. Comparing a `cocite` run on the new commit against the
pre-registered `text` runs would differ in **two** places, the memory, and how much
pasted-prefix junk survives into rule text. That is the confound the fairness rules exist to
prevent, so the baseline is re-run. The same reasoning applies to the `graph` arm if the §4a
conditional fires: it is run on this commit, never taken from the pre-registered set.

**`none` is exempt and is reused unchanged.** Verified: all three control runs make **generator
calls only** and log **zero** curator operations, so they never reach the sanitizer and are
invariant to this change.

**Reverting the regex was considered and rejected.** It would restore comparability for ~42h of
saved compute, but it selects code to fit a desired comparison, runs the experiment on knowingly
defective code, and biases toward the very null this experiment exists to move away from, since
the defect injects noise and noise shrinks differences.

## 4. Design

**This is the three-arm design, control, baseline, treatment**. `related_to` was the treatment in the pre-registered experiment and already has its
result there; co-citation replaces it. No arm is being dropped, the treatment is being swapped.

Three arms, seeds 0/1/2, `task_config_v2.json` (250 train / 100 val stratified / 200 test).

| arm | role | `--memory` | `--retrieval` | `--cocite` | source |
|---|---|---|---|---|---|
| none | control | `none` | `none` | — | **reused** from `results/formula/preregistered/seed*/none` |
| text | baseline | `text` | `flat` | off | re-run (§3) |
| cocite | **treatment** | `graph` | `graph` | **on** | new |

**Six new runs.** Everything else pinned exactly as the pre-registered set: `gemma4:e2b` in all
three roles, `--think_generator off`, `--num_ctx 32768`, `--dedup --dedup_threshold 0.88`,
`--eval_steps 125`, `--retrieval_budget 2500`, `--retrieval_seeds 5`, `--test_workers 1`,
`curator_frequency 1`, `max_num_rounds 3`, `num_epochs 1`. All six share one commit.

### 4a. The `graph` arm is a declared conditional, not a dropped arm

**If and only if the primary outcome (§5) clears the detection floor of §6, a `graph` arm
(`--memory graph --retrieval graph`, no `--cocite`) will be run on the same commit, three seeds,
and `graph → cocite` reported as the mechanism check.**

The rule and its direction are fixed here, before any run:

- **Primary significant → run `graph`.** A win raises the question *why*, and `graph → cocite` is
  the only pair differing in exactly one thing, which relation the graph holds. Attribution is
  worth buying once there is a gap to attribute.
- **Primary null → do not run `graph`.** There is no gap to attribute, and `graph → cocite` faces
  the same 50 clusters and the same ~4pp floor, so it would be null on any plausible effect size.
  Two nulls instead of one, for ~21h.

Declaring the direction now is the point. Deciding to add an arm *after* seeing a positive primary
would be a reaction to the number; a conditional fixed in advance is part of the design. **The
commitment binds in both directions:** if the primary is significant, the `graph` arm is run and
reported even if its result is unwelcome.

Any later variant of the co-citation gate queues **behind** this conditional, not ahead of it: a
significant primary buys the `graph` arm first, because that is the commitment made here.

**Until it is run, `text → cocite` is not decomposed**, and the write-up must say so: the gap is
attributable to the treatment as a whole: graph storage, the walk, `refines` eviction, annotation
*and* co-citation, not to the pairwise relation alone. The pre-registered `text → graph` +2.83pp
is context for that bundle, **not** a control for it: those runs predate the sanitiser fix (§3), so
subtracting one from the other compares an arm that has the fix against one that does not.

## 5. Outcomes, declared in advance

**Primary: `text → cocite`.** Clustered sign-flip permutation test on replicate-pooled scores
(`compare_arms.py::pooled_arm_scores` → `cluster_paired_stats`), blocking on formula, with the
cluster bootstrap CI. This is the headline claim, *relational memory beats linear memory* and
it is the **only** comparison declared primary.

**Secondary: `none → cocite`** and **`none → text`**, against the reused controls. These re-ask
the question the pre-registered experiment already answered positively (*does memory help at
all?*), so they are confirmatory rather than novel.

**Conditional: `graph → cocite`.** The mechanism check, run only if the primary clears the floor,
under the rule fixed in §4a. `text → cocite` licenses "this relational configuration beats linear
memory"; **only `graph → cocite` licenses "the pairwise relation is what did it."** Until it
exists, the second claim is not made.

**Declared a replication check, not a result, the new `none → text`.** Re-running the baseline
produces a second estimate of a question already answered. The pre-registered figures **remain the
answer**; new ones are reported as a replication check. Expected: a small level shift from the sanitiser fix, no change in the
gaps.

Three arms give three pairwise comparisons, rising to six if the conditional fires. Declaring the
primary in advance is what keeps the rest from becoming extra chances at significance.

## 6. Power, stated before the runs

The test split is 50 formulas × 4 instances, and instances of one formula fail together. Power
scales with **clusters, not rows**, and the measured ceiling is a **~3.5pp CI half-width even at
infinite replicates**, which is why more replicates were ruled out for the original experiment.

**`text → cocite` must reach roughly +4pp to exclude zero.** A gap of +3pp will be null.

**Declared now: a null below ~4pp is uninformative about the treatment, not evidence against it**,
and will be reported as an underpowered non-result rather than a negative one. 

**This floor is also the §4a trigger.** "Clears the detection floor" means the primary's cluster
bootstrap CI excludes zero, the same criterion used for the pre-registered result.

## 7. What would make a run mechanism-inconclusive

Read from `structural_stats()` **before quoting any accuracy**. Any of these means the treatment
was not delivered, and the run is inconclusive rather than negative:

- **`cocite_edges` near zero**: the gates never fired.
- **`cocite_max_degree` approaching the live node count**: inheritance concentrated mass into a
  hub, the walk reaches everything in one hop, and `related_to`'s failure is recreated.
- **`walk_only_share` near zero** (via `analyze_walk_share.py`, final-test pooled, **not** the
  run-cumulative counter in `structural_stats()`), the arms read the same rules.
  **Any reading of this taken before 2026-08-19 on a cocite run is void.** `analyze_walk_share.py`
  hardcoded the walk to `related_to`, which `--cocite` never writes, so it returned 0.0000 for
  every cocite run by construction while the run's own cumulative counter was right. Fixed by
  mirroring `graph.py:228`; verified inert on the pre-registered graph arms, which still read
  0.2066.
  **Necessary, never sufficient:** random rewiring scores 2.3–3.6× higher than a real graph, so
  clearing the floor shows the arms differ, it does not validate the edges. Held-out citation lift
  is what validates the edges, and it was measured before implementation (§1).

All `cocite_*` figures count live nodes only. Before that fix they included pairs whose nodes had
been withdrawn, which inflated `cocite_max_degree` and let `cocite_pair_coverage` exceed 1.0.

## 8. Deviations and disclosures carried over

- **The curator prompt is identical in all arms**, including the `refines` field only the graph
  arms act on. The text arm parses it and drops it. The baseline is honestly described as "ACE
  with an extended curator schema whose extra field it ignores".
- **The `refines` asymmetry is a second difference** between `text` and either graph arm, on top
  of retrieval. It is the disclosed cost of holding the curator prompt identical, and it means a
  `text → cocite` gap is not attributable to the pairwise relation alone, which is exactly why
  `graph → cocite` is reported beside it.
- **`num_epochs 1`** against the paper's 5. Disclosed deviation, identical in all arms.
- **`--seed` is inert at temperature 0**; replicates differ through GPU floating-point reduction
  order, which the ACE loop amplifies ~37×. No run here is described as reproducible.
- **Circularity in co-citation**: a pair can only be co-cited if co-rendered, and co-rendering is
  driven by the edges. Normalising by opportunity fixes numerator inflation but not the sampling
  bias. Measured coverage 41–47% of the pair space with no exploration mechanism.

## 9. Procedure, in order

1. Confirm `git status` clean and all six runs launch from one commit.
2. `check_memory_equivalence.py`, `check_prompts.py`, `check_finance_wiring.py` all pass.
3. Six runs into `results/formula/cocite_experiment/seed{0,1,2}/{text,cocite}`.
4. `check_learning_signal.py` on one run per arm, a dead signal invalidates that arm.
5. Analysis exactly as §5 via `compare_arms.py`; cost via `aggregate_run_costs.py`;
   `analyze_walk_share.py` on the cocite arm.
6. **Evaluate the §4a conditional and record the decision before acting on it.** If the primary
   cleared the floor, run the three `graph` arms into
   `results/formula/cocite_experiment/seed{0,1,2}/graph` on the same commit and report
   `graph → cocite`. If it did not, record that the conditional did not fire and stop.

   **Recorded 2026-08-23: the conditional did not fire.** The primary's cluster bootstrap CI is
   [−0.0200, +0.0600] and contains zero, so the three `graph` arms are not run and
   `text → cocite` stays undecomposed, exactly as §4a requires. `COCITE_RESULTS.md` §7 states
   that limitation where the result is claimed.

`compare_arms.py` will flag `git_commit` between the reused `none` runs and the new arms.
**The pre-registered set's licence: "the run-path diff is empty" does not apply here.** The
licence is narrower and must be stated when `--force` is used: `none` makes no curator calls, so
the one behavioural change cannot reach it.

## 10. Deviations log

| date | change | when | why it is not result-chasing |
|---|---|---|---|
| 08-17 | Co-citation designed and implemented | **after the three-arm results** | Post-hoc, stated in §0. Gates pinned from held-out **training** lift, never test. Correct-only filtering and decay were both tested offline and rejected on the evidence. |
| 08-18 | `_cocite_stats` counts live nodes only | before any run | Reporting only; the walk and the rendering already filtered. Fixed because the inflated figures landed on `cocite_max_degree`, the §7 gate. |
| 08-18 | `text` re-run rather than reused | before any run | §3. The alternative is reverting the sanitiser fix |
| 08-19 | `--cocite_min_rate` re-pinned **0.30 → 0.20** | after the seed-0 cocite run, **before its accuracy was read** | 0.20 was named as the fallback in §2 before any run, so the value is not new. The trigger is the §7 mechanism diagnostic, which is the ordering this pre-registration exists to enforce: 0.30 gave 14 live edges and a 0.0689 walk share, below every pre-registered graph arm. `final_test_results.json` for that run was never opened. The 0.30 run is kept at `results/formula/cocite_experiment/_superseded_rate030/` as the evidence. |
| 08-19 | `analyze_walk_share.py` walks `co_cited` when the run had `--cocite` | before any cocite result was quoted | A defect, not a choice: the script contradicted the run's own counter and the script was wrong. Verified inert on non-cocite runs, so no published figure moves. |
| 08-18 | `graph` moved from a run to a **declared conditional** (§4a) | before any run | The design is the three-arm control/baseline/treatment. `graph → cocite` faces the same ~4pp floor, so on a null primary it buys a second null for ~21h. The direction of the conditional is fixed here so that adding the arm after a positive primary is part of the design rather than a reaction to it. |
| 08-21 | `cocite` seed 2 **re-run** after the original was found byte-identical to seed 1 | after seeing both runs' results | Not a re-roll of an unwelcome number. The design pre-registered three replicates and three were run; the duplicate was found by hashing the prediction files, which is the check §8 mandates given that `--seed` is inert. The primary was already null under the two-replicate pooling (+3.25pp, p=0.1249) and stayed null at three (+2.17pp, p=0.3344), so no conclusion turns on the repair. The Ollama server was restarted to force a distinct state and the change verified with a 200-call `none` probe before the arm was launched. Duplicate archived at `results/formula/cocite_experiment/_determinism/`. |
| 08-21 | **Lift gate measured and rejected** | after the runs, offline, no new compute | The §5 primary was already null and this could only have become a *new* experiment, not a reinterpretation of this one. Measured with `calibrate_cocite_gate.py` at matched edge count on held-out generations: confidence reaches 17.53x base rate at k=10 against lift's 0.31x, and support floors of 5, 10, 20 and 40 never reverse the ordering at any k. Recorded here so the branch is closed on the record. See `COCITE_RESULTS.md` §6. |
| 08-23 | §4a conditional evaluated: **did not fire** | after the primary, per the rule fixed before any run | The commitment binds in both directions and the direction was fixed in advance. `text → cocite` therefore stays undecomposed and no arm was added after seeing the number. |
| 09-19 | **§6's "~4pp floor" relabelled; its wording is kept** | after the runs, arithmetic only, no new compute | ~4pp is the gap needed to reach significance (1.96 x the cluster SE of 2.03pp over 50 formulas), which a true effect of that size clears only about half the time. At 80% power the pooled design needs about **5.7pp** (2.80 x SE). No decision changes: the §4a trigger was defined as "the cluster bootstrap CI excludes zero", a significance criterion, so it fired or not exactly as before. The §6 declaration that a null below ~4pp is uninformative extends to ~5.7pp. The primary's CI, [−0.020, +0.060], cannot exclude an advantage of up to about 6pp. Corrected in place in `COCITE_RESULTS.md` §3, §5 and §6. |
