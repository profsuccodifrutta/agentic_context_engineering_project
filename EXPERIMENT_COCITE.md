# Experiment pre-registration — co-citation edges on `formula`

**Written 2026-08-18, before any run of this set.** Nothing below was chosen after seeing a test
number from it.

---

## 0. Status: this experiment is post-hoc, and says so first

The three-arm formula experiment (`EXPERIMENT_FORMULA.md`, results in
`FORMULA_RESULTS_PREREGISTERED.md`) is complete. It found that **memory helps** and that
**structure was not shown to**:

| question | answer | figures |
|---|---|---|
| Does memory help at all? | **Yes for `graph`, marginal for `text`** | `none → graph` +8.00pp, p=0.0146; `none → text` +5.17pp, p=0.0922 |
| Is relational memory better than linear? | **Not shown** | `text → graph` +2.83pp, CI [−0.017, +0.078], p=0.2905 |

This experiment was designed on 2026-08-17, **after** that null was analysed and diagnosed.
Designing an arm because you saw a null is a researcher degree of freedom, and no justification
makes it not one. What follows is pre-registered before its own runs, which is the most that can
be claimed for it, and it is reported as a follow-up rather than as part of the pre-registered
result. **The pre-registered nine runs are untouched and remain the answer to the three-arm
question.**

## 1. Why co-citation, specifically

The `text → graph` null has a mechanical diagnosis, not a statistical one. `related_to` is cosine
similarity between two rules' **texts**, while retrieval scores rules by cosine between the
**question** and each rule. Those are nearly the same signal, so the walk landed on rules that
relevance ranking would have selected anyway. Measured: a **random rewiring** at matched live-node
and edge count surfaced **2.3–3.6× more** novel rules than the real graph
(0.4761/0.3333/0.5011 against 0.2066/0.0939/0.2050). The treatment fired; its edges were largely
redundant with the scoring that chose the seeds.

The test for whether a relation earns a graph is **not** "could a text file store it" — anything
serialises. It is **is the relation a function of the items, or an irreducible fact about the
pair?**

| relation | pairwise? | derivable from item content? | needs a graph? |
|---|---|---|---|
| `refines` | no — unary ("this rule is dead") | — | no: a flat list with a `supersedes` field |
| `related_to` | yes | **yes** — cosine of two texts | no: a flat list plus an embedding index |
| `co_cited` | yes | **no** — exists only in accumulated history | **yes** |

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
`--cocite_min_opportunity 3 --cocite_min_rate 0.30 --cocite_half_life 0`.

Design decisions, each measured rather than assumed:

- **All citations, not correct-only.** Filtering to correct generations halves the edge count for
  no lift gain and loses in two of three seeds. Co-citation captures "these rules address the same
  kind of problem", which is orthogonal to whether the model got it right.
- **Decay implemented, defaulted off.** Half-life 100 measured +0.05 lift for ~10% fewer edges —
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

## 3. Why `text` and `graph` must be re-run

Between the pre-registered graph arm's commit `1acfe69` and this experiment's commit, the run-path
diff is ~800 insertions. **Exactly one change is behavioural for a non-cocite run** — established
by reading the diff, not assumed:

- `ace/ace.py` — a hoisted `_render(question)` producing the identical string, plus
  `record_generation`, a no-op on the `Memory` ABC that early-returns in `GraphMemory` when
  `cocite=False`.
- `logger.py` — a write-only `bullet_ids_rendered` field, plus removal of dead key-mixer plumbing.
- `llm.py`, `utils.py` — dead-code removal only.
- `ace/memory/graph.py` — gated behind `--cocite`, plus a reporting-only stats fix.
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
pre-registered `graph` runs would differ in **two** places — the relation, and how much
pasted-prefix junk survives into rule text. That is the confound the fairness rules exist to
prevent.

**`none` is exempt and is reused unchanged.** Verified: all three control runs make **generator
calls only** and log **zero** curator operations, so they never reach the sanitizer and are
invariant to this change.

**Reverting the regex was considered and rejected.** It would restore comparability for ~42h of
saved compute, but it selects code to fit a desired comparison, runs the experiment on knowingly
defective code, and — since the defect injects noise, and noise shrinks differences — biases
toward the very null this experiment exists to move away from.

## 4. Design

Four arms, seeds 0/1/2, `task_config_v2.json` (250 train / 100 val stratified / 200 test).

| arm | `--memory` | `--retrieval` | `--cocite` | source |
|---|---|---|---|---|
| none | `none` | `none` | — | **reused** from `results/formula/preregistered/seed*/none` |
| text | `text` | `flat` | off | re-run |
| graph | `graph` | `graph` | off | re-run |
| cocite | `graph` | `graph` | **on** | new |

Nine new runs. Everything else pinned exactly as the pre-registered set: `gemma4:e2b` in all three
roles, `--think_generator off`, `--num_ctx 32768`, `--dedup --dedup_threshold 0.88`,
`--eval_steps 125`, `--retrieval_budget 2500`, `--retrieval_seeds 5`, `--test_workers 1`,
`curator_frequency 1`, `max_num_rounds 3`, `num_epochs 1`. All nine share one commit.

## 5. Outcomes, declared in advance

**Primary — `text → cocite`.** Clustered sign-flip permutation test on replicate-pooled scores
(`compare_arms.py::pooled_arm_scores` → `cluster_paired_stats`), blocking on formula, with the
cluster bootstrap CI. This is the headline claim — *relational memory beats linear memory* — and
it is the **only** comparison declared primary.

**Secondary — `graph → cocite`.** The mechanism check. These two arms differ in exactly one thing,
which relation the graph holds, so this is where attribution to the pairwise relation lives.
`text → cocite` licenses "this relational configuration beats linear memory"; **only
`graph → cocite` licenses "the pairwise relation is what did it."** Both are reported.

**Secondary — `none → cocite`**, against the reused controls.

**Declared a replication check, not a result — the new `text → graph`.** Re-running these arms
produces a second estimate of a question already answered. The pre-registered **+2.83pp,
p=0.2905 remains the answer**; the new figure is reported as a replication check and **must not be
substituted for it whichever way it lands.** Expected: a small level shift in both arms from the
regex fix, no change in the gap.

Six pairwise comparisons are available across four arms. Declaring the primary in advance is what
keeps the remaining five from becoming five extra chances at significance.

## 6. Power, stated before the runs

The test split is 50 formulas × 4 instances, and instances of one formula fail together. Power
scales with **clusters, not rows**, and the measured ceiling is a **~3.5pp CI half-width even at
infinite replicates** — which is why more replicates were ruled out for the original experiment.

**`text → cocite` must reach roughly +4pp to exclude zero.** A gap of +3pp will be null.

**Declared now: a null below ~4pp is uninformative about the relation, not evidence against it**,
and will be reported as an underpowered non-result rather than a negative one. A reader is
entitled to know this before the number exists.

## 7. What would make a run mechanism-inconclusive

Read from `structural_stats()` **before quoting any accuracy**. Any of these means the treatment
was not delivered, and the run is inconclusive rather than negative:

- **`cocite_edges` near zero** — the gates never fired.
- **`cocite_max_degree` approaching the live node count** — inheritance concentrated mass into a
  hub, the walk reaches everything in one hop, and `related_to`'s failure is recreated.
- **`walk_only_share` near zero** (via `analyze_walk_share.py`, final-test pooled, **not** the
  run-cumulative counter in `structural_stats()`) — the arms read the same rules.
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
  `text → cocite` gap is not attributable to the pairwise relation alone — which is exactly why
  `graph → cocite` is reported beside it.
- **`num_epochs 1`** against the paper's 5. Disclosed deviation, identical in all arms.
- **`--seed` is inert at temperature 0**; replicates differ through GPU floating-point reduction
  order, which the ACE loop amplifies ~37×. No run here is described as reproducible.
- **Circularity in co-citation**: a pair can only be co-cited if co-rendered, and co-rendering is
  driven by the edges. Normalising by opportunity fixes numerator inflation but not the sampling
  bias. Measured coverage 41–47% of the pair space with no exploration mechanism.

## 9. Procedure, in order

1. Confirm `git status` clean and all nine runs launch from one commit.
2. `check_memory_equivalence.py`, `check_prompts.py`, `check_finance_wiring.py` — all pass.
3. Nine runs into `results/formula/cocite_experiment/seed{0,1,2}/{text,graph,cocite}`.
   **Read the first ten calls of every run**: under ~20 tok/s means the model was placed on CPU,
   and the run should be killed and restarted rather than left to take ~53h.
4. `check_learning_signal.py` on one run per arm — a dead signal invalidates that arm.
5. Analysis exactly as §5 via `compare_arms.py`; cost via `aggregate_run_costs.py`;
   `analyze_walk_share.py` on both graph arms.

`compare_arms.py` will flag `git_commit` between the reused `none` runs and the new three.
**The pre-registered set's licence — "the run-path diff is empty" — does not apply here.** The
licence is narrower and must be stated when `--force` is used: `none` makes no curator calls, so
the one behavioural change cannot reach it.

## 10. Deviations log

| date | change | when | why it is not result-chasing |
|---|---|---|---|
| 08-17 | Co-citation designed and implemented | **after the three-arm results** | Post-hoc, stated in §0. Gates pinned from held-out **training** lift, never test. Correct-only filtering and decay were both tested offline and rejected on the evidence. |
| 08-18 | `_cocite_stats` counts live nodes only | before any run | Reporting only; the walk and the rendering already filtered. Fixed because the inflated figures landed on `cocite_max_degree`, the §7 gate. |
| 08-18 | `text` and `graph` re-run rather than reused | before any run | §3. The alternative — reverting the sanitiser fix — would have selected code to fit the comparison. |
