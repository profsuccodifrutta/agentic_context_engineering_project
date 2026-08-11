# Formula results — pre-registered runs

**Status: seed 0 of 3 complete. The pre-registered decision rule cannot be evaluated yet.**

`EXPERIMENT_FORMULA.md` §5 pre-registers three seeds per arm and a rule that compares the
between-arm gap against the within-arm seed sd. At one seed there is no sd, so that rule is
undefined in either direction. What follows is a complete description of three runs, and it is
written up now so that the seed-1 and seed-2 analysis is a confirmation rather than a first look.

These are **not** the runs in `FORMULA_RESULTS.md`. Those were exploratory, are superseded, and
are excluded from the primary analysis for the reasons in their own §6.

---

## 1. Setup

| | |
|---|---|
| task | `formula` (`eval/finance`), 50 formulas |
| splits | `task_config_v2.json` — train 250 (5/formula), val **100 (stratified, 2/formula)**, test 200 (4/formula) |
| generator / reflector / curator | `gemma4:e2b` for all three roles |
| thinking | `--think_generator off`; reflector and curator at model default |
| `--num_ctx` | 32768 requested and **32768 served** (re-measured 2026-08-08) |
| `--seed` | 0 |
| `curator_frequency` / `num_epochs` / `max_num_rounds` | **1 / 1 / 3** |
| `eval_steps` | 125 (two checkpoints) |
| grow-and-refine | **on** — `--dedup --dedup_threshold 0.88` |
| retrieval | `none` / `flat` / `graph`, `--retrieval_budget 2500 --retrieval_seeds 5` |
| commit | `1acfe69` — **all three arms** |

Every difference from the exploratory runs is a defect fix or a pre-registered design change made
before any test number was seen (`EXPERIMENT_FORMULA.md` §10). One commit across all three arms,
so the protocol slip recorded in `FORMULA_RESULTS.md` §1 does not recur.

## 2. Final test accuracy (n = 200)

| arm | correct | accuracy | Wilson 95% | no_answer |
|---|---|---|---|---|
| `none` | 125 / 200 | **0.6250** | [0.556, 0.689] | 0 |
| `text` | 141 / 200 | **0.7050** | [0.638, 0.764] | 0 |
| `graph` | 143 / 200 | **0.7150** | [0.649, 0.773] | 0 |

Both memory arms now sit **above** the control, reversing the exploratory result's direction. The
paper reports Base 67.5 → ACE 85.5 on this benchmark at 671B; ours is 62.5 → 70.5 at 2B.

The Wilson intervals overlap heavily and are the wrong instrument here — they are marginal and
unpaired, and the test set is not iid. §3 is the reportable comparison.

## 3. Paired comparison

Quote the **clustered** row. The test split is 50 formulas × 4 instances, the instances of one
formula succeed and fail together, and both McNemar and Wilson report the precision of 200
independent draws for a design that has ~50.

| comparison | b (fixed) | c (broken) | McNemar p | clustered diff | bootstrap 95% CI | sign-flip p |
|---|---|---|---|---|---|---|
| none → text | 29 | 13 | 0.0195 | +0.0800 | [+0.0100, +0.1550] | 0.0548 |
| none → graph | 31 | 13 | 0.0096 | +0.0900 | [+0.0150, +0.1700] | **0.0364** |
| **text → graph** | **23** | **21** | **0.8804** | **+0.0100** | **[−0.0650, +0.0900]** | **0.9036** |

**The primary comparison is null.** `graph` beats `text` by one percentage point with a CI that
straddles zero and p=0.90. Whatever relational structure is doing, at this seed it is not
accuracy.

Three pairwise tests, no multiple-comparison correction. Under Bonferroni (α=0.0167) the
none→graph result does not clear either. The defensible one-sentence reading of seed 0 is:
*memory is worth roughly 8–9 points and that is marginally supported; structure is worth nothing
measurable.*

## 4. The mechanism was live — this is a negative result, not a broken run

`EXPERIMENT_FORMULA.md` §7 pre-registers the falsification: **graph ≈ text with `walk_only_share`
near zero means the arms were reading the same rules**, which is a broken run rather than
evidence about structure.

| checkpoint | rules rendered / generation | reached only by the walk | `walk_only_share` |
|---|---|---|---|
| step 125 (**the one tested**) | 19.17 | 2.88 | **0.1504** |
| step 250 | 19.32 | 3.44 | 0.1781 |

At the tested checkpoint, **15% of what the generator read was selected by following
`related_to` edges and would not have been selected by relevance alone** under the same budget,
across 817 retrieval calls. The treatment fired on essentially every generation.

So the arms genuinely read different things, and reading different things did not help. That is
the finding.

## 5. The difference is directionless, not absent

`text → graph` moves **44 of 200 rows** — 23 fixed, 21 broken, 120 correct in both, 36 wrong in
both. Blocked by formula: graph is better on 15 formulas, worse on 12, identical on 23.

This is the signature described in `CLAUDE.md` under "The learning signal": a memory with a large
effect and no direction. Structure changes about a fifth of the answers; the changes cancel.
Accuracy alone could not have told this apart from an inert treatment, which is why the paired
test is the reported one.

Where memory's real effect shows up is in consolidating partly-solved formulas:

| arm | formulas at 0/4 | 1/4 | 2/4 | 3/4 | 4/4 |
|---|---|---|---|---|---|
| `none` | 0 | 9 | 15 | 18 | 8 |
| `text` | 1 | 0 | 19 | 17 | 13 |
| `graph` | 0 | 3 | 14 | 20 | 13 |

Both memory arms convert 1/4 formulas into 2–4/4 formulas and grow the fully-solved count from 8
to 13. Neither rescues a formula the control failed completely, because there are none — the
control already gets at least one instance of all 50. `text` is the only arm that loses a formula
outright.

## 6. The loop is healthy, which is what makes the null credible

`check_learning_signal.py` passes all six checks on both memory arms. Against the exploratory
runs, whose null was produced by a broken pipeline:

| | exploratory | `text` (now) | `graph` (now) |
|---|---|---|---|
| `refines` non-null | 2 / 50 calls | 50 / 250 | **64 / 250** |
| `harmful` tags emitted | 0 | 94 | 159 |
| dedup supersessions | mechanism off | active | 145 (+36 curator-declared) |
| ADDs adding nothing new | ~57% | 34.9% | 34.1% |
| generator unparseable | — | 0 / 1026 | 0 / 1018 |
| distinct bullets cited | — | 95 | 105 |

The required-but-nullable `refines` field, the `helpful`-on-failure demotion, the environment
feedback fix and `--dedup` all did what they were introduced to do. The curator still pastes
rendered-bullet prefixes into rule text at a high rate (53.2% text / 47.1% graph of ADDs), but
`sanitize_curator_operations` strips them at run time and the *post*-sanitiser duplicate rate is
under the 50% gate in both arms.

## 7. Checkpoint selection

| arm | step 125 | step 250 | selected |
|---|---|---|---|
| `text` | val 0.7400 | val 0.7100 | **125** |
| `graph` | val 0.7400 | val 0.7400 | **125** (tie → earlier) |

Both arms tested their step-125 playbook, so the concern that the arms might be compared at
different checkpoints does not arise at this seed.

The graph arm's tie is exact but not vacuous: the two checkpoints disagree on **12 of 100
validation rows, 6 in each direction**. They are genuinely different objects that scored
identically. Validation therefore supplies *zero* evidence for preferring either, and the
pre-registered rule (`ace/ace.py:1000`, strict `acc > best_accuracy`) resolved it toward the
smaller playbook — 81 rules and 10,389 tokens against 130 rules and 18,944.

For the text arm the same 125-vs-250 comparison moves 17 rows and 0.03 accuracy, so its selection
is a real preference rather than a tie-break.

## 8. Cost

| arm | calls | tokens | LLM h | tok/s | memory tokens @125 |
|---|---|---|---|---|---|
| `none` | 200 | 201,829 | 0.31 | 108 → 108 | — |
| `text` | 1577 | 7,282,639 | 4.49 | 101 → 86 | 6,604 |
| `graph` | 1565 | 9,096,412 | 17.31 **⚠** | 96 → 90 (min 3) | 10,389 |

**⚠ The graph arm's hours are contaminated and must not be quoted as a cost of the arm.** Serving
throughput swung 36.3× mid-run during the endpoint incident logged in `EXPERIMENT_FORMULA.md` §9.
Tokens and accuracy are unaffected; only timing is.

Structure costs **+24.9% total tokens for +1.0pp accuracy (p=0.90)**. The split is the
informative part:

| role | `text` in-tok/call | `graph` in-tok/call | delta |
|---|---|---|---|
| generator | 3,070 | 3,410 | **+11%** |
| curator | 8,490 | 14,018 | **+65%** |

The generator figure is the retrieval budget working as designed: identical 2,500-token cap in
both arms, so the graph arm shows fewer rules for the same tokens and pays for its own
annotations. The curator is unbudgeted by design (`ace/ace.py:769` — a curator that sees a
retrieved slice cannot tell whether a rule already exists), and that is where the whole overhead
lands. At step 125 the same 81 rules render as 6,604 tokens flat and 10,389 as a graph, **+57%
for identical content**.

### Headroom warning for seeds 1 and 2

Curator prompt growth, measured (server tokens):

| arm | step 1 | step 250 | growth | ceiling |
|---|---|---|---|---|
| `text` | 1,531 | 13,509 | 48.1 tok/step | ~31,000 |
| `graph` | 1,375 | **25,665** | 97.6 tok/step | ~31,000 |

The graph arm finished at **83% of the practical ceiling**, with roughly 54 steps of headroom
left. §9's projection was 26,256 against 25,665 measured, so the model is good. But a seed that
accumulates slightly faster hits the wall, and the failure is silent — over the line the model
does not error, it thinks until the budget is gone and returns an empty response. Read the last
curator call's `prompt_num_tokens` on each remaining graph run.

## 9. Criticism

1. **One seed.** The pre-registered noise-vs-effect rule cannot run. Nothing here is a result
   about memory structures yet.
2. **Three pairwise tests, no correction.** Stated, not corrected. Under Bonferroni nothing
   clears.
3. **The extraction paths differ by 6.5pp.** Share of answers not read by a strict parse: none
   47.0%, graph 52.0%, text 53.5% — all LaTeX escape repairs, `no_answer=0` everywhere. The
   spread is small and runs *against* the graph arm's favour, but it is a between-arm difference
   that is not the independent variable.
4. **The graph arm's timing is unusable** (§8), so any efficiency claim rests on tokens alone.
5. **`contradicts` edges are off**, so "graph" here means `related_to` only. The treatment is
   weaker than the full design.
6. **`num_epochs=1` against the paper's 5**, a disclosed deviation, and the one most likely to
   under-train both memory arms relative to the published method.
7. **2B models throughout.** The paper's result is at 671B, and §6.1 of the exploratory write-up
   argues the curator is where small models break down. The curator-hygiene rates here (≈50% of
   ADDs carrying pasted prefixes) say that has not gone away, only that it is now sanitised.
8. **The checkpoint tie in the graph arm was resolved by a rule, not by evidence** (§7). A
   different tie-break would have tested a different playbook.

## 10. Conclusion, so far

At seed 0 with a healthy loop, a correctly-configured baseline and the retrieval mechanism
demonstrably firing on 15% of rendered rules:

- **Memory helps.** +8.0pp (text) and +9.0pp (graph) over the no-memory control, marginally
  supported once the test set's block structure is respected.
- **Structure does not.** +1.0pp, CI [−0.065, +0.090], p=0.90, for +24.9% tokens.
- **Structure is not inert** — it moves 44 of 200 rows and 27 of 50 formulas. It simply moves
  them in both directions equally.

Unlike the exploratory runs, this is informative about the hypothesis rather than about the
harness. Whether it is *reportable* depends on seeds 1 and 2.

## Reproducing

```
uv run python -m eval.finance.run --task_name formula --mode eval_only \
  --save_path results/formula/preregistered/seed0/none --memory none --retrieval none \
  --config_path ./eval/finance/data/task_config_v2.json \
  --generator_model gemma4:e2b --reflector_model gemma4:e2b --curator_model gemma4:e2b \
  --think_generator off --num_ctx 32768 --seed 0 \
  --dedup --dedup_threshold 0.88 --eval_steps 125 --retrieval_budget 2500 --retrieval_seeds 5

uv run python -m eval.finance.run --task_name formula --mode offline \
  --save_path results/formula/preregistered/seed0/text --memory text --retrieval flat \
  ... same flags ... --skip_initial_test

uv run python -m eval.finance.run --task_name formula --mode offline \
  --save_path results/formula/preregistered/seed0/graph --memory graph --retrieval graph \
  ... same flags ... --skip_initial_test
```

Analysis:

```
uv run python compare_arms.py results/formula/preregistered/seed0/*/ace_run_*
uv run python aggregate_run_costs.py results/formula/preregistered/seed0/*/ace_run_*
uv run python check_learning_signal.py results/formula/preregistered/seed0/text/ace_run_*
uv run python check_learning_signal.py results/formula/preregistered/seed0/graph/ace_run_*
```

---

## Appendix A. Notes for the final write-up — **delete this section when §1–§10 are rewritten**

Working notes only, added 2026-08-11 while seeds 1 and 2 were still running. Nothing here is a
result; it is the list of things that must not be forgotten once all nine runs land.

**A.1 — the noise floor is the headline, not the accuracy table.** Measured at two replicates
(`EXPERIMENT_FORMULA.md` §5): `none` moves 0.5pp / **1 row** between seed 0 and seed 1; `text`
moves 3.5pp / **37 rows** (15 fixed, 22 broke). The ACE loop amplifies serving nondeterminism
**~37×** in rows moved. The seed-0 graph−text gap was +1.0pp / 44 rows — smaller than the text
arm's movement against itself. **Lead with this.** The primary comparison is not merely null,
it is under-powered by a factor of ~3.5, and that was measured rather than assumed.

**A.2 — `--seed` is inert; say so plainly and early.** It never touches the data and at
temperature 0 the sampler never reads it (`EXPERIMENT_FORMULA.md` §4 has the measurement and the
divergence-shape evidence). §1's setup table row "`--seed` | 0" is *not* wrong, but on its own it
implies a control that does not exist. Rewrite the three runs as **replicates**, and state that
training order is fixed so the sd is a **lower bound** on true run-to-run variance.

**A.3 — the structure-stability test, once the seed-1 graph arm lands.** §4's amplification
figure compares `none` against `text`, so what is established is that *memory* amplifies
nondeterminism, **not** that *structure* does. Run the same paired-flip count on `graph`
seed 0 vs seed 1. If graph moves materially more than text's 37 rows, that is a real cost of
the treatment and belongs in §9's criticism list; if it moves less, that is a point in the
graph arm's favour and must be reported with equal prominence. No hypothesis was
pre-registered on it, so report it descriptively either way.

**A.4 — do not let seeds narrow anything.** The cluster bootstrap resamples formulas and all
replicates see the same 50, so the CI cannot tighten across seeds. Report mean ± sd across
replicates, and never sd/√3 as a standard error.

**A.5 — carry forward from the seed-0 draft.** The `walk_only_share` figures (0.1504 at step
125, 0.1781 at step 250) still matter: they are what proves the graph mechanism was live while
producing no benefit, which is the difference between a negative result and a broken run.
Likewise keep the §8 headroom warning — the graph curator peaked at 25,665 server tokens
against the ~31,000 ceiling, and the overflow failure is silent.

**A.6 — `compare_arms.py` will need `--force`** across the nine runs, because `git_commit`
differs. Justification and the `git diff` that licenses it: `EXPERIMENT_FORMULA.md` §9,
2026-08-10. Re-run that diff before quoting anything.

**A.7 — if seed 2 is not run, say so in §1 in the first paragraph**, with the reason, rather
than leaving the reader to count the runs. Two replicates with the deviation disclosed is
defensible; two replicates presented as if three were planned is not.

**A.8 — quote the pooled number as the headline for every arm pair.** `compare_arms.py` now
prints a `pooled across replicates` section above the pairwise block
(`EXPERIMENT_FORMULA.md` §5, procedure fixed 2026-08-12 before seed 2 ran). That is the
pre-registered primary statistic; the per-replicate results go beneath it, so a reader can see
pooling did not flatter the outcome. Three traps to avoid in the prose:

- **Never report the pooled row count as 600.** It is 200 rows in 50 blocks, scored over three
  replicates. Pooling removes run noise, not formula-sampling variance.
- **No McNemar on the pooled line** — it is undefined on averaged scores, and the tool omits it
  deliberately rather than by oversight.
- At R = 2 the pooled `none → text` was +6.5pp, CI [+0.005, +0.128], p=0.0600, where both
  individual replicates' CIs crossed zero. **If the R = 3 pooled result lands just either side
  of α, say so plainly and do not lean on the CI-excludes-zero framing alone** — the bootstrap
  CI and the permutation p are different procedures and need not agree at the boundary.
