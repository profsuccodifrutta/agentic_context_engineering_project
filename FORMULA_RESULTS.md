# Formula results — seed 0, exploratory

**Status: exploratory. One seed per arm. No conclusion is statistically licensed.**

`EXPERIMENT_FORMULA.md` pre-registers three seeds per arm and a decision rule that compares the
between-arm gap against the within-arm seed sd. At one seed there is no sd, so that rule cannot
be evaluated in either direction. Everything below is a description of three runs, not a result
about memory structures. It is written up because the *mechanism* it exposes changed the design
of the experiment — see §6.

These runs are superseded by the redesign in §7 and are excluded from the primary analysis.

---

## 1. Setup

| | |
|---|---|
| task | `formula` (`eval/finance`), 50 formulas |
| splits | train 250 (5/formula), val 300 (2–11/formula), test 200 (4/formula) |
| generator | `qwen3.5:2b`, `--think_generator off` |
| reflector / curator | `gemma4:e2b`, thinking left at model default |
| `--num_ctx` | 32768 requested; **~16,384 actually served** (Ollama clamps silently) |
| `--seed` | 0 |
| `curator_frequency` / `num_epochs` / `max_num_rounds` | 5 / 1 / 1 |
| `eval_steps` | 125 (two checkpoints) |
| grow-and-refine | **off** (`--use_bulletpoint_analyzer false`) |
| retrieval | none — whole memory injected on every call |
| commit | `b778bed` (none, text), `24703b2` (graph) |

The graph arm ran on a different commit from its siblings. The diff is `CLAUDE.md` and
`compare_arms.py` only (`git diff --name-only b778bed..HEAD`), and nothing in the run path
imports `compare_arms`, so it is behaviourally inert — but it is a protocol slip against the
pre-registration's "all runs share one SHA" rule and `compare_arms.py` requires `--force`
because of it. Recorded rather than waved away.

## 2. Final test accuracy (n = 200)

| arm | correct | accuracy | Wilson 95% | no_answer |
|---|---|---|---|---|
| `none` | 122 / 200 | **0.6100** | [0.541, 0.675] | 16 |
| `text` | 107 / 200 | **0.5350** | [0.466, 0.603] | 18 |
| `graph` | 109 / 200 | **0.5450** | [0.476, 0.613] | 15 |

Both memory arms land below the no-memory control. The paper reports the opposite direction on
this benchmark (Table 2, Formula, DeepSeek-V3.1-671B): Base LLM 67.5 → ACE with GT labels
**85.5**, a gain of +18.0. Ours is −7.5.

## 3. Paired comparison

The Wilson intervals above assume 200 independent trials. **They are not.** The test split is
50 formulas × 4 instances, and instances of one formula succeed or fail together — a playbook
carrying the wrong convention for "Sharpe Ratio" gets that formula's four rows wrong as a
block. The effective sample size is closer to 50 than 200. Both tests are reported.

| pair | b (fixed) | c (broken) | net | McNemar p | clustered diff | bootstrap 95% CI | sign-flip p |
|---|---|---|---|---|---|---|---|
| none → text | 17 | 32 | −15 | **0.0444** | −0.0750 | [−0.165, +0.010] | 0.1249 |
| none → graph | 19 | 32 | −13 | 0.0919 | −0.0650 | [−0.145, +0.010] | 0.1551 |
| text → graph | 24 | 22 | +2 | 0.8830 | +0.0100 | [−0.070, +0.090] | 0.9053 |

**The one significant result does not survive clustering.** none → text is p=0.0444 under
McNemar and p=0.1249 once the formula, not the row, is the unit of resampling. The point
estimate is unchanged; the interval crosses zero. So the honest statement is *"the memory arms
sit below the control by about 7 points, and at one seed with 50 blocks that gap is inside
noise"* — not *"the memory significantly hurt"*.

The clustered statistics are checked against their known limiting cases by
`uv run python compare_arms.py --selftest`: with singleton blocks the sign-flip test reproduces
McNemar exact to Monte Carlo error, and with perfectly correlated blocks of 4 it widens the
interval by exactly ×2.00, the design effect √4.

**text → graph is a wash with high churn.** 24 fixed, 22 broken, 16 formulas better and 16
worse. Not an inert memory — a memory with a large effect and no direction.

## 4. Where the loss came from

### 4.1 By target magnitude

| band | n | none | text | graph | text−none | graph−none |
|---|---|---|---|---|---|---|
| \|t\| < 1 (raw ratio) | 54 | 0.556 | 0.426 | 0.519 | **−13.0pp** | −3.7pp |
| 1–100 (percent-shaped) | 90 | 0.678 | 0.656 | 0.644 | −2.2pp | −3.3pp |
| > 100 (absolute/scaled) | 56 | 0.554 | 0.446 | 0.411 | **−10.7pp** | **−14.3pp** |

The middle band is nearly untouched and both tails collapse. The final playbook contains a
family of rules of the form *"always prioritize the conventional financial presentation format
(percentage) over the strict mathematical decimal ratio"* — correct for percent-shaped answers,
wrong for raw ratios and wrong for absolute quantities. A single over-general rule, applied
everywhere, is the shape of the damage.

### 4.2 Per formula

| pair | better | worse | equal |
|---|---|---|---|
| text vs none | 9 | 16 | 25 |
| graph vs none | 10 | 14 | 26 |
| graph vs text | 16 | 16 | 18 |

Worst hit (none → text): Jensen's Alpha 4→0, Enterprise Value 4→0, Dividend Yield 3→0,
Sharpe Ratio 4→1. Best: DDM 1→3, Future Value 0→2, ROA 0→2.

### 4.3 Not an extraction artefact

| arm | strict parse | escape repair | regex fallback | no answer |
|---|---|---|---|---|
| none | 134 (67.0%) | 20 (10.0%) | 30 (15.0%) | 16 (8.0%) |
| text | 100 (50.0%) | 58 (29.0%) | 24 (12.0%) | 18 (9.0%) |
| graph | 114 (57.0%) | 55 (27.5%) | 16 (8.0%) | 15 (7.5%) |

The arms lean on the recovery paths by materially different amounts — a 17.0pp spread in the
share not read by a strict parse — and `compare_arms.py` warns about it. It is not the driver
of the gap, though:

- The `[No answer]` losses are genuine model failures, not scoring losses. Only 3 of the 16 in
  the `none` arm were "field present but unreadable"; the rest emitted no `final_answer` at
  all. The only harness-side losses are in the *control*, i.e. against the direction of the gap.
- Conditioning on the 171 problems where **neither** arm lost the answer, the none→text gap is
  **−8.2pp** — wider than the headline. On the 76 where both parsed strictly, −5.3pp.

Every repair path runs only after a strict parse has already failed and was verified across all
1,791 logged generator responses to change 0 answers.

## 5. Cost

| arm | calls | tokens | LLM h | final memory |
|---|---|---|---|---|
| none | 200 | 0.44 M | 0.91 | — |
| text | 1,715 | 9.57 M | 7.09 | 75 rules / 6,270 tok |
| graph | 1,715 | 11.68 M | 7.63 | 63 active of 65 nodes / 7,590 tok |

The memory arms cost **21.5× and 26.3×** the control's tokens to end up below it. The graph
rendering is +22.0% tokens over the flat one for 16% fewer rules — about +44% per rule.

**The memory is spent out of the generator's own window.** Measured on the final test: the
`none` arm's longest generation is 15,997 output tokens on a 445-token prompt; the `text` arm's
is 9,293 on a 7,153-token prompt. Both land within ~60 tokens of 16,384, so the served context
is the wall. A 75-rule playbook is 7,103 prompt tokens at the median and therefore takes **43%
of the generator's window away from generation**. The memory arm is not "the same model plus
rules", it is "the same model plus rules and 6.7k fewer tokens to reason in" — an
arm-asymmetric constraint by construction, and a competing explanation for the loss.

## 6. The loop worked; the architecture did not

`check_learning_signal.py` exits 0 on both training arms. This is not a broken run.

| | text | graph |
|---|---|---|
| `harmful` tags emitted | 34 | 46 |
| `refines` present / non-null | 49 / 50, 6 non-null | 49 / 50, **2 non-null** |
| train accuracy | 0.540 | 0.540 |
| generations citing a bullet | 236 / 250 | 234 / 250 |
| generator unparseable | 6.8% | 6.8% |
| curator unparseable | 0% | 0% |

Three findings sit underneath that healthy scorecard, and together they are why the experiment
was redesigned rather than replicated at more seeds.

### 6.1 The curator was doing about a third less work than it appeared to

Measured with `replay_curator_sanitizer.py`:

| arm | ADDs | pasted rendered-bullet prefix | verbatim re-add | genuinely new |
|---|---|---|---|---|
| text | 75 | 43 (57%) | 23 (31%) | 52 (69%) |
| graph | 65 | 34 (52%) | 9 (14%) | 56 (86%) |
| all 12 MBPP runs | 263 | **0** | 0–12% | — |

The curator pasted the playbook's own rendering (`[calc-00013] helpful=5 harmful=0 :: `, once
nested three deep) into the rule text, so a snapshot of some *other* rule's counters entered
the generator prompt as rule text and never updated again. It also re-added rules verbatim,
which an ADD-only curator cannot notice. Both are now enforced in code
(`playbook_utils.sanitize_curator_operations`, commit `1713a8a`); neither is retroactively
repairable, because these runs trained on the corrupted memory.

### 6.2 The graph arm's one capability fired twice

`refines` — the curator naming a rule its new rule supersedes — is the graph arm's only
functional advantage over the flat playbook. It was non-null in **2 of 50** curator calls, so
2 of 65 nodes were ever withdrawn. The graph arm therefore reduces, in practice, to
*text plus 170 annotation edges*, and the dead heat against the text arm is what that predicts.

`related_to` edges are written by cosine similarity and read only for display text and
`structural_stats()`, which is never shown to a model. Nothing — generator, reflector or
curator — selects, filters, ranks or retrieves on connectivity.
`render_for_generator(query=...)` raises `NotImplementedError` in **both** arms.

### 6.3 The baseline was not the paper's method

This is the finding that matters. The paper names three key innovations (§3): the Reflector,
incremental delta updates, and **grow-and-refine** (§3.2) — *"A de-duplication step then prunes
redundancy by comparing bullets via semantic embeddings."* §4.7 calls it a key enabler.

`ace/core/bulletpoint_analyzer.py` **is** that mechanism, and the fairness rules required it off
in every run, on the reasoning that it merges bullets and the graph arm could not mirror the
removal. So:

- The `text` arm is ACE minus one of its three innovations, and it degenerates exactly as
  predicted — 31% duplicate ADDs, 75 rules, 43% of the generator window consumed. What was
  measured is not "linear memory doesn't help" but **"append-only memory without redundancy
  control doesn't help"**.
- `refines` is a sparse, LLM-declared re-implementation of the mechanism the baseline had been
  stripped of. The graph arm's "advantage" was a weaker version of something removed from its
  competitor.
- The fairness rule was applied to a component *of the method under replication*, and resolved
  by disabling it rather than by implementing it in both representations.

Two hyperparameters also drifted from the paper without being logged. Paper §4.2: *"We adopt a
batch size of 1... We set the maximum number of Reflector refinement rounds and the maximum
number of epoch in offline adaptation to 5."* These runs used `curator_frequency=5`,
`max_num_rounds=1`, `num_epochs=1`. The paper's own §4.6 reports that these knobs move results
only modestly, so they are unlikely to explain a 25-point reversal on their own — but they are
undisclosed deviations and are now in `EXPERIMENT_FORMULA.md` §8.

## 7. What replaces these runs

**Those runs exist now — see `FORMULA_RESULTS_PREREGISTERED.md` (seed 0 of 3 complete).**

Both memories get grow-and-refine (so the baseline is actually ACE) and query-conditioned
retrieval under an identical token budget — flat takes top-k by similarity, graph seeds
identically then expands over `related_to`. That makes the arms differ in exactly one thing:
whether the retrieved neighbourhood follows typed edges. Plus `curator_frequency=1`,
`max_num_rounds=3`, and a stratified validation split. Nine runs, one commit, cluster-robust
inference throughout.

## 8. Criticism

1. **One run per arm.** The pre-registered noise-vs-effect rule cannot run. Everything here is a
   description of three runs. Since measured on the pre-registered set, the floor this doc
   lacks is large: a memory arm moves 3.5pp and 37 of 200 rows between two nominally identical
   runs (`EXPERIMENT_FORMULA.md` §5), so **no gap reported below at or under ~4pp is
   distinguishable from re-running the same arm.**
2. **The headline significance is an artefact of the iid assumption**, and is reported as such
   in §3. Any earlier statement of "p=0.044" without the clustered figure beside it is wrong.
3. **Three pairwise tests, no multiple-comparison correction.** Stated, not corrected.
4. **The arms did not get equal generation headroom** (§5). This is inherent to whole-memory
   injection, not a defect, but it is a competing explanation for the entire result and it is
   the reason retrieval was added.
5. **Extraction-path asymmetry of 17.0pp** (§4.3). Conditioned away above, but it remains a
   difference between the arms that is not the independent variable.
6. **The graph arm's commit differs** from its siblings (§1). Inert, but a protocol slip.
7. **The magnitude bands in §4.1 were chosen after seeing the data.** They are descriptive, not
   a test, and nothing in §3 depends on them.
8. **Curator content corruption was present throughout** (§6.1) and is not repairable by
   rescoring, because it degraded the memory that was trained on.
9. **Two of the paper's hyperparameters were wrong** and undisclosed at run time (§6.3).
10. **The 2B/2B generator/curator pairing is far from the paper's 671B.** The paper claims
    robustness to a weak *Reflector* (§4.6) but says nothing about the Curator, and §6.1 is a
    Curator-capability failure. Grow-and-refine is plausibly what makes the paper's ACE tolerant
    of a sloppy curator; without it, ours was not.

## 9. Conclusion

At one seed, both memory arms sit ~6–7 points below the no-memory control, and that gap is
inside noise once the test set's block structure is accounted for. The graph and text arms are
indistinguishable (+1.0pp, p=0.91) while the graph costs 22% more tokens.

None of that is evidence about relational memory, because the mechanism that would make the
graph arm relational was never built (§6.2) and the baseline it was compared against was
missing one of the method's three components (§6.3). The runs are informative about the
*harness*, not the hypothesis.

## Reproducing

```
uv run python -m eval.finance.run --task_name formula --mode eval_only \
  --save_path results/formula/seed0_exploratory/none \
  --memory none --generator_model qwen3.5:2b --think_generator off --num_ctx 32768 --seed 0
uv run python -m eval.finance.run --task_name formula --mode offline \
  --save_path results/formula/seed0_exploratory/text \
  --memory text --generator_model qwen3.5:2b --think_generator off --num_ctx 32768 --seed 0 --skip_initial_test
uv run python -m eval.finance.run --task_name formula --mode offline \
  --save_path results/formula/seed0_exploratory/graph \
  --memory graph --generator_model qwen3.5:2b --think_generator off --num_ctx 32768 --seed 0 --skip_initial_test
```

Analysis:

```
uv run python compare_arms.py results/formula/seed0_exploratory/none \
  results/formula/seed0_exploratory/text results/formula/seed0_exploratory/graph --force
uv run python aggregate_run_costs.py results/formula/seed0_exploratory/none \
  results/formula/seed0_exploratory/text results/formula/seed0_exploratory/graph
uv run python check_learning_signal.py results/formula/seed0_exploratory/text \
  results/formula/seed0_exploratory/graph
uv run python replay_curator_sanitizer.py --root results
```
