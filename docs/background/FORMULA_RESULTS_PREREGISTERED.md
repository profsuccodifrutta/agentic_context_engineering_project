# Formula results — pre-registered runs

**Status: complete. All nine runs (3 arms × 3 replicates) finished. This is the final analysis.**

Headline, in one line: **memory helps; relational structure does not measurably beat linear
memory, and the design cannot detect a difference of the size observed.**

| question | answer | statistic |
|---|---|---|
| Does memory help at all? | **Yes for `graph`, marginal for `text`** | `none → graph` +8.00pp, p=0.0146; `none → text` +5.17pp, p=0.0922 |
| Is relational memory better than linear? | **Not shown** | `text → graph` +2.83pp, CI [−0.017, +0.078], p=0.2905 |

These are **not** the exploratory seed-0 runs, which are superseded and excluded for the reasons
recorded in `EXPERIMENT_FORMULA.md` §10.

---

## 1. Setup

| | |
|---|---|
| task | `formula` (`eval/finance`), 50 formulas |
| splits | `task_config_v2.json` — train 250 (5/formula), val 100 (stratified, 2/formula), test 200 (4/formula) |
| generator / reflector / curator | `gemma4:e2b` for all three roles |
| thinking | `--think_generator off`; reflector and curator at model default |
| `--num_ctx` | 32768 requested and 32768 served |
| replicates | `--seed 0`, `1`, `2` — **replicates, not seed variations**; see §3 |
| `curator_frequency` / `num_epochs` / `max_num_rounds` | 1 / 1 / 3 |
| `eval_steps` | 125 (two checkpoints) |
| grow-and-refine | on — `--dedup --dedup_threshold 0.88` |
| retrieval | `none` / `flat` / `graph`, `--retrieval_budget 2500 --retrieval_seeds 5` |
| commits | `1acfe69` (seed 0), `e6035e0` (seed 1), `fa3f2f7` (seed 2) |

**On the three commits.** `compare_arms.py` refuses the set without `--force` because `git_commit`
differs. Re-verified before forcing: `git diff --stat 1acfe69 fa3f2f7` touches only `*.md`,
`compare_arms.py` and `aggregate_run_costs.py`. **Zero bytes differ in `ace/`, `eval/`, `llm.py`,
`utils.py` or `playbook_utils.py`**, every commit in between is documentation or post-hoc
analysis. The nine runs executed identical code. Nothing else in the fairness audit differs.

## 2. Statistical approach

**The unit of analysis is the formula, not the row.** The test split is 50 formulas × 4
instances, and the four instances of one formula succeed and fail together, the model either
knows the formula or it does not. Treating 200 rows as 200 independent trials claims the
precision of 200 independent draws for a design that has about 50.

| what | used for | why |
|---|---|---|
| **Wilson 95% interval** | per-arm accuracy, §4 | reported for orientation only. It is marginal, unpaired and assumes iid rows. **Do not draw conclusions from it.** |
| **McNemar exact test (paired, two-sided)** | reported alongside, §5 | pairs the arms row by row, so it separates "the memory did nothing" from "the memory fixed 30 and broke 30". Its `b`/`c` counts are informative; **its p-value inherits the iid assumption and is anti-conservative here.** |
| **Cluster sign-flip permutation test** | **primary** | blocks on the formula and flips the sign of whole 4-row blocks. Makes no distributional assumption and respects the correlation. |
| **Cluster bootstrap 95% CI** | **primary** | resamples the 50 formulas with replacement, not the 200 rows. |

**Pooling across replicates ("seed-blocked").** Each row is scored as its *mean* correctness
across an arm's three replicates, a value in {0, ⅓, ⅔, 1} — and those real-valued rows go
through the same clustered test. This averages out run-to-run noise (§3). Procedure fixed
2026-08-12, before seed 2 ran.

Three things it does **not** do, all of which are easy to misreport:

- It does not add data. **The pooled analysis is 200 rows in 50 blocks**, scored over three
  replicates. It is never 600 rows.
- It does not narrow the formula-sampling variance, every replicate sees the same 50 formulas.
- McNemar is undefined on averaged scores, so it is deliberately absent from the pooled lines.

**Decision rule (pre-registered, `EXPERIMENT_FORMULA.md` §5).** Graph beats text only if the
clustered test is significant **and** the mean gap across replicates exceeds the within-arm
standard deviation. A gap smaller than the replicate spread is reported as noise regardless of p.
The same two-part standard is applied to the two secondary comparisons.

**Multiplicity.** Three pairwise comparisons, **no correction applied**; the Bonferroni threshold
would be α = 0.0167 and is quoted where it changes the reading.

**Machinery check.** `compare_arms.py --selftest` passes: singleton blocks reproduce McNemar
exactly, perfectly correlated blocks of *k* widen the interval by exactly √*k*, pooling identical
replicates changes nothing, pooling noisy replicates narrows the interval, and unequal replicate
counts are refused rather than silently pooled.

## 3. Why three replicates and why they are not "seeds"

`--seed` is **inert**. It reaches exactly one place, Ollama's sampler (`llm.py:215-216`), and at
`temperature=0.0` the sampler is greedy argmax and never consults it. It does not resample,
restratify, reorder or shuffle anything: the splits are fixed files and there is no
`random`/`shuffle`/`np.random` anywhere under `ace/`.

The runs diverge anyway, because GPU floating-point reduction order is not stable across batch
composition and layer placement, so a near-tied argmax flips. The three runs are therefore
**replicates of one configuration**, and their purpose is to measure **the noise floor**: how far
an arm moves when nothing meaningful changes. That is exactly the quantity the decision rule
compares the effect against, and it is undefined at n = 1.

**The floor, now measured at all three replicates** (rows moved between two runs of the same arm,
paired, out of 200):

| arm | 0↔1 | 0↔2 | 1↔2 | mean | accuracy sd |
|---|---|---|---|---|---|
| `none` | 1 | 1 | 0 | **0.7** | 0.0029 |
| `text` | 37 | 36 | 39 | **37.3** | 0.0301 |
| `graph` | 28 | 30 | 26 | **28.0** | 0.0231 |

The `none` arm carries no memory, so its prompt for a given row is identical across replicates and
its movement is raw decode noise: **1 row in 200**. Routed through the ACE loop the same noise
moves **37 rows**, one flipped token changes a reflection, then a curator operation, then the
playbook for every remaining step. **The loop amplifies serving nondeterminism ~37×.**

**This is the governing fact of the study.** Compare against the effect under test:

| | rows moved |
|---|---|
| `text` vs **itself** at another replicate | **37.3** |
| `text` vs `graph`, same replicate (44 / 31 / 26) | **33.7** |

**The between-arm difference is smaller than the within-arm noise.** The instrument cannot
resolve a difference of this size between two memory representations, and that was measured
rather than assumed.

Two consequences honoured throughout: replicates **cannot narrow** the CI (the bootstrap
resamples formulas, and all replicates see the same 50), and **sd/√3 is never used as a standard
error**. The floor is also an *underestimate*, because training order is held fixed.

## 4. Final test accuracy (n = 200 per run)

| arm | seed 0 | seed 1 | seed 2 | mean | sd | Wilson 95% on the mean † |
|---|---|---|---|---|---|---|
| `none` | 0.6250 | 0.6200 | 0.6200 | **0.6217** | 0.0029 | [0.553, 0.686] |
| `text` | 0.7050 | 0.6700 | 0.6450 | **0.6733** | 0.0301 | [0.606, 0.734] |
| `graph` | 0.7150 | 0.7150 | 0.6750 | **0.7017** | 0.0231 | [0.635, 0.761] |

† Computed at n = 200, **orientation only**. It is marginal, unpaired and assumes 200 independent
rows, which §2 explains this design does not have. The intervals overlap heavily and the
comparison that matters is the paired, clustered one in §5, do not read these as the result.

`no_answer` ≤ 1 in every run. The paper reports Base 67.5 → ACE 85.5 on this benchmark at 671B;
this is 62.2 → 67.3 at 2B.

Descriptively: **every one of the 6 memory runs scored above every one of the 3 control runs**
(lowest memory run 0.6450 > highest control 0.6250), and `graph` ≥ `text` in all three replicates.

## 5. The three comparisons

**Pooled across the three replicates, this is the pre-registered primary statistic.**
50 blocks, 200 rows, each row scored as its mean over 3 replicates.

| comparison | diff | cluster bootstrap 95% CI | sign-flip p | verdict |
|---|---|---|---|---|
| `none → text` | +0.0517 | [−0.0017, +0.1100] | 0.0922 | not significant |
| `none → graph` | **+0.0800** | **[+0.0200, +0.1450]** | **0.0146** | **significant** |
| **`text → graph`** | **+0.0283** | **[−0.0167, +0.0783]** | **0.2905** | **null** |

`none → graph` also clears the Bonferroni threshold (0.0146 < 0.0167). `none → text` does not
clear at either threshold.

**Per replicate**, so a reader can see pooling did not flatter the outcome:

| comparison | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `none → text` | +0.0800, p=0.0548 | +0.0500, p=0.1540 | +0.0250, p=0.5446 |
| `none → graph` | +0.0900, p=0.0364 | +0.0950, p=0.0211 | +0.0550, p=0.0938 |
| `text → graph` | +0.0100, p=0.9036 | +0.0450, p=0.2317 | +0.0300, p=0.4057 |

Direction is positive in **all nine** individual comparisons. Pooling tightened `none → graph`
from p ∈ [0.021, 0.094] to p = 0.0146, and tightened `none → text` from p ∈ [0.055, 0.545] to
p = 0.0922, the behaviour §2 predicts when run noise is averaged out.

**Where McNemar and the clustered test disagree, the clustered figure governs.** Example:
`none → text` at seed 0 alone is McNemar p=0.0195 against clustered p=0.0548. Reporting the McNemar
number would have turned a marginal result into a significant one by assuming independence the
data does not have.

### Applying the decision rule

| comparison | clustered significant? | mean gap > within-arm sd? | verdict |
|---|---|---|---|
| `none → text` | no (p=0.0922) | yes (5.17pp > 3.01pp) | **not established** |
| `none → graph` | yes (p=0.0146) | yes (8.00pp > 2.31pp) | **established** |
| `text → graph` | no (p=0.2905) | **no** (2.83pp < 3.01pp) | **noise** |

The primary comparison fails **both** halves of the rule.

### The trap to avoid

`graph` clears α against the control and `text` does not. **That is not evidence that `graph`
beats `text`.** A significant and a non-significant result do not make a significant difference,
the direct `text → graph` test is the one that answers that question, and it is null. The honest
reading is that the two memory arms sit on either side of the threshold with overlapping
intervals, and the design cannot separate them.

## 6. Structure is not inert, it is directionless

`text → graph` moves 44 / 31 / 26 rows across the three replicates while changing accuracy by
1–4.5pp. Blocked by formula:

| replicate | graph better | graph worse | tied |
|---|---|---|---|
| seed 0 | 11 | 12 | 27 |
| seed 1 | 12 | 5 | 33 |
| seed 2 | 12 | 7 | 31 |

Formulas are the test file's own labels (`compare_arms.load_clusters`), the same blocks the
clustered tests resample; the test file does not keep a formula's four rows together.

Structure changes a sixth of the answers and wins more formulas than it loses in two replicates
of three, losing one more than it wins in seed 0. **That is suggestive at most and cannot be made
significant here**: with three replicates a sign test's best possible two-sided p is 0.25. It is
reported as a direction worth testing at higher power, not as a result.

Where memory's real effect shows, per-formula solve counts (mean over replicates):

| arm | 0/4 | 1/4 | 2/4 | 3/4 | 4/4 |
|---|---|---|---|---|---|
| `none` | 8.7 | 4.3 | 11.0 | 6.0 | 20.0 |
| `text` | 6.0 | 4.0 | 8.3 | 12.7 | 19.0 |
| `graph` | 5.7 | 4.0 | 7.0 | 11.0 | 22.3 |

The control already solves 20 formulas completely and fails about 9 completely. Both memory arms
move formulas out of the unsolved and half-solved columns: 0/4 falls from 8.7 to 6.0 and 5.7, 2/4
from 11.0 to 8.3 and 7.0, and 3/4 roughly doubles, from 6.0 to 12.7 and 11.0. So memory does rescue
some formulas the control fails entirely. The arms part at the top: `text` leaves fully solved
formulas where the control has them (19.0 against 20.0), while `graph` adds about two (22.3). These
are averages of counts, not a paired per-formula comparison, so they describe where the gains sit
and carry no significance.

## 7. The mechanism was live, this is a negative result, not a broken run

`EXPERIMENT_FORMULA.md` §7 pre-registers the falsification: **graph ≈ text with `walk_only_share`
near zero means the arms read the same rules**, which is a broken run rather than evidence about
structure.

`walk_only_share` is the fraction of rendered rules the edge walk reached that relevance alone
would not have selected under the same budget. It is a counterfactual **inside the graph arm**,
what this arm rendered, against what a relevance-only ordering of *its own* rules would have
rendered under the same budget. It is not a comparison against the `text` arm, which holds
different rules under different ids from step 1, so no cross-arm set difference is defined.

Measured over the **200 final-test generations of each replicate**, the same calls that produced
the accuracies in §4, by `analyze_walk_share.py`:

| replicate | checkpoint tested | live rules | rendered per call | **`walk_only_share`** | generations the walk added nothing to | walk-added rules cited / relevance rules cited |
|---|---|---|---|---|---|---|
| seed 0 | step 125 | 81 | 19.2 | **0.1941** | 18/200 | 0.859 |
| seed 1 | step 250 | 105 | 17.6 | **0.0941** | 61/200 | 0.863 |
| seed 2 | step 125 | 73 | 19.8 | **0.2063** | 10/200 | 0.665 |

**9–21% of what the generator read on the test set was selected by following `related_to`
edges.** The arms genuinely read different things, and reading different things did not change
accuracy measurably. That is the finding.

The share is pooled, walk-only rules over all rules rendered, which is the definition the
harness uses. The mean of the per-generation shares is within 0.006 of it in every replicate
(0.1890 / 0.0905 / 0.2050), so nothing here depends on which of the two is quoted.

The last column is the citation rate of the walk-added rules divided by the citation rate of the
rules relevance would have selected anyway, both from `bullet_ids` in the same generations
(173/735 against 831/3031, 81/309 against 928/3055, 121/792 against 704/3066). Below 1 in every
replicate: the walk does not render noise, but it does not surface rules the generator prefers
over relevance's either.

### `walk_only_share` is a necessary condition, not a sufficient one

Added 2026-08-17. The pre-registered check treats this metric as the falsification hinge, and it
is a valid one **in the direction it was written**: near zero means the arms read the same rules
and the comparison says nothing. It does **not** mean a high value validates the edges.

Rewiring each replicate's graph at random, holding the live node set and the edge count fixed
(mean and sd over 10 random draws, scored through `GraphMemory.select_for_query`):

| replicate | live nodes | `related_to` edges | `related_to` | **random, same count** |
|---|---|---|---|---|
| seed 0 | 81 | 358 | 0.1941 | **0.4083** (sd 0.021) |
| seed 1 | 105 | 750 | 0.0941 | **0.3381** (sd 0.021) |
| seed 2 | 73 | 251 | 0.2063 | **0.4509** (sd 0.026) |

**Random edges score 2.1–3.6× higher than the real ones.** The metric rewards pointing *away*
from relevance, and arbitrary edges do that best, so it cannot distinguish a meaningful graph
from a scrambled one. Any run clearing the threshold has shown its mechanism fires, nothing
more.

The gap is itself a mechanism finding, and a sharper explanation for §5's null than power alone.
`related_to` is cosine similarity between rule *texts*, while retrieval scores rules by cosine
between the *question* and each rule. **These are close to the same signal**: if A is relevant and
B resembles A, B is probably also relevant, so the walk lands on rules the flat ranking would
have selected anyway. That is why the real graph surfaces *fewer* novel rules than random
rewiring. The treatment fired, but the edges it followed were largely redundant with the scoring
that chose the seeds, which is a statement about *this* graph, not about relational memory in
general.

> **Corrected 2026-08-17.** This table previously reported 0.1504 / 0.1090 / 0.1755, taken from
> `structural_stats()` at the tested checkpoint, and described them as holding "on essentially
> every generation". Those are **run-cumulative** figures: the counters at `ace/memory/graph.py:155-157`
> are never reset, so each is an average over every retrieval served since step 1, including
> the early steps where the playbook still fitted inside `--retrieval_budget` and the walk
> *could not* add anything — and `structural_stats()` is recorded at eval checkpoints only
> (`ace/ace.py:987`, `:1288`), so the final test contributed nothing to the number quoted for it.
> The figures above replace them and are measured on the test generations directly.
> **No conclusion changes**: the pre-registered condition is a floor, and every replicate clears
> it on either measurement. Note the correction is not one-directional, dilution pushes the
> cumulative figure down within a run, but the two quantities cover different question sets
> (train versus test), so seed 1 moved down while seeds 0 and 2 moved up. That is the reason to
> measure rather than adjust. The per-generation zero counts are new and worth keeping: seed 1
> got nothing from the walk on 30% of its generations and still ran at 0.0941 overall, so a
> substantial zero rate is compatible with a live mechanism.

### The graph arm reached that accuracy from ~40% fewer rules

`--retrieval_budget 2500` covers the **entire** generator-facing rendering, legend and
relation lines included, not just rule text. That is deliberate,
budgeting only rule text would hand the graph arm its annotations for free and reinstate the
+33% rendering asymmetry retrieval exists to remove. The structure has to pay for itself out of
the same allowance.

It does pay. Counting the rules actually present in the generator prompt, over the 200
final-test generations of each run, the same calls that produced the accuracies in §4:

| replicate | `text` rules rendered | `graph` rules rendered | `text` distinct | `graph` distinct |
|---|---|---|---|---|
| seed 0 | 30.2 | **19.2** | 26.9 | 16.8 |
| seed 1 | 31.9 | **17.6** | 26.2 | 15.9 |
| seed 2 | 30.2 | **19.8** | 28.0 | 16.6 |
| **mean** | **30.8** | **18.9** (−39%) | 27.0 | 16.4 |

("Distinct" discounts near-duplicates pairs of rendered rules at lexical Jaccard ≥ 0.6, so
that redundancy is not counted as content. It is a coarse measure and the two arms carry
similar duplicate fractions, 15–36% vs 20–33%, so it does not separate them.)

**Under an identical token budget the graph arm reads about 40% fewer rules than the flat arm,
and its accuracy is not lower**, nominally +2.83pp, which §5 establishes is null. The relation
lines are not free, and they consume roughly two rules' worth of budget for every three the
flat arm shows.

This is an efficiency observation, not a significance claim, and it must not be read as
evidence that structure helps: the honest statement is that **the graph arm gets at least the
same accuracy out of a substantially smaller set of rules**. It is worth recording for two
reasons. It is the clearest positive thing this experiment can say about relational structure,
whatever the walk selects is carrying more per token than relevance ranking alone. And it
sharpens the null in §5: the arms did not tie because the treatment was diluted into a large
common rule set, they tied while reading sets of very different size.

It also bounds a competing explanation. The memory arms lose generator context to their own
playbook (a 75-rule playbook is 43% of the served ~16,384-token window), so
"the graph lost because it saw less" and "the graph won because it saw less noise" are both
available a priori. Neither survives: both arms were capped at the same 2,500 tokens, the arm
reading fewer rules did not do worse, and the arm reading more did not do better.

## 8. Checkpoint selection, and loop health

**Both memory arms selected the same checkpoint in every replicate**, so the concern that arms
might be compared at different playbooks does not arise anywhere in this set:

| replicate | `text` | `graph` | both tested |
|---|---|---|---|
| seed 0 | 125 (val 0.7400) | 125 (val 0.7400, tie → earlier) | step 125 |
| seed 1 | 250 (val 0.7700) | 250 (val 0.7700) | step 250 |
| seed 2 | 125 (val 0.7700) | 125 (val 0.7400, tie → earlier) | step 125 |

`check_learning_signal.py` checks [1]–[5] pass on all six memory runs: `harmful` tags emitted
(94–159 per run), `refines` present in 248–250 of 250 curator calls with 50–126 non-null,
train accuracy inside the 0.20–0.85 band, and generator output 0–1/~1015 unparseable.

**Check [6], curator hygiene, fails on the `text` arm in two of three replicates.** This is the
one real asymmetry in the set:

| replicate | `text` ADDs adding nothing new | `graph` |
|---|---|---|
| seed 0 | 141/404 = **34.9%** | 154/452 = 34.1% |
| seed 1 | 375/695 = **54.0%** ✗ | 97/380 = 25.5% |
| seed 2 | 314/623 = **50.4%** ✗ | 169/448 = 37.7% |

Over 50% means the curator has largely stopped producing new knowledge. The `text` curator also
emitted far more ADDs (mean 574 vs 427) and pasted rendered-bullet prefixes more often. Two
qualifications, both in the arm's favour and both stated: the run itself was protected,
`sanitize_curator_operations` drops these before they land and the graph arm's figure is an
**upper bound** by construction, since a superseded rule is withdrawn from its curator's view and
may legitimately be re-added.

It was not pre-registered and is reported descriptively. **It was checked as a competing
explanation for §5 and does not survive.** For curator redundancy to cost the `text` arm
accuracy it would have to crowd distinct rules out of what the generator reads, and §7 measures
that directly: the near-duplicate fraction of the rendered slice is comparable in both arms, and
the `text` arm still reads *more* distinct rules than the `graph` arm (27.0 vs 16.4). The
redundancy is confined to the curator's unbudgeted view, which retrieval filters before the
generator sees it. Within the `text` arm across the three replicates the duplicate rate also
fails to track accuracy (34.9% → 0.7050, 54.0% → 0.6700, 50.4% → 0.6450, not monotone).
It remains an unexplained arm asymmetry worth reporting; it is not a mechanism for the §5 null.

One curator call in the seed-1 graph arm (step 105 of 250) returned truncated JSON and was
skipped, 1/250, logged in `curator_failures.txt`.

**The §8 headroom warning did not bite.** Peak curator prompt, server tokens, against the
~31,000 practical ceiling:

| replicate | `text` peak | `graph` peak | graph % of ceiling |
|---|---|---|---|
| seed 0 | 13,731 | 25,665 | 83% |
| seed 1 | 13,557 | 23,254 | 75% |
| seed 2 | 15,092 | 24,665 | 80% |

Zero empty curator responses in any run, the silent-overflow failure mode never fired.

## 9. Cost

Tokens only. **Do not quote per-arm LLM hours**: serving throughput swung 36.3× (seed-0 graph),
18.8× (seed-1 text) and 6.2× (seed-1 graph) mid-run. Tokens and accuracy are unaffected; timing
is not comparable.

| arm | total tokens (mean) | vs `text` | mean accuracy |
|---|---|---|---|
| `none` | 201,114 | −97.3% | 0.6217 |
| `text` | 7,394,721 | baseline | 0.6733 |
| `graph` | 8,824,057 | **+19.3%** | 0.7017 |

Per replicate the graph overhead is +24.9% / +20.1% / +13.2%.

**Structure costs +19.3% tokens for +2.83pp accuracy (p=0.29).** The split is the informative
part, the overhead is almost entirely in the curator, which is unbudgeted by design
(`ace/ace.py:769`: a curator that sees a retrieved slice cannot tell whether a rule already
exists). At the tested checkpoint the same rules render ~40–50% longer as a graph than as a flat
list (seed 0: 81 rules = 6,604 tokens flat vs 10,389 as a graph). The generator side is held
equal by the shared 2,500-token retrieval budget, so the graph arm shows *fewer* rules for the
same tokens and pays for its own annotations.

Whole experiment: ~49.3M tokens and ~50 LLM-hours across nine runs.

## 10. Criticism

1. **The primary comparison is under-powered by a factor of ~3.5, and this was measured** (§3).
   `text → graph` = +2.83pp against a within-arm noise floor of 3.01pp. **The null is a power
   limit, not evidence of equivalence.** Nothing here licenses "structure does not help"; it
   licenses "structure was not shown to help by an instrument that could not have shown it."
2. **The `text` curator failed hygiene check [6] at two of three replicates** (§8). The arms
   therefore differ in a second place besides the independent variable at those replicates.
3. **Three pairwise tests, no correction.** Stated, not corrected. Only `none → graph` survives
   Bonferroni.
4. **The `text` arm declines monotonically across replicates** (0.7050 → 0.6700 → 0.6450) while
   `graph` does not (0.7150 → 0.7150 → 0.6750). With three points this is not distinguishable
   from noise at sd 3.01pp, and no ordering exists among replicates for it to trend along, but
   it coincides with the hygiene failures and is worth a look at higher R.
5. **Timing is unusable** (§9), so any efficiency claim rests on tokens alone.
6. **`contradicts` edges are off**, so "graph" here means `related_to` only. The treatment is
   weaker than the full design.
7. **`num_epochs=1` against the paper's 5**, a disclosed deviation, and the one most likely to
   under-train both memory arms relative to the published method.
8. **2B models throughout.** The paper's result is at 671B. Roughly half of all curator ADDs
   still arrive with a pasted rendered-bullet prefix (35–65%); the sanitiser removes them, but
   the underlying curator competence is the known weak point of small models here.
9. **Replicates are not independent samples of the task**  training order is fixed, so the
   measured noise floor is a lower bound and the CI cannot tighten with more replicates.
10. **The graph's only retrieval-facing structure is one similarity-derived edge type, and it is
    largely redundant with the query scoring.** `refines` withdraws a node from rendering and
    needs no graph; `contradicts` is off. So the treatment reduces to `related_to`, which is
    cosine between rule texts, nearly the same signal as the cosine between question and rule
    that selects the seeds. Measured consequence (§7): random rewiring at matched density
    surfaces 2.1–3.6× *more* novel rules than the real edges do. **The null is therefore
    evidence about similarity-derived graphs specifically**, and a graph built on a relation that
    is not recoverable from rule text remains untested.

Resolved from the seed-0 draft: the extraction-path asymmetry no longer looks systematic. Share
of answers not read by a strict parse spans 41.5%–53.5% across the nine runs, but the *ordering*
of the arms flips between replicates (`text` is the highest at seed 0 and the lowest at seed 2),
which is what a nondeterminism artefact looks like rather than an arm effect.

## 11. Conclusion

Across nine runs with a healthy loop, a correctly-configured baseline, matched checkpoints in
every replicate, and the retrieval mechanism demonstrably firing on 9–21% of the rules the
generator read on the test set:

- **Memory helps.** +8.0pp for `graph` over the no-memory control (p=0.0146, CI excluding zero,
  surviving Bonferroni) and +5.2pp for `text` (p=0.0922, marginal). All six memory runs beat all
  three control runs. This is the solid result.
- **Relational memory is not shown to beat linear memory.** +2.83pp, CI [−0.017, +0.078],
  p=0.2905, for +19.3% tokens. It fails both halves of the pre-registered decision rule.
- **The null is a power limit, not equivalence.** The gap is smaller than the noise the ACE loop
  generates against itself between replicates. `graph` leads `text` in all three replicates and
  wins more formulas than it loses in two of them, a direction that this design is structurally
  unable to confirm.

What would settle it: more replicates (which buy power against run noise but not against
formula-sampling variance), a larger or multi-task test set (which buys blocks), a model whose
curator does not need sanitising, or `contradicts` edges via an LLM judge so the treatment is the
full design rather than `related_to` alone.

## Reproducing

```
# per seed S in {0,1,2}
uv run python -m eval.finance.run --task_name formula --mode eval_only \
  --save_path results/formula/preregistered/seed$S/none --memory none --retrieval none \
  --config_path ./eval/finance/data/task_config_v2.json \
  --generator_model gemma4:e2b --reflector_model gemma4:e2b --curator_model gemma4:e2b \
  --think_generator off --num_ctx 32768 --seed $S \
  --dedup --dedup_threshold 0.88 --eval_steps 125 --retrieval_budget 2500 --retrieval_seeds 5

uv run python -m eval.finance.run --task_name formula --mode offline \
  --save_path results/formula/preregistered/seed$S/text --memory text --retrieval flat \
  ... same flags ... --skip_initial_test

uv run python -m eval.finance.run --task_name formula --mode offline \
  --save_path results/formula/preregistered/seed$S/graph --memory graph --retrieval graph \
  ... same flags ... --skip_initial_test
```

Analysis (`--force` is required and licensed by the empty code diff in §1):

```
uv run python compare_arms.py --selftest
uv run python compare_arms.py --force results/formula/preregistered/seed*/*/ace_run_*
uv run python aggregate_run_costs.py results/formula/preregistered/seed*/*/ace_run_*
uv run python check_learning_signal.py results/formula/preregistered/seed*/{text,graph}/ace_run_*
```

`aggregate_run_costs.py` will report "seed differs across arms" on the nine-run set. That is the
audit assuming one triple; seed is the deliberately varied replicate index here, and the arms are
matched on every pinned field within each replicate.
