# Experiment pre-registration: ranked co-citation edges on `finer` and `formula`

**Written 2026-09-12, before any run of this set and before Stage A produced a single number.**
The offline measurements quoted below all come from *finished, committed* runs of the previous
experiments. Nothing here was chosen after seeing a test number from this one.

> **Status: nothing has been run. Stage A is not yet executed.** The go/no-go in section 9 has not
> fired in either direction.

---

## 0. Status: this is the third attempt, and it is exploratory

Three comparisons have been completed. All three are null:

| experiment | primary | result |
|---|---|---|
| `formula`, similarity edges | `text -> graph` | +2.83pp, CI [-0.017, +0.078], p=0.2905 |
| `formula`, co-citation edges | `text -> cocite` | +2.17pp, CI [-0.020, +0.060], p=0.3344 |
| `finer`, co-citation edges | `text -> cocite` | **-2.02pp, p=0.0601** |


The difference from the previous two attempts is that this one aims at a defect that has been
measured, and the fix is verified offline against saved artifacts before any compute is spent. The defect is stated in section 1, the fix in section 2, and the
offline evidence that the fix works is in section 3.

## 1. The defect: the graph can only reach an eighth of the memory

`docs/FINER_RESULTS.md` section 5 reports it and `docs/COCITE_RESULTS.md` section 5 shows the same
shape on `formula`. At the tested checkpoint:

| run | live rules | connected set | share | max degree |
|---|---|---|---|---|
| `finer` seed 0 | 224 | **30** | 13.4% | 15 |
| `finer` seed 1 | 226 | **31** | 13.7% | 13 |
| `finer` seed 2 | 226 | **31** | 13.7% | 13 |

**Seven rules in eight carry no edge at all.** The walk cannot reach them, so for those rules the
treatment arm is exactly the baseline arm: relevance-ranked flat retrieval. A mechanism that is
inert on 87% of the memory cannot be expected to move an aggregate.

Two causes. Both are measured, and both belong to the edge predicate, so neither is a property
of the task.

**The gate is a moving target.** Edges are not stored. `graph.py::_cocite_edge_pairs` recomputes
them from the counters on every read, so a pair whose co-render denominator grows while its
co-citation count does not falls below the rate gate with nothing deleted and no record made.
Measured over every rule that left the connected set between reported checkpoints on `finer`:
**0 were removed by de-duplication, 100% stayed live and had a grown denominator.** The connected
set therefore cannot grow with the playbook.

**A global rate ignores how much evidence a pair has.** Under `cocited / corendered >= 0.10`, a
pair seen 3 times and co-cited once (0.33) passes, while a pair seen 300 times and co-cited 25
times (0.083) fails. The gate systematically prefers the thinly observed pair. This is the same
metric mismatch recorded in `docs/EXPERIMENT_COCITE.md` section 2 in its confidence-against-lift
form, now in its evidence-depth form.

**What this experiment does not attack.** `FINER_RESULTS.md` section 6 reports a second bottleneck,
*tag* coverage: the playbook holds a rule for 84 of 137 gold tags, worth +21.0pp where a rule
exists and -17.1pp where none does. That is a property of what the curator wrote, and **no change
to the graph, the walk or the retrieval can create a rule for a tag the curator never addressed.**
This experiment targets **edge coverage only**.

## 2. The redesign

Five changes. Each is pointed at a measured defect and each is stated so that it can be turned off
independently.

### 2.1 Edge creation: rank, do not threshold

Both gates are deleted from the run path. For every rule, connect it to the **top `k` rules it
scores highest against**, unioned symmetrically:

```
    weight(a, b) = reward_sum(a, b) / (corendered(a, b) + beta)
    edges        = { (n, o) : o in top-k of n by weight, for every live n }
```

with **`beta = 10`** and **`k = 2`**, both fixed here from the offline measurements in section 3.

Three properties follow by construction, and they are the reason this form was chosen:

- **The moving target is gone.** A rank is relative. A growing denominator can only cost a pair its
  edge if another pair genuinely overtakes it. That is a real change in the evidence, where the
  old gate could drop an edge because the memory had simply got bigger.
- **Evidence buys score.** `beta` shrinks a rate toward zero in proportion to how little has been
  observed. It sits in the denominator, where an `alpha` in the numerator would credit pairs
  nothing has been observed about.
- **Degree is bounded.** Each node contributes `k` edges, so no rule can become a hub by being
  frequently rendered.

### 2.2 The reward counter, and outcome weighting

A **third counter** is added per pair. `cocited` and `corendered` keep their current meaning and
their current place in the sidecar; `reward_sum` accumulates the generation's score:

```
    corendered += 1                       for every co-rendered pair
    cocited    += 1                       when both rules are cited
    reward_sum += reward in [0, 1]        when both rules are cited
```

The reward is the generation's own accuracy: **fractional on `finer`** (`partial_credit`, so 3 of 4
tags adds 0.75) and **binary on `formula`**, whose `answer_is_correct` is a bool. Section 3.5 is the
measurement that says binary is safe there, and section 8 discloses the asymmetry.

The claim being tested is that **co-citation measures usage while outcome-weighted co-citation
measures success.** A pair co-cited 50 times on 45 wrong answers is evidence against that pair.

### 2.3 The walk: best-first over raw path strength

The breadth-first walk is replaced by a priority queue ordered on **cumulative raw path strength**:

```
    strength(seed)  = relevance(seed)
    strength(child) = strength(parent) * weight(parent, child)
```

Breadth-first order renders every hop-1 neighbour before any hop-2 one, so a 0.0071 edge is
rendered ahead of a 0.9 edge one hop further out. Raw products fix that, and the depth penalty
they impose falls out of the evidence itself, with no constant to tune.

**The products stay raw.** Normalising each node's best edge to 1.0 makes depth free along the
locally-best chain, so the walk runs five hops out on weak edges. Section 3.3 measures both.

**Strict precedence is kept.** Relevance is a cosine in roughly [0.2, 0.6]; path strength after one
hop is around 0.018. On one scale every unwalked rule outranks every walked one and the treatment
is erased. The emission order is therefore unchanged in shape from the shipped arm:

1. the relevance-chosen seeds,
2. the walk-reached rules, ordered among themselves by path strength,
3. the relevance tail, if the budget is not yet full.

This also keeps `walk_only_share` meaning exactly what it has always meant, so section 7's
falsification hinge and `analyze_walk_share.py` continue to apply without modification.

### 2.4 No relation lines

**The graph arm stops printing `co_cited:` lines entirely.** The walk uses the edges; the generator
never sees them.

The shipped arm carries two channels at once. Selection (which rules are rendered) is the
hypothesis. Annotation (a printed line saying these rules go together) is an
instruction to the model, and it is a second treatment. With both on, no result is attributable to
either. Section 3.4 measures what the annotation costs at the new edge density: **988 tokens, 39.5%
of the 2,500 budget.**

Removing it means both arms spend the full budget on rule text, render the same *kind* of object,
and differ in exactly one thing: which rules were selected. This is a deliberate departure from the
`formula` and `finer` designs.

### 2.5 Seeding: every sub-query is represented

Walk seeds are allocated **`ceil(retrieval_seeds / n_sub_queries)` per sub-query** rather than as a
global top-`k`. On `finer` that is 2 each, 8 total; on `formula`, which supplies one query, it is
`ceil(5/1) = 5`.

The defect it fixes is measured: under global seeding, all four `finer` sub-questions are
represented in **1.6% of generations**.

**The same allocation goes into the `text` arm.** Flat retrieval ranks globally too, by max over
sub-queries, so it starves sub-questions in exactly the same way. `TextPlaybookMemory` fills its
budget round-robin across the sub-queries' ranked lists. Leaving it out would make seed allocation
a second difference between the arms and void the comparison. **The baseline is made stronger, on
purpose, before the treatment is tested against it.**

### 2.6 What is deliberately not changed

- **`--dedup 0.86` stays on in both arms.** It is symmetric by construction: one decision rule
  (`refine.py::duplicate_groups`) applied by both memories, with `check_finance_wiring.py` [1e]
  asserting both arms reach the same visible rule set from the same operations. Turning it off
  would nearly double the memory: measured on
  `finer` seed 0, 224 live rules at 8,791 tokens become **422 rules at 16,634 tokens (1.89x)**
  against a curator prompt that already reads 11,252 at the last step and a measured ceiling of
  26,860.
- **`--no_curator_refines` is set in both arms on both tasks.** This *is* a real asymmetry and it is
  the one worth removing: the curator's `refines` field is a graph-only capability, and the text arm
  parses it and drops it. On `finer` it was already disabled. On the `formula` co-citation runs it
  fired **24 / 31 / 42 times**, so those runs carried it. Disabling it everywhere leaves the arms
  differing in the pairwise relation alone. If the capability is interesting, it is an ablation
  after the main runs.


## 3. The offline evidence, measured on committed runs before this was written

All figures are from `results/finer/final/` and `results/formula/cocite_experiment/`, read-only, no
LLM calls. They are what the redesign rests on, and they are reported here so the configuration
cannot later be said to have come from Stage A's output.

### 3.1 The predicate delivers the edge coverage

`finer` seed 0, 224 live rules:

| predicate | edges | connected | share | max degree | components |
|---|---|---|---|---|---|
| shipped gate (0.10) | 63 | 30 | 13.4% | 15 | 12 |
| **`c/(r+10)`, top-2** | 198 | **116** | **51.8%** | 21 | 1 |
| `c/(r+10)`, top-3 | 286 | 116 | 51.8% | 30 | 1 |
| `c/(r+10)`, top-5 | 431 | 116 | 51.8% | 35 | 1 |
| `(c+1)/(r+20)`, top-2 | 337 | 183 | 81.7% | 25 | 1 |

The reachable pool grows by 3.9x. Two readings fixed two constants:

- **`k` buys nothing but tokens.** The connected set is 116 at `k` = 2, 3 and 5 alike; only degree
  and relation-line cost move. `k = 2` is therefore the selected choice.
- **The 183 reached by `(c+1)/(r+20)` is an artifact of the `alpha`.** It connects pairs with no
  co-citation evidence at all, by giving them a prior. A rule never co-cited with anything has no
  pairwise evidence and should not be reachable by a pairwise relation, so `alpha` is dropped.

### 3.2 `beta = 10`

| form | 1/3 | 25/300 | evidence-rich pair wins? |
|---|---|---|---|
| `c/r` (shipped) | 0.3333 | 0.0833 | no |
| `c/(r+5)` | 0.1250 | 0.0820 | no |
| **`c/(r+10)`** | 0.0769 | **0.0806** | **yes** |
| `c/(r+20)` | 0.0435 | 0.0781 | yes |
| Wilson LCB, z=1 | 0.1356 | 0.0687 | no |
| Wilson LCB, z=1.96 | 0.0615 | 0.0571 | no |

`beta = 10` is chosen as the smallest value that inverts the pathological ordering, so the shrinkage
is no stronger than the requirement demands.

### 3.3 Best-first over raw products corrects breadth-first without replacing it

200 random seed draws, first 50 rules rendered, `finer` seed 0. Edge weights under `c/(r+10)` run
from 0.0071 to 0.3846, median 0.0461, so a strongest hop-2 path (0.385 squared = 0.148) beats the
weakest hop-1 edge (0.0071) by a factor of 20:

| ordering | set overlap with BFS | hop mix, h0/h1/h2/h3/h4/h5 |
|---|---|---|
| breadth-first (shipped) | 1.000 | 16 / 26 / 51 / 6 / 1 / 0 |
| **raw product** | **0.805** | 16 / 25 / 46 / 11 / 2 / 0 |
| per-node normalised | 0.646 | 16 / 15 / 22 / 21 / 13 / 7 |

Raw products change one rendered rule in five and go slightly deeper where the evidence justifies
it. Per-node normalisation goes much deeper (41% of renders at hop 3 or beyond, against 7% for
breadth-first) because it erases the absolute evidence scale the predicate was built to establish.
It is a different treatment and is not shipped.

### 3.4 Relation lines cost 39.5% of the budget at the new density

Induced subgraph over 200 walk-selected sets of 50 rules. `_render` already drops relations pointing
at unrendered rules, so this is the real charge:

| predicate | relation ids printed | tokens | share of 2,500 |
|---|---|---|---|
| shipped gate | 5.7 | 43 | 1.7% |
| **`c/(r+10)`, k=2** | **151.5** | **988** | **39.5%** |
| `c/(r+10)`, k=3 | 218.0 | 1,321 | 52.8% |

A walk-selected set is induced-dense by construction: it got there by following edges. A random
50-rule sample charges only 153 tokens, and quoting that number would understate the cost by 6.5x.
The 39.5 / 60.5 split reproduces the shape of the defect-invalidated seed-0 arms (53.6% relation
lines, 39.4% rule text), which is the second reason the lines are dropped.

### 3.5 Outcome weighting: fractional is required on `finer`, binary is safe on `formula`

Pairs co-cited at least 5 times over the logged training generations:

| task | reward | pairs | severed (`reward_sum = 0`) | conditional reward |
|---|---|---|---|---|
| `finer` seed 0 | all-four-exact (binary) | 90 | **53 (59%)** | median 0.000 |
| `finer` seed 1/2 | all-four-exact (binary) | 162 | **99 (61%)** | median 0.000 |
| `finer` seed 0 | **per-tag fractional** | 90 | **1 (1%)** | 0.000 to 0.475, CV 0.32 |
| `finer` seed 1/2 | **per-tag fractional** | 162 | **0** | 0.083 to 0.700, CV 0.32 |

`finer`'s all-four-exact base rate is 0.222, so binary weighting would sever three edges in five and
collapse the graph precisely where it is most needed. Fractional reward severs almost nothing and
still re-ranks. Weak pairs get out-competed instead of deleted, which is the behaviour wanted.

`formula` has no partial-credit metric, so its reward is binary. Measured on its three co-citation
runs, base rate 0.837 to 0.858:

| co-citation depth | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| pairs co-cited >= 1, severed | 14.6% | 18.6% | 21.4% |
| pairs co-cited >= 3, severed | **3.2%** | **2.0%** | **2.5%** |
| pairs co-cited >= 5, severed | 3.3% | 0.0% | 0.4% |
| share of severed pairs seen exactly once | **83%** | **95%** | **86%** |

The severance is concentrated almost entirely in pairs observed a single time on a single wrong
answer, which is the noise top-k is meant to discard. Evidence-rich pairs survive binary weighting
intact. **`formula` therefore ships binary and no partial-credit metric is built for it**.

### 3.6 `--cocite_min_opportunity` is already inert under ranking

| min_opportunity | connected | edges |
|---|---|---|
| 0 | 116 | 198 |
| 1 | 116 | 198 |
| 3 (shipped) | 115 | 197 |
| 5 | 115 | 197 |

Top-k already prunes thin pairs, and `beta` crushes what survives: a once-seen pair scores at most
1/11. The flag changes one node and one edge across its whole range and is **deleted from the run
path**. It stays readable in the analysis path so the committed sidecars still parse.

### 3.7 Where the two arms currently stand

Per-tag accuracy on the 441 `finer` test rows, recomputed through `finer_tag_counts`:

| arm | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| `text` | 0.4450 | 0.4439 | 0.4461 | **0.4450** | 0.0011 |
| `cocite` | 0.4150 | 0.4274 | 0.4320 | **0.4248** | 0.0088 |

The treatment is **2.02pp behind**. Section 6's futility gate is set against this baseline, and it
asks the new predicate for a **+3.6pp move from where the arm sits today** merely to reach the bar.
Carry the caveat from `FINER_RESULTS.md`: `finer` seeds 1 and 2 trained byte-identically, so these
sds understate the full loop noise.

## 4. Design

Two arms per task, three replicates each, plus the existing controls.

| arm | flags | what it answers |
|---|---|---|
| `text` | `--memory text --retrieval flat` | the ACE baseline, with round-robin seeding |
| `cocite_rank` | `--memory graph --retrieval graph --cocite --cocite_predicate rank` | does ranked, outcome-weighted structure help? |

**`text` is re-run, not reused.** Section 2.5 changes its seed allocation, so the committed runs are
a different configuration. They also differ on `git_commit`, which `compare_arms.py` refuses to
compare across without `--force`.

**The `none` control is reused.** It is deterministic on `finer` (five byte-identical runs across
three commits and two prompt sets) and nothing in this redesign reaches it: `--memory none` refuses
retrieval, so neither seeding nor the predicate touches it.

**The ablation, deferred to stage C:** `--memory graph --retrieval flat --cocite_predicate rank`.
Same graph, same rendering, same prompt length, relevance-ordered instead of edge-ordered. It is
what separates "edges choose what is read" from "the treatment arm reads fewer rules", and it is
the control for prompt length.


## 5. Outcomes, declared in advance

**Primary: `text -> cocite_rank` on `finer`.** Per-tag partial credit
(`compare_arms.py --metric finer_per_tag`), replicate-pooled through `pooled_arm_scores`, clustered
sign-flip permutation test with the cluster bootstrap CI, blocking on the row. This is the only
comparison declared primary.

**Secondary: `none -> cocite_rank` and `none -> text`.** Both re-ask a question already answered
positively (+8.84pp and +6.82pp, Holm < 0.0002). Reported as replication checks, not as new results.

**Conditional, stage D: the same primary on `formula`.** Run only if the `finer` primary clears the
futility gate and the stage C ablation is read. Section 10 is why this is the only stage capable of
producing a confirmatory claim, and why even it is weakened by everything that precedes it.

**Declared now, before any run:** the multiplicity family for this experiment is the two comparisons
above plus `formula`'s primary if stage D happens, and Holm is applied within it. It is **not**
pooled with the earlier experiments' families. 

## 6. Power and the futility gate, set from the data

The single-seed paired difference on `finer`, measured on the committed seed-0 pair: n = 441 rows,
sd of the per-row difference 0.330, **standard error 1.57pp**, 95% half-width 3.08pp.

A single seed is therefore a noisy instrument, and the gate has to be chosen knowing that:

| bar | kills a true +4pp effect | passes a true null |
|---|---|---|
| +4.4pp (80% power point) | **50%** | 0.3% |
| +3.1pp (95% bound) | 28% | 2.4% |
| **+1.6pp (1 SE)** | **6%** | **15%** |
| +0.0pp | 0.5% | 50% |

**The gate is asymmetric on purpose.** Killing a real effect ends the project; continuing on a null
costs about 14 hours of compute. A bar at +4pp kills a genuine +4pp effect on a coin flip.

> **Futility gate, fixed here: after seed 0 of `text` and `cocite_rank` on `finer`, stop if the
> point estimate of `text -> cocite_rank` is below +1.6pp.** No accuracy-dependent configuration
> change is permitted at that point. The remaining seeds and the `formula` runs are authorised only
> on a pass.

**What clearing the gate buys.** Clearing +1.6pp on one seed licenses more compute and nothing
else. The claim still rests on the three-replicate pooled analysis in section 5.

## 7. What would make a run mechanism-inconclusive

Read from `structural_stats()` and `analyze_walk_share.py` **before quoting any accuracy**. Any of
these means the treatment was not delivered and the run is inconclusive rather than negative. 

1. **Connected set below 40% of live rules.** Measured offline at 51.8%, so this fails only if the
   live graph behaves unlike the saved one.
2. **Max degree above 25% of the connected set.** Offline that is 21 of 116, or 18%. The bar is
   relative because the walk is no longer trapped by a hub the way a breadth-first walk was: raw
   path strength makes a hub's neighbours compete on evidence. It stays in as a diagnostic.
3. **Held-out co-citation lift at or below 1.0x.** Edges built on the first half of a run's training
   generations, scored on the unseen second half. If the edges do not predict co-citation at all,
   they are not evidence about pairs.
4. **`walk_only_share` below 0.05**, pooled over the final test generations via
   `analyze_walk_share.py`, never the run-cumulative counter in `structural_stats()`. Near zero
   means the arms read the same rules.
5. **Fewer than 20 distinct rendered rule sets over the first 25 test generations**, in every arm
   using retrieval. This is `check_learning_signal.py` [7].

**These are necessary conditions and nothing more.** Random rewiring at matched density scores 2.3 to 3.6x higher on
`walk_only_share` than a real similarity graph, so clearing these bars shows the arms differ; it
does not validate the edges. Gate 3 is what validates the edges.

## 8. Disclosures

Things this design does that a reader should be told without having to find them.

**The post-curate generation's reward is inflated, and it is counted anyway.**
`record_generation` fires twice per training step: once pre-train, once after the curator has
written rules from that same sample's failure. Measured on the three `finer` co-citation runs:

| | pre-train reward | post-curate reward | gap |
|---|---|---|---|
| seed 0 | 0.4012 | 0.5075 | **+10.6pp** |
| seed 1 | 0.4587 | 0.5463 | **+8.8pp** |
| seed 2 | 0.4587 | 0.5463 | **+8.8pp** |

The inflation is pair-specific: it accrues to pairs involving freshly written rules. Both sites are
kept, because `record_generation` being called identically in every arm from the same two places is
a fairness invariant, and because halving the pairwise evidence would undo the reachability this
redesign exists to buy. The expected behaviour is that the `corendered` denominator corrects an
over-rewarded new rule over subsequent pre-train generations. **That expectation is a hypothesis
and has not been verified.**

**Reward inherits through `refines`, and the store keeps no marker.**
`_inherit_cocitation` copies an ancestor's pair history onto its heir at full weight, for both the
`curator` and `dedup` origins, and it now copies `reward_sum` with `cocited` and `corendered`.
Inheriting only two of the three would divide a zero reward by a large historical denominator and
obliterate the heir. The cost is that every figure built on these counters, circularity included,
is an **upper bound**.

**Circularity is unchanged and still applies.** A pair can only be co-cited if it was co-rendered,
and co-rendering is driven by the edges, so the evidence for an edge is partly produced by the edge.
The exploration floor is question variety and the relevance-chosen seeds, which the edges cannot
suppress. No exploration mechanism is added in this version. Ranking makes the feedback loop
tighter than the gate did, because best-first order preferentially re-renders strong edges; the
counter-pressure is that `corendered` grows for a rule rendered into contexts where it is not cited,
which crushes its weight. **This too is a hypothesis.** Stage A metric 3 (held-out lift) is
the closest available check and it is a pre-run one.

## 9. Stage A: nine offline checks, and what fails the experiment before it starts

All read-only, no LLM, run against the committed `finer` and `formula` runs. **Declared here with
their bars before any of them is executed.**

| # | claim | bar | blocking? |
|---|---|---|---|
| 1 | the predicate reaches the memory | connected >= 40% of live | **yes** |
| 2 | no hub | max degree <= 25% of connected | **yes** |
| 3 | edges predict held-out co-citation | lift > 1.0x on the unseen second half | **yes** |
| 4 | reward weighting is not a no-op | top-2 sets differ from unweighted on >= 10% of nodes | no |
| 5 | the walk is not breadth-first in disguise | raw-product order differs from BFS on >= 10% of the rendered 50 | no |
| 6 | dropping relation lines buys rules | rendered rule count rises >= 30% at 2,500 tokens | **yes** |
| 7 | seeding covers every sub-question | 4 of 4 represented in >= 95% of generations, both arms | no |
| 8 | the `text` arm's imbalance is real | descriptive, quantifies what 2.5 repairs | no |
| 9 | binary reward does not collapse `formula` | severance < 25% at co-citation depth >= 3 | **yes** |

**If any of 1, 2, 3, 6 or 9 fails, no run is authorised.** 

Tooling: a new `calibrate_cocite_rank.py` covers 1 to 4 and 9; `analyze_walk_share.py` is extended
with a checkpoint replay for 5 to 8. `calibrate_cocite_gate.py` is retired in the same change, with
its held-out-lift scoring ported across rather than deleted.


## 10. What this experiment can and cannot license

**A win licenses the project's claim: relational memory beats linear memory on `finer`.** If the
primary clears with its CI excluding zero, the supported statement is that a memory which stores
pairwise usage-and-outcome history and retrieves along it outperforms a memory that retrieves by
relevance alone, under an identical loop, prompt set, data order, token budget and
de-duplication rule.

**What a win still does not license is the universal form of the claim**, that relational structure
helps in general. That needs more than two tasks and one model.


**Stage D on `formula` is the confirmatory step.** It is weakened by the configuration having been
selected on `finer`, and `formula`'s power ceiling is a ~3.5pp CI half-width even at infinite
replicates, so it can confirm a large effect and cannot resolve a small one.

**The worst case, stated plainly.** If Stage A blocks, or the futility gate fires, the finding is
that edge coverage was the diagnosis and fixing it did not move accuracy. Combined with the two
committed nulls, that is a coherent claim about the limits of relational memory in
this loop.

## 11. Procedure, in order

1. The implementation, in one commit with the documentation it describes:
   `graph.py` (third counter, ranked predicate, priority queue, no relation lines),
   `retrieval.py` and `text.py` (per-sub-query seeding, round-robin),
   `ace.py` (both `record_generation` calls moved after the grade, `partial_credit` plumbed),
   the runners' flags, `PINNED_FIELDS`, `structural_stats()`.
   `check_memory_equivalence.py` is run **before and after**, and both results go in the commit
   message: the `text` arm's round-robin change means it is no longer byte-identical to the
   pre-refactor path, so it is stated up front instead of discovered later.
2. Stage A. Report all nine metrics. **Stop for the go/no-go.**
3. On a pass: `finer` seed 0, both arms. **Stop and read section 7, then section 6's gate.**
4. On a pass: `finer` seeds 1 and 2, both arms. Pooled analysis, section 5.
5. Stage C: the `graph + retrieval flat` ablation, plus whatever replicate the section 5 analysis
   says is needed.
6. Stage D: `formula`, three replicates, both arms, if sections 5 and 6 authorise it.

## 12. Deviations log

*(Empty. Every deviation from the above gets a dated entry here, and a deviation that is not logged
before the next run is a deviation that did not happen.)*
