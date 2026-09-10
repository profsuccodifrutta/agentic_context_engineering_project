# `finer` results, XBRL tagging

**Status: complete. Six training runs (2 arms × 3 replicates) plus the control. This is the final
analysis.**

Pre-registration: `EXPERIMENT_FINER.md`, with the gates in §7 and the comparison family in §5, all
written before any run of this set.

Headline, in one line: **memory helps, the two memory
representations could not be separated, and the point estimate did not favour the relational one.**

| question | answer | statistic (per-tag, pooled) |
|---|---|---|
| **Primary.** Does relational memory beat linear memory? | **Not shown, and the point estimate is negative** | `text → cocite` **−2.02pp**, CI [−0.0412, +0.0008], p=0.0601 |
| Does memory help at all? | **Yes, both kinds, decisively** | `none → text` +8.84pp, Holm p=0.00006; `none → cocite` +6.82pp, Holm p=0.00012 |
| Did the treatment actually fire? | **Yes, in all three replicates** | a different rendered rule set on 439-440 of 441 test generations, `walk_only_share` 0.109-0.135, walk-added rules cited at 1.32-1.55× the relevance rate |

Twelve of the thirteen pre-registered gates pass. The one that fails is structural, and it is
reported as failed in §5.

The most useful finding here is not the headline.

Because finer asks the model to choose from a fixed list of 139 tags, we can separate questions according to whether the playbook contains a rule for the correct tag.

This reveals two opposite effects. When the playbook contains the correct tag, memory helps substantially: accuracy improves by about 20 percentage points. When the playbook does not contain the correct tag, memory hurts: accuracy falls by about 17 percentage points. The likely explanation is that the model treats the retrieved playbook rules as a shortlist of plausible answers and tends to choose from that shortlist even when the correct tag is not represented.

The overall +8.84pp memory gain is the net result of these two effects, weighted by how many questions fall into each group. The important lesson is therefore not simply that “memory helps,” but that memory helps most when it covers the correct answer and can hurt when it does not.

These are **not** the `results/finer/preregistered/` or `results/finer/repaired*/` runs, all of
which are invalidated for reasons recorded in `EXPERIMENT_FINER.md` §9 and §10.

---

## 1. What was tested

`finer` is XBRL tagging. One row is four sub-questions answered in a single generator call, and its
target is four comma-separated US GAAP tags drawn from a fixed **139-tag** vocabulary printed in
every prompt. 

**It is one of the ACE authors' own benchmarks.** The test split
`finer_test_subset_006_seed42.jsonl` is their subset, used unmodified; the generator and reflector
prompts are theirs verbatim, shared with `formula`; and all-four-exact is their metric, reported
here as the declared secondary. 

**Three properties of the data decide whether the experiment can run at all**, all measured before
any run and recorded in `EXPERIMENT_FINER.md` §2:

| property | why it matters | measured |
|---|---|---|
| which tags go together is **not** predictable from their names | the memory ranks rules by embedding similarity, so a relation that similarity already encodes adds nothing. This is what `related_to` failed on | over the **1,000-row training pool**, Spearman between tag-name cosine and true co-occurrence is **+0.026** (Pearson +0.248), and only 47 of the 200 most-co-occurring pairs (**23.5%**) are in the top 200 by name similarity |
| rules cited together are about the same material | the four sub-questions come from one filing section and are answered in one call, so co-citation is structural rather than incidental | word overlap inside a row is **0.350** by Jaccard against **0.038** between rows, 9.2× higher |
| a row gives a pairwise relation something to hold | a row whose four answers repeat one tag carries no pair | test rows carry **2.71** distinct tags on average, and **93.2%** carry two or more |

The arms are the project's standard three. `none` carries the identical generator prompt with the
playbook slot filled `(empty)`, which is what makes `none → text` a read on memory rather than on
prompt shape. `text` is the linear baseline at `--retrieval flat`. `cocite` is
`--memory graph --retrieval graph --cocite`.

**An edge exists iff** a pair was co-rendered at least 3 times **and** co-cited on at least 10% of
those occasions. The stored quantity is a rate rather than a count, so the predicate ranks pairs
*chosen* together rather than pairs *shown* often.

## 2. Setup

| | |
|---|---|
| task | `finer` (`eval/finance`), 139-tag US GAAP vocabulary printed in every prompt |
| splits | `task_config_v2.json`: train `finer_train_200.jsonl` (200), val `finer_val_150.jsonl` (150), test `finer_test_subset_006_seed42.jsonl` (441) |
| arms | `none` (control), `text` (linear baseline, `--retrieval flat`), `cocite` (`--memory graph --retrieval graph --cocite`) |
| all three roles | `gemma4:e2b`, `--think_generator off`, `--num_ctx 32768` requested and served |
| scoring | `finer_tag_counts`, positional tag match. Deterministic: **no LLM grades any output, in any arm** |
| replicates | seeds 0, 1, 2 (see §8: the seed flag is inert, these are server-state replicates) |
| grow-and-refine | on in every arm, `--dedup --dedup_threshold 0.86` |
| curator supersession | **`--no_curator_refines` in every arm** |
| retrieval budget | 2500 tokens, 5 walk seeds, identical in both memory arms |
| gates | `--cocite_min_opportunity 3 --cocite_min_rate 0.10 --cocite_half_life 0` |
| `curator_frequency` / `num_epochs` / `max_num_rounds` / `eval_steps` | 1 / 1 / 3 / 100 |
| prompts | `FINER_PROMPTS` (`ace/prompts/finer.py`), curator **v4** |
| commits | `8b5f011` (all six training runs), `c5014fb` (control) |
| cost | 41.8 h of LLM time, 73.0M tokens |

**Everything the generator reads is charged to the same 2500-token budget**, co-citation relation
lines and legend included. The treatment arm shows fewer rules for the same tokens, and the
structure has to pay for itself. That is the honest form of the question, and §5 measures what it
cost here.

**On the control's commit.** The `none` arm ran at `c5014fb` and the six training arms at
`8b5f011`, so `compare_arms.py` refuses the set without `--force`. Re-verified before forcing:
`git diff c5014fb 8b5f011` touches `ace/prompts/finer.py` (the **curator** substitutions only),
`eval/finance/data_processor.py` (`finer_retrieval_query`), `ace/memory/retrieval.py`
(`score_against_query` accepting a sequence), and **type annotations alone** in `ace/ace.py`,
`ace/memory/base.py`, `ace/memory/none.py`, `ace/memory/text.py`, `ace/memory/graph.py`. An
`eval_only --memory none --retrieval none` run reaches none of it: it runs no curator and no
reflector, `NoMemory.render_for_generator` returns `EMPTY_SLOT` whatever the query is, and
`score_against_query` is never called. The generator prompt object is unchanged. §8 gives the
empirical confirmation, which is stronger than the diff: five `none` runs across three commits are
byte-identical.

## 3. Statistical approach

**The unit of analysis is the row, not the tag.** A row's four tags come from one filing section
and are answered in one call, so they are correlated (measured ICC 0.33). Scoring a row as the
fraction of its four tags that are right absorbs that correlation *before* any resampling sees it,
and because every row carries exactly four tags the mean of those fractions is exactly per-tag
accuracy.

The 139 tags are the fixed answer vocabulary printed in every prompt. They are the task, not a
sample drawn from it, so they are fixed effects and **rows are the sampling unit**. Rows are
independent, measured: adjacent-row word Jaccard is 1.01× random, and 11 of 97,020 row pairs share
a sentence. There is no block structure above the
row, so `cluster_paired_stats` runs with one row per block.

| statistic | role |
|---|---|
| **Cluster sign-flip permutation test** | **primary.** No distributional assumption, paired by row index, α = 0.05. With singleton blocks it is the permutation form of McNemar, applied to real-valued scores where McNemar itself is undefined. |
| **Cluster bootstrap 95% CI** | **primary.** Resamples rows with replacement. |
| McNemar exact | all-four only. Its `b` and `c` count discordant *binary* pairs, so it is undefined on fractional scores and is deliberately absent from every per-tag line. |
| Wilson interval | not reported on per-tag. A marginal interval over 1,764 tag decisions would claim 4× the precision this design has, and `compare_arms.py` suppresses it. |

**Replicates are pooled before clustering.** Each row is scored as its mean across an arm's three
runs, and those real-valued rows go through the same test. **The pooled analysis is 441 rows, not
1,323**, and `sd/√3` is never a standard error.


**Decision rule, pre-registered.** An arm beats another only if the pooled primary test is
significant **and** the mean gap across replicates exceeds the within-arm standard deviation. A gap
smaller than the replicate spread is noise regardless of p.

`compare_arms.py --selftest` was run first and passes, including the two limits that bind here:
singleton blocks must reproduce McNemar exactly, and pooling identical replicates must change
nothing.

## 4. Accuracy

**Per-tag, the pre-registered primary.** n = 441 rows, 1,764 tags per run.

| arm | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| `none` | 0.3566 | 0.3566 | 0.3566 | **0.3566** | 0.0000 |
| `text` | 0.4450 | 0.4439 | 0.4461 | **0.4450** | 0.0011 |
| `cocite` | 0.4150 | 0.4274 | 0.4320 | **0.4248** | 0.0088 |

**All-four exact, the authors' metric:**

| arm | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| `none` | 0.0680 | 0.0680 | 0.0680 | **0.0680** | 0.0000 |
| `text` | 0.0998 | 0.1202 | 0.1315 | **0.1172** | 0.0161 |
| `cocite` | 0.1111 | 0.1020 | 0.1020 | **0.1051** | 0.0052 |

`no_answer` is 0 to 5 of 441 in every run, 0.0% to 1.1%. 

Pooled across the three replicates, per-tag, 441 rows one block each:

| comparison | diff | bootstrap 95% CI | sign-flip p | Holm | Westfall-Young | |
|---|---|---|---|---|---|---|
| **`text → cocite`** (primary) | **−2.02pp** | **[−0.0412, +0.0008]** | **0.0601** | 0.0601 | 0.0601 | **null** |
| `none → text` | +8.84pp | [+0.0552, +0.1213] | 0.00002 | **0.00006** | 0.00002 | significant |
| `none → cocite` | +6.82pp | [+0.0363, +0.1002] | 0.00006 | **0.00012** | 0.00016 | significant |

All-four secondary, same pooling:

| comparison | diff | CI | p | (Holm) |
|---|---|---|---|---|
| `text → cocite` | −1.21pp | [−0.0333, +0.0091] | 0.2915 | 0.2910 |
| `none → text` | +4.91pp | [+0.0189, +0.0794] | 0.0011 | 0.0059 |
| `none → cocite` | +3.70pp | [+0.0076, +0.0665] | 0.0149 | 0.0337 |

Both metrics agree on sign and ordering in all three comparisons. Per replicate, so a reader can
see that pooling did not create the pattern:

| comparison | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `none → text` | +8.84pp, p=0.0001 | +8.73pp, p=0.0001 | +8.96pp, p=0.0001 |
| `none → cocite` | +5.84pp, p=0.0021 | +7.09pp, p=0.0001 | +7.54pp, p=0.0001 |
| `text → cocite` | −3.00pp, p=0.0595 | −1.64pp, p=0.3402 | −1.42pp, p=0.3994 |

The direction is negative in all three matched comparisons and in **all nine** cross-pairings of a
`text` replicate against a `cocite` replicate, range −1.19pp to −3.12pp. As on `formula`, the
direction is consistent and the magnitude is inside the noise. Unlike `formula`, it points away
from the treatment.

**Applying the decision rule:**

| comparison | pooled test significant? | mean gap > within-arm sd? | verdict |
|---|---|---|---|
| `none → text` | **yes** (Holm 0.00006) | **yes** (8.84pp > 0.11pp) | **established** |
| `none → cocite` | **yes** (Holm 0.00012) | **yes** (6.82pp > 0.88pp) | **established** |
| `text → cocite` | **no** (p=0.0601) | yes (2.02pp > 0.11 / 0.88pp) | **not established** |

The primary fails the first half of the rule and passes the second, so the treatment is not
established. Two of the three replicates share a trained
playbook in both arms (§8), so the within-arm sd is a **floor** rather than a full estimate of loop
noise. The `formula` baseline's replicate sd was 2.57pp on genuinely distinct playbooks, larger
than the 2.02pp gap here, so against a fairly estimated spread this comparison would have failed
the second half of the rule as well.

**The trap, in both directions.** Both memory arms clear α against the control and `text` clears it
by more, which is not evidence that `text` beats `cocite`: two significant results of different
size do not make a significant difference, and the direct test is the one that answers it.
Symmetrically, a negative point estimate is not evidence that the baseline is better. −2.02pp with
an interval covering zero is compatible with the arms being equal and with the treatment being
modestly behind, and this design's resolution is about 2.4pp (§8), just above where the gap sits.


## 5. The treatment fired

### Twelve gates pass

| gate | requirement | `text` s0 / s1 / s2 | `cocite` s0 / s1 / s2 |
|---|---|---|---|
| rules cited per generation | mean ≥ 2.0, and ≥2 in 60% | 3.12 / 3.20 / 3.20; 90.5% / 93.0% / 93.0% | 4.69 / 4.68 / 4.68; 90.7% / 94.8% / 94.8% |
| rendered rule set varies by question | ≥ 20 distinct sets over the first 25 test generations | **25 / 25 / 25** | **25 / 25 / 25** |
| curator writes the requested form | ≥ 50% of **rendered** rules a single `IF…THEN <tag>` line | 82.3% / 82.1% / 82.1% | 87.2% / 86.7% / 86.7% |
| curator files it the same way in every arm | < 10% of **stored** rules outside `context_clues_and_indicators` | **0/224, 0/228, 0/228** | **0/224, 0/226, 0/226** |
| `walk_only_share` | ≥ 0.05 on the final-test pooled share | n/a | **0.1350 / 0.1094 / 0.1094** |
| held-out predictive lift of the shipped gate | > 1.0× base rate on generations it was not built from | n/a | **9.35× / 4.26× / 4.26×** |
| `cocite_max_degree` vs live nodes | must not approach the live node count | n/a | 15 / 13 / 13 against 224 / 226 / 226 |
| memory outgrows the budget | rendered per call < live rules | 60.8 / 63.6 / 63.6 of 224 / 228 / 228 | 51.3 / 47.1 / 47.1 of 224 / 226 / 226 |
| replicate distinctness | prediction hashes distinct within an arm | ✅ (but see §8) | ✅ (but see §8) |
| serving placement | first calls ≥ 20 tok/s | 105 / 102 / 104 | 103 / 102 / 102 |
| learning signal | `check_learning_signal.py` [1]-[6] | pass ×3 | pass ×3 |
| curator prompt at the last step | below the 26,860 measured ceiling | 11,283 / 11,252 / 11,252 | 12,346 / 13,002 / 13,002 |

### Retrieval, measured on the 441 test generations that produced §4

| | `text` s0 / s1 / s2 | `cocite` s0 / s1 / s2 |
|---|---|---|
| live rules at the tested checkpoint | 224 / 228 / 228 | 224 / 226 / 226 |
| rules rendered per call | 60.8 / 63.6 / 63.6 | **51.3 / 47.1 / 47.1** |
| distinct rendered rule sets, of 441 | 441 / 440 / 440 | 439 / 440 / 440 |
| distinct rules ever rendered | 213 / 208 / 208 | 200 / 199 / 199 |

Retrieval is fully alive in both arms: an essentially unique slice per question, and **as a union
over the 441 questions** it reaches 88% to 95% of the memory. That union is what relevance achieves
by being asked 441 different questions; a single call renders about 51 rules of 224, and the pool
the co-citation walk can reach is far smaller again (13%, below). 

**Under the same 2,500-token budget the treatment reads 23% fewer rules**, 48.5 against 62.7 on
average, because its co-citation relation lines are charged to the same allowance. That is the
design working as intended, and here the structure did not pay for itself.

### The walk surfaces rules the generator then uses

| replicate | live | rendered/call | `walk_only_share` | walk added nothing | walk-added cite rate | relevance cite rate | ratio |
|---|---|---|---|---|---|---|---|
| seed 0 | 224 | 51.3 | **0.1350** | 207/441 | 0.1075 | 0.0693 | **1.550** |
| seed 1 | 226 | 47.1 | **0.1094** | 121/441 | 0.1270 | 0.0960 | **1.324** |
| seed 2 | 226 | 47.1 | **0.1094** | 121/441 | 0.1412 | 0.0927 | **1.523** |

**11% to 14% of what the generator read came from following co-citation edges, and it went on to
cite those rules 1.3 to 1.6 times as often as the rules relevance had chosen.** The `formula`
co-citation arms reached 1.25× to 2.32× on the same measure, so the walk behaves the same way on
both tasks.

Two cautions carried from `COCITE_RESULTS.md`. `bullet_ids` is the model's self-report. And
`walk_only_share` is a **necessary condition, never a sufficient one**: a random rewiring at
matched density scores higher on it than a real graph, because the metric rewards pointing away
from relevance. Held-out lift is what validates the edges, and it is 4.3× to 9.4× base rate here.

### The gate that fails: the co-citation core does not grow

**What the gate is for.** The walk starts from the five highest relevance-scored rules and follows
co-citation edges outward, so a rule with no edge can never be reached whatever the question is.
The set of rules carrying at least one edge, the **connected set**, is therefore the entire pool
the treatment can draw on. The pre-registration's eleventh gate asks that this set *"grow with the playbook, not stay
at a fixed core"*, because if the playbook grows to two hundred rules while the connected set stays
small, every question's walk lands in the same handful and the treatment stops being a retrieval
mechanism. It becomes "the baseline, plus the same extra block of text on every call", the arms
differ by a constant rather than by a mechanism, and comparing their accuracy answers nothing.

**What was measured.** Recomputed from each checkpoint's saved graph at the shipped gate, counting
live nodes only:

| step | seed 0: live rules / edges / connected set | seeds 1-2: live rules / edges / connected set |
|---|---|---|
| 50 | 66 / 17 / 14 (21.2% of live) | 71 / 58 / 22 (31.0%) |
| 100 | 127 / 115 / 33 (26.0%) | 143 / 60 / 27 (18.9%) |
| 150 | 169 / 54 / 26 (15.4%) | 181 / 87 / 29 (16.0%) |
| **200** (the tested checkpoint) | **224 / 63 / 30 (13.4%)** | **226 / 73 / 31 (13.7%)** |

The playbook more than triples, from 66 rules to 224. The connected set does not move: 14, 33, 26,
30. Its share falls from 21-31% to about 13%. **On the literal wording the gate is not met, and it
is reported as failed rather than argued away.**

**The pre-registered fallback fired, and it selected nothing.** `EXPERIMENT_FINER.md` §9 declares
what to do when a structural gate fails at rate 0.10: pick the rate from the fixed candidate set
{0.05, 0.10, 0.15, 0.20} that maximises connected nodes subject to
`cocite_max_degree <= 0.25 x connected_nodes`, and re-run the `cocite` arm at it. The rule, the
candidate set and the objective were fixed before any arm ran, and only structural quantities are
read. Evaluated at the tested checkpoint:

| rate | seed 0: connected / max degree / bar | seeds 1-2: connected / max degree / bar |
|---|---|---|
| 0.05 | 81 / 32 / 20.3 | 115 / 41 / 28.8 |
| **0.10 (shipped)** | 30 / 15 / 7.5 | 31 / 13 / 7.8 |
| 0.15 | 17 / 14 / 4.3 | 19 / 12 / 4.8 |
| 0.20 | 14 / 10 / 3.5 | 14 / 4 / 3.5 |

No candidate satisfies the constraint, the nearest miss being 0.20 on seeds 1-2 at degree 4 against
a bar of 3.5. **So no rate was selected and no re-run was licensed**, which is the outcome the
pre-registration names explicitly. Nothing clears it because the bar scales with the connected set
and the connected set is small: at 30 connected nodes, any rule holding 8 edges fails. The rule was
written expecting a larger component than this task produced.

**The degeneration the gate guards against did not happen**, and that is measurable separately:

| | seed 0 | seed 1 |
|---|---|---|
| distinct sets of walk-added rules, over 441 test generations | **161** | **208** |
| mean rules the walk added per generation | 6.92 | 5.15 |
| generations where the walk added nothing | 207 of 441 | 121 of 441 |
| union of every rule the walk actually added | 28 | 27 |
| of the rules connected at step 100, still connected at step 150 | 18 of 33 | 15 of 27 |
| of those connected at step 150, still connected at step 200 | 16 of 26 | 15 of 29 |

The walk composed 161 and 208 different additions across the test set, so it is genuinely
query-dependent rather than a fixed block, and it is not a hub, at max degree 15 and 13 against 30
and 31 connected nodes. The small core is not even frozen: only about half its members survive from
one checkpoint to the next, so which rules carry edges keeps changing while how many do not.

**So the honest statement is narrower than either the gate or its failure suggests.** What matters
for §4's null is the size of the **pool**. At the tested checkpoint the walk could reach **30 and
31 connected rules of 224 and 226, about an eighth of the memory**, and across the 441 test
generations it actually surfaced 28 and 27 rules that relevance would not have selected. The other
seven-eighths is reachable by relevance alone, and it is what the baseline spends its extra 14
rules per call on.


## 6. Where the memory's benefit comes from

Rows by number of tags correct, mean over replicates, out of 441:

| arm | 0/4 | 1/4 | 2/4 | 3/4 | 4/4 |
|---|---|---|---|---|---|
| `none` | **129.0** | 117.0 | 103.0 | 62.0 | 30.0 |
| `text` | **84.7** | 116.3 | 103.0 | **85.3** | **51.7** |
| `cocite` | **87.3** | 124.3 | 109.3 | 73.7 | 46.3 |

Memory does two things and both are visible here: it **rescues total failures**, with 0/4 falling
from 129 to about 85, and it **completes rows**, with 4/4 rising from 30 to about 50. That is the
transfer the task was chosen for.


Paired, pooled per-row scores:

| | better | worse | tied |
|---|---|---|---|
| `text` vs `none` | 238 | 145 | 58 |
| `cocite` vs `none` | 229 | 143 | 69 |
| `cocite` vs `text` | 160 | **189** | 92 |

Per replicate, `cocite` loses more rows than it wins in all three (111/150, 132/141, 126/146).
Structure is not inert here either: it moves about 270 of 441 rows, and the rows it moves do not
come back in its favour.

### The playbook helps where it has a rule and hurts where it does not

**The unit here is one tag question rather than one row.** A row asks four sub-questions and each
wants one tag, so the 441 test rows contain **441 × 4 = 1,764 individual tag questions**, each with
one correct answer. Everything below counts those and calls them questions.

Every rule is a mapping (`IF <condition> THEN <Tag>`), so reading a playbook tells you which tags it
has advice about. The test's answers use **137** of the 139 printed tags; a playbook holds a rule
for about **84** of them and says nothing about the other 53. Split the 1,764 questions by whether
the *correct answer* is a tag the playbook covers, then score each pile twice, once with no memory
and once with the playbook, each arm judged against its **own** playbooks:

| coverage from | tags covered | questions | no memory | with memory | |
|---|---|---|---|---|---|
| `text` playbooks | 84 of 137 | 1,322 covered | 0.3109 | **0.5008** | **+19.0pp** |
| | | 442 uncovered | 0.4932 | **0.2783** | **−21.5pp** |
| `cocite` playbooks | 81 of 137 | 1,290 covered | 0.3155 | **0.4858** | **+17.0pp** |
| | | 474 uncovered | 0.4684 | **0.2588** | **−21.0pp** |

Weighted by pile size these reproduce the headline exactly, +8.84pp and +6.82pp.
`analyze_finer_coverage.py` asserts that identity against the **pooled statistic
`compare_arms.py` publishes**, not against a mean it recomputes itself, and exits non-zero if it
fails. It catches a decomposition scored on a different scale from the result it decomposes, which
is a defect this table has had. It cannot check how `covered` is defined: the identity holds for
any partition of the questions, so a wrong coverage rule moves both piles and still balances.

**The playbook behaves like a shortlist.** On the uncovered pile, when the model is wrong it names
a *covered* tag **68.9%** of the time in the `text` arm and **77.6%** in `cocite`, against the
control's **45.5%** and **56.0%** on the same questions. It is pulled toward the tags it has
written down, and on questions whose answer is not among them it does worse than having no memory
at all. 

**This is not a restatement of training frequency.** Coverage does track exposure, so the obvious
objection is that the split merely separates common tags from rare ones:

| training examples for the tag | test tags | how many the playbook covers |
|---|---|---|
| 0 | 17 | 0 (0%) |
| 1-2 | 26 | 9 (35%) |
| 3-5 | 48 | 32 (67%) |
| 6-10 | 28 | 26 (93%) |
| 11+ | 18 | 17 (94%) |

Holding exposure fixed, the split survives. `text` arm, all 1,764 questions:

| training examples | has a rule? | questions | no memory | with memory | |
|---|---|---|---|---|---|
| 0 | no | 134 | 0.493 | 0.104 | **−38.8pp** |
| 1-2 | yes | 109 | 0.211 | 0.382 | **+17.1pp** |
| 1-2 | no | 130 | 0.408 | 0.218 | **−19.0pp** |
| 3-5 | yes | 306 | 0.281 | 0.411 | **+13.0pp** |
| 3-5 | no | 127 | 0.598 | 0.454 | **−14.4pp** |
| 6-10 | yes | 385 | 0.377 | 0.480 | **+10.3pp** |
| 6-10 | no | 20 | 0.700 | 0.683 | −1.7pp |
| 11+ | yes | 522 | 0.301 | 0.594 | **+29.3pp** |
| 11+ | no | 31 | 0.290 | 0.301 | +1.1pp |

Within every band a rule is worth +10 to +29pp. Its absence costs 14 to 19pp in the two bands
that carry most of the uncovered questions, and the two smallest uncovered cells (20 and 31
questions) sit within a point of zero, so the penalty is clearest where there is enough data to
see it. Exposure strongly predicts whether a rule gets written; conditional on
exposure band, questions whose answer tag is covered do substantially better than questions whose
answer tag is not.

**The worst cell is the one the training split cannot reach.** Where a tag never appeared in
training, memory costs **38.8 points**, far more than in any other band. Those 134 questions are
where the shortlist effect operates with nothing to offset it: no rule can exist, and the model is
still pulled toward the 84 tags that do have one.

**Three qualifications.** The piles are not randomly assigned, since the curator writes rules for
tags it watched fail, so each pile is compared against itself and never one against the other.
**This measurement is only possible on `finer`**, because the answer space is a printed, enumerable
list; on `formula` the same interference could be present with no way to observe it.

**What it bounds.** Coverage is limited by exposure and exposure by the training split. **17 of the
137 test tags never appear in the 200 training rows**, which is 134 questions or 7.6% of the total,
so no amount of training on this split can reach them. 

## 7. Interpretation

**What this experiment licenses.**

- **Memory helps, and the effect is large and reproducible.** +8.84pp for linear memory and
  +6.82pp for relational, both with CIs excluding zero, both surviving Holm over the three
  pre-registered comparisons, and both confirmed on the secondary metric. 
- **The memory gain is largely explained by answer-tag coverage.** Where the playbook holds a rule
  for the correct tag it is worth about +20 points; where it does not it costs about 17, because
  the model treats the memory as a shortlist and picks from it anyway. 
- **Relational structure was not shown to beat linear memory**.


**What it does not license.**

- **Not "structure hurts".** The point estimate is −2.02pp against a design resolution of about
  2.4pp, and the interval covers zero. The negative sign in nine of nine pairings is worth testing
  at higher power, not reporting as a result.
- **Not "the treatment had a fair full-strength trial".** It was tested exactly as pre-registered,
  but under two measured constraints: its edges reached about an eighth of the memory, and paying
  for the relation lines out of the shared budget left it reading 23% fewer rules than the
  baseline. This design cannot separate "lost because it read less" from "lost because structure
  did not pay for itself"; a `--memory graph --retrieval flat` ablation would, and was not run.
- **Not a claim about causation from citations.** `bullet_ids` is the model's self-report. Held-out
  lift shows it carries real information; it is still not proof that a rule caused an answer.

**Taken with the `formula` co-citation experiment, the claim this project supports is about memory,
not about structure.** Memory helps where content transfers, on both tasks and by a similar margin.
Co-citation structure has not measurably moved accuracy on either, with the treatment verified
delivered before any accuracy was read both times.

**What would settle it**, roughly in order of how much this run says each is worth:

- **More epochs.** `--num_epochs 1` against the paper's 5 is the deviation costing the most here.
  Coverage is monotone in exposure (0%, 27%, 60%, 82%, 89% across the bands in §6) and a second
  pass doubles every tag's exposure. The loop learns from failures, 280 incorrect reflections
  against 27 correct, so a later epoch has fewer errors to write from, but the rows it still fails
  are disproportionately the ones with no rule yet.
- **More training data**, which raises a ceiling epochs cannot touch. 17 test tags never appear in
  the 200 training rows at all. The 250-row split leaves 13 unseen and the full 1,000-row pool
  leaves 7.
- **An edge predicate that reaches more than an eighth of the memory** (§5). The pool, not the
  quality of what the walk surfaces, is the binding limit on the treatment.
- **The `--memory graph --retrieval flat` ablation**, to separate "edges choose what is read" from
  "structure costs budget".
- **Independent replicates**, which this design cannot currently guarantee (§8), and a larger test
  set to buy the roughly 1pp of resolution this design is short of.

One caution on the first two: a bigger playbook faces harsher selection, since three times the
rules at the same 2,500-token budget is a 9% rendered fraction rather than 27%. Retrieval delivery
is not what binds today: of the 1,441 questions the seed-0 `text` playbook could answer, the
matching rule was rendered for that specific question **1,297 times, 90.0%** (93.4% in the other
two replicates, 85.3% to 88.0% in `cocite`). That is the figure to re-measure in such a run rather
than assume.

## 8. Disclosures

**The replicates.** `--seed` is inert: it reaches Ollama's sampler and nothing else, and at
`temperature=0.0` the sampler is greedy argmax and never consults it. Run-to-run divergence tracks
**opaque server state** instead, an uncontrolled and unrecorded property of the live endpoint that
fixes the order the tokens are computed in, so a near-tied argmax flips and the loop amplifies it.
Restarting Ollama between runs is the protocol used to force a new state, and it worked: all six
training runs produced distinct predictions. The mechanism beyond that is not established and this
document does not claim one.

**How far the state change reached is not the same in every pair.** Seeds 1 and 2 trained
byte-identically in both arms and diverged only at the final test:

| what | `text` seed 1 vs seed 2 | `cocite` seed 1 vs seed 2 |
|---|---|---|
| training generator calls | 680 shared, **0 responses differ** | 668 shared, **0 differ** |
| reflector calls | 307 shared, **0 differ** | 299 shared, **0 differ** |
| curator calls | 200 shared, **0 differ** | 200 shared, **0 differ** |
| `curator_operations_diff.jsonl` | 520 rows, differ **only in `timestamp`** | 475 rows, same |
| `best_playbook.txt` / `final_playbook.txt` | **byte-identical** | **byte-identical** |
| validation generations (150 × 2 checkpoints) | **300/300 identical** | **300/300 identical** |
| final-test prompts | **441/441 identical** | **441/441 identical** |
| final-test responses | **441/441 differ** | **441/441 differ** |

**The limitation is symmetric**, since it holds in both arms, so
neither arm's spread is measured on more draws than the other's. These are still three replicates
and are analysed as three: each produced its own 441 test generations, and those are what the
comparison resamples.

The floor, per-tag, rows moved out of 441:

| arm | 0↔1 | 0↔2 | 1↔2 | accuracy sd |
|---|---|---|---|---|
| `none` | n/a | n/a | n/a | **0.0000** |
| `text` | 264 | 267 | **187** | 0.0011 |
| `cocite` | 261 | 271 | **182** | 0.0088 |

The 1↔2 column is smaller than the others in both arms, which is the shared playbook showing
through. The `text` arm's shape matters too: about 265 of 441 rows change score between replicates
while accuracy moves 0.22pp, a large effect with no direction.

**The control is deterministic, measured rather than assumed.** Five `none` runs exist, spanning
2026-08-26 to 2026-09-02 across three commits and two prompt sets, and **all five
`test_test_predictions.jsonl` files are byte-identical** (`md5 094ed370...`) at 0.3566. Its
within-arm sd is exactly zero over five attempts, and the byte-identity across commits is the
empirical form of §2's argument that the intervening code changes cannot reach it. It is entered
three times in the pooled analysis, which `--selftest` proves is exact rather than an
approximation. A sixth run is not needed and on this evidence would not produce a different file.

**Power.** The pre-registered figure was a 4.7pp minimum detectable effect at 80% power for a
single replicate. Pooled at R=3 the equivalent for this pair is about 2.4pp at α=0.05, and the
observed gap is 2.02pp. The null is a power limit, not evidence of equivalence, and it cuts both
ways.

**Checkpoint selection.** Every one of the six runs selected step 200, the later of its two
checkpoints, so no arm was tested at a playbook its sibling did not also reach:

| replicate | `text` val@100 → @200 | `cocite` val@100 → @200 |
|---|---|---|
| seed 0 | 0.4150 → **0.4767** | 0.3650 → **0.4167** |
| seed 1 | 0.3917 → **0.4533** | 0.4283 → **0.4717** |
| seed 2 | 0.3917 → **0.4533** | 0.4283 → **0.4717** |

At seeds 1 and 2, validation ranked `cocite` 1.8pp above `text` while the test ranked it 1.7pp
below. Selection was correct and identical in both arms, and the discrepancy is unsurprising given
a 150-row validation set against a 441-row test.

**Loop health.** `check_learning_signal.py` [1]-[8] pass on all six runs.

| | `text` s0 / s1 / s2 | `cocite` s0 / s1 / s2 |
|---|---|---|
| train accuracy, per-tag (band 0.20-0.85) | 0.427 / 0.440 / 0.440 | 0.401 / 0.459 / 0.459 |
| generations citing a bullet | 199/200, 200/200, 200/200 | 200/200 ×3 |
| generator unparseable | 0.0% / 0.1% / 0.1% | 0.6% / 0.4% / 0.3% |
| curator unparseable | 0.5% / 0.0% / 0.0% | 0.0% ×3 |
| pasted rendered-bullet prefixes | **0 of 461 / 520 / 520** | **0 of 465 / 475 / 475** |
| ADDs adding nothing new (limit 50%) | 13.9% / 20.6% / 20.6% | 11.2% / 22.3% / 22.3% |



Grow-and-refine absorbed comparable shares in the two arms: `cocite` records 198 of 422 nodes
(46.9%) and 174 of 400 (43.5%) superseded by dedup, and `text` lands 224 of 397 and 228 of 413 ADDs
as live rules, so 43.6% and 44.8% absorbed. Both are under the 50% guard `--dedup_threshold 0.86`
was chosen against, and the arms end with nearly the same memory size. One curator call in the
`text` seed-0 run returned unusable JSON and was skipped, 1 of 200, logged in
`curator_failures.txt`.


## 9. Cost

Serving throughput was stable in every run, 88 to 112 tok/s with no run varying more than 1.2×, so
no timing figure is contaminated by the GPU-placement failure mode that affected earlier runs.

| arm | total tokens (mean) | vs `text` | LLM hours (mean) | mean per-tag accuracy |
|---|---|---|---|---|
| `none` | 1,594,323 | −86.1% | 1.74 | 0.3566 |
| `text` | 11,435,974 | baseline | 6.25 | 0.4450 |
| `cocite` | 12,372,551 | **+8.2%** | 7.12 (+13.9%) | 0.4248 |

Per replicate the treatment's token overhead is +6.8% / +8.6% / +8.2%. On `formula` the same
treatment ran 0.2% *cheaper* than the baseline, so structure is not free here where it was there: a
`finer` rule is a one-line mapping, which makes the relation lines a larger share of a smaller
rendering. **Structure cost +8.2% tokens and +13.9% wall time for −2.02pp of unresolvable
accuracy.** Whole experiment: 73.0M tokens and 41.8 LLM-hours across seven runs.

## 10. Reproducing these figures

```
# control, once. Deterministic on this task, see section 8
uv run python -m eval.finance.run --task_name finer --mode eval_only \
  --save_path results/finer/repaired_v2/seed0/none --memory none --retrieval none \
  --config_path ./eval/finance/data/task_config_v2.json \
  --generator_model gemma4:e2b --reflector_model gemma4:e2b --curator_model gemma4:e2b \
  --think_generator off --num_ctx 32768 --seed 0 \
  --dedup --dedup_threshold 0.86 --eval_steps 100 --retrieval_budget 2500 --retrieval_seeds 5

# per seed S in {0,1,2}
uv run python -m eval.finance.run --task_name finer --mode offline \
  --save_path results/finer/final/seed$S/text --memory text --retrieval flat \
  ... same flags ... --no_curator_refines --skip_initial_test --seed $S

uv run python -m eval.finance.run --task_name finer --mode offline \
  --save_path results/finer/final/seed$S/cocite --memory graph --retrieval graph --cocite \
  --cocite_min_opportunity 3 --cocite_min_rate 0.10 \
  ... same flags ... --no_curator_refines --skip_initial_test --seed $S
```

Analysis. `--force` is required, and is licensed by §2's argument about the control's commit:

```
uv run python compare_arms.py --selftest
uv run python compare_arms.py --metric finer_per_tag --force \
  results/finer/final/seed{0,1,2}/{text,cocite}/ace_run_* \
  results/finer/repaired_v2/seed0/none/ace_run_* \
  results/finer/repaired_v2/seed0/none/ace_run_* \
  results/finer/repaired_v2/seed0/none/ace_run_*
uv run python compare_arms.py --metric exact --force <same runs>
uv run python check_learning_signal.py results/finer/final/seed{0,1,2}/{text,cocite}/ace_run_*
uv run python analyze_walk_share.py results/finer/final/seed{0,1,2}/cocite/ace_run_*
uv run python calibrate_cocite_gate.py --shipped-rate 0.10 results/finer/final/seed*/cocite/ace_run_*
uv run python analyze_finer_coverage.py results/finer/final/seed*/{text,cocite}/ace_run_* results/finer/repaired_v2/seed0/none/ace_run_*
uv run python analyze_cocite_structure.py results/finer/final/seed*/cocite/ace_run_*
uv run python analyze_replicate_divergence.py results/finer/final/seed*/{text,cocite}/ace_run_*
uv run python aggregate_run_costs.py results/finer/final/seed*/*/ace_run_* \
  results/finer/repaired_v2/seed0/none/ace_run_*
```

Every figure in this document is recomputable from the commands above, which is a property
worth stating because it was not true when the file was first written: sections 5, 6 and 8 were
produced by one-off scripts, and that is how a scoring defect in section 6 survived a full review.
The `analyze_*` scripts above close that gap. `analyze_finer_coverage.py` **asserts** that the two
coverage piles reproduce the headline gap and exits non-zero otherwise. That assertion is a scale
check and nothing more: it fires when the decomposition is scored differently from the result, and
it is blind to how `covered` itself is defined, because the identity holds for any partition of the
questions. What the partition means is fixed in `analysis_common.mapping_targets` and has to be
read there, not inferred from the script passing.

Four things about reading these figures back.

**`--metric finer_per_tag` is required.** Without it `compare_arms.py` scores all-four-exact, the
declared secondary, and it prints an unmissable banner saying so.

**The control is passed three times** so `pooled_arm_scores` sees equal replicate counts. That is
exact rather than an approximation: the arm has one outcome (§8), and `compare_arms.py --selftest`
proves that pooling identical replicates reproduces the single-run result.

**`aggregate_run_costs.py` will report "seed differs across arms"** on the seven-run set. That is
the audit assuming one triple; seed is the deliberately varied replicate index here, and the arms
are matched on every pinned field within each replicate.

**`walk_only_share` must be read from `analyze_walk_share.py`, never from `structural_stats()`.**
The counter in the run artifacts is a run-cumulative average, diluted by every early step where the
playbook still fitted inside the budget, and it is written at eval checkpoints only, so the final
test contributes nothing to it.

The §6 coverage decomposition scores tag questions through `finer_tag_counts`, the same comparison
`compare_arms.py --metric finer_per_tag` uses, which is why its two piles reproduce the headline
gap exactly. Scoring it any other way breaks that identity.
