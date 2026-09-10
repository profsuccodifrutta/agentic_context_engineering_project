# Co-citation results, `formula`

**Status: complete. All nine runs (3 arms × 3 replicates) finished. This is the final analysis.**

Pre-registration: `EXPERIMENT_COCITE.md`, written 2026-08-18 before any run of this set.

**Added 2026-08-24:** §4.1 records the same model with no ACE prompt at all, as a reference point.
It is not an arm and takes no part in the analysis — this is a three-arm experiment.

Headline, in one line: **the graph's one irreducibly relational capability was built, delivered
and verified to work, and it still did not beat a flat list.**

| question | answer | statistic |
|---|---|---|
| **Primary.** Does relational memory beat linear memory? | **No** | `text → cocite` **+2.17pp**, CI [−0.020, +0.060], p=0.3344 |
| Does memory help at all? | **Yes, both kinds** | `none → cocite` +10.83pp, Holm p=0.0016; `none → text` +8.67pp, Holm p=0.0280 |
| Did the treatment actually fire? | **Yes, in all three replicates** | walk share 0.21 / 0.05 / 0.44; edges predict held-out co-citation at 3.6× to 11.6× base rate |

Both secondary results survive correction for the three comparisons this experiment declared
(§3.1). The primary null is unaffected by correction, since adjustment can only make a null more
likely.

The primary null is the informative part of this experiment, because for the first time the
mechanism behind it was fully instrumented. Sections 5 to 7 are where the result lives.

---

## 1. What was tested

The pre-registered three-arm experiment (`background/FORMULA_RESULTS_PREREGISTERED.md`) found that memory
helps and that structure was not shown to. Its diagnosis was mechanical rather than statistical:
the graph's only retrieval-facing relation, `related_to`, is cosine similarity between two rules'
**texts**, while retrieval scores rules by cosine between the **question** and each rule. Those
are nearly the same signal, so the walk landed on rules relevance would have picked anyway.

That suggests a sharper test. A relation earns a graph only if it is an irreducible fact about a
*pair*, not a function of the items:

| relation | pairwise | derivable from the rules' own text | needs a graph |
|---|---|---|---|
| `refines` | no (unary: "this rule is dead") | n/a | no: a flat list with a `supersedes` field |
| `related_to` | yes | **yes**, cosine of two texts | no: a flat list plus an embedding index |
| `co_cited` | yes | **no**, exists only in accumulated history | **yes** |

`--cocite` replaces similarity edges with co-citation edges: connect two rules when the generator
itself named them together in `bullet_ids`. Both arms already receive that citation stream
(`helpful` and `harmful` are derived from it today). A linear memory simply cannot hold the
pairwise part of it. So the comparison becomes: *same data, same loop, same token budget; one arm
can represent per-item facts, the other can represent facts about pairs.*

**An edge exists iff** the pair was co-rendered at least 3 times **and** was co-cited on at least
20% of those occasions. The stored quantity is a rate, not a count, so the rule ranks pairs
chosen together rather than pairs shown often.

## 2. Setup

| | |
|---|---|
| task | `formula` (`eval/finance`), 50 formulas, definitions stripped from the prompt |
| splits | `task_config_v2.json`: train 250 (5/formula), val 100 (stratified 2/formula), test 200 (4/formula) |
| arms | `none` (control, reused), `text` (linear baseline), `cocite` (`--memory graph --retrieval graph --cocite`) |
| all three roles | `gemma4:e2b`, `--think_generator off`, `--num_ctx 32768` requested and served |
| scoring | `answer_is_correct`, the extracted answer matched numerically against the target. Deterministic: **no LLM grades any output, in any arm** |
| replicates | seeds 0, 1, 2 (see §8: the seed flag is inert, these are server-state replicates) |
| grow-and-refine | on in every arm, `--dedup --dedup_threshold 0.88` |
| retrieval budget | 2500 tokens, 5 walk seeds, identical in both memory arms |
| gates | `--cocite_min_opportunity 3 --cocite_min_rate 0.20 --cocite_half_life 0` |
| reference point | `raw`, 3 runs, no ACE prompt at all (§4.1) — not an arm |
| cost | 27.5 h of LLM time, 45.1M tokens |

**Everything the generator reads is charged to the same 2500-token budget**, co-citation relation
lines and legend included. The treatment arm shows fewer rules for the same tokens, and the
structure has to pay for itself. That is the honest form of the question.

**`text` was re-run rather than reused** from the pre-registered set. The 2026-08-16
`_RENDERED_PREFIX` fix changed `sanitize_curator_operations`, which every arm executes, so the old
baseline was on a different code path. The `none` arms were reused, licensed by the fact that they
make no curator calls at all.

**Commits.** The three cocite-experiment commits (`8c317e8`, `0f42560`, `784c20d`) have an
**empty run-path diff**: `git diff --stat` over `ace/ eval/ llm.py utils.py playbook_utils.py
logger.py` returns nothing, and the only changes are two analysis scripts, two fairness audits and
documentation. The six new runs executed identical code. The reused `none` arms are on older
commits, whose run-path changes are confined to graph memory, co-citation plumbing, the curator
sanitiser, and two dead-code removals, none of which a `none` run reaches. That is the licence for
`compare_arms.py --force`, and it was verified rather than assumed.

## 3. Statistical approach

**The unit of analysis is the formula, not the row.** The test split is 50 formulas × 4 instances,
and the four instances of one formula succeed and fail together. Treating 200 rows as 200
independent trials claims the precision of 200 draws for a design that has about 50.

| statistic | role |
|---|---|
| Wilson 95% interval | orientation only. Marginal, unpaired, assumes iid rows. No conclusion is drawn from it. |
| McNemar exact (paired) | its `b` and `c` counts are informative: they separate "the memory did nothing" from "the memory fixed 15 and broke 15". Its p-value inherits the iid assumption and is anti-conservative here. |
| **Cluster sign-flip permutation test** | **primary.** Blocks on the formula and flips the sign of whole 4-row blocks. |
| **Cluster bootstrap 95% CI** | **primary.** Resamples the 50 formulas with replacement, not the 200 rows. |

**Replicates are pooled before clustering.** Each row is scored as its mean correctness across an
arm's three runs, and those real-valued rows go through the same clustered test. This averages out
run-to-run noise. It does **not** add data: every replicate sees the same 50 formulas, so the
block count stays 50 and the row count stays 200.

**The detection floor was declared before any run: roughly +4pp.** Power scales with clusters, not
rows, and the measured ceiling is a ~3.5pp CI half-width even at infinite replicates. A null below
4pp was pre-committed to be reported as an underpowered non-result rather than as evidence
against the treatment. §5 says why this null is stronger than that floor alone would allow.


Before analyzing the actual experiment, was verified that `compare_arms.py --selftest` behaves correctly in a bunch of carefully constructed edge cases, including cases where the mathematically correct answer is already known.
`compare_arms.py --selftest` was run first and passes on all fourteen limit cases, including the
ones where the answer is known in advance (singleton blocks must reproduce McNemar exact,
perfectly correlated blocks of *k* must widen the interval by exactly √*k*, pooling identical
replicates must change nothing, a family of one must not be adjusted, and Westfall-Young must
charge nothing for three copies of the same test, where Holm charges 3×).

### 3.1 Multiple comparisons

**The family is the three comparisons this experiment pre-registered**, one primary and two
secondaries, declared in `EXPERIMENT_COCITE.md` §5 before any run. Declaring it in advance is
what stops the grouping from being chosen after seeing which grouping passes.

The earlier three-arm formula experiment is reported as **motivation, not as a result**: what is
carried over from it is mechanism (the walk fired at 0.09 to 0.21; a random rewiring at matched
density surfaced 2.3× to 3.6× more novel rules than the real similarity graph, so the relation
was redundant with the ranking that chose the walk's seeds), and no significance claim is made
from its accuracies. Comparisons that carry no significance claim do not enter a family whose
purpose is to control false rejections.

Those runs are also a different system rather than an earlier replicate of this one. They ran
2026-08-08 to 08-13; commit `bc124eb` of 2026-08-16 fixed the curator sanitiser's rendered-prefix
regex, which had been leaking a pasted snapshot of another rule's counters into 82 rules across
their nine playbooks as rule text. Every run reported here is after that fix. So the m=6 table
below groups comparisons made on two different code paths, which is why it is a worst case and
not the natural family. It is still reported.

Resampled at 50,000 sign-flip draws, because an adjusted p-value near 0.05 cannot be resolved at
the 10,000 the pairwise block uses:

| comparison | raw p | **Holm** | Westfall-Young |
|---|---|---|---|
| `none → cocite` | 0.00054 | **0.0016** | 0.0013 |
| `none → text` | 0.01398 | **0.0280** | 0.0234 |
| `text → cocite` (primary) | 0.33437 | 0.3344 | 0.3344 |

**Both secondaries survive both procedures, and the primary is null before and after
correction**, the expected direction, since adjustment can only make a null more likely, so
multiplicity cannot threaten this experiment's headline claim.

**Holm is the figure to quote.** It is a step-down procedure controlling the family-wise error
rate under any dependence, which is the standard choice for a small pre-declared family.

**Westfall-Young is the sensitivity check.** Holm controls the error rate under *any* dependence,
which means it assumes the worst case; these comparisons are far from worst case because
`none → cocite` and `none → text` are driven by the same three control runs. Resampling all three
under one set of block sign-flips carries that correlation into the null. It matters at the
margin, and it matters more if the family is widened: **with the earlier experiment's three
comparisons included (m=6), Holm breaks and Westfall-Young does not.**

| m=6 family | raw p | Holm | Westfall-Young |
|---|---|---|---|
| `none → cocite` | 0.00054 | 0.0032 ✓ | 0.0022 ✓ |
| `none → text` | 0.01398 | 0.0655 ✗ | 0.0432 ✓ |
| formula `none → graph` | 0.01312 | 0.0655 ✗ | 0.0432 ✓ |

Holm fails there for a specific and instructive reason: it is a step-down procedure, so the two
comparisons at 0.013 and 0.014 break its chain together and neither can be rejected. Westfall-Young
rejects both because it knows they are largely the same evidence. The two rescued values sit at
0.0432 against a 0.05 threshold, close enough that the Monte Carlo error matters: at 50,000 draws
the 95% interval on that figure is roughly ±0.002, so it is under the line but not by much, and it
should be quoted rather than described as comfortable.

**The conclusion does not depend on the family declaration**, which is the point of reporting the
sensitivity rather than waiting to be asked for it.

## 4. Accuracy

| arm | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| `none` | 0.6250 | 0.6200 | 0.6200 | **0.6217** | 0.0029 |
| `text` | 0.6800 | 0.7150 | 0.7300 | **0.7083** | 0.0257 |
| `cocite` | 0.7350 | 0.7250 | 0.7300 | **0.7300** | 0.0050 |

Pooled across replicates, blocked on formula:

| comparison | diff | bootstrap 95% CI | sign-flip p | Holm (§3.1) | |
|---|---|---|---|---|---|
| **`text → cocite`** (primary) | **+2.17pp** | [−0.0200, +0.0600] | **0.3344** | 0.3344 | **null** |
| `none → cocite` | +10.83pp | [+0.0517, +0.1700] | 0.00054 | **0.0016** | significant |
| `none → text` | +8.67pp | [+0.0217, +0.1550] | 0.01398 | **0.0280** | significant |

**The primary gap is smaller than the baseline arm's own replicate spread.** `text` varies by
2.57pp across three runs of identical code on identical data; the arms differ by 2.17pp. A gap
below the within-arm sd is run noise, not an effect of the memory.

Per-seed, paired on the row:

| seed | `text` | `cocite` | fixed | broken | net | clustered diff | p |
|---|---|---|---|---|---|---|---|
| 0 | 0.6800 | 0.7350 | 22 | 11 | +11 | +5.50pp | 0.0492 |
| 1 | 0.7150 | 0.7250 | 15 | 13 | +2 | +1.00pp | 0.8698 |
| 2 | 0.7300 | 0.7300 | 15 | 15 | **0** | +0.00pp | 1.0000 |

Read the discordant counts, not the net. The two memories **disagree on 14% to 17% of the test
set in every replicate** while the net difference collapses to zero. This is not an inert
treatment. It is a treatment with a large effect and no direction

### 4.1 The model with no ACE prompt (reference point, not an arm)

`run_raw_baseline.py` asks the question directly, with no ACE prompt around it, no playbook slot,
no reflection section, no `bullet_ids`  holding the decoding settings and the scoring path
identical to every arm. Run three times on 2026-08-24.

| | per run | mean |
|---|---|---|
| `raw`, no ACE prompt | 0.3500 / 0.3450 / 0.3450 | **0.3467** |
| `none`, ACE prompt, empty playbook | 0.6250 / 0.6200 / 0.6200 | **0.6217** |

**This is not a fourth arm, and no statistic is computed on it.** It differs from the arms in
prompt structure, which is not this experiment's variable, so it cannot be a peer in the paired
tests and it does not enter the §3.1 family. This is a three-arm experiment: the memory question is
answered by `none → text` and `none → cocite` above.

It is on the record for one reason. The `none` control is easy to read as a base-LLM number and it
is not one, `none` is the same model under a playbook-shaped prompt that happens to be empty.

The 27.5pp between them is not attributed to anything here. It is whatever differs between the two
prompts, and these runs do not separate the pieces. The raw runs do emit far fewer output tokens
(36 per answer against 604), so the model is probably producing less reasoning, but that is an
observation rather than a mechanism. Either way it leaves the memory comparisons alone, because
`none` already has no memory.

## 5. The treatment fired, and that is what makes the null informative

A null is only worth reporting if the thing being tested actually happened. Three independent
gates, all declared in §7 of the pre-registration before any run:

| gate | seed 0 | seed 1 | seed 2 | verdict |
|---|---|---|---|---|
| live rules at the tested checkpoint | 69 | 103 | 147 | |
| co-citation edges | 18 | 12 | 152 | not near zero |
| max degree | 11 | 4 | 19 | no hub (would need to approach the live count) |
| **`walk_only_share`** | **0.2061** | **0.0512** | **0.4449** | the arms read different rules |
| **walk-added rules cited / relevance rules cited** | **1.252** | **2.319** | **1.492** | the walk is not rendering noise |
| held-out predictive lift of the shipped gate | 9.46× | 11.58× | 3.56× | the edges predict co-citation on unseen data |

**These six figures were corrected on 2026-08-28 and the conclusions are unchanged.**
`analyze_walk_share.py` kept the *first* generator log per test index, and validation evaluations
write under the same `generator_test_eval_<index>_*` prefix, so the indices below the
validation-set size came from a validation pass rather than the final test. It now keeps the last
file per index. `walk_only_share` moves by at most 0.0097 and the citation ratio by at most 0.184;
the spread across replicates is still eightfold (0.0512 to 0.4449), the walk still surfaces rules
the generator uses more often than relevance does, and the null in §4 is untouched. The figures
before the correction were 0.2094 / 0.0525 / 0.4352 and 1.279 / 2.269 / 1.308. The similarity-edge
arms of the pre-registered set move the same way, from 0.2066 / 0.0939 / 0.2050 to 0.1941 / 0.0941
/ 0.2063.

Two of these deserve emphasis.

**The edges are real.** Built on the first half of each run's training generations and scored on
the unseen second half, an edge predicts co-citation at 3.6× to 11.6× the base rate. Random edges
are 1.00× by construction. This is not a metric that can be gamed by pointing edges anywhere.

**The walk is not wasting the budget.** `walk_only_share` alone is a necessary condition and never
a sufficient one: a random rewiring at matched density scores 2.3× to 3.6× *higher* than a real
similarity graph, because the metric rewards pointing away from relevance and cannot tell a
meaningful edge from a scrambled one. The missing half is whether the generator then *uses* what
the walk added. It does, at 1.25 to 2.32 times the rate at which it uses the rules relevance
selected on its own.

**Then the decisive observation.** Across the three replicates the treatment varied by a factor of
**eight in walk share and thirteen in edge count**, and accuracy did not move at all:

| seed | edges | walk share | accuracy |
|---|---|---|---|
| 1 | 12 | 0.0512 | 0.7250 |
| 0 | 18 | 0.2061 | 0.7350 |
| 2 | 152 | 0.4449 | **0.7300** |

Seed 2 is the strongest delivery of this treatment ever measured in this repo: 152 edges, 44% of
what the generator read placed there by the graph rather than by relevance, and those rules cited
at 1.49 times the relevance rate. Paired against its own baseline it changed 30 of 200 answers and
netted exactly zero.

This is why the null reads as more than "underpowered". A pre-registered floor of 4pp says an
effect smaller than 4pp cannot be resolved. The dose-response evidence says something stronger:
inside the treatment arm, an eightfold increase in dose produced no gradient at all.

## 6. Where the null is not

Three candidate explanations were each closed on evidence, all offline, none costing a run.

**It is not the edge predicate.** The shipped gate is *confidence*, `P(both cited | both
rendered)`. Confidence favours pairs whose members are each cited often, frequent citation tracks
high relevance, and so the surviving edges point where relevance already looks. A *lift* gate,
dividing by what independent citation would have produced, is the principled repair and it was
measured with `calibrate_cocite_gate.py`, comparing the two rankings at matched edge count on
held-out generations. Both halves of the diagnosis hold: confidence edges connect rules cited
2.4× to 4.6× more often than the average rule, while lift edges sit at or below average, so lift
does remove the popularity bias. But lift's edges do not predict held-out co-citation at all:

| edges kept | confidence | lift |
|---|---|---|
| top 10 | 17.53× base rate | 0.31× |
| top 25 | 11.19× | 0.63× |
| top 50 | 7.63× | 0.80× |
| top 100 | 5.28× | 1.20× |
| top 200 | 3.70× | 2.93× |

Raising the support floor to 5, 10, 20 or 40 never reverses the ordering at any *k*. The two
converge only at *k*=200, where the edge set is a large fraction of the entire candidate space.
**The finding underneath is that on this material association and popularity are not separable.**
The pairs genuinely cited together are the pairs that get cited. No reweighting of the same
co-citation stream can both predict co-citation and point away from where relevance already
looks. That is a constraint on any predicate derived from this stream, not a defect of lift.

**It is not the read path.** §5 established that the walk surfaces rules the generator goes on to
cite, more often than relevance does in two of three replicates. The rules are read, named as
used, and accuracy does not move. Selecting *better* rules to show is therefore not the repair.

**It is not that the curator was blind to the structure.** `render_for_curator()` is an unfiltered
render, so the curator already reads every co-citation relation line plus its legend on every
call. The write-path version of this idea is already inside this null. It was also checked whether
co-citation communities are semantic duplicate families the curator could consolidate: they are
not. **Zero co-cited pairs reach the 0.88 de-duplication threshold** in any of the three runs, 0
of 18, 0 of 12 and 0 of 152. Connected rules *are* more alike than random ones, mean cosine 0.604,
0.642 and 0.416 within a community against 0.435, 0.418 and 0.391 over all live pairs, so the
edges are not joining unrelated rules. But none of them comes close to the threshold at which
grow-and-refine would merge two rules, which is what "not a duplicate family" has to mean here.
Recompute with `analyze_formula_disclosures.py`. An earlier version of this paragraph quoted
0.491 and 0.449 for the within-community figure and called it "barely above" the baseline; the
scripted measurement is higher than that and the "barely" does not survive it, though the
conclusion rests on the threshold count rather than on the gap.

## 7. Interpretation

**What this experiment licenses.**

- **Memory helps, and the effect is large and reproducible.** +8.67pp for linear memory and
  +10.83pp for relational, both with CIs excluding zero and both surviving Holm correction over
  the three pre-registered comparisons, on a task where the formula definition is
  stripped from the prompt so accumulated rules are directly reusable.
- **Relational structure was not shown to beat linear memory**, now on the strongest relation
  available to it. The earlier null could be blamed on `related_to` being derivable from rule
  text. This one cannot: `co_cited` is irreducible, was verified predictive on held-out data, and
  changed 14% to 17% of the answers.
- **The bottleneck is not what the generator reads.** Both loci where structure could act, the
  read path and the curator's view, were instrumented and neither moves accuracy. The natural next
  move after a retrieval null is a fourth edge predicate; the measurements say that would be the
  wrong move.

**What it does not license.**

- **Not "structure cannot help".** 50 clusters give a ~4pp floor and `formula_test.jsonl` has no
  more formulas in it, so effects below that are invisible here by construction. The dose-response
  flatness in §5 is suggestive, not a proof of zero.
- **Not a decomposition of `none → cocite`.** The §4a conditional (a plain `graph` arm, giving
  `graph → cocite`) fires only on a significant primary. It did not fire, so the 10.83pp is
  attributable to the treatment *as a whole*: graph storage, the walk, `refines` eviction,
  annotation and co-citation together, not to the pairwise relation alone.
- **Not a claim about causation from citations.** `bullet_ids` is the model's self-report. The
  held-out lift proves it carries real information; "the model says it used rule X" is still not
  "rule X caused the answer".

**Scope.** This needs itemised memory with stable ids and a generator that self-reports which
items it used. Both are ACE design properties, so the result generalises across ACE tasks, but not
to agent loops lacking either.

## 8. Disclosures

**The replicates.** `--seed` is inert: it reaches only Ollama's sampler, and at temperature 0 the
sampler is greedy argmax and never reads it. Run-to-run divergence tracks the state the Ollama
server was in when it loaded the model, which fixes layer placement and therefore the order of
floating-point reductions, so a near-tied argmax flips. The ACE loop then amplifies one flipped
token roughly 37×: a changed reflection becomes a changed curator operation, which becomes the
playbook for every remaining step.

Consequence, found and handled during this experiment: **the first `cocite` seed-2 run turned out
to be byte-identical to seed 1** (the same 458 curator operations, differing only in timestamps).
The design pre-registered three replicates; three were run; only afterwards did two of them turn
out to be the same run. The server was restarted to force a distinct state, verified with a
200-call probe, and the seed-2 arm was re-run. The duplicate is archived under
`results/formula/cocite_experiment/_determinism/`. The primary was null under both the two-replicate
and the three-replicate pooling (+3.25pp p=0.1249 and +2.17pp p=0.3344), so nothing about the
conclusion depends on this repair.

**The control arm, in full.** Its three runs were executed for the earlier three-arm experiment,
on older commits, and reused here. The licence is that a `none` arm accumulates nothing and makes
no curator or reflector calls at all, so none of the intervening run-path changes (graph memory,
co-citation plumbing, the curator sanitiser, two dead-code removals) can reach it. That was
verified by reading the diff rather than asserted.

The determinism phenomenon above affects it too: `none` seeds 1 and 2 are byte-identical, and a
post-restart probe reproduced seed 0 exactly, so the arm has three runs but only two distinct
outcomes.

**A fourth `none` run was made on 2026-08-24 to try to obtain a third distinct state, and it did
not produce one**: it came back byte-identical to seed 0, matching `md5` on the prediction files.
Five `none` runs now exist and land on exactly two outcomes, 0.6250 and 0.6200, in the ratio 3:2.
That is a further reading on the determinism finding above rather than a new replicate, and it is
kept at `results/formula/preregistered/seed2/none_rerun/`.

The pooled control stays on seeds 0/1/2 and nothing turns on that choice: swapping the re-run in
place of the seed-2 duplicate moves `none → text` to +8.50pp and `none → cocite` to +10.67pp, both
still significant under Holm, and leaves the primary `text → cocite` untouched at +2.17pp because
it does not involve this arm at all. It is why an across-seed sd here should be read as a count of
how many server states the runs happened to straddle rather than as a sampling distribution. Both
significant results rest on this arm, so all of it is stated rather than summarised.

**Extraction asymmetry.** The arms lean on the JSON escape-repair fallback by different amounts:
43.5% to 51.5% of answers across the six runs, with the seed-2 `cocite` run highest at 51.5%. The
fallback runs only after a strict parse has already failed and cannot alter a well-formed
response, but it is either a real effect of the memory on output shape or a scoring asymmetry, and
these data cannot distinguish them. `no_answer` totals are 0 in every `text` run and 1, 1, 0 in
the `cocite` runs.

**A second arm difference, disclosed and unavoidable.** The curator prompt is identical in all
three arms, including the `refines` field that only a graph arm acts on; the `text` arm parses it
and drops it. Holding the prompt identical is what keeps the arms from differing in two places at
the prompt level, and the cost is that they differ in eviction capability on top of retrieval. The
baseline is honestly described as "ACE with an extended curator schema whose extra field it
ignores".

**Post-hoc status.** This experiment was designed on 2026-08-17, after the pre-registered null was
analysed. Designing an arm after seeing a null is a researcher degree of freedom and no
justification makes it not one. What can be claimed is that it was pre-registered before its own
runs, that its gates were pinned from held-out *training* lift and never from a test row, and that
the one mid-experiment parameter change (the rate gate from 0.30 to 0.20) was triggered by the
mechanism diagnostic with that run's accuracy file unopened, to a value already named as the
fallback in the pre-registration.

**Other carried-over deviations.** `num_epochs 1` against the paper's 5, identical in all arms.
Circularity in co-citation is real but bounded: a pair can only be co-cited if co-rendered and
co-rendering is driven by the edges, but **43% to 77% of the live pair space is co-rendered at
least once** with no exploration mechanism at all (1,676 of 2,346, 4,048 of 5,253 and 4,631 of
10,731 at each run's tested checkpoint), because question variety does the work and the walk's
seeds are always relevance-chosen. Two things about that figure. An earlier version of this line
said 41% to 47%, which `analyze_formula_disclosures.py` does not reproduce under any definition of
"gets a chance". And the co-render weight it counts **includes what a refinement inherited from
the rule it superseded** (`graph.py::_inherit_cocitation` copies an ancestor's pair history onto
its heir at full weight), and the store keeps no marker separating the two, so 43% to 77% is an
upper bound on pairs genuinely shown together rather than a count of them. It bounds the
circularity in the conservative direction either way: the more of that space question variety
reaches on its own, the less the edges can be said to have manufactured their own evidence.

## 9. Cost

| arm | calls | tokens | LLM hours | memory at the tested checkpoint |
|---|---|---|---|---|
| `text` | 4,650 | 22.56M | 13.86 | 12,258 / 12,433 / 12,976 tokens |
| `cocite` | 4,658 | 22.51M | 13.61 | 10,146 / 8,769 / 13,410 tokens |

**The treatment is free.** The relational arm ran 0.2% cheaper in tokens for +2.17pp of
unresolvable accuracy. Serving throughput was stable in all six runs (95 to 105 tok/s first to
last), so no timing figure is contaminated by the GPU-placement failure mode that affected earlier
runs.

## 10. Reproducing these figures

```powershell
uv run python compare_arms.py --selftest
uv run python compare_arms.py --force (Get-ChildItem 'results\formula\cocite_experiment\seed*\*\ace_run_*' -Directory | Select-Object -ExpandProperty FullName) results\formula\preregistered\seed0\none\ace_run_* results\formula\preregistered\seed1\none\ace_run_* results\formula\preregistered\seed2\none\ace_run_*
uv run python analyze_walk_share.py results\formula\cocite_experiment\seed0\cocite\ace_run_*
uv run python calibrate_cocite_gate.py (Get-ChildItem 'results\formula\cocite_experiment\seed*\cocite\ace_run_*' -Directory | Select-Object -ExpandProperty FullName)
uv run python aggregate_run_costs.py (Get-ChildItem 'results\formula\cocite_experiment\seed*\*\ace_run_*' -Directory | Select-Object -ExpandProperty FullName)
uv run python analyze_cocite_structure.py (Get-ChildItem 'results\formula\cocite_experiment\seed*\cocite\ace_run_*' -Directory | Select-Object -ExpandProperty FullName)
uv run python analyze_replicate_divergence.py (Get-ChildItem 'results\formula\cocite_experiment\seed*\*\ace_run_*' -Directory | Select-Object -ExpandProperty FullName)
```

**The disclosures in sections 6 and 8 come from one further call**, which takes both run sets because the m=6 family spans two experiments and this experiment's control arm is the earlier set's, reused:

```powershell
uv run python analyze_formula_disclosures.py (Get-ChildItem 'results\formula\cocite_experiment\seed*\*\ace_run_*','results\formula\preregistered\seed*\none\ace_run_*' -Directory | Select-Object -ExpandProperty FullName) --family6 (Get-ChildItem 'results\formula\preregistered\seed*\*\ace_run_*' -Directory | Select-Object -ExpandProperty FullName)
```

**The extraction asymmetry needs no call of its own**: `compare_arms.py` already prints it, as "share of answers NOT read by a strict parse". Its version bounds the population to the indices actually scored, which matters here because validation writes 300 calls under the same `test_eval_<i>` prefix the 200 test calls use.

The second command prints the §3.1 m=3 multiplicity block. **The m=6 sensitivity cannot come from
the same call**: `compare_arms.py` labels arms by `memory_kind`, so this experiment's `cocite`
runs and the earlier experiment's `graph` runs collide under one label and would be pooled as six
replicates of one arm rather than compared as two experiments. It is computed instead by calling
the two public functions `holm()` and `westfall_young()` on the six comparisons' block-difference
vectors, which `block_diffs()` returns for any pair of pooled score maps. About ten lines, no new
machinery.

For the §8 robustness row, re-run the second command with
`results\formula\preregistered\seed2\none_rerun\ace_run_*` in place of the seed-2 `none`
path.

The §4.1 reference point:

```powershell
0,1,2 | ForEach-Object { uv run python run_raw_baseline.py --task_name formula --seed $_ `
    --save_path results\formula\raw_baseline\seed$_ }
```

Two numbers in this file therefore come from different resamplings of the same test: the pairwise
and pooled blocks use 10,000 sign-flip draws, and the multiplicity block uses 50,000 because an
adjusted p near 0.05 cannot be resolved at the coarser count. The tool prints that note itself.

`walk_only_share` must be read from `analyze_walk_share.py`, never from `structural_stats()`. The
counter in the run artifacts is a run-cumulative average diluted by every early step where the
playbook still fitted inside the budget, and it is written at eval checkpoints only, so the final
test contributes nothing to it. The recomputed and recorded values differ by roughly a third
(0.2061 against 0.1656 on seed 0).
