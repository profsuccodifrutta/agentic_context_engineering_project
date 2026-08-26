# Pre-registration: relational vs linear memory on `finer`

**Status: no arm has run.** Written 2026-08-25, before the probes.

---

## 1. Question

An ACE playbook is a list of rules. A linear memory can record facts about each rule on its own:
how often it helped, whether it is superseded. A relational memory can also record facts about
**pairs** of rules, and retrieval can follow those pairs outward from whatever relevance ranking
already selected.

**Does the pairwise representation produce better answers than the per-item one, holding
everything else identical?**

No direction is declared in advance. Both outcomes are informative and the design is built so
that either can be read.

## 2. The task

`finer` is XBRL tagging. Each row shows the model a fixed list of 140 US GAAP tags and asks four
questions of the form *"What is the best tag for entity X in sentence S?"*. The answer is four
comma-separated tags, one per question, in order.

Three properties of the data decide whether this experiment can work at all. All three were
measured before any run.

**The four questions in a row come from the same filing section.** Word overlap between the
sentences inside one row is 0.350 by Jaccard, against 0.038 for two sentences picked at random:
**9.2 times higher**. Rows are coherent; a row about a revolving credit facility asks four
questions about that facility.

**All four are answered in one generator call.** So whatever rules the model cites for question 1
and question 3 are cited together, in the same call, by construction. Test rows carry 2.71
distinct tags on average and 93.2% carry two or more, which is the raw material a pairwise
relation needs.

**Which tags go together is not predictable from their names.** This is the property that matters
most, because the memory stores rules and ranks them by embedding similarity. Over the 1000-row
training pool, the correlation between the embedding cosine of two tag names and how often those
tags actually co-occur is **Spearman +0.026** (Pearson +0.248), and only **47 of the 200
most-co-occurring pairs (23.5%)** appear in the top 200 by name similarity.
`StockRepurchaseProgramAuthorizedAmount` and `TreasuryStockSharesAcquired` co-occur at **36 times
base rate** while sitting at cosine 0.537.

So the association is real, dense, and invisible to the similarity ranking the baseline already
uses. A relational memory can hold it; a flat list plus an embedding index cannot reconstruct it.

## 3. Arms

Three arms, identical in every respect except the memory.

| arm | flags | mode | what it isolates |
|---|---|---|---|
| control | `--memory none --retrieval none` | `eval_only` | the accumulated content, by having none |
| baseline | `--memory text --retrieval flat` | `offline` | a linear memory with relevance retrieval |
| treatment | `--memory graph --retrieval graph --cocite` | `offline` | a memory that can also represent pairs |

Both memory arms score every live rule against the question and render the highest-scoring that
fit one shared token budget. The treatment then walks co-citation edges outward from those same
seeds, so it can surface a rule relevance alone would not have reached. **That walk is the entire
difference between the two arms.**

A co-citation edge connects two rules the generator itself named together in `bullet_ids`. It is
gated on a rate rather than a count: the pair must have been rendered together at least
`--cocite_min_opportunity` times and cited together on at least `--cocite_min_rate` of those.

**The control is not a bare LLM.** It carries the identical generator prompt with `(empty)` in the
playbook slot, so control-to-baseline isolates the accumulated content and holds prompt structure
fixed.

**Three mechanisms are deliberately off**, each because leaving it on would make the arms differ in
more than one place:

- **`refines`** (`--no_curator_refines`), the curator declaring one rule dead. It is a fact about a
  single rule and a flat list reproduces it with a `supersedes` field, so it is not the capability
  under test. The curator prompt is unchanged and still asks for the field; all three arms parse it
  and drop it.
- **`related_to`**, similarity edges between rule texts. §2 measured that similarity does not
  predict association here, so these edges would point where relevance already looks.
- **`contradicts`**, the NLI cross-encoder. It scores prescriptive advice as contradictory at
  saturation and does not work on this material. The treatment is therefore weaker than the full
  design, and that is disclosed rather than fixed.

De-duplication is on in both memory arms under one shared decision rule, so it cannot favour
either. It compares rule text by embedding similarity, which prunes redundancy but not staleness;
with `refines` off, neither arm can retire a rule superseded by something textually dissimilar.
That limitation is symmetric.

## 4. Data

Built by `make_finer_splits.py`, seed 42, deterministic. **Do not regenerate.** `--check`
re-derives and audits without writing.

| split | rows | tag decisions |
|---|---|---|
| train | 200 | 800 |
| val | 150 | 600 |
| **test** | **441** | **1764** |
| probe train / val / test | 50 / 20 / 20 | |

**The test split is the authors' own subset, unchanged.** Nothing is sampled from it, including
the probe.

**Train is 200 rows**, the first 200 of the audited 250-row split, so the stream a run sees is a
prefix of the stream the split builder produced. At that size 94 of the 137 test tags have three
or more training examples, backing 78.9% of test tag instances.

The length is set by the curator, not by coverage. The curator reads the whole memory with no
budget, so its prompt grows for the entire run, and the probes measured that growth at 100 tokens
per step in the baseline and 118 in the treatment. The curator was then measured against the live
endpoint: it answers a 26,860-token prompt and returns nothing at all at 30,896. At 250 steps the
treatment projects to 27,157 tokens and would curate past that point, silently, in the last part
of training and in one arm only. At 200 steps it projects to 21,997. The 43 tags this costs are
the price of a curator that is still working when the run ends. Full numbers in §9.

**Val is 150 rows**, drawn only from rows that share no sentence with the training pool.
Validation selects the checkpoint the final test runs on, so a val row the model trained on would
bias that choice.

**Leakage is zero.** The builder checks every pair of splits and refuses to write if any sentence
is shared. One overlap is deliberate and exempt: the probe's training rows are the first 50 of the
real training split, so the probe sees a prefix of the exact stream the real run will see.

## 5. Metrics and analysis

### Two metrics, and which is primary

One row is four tagging decisions bundled into one call, so there are two ways to score it, and
the harness already computes both:

| metric | definition | where it appears |
|---|---|---|
| **per-tag** | fraction of a row's four tags that are right | `final_results.json -> accuracy`; selects the checkpoint at `ace/ace.py:1012` |
| **all-four** | the row counts only if all four match | `correct/total`, and `is_correct` in the predictions file |

A finished run prints `Final Accuracy: 0.700 (52/441)` with both numbers correct and meaning
different things. This is written down so no report mixes them.

**Primary: per-tag.** Three reasons, none of them about which answer it gives:

1. The hypothesis is that a co-citation edge surfaces a companion tag's rule, which lifts a row
   from 2-of-4 to 3-of-4. All-four scores that at zero, so it is blind to the effect being tested.
2. The harness already selects checkpoints on per-tag. Making it primary keeps selection and
   evaluation on one scale.
3. All-four is roughly the fourth power of per-tag: at 70% per tag it is about 24%, at 50% about
   6%. A metric compressed against a floor cannot show a gap between arms.

**Disclosed against it:** per-tag is more sensitive than all-four, so it makes a positive easier to
obtain. It is chosen here, before any run, for the reasons above, and **all-four is reported
beside every primary figure and never omitted.** `compare_arms.py` prints a banner if it is run on
`finer` without `--metric finer_per_tag`, so the pair cannot be reported by halves.

### The unit of resampling is the row

**441 blocks, one per test row.** A row's score is its tag fraction. Aggregating its four decisions
into one number before any resampling is what handles the within-call correlation, and since every
row carries exactly four tags, the mean of those fractions is exactly per-tag accuracy.

Rows are independent of one another, measured on the test file: word overlap between adjacent rows
is 1.01 times that of random rows, and 11 of 97,020 row pairs share a sentence. Rows come from
different filings.

**Tags are not a second blocking dimension.** The 140 tags are the fixed answer vocabulary printed
in every prompt, identical in every row. They are the task, not a sample drawn from it. The
question being asked is "how would these arms do on new disclosure sentences", and under that
question tags are fixed and rows are what varies.

**Scope limit, stated in advance:** this licenses no claim about tags outside the 140. A reader who
wants that needs the same scores resampled by tag, which the saved predictions support and which
will be reported as a sensitivity check if the primary is significant. If the two disagree, the
more conservative governs.

### Tests

**Primary test:** the cluster sign-flip permutation test in `compare_arms.py::cluster_paired_stats`
with its cluster bootstrap confidence interval, over the 441 rows. Paired by row index, alpha 0.05.

McNemar is not printed on the primary metric, because its `b` and `c` count discordant *binary*
pairs and these scores are fractional. It is printed on the all-four secondary, where it is valid.

**Pooling across replicates.** Each row is scored as its mean across an arm's three runs, and those
real-valued rows go through the same test. Pooling removes run-to-run noise and does nothing to
row-sampling variance. **The pooled analysis is 441 rows, not 1323**, and `sd/sqrt(3)` is never a
standard error. `pooled_arm_scores` refuses unequal replicate counts.

**Family and correction.** Three comparisons on the primary metric, declared here: treatment vs
baseline (primary), baseline vs control, treatment vs control. Reported with **Holm** adjusted
p-values and **Westfall-Young** alongside. The all-four secondary covers the same three comparisons
and is reported uncorrected and labelled as such; no decision rests on it alone.

**Power** is computed from the probe and written into §9 before any arm runs, together with the
observed within-row correlation. Stating a resolvable effect before that correlation is known
would be a guess.

**Decision rule.** The treatment beats the baseline only if the pooled primary test is significant
**and** the mean gap across replicates exceeds the within-arm standard deviation. A gap smaller
than the replicate spread is noise regardless of p. The same standard applies to the other two
comparisons.

**Replicates must be distinct.** `--seed` reaches only the sampler, and at temperature 0 the
sampler never reads it, so two runs served under the same server state come out byte-identical.
Before pooling, `md5sum` every arm's prediction file. A duplicate hash is not a replicate: restart
the server and re-run. Pooling duplicates shrinks the within-arm standard deviation, which makes
the decision rule above fire more easily.

## 6. Configuration

All three roles are `gemma4:e2b`. `--think_generator off`, `--num_ctx 32768`, `--max_tokens 16384`,
`--num_epochs 1`, `--max_num_rounds 3`, `--curator_frequency 1`, `--test_workers 1`,
`--eval_steps 100`, `--retrieval_budget 2500`, `--retrieval_seeds 5`, `--dedup`,
`--dedup_threshold 0.86`, `--no_curator_refines`, and in the treatment
`--cocite_min_opportunity 3 --cocite_min_rate 0.20`. Every one of these is identical in all three
arms and audited by `compare_arms.py` and `aggregate_run_costs.py`.

**`--dedup_threshold` and `--cocite_min_rate` were derived from the probes**, not carried over
from anywhere: both are thresholds on quantities whose scale depends on what the rules look like,
and here the rules name US GAAP tags. The derivation is §9.

**The prompt budget was measured, not assumed.** A `finer` question is the constant 140-tag list
plus four sentences, and the probes put the whole generator prompt at 4,546 tokens in the
baseline and 4,721 in the treatment at the median, against a served window near 16,384. That
leaves about 11,600 tokens for generation, so the generator has room and the residual risk there
is runaway reasoning rather than prompt size, which is what `--think_generator off` addresses.

The curator is the constraint. It reads the whole memory with no budget, so its prompt grows for
the whole run, and it fails silently: over the wall the model spends its output budget on hidden
reasoning and returns an empty response, which drops one entire operation list without an error.
The ladder was measured against the live endpoint at 26,860 tokens working and 30,896 failing, and
the run length in §4 is set so that the treatment's projected 21,997 tokens at the last step keeps
an 18% margin under the lower of those.

## 7. What makes a run invalid

**No accuracy is compared until the treatment is shown to have fired.** These are reported per
replicate, ahead of any accuracy table.

| gate | requirement |
|---|---|
| **rules cited per generation** | mean **at least 2.0**, and at least 2 in **60%** of generations |
| live co-citation edges | not near zero |
| `cocite_max_degree` | must not approach the live node count, which is a hub rather than a graph |
| `walk_only_share` | **at least 0.05** on the final-test pooled share, from `analyze_walk_share.py` |
| held-out predictive lift of the shipped gate | above 1.0x base rate, on generations the gate was not built from |
| the memory outgrows the budget | rendered rules per call must be fewer than live rules, or retrieval selects nothing and the walk has nothing to add |
| replicate distinctness | prediction-file hashes distinct within every arm (§5) |
| serving placement | first ten `detailed_llm_logs/*.json` at **20 tok/s or more**, or the model is running on CPU |
| learning signal | `check_learning_signal.py` [1] to [6] pass; train accuracy inside 0.20 to 0.85, read on per-tag |
| curator prompt at the last step | below the ceiling measured by `check_curator_ceiling.py`, read off the run's own `prompt_num_tokens` |
| edges reach a growing share of the memory | the nodes carrying any co-citation edge must grow with the playbook, not stay at a fixed core |

**The citation gate is the one that stops the experiment.** The graph connects *rules*, not tags.
If the curator writes one rule per tag *family* rather than per tag, one citation covers four tags,
`bullet_ids` has length 1, and no pair can ever form. The data's own ceiling is 2.71 distinct tags
per row with 93.2% of rows carrying two or more, so a mean of 2.0 is a real bar. Below it, no arm
runs.

**The last gate is about shape, not size, and the probe is the reason it is here.** At 50 training
steps the co-citation edges covered 16 of the live rules and half of all pairs among those 16, a
dense core with a thin fringe rather than a sparse structure. That passes the degree gate, because
13 is nowhere near the live node count. But if the core stays that size while the playbook grows
to two hundred rules, the walk stops being query dependent: every seed that lands in the core
pulls in the same rules, and the treatment reduces to a fixed extra block of text that the
baseline does not get. Read the connected-node count against the live rule count at both
checkpoints. `walk_only_share` cannot see this, because a fixed block is still a block relevance
would not have chosen.

**`walk_only_share` is necessary but not sufficient.** It measures that the walk rendered rules
relevance would not have selected. A randomly rewired graph scores well on it, because the metric
rewards pointing away from relevance and arbitrary edges do that best. Clearing the floor shows the
arms read different rules and nothing more; held-out citation lift is what validates the edges.

**The relevance threat, declared in advance.** Because the four sentences in a row share a theme,
the question text is internally coherent, and relevance may already surface the companion rules on
its own. That would make the walk redundant. The argument runs both ways: relevance scores each
rule against a question mixing up to four themes, which blurs the ranking, while the walk starts
from seeds that landed on one theme and can pull in that theme's specific companions.
`walk_only_share` and the walk-added citation ratio measure it, and **either outcome is reported.**
This is written here so a null explained this way afterwards is not a post-hoc story.

## 8. Procedure

Nothing in steps 1 to 3 may be revisited once step 5 begins.

1. ~~Splits, configs, wiring, metrics, gates~~ **done 2026-08-25.**
2. ~~**Probe pair**, at provisional gates, on `probe_config.json`. Read §7 before any accuracy.~~
   **done 2026-08-26**, both arms, §7 read before any accuracy table.
3. ~~**Calibrate** `--dedup_threshold` and `--cocite_min_rate`, compute the within-row correlation
   and the resolvable effect, and write all four into §9.~~ **done 2026-08-26**, all four below.
4. Preflight: `check_memory_equivalence.py`, `check_prompts.py`, `check_finance_wiring.py`,
   `compare_arms.py --selftest`.
5. **Seed 0, three arms.** Stop and inspect against §7.
6. **Seeds 1 and 2.** Nothing changes except `--seed`; hash each prediction file before counting it
   as a replicate.
7. **Analyse.** `compare_arms.py --metric finer_per_tag` as the headline, `--metric exact` beside
   it, plus `aggregate_run_costs.py` and `analyze_walk_share.py`.

## 9. Deviations and amendment log

Deviations from the ACE authors' setup, all identical across arms:

- **`num_epochs = 1`** against the paper's five. Five multiplies every training call by five.
- **Train downsampled to 200 rows of 1000, val to 150 of 500.** Test unchanged (§4).
- **The curator schema carries a `refines` field the authors' does not**, applied identically in
  every arm and acted on by none of them here (§3).
- **`answer_is_correct` no longer calls bare `eval()` on model output.** The conversion never fires
  on tag names, so `ast.literal_eval` is behaviour-identical on this data and removes an execution
  path.
- **The prediction is normalised before comparison** (`normalize_finer_answer`): a doubled-quote
  wrapper and a JSON array of tags are unwrapped. Both survive `extract_answer` intact and both
  scored a correct answer wrong. Applied to the prediction only.
- **The reflector is told which of the four tags was wrong**, not only that the row was. The
  information was already computed and discarded. This changes the curator's inputs, not its
  instructions.

Append amendments below, dated. Anything added after step 5 begins must say so.

- *(2026-08-25)* Document created, before any probe. Splits built; the two metrics documented and
  per-tag declared primary with its disclosure; the row declared the resampling unit with its
  scope limit; the citation-granularity gate and the relevance threat declared with numeric
  thresholds. Code: `finer_tag_counts` and `normalize_finer_answer` in
  `eval/finance/data_processor.py`, per-tag reflector feedback, `compare_arms.py
  --metric finer_per_tag` with a self-test and a banner, `check_finance_wiring.py` [1i] to [1l]
  and [2c].

- *(2026-08-26)* **Probes read and the four step-3 numbers pinned.** Both arms ran 50 training
  steps on `probe_config.json` at the provisional gates, on GPU throughout (104 tok/s, flat).

  **`--dedup_threshold 0.86.`** The sweep replays each arm's recorded curator operations offline
  at every candidate:

  | threshold | baseline absorbed | live | treatment absorbed | live |
  |---|---|---|---|---|
  | 0.80 | 46.3% | 51 | 66.2% | 26 |
  | 0.84 | 40.4% | 56 | 49.3% | 36 |
  | **0.86** | **35.5%** | **60** | **40.3%** | **43** |
  | 0.88 | 34.4% | 61 | 37.5% | 43 |
  | 0.90 | 24.1% | 66 | 25.7% | 49 |

  0.86 is the highest threshold that still collapses the dominant restatement family. The curator
  writes the same rule with its two halves swapped, over and over: "Input vs. Outcome Distinction"
  and "Outcome vs. Input Distinction" appear four times between them, and 0.88 leaves three of
  those four standing at cosine 0.877 to 0.879. The pairs 0.86 declines to merge sit at 0.848 to
  0.860 and are genuinely different rules on inspection. Both arms stay under the 50% absorption
  guard and the gap between them, 4.8 points, is the smallest of any threshold that prunes the
  family.

  **`--cocite_min_rate 0.20`, unchanged.** Edges built on the first half of the probe's
  generations predict co-citation on the unseen second half at **5.48 times base rate**. The gate
  is nearly flat here: every value from 0.15 to 0.50 yields 23 to 31 edges at 5.48 to 5.72 times
  base rate, because a row that co-renders a pair usually cites both. 0.20 is kept for coverage,
  and confidence beats a lift ranking at matched edge count (5.92x against 3.64x at k=10).

  **Within-row correlation, ICC = 0.33.** Measured over 240 generations, and 0.40 over the 40 test
  rows alone. That is the correlation §5 absorbs by scoring a row as its tag fraction before any
  resampling, and it is why a tag-level interval would have overstated precision.

  **Resolvable effect: about 4.7 points.** The two probe arms scored the same 20 test rows, giving
  a paired per-row difference with sd 0.354. Over 441 test rows that is a minimum detectable
  effect of 4.7 points at 80% power and 5.5 at 90%. The gap must also exceed the within-arm
  replicate spread (§5), which is not yet known.

  **Run length cut from 250 training rows to 200**, and `--eval_steps` from 125 to 100 to keep two
  checkpoints. The reason is the curator ceiling, measured against the live endpoint rather than
  assumed: a 26,860-token curator prompt returns usable operations and a 30,896-token one returns
  nothing at all, spending its output budget on hidden reasoning. The probes put curator prompt
  growth at 100 tokens per step in the baseline and 118 in the treatment, so at 250 steps the
  treatment projects to 27,157 tokens and would cross the wall in the last part of training, in
  one arm only and with no error. At 200 steps the projections are 20,083 and 21,997. The
  alternative considered was keeping 250 rows at `--dedup_threshold 0.84`, which also clears the
  ceiling; it was rejected because 0.84 puts the treatment at 49.3% absorption against the 50%
  guard, with absorption still rising as the playbook grows. The cost is tag coverage, from 110 of
  137 test tags with three or more training examples down to 94, and from 87.5% of test tag
  instances backed down to 78.9%. The §6 projection of "about 14,600 tokens at step 250" that this
  replaces was an estimate made before any run and was low by roughly half.

  **Two tooling defects found by the probes and fixed before any arm.** `check_learning_signal.py`
  check [3] judged headroom on `is_correct`, which on `finer` is all-four-exact: it read 0.100 and
  0.060 and declared the signal dead on two healthy runs whose per-tag accuracy was 0.355 and
  0.270. It now bands on per-tag and prints both. And `compare_arms.py` printed no paired test at
  all for a fractional metric on a task with no block structure, which is exactly the state of
  step 5 with one replicate per arm; it now falls back to one block per row, the same fallback
  `report_pooled` already used, which `--selftest` asserts reproduces McNemar exact on binary
  scores.
