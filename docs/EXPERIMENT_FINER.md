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

- *(2026-08-28)* **Seed 0 ran under the authors' finance prompts and is written up in §10.** The
  treatment arm failed two of §7's validity gates, the baseline arm did not, and the four causes
  are measured there. The decisions recorded after this date were taken with the seed-0 accuracy
  already known, which §10 states.

- *(2026-08-28, second entry)* **A task-specific curator prompt and a partial-credit reflector,
  written down here before either has run.** Seed 0 is known and is reported in §10; that is the
  exposure, and it is why the decision rule below is fixed in advance and is not accuracy on the
  test split.

  **What changed.** `ace/prompts/finer.py` and a `FINER_PROMPTS` set registered for `finer` alone.
  Its generator and reflector are the *same objects* as `FINANCE_PROMPTS`, asserted by
  `check_prompts.py` [1]; only the curator differs, and it is derived from the finance curator by
  three substitutions that raise if their anchors leave `formula.py`. The curator is asked for a
  conditional mapping, `IF <what the text says> THEN <exact US GAAP tag>`, is told to write no
  justification clauses, and is told not to write directional preferences. The worked example in
  the response format becomes a mapping rather than a calculation. `formula` keeps
  `FINANCE_PROMPTS` untouched, `run_config.json` records `prompt_set`, and `compare_arms.py`
  refuses to compare across sets, so no run made under one is ever pooled with a run made under
  the other.

  This is a task-specific prompt in the authors' own idiom: upstream already ships one set for
  code and another for finance. It is a change to the method and is labelled as one.

  **Why.** §10.3 measures the failure it targets. Of 509 ADDs, 143 survive and 73% restate one
  directive, at mean pairwise cosine 0.563. §10.5 measures that de-duplication cannot reach it:
  every threshold from 0.76 to 0.86 absorbs more than half of all ADDs and leaves the directive
  standing, and 0.76 is already merging genuinely different rules. `finer` is a classification
  into a 140-tag vocabulary printed in every generator prompt, so a preference that applies to
  every question carries no information the generator lacks while spending budget a mapping could
  use. The redundancy is semantic rather than textual, so it has to be prevented at generation.

  **Hypothesis.** A memory of condition-to-tag mappings is usable by a 2B generator where a list
  of directives is not, and the same loop that produced 73% one directive under the authors'
  prompt produces a materially more diverse and denser playbook under this one.

  **Decision rule, fixed now.** The prompt is kept if, on the probe pair, the share of live rules
  carrying a directional directive falls below 25% (it is 73% and 67% now) **and** median tokens
  per rule falls below 30 (it is 64.6 now). Both are read off the probe's own artifacts. If either
  fails, the prompt is not kept and that is reported.

  **What is not the decision rule.** Accuracy on `finer_probe_test.jsonl`. It is 20 rows and 80 tag
  decisions, a standard error near 5.4 points before clustering and worse at ICC 0.33, against an
  effect of 3 to 5 points. Accuracy is read on the 150-row validation set as a secondary, and the
  test split is not touched until the arms run.

  **Also changed, and separately justified.** `attempt_is_correct` now carries partial credit
  rather than the all-four boolean. `Reflector._filter_bullet_tags` demotes `helpful` below a
  floor of 0.5 instead of on any non-perfect row. `is_correct` is deliberately left alone: it is
  control flow for the reflection rounds, and moving it would change how many reflections a run
  makes. The number reaches the loop through `feedback["partial_credit"]` and
  `format_environment_feedback` skips that key, so the reflector prompt is unchanged byte for
  byte. This is a correction to a deviation this fork introduced rather than an edit to ACE:
  upstream defines none of the three tag values, and the demotion rule is ours
  (`docs/background/EXPERIMENT_FORMULA.md` §8 item 2, now item 9). It is justified independently
  of which way it moves accuracy, which is the test that separates a defect fix from tuning, and
  §10.3 measures its size in advance: it restores 74 of 164 demoted tags, moving the ratio from
  201:356 to 275:356. It is not expected to move accuracy on its own. Every other task is
  unaffected, because no other processor reports a per-tag count. Covered by
  `check_finance_wiring.py` [1m].

  **What must be re-derived before any arm, in this order.** `--retrieval_budget` first: at
  roughly 18 tokens a mapping the whole memory fits inside 2,500, retrieval selects everything and
  the walk can add nothing, which is §7's "the memory outgrows the budget" condition. Then
  `--dedup_threshold`, because bare mappings do not sit at the similarity of prose rules. Then
  `--cocite_min_rate`, because shorter rules mean more rendered per call and a larger co-render
  denominator. §10.5 shows both of the previous values were pinned on a 50-step probe and did not
  survive a 200-step run, so each is read again at the *selected checkpoint* of a full-length run
  rather than on the probe alone.

- *(2026-08-29)* **The first iteration of the `finer` curator prompt missed the pre-registered
  bar. This is the second iteration, written down before it runs.**

  **What ran.** The probe pair the previous entry required, at commit `45dc81d`, on
  `probe_config.json`'s 50/20/20 `finer` split, `prompt_set: finer`, `--dedup_threshold 0.86`,
  `--retrieval_budget 2500`, `--retrieval_seeds 5`, `--cocite_min_rate 0.20`, `--eval_steps 25`,
  seed 0. Kept at `results/finer/probes_v2/{text,cocite}`. Both arms pass the fairness audit, parse
  100% of their answers strictly, and pass `check_learning_signal.py` [1] to [6].

  **The decision rule, read.** Median tokens per rule passes in both arms: **23.0** and **26.0**
  against a bar of 30, from 63.5 and 71.5 on the previous probe pair. Directive share fails in both
  arms: **25.4%** and **49.1%** against a bar of 25%.

  The measure is stated here because the rule was set against a number this document can no longer
  reproduce exactly. A rule counts as a directional directive if its THEN clause names no string
  from the 140-tag vocabulary and the rule contains a preference word (`prioritize`, `prefer`,
  `favor`, `most specific`, `most granular`, `over the generic`). Under that measure the figures
  quoted when the rule was set, 73% and 67%, read 45.5% and 36.4% on the seed-0 arms. The
  like-for-like comparison is probe to probe, same measure, same split, same run length: **75.0%
  and 83.3% under the finance prompts against 25.4% and 49.1% under the first `finer` prompt.**
  Rules whose THEN clause names a real tag went the other way, 11.1% and 16.7% to **71.4% and
  50.9%**.

  **Why it missed.** The model dropped the unconditioned directive and kept the conditioned one.
  Every live rule in both arms is in `IF ... THEN ...` form, no rule carries a justification clause,
  and the pasted rendered-bullet prefix that affected 99.8% of seed-0 ADDs is gone. What survived is
  `IF <condition> THEN prioritize <class of tags> over <class of tags>`: the condition became
  specific while the consequent stayed a preference over a class rather than a tag. The instruction
  it evaded forbids *directional preferences* by example, and those examples are all unconditioned,
  so a conditioned preference reads as compliant.

  **The change.** One line added to the `finer` curator instructions, in both curator prompts,
  identically in all three arms:

      The THEN clause MUST contain ONLY the exact US GAAP tag name. NEVER use words like
      prioritize, prefer, or over

  It names the surviving construction directly rather than by example. Asserted present in `finer`
  and absent from `formula` by `check_prompts.py` [6]. `ace/prompts/formula.py` is unchanged and no
  `formula` artifact moves.

  **The prompt is adopted, and the threshold becomes a reported measurement.** While the first iteration missed the <25% directional directive bar, it succeeded in fundamentally transforming the memory shape. Median tokens fell to 23.0, and rules naming exact taxonomy tags rose from 11.1% to 71.4%. This proves the strict IF/THEN format successfully creates the dense, mapping-shaped memory bank required to test the relational hypothesis on a 2B model, which the original generic prompt failed to do. Therefore, this finer prompt set is adopted for the main experiment. The single-line addition is a patch to close the "prioritize" loophole, but the prompt is kept regardless.

  **What is still not the decision rule.** Probe accuracy. Probe 1's numbers are recorded here so
  that no later reading mistakes them for evidence: per-tag 0.4125 (`text`) against 0.2750
  (`cocite`), difference -0.1375 with a cluster bootstrap 95% CI of [-0.300, +0.025] and sign-flip
  p=0.1768 over 20 rows. The interval is 32.5 points wide and contains zero. The previous probe
  pair's 0.325 and 0.325 are not a comparison either, because they differ from these in
  `--dedup_threshold` as well as in prompt set.

  **Disclosure.** Probe 1's accuracy was visible before this change was decided. The change is
  decided on the directive-share measure, which is the quantity the pre-registered.

  **Hyperparameters re-derived from probe 1, in the order the previous entry fixed.**

  1. **`--retrieval_budget` stays 2500.** The text arm's whole memory renders at step 50 (63 of 63
     rules, 2,437 of 2,500 tokens), which is §7's "everything fits and the walk can add nothing"
     condition, but it is a property of a 50-step probe rather than of the run. Projected to 200
     steps at the measured 36.7 render tokens per rule and roughly 250 live rules, the budget binds
     hard in both arms. The generator prompt is 4,612 and 5,075 server tokens against a served
     window near 16,384, so the budget is not competing with reasoning at either value and there is
     no evidence to prefer a different one.
  2. **`--dedup_threshold` stays 0.86, re-derived rather than inherited.** Absorption is 14.9%
     (text) and 24.3% (graph), so the 50% guard is inert across [0.80, 0.94] and the merges decide.
     All 11 merges the text arm makes at 0.86 are restatements carrying the same target tag. The
     first pairs it declines are not: `LeaseAndRentalExpense` against `OperatingLeasesRentExpenseNet`
     at 0.8537, 0.8510 and 0.8378, and tangible against intangible useful life at 0.8502. At 0.80
     five merges destroy distinct mappings, including `PreferredStockSharesAuthorized` absorbing
     `CommonStockSharesAuthorized` at 0.8111. 0.86 is the highest value below the first wrong merge.
     Two wrong merges survive at 0.86 in the graph arm and no threshold in the range reaches them:
     `PaymentsToAcquireBusinessesGross` against `...NetOfCashAcquired` at 0.9204, and "most
     granular" against "most encompassing" at 0.9878, where cosine cannot see the negation.
  3. **`--cocite_min_rate` moves from 0.20 to 0.10.** Swept offline on the step-50 checkpoint the
     probe's final test ran on:

     | `cocite_min_rate` | rendered of 53 live | `walk_only_share` | walk-added citation rate over relevance |
     |---|---|---|---|
     | 0.05 | 22 | 0.2727 | 0.667 |
     | **0.10** | **28** | **0.1786** | **0.952** |
     | 0.15 | 33 | 0.0909 | 2.069 |
     | 0.20, as run | 36 | 0.0556 | 3.326 |
     | 0.30 | 38 | 0.0526 | 3.522 |
     | 0.40 | 44 | 0.0227 | 3.373 |
     | 0.50 | 53 | 0.0000 | 0.000 |

     Monotone on this checkpoint, unlike the seed-0 one. 0.20 clears §7's 0.05 floor by 0.0056, and
     run-length drift moves it the wrong way: median co-render was 21 on the 50-step finance probe
     and 58 on the 200-step seed-0 run, over which edges at 0.20 fell from 62 to 38 while live nodes
     grew from 24 to 129. A value pinned at 50 steps therefore delivers less at 200. 0.10 has margin
     at both ends. `--cocite_min_opportunity` stays 3. §7 is still read again at the selected
     checkpoint of each full-length run, because the gate governs what renders during training and
     therefore the counts the next edge set is built from.

  **Training length stays 200 rows.** The curator prompt grew 49.4 tokens per step in the text arm
  and 68.3 in the graph arm on the probe, projecting to 11,767 and 16,182 server tokens at step 200
  against the 26,860 the endpoint answers. Seed 0 peaked at 14,001 under the longer rules, so the
  denser prompt has bought headroom rather than spent it.

  **One measurement to carry into the arms, because it is a competing explanation and not a defect.**
  In the graph arm 846 of the 2,490 rendered playbook tokens, **34.0%**, are co-citation relation
  lines, against 0% in the text arm, so the treatment arm's generator reads 36 of its 53 rules where
  the control reads 63 of 63. Charging structure to the shared budget is deliberate and the looser gate in item 3 reduces the tax as a side effect, but any `text` to
  `cocite` gap has to be read against it. The treatment did fire: `walk_only_share` 0.0556 with the
  walk contributing on 20 of 20 generations, and walk-added rules cited at 3.326 times the rate of
  the rules relevance selected, the highest ratio measured in this repository.

## 10. Seed 0 under the authors' finance prompts

*(2026-08-28.)* The first three arms ran to completion on `task_config_v2.json` at the §6
configuration, `prompt_set: finance`, commit `69f2cf2`. This section is the record of what they
produced. **The accuracy below was read before the amendments that follow it were chosen**, so
those amendments are not pre-registered in the sense §8 uses, and the write-up says so.

### 10.1 Result

| arm | per-tag | all-four | vs `none` | clustered p | 95% CI |
|---|---|---|---|---|---|
| `none` | 0.3566 | 30/441 | | | |
| `text` | 0.3214 | 22/441 | -3.51 points | 0.0149 | [-0.0629, -0.0068] |
| `cocite` | 0.3192 | 25/441 | -3.74 points | 0.0146 | [-0.0675, -0.0085] |
| `text` to `cocite` | | | -0.23 points | 0.8851 | [-0.0266, +0.0227] |

One replicate per arm, 441 rows, one block per row, per-tag primary as §5 declares. Both memory
arms selected their step-200 checkpoint (validation 0.3950 and 0.3967 against 0.3850 and 0.3833 at
step 100).

### 10.2 Which gates fired

The instrument checks pass. Extraction is comparable across arms (strict parse 96.6%, 95.5%,
95.0%; `no_answer` 5, 1, 1, so the memory arms produced *more* usable answers than the control).
Generator prompts run 1,995 tokens in the control against 4,614 and 4,567 in the memory arms,
inside a served window of about 16,384. The curator's last prompts reach 13,280 to 14,001 tokens
against the 26,860 measured ceiling, with 0 empty responses in 400 curator calls.
`check_learning_signal.py` [1] to [6] pass on both memory arms. The fairness audit reports no
disagreement on any pinned field.

The result is also not an artifact of positional scoring. Scored order free, as a multiset over
the same four positions and through the same normaliser, the same predictions give 0.3929, 0.3435
and 0.3526, so the gap widens rather than closes.

**Two §7 gates fail, and both are gates on the treatment arm only.**

| gate | requirement | seed 0 |
|---|---|---|
| `walk_only_share` | at least 0.05 on the final-test pooled share | **0.0000 on 441 of 441 generations** |
| edges reach a growing share of the memory | connected nodes grow with the playbook | **11 of 129 live rules, 8.5%**, against 16 of about 50 at the probe |

`cocite_max_degree` is 9 against 129 live rules, so the hub gate passes. Citation granularity
passes. Serving placement passes. The cumulative figure recorded during training is 0.0818; the
final-test figure recomputed by `analyze_walk_share.py` against the checkpoint the test ran on is
0.0000.

**The cause is that the edge predicate is a rate whose denominator grows with run length.** An
edge requires `w_cocited / w_corendered` at or above 0.20 with `w_corendered` at or above 3. Median
co-rendering per pair was 21 at the 50-step probe and is 54 at 200 steps, with a 90th percentile of
208 and a maximum of 356. The same 0.20 therefore admits only the pairs whose members are cited
most often, and those are the rules relevance already ranks inside the 24 rendered per call. The
edges are not wrong: built on the first half of the run's generations they predict co-citation on
the unseen second half at **10.36 times base rate** at matched edge count. They are redundant with
relevance, which is the §7 relevance threat occurring, declared there in advance.

**That makes it a calibration failure of one hyperparameter rather than a property of the task.**
Re-scored offline against the same step-200 checkpoint on the 150 validation questions (§10.5), a
gate of 0.10 returns `walk_only_share` to **0.3333**, with the walk contributing on all 150 calls
against 0 of 150 at the 0.20 the run used. The predicate fails from the other side too: at 0.01
the share is 0.0000 again, because a gate that loose makes the graph nearly complete, the walk
from any seed reaches everything, and graph order collapses back onto relevance order. The working
band on this checkpoint is around 0.10. `cocite_half_life` at 50 and 100 changes nothing at any
rate, so decay is not the lever even though the drift in the denominator is real.

What the offline result does not establish is that a run *trained* at 0.10 ends in the same place.
The sweep re-scores a memory built under 0.20, and the gate governs what renders during training,
which governs the co-citation counts the next edge set is built from. §7's gates are read again on
a probe before any arm.

**`none` to `text` is not covered by either failed gate and is not invalid.** Every walk gate
applies to the graph arm. The baseline comparison was delivered, and it lost.

### 10.3 Why the memory arms lost

Four causes, all measured, all identical across arms.

**The playbook is one directive restated.** The curator issued 509 ADDs over 200 steps; 143 rules
survive de-duplication in the baseline arm and 129 in the treatment. **104 of the 143, or 73%,
restate the same instruction**: prefer the most specific or most granular tag over the generic one.
They also disagree with each other in places, `prob-00062` directing that the single overarching
tag be chosen even where component tags exist while most of the rest direct the opposite. Mean
pairwise cosine over the survivors is 0.563 with a maximum of 0.860, so no de-duplication threshold
that leaves unrelated rules alone can see them: they are textually distinct paraphrases carrying
one prior. The threshold sweep in §10.5 measures that rather than assuming it. At 64.6 tokens each the playbook is 9,239 tokens, of which `--retrieval_budget` renders
24 rules per call.

**Near-total failure is the training signal.** ACE curates from failures. Over the 283 reflections
that match a training step, the generator's partial credit distributes as:

| tags right | steps |
|---|---|
| 0 of 4 | 94 |
| 1 of 4 | 76 |
| 2 of 4 | 64 |
| 3 of 4 | 35 |
| 4 of 4 | 14 |

Reflector tags come out helpful 201, harmful 356, neutral 257. The same check on the formula
co-citation arm reads helpful 725 against harmful 97 at a training accuracy of 0.720. The loop's
own judgement here is that the playbook it built is hurting, and the counters agree: the two
most-cited rules end at helpful 2 / harmful 17 and helpful 4 / harmful 11.

**Credit is assigned across four decisions at once.** `bullet_ids` is one list per generator call
covering four independent sub-questions, so `update_counts` credits or blames every cited rule for
all four together. With 0 of 4 on 35% of steps and 4 of 4 on 5%, most rules are blamed for
decisions they did not touch. This cannot be repaired by splitting the row: §2 rests the
co-citation premise on the four sub-questions coming from one filing section and being answered in
one call, so un-batching removes within-call co-citation and the treatment stops existing. The
batching is at once the task's worst property for credit assignment and the precondition of the
experiment.

**The grading rule discards part of the helpful signal.** `answer_is_correct` on `finer` is
all-four-exact, so `attempt_is_correct` is False on 186 of 200 steps and
`Reflector._filter_bullet_tags` demoted 164 `helpful` tags to neutral. Replayed against each step's
partial credit, 74 of those 164 sit on rows scoring 0.5 or 0.75; the other 90 sit on rows scoring
0 or 0.25 and would be demoted under any threshold. Restoring the 74 moves the ratio from 201:356
to 275:356, which is a real change to what the curator reads and is not on its own large enough to
account for the accuracy gap.

**Model capacity is a fifth factor and it is not separable here.** `gemma4:e2b` holds a 140-tag
taxonomy, four questions and 2,500 tokens of advice in one call, and its median response falls from
1,192 tokens in the control to 970 with the memory present. One model cannot separate capacity from
the four causes above, and no claim is made that it can.

### 10.4 What this licenses

It licenses a claim about **this playbook**, not about memory. A memory that is 73% one directive
is not a test of whether accumulated context helps, and the `none` to `text` result is reported as
what it is: under the authors' finance curator prompt, on a 2B model, the loop produced a
degenerate playbook and the arms carrying it lost by 3.51 and 3.74 points.

It licenses one methodological claim in full: **testing relational memory requires an underlying
memory bank that is dense and connected**, and the treatment arm here is the evidence. The `text`
to `cocite` comparison measured nothing, because the walk added no rule to any of the 441
final-test generations.

Neither the degenerate playbook nor the all-four grading rule is a confound. Both are identical in
all three arms and are properties of the system under test rather than differences between the
things being compared.

### 10.5 Offline diagnosis, 2026-08-28

Run against the saved checkpoints with no LLM and no re-run. The co-citation gate is applied at
read time in `GraphMemory._cocite_edge_pairs` and an explicit keyword beats the thresholds saved
with the graph, so any candidate setting can be scored against a finished checkpoint.
`analyze_walk_share.py` gained `--cocite-min-rate`, `--cocite-min-opportunity`,
`--cocite-half-life`, `--retrieval-budget`, `--retrieval-seeds` and `--checkpoint` for this. All
are inert when unset, and the gate figures quoted in §10.2 come from this path.

**A tooling defect was found first, and it is not confined to this run.** `test_generations` kept
the *first* log per index, and validation evaluations write under the same
`generator_test_eval_<index>_*` prefix, so on any offline run with `--eval_steps` set the indices
below the validation-set size came from a validation pass and the rest from the final test. It now
keeps the last file per index, which is the final test because the filenames carry a timestamp.
Corrected figures for the formula arms are in `COCITE_RESULTS.md`; the largest change there is
0.0097 and no conclusion moves. On this run the corrected final test still reads 0.0000 on 441 of
441 generations.

**The collapse is the checkpoint, not the question set.**

| checkpoint | live rules | validation | final test |
|---|---|---|---|
| step 100 | 84 | 0.1739 | 0.0833 |
| step 200, the one validation selected | 129 | 0.0000 | 0.0000 |

The treatment fired for the first half of training, above §7's 0.05 floor on both question sets,
and was off at the checkpoint the final test ran on. The edge set did not shrink to cause it: at
gate 0.20 the step-100 graph has 17 live edges over 8 nodes and the step-200 graph has 38 over 11.

**De-duplication is not the lever, and the shipped threshold was already past its own guard.**
Replaying the run's recorded curator operations offline at each candidate:

| threshold | baseline absorbed | live | treatment absorbed | live |
|---|---|---|---|---|
| off | 0.0% | 315 | 0.0% | 280 |
| 0.76 | 84.5% | 72 | 81.9% | 64 |
| 0.80 | 74.8% | 105 | 72.8% | 91 |
| 0.84 | 64.3% | 135 | 64.4% | 114 |
| 0.86, as run | 56.8% | 143 | 56.8% | 129 |

Every value tested is over the 50% absorption guard, including the one the run used. At 0.86 the
pass already absorbed 56.8% of ADDs and the 143 rules left standing are still 73% one directive,
so the redundancy that matters is not the redundancy cosine can see. Going lower does not reach it
either: 0.76 absorbs 84.5% and is merging genuinely different rules to get there. The generation
side has to change, because the pruning side cannot distinguish a restatement of the directive
from a rule about a different tag family.

**Both mis-set hyperparameters failed the same way.** `--dedup_threshold` was pinned at 0.86 on
the 50-step probe, where it absorbed 35.5%, and reaches 56.8% at 200 steps.
`--cocite_min_rate` was pinned at 0.20 on the same probe, where median co-rendering was 21, and
faces 54 at 200 steps. Both gate a quantity that grows with playbook size, and both were fixed
from a run an eighth as long. A probe can set the shape of a threshold on this task but not its
value, and §7 should be read at the selected checkpoint of a full-length run rather than on the
probe alone.
