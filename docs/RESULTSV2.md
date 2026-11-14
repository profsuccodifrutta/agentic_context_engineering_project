# Co-citation v2 results, `finer` and `formula`

**Status: complete. Twelve training runs (2 arms × 3 replicates × 2 tasks), with the `none`
controls reused. This is the final analysis.**

Pre-registration: `EXPERIMENT_COCITE_V2.md`. The outcomes are in §5, the gates in §7, and every
change since it was written in §12.

**Headline.** The redesign fixed what it set out to fix: the graph now reaches about half the
memory instead of an eighth. Accuracy still did not move. **The primary `text → cocite_rank` is
null on both tasks**: +1.15pp on `finer` (p=0.32) and +0.83pp on `formula` (p=0.72). Memory itself
helps clearly on `finer`. On `formula` the memory arms are ahead of the control, but neither gap
survives the multiplicity correction.

| question | `finer` | `formula` |
|---|---|---|
| **Primary.** Does ranked co-citation beat flat retrieval? | **No.** +1.15pp, Holm p=0.319 | **No.** +0.83pp, Holm p=0.724 |
| Does the treatment memory beat no memory? | **Yes.** +8.81pp, Holm p=0.00006 | Not after correction. +6.00pp, Holm p=0.145 |
| Does the baseline memory beat no memory? | **Yes.** +7.65pp, Holm p=0.00008 | Not after correction. +5.17pp, Holm p=0.176 |
| Was the treatment delivered? | Yes, every gate passes on all three runs | Yes, except gate 2 (hub), which fails on all three runs (§3) |

---

## 1. What is different from the first co-citation experiment

The first co-citation experiment (`COCITE_RESULTS.md`, `FINER_RESULTS.md`) was null on both
tasks. The diagnosis afterwards was that **the graph could only reach about an eighth of the
memory**. On `finer`, 30 of 224 live rules carried any edge, so for seven rules in eight the
treatment arm behaved exactly like the baseline. Two causes were measured. The edge rule was a
threshold recomputed on every read, so edges disappeared as the memory grew. The threshold was also
a global rate, which preferred thinly observed pairs (1 of 3) over well-observed ones (25 of 300).

v2 changes five things. Each change has its own flag:

| change | before (v1) | now (v2) | flag |
|---|---|---|---|
| **How an edge is made** | edge if co-cited on ≥ 10–20% of co-renderings (a threshold) | every rule links to its **top 2** partners by `reward / (corendered + 10)` (a rank) | `--cocite_predicate rank --cocite_k 2 --cocite_beta 10` |
| **What an edge counts** | how often two rules were cited together | how often they were cited together **on a good answer**: fractional on `finer` (3 of 4 tags = 0.75), 0/1 on `formula` | `--cocite_outcome_weighted` |
| **How the walk moves** | breadth-first from the seeds | **best-first** on path strength (relevance × edge weights), seeds always first | `--cocite_walk best_first` |
| **What the model is shown** | the selected rules plus `co_cited:` relation lines | the selected rules only; the walk uses the edges but the prompt doesn't show them | `--no_render_relations` |
| **How seeds are allocated** | global top 5 by relevance | split across `finer`'s four sub-questions (2 each); unchanged on `formula`, which has one query | `--subquery_seeding`, **in both arms** |



Everything else is held identical across arms: prompts, data order, `--num_ctx 32768`,
`--retrieval_budget 2500`, de-duplication (0.86 on `finer`, 0.88 on `formula`), model
(`gemma4:e2b` in every role) and splits (`task_config_v2.json`).

## 2. Runs and how they were compared

All under `results/<task>/v2/seed<n>/<arm>/`.

**Statistics, as pre-registered.** Replicates are pooled per row, then tested with a paired
sign-flip permutation test and a bootstrap CI. On `finer` the unit is the row (441), and the
metric is per-tag partial credit. On `formula` the unit is the formula (50 blocks of 4), and the
metric is exact match. Holm is applied within each task's family of three comparisons, and the two
tasks are not pooled. The decision rule is that an arm wins only if the test is significant **and**
the mean gap exceeds the within-arm sd.

**The fairness audit passed on every pinned field except `git_commit`.** The comparison was run
with `--force` after checking that this difference is harmless:
- The v2 training runs span commits `3764478`, `25d8fba` and `a46eeff`. The run-path diff between
  them is empty.
- The reused `none` controls ran at older commits. Between those commits and `a46eeff`, the changes
  that reach an `eval_only --memory none` run are type annotations, a removed unused helper, and
  `finer`-only code (the curator prompt, the retrieval query, and a scorer refactor shared by all
  arms). The `formula` scorer is untouched, and `formula` supplies no retrieval query.
- Empirically, a `formula` `none` re-run at `a46eeff` reproduced the old seed-0 control byte for
  byte. On `finer`, `none` has been byte-identical across five runs, three commits and two prompt
  sets.

## 3. The gates (§7): was the treatment delivered?

These were read before any accuracy was quoted. They are necessary conditions: passing shows the
arms read different rules, but it doesn't show the edges are good. Gate 3 is the one that tests the
edges themselves.

| run | 1. connected share (≥ 40%) | 2. max degree / connected (≤ 25%) | 3. held-out lift (> 1.0×) | 4. `walk_only_share` (≥ 0.05) | 5. distinct sets in first 25 (≥ 20) |
|---|---|---|---|---|---|
| `finer` s0 | 110/234 = 47.0% | 12/110 = 10.9% | 3.99× | 0.560 | 25 |
| `finer` s1 | 110/234 = 47.0% | 12/110 = 10.9% | 3.99× | 0.560 | 25 |
| `finer` s2 | 106/223 = 47.5% | 13/106 = 12.3% | 2.28× | 0.545 | 25 |
| `formula` s0 | 42/87 = 48.3% | 16/42 = **38.1%, fail** | 2.18× | 0.462 | 21 |
| `formula` s1 | 46/103 = 44.7% | 13/46 = **28.3%, fail** | 3.14× | 0.553 | 23 |
| `formula` s2 | 46/103 = 44.7% | 13/46 = **28.3%, fail** | 3.14× | 0.553 | 23 |

Gate 5 also passes on all six `text` runs (25 of 25 each). `check_learning_signal.py` reports no
failure on any of the twelve runs.

Each gate is read at the checkpoint the final test used: step 200 on `finer`, step 125 on
`formula` `cocite_rank`. Gates 1 and 2 come from the run's own `structural_stats()`. Gate 3 is
`calibrate_cocite_rank.py`'s held-out lift: edges are built on the first half of the logged
training generations and scored on the second half. That replay holds less evidence than the live
run, so it is conservative.

**The reach defect is fixed.** The connected set went from 13% of the memory on `finer` to 47%,
and `walk_only_share` went from 0.11–0.14 to 0.55–0.56. The walk now selects more than half of
every rendered slice, and on every test generation it selects a different set (441 of 441 on
`finer`).

**Gate 2 fails on all three `formula` runs, and it is reported as failed.** Its bar was inherited
from the breadth-first design, where a hub traps the walk: every walk that touches it is pulled
into the same neighbourhood. §7 itself says that under best-first "raw path strength makes a hub's
neighbours compete on evidence" and keeps the gate "as a diagnostic". The runs show no trapping:
- The walk adds a different set of rules on 187 of 200 test generations (seed 0) and on 196 of 200
  (seeds 1 and 2).
- On seed 0, the hub's neighbours take 45.6% of walk-added slots, close to their 38% share of the
  connected set.
- There is a small recurring core: on seed 0, three walk-added rules appear on 80–84% of calls.
  The hub-neighbour breakdown was measured on seed 0 only.

Every gate that measures delivery passes.

## 4. Results

### 4.1 `finer`: per-tag accuracy (primary metric)

| arm | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| `none` | 0.3566 | 0.3566 | 0.3566 | **0.3566** | 0.0000 |
| `text` | 0.4286 | 0.4354 | 0.4354 | **0.4331** | 0.0039 |
| `cocite_rank` | 0.4365 | 0.4609 | 0.4365 | **0.4446** | 0.0141 |

Pooled, 441 rows:

| comparison | role | diff | 95% CI | raw p | Holm p |
|---|---|---|---|---|---|
| `text → cocite_rank` | **primary** | **+1.15pp** | [−1.06, +3.42] | 0.319 | **0.319** |
| `none → cocite_rank` | secondary | +8.81pp | [+5.52, +12.04] | 0.00002 | **0.00006** |
| `none → text` | secondary | +7.65pp | [+4.16, +11.15] | 0.00004 | **0.00008** |

The primary also fails the decision rule on its own terms: the 1.15pp gap is smaller than the
treatment arm's 1.41pp sd across seeds.

**All-four exact (the authors' metric, secondary):** `none` 0.0680, `text` 0.1194, `cocite_rank`
0.1096. `text → cocite_rank` is **−0.98pp**, Holm p=0.465, with the opposite sign to per-tag.
`none → text` (Holm p=0.011) and `none → cocite_rank` (Holm p=0.013) are significant. The two
metrics disagreeing on the sign of a null gap is what a null looks like.


### 4.2 `formula`: exact match, clustered on the formula

| arm | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| `none` | 0.6250 | 0.6200 | 0.6200 | **0.6217** | 0.0029 |
| `text` | 0.6900 | 0.6750 | 0.6550 | **0.6733** | 0.0176 |
| `cocite_rank` | 0.6850 | 0.6800 | 0.6800 | **0.6817** | 0.0029 |

Pooled, 50 formula blocks, 200 rows:

| comparison | role | diff | 95% CI | raw p | Holm p |
|---|---|---|---|---|---|
| `text → cocite_rank` | **primary** | **+0.83pp** | [−2.83, +4.50] | 0.724 | **0.724** |
| `none → cocite_rank` | secondary | +6.00pp | [+0.50, +11.83] | 0.048 | **0.145** |
| `none → text` | secondary | +5.17pp | [−0.17, +10.83] | 0.088 | **0.176** |

`none → cocite_rank` is the one to word carefully. Its unadjusted CI excludes zero, but it is one
of three tests and does not survive Holm. It is not a significant result.

### 4.3 How small an effect could these runs have seen?

The 80% power point is 2.80 standard errors (deviation 7). The SE is read off each primary's
bootstrap CI:
- **`finer`**: SE ≈ 1.14pp. A gap needs ~2.2pp to reach significance, and the 80% power point is
  ~3.2pp.
- **`formula`**: SE ≈ 1.87pp. A gap needs ~3.7pp to reach significance, and the 80% power point is
  ~5.2pp.

Per the pre-registration, the nulls are read through their intervals and not against these figures.
The `finer` interval rules out an advantage for `cocite_rank` larger than about 3.4pp, and the
`formula` interval rules out one larger than about 4.5pp.

## 5. Interpretation

**Fixing reach did not move accuracy.** v1 left 87% of the memory unreachable by the walk, so a
null there could be blamed on the treatment barely acting. v2 removes that excuse. The walk now
chooses more than half of what the model reads, on every call, along edges that predict held-out
co-citation at 2–4× base rate. The primary is still null on both tasks. With v1, the project has
now tested relational retrieval in four configurations across two tasks, and none produced a
significant `text → graph` gain.

**The model rarely uses what the walk adds.** Walk-added rules are cited at **0.16–0.20×** the rate
of relevance-picked rules on `finer`, and at **0.40–0.61×** on `formula`:

| run | walk-added cited | relevance-picked cited | ratio |
|---|---|---|---|
| `finer` s0 / s1 / s2 | 1.2% / 1.3% / 1.4% | 7.8% / 7.5% / 6.9% | 0.16 / 0.17 / 0.20 |
| `formula` s0 / s1 / s2 | 4.4% / 7.3% / 7.3% | 11.1% / 12.0% / 12.0% | 0.40 / 0.61 / 0.61 |

In v1, with a small connected set, the same ratio was 1.3–1.5× on `finer` and 1.3–2.3× on
`formula`. The likely reading is that v1's walk added a few rules that relevance had narrowly
missed, while v2's walk fills half the slice with rules further from the question, and the
generator mostly ignores them. This is a citation frequency, which is the model's self-report. It
says how often the model names these rules, **not** whether they are good rules.

**So the edge set is no longer the bottleneck.** On `finer`, `FINER_RESULTS.md` §6 already showed
that the limit is **tag coverage**. Where the playbook holds a rule for the correct tag, accuracy
roughly doubles. Where it doesn't, the playbook does worse than no memory. No change to edges,
walk or seeding can write a rule for a tag the curator never addressed, and §1 of the
pre-registration declared that out of scope. These runs are consistent with that: the arms differ
only in which existing rules they show, and that is not where the accuracy is.

**Memory helps on `finer`; on `formula` it is ahead but not significant this time.** On `finer`,
both memory arms beat the control by 7.7–8.8pp at Holm p < 0.0001, replicating `FINER_RESULTS.md`
(+8.84pp and +6.82pp). On `formula`, both arms are 5–6pp ahead of the same control, but neither gap
survives Holm, while v1 found `none → cocite` +10.83pp at p=0.0008. Both v2 `formula` arms score
below their v1 counterparts (`text` 0.6733 against 0.7083, `cocite_rank` 0.6817 against 0.7300). The
v2 configuration differs from v1 in several places at once, including `--no_curator_refines` in
both arms, so this drop is reported and not attributed. Cross-experiment comparisons are not
controlled.

**What this licenses.** In this loop, with this model and on these two tasks, selecting rules by
outcome-weighted co-citation does not beat selecting them by relevance, even when the selection
reaches half the memory and changes on every call. It does **not** license "structure cannot
help". It licenses the claim that the read path is not where this method's accuracy is set.



## 6. Disclosures

- **Some replicates are copies.** Hashes of `best_playbook.txt` and the test predictions:
  - `finer` `text`: all three seeds selected the same playbook, and seeds 1 and 2 have
    byte-identical predictions.
  - `finer` `cocite_rank`: seeds 0 and 1 share a playbook, but their test predictions differ.
    Seed 2 is distinct.
  - `formula` `cocite_rank`: seeds 1 and 2 are byte-identical in both playbook and predictions.
  - `formula` `text`: all three seeds are distinct.

  The tests stay valid, because they resample rows or formulas and every replicate sees the same
  items. But the within-arm sd is a floor, not a full measure of loop noise, in three of the four
  training arms. Copies are never described as independent replicates. The cause is the
  server-state determinism documented in `docs/background/EXPERIMENT_FORMULA.md` §4: `--seed` is
  inert at temperature 0.
- **The `finer` `none` control is one run entered three times.** It is deterministic on this task,
  and the pooled analysis needs equal replicate counts per arm. This matches `FINER_RESULTS.md`.
- **The `formula` `none` control is the v1 control**, reused as deviation 6 specifies.
- **Gate 2 fails on `formula`** and is reported as measured (§3). No re-run was made and no
  deviation entry was added.
- **Held-out lift (gate 3) is computed on a replay** of logged pre-train generations, which omits
  each step's post-curate generation and does no inheritance, so it understates the live graph's
  evidence.

## 7. Reproduce

```
uv run python compare_arms.py --selftest
uv run python compare_arms.py --force --metric finer_per_tag \
  results/finer/v2/seed*/*/ace_run_* \
  results/finer/repaired_v2/seed0/none/ace_run_* \
  results/finer/repaired_v2/seed0/none/ace_run_* \
  results/finer/repaired_v2/seed0/none/ace_run_*
uv run python compare_arms.py --force --metric exact <same finer runs>
uv run python compare_arms.py --force \
  results/formula/v2/seed*/*/ace_run_* results/formula/preregistered/seed*/none/ace_run_*
uv run python calibrate_cocite_rank.py results/*/v2/seed*/cocite_rank/ace_run_*      # gate 3
uv run python analyze_walk_share.py results/*/v2/seed*/cocite_rank/ace_run_*         # gate 4, citation ratios
uv run python check_learning_signal.py results/*/v2/seed*/*/ace_run_*                # gate 5
```

Gates 1 and 2 are `structural_stats()` on each run's `best_playbook.graph.json`, loaded through
`calibrate_cocite_rank.checkpoint_memory(run_dir, weighted=True)`. The `formula` seed-0
hub-neighbour breakdown in §3 came from a one-off replay through `GraphMemory.select_for_query`,
not a repo tool.
