# V2 ablation: the treatment memory read by cosine alone

**Status: six `eval_only` runs complete (3 replicates × 2 tasks), run 2026-11-18/19.
Exploratory, not pre-registered.** Analysis plan written before the runs: `ABLATION_V2_COSINE_PLAN.md`.

## 1. Goal

The V2 primary comparison, `text → cocite_rank`, came out null on both tasks
(`RESULTSV2.md` §4). That comparison mixes two ways the graph can act:

1. **What the memory contains.** During training the walk changes what the generator reads, and
   therefore the reflections, the curator's additions and the rules that end up stored. The two arms
   finish training with different memories.
2. **How rules are selected at test time.** The walk versus relevance alone.

This ablation separates them. It takes the treatment's selected memory and runs the test phase with
flat cosine retrieval, under the same token budget and per-sub-question seeding as the baseline.
That gives three cells:

| cell | memory | test retrieval | runs |
|---|---|---|---|
| **A** | baseline (`text`) | cosine | existing V2 baseline arm |
| **B** | treatment (`cocite_rank`) | cosine | **new**, `results/<task>/v2_ablation_cosine/` |
| **C** | treatment (`cocite_rank`) | walk | existing V2 treatment arm |

- **C vs B** holds the memory fixed and varies selection, so it measures **the walk's effect at test time**.
- **A vs B** holds retrieval fixed and varies the memory, so it measures **the content effect**.
- **A vs C** is the existing V2 primary.

## 2. Is B the counterfactual it claims to be?

Before any accuracy was read, every generation of every run was replayed through
`GraphMemory.select_for_query`, the run's own selection code, on the treatment checkpoint:

| task | seed | B shows the relevance-only selection | C shows the walk selection | walk-only share | rules per prompt | live rules |
|---|---|---|---|---|---|---|
| formula | 0 | 200/200 | 200/200 | 0.46 | 32 | 87 |
| formula | 1 | 200/200 | 200/200 | 0.55 | 31 | 103 |
| formula | 2 | 200/200 | 200/200 | 0.55 | 31 | 103 |
| finer | 0 | 441/441 | 441/441 | 0.56 | 63–64 | 234 |
| finer | 1 | 441/441 | 441/441 | 0.56 | 63–64 | 234 |
| finer | 2 | 441/441 | 441/441 | 0.55 | 63–64 | 223 |

On every question, B and C read the same memory and differ only in selection, and that difference
is large: **about half of the rules in each prompt differ between the two cells**, on every row
but one.

## 3. Results

### 3.1 Accuracy per replicate

**Formula** (exact match):

| cell | seed 0 | seed 1 | seed 2 | pooled | sd |
|---|---|---|---|---|---|
| control (`none`, from `RESULTSV2.md`) | 0.6250 | 0.6200 | 0.6200 | 0.6217 | 0.0029 |
| A: baseline memory, cosine | 0.6900 | 0.6750 | 0.6550 | **0.6733** | 0.0176 |
| B: treatment memory, cosine | 0.7000 | 0.6750 | 0.6700 | **0.6817** | 0.0161 |
| C: treatment memory, walk | 0.6850 | 0.6800 | 0.6800 | **0.6817** | 0.0029 |

**FiNER** (per-tag accuracy, the primary metric):

| cell | seed 0 | seed 1 | seed 2 | pooled | sd |
|---|---|---|---|---|---|
| control (`none`, from `RESULTSV2.md`) | 0.3566 | 0.3566 | 0.3566 | 0.3566 | 0.0000 |
| A: baseline memory, cosine | 0.4286 | 0.4354 | 0.4354 | **0.4331** | 0.0039 |
| B: treatment memory, cosine | 0.4609 | 0.4626 | 0.4671 | **0.4635** | 0.0032 |
| C: treatment memory, walk | 0.4365 | 0.4609 | 0.4365 | **0.4446** | 0.0141 |

### 3.2 Paired comparisons

Statistics as in the V2 analysis: replicates pooled per row, then a cluster sign-flip test and a
cluster bootstrap CI (clustered on the formula on Formula, on the row on FiNER), computed through
`compare_arms.py`'s own `pooled_arm_scores` and `cluster_paired_stats`. The two ablation
comparisons form one exploratory family per task, Holm-corrected (m = 2).

| task | comparison | what it measures | difference | 95% CI | p | Holm p |
|---|---|---|---|---|---|---|
| formula | C → B | walk vs cosine, same memory | **+0.00pp** | [−4.17, +4.00] | 1.000 | 1.000 |
| formula | A → B | content, both cosine | **+0.83pp** | [−3.33, +5.17] | 0.763 | 1.000 |
| formula | A → C | V2 primary (`RESULTSV2.md`) | +0.83pp | [−2.83, +4.50] | 0.724 | — |
| finer | C → B | walk vs cosine, same memory | **+1.89pp** | [+0.06, +3.72] | 0.049 | **0.049** |
| finer | A → B | content, both cosine | **+3.04pp** | [+0.76, +5.33] | 0.011 | **0.023** |
| finer | A → C | V2 primary (`RESULTSV2.md`) | +1.15pp | [−1.06, +3.42] | 0.319 | — |

A positive difference means the second cell scores higher.

**Detection limits** (SE read off each CI): Formula needs about 4.1–4.2pp to reach significance and
5.8–6.1pp for 80% power. FiNER needs 1.8pp (C → B) and 2.3pp (A → B) to reach significance, and
2.6pp and 3.3pp for 80% power.


## 4. Interpretation

**Formula: the walk makes no difference to accuracy.** The same memory scores 0.6817 whichever way it
is read, although about half of each prompt differs between the two readings, and the rows it moves
cancel out (25 up, 26 down). The two memories are also equally useful when both are read by cosine.
This supports the reading in thesis 10.1: accuracy is set by the few most relevant rules, which both
selections share, because the walk starts from the relevance seeds. The rest of the prompt can be
swapped without changing the outcome. The intervals rule out an effect of the walk larger than about
4pp in either direction.

**FiNER: taken at face value, the treatment loop wrote a better memory, and the walk wasted most of
that advantage.** Read by cosine, the treatment memory beats the baseline memory by +3.04pp
(Holm p = 0.023). Read by the walk, the same memory scores 1.89pp lower (Holm p = 0.049). That leaves
the +1.15pp null of the V2 primary: the two effects roughly cancel. This fits V2's low citation
ratio on FiNER (`RESULTSV2.md` §5): the walk fills about half of each prompt with rules the model
cites at 0.16–0.20× the rate of relevance rules, and those rules displace relevant rules the model
would have used.


**What this licenses for the thesis.** On Formula, the walk is inert at test time, which is direct
support for section 10.1. On FiNER, pending the drift check, the null between arms hides two opposite effects:
a better memory, and a selection that wastes it. 

## 5. Reproduce

```
uv run python compare_arms.py --force results/formula/v2_ablation_cosine/seed*/ace_run_* results/formula/v2/seed*/cocite_rank/ace_run_*
uv run python compare_arms.py --force --metric finer_per_tag results/finer/v2_ablation_cosine/seed*/ace_run_* results/finer/v2/seed*/cocite_rank/ace_run_*
```
`--force` covers the git-commit difference only. The A → B pairs label both cells `text`, so they
were computed by calling `pooled_arm_scores`, `cluster_paired_stats` and `holm` from `compare_arms.py`
directly with distinct labels. The selection replay and the dose test call
`GraphMemory.select_for_query` through `analyze_walk_share.py`'s loaders.
