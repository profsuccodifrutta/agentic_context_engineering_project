# V2 ablation: treatment memory read by cosine — analysis plan

What the results file (`docs/ABLATION_V2_COSINE_RESULTS.md`) must cover once the six runs finish.
Written before any ablation result was seen.

## The question
The V2 comparison `text → cocite_rank` mixes two effects: what the memory contains (the walk shaped
training, so the arms wrote different rules) and how rules are selected at test time. Reading the
treatment's selected memory with cosine alone separates them.

## The cells (both tasks, 3 replicates each, V2 settings)

| cell | memory | test retrieval | source |
|---|---|---|---|
| A | baseline (`text`) | cosine | existing: `results/<task>/v2/seed*/text/` |
| B | treatment (`cocite_rank`) | cosine | **new**: `results/<task>/v2_ablation_cosine/seed*/` |
| C | treatment (`cocite_rank`) | walk | existing: `results/<task>/v2/seed*/cocite_rank/` |

## Comparisons
1. **C vs B: the walk's effect at test time.** Same rules, questions and budget; only selection differs.
2. **A vs B: the content effect.** Same retrieval; only the memory differs.
3. **A vs C** is the existing V2 primary. Quote it from `docs/RESULTSV2.md`, don't re-derive it.

Statistics as in the thesis (Chapter 5): replicates pooled per row, cluster sign-flip test and cluster
bootstrap CI (formula clusters on Formula; rows and `--metric finer_per_tag` on FiNER), the pp
difference, and the SE / 1.96 SE / 2.80 SE figures. Comparisons 1 and 2 are an **exploratory**
family, Holm-corrected within each task (m = 2), and never presented as pre-registered.
Tooling: `compare_arms.py` for B vs C. For A vs B both cells are labelled `text`, so run them as a
separate pair, or call `pooled_arm_scores` / `cluster_paired_stats` directly. Never compute a
statistic by hand.

## Checks before reading any accuracy
- Each run is complete: 200 / 441 predictions, no failed or empty generations beyond the usual rate.
- The prompts are what the ablation claims: B's test prompts contain no relation lines, and the rule
  set B rendered matches the relevance-only selection `analyze_walk_share.py` computes on the same
  checkpoint (check a sample of generations). Otherwise B is not the counterfactual.
- Throughput was GPU-speed, and the endpoint answered with the header.
- Replicates: hash the prediction files; note that Formula seeds 1/2 and FiNER seeds 0/1 share a
  playbook, so B has 2 distinct memories per task, as C does.

## Interpretation (state it per task)
- **B ≈ C:** the walk-added rules make no difference to accuracy, supporting thesis 10.1.
- **B > C:** the walk hurts at test time. Cross-reference V2's low citation ratio (0.16–0.61×).
- **B < C:** the walk helps at test time, and the arm-level null means content differences offset it.
- **A vs B:** whether the treatment loop wrote a better or worse memory, read by cosine.
- Nulls are read through their intervals, not as equality. No chaining of non-significant results:
  each conclusion comes from its own direct comparison.
- Supporting detail where useful: per-row discordant counts, and whether the effect sits in rows
  where the walk added many rules.

## Limits to state
One model; exploratory family; differences in content come from training under different loops,
so their cause isn't identified; the walk can't be tested on the baseline memory, which has no edges.
