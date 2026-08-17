# TODOs — what to do next

Last updated 2026-08-16. Working checklist, not a research document. For the reasoning behind
any item, follow the link on it.

## Where things stand

| | status |
|---|---|
| **formula** | ✅ **done** — 9 runs, analysed, written up in `FORMULA_RESULTS_PREREGISTERED.md` |
| **MBPP** | 🔜 pre-registered in `EXPERIMENT_MBPP.md`, **arms not run**. Steps 1–4 done: splits, generator, probes, `--dedup_threshold 0.86`. Next is step 5. |
| open question on formula | ✅ closed — see item 1 |

**Result so far, in one line:** memory helps (`none → graph` +8.0pp, p=0.0146); relational
structure was **not** shown to beat linear memory (`text → graph` +2.8pp, p=0.2905, null).

---

## Do these in order

### 1. Diagnose the text-arm curator hygiene ✅ **done 2026-08-16**

**Outcome: not a threat to the result. MBPP is unblocked.**

- De-duplication *was* active in the `text` arm (`dedup: true`, 0.88) and absorbed 128/170/150
  rules per seed. The arm was not running an append-only memory.
- The redundancy never reaches the generator. Under the shared 2,500-token budget the `text`
  arm reads 30.8 rules on average of which ~27.0 are distinct; `graph` reads 18.9. Retrieval
  filters the bloat out of the curator's unbudgeted view. Written up in
  `FORMULA_RESULTS_PREREGISTERED.md` §7–§8.
- Duplicate rate does not track accuracy within the arm across the three seeds.

Two by-products, both already applied:
- **Fixed** `_RENDERED_PREFIX`, which pinned ids to `\w+-\d{5}` and leaked 82 pasted prefixes
  into the formula playbooks because the model invents ids. Verified inert on MBPP.
- **Recorded** the one genuine second arm difference: the `text` arm parses the curator's
  `refines` field and drops it (50/126/102 declarations discarded), while `graph` acts on
  36/35/44. Disclosed in `CLAUDE.md`. It cannot inflate a `text → graph` claim because there
  is no such claim — that comparison is null — but it does mean part of `none → graph` may be
  eviction rather than relational retrieval.

---

### 2. MBPP preflight ⬜
**Cost: ~2 minutes. No LLM calls except the last line.**

```powershell
docker ps
uv run python check_memory_equivalence.py
uv run python check_prompts.py
uv run python compare_arms.py --selftest
uv run python check_model.py --model gemma4:e2b --think off
```
All must pass. Docker must be running or **every** MBPP run dies before its first LLM call.

---

### 3. MBPP probes ✅ **done 2026-08-17**

**Both arms ran. Gates [1] [2] [4] [6] pass; [3] passes at 0.667 / 0.767 so the §7 ceiling
branch does not trigger — `gemma4:e2b` stands.** [5] fails on the graph arm at 15.1% vs a 15%
limit; traced to the probe's 10-problem test split (2 chronic offenders, 5 calls per train
step), 1.7% on a real 257-row set, and disclosed in the `EXPERIMENT_MBPP.md` amendment log
rather than waived. `walk_only_share` is 0.000 — expected, the playbook never exceeded 1,441
tokens against the 2,500 budget, so retrieval never bound. Command kept below for reference.

**Cost: ~1 hour.**

```powershell
$probe = @(
  '--task_name','mbpp_code_exec',
  '--config_path','./eval/mbpp_code_exec/data/probe_config.json',
  '--generator_model','gemma4:e2b','--reflector_model','gemma4:e2b','--curator_model','gemma4:e2b',
  '--think_generator','off','--num_ctx','32768','--seed','0',
  '--dedup','--dedup_threshold','0.90','--eval_steps','15',
  '--retrieval_budget','2500','--retrieval_seeds','5','--test_workers','1','--skip_initial_test'
)
uv run python -m eval.mbpp_code_exec.run @probe --mode offline --save_path results/mbpp/probes_v2/text  --memory text  --retrieval flat
uv run python -m eval.mbpp_code_exec.run @probe --mode offline --save_path results/mbpp/probes_v2/graph --memory graph --retrieval graph
uv run python check_learning_signal.py results/mbpp/probes_v2/text results/mbpp/probes_v2/graph
```

**⚠️ Decision point — read check [3] in the output.** `gemma4:e2b` scores 0.8521 on MBPP with no
memory, so train accuracy may land above the 0.20–0.85 band. `EXPERIMENT_MBPP.md` §7 has the
response already written down: either run anyway and lead the results with the ceiling caveat,
or switch the generator. **Decide before seeing any test number.**

➡️ *Paste me the output and I'll tell you which branch applies.*

---

### 4. Pick `--dedup_threshold` ✅ **done 2026-08-17 → `0.86`**

The 50% absorption rule did not select: nothing in [0.80, 0.94] approaches it, and at 0.90 dedup
absorbs **zero** rules while 14 of 20 rules in one section restate each other — grow-and-refine
would have been inert for the whole run, which is the seed-0 formula defect all over again.
0.80–0.86 is one plateau collapsing an identical merge set, so 0.86 is its top. All ten merges
were read first; every one is a genuine restatement, no false positive. Full evidence in the
`EXPERIMENT_MBPP.md` amendment log.

```powershell
uv run python calibrate_dedup_threshold.py results/mbpp/probes_v2/text results/mbpp/probes_v2/graph
uv run python calibrate_dedup_threshold.py results/mbpp/probes_v2/text results/mbpp/probes_v2/graph --show-merges 0.86
```

---

### 5. MBPP seed 0 — three arms ⬜
**Cost: ~7 hours. Run seed 0 alone first and inspect it**, the way you did on formula.

```powershell
$common = @(
  '--task_name','mbpp_code_exec',
  '--config_path','./eval/mbpp_code_exec/data/task_config_v2.json',
  '--generator_model','gemma4:e2b','--reflector_model','gemma4:e2b','--curator_model','gemma4:e2b',
  '--think_generator','off','--num_ctx','32768',
  '--dedup','--dedup_threshold','0.86',
  '--eval_steps','40','--retrieval_budget','2500','--retrieval_seeds','5','--test_workers','1'
)

$s = 0
uv run python -m eval.mbpp_code_exec.run @common --seed $s --mode eval_only `
  --save_path "results/mbpp/preregistered/seed$s/none"  --memory none  --retrieval none
uv run python -m eval.mbpp_code_exec.run @common --seed $s --mode offline --skip_initial_test `
  --save_path "results/mbpp/preregistered/seed$s/text"  --memory text  --retrieval flat
uv run python -m eval.mbpp_code_exec.run @common --seed $s --mode offline --skip_initial_test `
  --save_path "results/mbpp/preregistered/seed$s/graph" --memory graph --retrieval graph
```

**Then stop and check before spending more:**
```powershell
uv run python check_learning_signal.py results/mbpp/preregistered/seed0/text results/mbpp/preregistered/seed0/graph
uv run python aggregate_run_costs.py results/mbpp/preregistered/seed0/*/ace_run_*
uv run python analyze_walk_share.py results/mbpp/preregistered/seed0/graph
```
**The `walk_only_share` gate is the final-test pooled share from `analyze_walk_share.py` — not
the number in `structural_stats()`, which is a run-cumulative average that excludes the test
entirely. `≥ 0.05` = the treatment fired; `< 0.05` = report the run as *mechanism-inconclusive*,
not as a negative result about structure** (`EXPERIMENT_MBPP.md` §8, fixed 2026-08-17 before any
arm ran). Expect a lower value than formula's 0.09–0.21: the budget only starts binding around
step 70 of 120 and the final playbook is ~1.6× the budget against formula's 2.8×. A zero does
**not** mean the arms agree — they still diverge through `refines` eviction and the graph's
annotation lines. What it costs is attribution.

Also record, for the results table: **the per-arm rate of generations that do not parse as
Python.** The earlier gemma runs showed 0/257 in `none` against 6/343 in `text`; if that
replicates it is a scoring-path artifact worth ~3–4 rows against effects of ~11–17, and it must
be reported beside the accuracies rather than repaired (the truncation is inside the model's own
valid JSON, so nothing is recoverable).

---

### 6. MBPP seeds 1 and 2 ⬜
**Cost: ~14 hours.** Same block as step 5 with `$s = 1`, then `$s = 2`. Nothing may change
between seeds except `--seed`.

---

### 7. Analyse and write up ⬜
**Cost: free.**

```powershell
uv run python compare_arms.py --selftest
uv run python compare_arms.py --force (Get-ChildItem 'results\mbpp\preregistered\seed*\*\ace_run_*' -Directory | Select-Object -ExpandProperty FullName)
uv run python aggregate_run_costs.py (Get-ChildItem 'results\mbpp\preregistered\seed*\*\ace_run_*' -Directory | Select-Object -ExpandProperty FullName)
```
Quote the **pooled** line as the headline. `--force` is only legitimate if `git_commit` is the
sole difference **and** the code diff over `ace/ eval/ llm.py utils.py playbook_utils.py` is
empty — check it, don't assume.

---

## Do NOT do these

- ❌ **More formula replicates.** Measured: even infinite replicates leave the CI at ~3.5pp
  half-width, which still contains zero for a 2.8pp effect. `formula_test.jsonl` is capped at 50
  formulas, so the power ceiling cannot be raised. Another 50 hours would change nothing.
- ❌ **Cite MBPP numbers from `MBPP_RESULTS.md`.** Those runs predate `--retrieval`, so the graph
  arm's treatment never fired. Superseded — see `EXPERIMENT_MBPP.md` §9.
- ❌ **Say "structure helps."** The data says memory helps and structure was not shown to.

## Later / optional

- **`--memory graph --retrieval flat` ablation** on formula (3 runs, ~10h). Separates "edges
  choose what is read" from "edges are shown alongside what is read". The only remaining
  question formula still has the power to answer.
- **LLM judge for `contradicts` edges** — future work, no implementation yet. Best run as a
  *fourth arm* (`graph` vs `graph+contradicts`) so it isolates the extra edge type and leaves
  the existing three-arm result untouched. Note it only earns the graph if it changes what the
  generator **reads**; if it merely evicts, a flat list reproduces it.
- **`num_epochs=5`** — the paper's setting, and our largest disclosed deviation. 5× the cost, and
  it moves both memory arms, so it does not isolate structure.
