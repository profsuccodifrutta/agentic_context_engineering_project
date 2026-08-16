# TODOs — what to do next

Last updated 2026-08-16. Working checklist, not a research document. For the reasoning behind
any item, follow the link on it.

## Where things stand

| | status |
|---|---|
| **formula** | ✅ **done** — 9 runs, analysed, written up in `FORMULA_RESULTS_PREREGISTERED.md` |
| **MBPP** | 🔜 pre-registered in `EXPERIMENT_MBPP.md`, **not run**. Setup steps 1–2 done. |
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

### 3. MBPP probes ⬜
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

### 4. Pick `--dedup_threshold` ⬜
**Cost: free — offline replay of the probe's ADDs, no LLM.**

Do **not** reuse formula's 0.88; it was calibrated on financial-convention rules, not style
advice. Rule: largest threshold whose absorption stays under 50% in **both** memory arms.

➡️ *After step 3, ask me to run the sweep. I'll give you the number to paste into step 5.*

---

### 5. MBPP seed 0 — three arms ⬜
**Cost: ~7 hours. Run seed 0 alone first and inspect it**, the way you did on formula.

```powershell
$common = @(
  '--task_name','mbpp_code_exec',
  '--config_path','./eval/mbpp_code_exec/data/task_config_v2.json',
  '--generator_model','gemma4:e2b','--reflector_model','gemma4:e2b','--curator_model','gemma4:e2b',
  '--think_generator','off','--num_ctx','32768',
  '--dedup','--dedup_threshold','<FROM STEP 4>',
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
```
The one that invalidates everything: **`walk_only_share` near zero in the graph arm** means the
treatment never fired and the run is broken, not negative.

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
