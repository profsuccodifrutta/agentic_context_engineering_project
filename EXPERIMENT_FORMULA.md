# Pre-registration: relational vs linear memory on `formula`

**Written 2026-08-04, before any `formula` run.** Nothing below may be changed after the
first test number is observed. If something has to change, the change and its date go in
§8 with a reason, and the affected result is reported as exploratory.

## 1. Question

Does representing an agent's accumulated rules as a typed relation graph produce better
task performance than the same rules held as a flat list?

## 2. Why this task

MBPP returned a null (`MBPP_RESULTS.md`) and, more importantly, is structurally unsuited:
its problems are independent, so nothing accumulated on one transfers to another and the
curator can only write style advice. `formula` has the transfer structure MBPP lacks:

- 50 distinct finance formulas, **all 50 present in train, val and test**
- 5 training instances per formula (subsampled), ~4 test instances per formula
- zero verbatim question overlap between splits
- the authors' preprocessing **strips the formula definition from the prompt**, so the
  model must recall it — making the formula itself accumulable playbook content
- per-formula conventions are genuinely error-prone: percent vs fraction, rounding, sign,
  unit scaling

It is also the task the ACE authors used, so this doubles as a replication check on the
implementation.

## 3. Arms

Three, differing in exactly one flag:

| arm | `--memory` | mode |
|---|---|---|
| control | `none` | `eval_only` |
| baseline | `text` | `offline` |
| treatment | `graph` | `offline` |

Everything else held identical: `qwen3.5:2b` generator with reasoning off,
`gemma4:e2b` reflector and curator at model default, `num_ctx=32768`,
`temperature=0.0`, `max_tokens=16384`, `max_num_rounds=1`, `curator_frequency=5`,
**`eval_steps=125`**, `test_workers=1`, bulletpoint analyzer off, `contradicts` edges off,
`related_threshold=0.60`, `keep_superseded=false`. Train order is file order, no shuffle.

`eval_steps=125` gives two validation checkpoints (steps 125 and 250), matching the MBPP
design. See §9 — at 60 it would give four, and validation is 300 rows, so the checkpoints
alone would be 1,200 generations per training arm and would dominate the budget.

`--seed 0`, `--seed 1`, `--seed 2` for each arm: **9 runs**.

## 4. Data

Fixed now, not after seeing results.

| split | file | n |
|---|---|---|
| train | `formula_train_subset_250.jsonl` | 250 (5 per formula, stratified) |
| val | `formula_val_subset_300.jsonl` | 300 |
| test | `formula_test.jsonl` | **200** |

Test is the authors' split, untouched, and is not looked at until §6 runs.

## 5. Primary outcome and analysis

**Primary metric:** accuracy on the 200-row test set, using the best-validation checkpoint,
scored by `DataProcessor.answer_is_correct` (exact float equality after
`normalize_numeric`).

**Primary comparison:** graph vs text, paired by problem index, McNemar exact two-sided,
α = 0.05, seed-blocked across the three seeds. Reported with the discordant counts `b`
and `c`, never the p-value alone.

**Secondary:** text vs none and graph vs none, same test. Three pairwise tests are reported
together with no correction; any single p < 0.05 is interpreted accordingly.

**Power, stated in advance:** n = 200 gives a Wilson half-width of about ±7pp. This design
can detect a large effect and cannot detect a small one. A null is therefore evidence about
large effects only, and will be reported that way.

**Decision rule.** Graph beats text only if the paired test is significant *and* the mean
gap across the three seeds exceeds the within-arm standard deviation. A gap smaller than
the seed spread is reported as noise regardless of p.

## 6. Procedure, in order

1. `check_prompts.py`, `check_finance_wiring.py`, `check_memory_equivalence.py` all pass.
2. Working tree committed; all 9 runs share one git SHA.
3. **Probe** — `--memory text --mode offline --config_path ./eval/finance/data/probe_config.json`
   (50 train / 20 val / 20 test, 10 formulas × 5 training instances, `--eval_steps 25`). Then
   `check_learning_signal.py`. Gates, all of which must pass before spending the rest:
   - `harmful` emitted at least once
   - `refines` present in every curator response, at least one non-null
   - train accuracy inside 0.20–0.85
   - output integrity within limits
4. **Floor/ceiling gate** — `none` arm at seed 0 on the real test set. If accuracy is below
   0.15 or above 0.85 the design is dead: no memory can move it, and the generator must be
   reconsidered before any further run. This decision is made on the *control* arm only.
5. **The nine runs.**
6. Analysis exactly as §5, via `compare_arms.py`; cost via `aggregate_run_costs.py`.

## 7. What would falsify the hypothesis

Graph ≈ text on the primary comparison, with `refines` demonstrably firing (non-null in
curator responses, nodes withdrawn from the rendered playbook). That is a real negative
result about relational structure, and it will be reported as one.

Graph ≈ text with `refines` *not* firing is not a result about structure — it is a broken
run, and would be reported as such.

## 8. Deviations from the ACE authors' setup

Recorded now, and to be repeated in the write-up.

1. **Models.** `qwen3.5:2b` / `gemma4:e2b` on a local Ollama endpoint, not DeepSeek-V3.1.
   Absolute accuracy is therefore not comparable to the published numbers; only the
   between-arm contrast is.
2. **Prompts** are the authors' finance prompts (`ace/prompts/formula.py`) with two changes,
   both applied identically in all three arms:
   - `refines`, required-but-nullable, added to the curator schema. It is the graph arm's
     only capability; without it the treatment does not exist. Required-but-nullable
     because on MBPP an optional field was emitted 0 times in 24 calls.
   - Tag semantics added to the reflector: tags judged by causal impact on this attempt,
     `helpful` unavailable when the environment reports failure. Upstream defines none of
     the three tag values, and on this stack that produced `harmful` 0 times in 120
     reflections, with 15 `helpful` tags on attempts that had just failed.
3. **`normalize_numeric`** is applied to predictions and targets before comparison ($ , %
   commas, quotes, scale words, and a lone number inside a sentence). The authors relied on
   a model that obeys "no explanations or other text". Identical in all arms, including the
   control. The un-normalised numbers will be reported alongside.
4. **Train subsampled to 250** of 500, stratified to exactly 5 instances per formula, to
   afford three seeds. Val and test untouched.
5. **`contradicts` edges disabled.** The NLI cross-encoder scores prescriptive advice as
   contradictory at saturation. The graph arm ships with `related_to` and `refines` only.
6. **`--think_generator off`.** Both models are reasoning models; left on, the generator
   spends its output budget before answering. Reflector and curator keep model default.

## 9. Deviations log

**2026-08-04, first probe run, before any arm.** Two extraction defects found by the probe
and fixed in `utils.extract_answer`. Both only run after the strict parse has already
failed, so neither can change how a well-formed response is read, and both are in shared
code that runs identically in all three arms.

- *Invalid JSON escapes.* The generator writes its working in LaTeX — `\frac`, `\sigma`,
  `\approx` — inside the `reasoning` field. Those are not legal JSON escapes, so
  `json.loads` rejected the whole envelope even though the answer field was intact:
  **22 of 29 unparseable probe responses**. `repair_json_escapes` retries once with the
  unrecognised backslashes escaped.
- *Unquoted numeric answer.* Both regex fallbacks required `final_answer` to be a quoted
  string. This task's answer is a number and the model writes it bare
  (`"final_answer": 229.27`, sometimes with a stray closing quote): **15 of 29** were
  discarded with the correct answer present in the text. A numeric-literal fallback was
  added, restricted to a number so it cannot swallow a fragment of a broken object.

Effect on the probe: usable answers rose from 28 to 43 of 46. Effect on the completed MBPP
runs: **none** — re-scoring all three arms gives 86 / 93 / 86 exactly as before, so
`MBPP_RESULTS.md` is unchanged. The probe was re-run after the fix; the pre-fix probe is
discarded.

**2026-08-04, second probe run, before any arm.** Reflector reasoning-trace cap added
(`ace/core/reflector.py::MAX_TRACE_TOKENS = 4000`). At training step 44 the reflector
returned empty fifteen consecutive times. Cause: the generator's reasoning trace is passed
into the reflector prompt verbatim and is the only field there with no bound; a 36,268-char
trace made up 89.5% of a 40,522-char prompt (14,675 tokens). **`--num_ctx 32768` is not what
is served** — Ollama clamps it to the model's own trained context, and on `gemma4:e2b` every
reflector call at or below 12,433 tokens succeeded while the 14,822-token one was left 147
tokens of output budget, spent them all on hidden reasoning, and produced nothing. The step's
learning signal was lost and fifteen retries were paid for it.

The cap keeps the trace's head and tail with a marker between; on the real failing call it
takes the prompt from 14,675 to 4,890 tokens. Applied identically in every arm, and
deliberately **not** applied to `bullets_used`, which is memory content, renders longer in
the graph arm, and would bias the comparison if truncated. It strictly increases the
information the reflector receives — the status quo was an empty response.

The 3 remaining unusable probe responses are genuine model failures, not harness defects:
the generator produced 3k–16k tokens of reasoning and never emitted `final_answer`. At
~6.5% that is inside check [5]'s limit and is reported rather than corrected. Note the cost
shape it implies — median response 723 tokens, p90 8,128 — the tail dominates the budget.

**2026-08-04, before any formula run.** `eval_steps` 60 → 125. Reason: budget, decided on
call counts alone with no test number observed. Validation is 300 rows and 250 training
steps at `eval_steps=60` would fire four checkpoints, i.e. 1,200 validation generations per
training arm — more than every other phase combined, and roughly doubling the cost of the
nine runs. 125 gives two checkpoints, the same number MBPP used. The cost is coarser
checkpoint selection (2 candidates per arm instead of 4), which is a known and already
stated weakness of the MBPP design (`MBPP_RESULTS.md` §5.4) and is now identical across
arms by construction.
