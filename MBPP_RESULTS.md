# MBPP results: linear vs relational memory in ACE

> ## ⚠ SUPERSEDED — exploratory. Do not cite these numbers.
>
> **These runs do not implement the design being tested.** They predate `--retrieval`, so the
> graph arm's actual treatment — *retrieval follows `related_to` edges* — **never ran**; and
> they predate `--dedup`, so the baseline was ACE minus one of its three components. One run
> per arm, no noise floor. Full itemisation in **`EXPERIMENT_MBPP.md` §9**.
>
> **What this document is still good for**, and why it is kept:
> - **§2** — the `qwen3.5:2b` bracket-wrapping scoring defect, its arm-specificity, and the
>   guarded fix. Fixed in the live path since; the write-up is the record of how it was found.
> - **§5** — the criticism list, which is the input to the new pre-registration.
> - **§6** — *why MBPP is the wrong instrument for accumulation*. This is the durable finding
>   and it survives everything above: MBPP problems share no transferable content, so a curator
>   can only accumulate style advice.
>
> **The experiment that will produce citable MBPP numbers is pre-registered in
> `EXPERIMENT_MBPP.md` and has not been run.** For results, see
> `FORMULA_RESULTS_PREREGISTERED.md` (complete, nine runs).

Report date: 2026-08-04. Runs executed 2026-08-03/04, all from commit
`82b0b6568ce0b7bfd6aaf917ecc387724f3acf2e`.

**Summary.** Three arms — no memory, flat playbook, relation graph — were compared on
MBPP-sanitized. After correcting a scoring defect found during analysis, the flat playbook
scores highest (0.362) and the graph arm ties the memory-free control (0.335). **No pairwise
difference is statistically significant** (McNemar exact, all p > 0.4). The memory is not
inert: it changes the outcome on roughly a quarter of test problems. It changes them in
both directions in almost equal measure.

This is reported as a null. The rest of the document is mostly about why that null is
weaker evidence than it looks, and what would have to change to get a real answer.

---

## 1. Setup

| | |
|---|---|
| task | `mbpp_code_exec`, MBPP-sanitized |
| splits | 120 train / 43 val / 257 test |
| generator | `qwen3.5:2b`, reasoning suppressed (`--think_generator off`) |
| reflector, curator | `gemma4:e2b`, reasoning left at model default |
| decoding | `temperature=0.0`, `seed=0`, `num_ctx=32768` (pinned), `max_tokens=16384`, streamed |
| loop | 1 epoch, `max_num_rounds=1`, `curator_frequency=5`, `eval_steps=60`, `test_workers=1` |
| bulletpoint analyzer | off (it performs merges the graph arm cannot mirror) |
| graph settings | `related_threshold=0.60`, `contradicts` **disabled**, `keep_superseded=false` |

Arms differ in exactly one flag, `--memory`:

- **`none`** — no accumulated memory, `eval_only`. The pure-LLM control.
- **`text`** — upstream flat playbook. The ACE baseline.
- **`graph`** — the same rules as nodes in a typed relation graph, with `refines` withdrawing
  superseded rules from the rendered prompt.

`aggregate_run_costs.py` audits the three `run_config.json` files and reports **no fairness
violations**: identical models, seed, `num_ctx`, per-role reasoning flags, `test_workers`,
`curator_frequency`, `eval_steps`, `max_num_rounds`, analyzer setting and git commit.

The curator prompt is byte-identical in all three arms, including the `refines` field that
only the graph arm acts on. The baseline is therefore honestly described as *ACE with an
extended curator schema whose extra field it ignores*.

---

## 2. The scoring defect

Found while auditing the results, not before. It matters enough to come before the numbers.

`utils.extract_answer` pulls the program out of the generator's JSON envelope.
`qwen3.5:2b` sometimes writes the program as a **JSON array literal inside the string
value**:

```json
"final_answer": "[\"def min_of_three(a,b,c):\n    return min([a,b,c])\"]"
```

After `json.loads`, the value is a `str` that still begins `["` and ends `"]`. The sandbox
receives a bare expression instead of a function definition, every assertion fails on an
undefined name, and the sample scores zero. The code inside is correct.

`unwrap_quoted_source` already handled the plain-quote spelling of this habit but not the
bracketed one. Frequency, over all parsed generator responses in each run:

| arm | generator calls | responses whose `final_answer` parsed as a string | bracket-wrapped | rate |
|---|---:|---:|---:|---:|
| none  | 257 | 230 | **0** | 0.0% |
| text  | 667 | 599 | **68** | 11.4% |
| graph | 660 | 552 | **6** | 1.1% |

**The defect was almost exclusive to one arm.** In the final test set specifically it
mangled 28 of the text arm's 257 answers (10.9%) and none in the other two arms. The
between-arm accuracy differences being argued about were 5 problems.

### 2.1 What was fixed, and why it cannot have favoured one arm

`utils.unwrap_quoted_source` now recognises `["…"]` and `['…']` alongside the quote forms.
The `ast.parse` guard is unchanged: it unwraps only when the answer does not already parse
*and* unwrapping makes it parse. That makes the fix **one-directional by construction** — it
can only fire on an answer that already scored zero, so it can turn 0 into 1 and never 1
into 0. `extract_answer` additionally unwraps a genuine one-element JSON array, a defensive
fix for a spelling not present in these runs.

`rescore_from_logs.py` then re-scored all three arms from the saved generator logs — same
responses, corrected extraction, re-executed in the sandbox. No LLM calls.

Re-extracting the whole final test both ways, with the bracket forms as the only variable:

| arm | test answers | changed by the fix | already parseable | became parseable | broke |
|---|---:|---:|---:|---:|---:|
| none  | 257 | **0** | 0 | 0 | 0 |
| text  | 257 | 28 | **0** | 28 | **0** |
| graph | 257 | **0** | 0 | 0 | 0 |

The fix touched **zero** answers outside the text arm, so it cannot have moved that arm
relative to the others by altering them. Within the text arm every change was
unparseable → parseable; nothing that already worked was touched. Of the 28 recovered
answers, **12 then passed their assertions** and 16 still failed on their logic — which is
the expected shape. Recovering the text of a program does not make the program right, and
28 out of 28 flipping to correct would have indicated a scoring bug rather than a fix.

**The rescore repairs the measurement. It does not repair the training.** During training
the text arm emitted 29 bracket-wrapped generations, of which 26 contain valid Python. At
run time those were scored as failures, and those failures were fed to the reflector and
then to the curator. The text arm's playbook was therefore built from a partly spurious
error signal that the graph arm (4 such generations) largely did not see. No offline
correction can undo that; only re-running can.

### 2.2 An attribution I could not support

The obvious hypothesis is that the flat playbook's top-cited rule, `[code-00002] "Use list
comprehensions or generator expressions…"`, primed the bracket habit. It does not hold up:
`code-00002` is cited in 86.8% of the text arm's 68 bracket-wrapped generations and 80.8% of
its 531 normal ones.
That is not a discriminating signal. **The habit is strongly arm-specific and I cannot say
why.** Recording it as unexplained rather than guessing.

---

## 3. Results

### 3.1 Final test accuracy (n = 257)

| arm | correct | accuracy | Wilson 95% CI | no answer | unparseable |
|---|---:|---:|:--:|---:|---:|
| none  | 86 | 0.3346 | [0.280, 0.394] | 3 | 22 |
| text  | 93 | **0.3619** | [0.306, 0.422] | 7 | 10 |
| graph | 86 | 0.3346 | [0.280, 0.394] | 10 | 12 |

As originally scored, before the fix: none 86 (0.3346), text **81** (0.3152), graph 86
(0.3346). Only the text arm moved (+12 problems); the other two reproduced exactly, which
is the control that shows the fix is not a general scoring liberalisation.

All three intervals overlap heavily. On n = 257 the half-width is about ±5.8pp, so this
experiment could only ever have detected differences larger than roughly 6 points.

### 3.2 Paired comparison (McNemar exact, two-sided)

`b` = problems the second arm fixed; `c` = problems it broke.

| comparison | both right | both wrong | b | c | net | p |
|---|---:|---:|---:|---:|---:|---:|
| none → text  | 62 | 140 | 31 | 24 | +7 | 0.419 |
| none → graph | 55 | 140 | 31 | 31 | **0** | 1.000 |
| text → graph | 54 | 132 | 32 | 39 | −7 | 0.477 |

Three pairwise tests, no multiple-comparison correction applied; none approaches
significance so it makes no difference here.

**The interesting number is not p, it is `b + c`.** Adding a flat playbook changes the
outcome on 55 of 257 problems (21%). Adding a graph changes 62 (24%). Swapping flat for
graph changes 71 (28%). The memory is doing a great deal of work and almost exactly
cancelling itself out. `none → graph` at b = c = 31 is the cleanest statement of it: 31
problems fixed, 31 broken, zero net.

A memory with a large, directionless effect is a different diagnosis from a memory that is
ignored — and it points at rule *quality*, not at rule *representation*.

### 3.3 Cost

| arm | LLM calls | tokens | LLM hours | memory tokens |
|---|---:|---:|---:|---:|
| none  | 257 |   251,990 | 0.25 | – |
| text  | 811 | 2,089,239 | 2.08 | 1,169 |
| graph | 804 | 1,808,977 | 1.64 | 1,265 |

Memory costs roughly **8× the tokens of the pure-LLM control** for no measurable accuracy
gain.

One finding worth keeping: the graph rendering was projected to cost about +33% tokens over
the flat one, and came in at **+8%** (1,265 vs 1,169). `refines` withdrew 6 of 26 nodes from
the prompt, and that eviction paid for most of the annotation overhead. The graph arm also
used 13% *fewer* total tokens than the text arm, because the text arm spent more on
regeneration after reflection. So the relational layer is close to free at this scale. It
simply did not buy anything.

---

## 4. The loop itself worked

Three defects fixed before these runs — a never-emitted `harmful` tag, a never-emitted
`refines` field, and a reflector that never saw why code failed — were all verified live:

| | text | graph |
|---|---|---|
| `harmful` tags emitted | 47 | 24 |
| `refines` present in curator responses | 24/24 | 24/24 |
| `refines` non-null (real supersession) | 8 | 6 |
| generations citing a playbook rule | 117/120 | 113/120 |
| distinct rules cited | 15 | 25 |
| train accuracy (headroom) | 0.300 | 0.358 |
| generator output unparseable (after fix) | 9.1% | 7.7% |

The graph arm ended with 26 nodes of which 6 were withdrawn by `refines`, 80 `related_to`
edges, 7 connected components. The mechanism under test fired. It did not help.

`check_learning_signal.py` passes on both arms after the extraction fix; before it, the text
arm failed check [5] at 18.9% unrunnable.

---

## 5. Criticism

Stated fully, because the null is only as strong as this list is short — and it is not
short.

1. **Single run per arm.** One run per arm at `seed=0`. There is no estimate of run-to-run
   variance, so it is not possible to say whether a 2.7pp gap is an effect or noise. This
   is the single biggest weakness. Everything below is secondary to it.
   **Since measured, and worse than it looked:** re-running an arm changes the answer even
   with every flag identical, because the serving stack is not bit-reproducible and the ACE
   loop amplifies that — on `formula`, a memory arm moved 3.5pp and 37 of 200 rows between
   two nominally identical runs (`EXPERIMENT_FORMULA.md` §5). A 2.7pp gap is well inside
   that. Note also that `--seed` does not fix this: it never reaches the data and at
   temperature 0 the sampler never reads it (§4 there, and the `--seed` gotcha in
   `CLAUDE.md`).

2. **Underpowered.** n = 257 gives ±5.8pp. Two of the three observed gaps are smaller than
   that. The experiment was not capable of resolving the differences it was run to detect.

3. **The training signal was not equalised — this is the deepest problem with the corrected
   number.** §2.1: the text arm trained under ~26 spurious failures the graph arm did not
   have. The rescore corrects the score, not the playbook that produced it. Strictly, the
   corrected text number is *the text arm's playbook, built under a noisier signal, scored
   fairly* — not *a fair text arm*. Whether that noise helped or hurt its playbook is
   unknown and unknowable without re-running.

4. **Checkpoint selection on 43 validation rows.** The text arm's best checkpoint was step
   120 (val 0.372), the graph arm's step 60 (val 0.372, against 0.279 at step 120). A tie on
   43 samples decided which playbook each arm carried into the test set. That is model
   selection driven by noise, and it differed between arms.

5. **The playbooks are degenerate.** The text arm's 24 rules include **14 near-verbatim
   restatements** of "favor immutable data types / avoid mutating shared state". Its
   most-cited rule ended at `helpful=18 harmful=40` — net harmful and cited in 81% of
   generations. The graph arm has the same problem in milder form: `prob-00007/8/9/13` are
   one sentence written four times, and `refines` withdrew only 6. Neither memory
   accumulated content worth structuring.

6. **`contradicts` edges were disabled.** The NLI cross-encoder scores prescriptive advice
   as contradictory at saturation (19 edges, ~3 genuine, "prefer pure functions" vs "prefer
   function composition" at 1.000). The graph arm therefore shipped with `related_to` and
   `refines` only, i.e. a reduced version of the treatment.

7. **The `keep_superseded` ablation was not run.** Without it, "the graph annotated the
   rules" and "the graph withdrew stale ones" are not separated. Given the null, running it
   would not have been informative, but it means the graph's mechanism is untested in
   isolation.

8. **A known harness wart.** `ace.py` calls `log_bullet_usage` twice for correct training
   samples and once for incorrect ones. Any raw row count over-reports training accuracy
   (e.g. 0.537 instead of 0.367 on an earlier probe). `check_learning_signal.py` deduplicates
   on `(epoch, step)`; the figures in §4 are the deduplicated ones and match ACE's own
   `pre_train_accuracy`.

9. **Not blinded, and analysed by the same person who fixed the harness.** The extraction
   defect was found after seeing the results, so the fix was made with knowledge of which
   arm it would help. Three things bound how much that can matter: the fix lives in shared
   code and ran identically on all three arms; it is guarded so it can only ever raise a
   zero, never lower a one; and measurably it altered 0 answers in the two arms it did not
   help (§2.1). It remains a post-hoc correction, and a pre-registered re-run is the only
   thing that fully removes the concern.

10. **The reported figure of 22 was wrong in the first draft of this report.** The correct
    count of mangled text-arm answers is 28; 22 came from a coarser diagnostic that missed
    cases with trailing punctuation. Corrected 2026-08-04. The accuracy figures were never
    affected — they came from the sandbox, not from that count.

11. **The arms were scored through different amounts of extraction machinery, and §2.1 did
    not measure that.** §2.1 shows the *bracket-wrapper fix* was arm-neutral, which it was.
    It does not show that the extractor as a whole treated the arms alike. Classifying every
    final-test answer by the path that recovered it (added to `compare_arms.py`
    2026-08-05, after this report's first draft) gives:

    | arm | strict JSON parse | escape repair | regex fallback | no answer | not strict |
    |---|---|---|---|---|---|
    | none  | 230 (89.5%) | 1 (0.4%) | 23 (8.9%)  | 3 (1.2%)  | 10.5% |
    | text  | 231 (89.9%) | 0        | 19 (7.4%)  | 7 (2.7%)  | 10.1% |
    | graph | 208 (80.9%) | 0        | 39 (15.2%) | 10 (3.9%) | 19.1% |

    The graph arm needed the regex fallback about twice as often as the other two — an
    **8.9-point spread**, larger than any accuracy difference reported here. Two readings
    are available and this design cannot separate them: the graph arm's longer rendered
    playbook genuinely changed how the generator formatted its output, or the graph arm was
    scored more leniently because more of its answers were rescued by pattern-matching. The
    fallbacks are guarded — each runs only after a strict parse has failed and cannot alter
    a well-formed response — so this does not invalidate an individual score. It does mean
    the graph-vs-others contrast carries a scoring-path confound that was not reported in
    the first draft. `compare_arms.py` now prints this table and warns above a 5-point
    spread, so the formula arms cannot repeat the omission.

---

## 6. Interpretation

MBPP appears to be the wrong instrument for this question, independent of the null.

**Its problems share no transferable content.** Each MBPP task is a self-contained
algorithmic exercise. Nothing learned on problem 7 applies to problem 84 except style. So
the only thing a curator can accumulate is style advice — and §5.5 shows that is exactly
what it accumulated, in triplicate. Asking whether a graph organises that better than a list
is asking which container better organises noise.

The paired counts in §3.2 support this reading: memory flips a quarter of problems in both
directions. That is the signature of advice that is sometimes apt and sometimes actively
misleading, not of advice that is being ignored.

For contrast, the `formula` task in `eval/finance` — which the ACE authors used — has
**50 distinct formulas, all 50 present in train, val and test**, roughly 10 training and 4
test instances each, with zero verbatim question overlap between splits. A convention
learned on the 10 training instances of a formula applies directly to its 4 test instances.
The conventions are also genuinely error-prone: percent versus fraction
(`Operating Margin = … × 100` → `15.0`), rounding, sign, unit scaling. That is transferable
content, and it is a setting where 50 topic clusters give a relational memory something to
organise.

## 7. Conclusion

On MBPP-sanitized, with `qwen3.5:2b` as generator and a single seed, **neither flat nor
relational memory produced a detectable improvement over no memory**, and the two memory
representations did not differ from each other. Flat memory scored numerically highest;
the difference is not significant and is smaller than the confidence interval.

~~The graph mechanism worked as designed — `refines` withdrew 6 of 26 rules and the relational
rendering cost only 8% extra tokens — so the null is about the value of the structure, not
about whether it was implemented.~~

**Retracted 2026-08-15. This sentence was wrong, and it is the reason the whole report is
superseded.** What fired here was `refines`, i.e. eviction — which a flat list reproduces with
a `supersedes` field and a render-time filter, and which therefore needs no graph. The graph's
actual capability, *retrieval following `related_to` edges*, did not exist until 2026-08-06:
`render_for_generator(query=...)` raised `NotImplementedError` in both arms at this commit. So
the null **is** about whether the treatment was implemented. Edges were built, then read only
for display text and `structural_stats()`, which no model ever sees.

Given §5.1 (single seed), §5.3 (unequal training signal), §6 (benchmark unsuited to
accumulation) and the retraction above, **this result should be read as "MBPP cannot answer the
question", not as "relational memory does not work".**

**What happened next, added 2026-08-15.** The `formula` experiment ran instead, with three
replicates per arm and per-problem predictions saved by default, both of which these runs
lacked. It is complete and reported in `FORMULA_RESULTS_PREREGISTERED.md`: memory helps
(`none → graph` +8.0pp, p=0.0146), structure does not measurably (`text → graph` +2.8pp,
p=0.2905 — null, and under-powered against a noise floor it cannot clear).

**`EXPERIMENT_MBPP.md` is the pre-registration for the MBPP re-run**, and it takes §5 of this
document as its starting point. Its three runs per arm are **replicates**, not seed variations —
`--seed` varies nothing (`EXPERIMENT_FORMULA.md` §4) — and what they buy is a measured noise
floor rather than a tighter interval.

---

## Reproducing

```
uv run python rescore_from_logs.py results/mbpp/arms_qwen_seed0/none \
  results/mbpp/arms_qwen_seed0/text results/mbpp/arms_qwen_seed0/graph
uv run python compare_arms.py results/mbpp/arms_qwen_seed0/none \
  results/mbpp/arms_qwen_seed0/text results/mbpp/arms_qwen_seed0/graph
uv run python aggregate_run_costs.py results/mbpp/arms_qwen_seed0/none \
  results/mbpp/arms_qwen_seed0/text results/mbpp/arms_qwen_seed0/graph
uv run python check_learning_signal.py results/mbpp/arms_qwen_seed0/text \
  results/mbpp/arms_qwen_seed0/graph
```

`rescore_from_logs.py` needs Docker. The superseded `gemma4:e2b` pilot from 2026-08-02 used to
sit alongside the control in the same directory and had to be excluded by hand; it now lives in
`results/mbpp/learning_signal_gemma/`, so the three paths above are exactly this comparison.

Per-problem outcomes for every arm are in `rescored_test_predictions.jsonl` inside each run
directory.
