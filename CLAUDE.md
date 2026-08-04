# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ACE (Agentic Context Engineering) — LLMs self-improve by evolving accumulated rules through a
Generator / Reflector / Curator loop instead of fine-tuning weights. Fork of `ace-agent/ace`
(arXiv:2510.04618).

This fork's research question: **does relational memory beat linear memory for agent execution?**
Three arms, differing only in `--memory`:

| arm | what it is | what it answers |
|---|---|---|
| `none` | no accumulated memory, `eval_only` only | does memory help at all? |
| `text` | upstream flat playbook | the ACE baseline |
| `graph` | same rules as nodes in a typed relation graph | does *structure* help? |

Everything else is held identical across arms — same prompts, same data order, same seed, same
context window. That is the point, and it constrains most design decisions here.

The graph's actual capability is **`refines`**: the curator names an existing rule its new rule
supersedes, and the superseded rule stops being rendered into the prompt while staying in the graph.
That gives the graph arm eviction. The flat playbook cannot do it — it would need `DELETE`, which
upstream ACE never implemented, so its first draft of a rule keeps being injected forever alongside
every later correction. `related_to` edges are annotation on top; `--keep_superseded` is the ablation
that separates the two.

- `ace/memory/` — the swap seam. `base.py` (Memory ABC), `text.py` (flat playbook, delegates to
  `playbook_utils` so the baseline is provably unchanged), `graph.py` (NetworkX), `embeddings.py`
- `ace/core/` — `generator.py`, `reflector.py`, `curator.py`, `bulletpoint_analyzer.py`
- `ace/prompts/` — prompt templates, one per core module
- `eval/<task>/` — one dir per benchmark, each with its own `run.py` + `data_processor.py`
- `llm.py`, `utils.py`, `logger.py`, `playbook_utils.py` — shared infra at repo root
- `ace-appworld/` is an uninitialized submodule — ignore it

## Setup and running

Package manager is **uv** (`uv sync`). Python 3.12 pinned in `.python-version`. Evals run as modules
from the repo root:

```
uv run python -m eval.mbpp_code_exec.run --mode offline   --save_path results --memory text  --num_ctx 32768 --seed 0
uv run python -m eval.mbpp_code_exec.run --mode offline   --save_path results --memory graph --num_ctx 32768 --seed 0
uv run python -m eval.mbpp_code_exec.run --mode eval_only --save_path results --memory none  --num_ctx 32768 --seed 0
```

`--memory none` refuses `offline`/`online`: it accumulates nothing, so training would spend
generator, reflector and curator calls on a memory that provably cannot change.

Modes: `offline`, `online`, `eval_only` (pass `--initial_playbook_path` for `eval_only`).
No `--num_samples` flag — to run a subset, point `--config_path` at a config referencing a truncated
JSONL.

The `formula` task in `eval/finance` is the second benchmark, and the one the ACE authors used:

```
uv run python -m eval.finance.run --task_name formula --mode offline --save_path results/formula_text \
  --memory text --generator_model qwen3.5:2b --think_generator off --num_ctx 32768 --seed 0 \
  --skip_initial_test
```

Its flags are identical to MBPP's by design. Splits default to `task_config.json`
(250 train / 300 val / 200 test); `probe_config.json` is the 50/20/20 probe. Unlike MBPP it needs
no Docker. **`formula` is the better instrument for the research question**: 50 formulas appear in
every split with ~5 training and ~4 test instances each, and the authors' preprocessing strips the
formula definition from the prompt, so what the playbook accumulates is directly reusable. MBPP's
problems are independent, so nothing transfers and the curator can only write style advice — which
is what it wrote. See `MBPP_RESULTS.md` and `EXPERIMENT_FORMULA.md`.

There is no test suite. Four no-network checks stand in for one; run them before spending compute:

- `uv run python check_memory_equivalence.py` — proves the text arm is byte-identical to the
  pre-refactor code path, checks the cross-arm invariants, and runs the whole training loop against
  both arms with the LLM stubbed.
- `uv run python check_prompts.py` — proves the MBPP generator prompt is unchanged by prompt
  selection, and that the finance prompts bind their fields correctly. **The finance generator
  template orders its fields playbook → reflection → question → context while
  `Generator.generate` supplies question, context, playbook, reflection**; bound positionally it
  puts the question under the "Playbook" heading in every call, silently. `PromptSet.format_generator`
  is what prevents that, and this check is what proves it.
- `uv run python check_finance_wiring.py` — runs the full offline loop on the real probe splits for
  the text and graph arms with the LLM stubbed; covers processor signatures, prompt selection,
  environment feedback and the artifacts the analysis tools read.
- `uv run python check_learning_signal.py <run_dir>` — see "The learning signal" below.

`uv run python check_streaming.py` is the companion check that *does* hit the network: it confirms
streamed calls return content, still report token usage, and produce byte-identical output to
non-streamed ones. Run it after any change to the transport in `llm.py` — a server that quietly
drops usage yields a complete-looking run with no cost data.

`uv run python calibrate_graph_thresholds.py <playbook.txt>` prints what each edge threshold actually
produces on a real playbook. Use it before trusting `--contradicts_threshold`.

After runs finish, `uv run python aggregate_run_costs.py <run_dir> [<run_dir> ...]` (or `--root
results`) sums `detailed_llm_logs/` into tokens, latency and cost per role and per loop phase, writes
`cost_summary.json` into each run dir, and prints a cross-arm table plus a fairness audit. Nothing
else in the harness aggregates cost, so "the graph arm cost X% more tokens for Y accuracy" is only
derivable here. The audit fails a run whose `num_ctx` varies call-to-call (auto-sizing — the arms got
different context windows), whose `num_ctx` was never recorded at all, that had the bulletpoint
analyzer on, or that differs from its sibling arms in seed, models, `test_workers`,
`curator_frequency`, `eval_steps` or `max_num_rounds`.

**`uv run python compare_arms.py <run_dir> ...` is where cross-arm claims come from.** Accuracy
alone cannot tell "the memory did nothing" apart from "the memory fixed 31 problems and broke 31" —
and on MBPP it was the second. It reads `*_test_predictions.jsonl`, prints Wilson intervals, and
runs a **paired McNemar exact test** per arm pair reporting the discordant counts `b` and `c`, not
just p. Given several runs of the same arm it also prints mean ± sd across seeds; **a between-arm
gap smaller than the within-arm sd is seed noise.** It refuses to compare runs that differ on
anything the fairness rules pin, unless `--force`.

`uv run python rescore_from_logs.py <run_dir>` re-scores a finished run's final test from its saved
generator logs — same responses, current extractor, re-executed. No LLM calls. Use it when a
scoring-side defect is found after the fact instead of re-running arms. It writes
`rescored_test_predictions.jsonl`, which `compare_arms.py` prefers over the live one. It is also the
only way to score runs from before per-problem predictions were persisted.

**`contradicts` edges are off by default (`--enable_contradicts`).** The NLI cross-encoder does not
work on this material: calibrated against a real 24-rule playbook it emitted 19 edges of which ~3
were genuine, scoring "Prioritize writing pure functions" against "Prioritize function composition"
at 1.000 confidence. NLI is trained on descriptive premise/hypothesis pairs; these rules are
prescriptive advice, and the model reads any two directives as competing. Scores are saturated (19
edges at 0.30, 14 at 0.95) so no threshold separates them. The graph arm currently ships with
`related_to` only. Re-enabling it needs a detector that understands prescriptive text — an LLM judge
at ADD time is the obvious candidate.

## Experimental fairness rules

Violating any of these silently invalidates a comparison:

- **Pin `--num_ctx` to the same value in both arms.** `llm.py` otherwise derives Ollama's context
  window from prompt length, and the graph rendering is longer than the flat one (+33% tokens with
  `related_to` only, +109% with contradicts on), so the arms would be decoded under different context
  windows. ACE warns at startup when it is unset.
- `--seed` defaults to 0. Without a seed the Ollama server RNG is random even at `temperature=0.0`.
- **`--think` must match across arms** (audited). It sends Ollama's reasoning switch;
  `--think_generator` / `--think_reflector` / `--think_curator` override it per role. **Set it
  per role, not globally.** The roles have opposite economics: the generator is ~80% of all calls,
  so thinking there is what a reasoning model blows its budget on, while the reflector and curator
  make only ~145 calls between them and those calls *are* the learning signal. `gemma4:e2b` is
  itself a reasoning model and was thinking in every run before this flag existed, so a blanket
  `--think off` would silently change the two components whose judgement was just repaired. The
  working configuration is `--think_generator off` alone.
  **The mechanism matters: use `reasoning_effort`, not `think`.** Ollama's own top-level `think`
  field belongs to `/api/chat` and is silently dropped on the OpenAI-compatible route we use.
  Measured on `qwen3.5:2b` against `ollama.caduceus.lol`, same vowel-counting prompt:

  | mechanism | tokens | time | reasoning |
  |---|---|---|---|
  | `think` | 5,716 | 52.8 s | 22,923 chars — ignored, and the harder prompt returned *nothing* |
  | `reasoning_effort` | 169 | 1.8 s | suppressed |

  `llm.py` defaults `think_mechanism` to `reasoning_effort` for that reason. It also records
  `reasoning_chars` on every call and warns once if `think=False` still returns reasoning, because
  a server that does not recognise a key ignores it rather than erroring — the failure is silent
  and looks like success. Verify any candidate with
  `uv run python check_model.py --model <name> --think off` before probing: one minute, no Docker,
  tries each mechanism in turn and also checks the model can hold the output schema. A reasoning
  model left thinking turns hard problems into `no_answer`, which is the same dead end as no
  headroom.
- **`--use_bulletpoint_analyzer` must be off.** It merges bullets, which is a removal the graph arm
  cannot mirror; ACE raises rather than silently skipping it for the graph.
- Hold `test_workers`, `curator_frequency`, `eval_steps`, `max_tokens` and all three model names
  identical. Train order is already deterministic (file order, no shuffle).
- `stats()` must return the same key set for every memory kind — it is interpolated into the curator
  prompt, so an extra key there is a prompt difference disguised as a metric. Structure-specific
  numbers go in `structural_stats()`, which is recorded but never shown to a model.
- **The curator prompt is identical in all three arms**, including the `refines` field that only the
  graph arm acts on. That is deliberate: if the graph arm's curator were prompted differently, the
  arms would differ in two places and no result could be attributed to the memory. The text arm
  simply ignores the field. Describe the baseline honestly as "ACE with an extended curator schema
  whose extra field it ignores".
- `best_memory` must be assigned via `.copy()`, never `=`. The old code relied on strings being
  immutable; a graph assigned directly would alias and "best" would track "current".
- **Prompts are per task, and the two sets bind their generator fields differently.**
  `get_prompts(task_name)` returns a `PromptSet`; anything unrecognised gets the code prompts, so
  adding a task never changes an existing one. `ace/prompts/formula.py` holds the ACE authors'
  finance prompts verbatim apart from two documented deltas (`refines` in the curator, tag
  semantics in the reflector), both applied identically in every arm — see `EXPERIMENT_FORMULA.md`
  §8. Never format a generator template directly: the finance one orders its fields
  playbook → reflection → question → context and the code one is positional, so
  `PromptSet.format_generator` exists to keep that from silently scrambling every call.
  `run_config.json` records `prompt_set`; two runs on different sets are not comparable.

## The learning signal

Run `20260802_220602` (text arm, full offline) produced test 0.8444 vs the `none` arm's 0.8521 —
but paired per-problem it fixed 17 and broke 19 (McNemar p=0.87). Not an inert memory: a memory
with a large effect and no direction. Three defects underneath it, all now fixed, all shared by
every arm:

- **`harmful` was never emitted.** Across 120 reflections the tags were helpful×98, neutral×47,
  harmful×0 — including 15 `helpful` tags on attempts that had just failed their assertions. The
  reflector prompt now defines the tags by causal impact on *this* attempt and states that
  `helpful` is unavailable when the environment reports failure, and
  `Reflector._filter_bullet_tags` enforces it by demoting `helpful`→`neutral` when
  `attempt_is_correct=False`. It never manufactures `harmful` — blame stays a model judgement, and
  all-`neutral` is a legitimate answer. Prompt *and* code, because the prompt alone is what failed.
- **`refines` was never emitted** — 0 of 24 curator calls, not even as `null`, while 15 of the 25
  final rules sat in 6 near-duplicate clusters ("favor immutable data types" appeared three times).
  Since `refines` is the graph arm's only real capability, a graph run under that curator would
  have been a guaranteed null. The field is now **required-but-nullable** in both curator prompts,
  which is the standard fix for a model skipping an optional key. The validator already tolerated
  `null`, so nothing downstream changed.
- **The reflector never saw why the code failed.** `environment_feedback` was the hardcoded string
  `"Predicted answer does not match ground truth"`, while the sandbox already had
  `Test failed: <assertion> - <exception>` in stderr and threw it away. `answer_is_correct` now
  takes an optional `feedback` dict out-parameter; `utils.check_answer` fills it when the processor
  supports it (detected from the signature, so the stale `finance`/`mind2web` processors still
  work) and `utils.format_environment_feedback` renders it, degrading to the old one-liner when
  nothing was captured. This is the reason accumulated rules came out as generic style advice
  rather than task strategy — the reflector was guessing at causes.

`uv run python check_learning_signal.py <run_dir>` re-checks all of this on any finished run and
exits non-zero if the signal is dead. Run it on a short probe
(`--config_path ./eval/mbpp_code_exec/data/probe_config.json`, 25 train / 10 val / 10 test) before
committing to a full arm — the failed run above looked healthy in every artifact it wrote.

Do **not** tune the curator prompt toward MBPP to make the rules more specific. Fixing the
curator's *inputs* is a defect fix; steering its *instructions* toward the benchmark after seeing
the results is a researcher degree of freedom that would make any positive result unpublishable.
If the prompt is changed, change it once, before looking at test numbers, identically in all arms.

## Gotchas

- **`eval/mbpp_code_exec` and `eval/finance` (`formula`) work; `mind2web*` does not.** `utils.py`'s
  eval path calls `answer_is_correct` with three arguments and `evaluate_accuracy` with three; the
  `mind2web*` processors still define the upstream two-arg signatures. `finer` shares the finance
  processor and has the right signatures but has never been run here.
- **The LLM backend is hardcoded to a personal tunnel.** `utils.py::initialize_clients()` ignores
  `--api_provider` and always targets `https://ollama.caduceus.lol/v1` with an empty key and a
  `bypass: true` header. Expected dev setup, not a bug. `--api_provider` still selects
  `max_completion_tokens` vs `max_tokens` in `llm.py`.
- **Docker must be running for any MBPP run.** `DataProcessor.__init__` constructs a `DockerSandbox`
  and pulls `python:3.10-slim`, before any LLM call.
- **`ace/native_deps.py` must be called before importing sentence-transformers.** On this Windows
  install, importing it first kills the interpreter with an access violation inside pyarrow. Without
  the guard `import ace.core` crashes outright and no run starts.
- **A scoring-side defect can be arm-specific, and that invalidates a comparison.** The same
  double-stringification habit also appears bracketed: `"final_answer": "[\"def f(): ...\"]"`.
  It made **28 of the MBPP text arm's 257 test answers unreadable (10.9%) and none in the other
  two arms**; 12 of the 28 held code that passes, so the arm was scored 12 problems low. Measured
  across all parsed generations: text 68/599, graph 6/552, none 0/230. Cause unknown; the obvious hypothesis
  (the top-cited "use list comprehensions" rule) does not hold, at 86.8% citation in wrapped
  generations against 80.8% in normal ones. `unwrap_quoted_source` now handles `["…"]`/`['…']`,
  and `extract_answer` unwraps a genuine one-element array too. **Rescoring repairs the
  measurement, never the training** — those samples were scored as failures at run time and fed
  to the reflector and curator, so the text arm's playbook was built from a noisier signal than
  the graph arm's. Only re-running fixes that.
- **A numeric task breaks `extract_answer` in ways a code task never did.** Two defects, both
  found by the first `formula` probe, both fixed on the extraction side rather than by touching
  the authors' frozen prompt: (1) the generator writes its working in LaTeX inside `reasoning`
  — `\frac`, `\sigma`, `\approx` — which are not legal JSON escapes, so `json.loads` rejected
  the whole envelope with the answer field intact (22 of 29 unparseable probe responses);
  `repair_json_escapes` retries once with the unrecognised backslashes doubled. (2) Both regex
  fallbacks demanded `final_answer` be a *quoted string*, but a numeric answer is written bare
  (`"final_answer": 229.27`, sometimes with a stray closing quote), losing 15 of 29 with the
  right answer sitting in the text. Both paths run only after a strict parse fails, so they
  can never alter a well-formed response — verified: re-scoring the three MBPP arms with them
  in place returns 86/93/86, unchanged. Covered by `check_finance_wiring.py` [1b].
- **`--num_ctx 32768` is not what the server actually serves.** Ollama clamps `num_ctx` to the
  model's own trained context, silently. Measured on `gemma4:e2b`: every reflector call at or
  below **12,433 tokens** succeeded and one at **14,822** was left 147 tokens of output budget,
  spent them all on hidden reasoning, and returned empty fifteen times in a row. Pinning
  `--num_ctx` is still required — the clamp is deterministic, so both arms get the same window —
  but do not assume you have 32k of headroom. Budget prompts against ~12k for this model.
- **Every role's JSON goes through a different reader, and both had the same blind spot.**
  The generator is parsed by `utils.extract_answer`; the reflector and curator by
  `playbook_utils.extract_json_from_text`. LaTeX in a string field (`\sigma`, `\%`,
  `\approx`) is not valid JSON escaping and rejected **24 of 50 schema-perfect reflector
  responses (48%)** on the formula probe. Both readers now retry once through
  `repair_json_escapes`, after a strict parse has already failed. Note the shape of the
  near-miss: the reflector has a bracket-depth scan fallback that recovered the tags anyway,
  so the damage was invisible in the results and only check [5] showed it — and the **curator
  has no fallback**, so one macro in its response drops a whole operation list silently.
  `repair_json_escapes` must consume valid escapes *whole*: doubling by lookahead alone skips
  a `\\` pair, rescans its second character, and turns `\\approx` into a run of three that is
  still invalid.
- **A runaway generator trace can take the reflector down with it.** The generator's reasoning
  goes into the reflector prompt verbatim, and it is the one field there with no natural bound.
  On the formula probe a 36,268-char trace made up 89.5% of a 40,522-char reflector prompt,
  overflowed the served context, and cost 15 retries plus that step's entire learning signal.
  `Reflector._truncate_trace` now caps it at `MAX_TRACE_TOKENS` (4000), keeping head and tail
  with a marker between. Never extend this to `bullets_used`: that is memory content, it renders
  longer in the graph arm, and truncating it would bias the comparison.
- **Runaway reasoning is a real failure mode on `formula`, not a bug.** `qwen3.5:2b` takes the
  authors' "show your reasoning step-by-step" literally: median response 723 tokens but p90
  8,128 and a tail that reaches `max_tokens`, occasionally emitting no `final_answer` at all
  (~6.5% on the probe). The tail dominates run cost. Do not "fix" it by editing the generator
  prompt — that is the authors' text and steering it toward the benchmark forfeits the
  replication claim.
- **Per-problem predictions must be persisted or no paired test is possible later.**
  `utils.evaluate_test_set` computed per-sample correctness and wrote only aggregates, which is
  why the first three arms needed `rescore_from_logs.py` to be compared at all. It now writes
  `<prefix>_test_predictions.jsonl`. Don't remove it: an unpaired comparison at n=200-257 cannot
  distinguish an inert memory from a directionless one.
- **Some models double-quote the code inside the JSON string.** The generator prompt asks for
  the program as a JSON string field, and `qwen3.5:2b` stringifies it twice, emitting
  `"final_answer": "\"def f():\n    ...\n\""`. After `json.loads` the value still carries literal
  quote characters at both ends, so the sandbox sees a bare string literal instead of a function
  definition and every assertion fails on a name that was never defined. This scores **zero while
  looking exactly like a reasoning problem** — the first probe came back 0.000 train accuracy.
  `utils.unwrap_quoted_source()` strips the wrapper, but only when the answer does not already
  parse and unwrapping makes it parse, so it can turn a syntax error into working code and can
  never alter code that was fine. Measured: 86 of 97 qwen answers were wrapped (10% → 78% parsing,
  test 0.10 → 0.30 rescored), against **0 of 790** for `gemma4:e2b` — which is why the extractor
  had never been exercised on this, and why the fix does not touch the gemma runs at all.
- **Emoji in diagnostics used to kill runs on Windows.** 37 print sites across 8 files carry
  emoji, and on Windows stdout falls back to the ANSI codepage whenever it is not a terminal
  (redirected, piped, captured by a task runner), so printing one raised `UnicodeEncodeError`.
  The worst offenders are inside `llm.py`'s empty-response handler, so it fired exactly when a
  call had already failed — the real error was replaced by an encoding traceback and the process
  died. `llm.py::_make_console_encoding_safe()` now runs at import and relaxes the streams' error
  policy to `replace` (encoding untouched, so output is byte-identical wherever it already
  worked). Every entry point imports `llm`, so this covers the repo. Don't remove it because the
  emoji "look fine in the terminal" — they do, right up until you pipe the run to a log.
- **`llm.py` is Ollama-specific.** It auto-sizes `num_ctx` via `extra_body` and has a bespoke
  empty-response retry path that escalates temperature above 0.0 after `empty_retry_escalate_after`
  failures — `call_info["empty_retries"]` records when a call stopped being greedy.
- **Calls stream by default (`stream=True`).** Non-streamed requests send nothing until generation
  finishes, so the tunnel's reverse proxy cannot tell a slow completion from a hung origin and kills
  it with an HTTP 524 — this killed a text-arm run at step 117/120 on a call emitting ~8k tokens.
  Streaming keeps bytes flowing so wall-clock generation length stops mattering. It is a transport
  change only: same model, seed, temperature and `num_ctx`, so output is byte-identical and arms run
  streamed and non-streamed stay comparable. Token counts are *not* recoverable from a stream, so
  `stream_options={"include_usage": True}` is sent alongside; if a server ignores it, `call_info`
  token counts come back `None`, `aggregate_run_costs.py` has nothing to sum, and `llm.py` prints a
  one-time warning. Run `uv run python check_streaming.py` against the live endpoint before
  trusting it — it verifies content, usage, and streamed/non-streamed output equivalence.
- **`--max_tokens` defaults to `16384` (`eval/mbpp_code_exec/run.py:35`), not the `4096` in the
  `ACE` and `timed_llm_call` signatures.** The backend does honour it — a call capped at exactly
  4096 when asked to — but 16384 is a very loose ceiling: at ~112 tok/s a maximal generation runs
  ~146s, which is why long generator calls could outlast a proxy read timeout. Real runs peak around
  8-9k tokens, so lowering it would be safe, but it changes generation and would invalidate any
  already-completed arm. Do not change it mid-experiment.
- **`ACEBatch` is out of scope** and predates the memory interface; it still owns a playbook string.
  MBPP imports `ACE` directly.
- **`analyze_iterations.py` has a hardcoded `run_save_path` in `main()`** — edit before running.

## Style and lint

Lint with `uv run ruff check .` (`ruff.toml`: line-length 120, rules `E`/`F`/`I`). ~163 pre-existing
findings, mostly from `from utils import *` in `ace/ace.py` and `ace/ace_batch.py`. Don't mass-fix or
run `--fix` across the repo; clean up only lines you are already editing. The codebase is not
`ruff format`-ed — don't reformat whole files.

## Git workflow

Commit directly to `main` — solo research repo, no PR ceremony. `origin` is the fork
(`profsuccodifrutta/agentic_context_engineering_project`); `upstream` is `ace-agent/ace`.
