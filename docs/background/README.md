# Background

Nothing in this folder is a result of the current system. It is kept because it is the record of
how the design got to where it is, and deleting a pre-registration after the fact is worse than
keeping it.

The publication-facing documents are one level up in `docs/`.

| document | why it is here |
|---|---|
| [`EXPERIMENT_FORMULA.md`](EXPERIMENT_FORMULA.md) | Pre-registration for the three-arm `related_to` experiment on `formula`. |
| [`FORMULA_RESULTS_PREREGISTERED.md`](FORMULA_RESULTS_PREREGISTERED.md) | Its nine runs and their analysis. |
| [`EXPERIMENT_MBPP.md`](EXPERIMENT_MBPP.md) | Pre-registration for an MBPP experiment. **No arm ever ran.** |
| [`MBPP_RESULTS.md`](MBPP_RESULTS.md) | An earlier MBPP run set, on a configuration this repository no longer uses. |

## Why each one is background

**The `related_to` experiment on `formula`** ran on the current harness and is a real null. It is
here because its treatment was superseded rather than because its measurement was wrong.
`related_to` is cosine between two rules' texts, while retrieval scores rules by cosine between the
question and each rule: the same encoder in the same space, so the walk landed where relevance was
already looking. A random rewiring at matched edge count surfaced 2.3x to 3.6x *more* novel rules
than the real graph. That finding is what the co-citation experiment was built on, and it is the
reason to read these two files. Its accuracy numbers are superseded by the co-citation runs and
should not be quoted as this repository's result.

**The MBPP pre-registration** was written and never executed. Probing showed the treatment could
not be delivered on that task: the co-citation gate admitted zero edges, the generator named 1.47
rules per answer with a hard maximum of 2, and the whole playbook fit inside the retrieval budget,
so the walk had nothing to add. It is kept open and unrun rather than deleted, because a
pre-registration that vanishes from the history is a worse record than one that says plainly it was
abandoned and why.

**The earlier MBPP runs** are not a measurement of this system at all. They predate `--retrieval`,
so the graph arm's treatment never fired; they predate `--dedup`, so the baseline was missing one
of ACE's three components; they ran the curator once every five steps with one reflector round
against the current one and three; and they used a different generator model. Their §2, §5 and §6
mechanism write-up is still worth reading. Their numbers are not.
