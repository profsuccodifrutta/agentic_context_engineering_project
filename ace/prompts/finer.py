"""Curator prompts for the `finer` task.

`finer` shares the ACE authors' finance generator and reflector verbatim with `formula`; only
the curator differs, and the difference is derived from the finance text by the substitutions
below rather than written out again, so any upstream edit to `formula.py` either flows through
or raises. The generator and reflector are re-exported unchanged so `FINER_PROMPTS` is a
complete set.

**Why the curator differs.** Under the shared finance prompt the seed-0 arms produced a
playbook of 143 live rules of which 73% restate one directive, "prefer the most specific or
most granular tag over the generic one", at mean pairwise cosine 0.563. De-duplication cannot
reach that: every threshold measured absorbs more than half of all ADDs and still leaves the
directive standing (docs/EXPERIMENT_FINER.md section 10.5). The redundancy is semantic rather
than textual, so it has to be prevented at generation.

`finer` is a classification into a fixed 140-tag vocabulary that is printed in every generator
prompt. A directive that applies to every question therefore carries no information the
generator does not already have, while spending budget that a condition-to-tag mapping could
use. The substitutions ask for the mapping.

**Why the section is pinned (v3, 2026-09-02).** The curator prompt is identical in every arm,
but the *playbook rendering* inside it is not: the graph arm's carries co-citation relation
lines. Measured on two independent seed-0 sets, that is enough to move where the curator files
its rules. The `strategies_and_insights` share of live rules is 106 of 207 and 127 of 206 in the
treatment arm against 0 of 223 and 0 of 227 in the baseline arm. The arms therefore differ
in what the curator writes, not only in how the memory is structured, which is the confound the
shared-prompt rule exists to prevent. Pinning the section at the schema line removes the degree
of freedom in every arm at once. See docs/EXPERIMENT_FINER.md section 9, 2026-09-02.

**Why one prohibition survives (v4, 2026-09-03).** The block is otherwise positive: every other
line states what to emit. The v3 rewrite dropped the explicit ban on preference words as well, on
the argument that naming a construction puts it in the prompt. That was measured and it is false
for this model. Under v3 the strict mapping share at the 50-step probe fell to 3.0% in the
baseline and 22.1% in the treatment, against 50.8% and 74.3% under v2, and the cause is a wrapper
in front of an otherwise correct tag: `THEN prioritize mapping to <Tag>`. What v3 did improve is
content - the share of rules naming one of the 140 printed tags rose to 79.1% and 73.5%, the highest
measured - so the ban is restored and the rest of the rewrite is kept.

This is a task-specific prompt in the authors' own idiom: upstream already ships one set for
code and another for finance. It is registered for `finer` alone, `formula` keeps
`FINANCE_PROMPTS` untouched, and `run_config.json` records `prompt_set` so the two are never
compared to each other by accident.
"""

from . import formula as _formula

GENERATOR_PROMPT = _formula.GENERATOR_PROMPT
REFLECTOR_PROMPT = _formula.REFLECTOR_PROMPT
REFLECTOR_PROMPT_NO_GT = _formula.REFLECTOR_PROMPT_NO_GT

# --- the substitutions, applied to both curator prompts ---------------------------------

_INSTRUCTIONS_ANCHOR = "- Be concise and specific - each addition should be actionable"

_INSTRUCTIONS_ADDED = """- Be concise and specific - each addition should be actionable
- Write every bullet as one mapping: IF <what the text says> THEN <ExactTagName>
- The THEN clause is the tag name alone, copied exactly from the taxonomy
- One mapping per bullet. Two condition-to-tag pairs means two ADD operations
- Every bullet names a tag
- The THEN clause MUST contain ONLY the exact US GAAP tag name. NEVER use words like prioritize, prefer, or over"""

# The ADD schema line, where the model actually emits the field. Pinning it here rather than in
# the instruction block is what makes the section constant across arms.
_SECTION_ANCHOR = "    - section: the section to add the new bullet to"

_SECTION_REPLACEMENT = '    - section: must be exactly "context_clues_and_indicators"'

# The same field in the generic operation schema above the ADD block. ADD is the only operation
# the finance curator defines, so pinning both leaves no line that offers a choice of section.
_FIELD_ANCHOR = "  - section: the section to add the bullet to"

_FIELD_REPLACEMENT = '  - section: must be exactly "context_clues_and_indicators"'

_CONTENT_ANCHOR = (
    "    - content: the new content of the bullet. Note: no need to include the bullet_id in "
    "the content like '[ctx-00263] helpful=1 harmful=0 ::', the bullet_id will be added by the system."
)

_CONTENT_REPLACEMENT = _CONTENT_ANCHOR + " The content is one line: IF <condition> THEN <ExactTagName>."

_EXAMPLE_ANCHOR = """      "section": "formulas_and_calculations",
      "content": "[New calculation method...]","""

_EXAMPLE_REPLACEMENT = """      "section": "context_clues_and_indicators",
      "content": "IF the text reports cash paid to acquire a business THEN PaymentsToAcquireBusinessesGross","""

_SUBSTITUTIONS = (
    (_INSTRUCTIONS_ANCHOR, _INSTRUCTIONS_ADDED),
    (_SECTION_ANCHOR, _SECTION_REPLACEMENT),
    (_FIELD_ANCHOR, _FIELD_REPLACEMENT),
    (_CONTENT_ANCHOR, _CONTENT_REPLACEMENT),
    (_EXAMPLE_ANCHOR, _EXAMPLE_REPLACEMENT),
)


def _derive(source: str) -> str:
    """Apply the substitutions, raising if any anchor has drifted out of `formula.py`.

    A silent no-op here would ship the finance prompt under the `finer` label and make the
    two prompt sets identical while `run_config.json` claimed otherwise.
    """
    for anchor, replacement in _SUBSTITUTIONS:
        if anchor not in source:
            raise AssertionError(f"finer curator prompt anchor not found in formula.py: {anchor[:60]!r}")
        source = source.replace(anchor, replacement, 1)
    return source


CURATOR_PROMPT = _derive(_formula.CURATOR_PROMPT)
CURATOR_PROMPT_NO_GT = _derive(_formula.CURATOR_PROMPT_NO_GT)
