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
use. The substitutions ask for the mapping and forbid the directive.

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
- Write every bullet as a conditional mapping in the form: IF <what the text says> THEN <exact US GAAP tag>
- Copy the tag string exactly as it appears in the taxonomy. Never paraphrase or abbreviate a tag name
- Write no justification. Do not append clauses such as "This ensures accurate reporting", "Contextual analysis is crucial", or "to ensure alignment with granular reporting requirements". The mapping alone is the bullet
- Do not write directional preferences. "Prefer the most specific tag", "prioritize granular over general" and "contextual relevance dictates the choice" apply to every question in this task, so they tell the reader nothing and displace mappings that would
- One mapping per bullet. If the reflection yields two distinct condition-to-tag pairs, emit two ADD operations
- A bullet that names no tag is not worth adding
- The THEN clause MUST contain ONLY the exact US GAAP tag name. NEVER use words like prioritize, prefer, or over"""

_CONTENT_ANCHOR = (
    "    - content: the new content of the bullet. Note: no need to include the bullet_id in "
    "the content like '[ctx-00263] helpful=1 harmful=0 ::', the bullet_id will be added by the system."
)

_CONTENT_REPLACEMENT = _CONTENT_ANCHOR + (
    " The content must be a single IF <condition> THEN <exact tag> mapping, with no explanation "
    "after it."
)

_EXAMPLE_ANCHOR = """      "section": "formulas_and_calculations",
      "content": "[New calculation method...]","""

_EXAMPLE_REPLACEMENT = """      "section": "context_clues_and_indicators",
      "content": "IF the text reports cash paid to acquire a business THEN PaymentsToAcquireBusinessesGross","""

_SUBSTITUTIONS = (
    (_INSTRUCTIONS_ANCHOR, _INSTRUCTIONS_ADDED),
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
