"""
Prompts module for ACE system.
Contains all prompts for Generator, Reflector, and Curator agents.

Prompts are selected per task through `get_prompts(task_name)`. The code set
(`generator.py`, `reflector.py`, `curator.py`) is the default and is what every existing
caller gets when no task name is supplied, so behaviour is unchanged by default.

The `PromptSet` wrapper exists because the two sets bind their generator placeholders
differently: the code prompt is positional, the finance prompt is named. That difference is
invisible at the call site and would scramble every field if it were not handled here, so
formatting goes through `format_generator` rather than `.format` directly.
"""

from dataclasses import dataclass
from typing import Optional

from . import formula as _formula
from .curator import *  # noqa: F401,F403
from .curator import CURATOR_PROMPT, CURATOR_PROMPT_NO_GT
from .generator import *  # noqa: F401,F403
from .generator import GENERATOR_PROMPT
from .reflector import *  # noqa: F401,F403
from .reflector import REFLECTOR_PROMPT, REFLECTOR_PROMPT_NO_GT


@dataclass(frozen=True)
class PromptSet:
    """One task's prompts, plus how its generator template binds its fields."""

    name: str
    generator: str
    reflector: str
    reflector_no_gt: str
    curator: str
    curator_no_gt: str
    #: "positional" formats with four ordered arguments, "named" with four keywords.
    generator_binding: str = "positional"

    def format_generator(self, question: str, context: str, playbook: str,
                         reflection: str) -> str:
        if self.generator_binding == "named":
            return self.generator.format(question=question, context=context,
                                         playbook=playbook, reflection=reflection)
        # Argument order is load-bearing and matches the historical call site exactly.
        return self.generator.format(question, context, playbook, reflection)


CODE_PROMPTS = PromptSet(
    name="code",
    generator=GENERATOR_PROMPT,
    reflector=REFLECTOR_PROMPT,
    reflector_no_gt=REFLECTOR_PROMPT_NO_GT,
    curator=CURATOR_PROMPT,
    curator_no_gt=CURATOR_PROMPT_NO_GT,
    generator_binding="positional",
)

FINANCE_PROMPTS = PromptSet(
    name="finance",
    generator=_formula.GENERATOR_PROMPT,
    reflector=_formula.REFLECTOR_PROMPT,
    reflector_no_gt=_formula.REFLECTOR_PROMPT_NO_GT,
    curator=_formula.CURATOR_PROMPT,
    curator_no_gt=_formula.CURATOR_PROMPT_NO_GT,
    generator_binding="named",
)

#: Tasks whose prompts are not the code set. Anything absent falls back to CODE_PROMPTS,
#: so adding a task never silently changes an existing one.
PROMPT_SETS = {
    "formula": FINANCE_PROMPTS,
    "finer": FINANCE_PROMPTS,
    "mbpp_code_exec": CODE_PROMPTS,
}


def get_prompts(task_name: Optional[str] = None) -> PromptSet:
    """Return the prompt set for a task, defaulting to the code prompts."""
    return PROMPT_SETS.get(task_name, CODE_PROMPTS)


__all__ = [
    # Generator prompts
    'GENERATOR_PROMPT',

    # Reflector prompts
    'REFLECTOR_PROMPT',
    'REFLECTOR_PROMPT_NO_GT',

    # Curator prompts
    'CURATOR_PROMPT',
    'CURATOR_PROMPT_NO_GT',

    # Per-task selection
    'PromptSet',
    'CODE_PROMPTS',
    'FINANCE_PROMPTS',
    'PROMPT_SETS',
    'get_prompts',
]
