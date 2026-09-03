"""
No memory at all: the pure-LLM control arm.

This is the floor of the experiment. The text and graph arms answer "does memory
structure matter"; this one answers the prior question "does accumulated memory
matter at all", by running the same generator, on the same problems, with nothing
accumulated.

It deliberately keeps the same generator prompt and fills the playbook slot with a
placeholder rather than removing the section. Deleting the section would change the
prompt template, making this arm differ from the others in two ways -- prompt
structure and memory content -- when only the second is the intended comparison.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

from .base import Memory

EMPTY_SLOT = "(empty)"


class NoMemory(Memory):
    """A memory that never accumulates anything."""

    kind = "none"

    def __init__(self, *_args, **_kwargs):
        # Accepts and ignores the other arms' settings so create_memory can pass
        # a uniform kwargs dict.
        pass

    # --- reading -----------------------------------------------------------

    def render_for_generator(self, query: Optional[Union[str, Sequence[str]]] = None) -> str:
        return EMPTY_SLOT

    def render_for_curator(self) -> str:
        return EMPTY_SLOT

    def render_bullets(self, bullet_ids: List[str]) -> str:
        # The same sentinel the flat playbook returns when nothing was cited, which
        # is always the case here: there is nothing to cite.
        return "(No bullets used by generator)"

    def stats(self) -> Dict[str, Any]:
        return {
            "total_bullets": 0,
            "high_performing": 0,
            "problematic": 0,
            "unused": 0,
            "by_section": {},
        }

    def snapshot(self) -> str:
        return EMPTY_SLOT

    # --- writing -----------------------------------------------------------

    def apply_operations(self, operations: List[Dict[str, Any]], step: int = 0) -> List[str]:
        return []

    def update_counts(self, bullet_tags: List[Dict[str, str]]) -> None:
        return None

    # --- lifecycle ---------------------------------------------------------

    def copy(self) -> "NoMemory":
        return NoMemory()

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.snapshot())

    @classmethod
    def load(cls, path: str) -> "NoMemory":
        return cls()
