"""
Swappable memory representations for the ACE loop.

`create_memory` is the single construction point, so adding an arm to the
experiment means adding a branch here rather than touching the training loop.
"""

from typing import Any, Optional

from .base import EMPTY_PLAYBOOK, Memory
from .graph import GraphMemory
from .none import NoMemory
from .text import TextPlaybookMemory

# The three experimental arms, from least to most structure.
MEMORY_KINDS = ("none", "text", "graph")


def create_memory(kind: str = "text", initial: Optional[str] = None, **kwargs: Any) -> Memory:
    """
    Build a memory of the requested kind.

    Args:
        kind: "none" for the pure-LLM control, "text" for the flat playbook
              baseline, "graph" for the relation graph.
        initial: Contents of an --initial_playbook_path file, if one was given.
                 The graph arm accepts a flat playbook here and derives its edges
                 from it, so a graph run can start from a baseline playbook.
        **kwargs: Kind-specific settings (the graph thresholds).
    """
    if kind == "none":
        return NoMemory()

    if kind == "text":
        # Only the settings a flat playbook can act on. The graph thresholds are silently
        # irrelevant here rather than an error, so a caller can pass one settings dict for
        # every arm and let each memory take what applies to it -- which is what keeps the
        # arms configured from a single place.
        return TextPlaybookMemory(initial, **{
            key: value for key, value in kwargs.items()
            if key in ("backend", "retrieval", "retrieval_budget", "retrieval_seeds")
        })

    if kind == "graph":
        if initial:
            return GraphMemory.from_playbook_text(initial, **kwargs)
        return GraphMemory(**kwargs)

    raise ValueError(f"Unknown memory kind: {kind!r}. Expected one of {MEMORY_KINDS}.")


__all__ = [
    "EMPTY_PLAYBOOK",
    "GraphMemory",
    "MEMORY_KINDS",
    "Memory",
    "NoMemory",
    "TextPlaybookMemory",
    "create_memory",
]
