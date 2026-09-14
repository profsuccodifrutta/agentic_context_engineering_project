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
            if key in ("backend", "retrieval", "retrieval_budget", "retrieval_seeds",
                       "subquery_seeding")
        })

    if kind == "graph":
        if initial:
            return GraphMemory.from_playbook_text(initial, **kwargs)
        return GraphMemory(**kwargs)

    raise ValueError(f"Unknown memory kind: {kind!r}. Expected one of {MEMORY_KINDS}.")



def build_memory_kwargs(args):
    """The settings each memory is constructed with, for the arm this run is.

    Graph-only settings reach the graph arm alone. `subquery_seeding` is the exception and
    reaches every arm that retrieves: flat retrieval ranks globally too, so a text arm left on a
    global ranking while the graph arm balances across sub-questions makes slot allocation a
    second difference between them. That is not hypothetical -- this function exists because
    the setting was gated behind `args.memory == "graph"` and silently never reached the text
    arm. Both runners import this one function, and `check_finance_wiring.py` [1o] calls it for
    every arm.
    """
    if args.memory == "graph":
        return {
            "related_threshold": args.related_threshold,
            "nli_gate": args.nli_gate,
            "contradicts_threshold": args.contradicts_threshold,
            "enable_contradicts": args.enable_contradicts,
            "keep_superseded": args.keep_superseded,
            "cocite": args.cocite,
            "cocite_min_opportunity": args.cocite_min_opportunity,
            "cocite_min_rate": args.cocite_min_rate,
            "cocite_half_life": args.cocite_half_life,
            "cocite_predicate": args.cocite_predicate,
            "cocite_beta": args.cocite_beta,
            "cocite_k": args.cocite_k,
            "cocite_outcome_weighted": args.cocite_outcome_weighted,
            "cocite_walk": args.cocite_walk,
            "render_relations": not args.no_render_relations,
            "subquery_seeding": args.subquery_seeding,
            "curator_refines": not args.no_curator_refines,
        }
    if args.memory == "text":
        return {"subquery_seeding": args.subquery_seeding}
    return {}


__all__ = [
    "EMPTY_PLAYBOOK",
    "GraphMemory",
    "MEMORY_KINDS",
    "Memory",
    "NoMemory",
    "TextPlaybookMemory",
    "build_memory_kwargs",
    "create_memory",
]
