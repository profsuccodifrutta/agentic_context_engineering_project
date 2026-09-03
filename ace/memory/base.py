"""
Memory interface for the ACE loop.

The playbook has always been a plain string with exactly one generation-time
injection point (ace/core/generator.py formats it into GENERATOR_PROMPT). The
generator, reflector and curator never see structure -- they see rendered text.
That makes the memory representation swappable behind a single seam, which is
what this interface is: the loop holds a Memory, and every memory implementation
answers the same questions in the same order.

Two implementations exist:

- TextPlaybookMemory: the flat playbook, delegating to the original
  playbook_utils functions so the baseline arm is provably unchanged.
- GraphMemory: rules as nodes in a typed relation graph.

Both are ADD-only, matching the only curator operation the system implements.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union

from utils import count_tokens

# The seven sections a fresh playbook is seeded with. Kept byte-identical to the
# original ACE._initialize_empty_playbook() so a text-arm run is unaffected by the
# refactor.
EMPTY_PLAYBOOK = """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""


def sort_by_mint_order(bullet_ids: List[str]) -> List[str]:
    """
    Order ids by their numeric suffix, i.e. the order they were minted in.

    Implementations recover the ids they just added by different routes -- the
    text arm diffs the rendered playbook, the graph arm collects them as it goes --
    and those routes disagree on ordering. Sorting by the zero-padded counter gives
    both the same answer, so downstream logs are comparable across arms.
    """
    def key(bullet_id: str):
        _, _, tail = bullet_id.rpartition("-")
        return (0, int(tail)) if tail.isdigit() else (1, bullet_id)

    return sorted(bullet_ids, key=key)


class Memory(ABC):
    """Accumulated rules, in whatever representation, rendered on demand as text."""

    # --- reading -----------------------------------------------------------

    @abstractmethod
    def render_for_generator(self, query: Optional[Union[str, Sequence[str]]] = None) -> str:
        """
        Render the memory for injection into the generator prompt.

        `query` may be one string or several. A task whose row is several independent
        sub-questions supplies one entry each, and `score_against_query` takes each rule's best
        match against any of them; `finer` does since 2026-09-03. A single string behaves as it
        always has.

        `query` is accepted so retrieval can later be switched on for both
        implementations without touching the training loop. It is None for the
        whole-memory injection this experiment compares, which mirrors what the
        baseline does today: the entire playbook, every call.
        """

    @abstractmethod
    def render_for_curator(self) -> str:
        """Render the memory for the curator prompt."""

    @abstractmethod
    def render_bullets(self, bullet_ids: List[str]) -> str:
        """Render just the cited rules, for the reflector to grade."""

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """
        Summary counts, surfaced in the curator prompt and in train_results.json.

        Every implementation must return the same key set. The curator prompt
        interpolates this dict, so an arm that returned extra keys here would be
        prompting its curator differently -- a confound hidden inside a metric.
        Structure-specific numbers belong in structural_stats() instead, which is
        recorded but never shown to a model.
        """

    @abstractmethod
    def snapshot(self) -> str:
        """Human-readable form written to *_playbook.txt and diffed between runs."""

    # --- writing (ADD-only) ------------------------------------------------

    @abstractmethod
    def apply_operations(self, operations: List[Dict[str, Any]], step: int = 0) -> List[str]:
        """Apply curator operations. Returns the ids of rules that were added."""

    def record_generation(self, rendered_ids, cited_ids, step: int = 0) -> None:
        """Observe one generation: what retrieval showed, and what the model said it used.

        A no-op by default, and deliberately so. Every memory kind receives the same call from
        the same place in the loop, so the arms cannot differ in *when* evidence is offered --
        only in whether the representation can hold it. A flat playbook can store the unary
        part of this signal (that is what helpful/harmful already are) but has nowhere to put
        a fact about a *pair*, which is the whole point of the comparison.
        """

    @abstractmethod
    def update_counts(self, bullet_tags: List[Dict[str, str]]) -> None:
        """Apply the reflector's helpful/harmful tags in place."""

    def refine(self, backend: Any, threshold: float) -> Dict[str, Any]:
        """
        Prune rules that repeat an earlier rule -- the paper's grow-and-refine (§3.2).

        The *decision* is shared (memory/refine.py::duplicate_groups) so that every arm
        withdraws the same rules from the same accumulated set; only the way the outcome is
        stored is the implementation's business. An arm whose de-duplication rule differed
        from its sibling's would be running a second treatment alongside the memory
        structure, which is exactly what the memory interface exists to prevent.

        Returns a report dict; the default is a no-op for memories that hold nothing.
        """
        return {"groups": 0, "absorbed": 0, "remaining": 0}

    # --- lifecycle ---------------------------------------------------------

    @abstractmethod
    def copy(self) -> "Memory":
        """
        Return an independent copy.

        Required, not convenience: the loop tracks a best-so-far memory with
        `self.best_memory = self.memory`. That was only safe while the playbook
        was an immutable string that every mutation rebound. A mutable memory
        assigned this way would alias, and "best" would silently follow
        "current" for the rest of the run.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Write to `path` (a *.txt destination), plus any sidecar the impl needs."""

    # --- shared ------------------------------------------------------------

    def structural_stats(self) -> Dict[str, Any]:
        """
        Representation-specific measurements for train_results.json.

        Never rendered into a prompt, so implementations may return whatever
        their structure supports without affecting what any model sees.
        """
        return {}

    def num_tokens(self) -> int:
        """Token count of the generator-facing rendering (tiktoken cl100k_base)."""
        return count_tokens(self.render_for_generator())

    def __len__(self) -> int:
        """Character length of the generator-facing rendering."""
        return len(self.render_for_generator())
