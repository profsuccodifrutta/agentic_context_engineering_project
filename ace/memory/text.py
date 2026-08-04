"""
Flat-playbook memory: the original ACE representation, behind the Memory interface.

Every method delegates to the existing playbook_utils functions rather than
reimplementing them. That is deliberate and load-bearing for the experiment: the
text arm must behave exactly as it did before the Memory refactor, so that any
difference measured against the graph arm is attributable to the memory
structure and not to a rewrite of the baseline.
"""

from typing import Any, Dict, List, Optional

from playbook_utils import (
    apply_curator_operations,
    extract_playbook_bullets,
    get_next_global_id,
    get_playbook_stats,
    parse_playbook_line,
    update_bullet_counts,
)

from .base import EMPTY_PLAYBOOK, Memory, sort_by_mint_order


class TextPlaybookMemory(Memory):
    """The playbook as a single string of '[id] helpful=X harmful=Y :: content' lines."""

    kind = "text"

    def __init__(self, playbook: Optional[str] = None, next_global_id: Optional[int] = None):
        self.playbook = playbook if playbook else EMPTY_PLAYBOOK
        # Derive rather than default to 1, so seeding from an existing playbook does
        # not re-mint ids that already exist in it.
        self.next_global_id = (
            next_global_id if next_global_id is not None else get_next_global_id(self.playbook)
        )

    # --- reading -----------------------------------------------------------

    def render_for_generator(self, query: Optional[str] = None) -> str:
        if query is not None:
            raise NotImplementedError(
                "Retrieval is not implemented for TextPlaybookMemory. Enabling it for "
                "one arm only would confound the comparison; add it to both arms together."
            )
        return self.playbook

    def render_for_curator(self) -> str:
        return self.playbook

    def render_bullets(self, bullet_ids: List[str]) -> str:
        return extract_playbook_bullets(self.playbook, bullet_ids)

    def stats(self) -> Dict[str, Any]:
        return get_playbook_stats(self.playbook)

    def snapshot(self) -> str:
        return self.playbook

    # --- writing -----------------------------------------------------------

    def apply_operations(self, operations: List[Dict[str, Any]], step: int = 0) -> List[str]:
        if not operations:
            return []

        before = set(self._bullet_ids())
        self.playbook, self.next_global_id = apply_curator_operations(
            self.playbook, operations, self.next_global_id
        )
        # Diff rather than re-deriving the minted ids, so id minting stays owned by
        # apply_curator_operations alone and cannot drift out of sync here.
        return sort_by_mint_order([bid for bid in self._bullet_ids() if bid not in before])

    def update_counts(self, bullet_tags: List[Dict[str, str]]) -> None:
        if not bullet_tags:
            return
        self.playbook = update_bullet_counts(self.playbook, bullet_tags)

    # --- lifecycle ---------------------------------------------------------

    def copy(self) -> "TextPlaybookMemory":
        return TextPlaybookMemory(self.playbook, next_global_id=self.next_global_id)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.playbook)

    @classmethod
    def load(cls, path: str) -> "TextPlaybookMemory":
        with open(path, "r", encoding="utf-8") as f:
            return cls(f.read())

    # --- internals ---------------------------------------------------------

    def _bullet_ids(self) -> List[str]:
        ids = []
        for line in self.playbook.split("\n"):
            parsed = parse_playbook_line(line)
            if parsed:
                ids.append(parsed["id"])
        return ids
