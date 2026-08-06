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
    format_playbook_line,
    get_next_global_id,
    get_playbook_stats,
    parse_playbook_line,
    update_bullet_counts,
)
from utils import count_tokens

from .base import EMPTY_PLAYBOOK, Memory, sort_by_mint_order
from .embeddings import EmbeddingBackend
from .refine import duplicate_groups
from .retrieval import (
    DEFAULT_RETRIEVAL_BUDGET,
    DEFAULT_RETRIEVAL_SEEDS,
    fit_to_budget,
    flat_order,
    score_against_query,
)


class TextPlaybookMemory(Memory):
    """The playbook as a single string of '[id] helpful=X harmful=Y :: content' lines."""

    kind = "text"

    def __init__(self, playbook: Optional[str] = None, next_global_id: Optional[int] = None,
                 backend: Optional[Any] = None, retrieval: str = "none",
                 retrieval_budget: int = DEFAULT_RETRIEVAL_BUDGET,
                 retrieval_seeds: int = DEFAULT_RETRIEVAL_SEEDS):
        self.playbook = playbook if playbook else EMPTY_PLAYBOOK
        # Derive rather than default to 1, so seeding from an existing playbook does
        # not re-mint ids that already exist in it.
        self.next_global_id = (
            next_global_id if next_global_id is not None else get_next_global_id(self.playbook)
        )
        if retrieval not in ("none", "flat"):
            raise ValueError(
                f"TextPlaybookMemory cannot do {retrieval!r} retrieval -- a flat playbook has "
                f"no edges to walk. Use 'flat' for the baseline arm."
            )
        self.retrieval = retrieval
        self.retrieval_budget = retrieval_budget
        self.retrieval_seeds = retrieval_seeds
        self.backend = backend if backend is not None else (
            EmbeddingBackend() if retrieval != "none" else None
        )
        self._vectors: Dict[str, Any] = {}

    # --- reading -----------------------------------------------------------

    def render_for_generator(self, query: Optional[str] = None) -> str:
        """
        The whole playbook, or the part of it this question is about.

        A query with retrieval off is not an error, it is a caller that does not know which
        arm it is in -- the loop passes the question unconditionally and the memory decides.
        """
        if query is None or self.retrieval == "none":
            return self.playbook
        return self._render_retrieved(query)

    def _render_retrieved(self, query: str) -> str:
        entries = [e for e in (parse_playbook_line(line)
                               for line in self.playbook.split("\n")) if e]
        if not entries:
            return self.playbook

        by_id = {e["id"]: e for e in entries}
        # _vectors is filled in place, so the playbook is embedded once per new rule rather
        # than once per generation.
        scores = score_against_query(
            query, [(e["id"], e["content"]) for e in entries], self.backend, self._vectors)

        chosen = fit_to_budget(
            flat_order(scores),
            lambda rule_id, _picked: count_tokens(by_id[rule_id]["raw_line"]),
            self._render_subset,
            self.retrieval_budget,
        )
        return self._render_subset(chosen)

    def _render_subset(self, keep: List[str]) -> str:
        """Selected rules only, in playbook order, with every section header preserved.

        Order and headers are kept because the graph arm keeps them too: a rule reordered by
        relevance would change what the model reads independently of which rules it reads,
        and that is a second treatment.
        """
        wanted = set(keep)
        lines = []
        for line in self.playbook.split("\n"):
            entry = parse_playbook_line(line)
            if entry is None:
                lines.append(line)
            elif entry["id"] in wanted:
                lines.append(line)
        return "\n".join(lines)

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

    def refine(self, backend: Any, threshold: float) -> Dict[str, Any]:
        """
        Drop absorbed lines outright, which is what the paper's de-duplication does.

        The counters of an absorbed rule are added to the survivor rather than discarded: the
        two rules said the same thing, so the evidence about that advice was split across
        them, and throwing half of it away would make a rule the reflector had graded twice
        look untried.
        """
        parsed = [parse_playbook_line(line) for line in self.playbook.split("\n")]
        items = [(p["id"], p["content"]) for p in parsed if p]
        groups = duplicate_groups(items, backend, threshold)
        if not groups:
            return {"groups": 0, "absorbed": 0, "remaining": len(items)}

        gained = {}
        absorbed = set()
        by_id = {p["id"]: p for p in parsed if p}
        for survivor, victims in groups.items():
            helpful = sum(by_id[v]["helpful"] for v in victims)
            harmful = sum(by_id[v]["harmful"] for v in victims)
            gained[survivor] = (helpful, harmful)
            absorbed.update(victims)

        kept = []
        for line, entry in zip(self.playbook.split("\n"), parsed):
            if entry is None:
                kept.append(line)
                continue
            if entry["id"] in absorbed:
                continue
            if entry["id"] in gained:
                helpful, harmful = gained[entry["id"]]
                kept.append(format_playbook_line(
                    entry["id"], entry["helpful"] + helpful, entry["harmful"] + harmful,
                    entry["content"]))
            else:
                kept.append(line)

        self.playbook = "\n".join(kept)
        return {"groups": len(groups), "absorbed": len(absorbed),
                "remaining": len(items) - len(absorbed)}

    # --- lifecycle ---------------------------------------------------------

    def copy(self) -> "TextPlaybookMemory":
        clone = TextPlaybookMemory(
            self.playbook, next_global_id=self.next_global_id, backend=self.backend,
            retrieval=self.retrieval, retrieval_budget=self.retrieval_budget,
            retrieval_seeds=self.retrieval_seeds,
        )
        # Rule text never changes for a given id -- the curator only ADDs and refine() only
        # deletes -- so the cache stays valid in the copy. Sharing the backend is safe for the
        # same reason the graph shares it: the models are read-only.
        clone._vectors = dict(self._vectors)
        return clone

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
