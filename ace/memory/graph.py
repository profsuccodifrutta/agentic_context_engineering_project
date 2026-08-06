"""
Graph memory: accumulated rules as nodes in a typed relation graph.

This is the treatment arm of the linear-vs-relational comparison. Rules are
identical in content and identifier to what the flat playbook would hold -- the
same curator, the same ADD-only operation vocabulary, the same helpful/harmful
counters. What differs is that rules also carry typed relations to each other,
and that those relations are rendered into the generator prompt.

Three relation types exist:

  refines      declared by the curator when a new rule supersedes an existing one.
               This is the arm's actual capability -- see below.
  related_to   cosine similarity between rule embeddings, above related_threshold.
               Always on.
  contradicts  NLI contradiction probability above contradicts_threshold, scored
               only for pairs already above nli_gate. OFF by default -- see below.

refines is what makes this more than an annotated list. A superseded rule stops
being rendered into the generator prompt while remaining in the graph, so the graph
arm gains eviction: stale guidance stops competing for the model's attention, and
the rendered memory can shrink as well as grow. The flat playbook cannot do this.
It would need a DELETE operation, which upstream ACE never implemented, so its
first draft of a rule keeps being injected forever alongside every later correction.

Crucially this stays ADD-only in the sense that matters: nothing is ever removed
from the graph. The superseded rule keeps its id, its content, its counters and its
edges, and is still written to disk. Supersession is a rendering decision, which is
why --keep_superseded can switch it off without changing what was stored.

contradicts is disabled by default because an off-the-shelf NLI cross-encoder does
not work on this material. Calibrated against a real 24-rule playbook it produced
19 contradiction edges of which roughly 3 were genuine, scoring pairs like
"Prioritize writing pure functions" against "Prioritize function composition" at
1.000 confidence. NLI models are trained on descriptive premise/hypothesis pairs,
while these rules are prescriptive advice that mostly opens with "Prioritize", and
the model reads any two directives as competing. The scores are saturated rather
than graded (19 edges at threshold 0.30, still 14 at 0.95), so no threshold
separates signal from noise. Run calibrate_graph_thresholds.py before enabling it.

The similarity gate keeps contradiction detection cheap when it is enabled: rules
that genuinely conflict are almost always topically close. Combined with embedding
only the newly added rule, edge construction is O(n) per ADD rather than O(n^2).

Relations are stored as single directed edges but are semantically symmetric, and
every reader here treats them that way. depends_on and refines are intentionally
absent: neither is recoverable from embeddings or NLI, so both would require
extending the curator's output schema, which changes the baseline's write path.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from playbook_utils import (
    format_playbook_line,
    normalize_section_name,
    parse_playbook_line,
)
from utils import count_tokens, get_section_slug

from .base import EMPTY_PLAYBOOK, Memory, sort_by_mint_order
from .embeddings import EmbeddingBackend
from .refine import duplicate_groups
from .retrieval import (
    DEFAULT_RETRIEVAL_BUDGET,
    DEFAULT_RETRIEVAL_SEEDS,
    RETRIEVAL_MODES,
    fit_to_budget,
    flat_order,
    graph_order,
    score_against_query,
)

RELATED_TO = "related_to"
CONTRADICTS = "contradicts"
REFINES = "refines"

CONFLICTS_HEADER = "## RULE CONFLICTS"
CONFLICTS_NOTE = ("These rules disagree with each other. Prefer the one with the better "
                  "track record, and do not apply both.")


def build_legend(include_contradicts: bool, include_refines: bool) -> str:
    """
    Explain the notation to the model.

    Built rather than fixed so the legend only ever describes relations the model
    will actually encounter. Documenting a relation type while emitting none of it
    would spend prompt tokens teaching a notation that never appears.
    """
    lines = [
        "The knowledge below is a graph, not a list. Each rule carries typed",
        "relations to other rules, shown as indented lines beneath it:",
        "  related_to  -- covers closely related ground; consider them together",
    ]
    if include_refines:
        lines.append("  refines     -- replaces an older rule, which has been withdrawn")
    if include_contradicts:
        lines.append("  contradicts -- gives conflicting advice; they cannot both be followed")
    lines += [
        "Counters record each rule's track record: helpful=X is how often it contributed to a",
        "correct solution, harmful=Y how often it contributed to a wrong one.",
    ]
    return "\n".join(lines)


class GraphMemory(Memory):
    """Rules as graph nodes with related_to / contradicts edges between them."""

    kind = "graph"

    def __init__(
        self,
        related_threshold: float = 0.60,
        nli_gate: float = 0.45,
        contradicts_threshold: float = 0.50,
        enable_contradicts: bool = False,
        keep_superseded: bool = False,
        backend: Optional[EmbeddingBackend] = None,
        sections: Optional[List[str]] = None,
        next_global_id: int = 1,
        retrieval: str = "none",
        retrieval_budget: int = DEFAULT_RETRIEVAL_BUDGET,
        retrieval_seeds: int = DEFAULT_RETRIEVAL_SEEDS,
    ):
        if retrieval not in RETRIEVAL_MODES:
            raise ValueError(f"unknown retrieval mode {retrieval!r}; "
                             f"expected one of {list(RETRIEVAL_MODES)}")
        self.retrieval = retrieval
        self.retrieval_budget = retrieval_budget
        self.retrieval_seeds = retrieval_seeds
        self.related_threshold = related_threshold
        self.nli_gate = nli_gate
        self.contradicts_threshold = contradicts_threshold
        self.enable_contradicts = enable_contradicts
        # Ablation switch: keep superseded rules in the prompt, so a run can separate
        # "the graph annotated the rules" from "the graph withdrew stale rules".
        self.keep_superseded = keep_superseded
        self.backend = backend if backend is not None else EmbeddingBackend()
        self.next_global_id = next_global_id

        # Ordered display headers, mirroring the flat playbook's section list so both
        # arms offer the curator the same section vocabulary.
        self.sections = sections if sections is not None else _default_sections()

        self.graph = nx.MultiDiGraph()
        self._embeddings: Dict[str, Any] = {}
        self._insertion_counter = 0

    # --- reading -----------------------------------------------------------

    def render_for_generator(self, query: Optional[str] = None) -> str:
        """
        The live rules, or the neighbourhood of the ones this question is about.

        `graph` retrieval is the treatment: the same top-k seeds the flat arm would take,
        then outward along `related_to`. A neighbour that similarity alone would never have
        surfaced is what the flat arm structurally cannot render, and it is the only place in
        this system where connectivity changes what a model reads.
        """
        if query is None or self.retrieval == "none":
            return self._render(with_legend=True)

        active = self.active_nodes()
        if not active:
            return self._render(with_legend=True)

        scores = score_against_query(
            query, [(n, self.graph.nodes[n]["content"]) for n in active],
            self.backend, self._embeddings)

        if self.retrieval == "flat":
            # The ablation: graph storage, graph rendering, but relevance-only selection.
            # Running it against the graph walk isolates "edges choose what is read" from
            # "edges are shown alongside what is read".
            order = flat_order(scores)
        else:
            live = set(active)
            order = graph_order(
                scores,
                lambda node_id: [n for n in self._neighbours(node_id, RELATED_TO) if n in live],
                self.retrieval_seeds,
            )

        chosen = fit_to_budget(
            order,
            self._render_cost,
            lambda keep: self._render(with_legend=True, only=set(keep)),
            self.retrieval_budget,
        )
        return self._render(with_legend=True, only=set(chosen))

    def _render_cost(self, node_id: str, picked: List[str]) -> int:
        """Tokens this rule adds: its own line, plus relations to rules already selected.

        An estimate, because a later selection can add a relation line to an earlier rule.
        fit_to_budget measures the real rendering afterwards and trims.
        """
        visible = set(picked) | {node_id}
        hidden = {n for n in self.graph.nodes() if n not in visible}
        lines = [self._format_node(node_id)] + self._format_relations(node_id, hidden)
        return count_tokens("\n".join(lines))

    def render_for_curator(self) -> str:
        return self._render(with_legend=True)

    def render_bullets(self, bullet_ids: List[str]) -> str:
        """
        Render cited rules for the reflector.

        Deliberately identical to the flat playbook's rendering, with no relation
        annotations: the reflector's job is to grade rules, not to reason over
        structure, and keeping this step identical across arms confines the
        experimental difference to the generator prompt.
        """
        if not bullet_ids:
            return "(No bullets used by generator)"

        wanted = set(bullet_ids)
        lines = [
            self._format_node(node_id)
            for node_id in self._nodes_in_reading_order()
            if node_id in wanted
        ]

        if not lines:
            return "(Generator referenced bullet IDs but none were found in playbook)"
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        """
        Same key set and same bucket rules as get_playbook_stats().

        Counts only rules the curator can actually see. Reporting withdrawn rules
        here would describe a playbook that does not match the one rendered directly
        above it in the same prompt.
        """
        stats = {
            "total_bullets": 0,
            "high_performing": 0,
            "problematic": 0,
            "unused": 0,
            "by_section": {},
        }

        for node_id in self.active_nodes():
            data = self.graph.nodes[node_id]
            helpful, harmful = data["helpful"], data["harmful"]
            stats["total_bullets"] += 1

            if helpful > 5 and harmful < 2:
                stats["high_performing"] += 1
            elif harmful >= helpful and harmful > 0:
                stats["problematic"] += 1
            elif helpful + harmful == 0:
                stats["unused"] += 1

            section = data["section_header"]
            bucket = stats["by_section"].setdefault(
                section, {"count": 0, "helpful": 0, "harmful": 0}
            )
            bucket["count"] += 1
            bucket["helpful"] += helpful
            bucket["harmful"] += harmful

        return stats

    def structural_stats(self) -> Dict[str, Any]:
        """Graph shape over training. Recorded only; never shown to a model."""
        node_count = self.graph.number_of_nodes()
        related = self._edges_of_type(RELATED_TO)
        contradicts = self._edges_of_type(CONTRADICTS)
        refines = self._edges_of_type(REFINES)
        superseded = self.superseded_nodes()
        possible_pairs = node_count * (node_count - 1) / 2

        undirected = nx.Graph()
        undirected.add_nodes_from(self.graph.nodes())
        undirected.add_edges_from((u, v) for u, v, _ in self.graph.edges(keys=True))

        # How often a contradicts edge coincides with an actually-harmful rule. The
        # secondary result: does relational structure predict the counter signal?
        contradicting_nodes = {n for edge in contradicts for n in edge[:2]}
        harmful_nodes = {
            n for n in self.graph.nodes() if self.graph.nodes[n]["harmful"] > 0
        }

        return {
            "node_count": node_count,
            # The gap between these two is the eviction the flat playbook cannot do:
            # rules accumulated versus rules still competing for the model's attention.
            "active_node_count": node_count - len(superseded),
            "superseded_node_count": len(superseded),
            "edge_count": self.graph.number_of_edges(),
            "related_to_edges": len(related),
            "contradicts_edges": len(contradicts),
            "refines_edges": len(refines),
            # Split, because they answer different questions. The curator-declared count is
            # the graph arm's headline capability -- 2 in 50 calls on the seed-0 run -- and a
            # total that silently absorbed de-duplication edges would hide exactly that.
            "refines_edges_curator": sum(
                1 for _, _, d in refines if d.get("origin", "curator") == "curator"),
            "refines_edges_dedup": sum(
                1 for _, _, d in refines if d.get("origin") == "dedup"),
            "keep_superseded": self.keep_superseded,
            "mean_degree": (2 * undirected.number_of_edges() / node_count) if node_count else 0.0,
            "connected_components": nx.number_connected_components(undirected) if node_count else 0,
            "contradiction_density": (len(contradicts) / possible_pairs) if possible_pairs else 0.0,
            "nodes_in_contradiction": len(contradicting_nodes),
            "nodes_with_harmful_count": len(harmful_nodes),
            "contradicting_and_harmful": len(contradicting_nodes & harmful_nodes),
            "contradicts_enabled": self.enable_contradicts,
            "nli_available": self.backend.nli_available if self.enable_contradicts else None,
            "related_threshold": self.related_threshold,
            "nli_gate": self.nli_gate,
            "contradicts_threshold": self.contradicts_threshold,
        }

    def snapshot(self) -> str:
        """Readable rendering written to *_playbook.txt, so runs stay diffable."""
        return self._render(with_legend=False)

    # --- writing -----------------------------------------------------------

    def apply_operations(self, operations: List[Dict[str, Any]], step: int = 0) -> List[str]:
        """
        Apply ADD operations. Section routing and id minting mirror
        apply_curator_operations() exactly, so ids are comparable across arms.
        """
        added = []
        refinements = []

        for op in operations:
            if op.get("type") != "ADD":
                # Matches the flat playbook: non-ADD operations are dropped, not
                # applied. Nothing in this system emits them today.
                continue

            content = op.get("content", "")
            if not content:
                continue

            section_key = _normalize_section(op.get("section", "general"))
            if section_key not in self._section_keys() and section_key != "general":
                print(f"Warning: Section '{op.get('section')}' not found, adding to OTHERS")
                section_key = "others"
            if section_key == "general":
                section_key = "others"

            node_id = f"{get_section_slug(section_key)}-{self.next_global_id:05d}"
            self.next_global_id += 1

            self._add_node(node_id, section_key, content, step)
            added.append(node_id)
            print(f"  Added bullet {node_id} to section {section_key}")

            target = op.get("refines")
            if target:
                refinements.append((node_id, target))

        if added:
            self._link_new_nodes(added, step)

        # After the nodes exist, so a refines target added in the same batch is
        # resolvable -- and rejectable, since superseding a rule from the same
        # response means the curator contradicted itself rather than improved on
        # anything the generator ever used.
        for node_id, target in refinements:
            self._add_refines(node_id, target, set(added), step)

        return sort_by_mint_order(added)

    def refine(self, backend: Any, threshold: float) -> Dict[str, Any]:
        """
        Withdraw absorbed rules from the prompt without removing them from the graph.

        Same decision as the flat playbook (memory/refine.py), different storage: the flat arm
        deletes the line, here the survivor gains a `refines` edge and the existing
        supersession machinery stops rendering the absorbed node. Its content, counters and
        edges stay on disk, which is what makes --keep_superseded a real ablation.

        The edge carries origin="dedup" so structural_stats can separate it from a
        curator-declared supersession. Conflating them would make the graph arm's headline
        capability uncountable: on the seed-0 run the curator declared 2 in 50 calls, and a
        refines_edges figure that silently mixed in de-duplication would hide that.

        Counters are added to the survivor for the same reason as in the flat arm: the two
        rules said the same thing, so the evidence about that advice was split across them.
        """
        active = self.active_nodes()
        items = [(node_id, self.graph.nodes[node_id]["content"]) for node_id in active]
        groups = duplicate_groups(items, backend, threshold, vectors=self._embeddings)
        if not groups:
            return {"groups": 0, "absorbed": 0, "remaining": len(items)}

        absorbed = 0
        for survivor, victims in groups.items():
            for victim in victims:
                if not self._add_refines(survivor, victim, set(), step=-1, origin="dedup"):
                    continue
                self.graph.nodes[survivor]["helpful"] += self.graph.nodes[victim]["helpful"]
                self.graph.nodes[survivor]["harmful"] += self.graph.nodes[victim]["harmful"]
                absorbed += 1

        return {"groups": len(groups), "absorbed": absorbed,
                "remaining": len(items) - absorbed}

    def _add_refines(self, node_id: str, target: str, batch: set, step: int,
                     origin: str = "curator") -> bool:
        """
        Record that `node_id` supersedes `target`, if that claim is admissible.

        Every rejection below costs one relation and leaves the new rule in place.
        A bad supersession is worse than a missing one: it withdraws a rule the
        generator can no longer see, so the bar for accepting one is deliberately
        higher than for related_to.
        """
        if target == node_id:
            return False

        if target not in self.graph:
            print(f"Warning: {node_id} claims to refine unknown rule {target}; ignoring")
            return False

        if target in batch:
            print(f"Warning: {node_id} refines {target} from the same curator response; ignoring")
            return False

        # A cycle would leave every rule in it superseded and none of them rendered,
        # silently emptying a section.
        if self._supersedes(target, node_id):
            print(f"Warning: {node_id} refines {target} would close a cycle; ignoring")
            return False

        self.graph.add_edge(node_id, target, type=REFINES, score=1.0, created_step=step,
                            origin=origin)
        print(f"  {node_id} supersedes {target}"
              f"{' (duplicate)' if origin == 'dedup' else ''}")
        return True

    def _supersedes(self, source: str, target: str) -> bool:
        """True when source already refines target, directly or through a chain."""
        seen = set()
        frontier = [source]
        while frontier:
            current = frontier.pop()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(
                other for _, other, data in self.graph.out_edges(current, data=True)
                if data.get("type") == REFINES
            )
        return False

    def update_counts(self, bullet_tags: List[Dict[str, str]]) -> None:
        if not bullet_tags:
            return

        # Last tag wins per id and each rule moves at most one step per call,
        # matching update_bullet_counts()'s dict-based collapse.
        tag_map = {}
        for tag in bullet_tags:
            if not isinstance(tag, dict):
                continue
            bullet_id = tag.get("id") or tag.get("bullet", "")
            if bullet_id:
                tag_map[bullet_id] = tag.get("tag", "neutral")

        unmatched = []
        for bullet_id, tag in tag_map.items():
            if bullet_id not in self.graph:
                unmatched.append(bullet_id)
                continue
            if tag == "helpful":
                self.graph.nodes[bullet_id]["helpful"] += 1
            elif tag == "harmful":
                self.graph.nodes[bullet_id]["harmful"] += 1

        if unmatched:
            preview = ", ".join(sorted(unmatched)[:5])
            suffix = ", ..." if len(unmatched) > 5 else ""
            print(f"Warning: {len(unmatched)} tagged id(s) not found in graph: {preview}{suffix}")

    # --- lifecycle ---------------------------------------------------------

    def copy(self) -> "GraphMemory":
        clone = GraphMemory(
            related_threshold=self.related_threshold,
            nli_gate=self.nli_gate,
            contradicts_threshold=self.contradicts_threshold,
            enable_contradicts=self.enable_contradicts,
            keep_superseded=self.keep_superseded,
            backend=self.backend,  # models are read-only, safe to share
            sections=list(self.sections),
            next_global_id=self.next_global_id,
            retrieval=self.retrieval,
            retrieval_budget=self.retrieval_budget,
            retrieval_seeds=self.retrieval_seeds,
        )
        clone.graph = self.graph.copy()
        clone._embeddings = dict(self._embeddings)
        clone._insertion_counter = self._insertion_counter
        return clone

    def save(self, path: str) -> None:
        """Write the readable snapshot, plus the graph itself as a sibling JSON."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.snapshot())

        with open(graph_path_for(path), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": "ace-graph-memory/1",
            "next_global_id": self.next_global_id,
            "sections": list(self.sections),
            "thresholds": {
                "related_threshold": self.related_threshold,
                "nli_gate": self.nli_gate,
                "contradicts_threshold": self.contradicts_threshold,
                "enable_contradicts": self.enable_contradicts,
                "keep_superseded": self.keep_superseded,
            },
            "graph": nx.node_link_data(self.graph, edges="edges"),
        }

    @classmethod
    def load(cls, path: str, **kwargs) -> "GraphMemory":
        """
        Rebuild from a saved graph, or seed from a flat playbook.

        Seeding from a flat playbook is what makes a graph run startable from a
        baseline playbook: nodes come from the bullet lines, and edges are then
        derived from scratch by the usual similarity and NLI pass.
        """
        sidecar = graph_path_for(path)
        if os.path.exists(sidecar):
            with open(sidecar, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f), **kwargs)

        with open(path, "r", encoding="utf-8") as f:
            return cls.from_playbook_text(f.read(), **kwargs)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], **kwargs) -> "GraphMemory":
        # Thresholds saved with the graph are the defaults on reload; an explicit
        # keyword still wins, so a run can be resumed with different settings.
        thresholds = payload.get("thresholds", {})
        for key in ("related_threshold", "nli_gate", "contradicts_threshold",
                    "enable_contradicts", "keep_superseded"):
            if key not in kwargs and key in thresholds:
                kwargs[key] = thresholds[key]

        memory = cls(sections=payload.get("sections"), **kwargs)
        memory.graph = nx.node_link_graph(
            payload["graph"], directed=True, multigraph=True, edges="edges"
        )
        memory.next_global_id = payload.get("next_global_id", 1)
        memory._insertion_counter = 1 + max(
            (memory.graph.nodes[n].get("order", 0) for n in memory.graph.nodes()), default=-1
        )
        return memory

    @classmethod
    def from_playbook_text(cls, text: str, **kwargs) -> "GraphMemory":
        """Parse a flat playbook into nodes, then derive edges over the whole set."""
        memory = cls(**kwargs)

        section_key = "others"
        added = []
        for line in (text or "").split("\n"):
            stripped = line.strip()
            if stripped.startswith("##"):
                section_key = _normalize_section(stripped[2:].strip())
                continue

            parsed = parse_playbook_line(line)
            if not parsed:
                continue

            memory._add_node(
                parsed["id"],
                section_key if section_key in memory._section_keys() else "others",
                parsed["content"],
                step=0,
                helpful=parsed["helpful"],
                harmful=parsed["harmful"],
            )
            added.append(parsed["id"])

        memory.next_global_id = _next_id_from(added)
        if added:
            memory._link_new_nodes(added, step=0)
        return memory

    # --- graph construction ------------------------------------------------

    def _add_node(self, node_id, section_key, content, step, helpful=0, harmful=0):
        self.graph.add_node(
            node_id,
            section=section_key,
            section_header=self._section_header(section_key),
            content=content,
            helpful=helpful,
            harmful=harmful,
            created_step=step,
            order=self._insertion_counter,
        )
        self._insertion_counter += 1

    def _link_new_nodes(self, new_ids: List[str], step: int):
        """
        Attach newly added rules to the rules already present.

        Only the new rules are embedded; existing vectors are cached. Contradiction
        scoring is restricted to pairs that already cleared nli_gate, which is what
        keeps this from becoming an all-pairs NLI sweep.
        """
        contents = [self.graph.nodes[n]["content"] for n in new_ids]
        vectors = self.backend.encode(contents)
        for node_id, vector in zip(new_ids, vectors):
            self._embeddings[node_id] = vector

        new_set = set(new_ids)
        existing = list(self.graph.nodes())
        nli_candidates: List[Tuple[str, str]] = []

        for node_id in new_ids:
            vector = self._embeddings[node_id]
            for other in existing:
                if other == node_id:
                    continue
                # A pair of rules added in the same call is visible from both sides.
                # Processing it only from the later one links it exactly once.
                if other in new_set and not self._is_earlier(other, node_id):
                    continue

                other_vector = self._embeddings.get(other)
                if other_vector is None:
                    continue

                similarity = float(vector @ other_vector)
                if similarity >= self.related_threshold:
                    self.graph.add_edge(
                        node_id, other, type=RELATED_TO,
                        score=round(similarity, 4), created_step=step,
                    )
                if self.enable_contradicts and similarity >= self.nli_gate:
                    nli_candidates.append((node_id, other))

        self._score_contradictions(nli_candidates, step)

    def _score_contradictions(self, candidates: List[Tuple[str, str]], step: int):
        if not self.enable_contradicts or not candidates or not self.backend.nli_available:
            return

        # NLI is directional, so each pair is scored both ways and the stronger
        # reading wins. "Never use bare except" vs "bare except is fine here"
        # reads as a contradiction in one order more clearly than the other.
        forward = [
            (self.graph.nodes[a]["content"], self.graph.nodes[b]["content"])
            for a, b in candidates
        ]
        backward = [(b, a) for a, b in forward]

        forward_scores = self.backend.contradiction_scores(forward)
        backward_scores = self.backend.contradiction_scores(backward)
        if not forward_scores or not backward_scores:
            return

        for (a, b), fwd, bwd in zip(candidates, forward_scores, backward_scores):
            score = max(fwd, bwd)
            if score >= self.contradicts_threshold:
                self.graph.add_edge(
                    a, b, type=CONTRADICTS, score=round(score, 4), created_step=step
                )

    # --- rendering ---------------------------------------------------------

    def _render(self, with_legend: bool, only: Optional[set] = None) -> str:
        """Render the live rules, or `only` of them when retrieval has made a selection.

        A relation pointing at a rule that was not selected is dropped, exactly as one
        pointing at a superseded rule is: telling the model to consider a rule it cannot read
        spends tokens to create a dangling reference.
        """
        superseded = self.superseded_nodes()
        hidden = superseded if only is None else superseded | {
            node for node in self.graph.nodes() if node not in only
        }
        if with_legend:
            blocks = [build_legend(self.enable_contradicts, include_refines=bool(superseded))]
        else:
            blocks = []

        by_section, ordered_keys = self._group_by_section()

        for key in ordered_keys:
            lines = [f"## {self._section_header(key)}"]
            for node_id in by_section.get(key, []):
                if node_id in hidden:
                    continue
                lines.append(self._format_node(node_id))
                lines.extend(self._format_relations(node_id, hidden))
            blocks.append("\n".join(lines))

        conflicts = self._render_conflicts()
        if conflicts:
            blocks.append(conflicts)

        return "\n\n".join(blocks)

    def _format_node(self, node_id: str) -> str:
        data = self.graph.nodes[node_id]
        return format_playbook_line(node_id, data["helpful"], data["harmful"], data["content"])

    def _format_relations(self, node_id: str, superseded: Optional[set] = None) -> List[str]:
        superseded = superseded if superseded is not None else set()
        lines = []

        # What this rule replaced, stated outward only: the withdrawn rule is not
        # rendered, so listing it from the other side would point at nothing.
        #
        # De-duplication supersessions are excluded. They are bookkeeping, not knowledge --
        # and the flat arm records nothing at all when it drops a duplicate line, so
        # rendering them here would charge the graph arm tokens for the same decision and
        # reintroduce the asymmetry retrieval exists to remove.
        replaced = sorted(
            target for _, target, data in self.graph.out_edges(node_id, data=True)
            if data.get("type") == REFINES and data.get("origin", "curator") == "curator"
        )
        if replaced:
            lines.append(f"    {REFINES}: {', '.join(replaced)} (withdrawn)")

        for edge_type in (RELATED_TO, CONTRADICTS):
            # Never point a live rule at a withdrawn one; the model would be told to
            # consider a rule it cannot read.
            neighbours = [n for n in self._neighbours(node_id, edge_type) if n not in superseded]
            if neighbours:
                lines.append(f"    {edge_type}: {', '.join(neighbours)}")
        return lines

    def _render_conflicts(self) -> str:
        """
        Pair up contradicting rules with their track records.

        This block is the experimental treatment. A flat playbook can hold two
        conflicting rules but has no way to say that they conflict, so a generator
        reading it has no signal to arbitrate between them.
        """
        superseded = self.superseded_nodes()
        pairs = []
        for source, target, data in self.graph.edges(data=True):
            if data.get("type") != CONTRADICTS:
                continue
            if source in superseded or target in superseded:
                # A conflict with a withdrawn rule is not a live conflict.
                continue
            a, b = sorted((source, target))
            pairs.append((a, b))

        if not pairs:
            return ""

        lines = [CONFLICTS_HEADER, CONFLICTS_NOTE]
        for a, b in sorted(set(pairs)):
            lines.append(f"{self._describe(a)}  <->  {self._describe(b)}")
        return "\n".join(lines)

    def _describe(self, node_id: str) -> str:
        data = self.graph.nodes[node_id]
        return f"{node_id} (helpful={data['helpful']} harmful={data['harmful']})"

    # --- internals ---------------------------------------------------------

    def _neighbours(self, node_id: str, edge_type: str) -> List[str]:
        """Edges are stored directed but read symmetrically."""
        found = set()
        for _, target, data in self.graph.out_edges(node_id, data=True):
            if data.get("type") == edge_type:
                found.add(target)
        for source, _, data in self.graph.in_edges(node_id, data=True):
            if data.get("type") == edge_type:
                found.add(source)
        return sorted(found)

    def _edges_of_type(self, edge_type: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        return [
            (u, v, d) for u, v, d in self.graph.edges(data=True) if d.get("type") == edge_type
        ]

    def superseded_nodes(self) -> set:
        """
        Rules withdrawn from the prompt because something refines them.

        A rule is superseded as soon as anything refines it, so chains collapse for
        free: if C refines B and B refines A, both A and B have an incoming edge and
        only C survives into the prompt.
        """
        if self.keep_superseded:
            return set()
        return {
            target for _, target, data in self.graph.edges(data=True)
            if data.get("type") == REFINES
        }

    def active_nodes(self) -> List[str]:
        """Rules the generator can currently see."""
        superseded = self.superseded_nodes()
        return [n for n in self._nodes_in_reading_order() if n not in superseded]

    def _group_by_section(self) -> Tuple[Dict[str, List[str]], List[str]]:
        """Nodes bucketed by section, plus the section order to read them in."""
        known_keys = self._section_keys()
        by_section: Dict[str, List[str]] = {key: [] for key in known_keys}

        for node_id in self._ordered_nodes():
            by_section.setdefault(self.graph.nodes[node_id]["section"], []).append(node_id)

        # Sections outside the standard list are read last, so a rule can never
        # silently disappear just because its section is unrecognised.
        extra_keys = [key for key in by_section if key not in known_keys]
        return by_section, known_keys + extra_keys

    def _nodes_in_reading_order(self) -> List[str]:
        """
        Section order first, insertion order within a section.

        This is the order a flat playbook is read in, and matching it keeps the
        reflector's view and the by_section stats identical across arms. Ordering
        by insertion alone would hand the reflector the same bullets in a different
        sequence, which is a prompt difference the experiment did not intend.
        """
        by_section, ordered_keys = self._group_by_section()
        return [node_id for key in ordered_keys for node_id in by_section.get(key, [])]

    def _ordered_nodes(self) -> List[str]:
        """Pure insertion order, used to build the section buckets."""
        return sorted(self.graph.nodes(), key=lambda n: self.graph.nodes[n].get("order", 0))

    def _is_earlier(self, a: str, b: str) -> bool:
        return self.graph.nodes[a].get("order", 0) < self.graph.nodes[b].get("order", 0)

    def _section_keys(self) -> List[str]:
        return [_normalize_section(header) for header in self.sections]

    def _section_header(self, section_key: str) -> str:
        for header in self.sections:
            if _normalize_section(header) == section_key:
                return header
        return section_key.replace("_", " ").upper()


def graph_path_for(path: str) -> str:
    """Sidecar JSON path for a given *_playbook.txt destination."""
    stem, _ = os.path.splitext(path)
    return f"{stem}.graph.json"


def _normalize_section(name: str) -> str:
    """Same normalization apply_curator_operations() uses on section names."""
    return normalize_section_name(name)


def _default_sections() -> List[str]:
    return [
        line.strip()[2:].strip()
        for line in EMPTY_PLAYBOOK.split("\n")
        if line.strip().startswith("##")
    ]


def _next_id_from(bullet_ids: List[str]) -> int:
    highest = 0
    for bullet_id in bullet_ids:
        _, _, tail = bullet_id.rpartition("-")
        if tail.isdigit():
            highest = max(highest, int(tail))
    return highest + 1
