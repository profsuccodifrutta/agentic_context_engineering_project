"""
Grow-and-refine: the paper's de-duplication step, as one decision rule both arms obey.

ACE lists three key innovations (arXiv:2510.04618 §3): a dedicated Reflector, incremental
delta updates, and grow-and-refine (§3.2) -- *"bullets with new identifiers are appended,
while existing bullets are updated in place (e.g., incrementing counters). A de-duplication
step then prunes redundancy by comparing bullets via semantic embeddings."* §4.7 calls it a
key enabler of the method.

This repo shipped it as `ace/core/bulletpoint_analyzer.py` and then turned it off in every
run, because it merges bullets and the graph arm could not mirror the removal. The cost of
that decision is measured in EXPERIMENT_FORMULA.md §10: the flat arm was ACE minus one of its
three components, it degenerated into 75 rules of which 23 were verbatim re-adds, and the
graph arm's `refines` was a sparse LLM-declared re-implementation of the mechanism its
competitor had been stripped of. Neither arm was the thing it was supposed to be.

The fix is symmetric rather than restorative. **One decision rule, computed here, applied by
each memory in its own representation.** The flat playbook prunes the line, which is what the
paper does; the graph withdraws the node from rendering while keeping it, its counters and its
edges in the graph. Both end up showing the same rules. What differs is storage -- which is
the independent variable, and now the only one.

Two deliberate departures from BulletpointAnalyzer, both to protect the comparison:

  * **Embeddings only, no LLM.** The analyzer asks a model to write a merged bullet. That
    would add a call whose output could differ between arms, reintroduce seed sensitivity into
    a step that is supposed to be deterministic, and put an LLM's judgement inside the
    treatment. Here the survivor's text is kept verbatim.
  * **Similarity is measured against the survivor, not transitively.** If A~B and B~C but not
    A~C, C survives. Transitive closure at threshold 0.90 chains unrelated rules together
    through a corridor of near-neighbours, and the chain it builds depends on iteration order.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

#: What BulletpointAnalyzer used, so "duplicate" means one thing repo-wide.
DEFAULT_DEDUP_THRESHOLD = 0.90


def duplicate_groups(
    items: Sequence[Tuple[str, str]],
    backend: Any,
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
    vectors: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[str]]:
    """
    Find rules that say the same thing as an earlier rule.

    Args:
        items: (rule_id, content) in the order the memory renders them. Both memories render
            section-first then insertion-order, so this sequence is identical across arms for
            the same accumulated rules -- which is what makes the decision identical.
        backend: EmbeddingBackend. `encode` L2-normalises, so a dot product is cosine.
        threshold: cosine at or above which two rules are the same rule.
        vectors: already-computed embeddings by rule id. GraphMemory caches these as it builds
            edges; reusing them is a speed optimisation only, since they come from the same
            encoder over the same text.

    Returns:
        {survivor_id: [absorbed_id, ...]}, containing only survivors that absorbed something.

    The earliest rule survives. That keeps ids stable across a refine pass, preserves the
    accumulated helpful/harmful history rather than restarting it, and makes the outcome
    independent of which rule the curator happened to write more fluently.
    """
    if len(items) < 2:
        return {}

    known = vectors or {}
    missing = [(rule_id, content) for rule_id, content in items if rule_id not in known]
    if missing:
        encoded = backend.encode([content for _, content in missing])
        known = dict(known)
        for (rule_id, _), vector in zip(missing, encoded):
            known[rule_id] = vector

    absorbed_by: Dict[str, str] = {}
    groups: Dict[str, List[str]] = {}

    for position, (survivor, _) in enumerate(items):
        if survivor in absorbed_by:
            continue
        for candidate, _ in items[position + 1:]:
            if candidate in absorbed_by:
                continue
            similarity = float(known[survivor] @ known[candidate])
            if similarity >= threshold:
                absorbed_by[candidate] = survivor
                groups.setdefault(survivor, []).append(candidate)

    return groups
