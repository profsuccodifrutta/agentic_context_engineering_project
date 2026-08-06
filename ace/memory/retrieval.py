"""
Query-conditioned retrieval: the mechanism that makes the comparison a test of structure.

ACE (arXiv:2510.04618 §3.1) lists *fine-grained retrieval, "so the Generator can focus on the
most pertinent knowledge"* as a property of the itemized bullet design. Neither arm here
implemented it -- `render_for_generator(query=...)` raised NotImplementedError in both -- so
every call injected the whole memory, and `related_to` edges were written by cosine similarity
and then read only for display text and `structural_stats()`, which no model ever sees.

That left the experiment unable to answer its own question. The only functional graph mechanism
(`refines`, i.e. eviction) needs no graph: a flat playbook with a `supersedes` field and a
render-time filter reproduces it exactly. The only mechanism that needs a graph (`related_to`)
did nothing. So a win either way was attributable to eviction, not to relational structure.

Retrieval closes that gap, and it has to be built in **both** arms at once or it becomes the
confound instead of the treatment:

    flat    score every rule against the question, render the highest-scoring that fit.
    graph   take the same top-k by the same score as seeds, then walk `related_to` outward,
            rendering neighbours of the seeds that similarity alone would not have surfaced.

Same encoder, same budget, same seed scoring, same prompt template, same emission order. The
single difference is whether the neighbourhood follows edges. That is the hypothesis, stated
as code.

**The budget covers the entire generator-facing rendering, legend and relation lines included.**
It would be easier to budget only the rule text, but that hands the graph arm its annotations
for free and reintroduces the +33% asymmetry retrieval exists to remove. Charging structure to
the same budget is the honest form of the question: the graph arm shows fewer rules for the
same tokens, and the structure has to pay for itself.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from utils import count_tokens

RETRIEVAL_MODES = ("none", "flat", "graph")

#: A budget, not a limit. Chosen against the measured problem: on the seed-0 formula run a
#: 75-rule playbook was 7,103 prompt tokens at the median, i.e. 43% of the generator's served
#: ~16,384-token window spent before it began to reason.
DEFAULT_RETRIEVAL_BUDGET = 2500

#: How many rules the graph arm seeds its walk from. Small on purpose: seeding with everything
#: makes the walk irrelevant and the arm reduces to flat retrieval.
DEFAULT_RETRIEVAL_SEEDS = 5


def score_against_query(
    query: str,
    items: Sequence[Tuple[str, str]],
    backend: Any,
    vectors: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Cosine between the question and each rule. `encode` L2-normalises, so dot is cosine.

    `vectors` is filled in place when given, so a memory can hold one cache across calls.
    Retrieval runs on every generation, and re-embedding the whole playbook each time would
    make it the dominant cost of the arm.
    """
    if not items:
        return {}

    known = vectors if vectors is not None else {}
    missing = [(rule_id, content) for rule_id, content in items if rule_id not in known]
    if missing:
        encoded = backend.encode([content for _, content in missing])
        for (rule_id, _), vector in zip(missing, encoded):
            known[rule_id] = vector

    query_vector = backend.encode([query])[0]
    return {rule_id: float(query_vector @ known[rule_id]) for rule_id, _ in items}


def flat_order(scores: Dict[str, float]) -> List[str]:
    """Every rule, most relevant first. The id breaks ties so the order is deterministic."""
    return sorted(scores, key=lambda rule_id: (-scores[rule_id], rule_id))


def graph_order(
    scores: Dict[str, float],
    neighbours: Callable[[str], Sequence[str]],
    seeds: int,
) -> List[str]:
    """Seeds by relevance, then outward along the edges, breadth first.

    Within a hop, more relevant first -- so the walk is deterministic and a neighbour that is
    also independently relevant is preferred, but a neighbour that is *only* reachable still
    outranks every unconnected rule. That is the whole treatment: relevance alone cannot
    surface it, and the flat arm never will.

    Rules the walk never reaches are appended in relevance order, so the graph arm can still
    fill its budget when the memory is sparsely connected. Without that tail a disconnected
    graph would render less than the flat arm from the same budget, and the comparison would
    be measuring connectivity of the embedding space rather than the value of the structure.
    """
    ordered: List[str] = []
    seen = set()

    frontier = [rid for rid in flat_order(scores)[:seeds]]
    for rule_id in frontier:
        ordered.append(rule_id)
        seen.add(rule_id)

    while frontier:
        nxt = []
        for rule_id in frontier:
            for other in neighbours(rule_id):
                if other in seen or other not in scores:
                    continue
                seen.add(other)
                nxt.append(other)
        nxt.sort(key=lambda rid: (-scores[rid], rid))
        ordered.extend(nxt)
        frontier = nxt

    ordered.extend(rid for rid in flat_order(scores) if rid not in seen)
    return ordered


def fit_to_budget(
    order: Sequence[str],
    estimate: Callable[[str, List[str]], int],
    render: Callable[[List[str]], str],
    budget: int,
) -> List[str]:
    """Take rules down the priority list while the rendering fits in `budget` tokens.

    Two passes because the estimate cannot be exact: in the graph arm a rule's cost includes
    the relation lines pointing at rules already chosen, and choosing a later rule can add a
    line to an earlier one. The greedy pass uses the estimate; the trim pass measures the real
    rendering and drops from the tail until it fits. Usually zero or one trim.

    The greedy pass starts already charged for the empty rendering -- section headers, and the
    graph arm's legend. Leaving that out made the estimate optimistic by a fixed amount and
    handed the whole job to the trim, which pops from the tail: at a small budget both the
    walk order and the relevance order then collapse onto the same single rule and the arms
    stop differing at all. It is also the honest accounting, since the legend is part of what
    the generator reads.

    A rule that does not fit is skipped rather than ending the scan, so a single long rule near
    the top cannot truncate the whole memory. At least one rule is always rendered, even if it
    alone exceeds the budget -- an empty memory is not a smaller memory, it is a different arm.
    """
    chosen: List[str] = []
    spent = count_tokens(render([]))
    for rule_id in order:
        cost = estimate(rule_id, chosen)
        if chosen and spent + cost > budget:
            continue
        chosen.append(rule_id)
        spent += cost

    while len(chosen) > 1 and count_tokens(render(chosen)) > budget:
        chosen.pop()
    return chosen
