"""
Query-conditioned retrieval: the mechanism that makes the comparison a test of structure.

ACE (arXiv:2510.04618) passes the whole playbook to the generator and lets the model select the
relevant bullets itself; §3.1's "fine-grained retrieval" refers to that in-prompt selection, not
to a retrieval step. Retrieval is therefore an extension made for this experiment. Before it,
`render_for_generator(query=...)` raised NotImplementedError in both arms, so
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

import heapq
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from utils import count_tokens

RETRIEVAL_MODES = ("none", "flat", "graph")

#: A budget, not a limit. Chosen against the measured problem: on the seed-0 formula run a
#: 75-rule playbook was 7,103 prompt tokens at the median, i.e. 43% of the generator's served
#: ~16,384-token window spent before it began to reason.
DEFAULT_RETRIEVAL_BUDGET = 2500

#: How many rules the graph arm seeds its walk from. Small on purpose: seeding with everything
#: makes the walk irrelevant and the arm reduces to flat retrieval.
DEFAULT_RETRIEVAL_SEEDS = 5


def per_query_scores(
    query: Union[str, Sequence[str]],
    items: Sequence[Tuple[str, str]],
    backend: Any,
    vectors: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, float]]:
    """One score map per sub-query, in the order the sub-queries were supplied.

    `score_against_query` collapses the sub-queries with `max`, which is the right answer for
    *ranking* a rule but throws away which sub-question wanted it. Round-robin allocation needs
    that, so this returns the maps unreduced. A single string yields one map and
    `score_against_query` is exactly its `max`, so nothing that supplies one string changes.
    """
    if not items:
        return []

    known = vectors if vectors is not None else {}
    missing = [(rule_id, content) for rule_id, content in items if rule_id not in known]
    if missing:
        encoded = backend.encode([content for _, content in missing])
        for (rule_id, _), vector in zip(missing, encoded):
            known[rule_id] = vector

    queries = [query] if isinstance(query, str) else [q for q in query if q]
    if not queries:
        return [{rule_id: 0.0 for rule_id, _ in items}]
    query_vectors = backend.encode(queries)
    return [{rule_id: float(qv @ known[rule_id]) for rule_id, _ in items}
            for qv in query_vectors]


def score_against_query(
    query: Union[str, Sequence[str]],
    items: Sequence[Tuple[str, str]],
    backend: Any,
    vectors: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Cosine between the question and each rule. `encode` L2-normalises, so dot is cosine.

    **A query may be several strings, and a rule then scores as its best match against any one
    of them.** One `finer` row is four independent sub-questions answered in a single call, and
    scoring against their concatenation makes a rule that is *slightly* related to all four beat
    a rule that answers one of them exactly. Measured on a real checkpoint against held-out
    validation questions, that is what happened, and taking the max over sub-queries reverses it:
    docs/EXPERIMENT_FINER.md section 9, 2026-09-03, carries the table. A single string is one
    sub-query, so every task that supplies one behaves exactly as before.

    `vectors` is filled in place when given, so a memory can hold one cache across calls.
    Retrieval runs on every generation, and re-embedding the whole playbook each time would
    make it the dominant cost of the arm.
    """
    per_query = per_query_scores(query, items, backend, vectors)
    if not per_query:
        return {}
    return allocate(per_query, subquery_seeding=False)[0]


def allocate(per_query: Sequence[Dict[str, float]], subquery_seeding: bool):
    """(each rule's best-match score, the order the retrieval budget is filled in).

    **The fairness guarantee between the arms lives here and nowhere else.** Both memories fill
    their budget in this order, and the graph arm's walk appends its relevance tail in it, so
    the flat arm and the treatment can never allocate slots by different rules. With
    `subquery_seeding` the order is round-robin across the sub-queries' own rankings; without
    it, and always on a task that supplies one query, it is the global relevance order.
    """
    scores = {rule_id: max(sub[rule_id] for sub in per_query) for rule_id in per_query[0]}
    if subquery_seeding and len(per_query) > 1:
        return scores, round_robin_order(per_query)
    return scores, flat_order(scores)


def seed_groups(per_query: Sequence[Dict[str, float]], scores: Dict[str, float],
                total_seeds: int, subquery_seeding: bool) -> List[List[str]]:
    """The walk's seeds, as one list per sub-query they were allocated for.

    With `subquery_seeding` each sub-query contributes its own top `ceil(total / count)`; without
    it there is a single group, the global top `total_seeds`. Returned grouped so a check can ask
    whether every sub-question is represented without re-deriving which rules belong to which.
    """
    if not subquery_seeding or len(per_query) <= 1:
        return [flat_order(scores)[:total_seeds]]
    take = seeds_per_query(total_seeds, len(per_query))
    return [flat_order(sub)[:take] for sub in per_query]


def round_robin_order(per_query: Sequence[Dict[str, float]]) -> List[str]:
    """Every rule, taken one at a time from each sub-query's own ranking in turn.

    A global ranking over `max` starves a sub-question: on `finer` all four sub-questions were
    represented in 1.6% of generations. Taking in turn guarantees each one is represented in
    proportion to how many slots there are, whatever the budget turns out to fit.

    **One sub-query degenerates to `flat_order` exactly**, so `formula` and MBPP are unchanged.
    Ties inside a sub-query break on the id, and a rule already taken is skipped rather than
    re-emitted, so the order is total and deterministic.
    """
    if not per_query:
        return []
    if len(per_query) == 1:
        return flat_order(per_query[0])

    rankings = [flat_order(scores) for scores in per_query]
    cursors = [0] * len(rankings)
    ordered: List[str] = []
    seen = set()
    while len(seen) < len(per_query[0]):
        progressed = False
        for index, ranking in enumerate(rankings):
            while cursors[index] < len(ranking) and ranking[cursors[index]] in seen:
                cursors[index] += 1
            if cursors[index] >= len(ranking):
                continue
            rule_id = ranking[cursors[index]]
            cursors[index] += 1
            seen.add(rule_id)
            ordered.append(rule_id)
            progressed = True
        if not progressed:
            break
    return ordered


def seeds_per_query(total_seeds: int, query_count: int) -> int:
    """How many walk seeds each sub-query gets.

    `ceil(total / count)` rather than a split, so every sub-query is represented even when
    there are more sub-queries than seeds. On one sub-query it returns `total_seeds`, which is
    the shipped behaviour; on `finer`'s four it returns 2, for 8 seeds rather than 5.
    """
    if query_count <= 1:
        return total_seeds
    return max(1, math.ceil(total_seeds / query_count))


def flat_order(scores: Dict[str, float]) -> List[str]:
    """Every rule, most relevant first. The id breaks ties so the order is deterministic."""
    return sorted(scores, key=lambda rule_id: (-scores[rule_id], rule_id))


def graph_order(
    scores: Dict[str, float],
    neighbours: Callable[[str], Sequence[str]],
    seeds: Union[int, Sequence[str]],
    tail: Optional[Sequence[str]] = None,
) -> List[str]:
    """Seeds by relevance, then outward along the edges, breadth first.

    `seeds` is a count, taken off the top of the relevance ranking, or an explicit list of
    seed ids when the caller allocated them itself (round-robin across sub-queries).

    `tail` is the order unreached rules are appended in, and it must be the order the flat arm
    would use. The counterfactual `walk_only_share` compares against is the flat arm's
    selection, so a tail ranked by a *different* rule turns ordinary relevance picks into false
    "walk-only" rules: with per-sub-query seeding the flat arm allocates round-robin, and a
    global-relevance tail here disagreed with it on rules no edge ever touched. Defaults to the
    global relevance order, which is what every run before sub-query seeding used.

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

    frontier = (list(seeds) if not isinstance(seeds, int)
                else flat_order(scores)[:seeds])
    frontier = [rid for rid in frontier if rid in scores]
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

    ordered.extend(rid for rid in (flat_order(scores) if tail is None else tail)
                   if rid not in seen and rid in scores)
    return ordered


def weighted_walk_order(
    scores: Dict[str, float],
    weighted_neighbours: Callable[[str], Sequence[Tuple[str, float]]],
    seeds: Sequence[str],
    tail: Optional[Sequence[str]] = None,
) -> List[str]:
    """Seeds by relevance, then outward by cumulative raw path strength, best first.

        strength(seed)  = relevance(seed)
        strength(child) = strength(parent) * weight(parent, child)

    **Why not breadth first.** Breadth first renders every hop-1 neighbour before any hop-2
    one, so on a real `finer` graph a 0.0071 edge is rendered ahead of a 0.9 edge one hop
    further out. Multiplying along the path fixes that ordering and makes depth expensive in
    proportion to how weak the evidence is, with no constant to tune.

    **Raw products, never per-node normalised.** Dividing each node's edges by its own best
    edge makes depth free along the locally-best chain: measured over 200 seed draws, that
    puts 41% of the rendered slice at hop 3 or beyond against breadth first's 7%, because it
    erases the absolute evidence scale the weights carry. That is a different treatment.

    **Strict precedence over the relevance tail.** Relevance is a cosine, roughly [0.2, 0.6];
    path strength after one hop is around 0.018. Merging them into one queue would render
    every unwalked rule before any walked one and erase the treatment, so the tail is appended
    after the walk is exhausted, exactly as `graph_order` does it.
    """
    seed_set = {rule_id for rule_id in seeds if rule_id in scores}
    ordered: List[str] = []
    best: Dict[str, float] = {}
    # Tier before strength, so every seed is emitted before any walked rule whatever the
    # arithmetic says. Relevance and path strength are not on one scale -- a seed at 0.2 and a
    # child of a 0.9 seed at 0.23 are not comparable quantities -- and the emission order the
    # pre-registration declares is seeds, then the walk, then the tail.
    heap: List[Tuple[int, float, str]] = []

    for rule_id in seeds:
        if rule_id not in scores:
            continue
        strength = scores[rule_id]
        if strength > best.get(rule_id, float("-inf")):
            best[rule_id] = strength
            heapq.heappush(heap, (0, -strength, rule_id))

    done = set()
    while heap:
        _tier, negated, rule_id = heapq.heappop(heap)
        if rule_id in done:
            continue
        done.add(rule_id)
        ordered.append(rule_id)
        strength = -negated
        for other, weight in weighted_neighbours(rule_id):
            if other in done or other not in scores or other in seed_set:
                continue
            reached = strength * weight
            if reached > best.get(other, 0.0):
                best[other] = reached
                heapq.heappush(heap, (1, -reached, other))

    # Same order the flat arm allocates in, for the reason given in `graph_order`.
    ordered.extend(rid for rid in (flat_order(scores) if tail is None else tail)
                   if rid not in done and rid in scores)
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
