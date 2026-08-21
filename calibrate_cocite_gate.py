"""Held-out comparison of co-citation edge predicates: confidence against lift.

The shipped gate is *confidence*, P(both cited | both rendered) >= --cocite_min_rate. Confidence
favours pairs whose members are each cited often, frequent citation tracks high relevance, and so
the surviving edges point where relevance already looks. Lift divides that by what independent
citation would have produced, so it ranks pairs by how much co-citation exceeds chance rather
than by popularity.

Both are rankings over the same candidate pairs, so the honest comparison is at matched edge
count: build each ranking on the first half of a run's generations, take the top k of each, and
measure on the unseen second half how much better than base rate each set predicts co-citation.
No LLM, no network, reads a finished run only.

    uv run python calibrate_cocite_gate.py <run_dir> [<run_dir> ...] [--min-opportunity 3]
    uv run python calibrate_cocite_gate.py --selftest

ponytail: the pair stream is rebuilt from `bullet_usage_log.jsonl`, which records the pre-train
generation of each step. The live memory also observes the post-curate generation through
`record_generation`, so the counts here are roughly half of what a run accumulates. That is
applied identically to both predicates, so the ranking comparison is unaffected; only the
absolute opportunity counts are. Recover the post-curate half from `detailed_llm_logs/` prompts
if an absolute threshold ever needs to be pinned from this script.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from itertools import combinations

#: Rule ids the harness emits look like `calc-00030`. The generator also cites section headings
#: ("FORMULAS & CALCULATIONS"), which are not rules and must not enter the pair space.
RULE_ID = re.compile(r"^[a-z]+-\d+$")


def generations(run_dir):
    """Each logged generation as (rendered_ids, cited_ids), in run order."""
    path = os.path.join(run_dir, "bullet_usage_log.jsonl")
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            rendered = {b for b in (row.get("bullet_ids_rendered") or []) if RULE_ID.match(b)}
            # A citation only counts as evidence about a pair if the rule was actually on the
            # page; anything else is the model naming an id it was never shown.
            cited = {b for b in (row.get("bullet_ids_used") or []) if RULE_ID.match(b)} & rendered
            if len(rendered) >= 2:
                out.append((rendered, cited))
    return out


def accumulate(rows):
    """Pair and per-rule counts over a slice of generations."""
    pair = defaultdict(lambda: [0, 0])  # (a, b) -> [cocited, corendered]
    rule = defaultdict(lambda: [0, 0])  # a -> [cited, rendered]
    for rendered, cited in rows:
        for a in rendered:
            rule[a][1] += 1
            if a in cited:
                rule[a][0] += 1
        for a, b in combinations(sorted(rendered), 2):
            slot = pair[(a, b)]
            slot[1] += 1
            if a in cited and b in cited:
                slot[0] += 1
    return pair, rule


def rankings(pair, rule, min_opportunity):
    """Candidate pairs scored by each predicate, best first."""
    conf, lift = [], []
    for key, (cocited, corendered) in pair.items():
        if corendered < min_opportunity:
            continue
        c = cocited / corendered
        conf.append((c, key))
        a, b = key
        pa = rule[a][0] / rule[a][1] if rule[a][1] else 0.0
        pb = rule[b][0] / rule[b][1] if rule[b][1] else 0.0
        expected = pa * pb
        # A pair whose members are never cited alone has no baseline to beat; chance is zero and
        # the ratio is undefined, so it cannot be ranked and is dropped rather than given +inf.
        lift.append((c / expected, key) if expected > 0 else (0.0, key))
    conf.sort(key=lambda t: (-t[0], t[1]))
    lift.sort(key=lambda t: (-t[0], t[1]))
    return conf, lift


def evaluate(rows, edges):
    """(co-render events touching an edge, of those co-cited) on held-out generations."""
    opportunities = hits = 0
    for rendered, cited in rows:
        for a, b in combinations(sorted(rendered), 2):
            if (a, b) in edges:
                opportunities += 1
                if a in cited and b in cited:
                    hits += 1
    return opportunities, hits


def base_rate(rows):
    opportunities = hits = 0
    for rendered, cited in rows:
        for a, b in combinations(sorted(rendered), 2):
            opportunities += 1
            if a in cited and b in cited:
                hits += 1
    return hits, opportunities


def report(run_dir, min_opportunity, ks, shipped_rate):
    rows = generations(run_dir)
    if len(rows) < 20:
        print(f"  {os.path.basename(run_dir)}: only {len(rows)} usable generations, skipped")
        return None
    cut = len(rows) // 2
    train, held = rows[:cut], rows[cut:]
    pair, rule = accumulate(train)
    conf, lift = rankings(pair, rule, min_opportunity)
    hits, opportunities = base_rate(held)
    base = hits / opportunities if opportunities else 0.0

    print(f"\n{os.path.basename(run_dir)}")
    print(f"  {len(rows)} generations, split {len(train)} / {len(held)}")
    print(f"  candidate pairs at opportunity >= {min_opportunity}: {len(conf)}")
    print(f"  held out: {opportunities} co-render events, {hits} co-cited, base rate {base:.4f}")
    if not conf or base == 0.0:
        print("  nothing to rank")
        return None

    print(f"  {'k':>5}  {'confidence':>22}  {'lift':>22}")
    print(f"  {'':>5}  {'events  cocited  x base':>22}  {'events  cocited  x base':>22}")
    out = {}
    for k in ks:
        if k > len(conf):
            continue
        cells = []
        for ranking in (conf, lift):
            edges = {key for _, key in ranking[:k]}
            opp, hit = evaluate(held, edges)
            ratio = (hit / opp) / base if opp else 0.0
            cells.append((opp, hit, ratio))
            out.setdefault(k, []).append(ratio)
        print(f"  {k:5d}  {cells[0][0]:6d} {cells[0][1]:8d} {cells[0][2]:8.2f}x  "
              f"{cells[1][0]:6d} {cells[1][1]:8d} {cells[1][2]:8.2f}x")

    shipped = {key for score, key in conf if score >= shipped_rate}
    opp, hit = evaluate(held, shipped)
    ratio = (hit / opp) / base if opp else 0.0
    print(f"  shipped gate (confidence >= {shipped_rate}, opportunity >= {min_opportunity}): "
          f"{len(shipped)} edges, {opp} events, {ratio:.2f}x base")
    return out


def selftest():
    """Two limits where the answer is known before running."""
    # A pair that is always co-cited when co-rendered, against members that are otherwise never
    # cited: confidence 1.0, and lift far above 1 because chance predicts almost nothing.
    rows = []
    for i in range(20):
        rows.append(({"a-1", "a-2", "a-3"}, {"a-1", "a-2"} if i % 2 == 0 else set()))
    pair, rule = accumulate(rows)
    assert pair[("a-1", "a-2")] == [10, 20], pair[("a-1", "a-2")]
    assert rule["a-3"] == [0, 20], rule["a-3"]
    conf, lift = rankings(pair, rule, 3)
    assert conf[0][1] == ("a-1", "a-2") and abs(conf[0][0] - 0.5) < 1e-9
    # p(a-1)=p(a-2)=0.5, so chance is 0.25 and lift is 2.0.
    assert lift[0][1] == ("a-1", "a-2") and abs(lift[0][0] - 2.0) < 1e-9
    # A pair never co-cited scores zero under both.
    assert all(score == 0.0 for score, key in conf if key != ("a-1", "a-2"))
    # Held-out evaluation of an edge that always fires returns the rate itself.
    opp, hit = evaluate(rows, {("a-1", "a-2")})
    assert (opp, hit) == (20, 10), (opp, hit)
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*")
    ap.add_argument("--min-opportunity", type=int, default=3,
                    help="pairs co-rendered fewer times than this are not ranked (default 3)")
    ap.add_argument("--k", type=int, nargs="+", default=[10, 25, 50, 100, 200],
                    help="edge counts at which the two rankings are compared")
    ap.add_argument("--shipped-rate", type=float, default=0.20,
                    help="the confidence gate the runs shipped with (default 0.20)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return 0
    if not args.run_dirs:
        ap.error("give at least one run directory, or --selftest")

    summary = defaultdict(list)
    for run_dir in args.run_dirs:
        out = report(run_dir, args.min_opportunity, args.k, args.shipped_rate)
        if out:
            for k, (conf_ratio, lift_ratio) in out.items():
                summary[k].append((conf_ratio, lift_ratio))

    if summary:
        print("\n--- across runs, mean x base rate ---")
        print(f"  {'k':>5}  {'confidence':>11}  {'lift':>11}  {'runs':>5}")
        for k in sorted(summary):
            vals = summary[k]
            mean_conf = sum(v[0] for v in vals) / len(vals)
            mean_lift = sum(v[1] for v in vals) / len(vals)
            print(f"  {k:5d}  {mean_conf:10.2f}x  {mean_lift:10.2f}x  {len(vals):5d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
