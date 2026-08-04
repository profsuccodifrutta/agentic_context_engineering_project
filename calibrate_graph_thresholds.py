#!/usr/bin/env python3
"""
Pick GraphMemory's edge thresholds by looking at what they actually produce.

    uv run python calibrate_graph_thresholds.py <playbook.txt> [--top 40]

Why this exists: the contradicts edge is the treatment in the linear-vs-relational
comparison, and NLI cross-encoders are trained on descriptive premise/hypothesis
pairs, not on prescriptive advice. Two rules that both begin "Prioritize ..." read
as competing directives to the model even when they are complementary, so the
default threshold over-fires on a real playbook. A conflicts block that is mostly
spurious does not test the hypothesis -- it injects noise into the generator prompt
and would plausibly make the graph arm lose for reasons unrelated to structure.

So: run this on a real playbook, read the ranked pairs, and set
--contradicts_threshold at the point where the pairs stop being real disagreements.
Report the chosen value alongside the results.
"""

import argparse
import sys
from itertools import combinations

from ace.memory.embeddings import EmbeddingBackend
from playbook_utils import parse_playbook_line


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("playbook", help="Path to a *_playbook.txt to calibrate against")
    parser.add_argument("--top", type=int, default=40,
                        help="How many top-scoring contradiction candidates to print")
    parser.add_argument("--nli_gate", type=float, default=0.45,
                        help="Similarity floor below which pairs are not scored at all")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.playbook, "r", encoding="utf-8") as f:
        text = f.read()

    bullets = []
    for line in text.split("\n"):
        parsed = parse_playbook_line(line)
        if parsed:
            bullets.append((parsed["id"], parsed["content"]))

    if len(bullets) < 2:
        print(f"Need at least 2 bullets to calibrate; found {len(bullets)}.")
        return 1

    print(f"{len(bullets)} rules from {args.playbook}\n")

    backend = EmbeddingBackend()
    vectors = backend.encode([content for _, content in bullets])

    pairs = []
    for (i, (id_a, text_a)), (j, (id_b, text_b)) in combinations(enumerate(bullets), 2):
        similarity = float(vectors[i] @ vectors[j])
        pairs.append((similarity, id_a, text_a, id_b, text_b))

    # --- related_to ------------------------------------------------------
    print("=" * 78)
    print("related_to  (cosine similarity)")
    print("=" * 78)
    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        count = sum(1 for p in pairs if p[0] >= threshold)
        print(f"  >= {threshold:.2f}: {count:4d} edges  "
              f"({count / len(bullets):.1f} per rule)")

    # --- contradicts -----------------------------------------------------
    candidates = [p for p in pairs if p[0] >= args.nli_gate]
    print()
    print("=" * 78)
    print(f"contradicts  (NLI, scored on the {len(candidates)} pairs above "
          f"similarity {args.nli_gate})")
    print("=" * 78)

    if not candidates:
        print("  no pairs cleared the gate; lower --nli_gate")
        return 0

    forward = [(p[2], p[4]) for p in candidates]
    backward = [(p[4], p[2]) for p in candidates]
    forward_scores = backend.contradiction_scores(forward)
    backward_scores = backend.contradiction_scores(backward)

    if not forward_scores:
        print("  NLI model unavailable; cannot calibrate contradicts")
        return 1

    scored = sorted(
        ((max(f, b), p[1], p[2], p[3], p[4]) for p, f, b in
         zip(candidates, forward_scores, backward_scores)),
        reverse=True,
    )

    for threshold in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
        count = sum(1 for s in scored if s[0] >= threshold)
        print(f"  >= {threshold:.2f}: {count:4d} edges")

    print()
    print(f"--- top {min(args.top, len(scored))} candidates, highest score first ---")
    print("Read down until the pairs stop being real disagreements; that score is your cut.\n")
    for score, id_a, text_a, id_b, text_b in scored[:args.top]:
        print(f"[{score:.3f}] {id_a}  <->  {id_b}")
        print(f"          {text_a[:110]}")
        print(f"          {text_b[:110]}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
