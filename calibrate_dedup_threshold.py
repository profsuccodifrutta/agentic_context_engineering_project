#!/usr/bin/env python3
"""Choose `--dedup_threshold` for a benchmark by replaying a probe's curator operations.

Grow-and-refine (arXiv:2510.04618 3.2) prunes redundancy by comparing bullets via semantic
embeddings, and the cosine at which two rules count as one rule is the only free parameter in
it. It has to be set per benchmark -- rules about financial conventions and rules about Python
style do not sit at the same similarity -- and it has to be set from artifacts that exist
before any test number does, or it becomes a researcher degree of freedom.

This replays a finished probe's curator operations into a fresh memory of the same kind, at
each candidate threshold, applying `refine()` on the same schedule the training loop does. No
LLM, no network, no Docker, and it never writes to the run directory.

    uv run python calibrate_dedup_threshold.py results/mbpp/probes_v2/text results/mbpp/probes_v2/graph
    uv run python calibrate_dedup_threshold.py results/mbpp/probes_v2/text --show-merges 0.86

**The rate alone must not decide it.** A threshold low enough to collapse two genuinely
different rules costs the arm knowledge, and one high enough to prune nothing leaves the text
arm as ACE minus one of its three components -- the exact defect EXPERIMENT_FORMULA.md 10
measures on the seed-0 runs. So the rule is:

  1. absorption must stay under 50% in **both** memory arms (the guard against over-merging);
  2. of the thresholds that satisfy (1), take the highest one that still prunes the benchmark's
     restatement family, and **read every merge it makes** with `--show-merges` to confirm each
     is a restatement rather than two distinct rules.

Two notes on reading the output. The ADD count moves with the threshold: more absorption means
a smaller playbook in front of the curator, so `sanitize_curator_operations` drops fewer of its
ADDs as re-adds of something visible. And the arms legitimately differ -- the graph arm's
curator-declared `refines` withdraws rules before dedup runs, so fewer live candidates remain.
That is the treatment, not an asymmetry in the decision rule, which is shared code.
"""
import argparse
import glob
import json
import os
import re
import sys

from ace.memory import create_memory
from ace.memory.embeddings import EmbeddingBackend
from ace.memory.refine import DEFAULT_DEDUP_THRESHOLD, duplicate_groups
from playbook_utils import extract_json_from_text, parse_playbook_line, sanitize_curator_operations

DEFAULT_THRESHOLDS = (0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94)


def _step_of(path):
    match = re.search(r"_s_(\d+)_", os.path.basename(path))
    return int(match.group(1)) if match else 0


def _resolve(path):
    """Accept either a run directory or the parent --save_path nests it under."""
    if os.path.isdir(os.path.join(path, "detailed_llm_logs")):
        return path
    found = sorted(glob.glob(os.path.join(path, "ace_run_*")))
    return found[0] if found else None


def curator_operations(run_dir):
    """(step, operations) in step order, as Curator.curate parsed them at run time.

    Read from the raw responses rather than curator_operations_diff.jsonl, because the diff log
    records only what was applied and drops the fields (notably `refines`) that decide what the
    graph arm withdraws before dedup ever runs.
    """
    stream = []
    paths = glob.glob(os.path.join(run_dir, "detailed_llm_logs", "curator_*.json"))
    for path in sorted(paths, key=_step_of):
        with open(path, encoding="utf-8") as fh:
            entry = json.load(fh)
        parsed = extract_json_from_text(entry.get("response") or "")
        if isinstance(parsed, dict) and parsed.get("operations"):
            stream.append((_step_of(path), parsed["operations"]))
    return stream


def replay(run_dir, kind, threshold, backend):
    """Rebuild the memory from the recorded operations, refining at `threshold` each step."""
    memory = create_memory(kind, backend=backend) if kind == "graph" else create_memory(kind)
    adds = absorbed = 0

    for step, operations in curator_operations(run_dir):
        kept, _ = sanitize_curator_operations(operations, memory.render_for_curator())
        adds += sum(1 for op in kept if str(op.get("type", "")).upper() == "ADD")
        memory.apply_operations(kept, step=step)
        if threshold is not None:
            absorbed += memory.refine(backend, threshold)["absorbed"]

    return adds, absorbed, memory


def live_rules(memory):
    """(id, content) for everything the memory still shows, in render order."""
    rules = []
    for line in memory.render_for_curator().split("\n"):
        parsed = parse_playbook_line(line)
        if parsed:
            rules.append((parsed["id"], parsed["content"]))
    return rules


def show_merges(run_dir, kind, threshold, backend):
    """Print every pair `threshold` would collapse, so the choice is made by reading them."""
    _adds, _absorbed, memory = replay(run_dir, kind, None, backend)
    rules = live_rules(memory)
    if len(rules) < 2:
        print("  (fewer than two live rules)")
        return

    ids = [rule_id for rule_id, _ in rules]
    vectors = dict(zip(ids, backend.encode([content for _, content in rules])))
    groups = duplicate_groups(rules, backend, threshold, vectors)
    by_id = dict(rules)

    merged = sum(len(v) for v in groups.values())
    print(f"  {len(rules)} live rules, {merged} absorbed into {len(groups)} survivor(s)")
    for survivor, absorbed_ids in groups.items():
        print(f"    keep [{survivor}] {by_id[survivor][:104]}")
        for absorbed_id in absorbed_ids:
            similarity = float(vectors[survivor] @ vectors[absorbed_id])
            print(f"      drop [{absorbed_id}] sim={similarity:.4f} {by_id[absorbed_id][:96]}")

    pairs = sorted(
        (float(vectors[a] @ vectors[b]), a, b)
        for i, a in enumerate(ids) for b in ids[i + 1:]
    )
    near = [p for p in reversed(pairs) if p[0] < threshold][:5]
    if near:
        print(f"  highest-similarity pairs left alone at {threshold}:")
        for similarity, a, b in near:
            print(f"    sim={similarity:.4f}  [{a}] {by_id[a][:70]}")
            print(f"                  [{b}] {by_id[b][:70]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+",
                        help="probe run directories, one per memory arm")
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--show-merges", type=float, metavar="THR",
                        help="print the pairs this threshold collapses, and the near misses")
    args = parser.parse_args()

    runs = []
    for path in args.run_dirs:
        run_dir = _resolve(path)
        if not run_dir:
            print(f"no run with detailed_llm_logs/ under {path}")
            return 2
        with open(os.path.join(run_dir, "run_config.json"), encoding="utf-8") as fh:
            kind = json.load(fh).get("memory_kind", "text")
        if kind == "none":
            print(f"skipping {path}: the control accumulates nothing to de-duplicate")
            continue
        runs.append((path, run_dir, kind))

    if not runs:
        print("nothing to calibrate -- pass at least one text or graph probe")
        return 2

    backend = EmbeddingBackend()

    if args.show_merges is not None:
        for path, run_dir, kind in runs:
            print(f"\n{kind} -- {path} at threshold {args.show_merges}")
            show_merges(run_dir, kind, args.show_merges, backend)
        return 0

    header = f"{'thr':>6}"
    for _path, _run_dir, kind in runs:
        header += f" | {kind + ' ADDs':>11} {'absorbed':>8} {'rate':>6} {'live':>5}"
    print(header)
    print("-" * len(header))

    for threshold in [None] + list(args.thresholds):
        label = "  off " if threshold is None else f"{threshold:6.2f}"
        row = label
        over = False
        for _path, run_dir, kind in runs:
            adds, absorbed, memory = replay(run_dir, kind, threshold, backend)
            rate = absorbed / adds if adds else 0.0
            over = over or rate >= 0.50
            row += (f" | {adds:>11} {absorbed:>8} {rate:>5.1%} "
                    f"{memory.stats().get('total_bullets', 0):>5}")
        print(row + ("   <-- over the 50% limit" if over else ""))

    print(f"\nrepo default is {DEFAULT_DEDUP_THRESHOLD}. Absorption under 50% is necessary, not "
          f"sufficient -- rerun with --show-merges <thr> and read every pair before choosing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
