#!/usr/bin/env python3
"""Replay a finished run's curator operations through the sanitiser and report the effect.

`Curator.curate` strips rendered-bullet prefixes pasted into a rule's text and drops ADDs
that repeat a rule already in the playbook (playbook_utils.sanitize_curator_operations).
This applies it after the fact to runs that were executed before it existed, which is where
the measured rates in docs/background/EXPERIMENT_FORMULA.md come from -- and, run against the MBPP arms, is
the check that the sanitiser is inert on clean input.

    uv run python replay_curator_sanitizer.py results/formula/seed0_exploratory/text
    uv run python replay_curator_sanitizer.py --root results
    uv run python replay_curator_sanitizer.py results/formula/seed0_exploratory/text --show 10

Reads curator_operations_diff.jsonl only. No network, no LLM, no Docker, and it never writes
to the run directory.

Caveat on the duplicate count: the diff log does not snapshot the playbook, so the playbook
each curator call saw is rebuilt from the ADDs that survived earlier calls. The memory is
ADD-only, so that is exact for the text arm. In the graph arm a superseded rule is withdrawn
from render_for_curator() and may legitimately be re-added, so the duplicate count there is
an upper bound.
"""
import argparse
import collections
import json
import os
import sys

from playbook_utils import format_playbook_line, sanitize_curator_operations, strip_rendered_prefix


def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def replay(run_dir, show=0):
    """Return (totals, examples) for one run directory."""
    rows = _read_jsonl(os.path.join(run_dir, "curator_operations_diff.jsonl"))
    adds = [r for r in rows if r.get("operation_type") == "ADD"]

    batches = collections.OrderedDict()
    for row in adds:
        batches.setdefault(row.get("call_id"), []).append(row)

    playbook_lines = []
    totals = collections.Counter()
    examples = []

    for rows_in_call in batches.values():
        ops = [{"type": "ADD", "section": r.get("section", "others"),
                "content": r.get("content", "")} for r in rows_in_call]
        kept, report = sanitize_curator_operations(ops, "\n".join(playbook_lines))
        totals.update(report)

        kept_contents = {op.get("content") for op in kept}
        for op in ops:
            cleaned, removed = strip_rendered_prefix(op["content"])
            if removed:
                examples.append(("stripped", op["content"]))
            if cleaned not in kept_contents:
                examples.append(("dropped" if cleaned else "emptied", op["content"]))

        for op in kept:
            content = " ".join((op.get("content") or "").split())
            playbook_lines.append(
                format_playbook_line(f"add-{len(playbook_lines) + 1:05d}", 0, 0, content))

    totals["calls"] = len(batches)
    return totals, examples[:show] if show else []


def _expand_run_dirs(path):
    if os.path.exists(os.path.join(path, "curator_operations_diff.jsonl")):
        return [path]
    if not os.path.isdir(path):
        return []
    return sorted(
        os.path.join(path, name) for name in os.listdir(path)
        if os.path.exists(os.path.join(path, name, "curator_operations_diff.jsonl"))
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*", help="run directories, or a parent containing them")
    ap.add_argument("--root", help="scan every run under this directory, recursively")
    ap.add_argument("--show", type=int, default=0,
                    help="print up to N affected contents per run")
    args = ap.parse_args()

    resolved = []
    for path in args.run_dirs:
        found = _expand_run_dirs(path)
        if not found:
            print(f"no curator_operations_diff.jsonl under {path}")
            return 2
        resolved.extend(found)

    if args.root:
        for dirpath, _dirnames, filenames in os.walk(args.root):
            if "curator_operations_diff.jsonl" in filenames:
                resolved.append(dirpath)
        resolved = sorted(set(resolved))

    if not resolved:
        print("nothing to replay -- pass a run directory or --root")
        return 2

    print(f"{'run':<52} {'ADDs':>5} {'strip':>7} {'dup':>7} {'empty':>7} {'kept':>7}")
    print("-" * 92)
    for run_dir in resolved:
        totals, examples = replay(run_dir, args.show)
        n = totals["adds"]
        if not n:
            print(f"{os.path.basename(os.path.normpath(run_dir)):<52} {'-':>5}  (no ADDs)")
            continue
        label = os.path.basename(os.path.normpath(run_dir))[:52]
        print(f"{label:<52} {n:>5} "
              f"{totals['stripped']:>3} {totals['stripped'] / n:>6.0%} "
              f"{totals['deduped']:>3} {totals['deduped'] / n:>6.0%} "
              f"{totals['emptied']:>3} {totals['emptied'] / n:>6.0%} "
              f"{totals['kept']:>3} {totals['kept'] / n:>6.0%}")
        for kind, content in examples:
            print(f"      {kind:<9} {content[:110]!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
