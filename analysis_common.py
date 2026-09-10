"""Helpers the read-only `analyze_*.py` scripts share.

These scripts recompute figures in the results documents from finished runs. They kept private
copies of the same four things -- how to find a run directory, how to name an arm, how to read a
stored rule line, and how to decide which tag a rule maps to -- and a copy that drifts is how a
results file ends up with a number nothing reproduces. One definition each, here.

Nothing in this module is on the run path. It reads artifacts and never imports `ace`.
"""

import collections
import glob
import json
import os
import re

#: A stored rule as the playbook renders it: `[id] helpful=N harmful=N :: <content>`.
#: The id pattern is deliberately loose. `check_learning_signal._RENDERED_RULE` is the canonical
#: form and matches `[\w-]+`, because the curator invents ids that do not follow the harness's
#: own `<prefix>-<5 digits>` shape and pinning the digit count silently drops those rules.
RULE_LINE = re.compile(r"^\s*\[([\w-]+)\]\s+helpful=\d+\s+harmful=\d+\s*::\s*(.*)$")

_THEN = re.compile(r"\bTHEN\b", re.IGNORECASE)
_TOKEN = re.compile(r"\b([A-Za-z][A-Za-z0-9]*)\b")
#: The drift forms the curator produces when it ignores the requested `THEN <Tag>` shape.
_VERB_LED = re.compile(
    r"\b(?:select|use|choose|prioritize|prefer|map(?:ping)?)\b(?:\s+\w+){0,3}?\s+", re.IGNORECASE
)
#: Where the clause stops naming what the rule *wants*. The curator's commonest shape is
#: `prioritize X over Y`, and Y is the tag it is steering away from: reading it as covered would
#: credit the playbook with advice about a tag it tells the model not to use.
_CONTRAST = re.compile(r"\b(?:over|rather than|instead of|not|unless|except|avoid)\b",
                       re.IGNORECASE)


def mapping_targets(content, vocabulary):
    """Vocabulary tags a rule names as its mapping target.

    **Scans the whole clause after `THEN`, not just the token immediately following it.** The
    earlier version alternated `THEN\\s+(\\w+)` with a verb pattern, so on the very common drift
    form `IF ... THEN prioritize mapping to PaymentsToAcquireBusinessesGross` the first branch
    matched `THEN prioritize`, captured the verb, found it was not a tag, and consumed the text
    the tag was in. `finditer` then resumed past it and the verb branch never fired, because
    `map\\s+to` does not match `mapping to`. Every rule written that way was counted as covering
    nothing, and the coverage tables were built on that.

    A rule with no `THEN` at all is read through the verb forms instead, which keeps the scan off
    tags that appear only in an `IF` condition.
    """
    found = set()
    for match in _THEN.finditer(content):
        found |= _targets_in(content[match.end():], vocabulary)
    if found or _THEN.search(content):
        return found
    for match in _VERB_LED.finditer(content):
        found |= _targets_in(content[match.end():], vocabulary)
    return found


def _targets_in(clause, vocabulary):
    """Vocabulary tags named in one clause, up to the first contrastive marker."""
    stop = _CONTRAST.search(clause)
    if stop:
        clause = clause[:stop.start()]
    return {token.lower() for token in _TOKEN.findall(clause) if token.lower() in vocabulary}


def covered_tags(playbook_path, vocabulary):
    """Vocabulary tags this playbook's rules name as mapping targets."""
    tags = set()
    with open(playbook_path, encoding="utf-8") as handle:
        for line in handle:
            match = RULE_LINE.match(line.rstrip())
            if match:
                tags |= mapping_targets(match.group(2), vocabulary)
    return tags


def graph_snapshot(path):
    """Read one `.graph.json` sidecar: (live ids, {id: text}, live pairs, thresholds).

    A node is live when nothing `refines` it, which is the same rule `GraphMemory.active_nodes`
    applies, and pairs touching a superseded node are dropped because neither consumer can follow
    them. Returns None when the file is not there, so a `text` arm is skipped rather than raising.

    Pairs are `(a, b, cocited, corendered)`. **`corendered` includes weight inherited through
    `refines`**: `graph.py::_inherit_cocitation` copies an ancestor's pair history onto the heir
    at full weight, and the store keeps no marker saying so. Nothing here can separate a pair that
    was really co-rendered from one that inherited the count, and any figure built on `corendered`
    has to say so rather than imply otherwise.
    """
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        blob = json.load(handle)
    graph = blob["graph"]
    superseded = {edge["target"] for edge in graph["edges"] if edge.get("type") == "refines"}
    text = {node["id"]: node.get("content", "")
            for node in graph["nodes"] if node["id"] not in superseded}
    pairs = [(a, b, cocited, corendered)
             for a, b, cocited, corendered, _step in blob.get("cocite_pairs", [])
             if a in text and b in text]
    return set(text), text, pairs, blob.get("thresholds", {})


def cocite_gates(thresholds, min_opportunity=None, min_rate=None):
    """The gate a run was built with, with optional overrides.

    Read from the sidecar rather than defaulted, because the runner default (0.30) is not what
    either completed experiment shipped: `formula` used 0.20 and `finer` 0.10, and a hardcoded
    fallback silently rescores a run under a gate it never had.
    """
    floor = min_opportunity if min_opportunity is not None else thresholds.get(
        "cocite_min_opportunity", 3)
    rate = min_rate if min_rate is not None else thresholds.get("cocite_min_rate", 0.20)
    return floor, rate


def cocite_adjacency(pairs, min_opportunity, min_rate):
    """Rule -> its co-citation neighbours, under the gate as it is evaluated at read time.

    Edges are not stored. `graph.py::_cocite_edge_pairs` recomputes them from the counters on
    every read, so a pair whose co-render denominator grows while its co-citation count does not
    will drop below the rate gate with nothing deleted. This mirrors that predicate exactly.
    """
    graph = collections.defaultdict(set)
    for a, b, cocited, corendered in pairs:
        if corendered >= min_opportunity and cocited / corendered >= min_rate:
            graph[a].add(b)
            graph[b].add(a)
    return graph


def prediction_rows(run_dir):
    """The run's test predictions as parsed rows, or None.

    File precedence comes from `compare_arms.PREDICTION_FILES`, so a rescored run is preferred
    here exactly as it is there.
    """
    from compare_arms import PREDICTION_FILES

    for name in PREDICTION_FILES:
        path = os.path.join(run_dir, name)
        if not os.path.isfile(path):
            continue
        rows = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    return None


def components(edges):
    """Connected components of an undirected edge list, as a list of id sets of size 2 or more.

    Union-find with path halving. Isolated nodes are not returned: a component of one has no
    pair in it, and every consumer here asks a question about pairs.
    """
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in edges:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b
    groups = collections.defaultdict(set)
    for node in parent:
        groups[find(node)].add(node)
    return [group for group in groups.values() if len(group) > 1]


def run_dirs_under(path):
    """The run directories at or under `path`.

    A directory holding `detailed_llm_logs/` is itself a run; anything else is treated as a
    parent and its `ace_run_*` children are returned.
    """
    if os.path.isdir(os.path.join(path, "detailed_llm_logs")):
        return [path]
    return sorted(glob.glob(os.path.join(path, "ace_run_*")))


def collect_runs(paths):
    """Every run directory under the given paths that carries a `run_config.json`."""
    runs = []
    for path in paths:
        for run_dir in run_dirs_under(path):
            if os.path.isfile(os.path.join(run_dir, "run_config.json")):
                runs.append(run_dir)
    return runs


def read_config(run_dir):
    with open(os.path.join(run_dir, "run_config.json"), encoding="utf-8") as handle:
        return json.load(handle)


def arm_of(run_dir, config=None):
    """The arm name this run belongs to.

    `run_config.json` records `memory_kind`, which cannot separate a `graph` arm from a
    `graph + --cocite` one: both are `graph`. The co-citation flag lives in `memory_settings`,
    so it is read here and the two are given different names.
    """
    config = config if config is not None else read_config(run_dir)
    kind = config.get("memory_kind")
    if kind in ("none", "text"):
        return kind
    return "cocite" if (config.get("memory_settings") or {}).get("cocite") else "graph"


def replicate_label(run_dir):
    """A short label that distinguishes runs of one arm, preferring the `seed<N>` directory.

    Run directories are all `ace_run_<timestamp>_<task>_<mode>`, so their basenames share a long
    suffix and truncating them prints the same string for every run.
    """
    parts = os.path.normpath(run_dir).split(os.sep)
    for part in reversed(parts[:-1]):
        if part.startswith("seed"):
            return part
    return os.path.basename(run_dir)
