#!/usr/bin/env python3
"""Aggregate token cost, latency and accuracy for one or more ACE runs.

Nothing in the harness sums the per-call logs, so "the graph arm cost X% more tokens for Y%
accuracy" -- the headline of the linear-vs-relational comparison -- is not derivable from any
file a run writes. This walks detailed_llm_logs/ and produces it.

It also audits the cross-arm invariants that silently invalidate the comparison when broken:
a num_ctx that varies call-to-call means --num_ctx was left unset and Ollama auto-sized the
context window per prompt length, which alone makes two arms incomparable.

    uv run python aggregate_run_costs.py results/arm_text/ace_run_...
    uv run python aggregate_run_costs.py --root results          # discover every run
    uv run python aggregate_run_costs.py run_a run_b run_c       # cross-arm comparison

Writes cost_summary.json into each run directory and prints a report.
"""
import argparse
import json
import os
from collections import defaultdict

# Calls that reached the model and reported usage vs. calls that died. The failure paths in
# llm.py build a call_info with no usage fields, so their tokens are unrecoverable -- they are
# counted separately rather than folded into the totals as zeros.
TOKEN_KEYS = ("prompt_num_tokens", "response_num_tokens")


def classify_phase(role, call_id):
    """Bucket a call by where in the loop it happened. Mirrors the call_id scheme in ace.py."""
    if role == "curator":
        return "curator"
    if call_id.startswith("test_eval"):
        return "test/val generation"
    if "_post_curate" in call_id:
        return "post-curator generation"
    if "_gen_initial" in call_id:
        return "initial generation"
    if "_post_reflect" in call_id:
        return "regeneration after reflection"
    if "reflect_on_correct" in call_id:
        return "reflection (correct)"
    if "_round_" in call_id:
        return "reflection (incorrect)"
    return f"other ({role})"


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def collect_calls(log_dir):
    """Walk detailed_llm_logs/ once, retaining only the numeric fields."""
    agg = {
        "total_calls": 0,
        "ok_calls": 0,
        "failed_calls": 0,
        "failed_wall_s": 0.0,
        "by_role": defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "time_s": 0.0}),
        "by_phase": defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "time_s": 0.0}),
        "num_ctx_seen": defaultdict(int),
        "seeds_seen": defaultdict(int),
        "models_seen": defaultdict(int),
        "nongreedy_calls": 0,
        "empty_response_incorrect": 0,
        "first_call": None,
        "last_call": None,
    }
    if not os.path.isdir(log_dir):
        return agg

    for name in os.listdir(log_dir):
        if not name.endswith(".json"):
            continue
        rec = _read_json(os.path.join(log_dir, name))
        if rec is None:
            continue

        agg["total_calls"] += 1
        role = rec.get("role", "?")
        phase = classify_phase(role, rec.get("call_id", ""))
        elapsed = rec.get("total_time") or 0.0

        if rec.get("model"):
            agg["models_seen"][rec["model"]] += 1
        if rec.get("training_marked_incorrect_due_to_empty_response") or \
                rec.get("test_marked_incorrect_due_to_empty_response"):
            agg["empty_response_incorrect"] += 1

        has_usage = all(rec.get(k) is not None for k in TOKEN_KEYS)
        if not has_usage:
            # Abandoned/errored call: wall time is known, tokens are not.
            agg["failed_calls"] += 1
            agg["failed_wall_s"] += elapsed
        else:
            agg["ok_calls"] += 1
            tin, tout = rec["prompt_num_tokens"], rec["response_num_tokens"]
            for bucket, key in ((agg["by_role"], role), (agg["by_phase"], phase)):
                bucket[key]["calls"] += 1
                bucket[key]["in"] += tin
                bucket[key]["out"] += tout
                bucket[key]["time_s"] += elapsed
            # num_ctx/seed only exist on the success path; that is where the confound lives.
            agg["num_ctx_seen"][rec.get("num_ctx")] += 1
            agg["seeds_seen"][rec.get("seed")] += 1
            if rec.get("empty_retries"):
                agg["nongreedy_calls"] += 1

        stamp = rec.get("datetime")
        if stamp:
            if agg["first_call"] is None or stamp < agg["first_call"]:
                agg["first_call"] = stamp
            if agg["last_call"] is None or stamp > agg["last_call"]:
                agg["last_call"] = stamp

    agg["by_role"] = dict(agg["by_role"])
    agg["by_phase"] = dict(agg["by_phase"])
    agg["num_ctx_seen"] = {str(k): v for k, v in agg["num_ctx_seen"].items()}
    agg["seeds_seen"] = {str(k): v for k, v in agg["seeds_seen"].items()}
    agg["models_seen"] = dict(agg["models_seen"])
    return agg


def collect_accuracy(run_dir):
    """Pull the accuracy numbers a run reports, from wherever each one lives."""
    out = {}
    for phase, fname in (("initial_test", "initial_test_results.json"),
                         ("final_test", "final_test_results.json"),
                         ("test", "test_test_results.json")):
        data = _read_json(os.path.join(run_dir, fname))
        if data:
            block = data.get("test_results", data)
            out[phase] = {k: block.get(k) for k in ("accuracy", "correct", "total", "no_answer")}

    train = _read_json(os.path.join(run_dir, "train_results.json"))
    if isinstance(train, dict):
        out["best_val_accuracy"] = train.get("best_accuracy")
        checkpoints = train.get("results") or []
        if checkpoints:
            out["checkpoints"] = [
                {
                    "step": c.get("step"),
                    "val_accuracy": (c.get("val_result") or {}).get("accuracy"),
                    "train_pre": (c.get("train_result") or {}).get("pre_train_accuracy"),
                    "train_post": (c.get("train_result") or {}).get("post_train_accuracy"),
                    "memory_tokens": c.get("playbook_num_tokens"),
                    "total_bullets": (c.get("playbook_stats") or {}).get("total_bullets"),
                    "structural_stats": c.get("structural_stats"),
                }
                for c in checkpoints
            ]
            out["final_memory_tokens"] = checkpoints[-1].get("playbook_num_tokens")
    return out


def summarize_run(run_dir):
    cfg_file = _read_json(os.path.join(run_dir, "run_config.json")) or {}
    cfg = cfg_file.get("config", {})
    calls = collect_calls(os.path.join(run_dir, "detailed_llm_logs"))

    total_in = sum(v["in"] for v in calls["by_role"].values())
    total_out = sum(v["out"] for v in calls["by_role"].values())
    llm_seconds = sum(v["time_s"] for v in calls["by_role"].values()) + calls["failed_wall_s"]

    wall_seconds = None
    if calls["first_call"] and calls["last_call"]:
        from datetime import datetime as _dt
        try:
            wall_seconds = (_dt.fromisoformat(calls["last_call"])
                            - _dt.fromisoformat(calls["first_call"])).total_seconds()
        except ValueError:
            wall_seconds = None

    return {
        "run_dir": run_dir,
        "run_name": os.path.basename(os.path.normpath(run_dir)),
        "identity": {
            # "text?" marks a run predating the memory refactor: it has no --memory flag recorded,
            # so it is the flat playbook by construction but was not labelled as such.
            "memory_kind": cfg.get("memory", cfg_file.get("memory_kind", "text?")),
            "mode": cfg_file.get("mode") or cfg.get("mode"),
            "git_commit": cfg_file.get("git_commit") or cfg.get("git_commit"),
            "generator_model": cfg_file.get("generator_model"),
            "reflector_model": cfg_file.get("reflector_model"),
            "curator_model": cfg_file.get("curator_model"),
            "num_ctx_flag": cfg.get("num_ctx"),
            "seed_flag": cfg.get("seed"),
            "think_flag": cfg.get("think"),
            "test_workers": cfg.get("test_workers"),
            "curator_frequency": cfg.get("curator_frequency"),
            "eval_steps": cfg.get("eval_steps"),
            "max_num_rounds": cfg.get("max_num_rounds"),
            "use_bulletpoint_analyzer": cfg.get("use_bulletpoint_analyzer"),
            "dedup": cfg_file.get("dedup", cfg.get("dedup")),
            "dedup_threshold": cfg_file.get("dedup_threshold", cfg.get("dedup_threshold")),
            "retrieval": cfg_file.get("retrieval", cfg.get("retrieval")),
            "retrieval_budget": cfg_file.get("retrieval_budget", cfg.get("retrieval_budget")),
            "retrieval_seeds": cfg_file.get("retrieval_seeds", cfg.get("retrieval_seeds")),
            "memory_settings": cfg_file.get("memory_settings"),
        },
        "calls": {
            "total": calls["total_calls"],
            "with_usage": calls["ok_calls"],
            "without_usage": calls["failed_calls"],
            "by_role": calls["by_role"],
            "by_phase": calls["by_phase"],
        },
        "tokens": {
            "prompt": total_in,
            "completion": total_out,
            "total": total_in + total_out,
            "unaccounted_calls": calls["failed_calls"],
        },
        "time": {
            "llm_seconds": llm_seconds,
            "llm_hours": llm_seconds / 3600.0,
            "wall_seconds": wall_seconds,
            "wall_hours": wall_seconds / 3600.0 if wall_seconds else None,
            "failed_call_seconds": calls["failed_wall_s"],
            "first_call": calls["first_call"],
            "last_call": calls["last_call"],
        },
        "integrity": {
            "num_ctx_observed": calls["num_ctx_seen"],
            "seeds_observed": calls["seeds_seen"],
            "models_observed": calls["models_seen"],
            "nongreedy_calls": calls["nongreedy_calls"],
            "empty_response_marked_incorrect": calls["empty_response_incorrect"],
        },
        "accuracy": collect_accuracy(run_dir),
    }


def audit(summaries):
    """Cross-arm invariants. Each violation makes the comparison unsound, so they are loud."""
    problems, notes = [], []

    for s in summaries:
        ctxs = s["integrity"]["num_ctx_observed"]
        if set(ctxs) == {"None"}:
            # Not the same as "consistent": the field is simply absent, so this run carries no
            # evidence either way and must not be read as having passed the check.
            problems.append(
                f"{s['run_name']}: no num_ctx recorded on any call -- this run predates the "
                f"--num_ctx flag, so the context window it was served is unknown and cannot be "
                f"matched against a new run."
            )
        elif len(ctxs) > 1:
            problems.append(
                f"{s['run_name']}: {len(ctxs)} distinct num_ctx values in one run "
                f"({', '.join(sorted(ctxs))}) -- --num_ctx was unset, so Ollama auto-sized the "
                f"context window per prompt. The arms were decoded under different windows."
            )
        if s["integrity"]["nongreedy_calls"]:
            notes.append(f"{s['run_name']}: {s['integrity']['nongreedy_calls']} call(s) escalated "
                         f"temperature above 0.0 after empty responses (no longer greedy)")
        if s["calls"]["without_usage"]:
            notes.append(f"{s['run_name']}: {s['calls']['without_usage']} call(s) recorded no token "
                         f"usage ({s['time']['failed_call_seconds']:.0f}s of wall time); their "
                         f"tokens are absent from the totals")
        if s["identity"]["use_bulletpoint_analyzer"]:
            problems.append(f"{s['run_name']}: use_bulletpoint_analyzer was ON -- it merges bullets, "
                            f"a removal the graph arm cannot mirror")

    if len(summaries) > 1:
        def spread(getter, label):
            values = {}
            for s in summaries:
                values.setdefault(repr(getter(s)), []).append(s["run_name"])
            if len(values) > 1:
                detail = "; ".join(f"{v} -> {', '.join(runs)}" for v, runs in values.items())
                problems.append(f"{label} differs across arms: {detail}")

        spread(lambda s: s["identity"]["seed_flag"], "seed")
        spread(lambda s: s["identity"]["num_ctx_flag"], "num_ctx")
        spread(lambda s: s["identity"]["think_flag"], "think")
        spread(lambda s: s["identity"]["generator_model"], "generator_model")
        spread(lambda s: s["identity"]["reflector_model"], "reflector_model")
        spread(lambda s: s["identity"]["curator_model"], "curator_model")
        spread(lambda s: s["identity"]["test_workers"], "test_workers")
        spread(lambda s: s["identity"]["curator_frequency"], "curator_frequency")
        spread(lambda s: s["identity"]["eval_steps"], "eval_steps")
        spread(lambda s: s["identity"]["max_num_rounds"], "max_num_rounds")
        # Grow-and-refine changes what the memory contains, so an arm that ran it
        # against one that did not is comparing two different accumulation rules.
        spread(lambda s: s["identity"]["dedup"], "dedup")
        spread(lambda s: s["identity"]["dedup_threshold"], "dedup_threshold")
        # The retrieval *mode* is part of the treatment and is meant to differ, but the
        # budget it spends is not: an arm given more tokens to retrieve into is simply given
        # a bigger memory. Mixing retrieval with no-retrieval is not a comparison at all.
        spread(lambda s: s["identity"]["retrieval_budget"], "retrieval_budget")
        spread(lambda s: s["identity"]["retrieval_seeds"], "retrieval_seeds")
        modes = {s["identity"]["retrieval"] for s in summaries}
        if len(modes) > 1 and ("none" in modes or None in modes):
            problems.append(
                f"some arms retrieve and some inject the whole memory: {sorted(map(str, modes))} "
                f"-- that is a difference in how much memory the generator sees, not in how it "
                f"is structured"
            )

    return problems, notes


def print_run(s):
    ident = s["identity"]
    print("=" * 92)
    print(f"{s['run_name']}   memory={ident['memory_kind']}   mode={ident['mode']}")
    if ident["git_commit"]:
        print(f"  commit {ident['git_commit']}")
    print(f"  models  gen={ident['generator_model']} refl={ident['reflector_model']} "
          f"cur={ident['curator_model']}")
    # think is per-role, so collapse it to one token when the roles agree and spell it out
    # when they do not -- "off" and "gen=off/refl=default" are very different experiments.
    think = ident["think_flag"]
    if isinstance(think, dict):
        think = ("/".join(f"{role[:4]}={value}" for role, value in think.items())
                 if len(set(think.values())) > 1 else next(iter(think.values()), None))
    print(f"  flags   num_ctx={ident['num_ctx_flag']} seed={ident['seed_flag']} "
          f"think={think} workers={ident['test_workers']} "
          f"analyzer={ident['use_bulletpoint_analyzer']} dedup={ident['dedup']}"
          f"{'@' + str(ident['dedup_threshold']) if ident['dedup'] else ''}")
    if ident["retrieval"] and ident["retrieval"] != "none":
        print(f"  retrieval {ident['retrieval']} budget={ident['retrieval_budget']} "
              f"seeds={ident['retrieval_seeds']}")
    if ident["memory_settings"]:
        print(f"  memory  {ident['memory_settings']}")

    print(f"\n  calls {s['calls']['total']} "
          f"({s['calls']['with_usage']} with usage, {s['calls']['without_usage']} without)")
    print(f"  {'role':<12}{'calls':>7}{'in tok':>12}{'out tok':>12}{'tok/call':>10}"
          f"{'time s':>10}{'s/call':>9}")
    for role, v in sorted(s["calls"]["by_role"].items()):
        per = (v["in"] + v["out"]) / v["calls"] if v["calls"] else 0
        print(f"  {role:<12}{v['calls']:>7}{v['in']:>12,}{v['out']:>12,}{per:>10.0f}"
              f"{v['time_s']:>10.0f}{v['time_s'] / max(v['calls'], 1):>9.1f}")
    tk = s["tokens"]
    print(f"  {'TOTAL':<12}{s['calls']['with_usage']:>7}{tk['prompt']:>12,}{tk['completion']:>12,}"
          f"{'':>10}{s['time']['llm_seconds']:>10.0f}")

    print("\n  by phase:")
    for phase, v in sorted(s["calls"]["by_phase"].items(), key=lambda kv: -kv[1]["calls"]):
        print(f"    {phase:<32}{v['calls']:>6} calls{(v['in'] + v['out']):>12,} tok"
              f"{v['time_s'] / 3600:>8.2f} h")

    t = s["time"]
    line = f"\n  LLM time {t['llm_hours']:.2f} h"
    if t["wall_hours"]:
        ratio = t["llm_seconds"] / t["wall_seconds"] if t["wall_seconds"] else 0
        line += f"   wall {t['wall_hours']:.2f} h   parallelism {ratio:.2f}x"
    print(line)

    acc = s["accuracy"]
    if acc:
        print("\n  accuracy:")
        for key in ("initial_test", "final_test", "test"):
            if key in acc and acc[key].get("accuracy") is not None:
                a = acc[key]
                print(f"    {key:<14}{a['accuracy']:.4f}  ({a['correct']}/{a['total']}, "
                      f"no_answer={a['no_answer']})")
        if acc.get("best_val_accuracy") is not None:
            print(f"    {'best val':<14}{acc['best_val_accuracy']:.4f}")
        for c in acc.get("checkpoints", []):
            struct = c.get("structural_stats")
            extra = ""
            if struct:
                extra = (f"  nodes={struct.get('active_node_count')}/{struct.get('node_count')} "
                         f"refines={struct.get('refines_edges')} "
                         f"related={struct.get('related_to_edges')}")
            print(f"    step {c['step']:<9}val={c['val_accuracy']:.4f} "
                  f"train {c['train_pre']:.3f}->{c['train_post']:.3f} "
                  f"mem={c['memory_tokens']} tok, {c['total_bullets']} rules{extra}")
    print()


def short_name(run_name):
    """ace_run_20260731_225716_mbpp_code_exec_offline -> 20260731_225716"""
    trimmed = run_name[len("ace_run_"):] if run_name.startswith("ace_run_") else run_name
    for marker in ("_mbpp", "_finance", "_mind2web"):
        if marker in trimmed:
            trimmed = trimmed.split(marker)[0]
            break
    return trimmed[:17]


def print_comparison(summaries):
    print("=" * 92)
    print("CROSS-ARM COMPARISON")
    print("=" * 92)
    print(f"{'run':<18}{'memory':<8}{'calls':>7}{'tokens':>12}{'LLM h':>8}{'final acc':>11}"
          f"{'mem tok':>9}{'vs base tok':>13}")

    base = next((s for s in summaries if s["identity"]["memory_kind"] == "text"), None)
    base_tokens = base["tokens"]["total"] if base else None

    for s in summaries:
        acc = s["accuracy"]
        final = acc.get("final_test") or acc.get("test") or {}
        acc_str = f"{final['accuracy']:.4f}" if final.get("accuracy") is not None else "-"
        mem_tok = acc.get("final_memory_tokens")
        delta = "-"
        if base_tokens and s is not base:
            delta = f"{100.0 * (s['tokens']['total'] - base_tokens) / base_tokens:+.1f}%"
        elif s is base:
            delta = "baseline"
        print(f"{short_name(s['run_name']):<18}{str(s['identity']['memory_kind']):<8}"
              f"{s['calls']['total']:>7}{s['tokens']['total']:>12,}{s['time']['llm_hours']:>8.2f}"
              f"{acc_str:>11}{str(mem_tok if mem_tok is not None else '-'):>9}{delta:>13}")
    print()


def discover(root):
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        if "detailed_llm_logs" in dirnames:
            found.append(dirpath)
            dirnames[:] = []
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="*", help="Run directories (each containing detailed_llm_logs/)")
    parser.add_argument("--root", help="Discover every run directory beneath this path")
    parser.add_argument("--no_write", action="store_true",
                        help="Print only; do not write cost_summary.json into each run directory")
    args = parser.parse_args()

    # A positional argument may be a run directory or a parent holding several -- runs are
    # saved under results/<arm>/ace_run_<timestamp>_.../, so `results/arm_text` is the
    # natural thing to type. Without this it was accepted as a run, found no logs, and
    # wrote an all-zero cost_summary.json that looked like a completed analysis.
    run_dirs = []
    for path in args.run_dirs:
        run_dirs.extend(discover(path) or [path])
    if args.root:
        run_dirs.extend(discover(args.root))
    if not run_dirs:
        parser.error("give at least one run directory, or --root to discover them")
    run_dirs = list(dict.fromkeys(run_dirs))

    summaries = []
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            print(f"skipping {run_dir}: not a directory")
            continue
        s = summarize_run(run_dir)
        summaries.append(s)
        if not args.no_write:
            with open(os.path.join(run_dir, "cost_summary.json"), "w", encoding="utf-8") as f:
                json.dump(s, f, indent=2)

    for s in summaries:
        print_run(s)

    if len(summaries) > 1:
        print_comparison(summaries)

    problems, notes = audit(summaries)
    if problems:
        print("FAIRNESS PROBLEMS -- these invalidate a comparison:")
        for p in problems:
            print(f"  [X] {p}")
        print()
    if notes:
        print("notes:")
        for n in notes:
            print(f"  [!] {n}")
        print()
    if not problems:
        print("no fairness violations detected across the runs supplied.")

    if not args.no_write:
        print(f"wrote cost_summary.json into {len(summaries)} run director"
              f"{'y' if len(summaries) == 1 else 'ies'}")


if __name__ == "__main__":
    main()
