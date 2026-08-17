#!/usr/bin/env python3
"""Run ACE on the finance tasks (`formula`, `finer`).

Mirrors eval/mbpp_code_exec/run.py flag for flag. The finance entry point was still the
upstream copy: no --memory, no --num_ctx, no --seed, no --think. Anything that differs
between the two evals is a confound if results are ever compared, so the argparse block is
deliberately identical apart from the task name and config path.

Its upstream defaults were test_workers=20, max_num_rounds=3, curator_frequency=1, and all
three were overridden to the MBPP runner's for that parity. test_workers stays at 1 -- the
fairness rules pin it -- but the other two were a silent, undisclosed departure from the
paper (4.2: batch size 1, five reflector rounds, five epochs) and are restored as of
2026-08-06. See EXPERIMENT_FORMULA.md 8. num_epochs remains 1, disclosed: five multiplies
every training-phase call by five.

    uv run python -m eval.finance.run --task_name formula --mode offline \
        --save_path results/formula/preregistered/seed0/text --memory text --dedup --retrieval flat \
        --config_path ./eval/finance/data/task_config_v2.json \
        --generator_model qwen3.5:2b --think_generator off --num_ctx 32768 --seed 0
"""

import os  # noqa: I001  -- import order mirrors eval/mbpp_code_exec/run.py exactly
import json
import argparse
import logging

from eval.finance.data_processor import DataProcessor, load_data
from ace.ace import ACE
from utils import initialize_clients  # noqa: F401  -- kept for parity with the MBPP runner

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run ACE on a finance task')
    parser.add_argument("--task_name", type=str, default="formula",
                        choices=["formula", "finer"])
    parser.add_argument("--mode", type=str, default="offline",
                        choices=['offline', 'online', 'eval_only'],
                        help="Run mode: 'offline' for offline training, "
                             "'online' for online training, 'eval_only' for evaluation only")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save results and playbooks")
    parser.add_argument("--initial_playbook_path", type=str, default=None,
                        help="Optional path to an existing playbook to start with")
    parser.add_argument("--config_path", type=str,
                        default="./eval/finance/data/task_config.json",
                        help="Path to the task configuration JSON file")

    parser.add_argument("--generator_model", type=str, default="gemma4:e2b")
    parser.add_argument("--reflector_model", type=str, default="gemma4:e2b")
    parser.add_argument("--curator_model", type=str, default="gemma4:e2b")
    parser.add_argument("--api_provider", type=str, default="groq")
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_num_rounds", type=int, default=3,
                        help="Reflector refinement rounds on a failed sample. The paper uses 5 "
                             "(4.2) and the upstream finance runner defaulted to 3. It was 1 "
                             "here until 2026-08-06 and undisclosed -- EXPERIMENT_FORMULA.md 8")

    parser.add_argument("--curator_frequency", type=int, default=1,
                        help="Curate every Nth step. 1 is the paper's batch size of 1, "
                             "'constructing a delta context from each sample' (4.2), and was "
                             "the upstream finance default. It was 5 here until 2026-08-06 "
                             "and undisclosed")
    parser.add_argument("--eval_steps", type=int, default=60)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--online_eval_frequency", type=int, default=50)

    parser.add_argument("--playbook_token_budget", type=int, default=80000)
    parser.add_argument("--json_mode", action="store_true", help="Enable JSON mode for LLM calls")
    parser.add_argument("--no_ground_truth", action="store_true",
                        help="Do not provide ground truth during online learning")
    parser.add_argument("--test_workers", type=int, default=1,
                        help="Number of parallel workers for evaluation")
    parser.add_argument("--use_bulletpoint_analyzer", action="store_true",
                        help="Use bulletpoint analyzer for playbook de-duplication")
    parser.add_argument("--skip_initial_test", action="store_true",
                        help="Skip initial test evaluation in offline mode to save time")

    # Memory representation. This is the independent variable of the linear-vs-relational
    # comparison: run the same command three times, changing only --memory.
    parser.add_argument("--memory", type=str, default="text", choices=["none", "text", "graph"],
                        help="Memory representation: 'none' for the pure-LLM control (eval_only "
                             "only), 'text' for the flat playbook baseline, 'graph' for the typed "
                             "relation graph")
    parser.add_argument("--related_threshold", type=float, default=0.60,
                        help="[graph] Cosine similarity above which two rules get a related_to edge")
    parser.add_argument("--keep_superseded", action="store_true",
                        help="[graph] Keep refined-away rules in the prompt. Ablation switch: it "
                             "separates 'the graph annotated the rules' from 'the graph withdrew "
                             "stale ones', which is the capability the flat playbook lacks")
    parser.add_argument("--enable_contradicts", action="store_true",
                        help="[graph] Derive contradicts edges with an NLI cross-encoder. OFF by "
                             "default: calibrated against a real playbook it produced 19 edges of "
                             "which ~3 were genuine, scoring complementary advice at 1.000 "
                             "confidence. Run calibrate_graph_thresholds.py before enabling")
    parser.add_argument("--nli_gate", type=float, default=0.45,
                        help="[graph] Cosine similarity a pair must clear before it is scored for "
                             "contradiction; keeps NLI off all-pairs")
    parser.add_argument("--contradicts_threshold", type=float, default=0.50,
                        help="[graph] NLI contradiction probability above which two rules get a "
                             "contradicts edge")

    parser.add_argument("--cocite", action="store_true",
                        help="[graph] Build association edges from CO-CITATION instead of text "
                             "similarity: connect rules the generator itself named together in "
                             "bullet_ids. Replaces related_to rather than joining it, so a "
                             "graph+cocite arm differs from a graph arm in exactly one thing -- "
                             "which relation the graph holds. This is the relation a flat "
                             "playbook provably cannot represent: related_to is a function of the "
                             "two rules' text, so a list plus an embedding index reproduces it, "
                             "while co-citation exists only in accumulated history")
    parser.add_argument("--cocite_min_opportunity", type=float, default=3,
                        help="[graph] How often two rules must have been rendered together before "
                             "their co-citation rate is trusted at all")
    parser.add_argument("--cocite_min_rate", type=float, default=0.30,
                        help="[graph] P(both cited | both rendered) above which two rules get a "
                             "co_cited edge. A rate, not a count: a raw count would rank whatever "
                             "is rendered most often rather than what is chosen together")
    parser.add_argument("--cocite_half_life", type=float, default=0.0,
                        help="[graph] Steps for co-citation evidence to decay by half. 0 disables "
                             "decay, which is the default: measured on the pre-registered formula "
                             "runs, a half-life of 100 moved held-out citation lift by +0.05 for "
                             "~10%% fewer edges, which does not justify a hyperparameter")

    parser.add_argument("--dedup", action="store_true",
                        help="Run grow-and-refine (arXiv:2510.04618 section 3.2) after each "
                             "curation: prune a rule that repeats an earlier one, comparing by "
                             "semantic embedding. Applies to EVERY memory kind under one shared "
                             "decision rule, so unlike --use_bulletpoint_analyzer it does not "
                             "advantage an arm. Off by default, which reproduces the "
                             "pre-2026-08-06 behaviour exactly")
    parser.add_argument("--dedup_threshold", type=float, default=0.90,
                        help="Cosine at or above which two rules count as the same rule. Must "
                             "match across arms; the fairness audit checks it")

    parser.add_argument("--retrieval", choices=["none", "flat", "graph"], default="none",
                        help="What the generator is shown. 'none' injects the whole memory on "
                             "every call, which is what every run before 2026-08-06 did. "
                             "'flat' renders the rules most similar to the question; 'graph' "
                             "seeds with the same top-k and then walks related_to outward. "
                             "'graph' requires --memory graph; run --memory graph --retrieval "
                             "flat as the ablation that isolates edges-choose-what-is-read "
                             "from edges-are-shown-alongside-it")
    parser.add_argument("--retrieval_budget", type=int, default=2500,
                        help="Token ceiling on the ENTIRE generator-facing rendering, legend "
                             "and relation lines included. Budgeting only the rule text would "
                             "hand the graph arm its annotations for free. Must match across "
                             "arms; the fairness audit checks it")
    parser.add_argument("--retrieval_seeds", type=int, default=5,
                        help="[graph] How many rules the walk starts from. Seeding with "
                             "everything makes the walk irrelevant and reduces the arm to flat "
                             "retrieval. Must match across arms")

    # Sampling controls. Both must be identical across arms for the comparison to hold.
    parser.add_argument("--num_ctx", type=int, default=None,
                        help="Pin the Ollama context window. Leave unset to auto-size from "
                             "prompt length, but pin it (e.g. 32768) when comparing memory "
                             "kinds -- the two render to different lengths, so auto-sizing "
                             "would serve them different context windows")
    parser.add_argument("--seed", type=int, default=0,
                        help="Ollama sampling seed. Without one the server RNG is seeded "
                             "randomly even at temperature=0.0")
    parser.add_argument("--think", choices=["default", "on", "off"], default="default",
                        help="Ollama reasoning switch, applied to every role that does not "
                             "override it below. 'default' leaves the model alone; 'off' for "
                             "a reasoning model (Qwen3, R1) -- every prompt here demands a "
                             "bare JSON object, and a thinking model spends its output budget "
                             "before answering. llm.py warns if the server ignores it")
    for role in ("generator", "reflector", "curator"):
        parser.add_argument(f"--think_{role}", choices=["default", "on", "off"], default=None,
                            # %% because argparse %-formats help strings before printing.
                            help=f"override --think for the {role} alone. The roles have "
                                 f"opposite needs: the generator is ~80%% of all calls, so "
                                 f"thinking there is the expensive one; the reflector and "
                                 f"curator make the judgement calls that are the learning "
                                 f"signal, so thinking there is the one worth paying for")

    return parser.parse_args()


def preprocess_data(task_name, config, mode):
    """Load and preprocess data."""
    processor = DataProcessor(task_name=task_name)

    # For online and eval_only modes, only load test data
    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None

        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            raise ValueError(f"{mode} mode requires test data in config.")

        logger.info(f"{mode} mode: Testing on {len(test_samples)} examples")

    # For offline mode, load train, val, and optionally test data
    else:
        train_samples = load_data(config["train_data"])
        val_samples = load_data(config["val_data"])
        train_samples = processor.process_task_data(train_samples)
        val_samples = processor.process_task_data(val_samples)

        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            test_samples = []

        logger.info(f"Offline mode: Training on {len(train_samples)} examples, "
                    f"validating on {len(val_samples)}, testing on {len(test_samples)}")

    return train_samples, val_samples, test_samples, processor


def load_initial_playbook(path):
    """Load initial playbook if provided."""
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return None


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Load task configuration
    with open(args.config_path, 'r', encoding='utf-8') as f:
        task_config = json.load(f)

    # Preprocess data
    train_samples, val_samples, test_samples, data_processor = \
        preprocess_data(args.task_name, task_config[args.task_name], args.mode)

    # Load initial playbook (or use empty if None provided)
    initial_playbook = load_initial_playbook(args.initial_playbook_path)
    if initial_playbook:
        logger.info(f"Loaded initial playbook from {args.initial_playbook_path}\n")
    else:
        logger.info("Using empty playbook as initial playbook\n")

    # Graph-only settings; ignored by the other arms
    memory_kwargs = {}
    if args.memory == "graph":
        memory_kwargs = {
            "related_threshold": args.related_threshold,
            "nli_gate": args.nli_gate,
            "contradicts_threshold": args.contradicts_threshold,
            "enable_contradicts": args.enable_contradicts,
            "keep_superseded": args.keep_superseded,
            "cocite": args.cocite,
            "cocite_min_opportunity": args.cocite_min_opportunity,
            "cocite_min_rate": args.cocite_min_rate,
            "cocite_half_life": args.cocite_half_life,
        }

    # Per-role reasoning switch: --think sets the floor, --think_<role> overrides it.
    # Resolved here rather than inside ACE so the exact per-role values land in config
    # and therefore in run_config.json, where the fairness audit can compare them.
    think_flags = {role: (getattr(args, f"think_{role}") or args.think)
                   for role in ("generator", "reflector", "curator")}
    think_settings = {role: {"default": None, "on": True, "off": False}[value]
                      for role, value in think_flags.items()}

    ace_system = ACE(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=args.max_tokens,
        initial_playbook=initial_playbook,
        use_bulletpoint_analyzer=args.use_bulletpoint_analyzer,
        memory_kind=args.memory,
        memory_kwargs=memory_kwargs,
        dedup=args.dedup,
        dedup_threshold=args.dedup_threshold,
        retrieval=args.retrieval,
        retrieval_budget=args.retrieval_budget,
        retrieval_seeds=args.retrieval_seeds,
        num_ctx=args.num_ctx,
        seed=args.seed,
        think=think_settings,
        # Selects the finance prompt set. Passed explicitly rather than inferred inside
        # ACE so a run that got the wrong prompts is visible in run_config.json.
        task_name=args.task_name
    )

    config = {
        'num_epochs': args.num_epochs,
        'max_num_rounds': args.max_num_rounds,
        'curator_frequency': args.curator_frequency,
        'eval_steps': args.eval_steps,
        'online_eval_frequency': args.online_eval_frequency,
        'save_steps': args.save_steps,
        'playbook_token_budget': args.playbook_token_budget,
        'task_name': args.task_name,
        'mode': args.mode,
        'json_mode': args.json_mode,
        'no_ground_truth': args.no_ground_truth,
        'save_dir': args.save_path,
        'test_workers': args.test_workers,
        'initial_playbook_path': args.initial_playbook_path,
        'use_bulletpoint_analyzer': args.use_bulletpoint_analyzer,
        'api_provider': args.api_provider,
        'skip_initial_test': args.skip_initial_test,
        'memory': args.memory,
        'related_threshold': args.related_threshold,
        'keep_superseded': args.keep_superseded,
        'cocite': args.cocite,
        'cocite_min_opportunity': args.cocite_min_opportunity,
        'cocite_min_rate': args.cocite_min_rate,
        'cocite_half_life': args.cocite_half_life,
        'enable_contradicts': args.enable_contradicts,
        'nli_gate': args.nli_gate,
        'contradicts_threshold': args.contradicts_threshold,
        'dedup': args.dedup,
        'dedup_threshold': args.dedup_threshold,
        'retrieval': args.retrieval,
        'retrieval_budget': args.retrieval_budget,
        'retrieval_seeds': args.retrieval_seeds,
        'num_ctx': args.num_ctx,
        'seed': args.seed,
        'think': think_flags
    }

    results = ace_system.run(
        mode=args.mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        data_processor=data_processor,
        config=config
    )

    logger.info(f"ACE run completed with results: {results}")


if __name__ == "__main__":
    main()
