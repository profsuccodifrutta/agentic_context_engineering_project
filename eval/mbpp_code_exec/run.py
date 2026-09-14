
#!/usr/bin/env python3
import os
import json
import argparse
from datetime import datetime
import logging

from eval.mbpp_code_exec.data_processor import DataProcessor, load_data
from ace.ace import ACE
from ace.memory import build_memory_kwargs
# Imported rather than re-typed: a default that drifts between the runner and the memory is a
# silent fairness bug, since run_config.json would record one value and GraphMemory use another.
from ace.memory.graph import (
    DEFAULT_COCITE_BETA,
    DEFAULT_COCITE_HALF_LIFE,
    DEFAULT_COCITE_K,
    DEFAULT_COCITE_MIN_OPPORTUNITY,
    DEFAULT_COCITE_MIN_RATE,
    DEFAULT_COCITE_PREDICATE,
    DEFAULT_COCITE_WALK,
) 
from utils import initialize_clients

logger = logging.getLogger(__name__)
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run ACE on MBPP Code Execution task')
    parser.add_argument("--task_name", type=str, default="mbpp_code_exec") 
    parser.add_argument("--mode", type=str, default="offline",
                        choices=['offline', 'online', 'eval_only'],
                        help="Run mode: 'offline' for offline training, "
                             "'online' for online training, 'eval_only' for evaluation only")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save results and playbooks")
    parser.add_argument("--initial_playbook_path", type=str, default=None,
                        help="Optional path to an existing playbook to start with")
    parser.add_argument("--config_path", type=str, 
                        default="./eval/mbpp_code_exec/data/task_config.json", 
                        help="Path to the task configuration JSON file")
    
    # DEFAULTS FOR GROQ 
    parser.add_argument("--generator_model", type=str, default="gemma4:e2b")
    parser.add_argument("--reflector_model", type=str, default="gemma4:e2b")
    parser.add_argument("--curator_model", type=str, default="gemma4:e2b")
    parser.add_argument("--api_provider", type=str, default="groq") 
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_num_rounds", type=int, default=3,
                        help="Reflector refinement rounds on a failed sample. The paper uses 5 "
                             "(4.2) and the upstream finance runner defaulted to 3. It was 1 "
                             "here until 2026-08-06 and undisclosed -- docs/background/EXPERIMENT_FORMULA.md 8")
    
    
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
    parser.add_argument("--no_ground_truth", action="store_true", help="Do not provide ground truth during online learning")
    parser.add_argument("--test_workers", type=int, default=1, help="Number of parallel workers for evaluation")
    parser.add_argument("--use_bulletpoint_analyzer", action="store_true", help="Use bulletpoint analyzer for playbook de-duplication")
    parser.add_argument("--skip_initial_test", action="store_true", help="Skip initial test evaluation in offline mode to save time")

    # Memory representation. This is the independent variable of the linear-vs-relational
    # comparison: run the same command twice, changing only --memory.
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
    parser.add_argument("--cocite_min_opportunity", type=float,
                        default=DEFAULT_COCITE_MIN_OPPORTUNITY,
                        help="[graph] How often two rules must have been rendered together before "
                             "their co-citation rate is trusted at all")
    parser.add_argument("--cocite_min_rate", type=float, default=DEFAULT_COCITE_MIN_RATE,
                        help="[graph] P(both cited | both rendered) above which two rules get a "
                             "co_cited edge. A rate, not a count: a raw count would rank whatever "
                             "is rendered most often rather than what is chosen together")
    parser.add_argument("--cocite_half_life", type=float, default=DEFAULT_COCITE_HALF_LIFE,
                        help="[graph] Steps for co-citation evidence to decay by half. 0 disables "
                             "decay, which is the default: measured on the pre-registered formula "
                             "runs, a half-life of 100 moved held-out citation lift by +0.05 for "
                             "~10%% fewer edges, which does not justify a hyperparameter")

    parser.add_argument("--cocite_predicate", choices=("gate", "rank"),
                        default=DEFAULT_COCITE_PREDICATE,
                        help="[graph] How a co-citation edge is decided. 'gate' is the shipped "
                             "predicate: both thresholds above must be met. 'rank' removes the "
                             "thresholds entirely -- every rule keeps its top --cocite_k partners "
                             "by reward/(corendered+beta), unioned symmetrically. Measured on the "
                             "finer arms 'gate' reaches 30 of 224 live rules and 'rank' reaches "
                             "116, and a rank cannot silently drop an edge as the memory grows, "
                             "because it only moves when another pair genuinely overtakes it")
    parser.add_argument("--cocite_beta", type=float, default=DEFAULT_COCITE_BETA,
                        help="[graph] Denominator offset in reward/(corendered+beta), which "
                             "shrinks a rate toward zero in proportion to how little has been "
                             "observed. NOT a numerator prior: an alpha would credit pairs "
                             "nothing has been seen about. 10 is the smallest value that ranks a "
                             "25/300 pair above a 1/3 one")
    parser.add_argument("--cocite_k", type=int, default=DEFAULT_COCITE_K,
                        help="[graph] Partners each rule nominates under --cocite_predicate rank. "
                             "Bounding what a rule nominates is what stops it becoming a hub; "
                             "being nominated by many rules is popularity the walk may see. "
                             "Measured, the connected set is identical at k=2, 3 and 5, so k buys "
                             "degree and tokens and no reach")
    parser.add_argument("--cocite_outcome_weighted", action="store_true",
                        help="[graph] Weight a co-citation by what the generation was worth "
                             "instead of counting that it happened. Fractional where the task "
                             "reports partial credit (finer: 3 of 4 tags banks 0.75), binary "
                             "where it does not. A pair cited 50 times on 45 wrong answers stops "
                             "outranking one that works")
    parser.add_argument("--cocite_walk", choices=("breadth", "best_first"),
                        default=DEFAULT_COCITE_WALK,
                        help="[graph] How the walk traverses the edges. 'breadth' renders every "
                             "hop-1 neighbour before any hop-2 one, so a near-zero edge is read "
                             "ahead of a strong edge one hop further out. 'best_first' orders a "
                             "priority queue on cumulative raw path strength (relevance times "
                             "the product of the edge weights), which fixes that and makes depth "
                             "cost what the evidence says it should. Independent of "
                             "--cocite_predicate, so either walk can be run over either edge set")
    parser.add_argument("--subquery_seeding", action="store_true",
                        help="Give every sub-question its own retrieval slots instead of one "
                             "global ranking: ceil(--retrieval_seeds / sub-questions) seeds each "
                             "for the walk, and round-robin across their rankings for the budget. "
                             "MUST be set in both arms or neither -- flat retrieval ranks "
                             "globally too, so setting it on one side makes slot allocation a "
                             "second difference between the arms. On a task that supplies one "
                             "query (formula, MBPP) it changes nothing. Measured on finer, global "
                             "seeding represents all four sub-questions in 1.6%% of generations")

    parser.add_argument("--no_render_relations", action="store_true",
                        help="[graph] Stop printing relation lines to the generator. The walk "
                             "still follows the edges; the model is simply not shown them. Both "
                             "arms then render the same object -- a flat list of rules, same "
                             "format, same budget -- so the only difference left between them is "
                             "which rules were selected. Measured, the lines cost 988 tokens of a "
                             "2500 budget at the v2 edge density")

    parser.add_argument("--no_curator_refines", action="store_true",
                        help="Ignore the curator's refines field in the graph arm. It is unary "
                             "and reproducible by a flat list, so it is not the graph capability "
                             "under test; disabling it leaves the arms differing only in whether "
                             "the memory can represent pairwise facts. Does not affect --dedup.")
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
    else: # This is the offline mode block
        train_samples = load_data(config["train_data"])
        val_samples = load_data(config["val_data"])
        train_samples = processor.process_task_data(train_samples)
        val_samples = processor.process_task_data(val_samples)
        
        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            test_samples = []  # Default if no test data configured
        
        logger.info(f"Offline mode: Training on {len(train_samples)} examples, "
                    f"validating on {len(val_samples)}, testing on {len(test_samples)}")
    
    return train_samples, val_samples, test_samples, processor




def load_initial_playbook(path):
    """Load initial playbook if provided."""
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return None


def main():
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Load task configuration
    with open(args.config_path, 'r') as f:
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
    
    memory_kwargs = build_memory_kwargs(args)

    # Per-role reasoning switch: --think sets the floor, --think_<role> overrides it.
    # Resolved here rather than inside ACE so the exact per-role values land in config
    # and therefore in run_config.json, where the fairness audit can compare them.
    think_flags = {role: (getattr(args, f"think_{role}") or args.think)
                   for role in ("generator", "reflector", "curator")}
    think_settings = {role: {"default": None, "on": True, "off": False}[value]
                      for role, value in think_flags.items()}

    # Initialize ACE
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
        think=think_settings
    )
    
    # Configure ACE run parameters
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
        # Pinned across arms by both fairness audits: each changes which rules the generator
        # reads. `cocite_predicate` is the exception in spirit -- like `memory` and
        # `retrieval` it names the treatment -- but it is audited anyway, because pooling a
        # `gate` arm with a `rank` arm as replicates of one `graph` arm is exactly the
        # mistake `memory_kind` alone cannot catch.
        'cocite_predicate': args.cocite_predicate,
        'cocite_beta': args.cocite_beta,
        'cocite_k': args.cocite_k,
        'cocite_outcome_weighted': args.cocite_outcome_weighted,
        'render_relations': not args.no_render_relations,
        'cocite_walk': args.cocite_walk,
        # Not graph-only: it reaches the text arm too, so it is audited on every arm that
        # retrieves. The control is exempt, because `--memory none` refuses retrieval.
        'subquery_seeding': args.subquery_seeding,
        # Recorded here as well as in memory_settings, under the flag's own name. The audit in
        # compare_arms.py looks for `no_curator_refines` and nothing wrote it, so it fell back
        # to the default and read False for every run whether the flag was passed or not --
        # which meant it could never catch two graph arms disagreeing about it.
        'no_curator_refines': args.no_curator_refines,
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

