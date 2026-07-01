
#!/usr/bin/env python3
import os
import json
import argparse
from datetime import datetime
import logging

from eval.mbpp_code_exec.data_processor import DataProcessor, load_data
from ace.ace import ACE 
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
    parser.add_argument("--generator_model", type=str, default="gemma4:e4b")
    parser.add_argument("--reflector_model", type=str, default="gemma4:e4b")
    parser.add_argument("--curator_model", type=str, default="gemma4:e4b")
    parser.add_argument("--api_provider", type=str, default="groq") 
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_num_rounds", type=int, default=3)
    
    # Set frequencies to 1 for tiny dataset
    parser.add_argument("--curator_frequency", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--online_eval_frequency", type=int, default=50)
    
    parser.add_argument("--playbook_token_budget", type=int, default=80000) # Lowered for test it was default=80000
    parser.add_argument("--json_mode", action="store_true", help="Enable JSON mode for LLM calls")
    parser.add_argument("--no_ground_truth", action="store_true", help="Do not provide ground truth during online learning")
    parser.add_argument("--test_workers", type=int, default=1, help="Number of parallel workers for evaluation")
    parser.add_argument("--use_bulletpoint_analyzer", action="store_true", help="Use bulletpoint analyzer for playbook de-duplication")

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
    
    # Initialize ACE
    
    ace_system = ACE(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=args.max_tokens,
        initial_playbook=initial_playbook
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
        'api_provider': args.api_provider
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


