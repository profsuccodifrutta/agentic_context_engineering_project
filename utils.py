#!/usr/bin/env python3
import os
import re
import json
import openai
import tiktoken
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

def initialize_clients(api_provider):
    """Initialize separate clients for generator, reflector, and curator"""
    
    api_key = ""
    base_url = "https://ollama.caduceus.lol/v1"
    headers = {"bypass": "true"}

    generator_client = openai.OpenAI(api_key=api_key, base_url=base_url, default_headers=headers)
    reflector_client = openai.OpenAI(api_key=api_key, base_url=base_url, default_headers=headers)
    curator_client = openai.OpenAI(api_key=api_key, base_url=base_url, default_headers=headers)

    return generator_client, reflector_client, curator_client

def get_section_slug(section_name):
    """Convert section name to slug format (3-5 chars)"""
    # Common section mappings - updated to match original sections
    slug_map = {
        "financial_strategies_and_insights": "fin",
        "formulas_and_calculations": "calc",
        "code_snippets_and_templates": "code",
        "common_mistakes_to_avoid": "err",
        "problem_solving_heuristics": "prob",
        "context_clues_and_indicators": "ctx",
        "others": "misc",
        "meta_strategies": "meta"
    }
    
    # Clean and convert to snake_case
    clean_name = section_name.lower().strip().replace(" ", "_").replace("&", "and")
    
    if clean_name in slug_map:
        return slug_map[clean_name]
    
    # Generate slug from first letters
    words = clean_name.split("_")
    if len(words) == 1:
        return words[0][:4]
    else:
        return "".join(w[0] for w in words[:5])

def extract_boxed_content(text):
    """Helper function to extract content from \\boxed{} format"""
    pattern = r'\\boxed\{'
    match = re.search(pattern, text)
    if not match:
        return None
    
    start = match.end() - 1  # Position of opening brace
    brace_count = 0
    i = start
    
    while i < len(text):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start + 1:i]  # Content between braces
        i += 1
    return None

def extract_answer(response):
    """Extract final answer from model response, handling bad JSON and markdown."""
    try:
        clean_response = response.strip()
        
        # Strip external markdown if the LLM wrapped the whole JSON in it
        if "```json" in clean_response:
            clean_response = clean_response.split("```json")[1].split("```")[0].strip()
        
        # Isolate the JSON object (first { to last })
        start_idx = clean_response.find('{')
        end_idx = clean_response.rfind('}')
        if start_idx != -1 and end_idx != -1:
            clean_response = clean_response[start_idx:end_idx+1]

        # Use strict=False to safely handle literal newlines without brittle regexes
        parsed = json.loads(clean_response, strict=False)
        answer = str(parsed.get("final_answer", "No final answer found"))
        
        # Clean up inner markdown (if LLM put ```python inside the JSON string)
        if answer != "No final answer found":
            answer = answer.replace("```python\n", "")
            answer = answer.replace("```python", "")
            answer = answer.replace("```", "")
            
            # Re-convert escaped newlines back to literal newlines for Python execution
            answer = answer.replace("\\n", "\n")
            
            # --- SAFETY FIXES ---
            answer = answer.replace('\xa0', ' ')
            answer = answer.replace('\t', '    ')
            answer = answer.strip()
            
        return answer
        
    except Exception as e:
        #  FALLBACK REGEX (If JSON parsing completely fails) 
        
        # FIXED: The regex now correctly skips escaped quotes (\") inside the code!
        matches = re.findall(r'"final_answer"\s*:\s*"((?:[^"\\]|\\.)*)"', response, re.DOTALL)
        if matches:
            answer = matches[-1]
            answer = answer.replace('\\"', '"') # Restore escaped quotes
            answer = answer.replace("```python", "").replace("```", "").replace("\\n", "\n")
            answer = answer.replace('\xa0', ' ').replace('\t', '    ').strip()
            return answer
        
        # Try finding single quotes: 'final_answer': '...'
        matches = re.findall(r"'final_answer'\s*:\s*'((?:[^'\\]|\\.)*)'", response, re.DOTALL)
        if matches:
            answer = matches[-1]
            answer = answer.replace("\\'", "'") # Restore escaped quotes
            answer = answer.replace("```python", "").replace("```", "").replace("\\n", "\n")
            answer = answer.replace('\xa0', ' ').replace('\t', '    ').strip()
            return answer

        # If everything fails, print the raw output to see how the LLM broke it
        print(f"\n[Extraction Error] Failed to parse code. Raw LLM response:\n{response[:250]}...\n")
        return "raise ValueError('No valid code extracted from LLM')"
    
enc = tiktoken.get_encoding("cl100k_base")
def count_tokens(prompt: str) -> int:
    return len(enc.encode(prompt))


def evaluate_single_test_sample(args_tuple, data_processor) -> Tuple[Dict, str]:
    """
    Evaluate a single test sample - task-agnostic implementation.
    
    Args:
        args_tuple: Tuple of (index, task_dict, generator, playbook, max_tokens, log_dir, use_json_mode)
        data_processor: DataProcessor instance with answer_is_correct method
    """
    (i, task_dict, generator, playbook, max_tokens, log_dir, use_json_mode) = args_tuple
    try:
        context = task_dict["context"]
        question = task_dict["question"]
        target = task_dict["target"]

        gen_response, bullet_ids, call_info = generator.generate(
            question=question,
            playbook=playbook,
            context=context,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"test_eval_{i}",
            log_dir=log_dir
        )

        final_answer = extract_answer(gen_response)
        
        
        is_correct = data_processor.answer_is_correct(final_answer, target, task_dict["original_mbpp_item"])

        return {
            "index": i,
            "final_answer": final_answer,
            "target": target,
            "is_correct": is_correct,
            "success": True # Indicate that the generation itself was successful, even if answer is wrong
        }, None

    except Exception as e:
        return {
            "index": i,
            "final_answer": "ERROR_DURING_GENERATION_OR_EXECUTION",
            "target": task_dict.get("target", "UNKNOWN"),
            "is_correct": False,
            "success": False # Indicate that the generation/evaluation process failed for this sample
        }, f"Error evaluating sample {i}: {type(e).__name__}: {str(e)}"


def evaluate_test_set(data_processor, generator, playbook, test_samples,
                      max_tokens=4096, log_dir=None, max_workers=20, 
                      use_json_mode=False) -> Tuple[Dict, Dict]:
    """
    Parallel evaluation of test set - task-agnostic implementation.
    
    Args:
        data_processor: DataProcessor instance with answer_is_correct and evaluate_accuracy methods
        generator: Generator instance
        playbook: Current playbook string
        test_samples: List of test samples
        max_tokens: Max tokens for generation
        log_dir: Directory for logs
        max_workers: Number of parallel workers
        use_json_mode: Whether to use JSON mode
        
    Returns:
        Tuple of (results_dict, error_logs_dict)
    """
    print(f"\n{'='*40}")
    print(f"EVALUATING TEST SET - {len(test_samples)} samples, {max_workers} workers")
    print(f"{'='*40}")

    args_list = [
        (i, sample, generator, playbook, max_tokens, log_dir, use_json_mode)
        for i, sample in enumerate(test_samples)
    ]

    
    all_sample_results = []
    error_messages = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_args = {
            executor.submit(evaluate_single_test_sample, args, data_processor): args 
            for args in args_list
        }

        for i, future in enumerate(as_completed(future_to_args), 1):
            sample_result, error_msg = future.result()
            
            if sample_result: 
                all_sample_results.append(sample_result)
            if error_msg:
                error_messages.append(error_msg)

            if i % 50 == 0:
                processed_count = len(all_sample_results)
                current_correct = sum(1 for r in all_sample_results if r["is_correct"])
                current_accuracy = current_correct / processed_count if processed_count > 0 else 0
                print(f"Progress: {i}/{len(args_list)}, Processed: {processed_count}, Accuracy: {current_accuracy:.3f}")
    
    if all_sample_results:
        predictions = [r["final_answer"] for r in all_sample_results]
        ground_truths = [r["target"] for r in all_sample_results]
        
        original_items_for_accuracy = [test_samples[r["index"]]["original_mbpp_item"] for r in all_sample_results]

        overall_accuracy = data_processor.evaluate_accuracy(predictions, ground_truths, original_items_for_accuracy)
        
        total_correct = sum(1 for r in all_sample_results if r["is_correct"])
        total_total = len(all_sample_results)
        total_no_answer = sum(1 for r in all_sample_results if r["final_answer"] == "No final answer found")
        
        final_results = {
            "accuracy": overall_accuracy,
            "correct": total_correct,
            "total": total_total,
            "no_answer": total_no_answer
        }
        
        error_logs_list = []
        for r in all_sample_results:
            if not r["is_correct"]: # and r["success"] is True, meaning generation was okay but answer was wrong
                error_logs_list.append({
                    "index": r["index"],
                    "prediction": r["final_answer"],
                    "ground_truth": r["target"]
                })
            elif not r["success"]: # Generation/execution failed
                 error_logs_list.append({
                    "index": r["index"],
                    "prediction": r["final_answer"], # This would be "ERROR_DURING_GENERATION_OR_EXECUTION"
                    "ground_truth": r["target"],
                    "error_type": "GENERATION_OR_EXECUTION_FAILURE"
                })

        if error_messages:
            error_logs_list.append({"general_errors": error_messages})


        error_logs_dict = {
            "accuracy": overall_accuracy, 
            "errors": error_logs_list
        }
        
        print(f"\n📊 Final Accuracy: {overall_accuracy:.3f} ({total_correct}/{total_total})")
    else:
        final_results = {"accuracy": 0.0, "correct": 0, "total": 0, "no_answer": 0}
        error_logs_dict = {"accuracy": 0.0, "errors": error_messages}
        print(f"\n📊 No valid results processed!")
        
    return final_results, error_logs_dict


