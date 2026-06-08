import os
import json
import re
from collections import defaultdict

def analyze_llm_call_logs(log_dir):
    """
    Analyzes detailed LLM call logs to extract iteration counts per sample.

    Args:
        log_dir (str): Path to the directory containing detailed LLM JSON logs.

    Returns:
        dict: A dictionary where keys are sample IDs (step_id) and values are
              dictionaries containing counts of generator, reflector, and curator
              calls for that sample.
    """
    sample_iteration_counts = defaultdict(lambda: defaultdict(int))
    
    # Updated Regex to extract step_id and agent role from call_id
    # Handles formats like:
    # train_e_1_s_1_gen_initial
    # train_e_1_s_1_round_0
    # train_e_1_s_1_post_reflect_round_0
    # train_e_1_s_1_post_curate
    # online_train_s_1_gen_initial
    # online_train_s_1_chunk_1
    # test_eval_0, test_eval_1, etc. (for evaluation only)
    # Added _reflect_on_correct and _reflect_no_gt to the optional action group
    call_id_pattern = re.compile(
        r"^(train_e_\d+_s_\d+|online_train_s_\d+|test_eval_\d+)" # Base sample identifier
        r"(_gen_initial|_round_\d+|_post_reflect_round_\d+|_post_curate|_reflect_on_correct|_reflect_no_gt)?(_chunk_\d+)?$" # Optional action and chunk
    )


    for filename in os.listdir(log_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(log_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    log_entry = json.load(f)
                    
                call_id = log_entry.get("call_id")
                role = log_entry.get("role")

                if call_id and role:
                    match = call_id_pattern.match(call_id)
                    if match:
                        step_id_prefix = match.group(1) # e.g., "train_e_1_s_1", "online_train_s_1", or "test_eval_0"
                        
                        sample_id = step_id_prefix

                        
                        if role.lower() == "generator": 
                            sample_iteration_counts[sample_id]["generator_calls"] += 1
                        elif role.lower() == "reflector":
                            sample_iteration_counts[sample_id]["reflector_calls"] += 1
                        elif role.lower() == "curator":
                            sample_iteration_counts[sample_id]["curator_calls"] += 1
                        elif role.lower() == "bulletpoint_analyzer":
                            sample_iteration_counts[sample_id]["bulletpoint_analyzer_calls"] += 1

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return sample_iteration_counts


def main():
    # Directly set the run_save_path to your specific timestamped directory.
    run_save_path = "./results/mbpp_offline_run/ace_run_20260531_230836_mbpp_code_exec_offline" 

    detailed_log_dir = os.path.join(run_save_path, "detailed_llm_logs")
    output_file = os.path.join(run_save_path, "llm_iteration_counts.json")

 
    if not os.path.exists(detailed_log_dir):
        print(f"Error: Detailed LLM log directory not found at {detailed_log_dir}")
        print("Please ensure the 'run_save_path' points to the correct timestamped ACE run directory.")
        return

    print(f"Analyzing LLM call logs in: {detailed_log_dir}")
    iteration_counts = analyze_llm_call_logs(detailed_log_dir)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(iteration_counts, f, indent=2, ensure_ascii=False)

    print(f"\nAnalysis complete. Iteration counts saved to: {output_file}")
    total_generator_calls = sum(counts.get("generator_calls", 0) for counts in iteration_counts.values())
    total_reflector_calls = sum(counts.get("reflector_calls", 0) for counts in iteration_counts.values())
    total_curator_calls = sum(counts.get("curator_calls", 0) for counts in iteration_counts.values())
    print(f"Total Generator calls across all samples: {total_generator_calls}")
    print(f"Total Reflector calls across all samples: {total_reflector_calls}")
    print(f"Total Curator calls across all samples: {total_curator_calls}")

if __name__ == "__main__":
    main()

