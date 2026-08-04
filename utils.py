#!/usr/bin/env python3
import os
import re
import ast
import json
import inspect
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

def get_original_item(task_dict):
    """Return the raw task record that the data processor expects.

    process_task_data() wraps the raw dataset row (which carries test_setup_code
    and test_list) under "original_mbpp_item". Passing the processed record to
    answer_is_correct() instead makes both of those fields resolve empty, so the
    sandbox executes the prediction with no assertions attached and scores any
    code that merely imports and defines as correct.
    """
    if isinstance(task_dict, dict) and "original_mbpp_item" in task_dict:
        return task_dict["original_mbpp_item"]
    return task_dict

# Cached per processor class: only mbpp_code_exec accepts the feedback out-parameter,
# and the other processors in this repo still carry the upstream signature.
_FEEDBACK_CAPABLE_PROCESSORS = {}

def check_answer(data_processor, predicted, target, original_item, feedback=None):
    """answer_is_correct(), additionally capturing execution output when supported.

    Pass a dict as `feedback` to have it filled in place with whatever the processor
    knows about the run (for MBPP: sandbox status, stdout, stderr). Support is detected
    from the signature rather than assumed, so processors that were never updated keep
    working and simply return no feedback.
    """
    if feedback is None:
        return data_processor.answer_is_correct(predicted, target, original_item)

    cls = type(data_processor)
    supports = _FEEDBACK_CAPABLE_PROCESSORS.get(cls)
    if supports is None:
        try:
            supports = "feedback" in inspect.signature(data_processor.answer_is_correct).parameters
        except (TypeError, ValueError):
            supports = False
        _FEEDBACK_CAPABLE_PROCESSORS[cls] = supports

    if supports:
        return data_processor.answer_is_correct(predicted, target, original_item, feedback=feedback)
    return data_processor.answer_is_correct(predicted, target, original_item)

def format_environment_feedback(is_correct, feedback=None):
    """Render the ENVIRONMENT FEEDBACK block of the reflector prompt.

    The verdict line is the string the loop used to pass on its own, so a processor that
    captured nothing degrades to exactly the previous behavior rather than to an empty
    section.
    """
    verdict = ("Predicted answer matches ground truth" if is_correct
               else "Predicted answer does not match ground truth")
    if not feedback:
        return verdict

    # Execution tasks keep the sandbox wording verbatim -- the MBPP arms were run against
    # this exact text and changing it would make them non-reproducible.
    if "stderr" in feedback or "stdout" in feedback:
        parts = [verdict, f"Sandbox status: {feedback.get('status')}"]
        stderr = (feedback.get("stderr") or "").strip()
        stdout = (feedback.get("stdout") or "").strip()
        if stderr:
            parts.append(f"Sandbox stderr:\n{stderr}")
        if stdout:
            parts.append(f"Sandbox stdout:\n{stdout}")
        return "\n\n".join(parts)

    # Comparison tasks: show expected against got. A reflector told only "does not match
    # ground truth" has to guess at the cause, which is how a memory ends up full of
    # generic advice instead of the convention that was actually violated.
    parts = [verdict]
    if feedback.get("status") is not None:
        parts.append(f"Status: {feedback['status']}")
    if "expected" in feedback:
        parts.append(f"Expected answer: {feedback['expected']}")
    if "got" in feedback:
        parts.append(f"Predicted answer: {feedback['got']}")
    normalized = feedback.get("got_normalized")
    if normalized is None and feedback.get("status") == "unparseable_prediction":
        parts.append("The predicted answer could not be read as a single number.")
    for key, value in feedback.items():
        if key in ("status", "expected", "got", "expected_normalized", "got_normalized"):
            continue
        if value not in (None, ""):
            parts.append(f"{key}: {value}")
    return "\n\n".join(parts)

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
    
    # Clean and convert to snake_case. The hyphen replacement matters: without it
    # "PROBLEM-SOLVING HEURISTICS" normalizes to "problem-solving_heuristics",
    # misses the slug_map key below, and falls through to the initials branch,
    # minting the 2-letter slug "ph" instead of "prob". Kept in sync with
    # playbook_utils.normalize_section_name (imported there, not here, to avoid a cycle).
    clean_name = section_name.lower().strip().replace(" ", "_").replace("&", "and").replace("-", "_")
    
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

def unwrap_quoted_source(answer):
    """Remove a redundant quote pair the model wrapped around the whole program.

    The generator prompt asks for the code as a JSON string, and some models stringify it
    twice, emitting `"final_answer": "\\"def f():\\n    ...\\n\\""`. After json.loads the
    value still carries literal quote characters at both ends, so the sandbox sees a bare
    string literal instead of a function definition: every assertion then fails on a name
    that was never defined, and the run scores zero while looking like a reasoning problem.
    Measured on qwen3.5:2b, 94 of 110 answers came back like this and only 10% of answers
    parsed as Python at all; unwrapping takes that to 78%.

    The bracketed forms are the same habit spelled as an array literal: the value arrives
    as the string `["def f():\\n    ..."]`, so json.loads yields a str that still carries
    brackets and quotes. Measured on the MBPP arms: 68 of 599 parsed generations in the
    text arm, 6 in the graph arm, 0 in the memory-free arm -- 28 of them in a single final
    test set, 10.9% of that arm's answers unreadable for formatting rather than reasoning.

    Only applied when the answer does not already parse and unwrapping makes it parse, so
    it can turn a syntax error into working code but can never alter code that was fine.
    """
    if not answer or not isinstance(answer, str):
        return answer
    try:
        ast.parse(answer)
        return answer
    except SyntaxError:
        pass

    stripped = answer.strip()
    # Triple quotes first: '"' is a prefix of '"""', so the shorter form would match and
    # leave two stray quotes behind. The bracketed forms cover a one-element JSON array
    # that reached here as text -- extract_answer unwraps that case before stringifying,
    # but the regex fallback path never parses the JSON and so still sees `["def f()..."]`.
    for quote, closing in (('["', '"]'), ("['", "']"),
                           ('"""', '"""'), ("'''", "'''"),
                           ('"', '"'), ("'", "'")):
        if not stripped.startswith(quote):
            continue
        inner = stripped[len(quote):]
        # The closing quote is not always last: models emit `...";` and, when the JSON
        # extraction overcaptures, `..."}`. Trailing punctuation is never valid Python
        # here, so dropping it costs nothing and the ast.parse guard below still decides.
        tail = inner.rstrip().rstrip(";,}") if closing.endswith("]") else inner.rstrip().rstrip(";,}]")
        if tail.rstrip().endswith(closing):
            inner = tail.rstrip()[:-len(closing)]
        try:
            ast.parse(inner)
            return inner
        except SyntaxError:
            continue
    return answer

# Backslash not followed by one of JSON's own escapes. LaTeX in a reasoning field is full
# of these -- \frac, \sigma, \approx -- and one of them invalidates the whole envelope even
# though the answer field itself is fine.
_INVALID_JSON_ESCAPE = re.compile(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})')


def repair_json_escapes(text):
    """Escape the backslashes JSON does not recognise, leaving the valid ones alone.

    Only ever called after a strict parse has already failed, so it cannot change how a
    well-formed response is read. Measured on the formula probe: the generator writes its
    working in LaTeX ("$$Yield = \\frac{Dividend}{Price}$$"), and `\\f` aside, none of those
    sequences is a legal JSON escape -- 22 of 29 unparseable responses failed on exactly
    that, with a perfectly good answer sitting in the next field.
    """
    return _INVALID_JSON_ESCAPE.sub(r'\\\\', text)


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
        try:
            parsed = json.loads(clean_response, strict=False)
        except json.JSONDecodeError:
            # Second chance before falling through to the regexes: an unrecognised escape
            # anywhere in the object -- usually LaTeX in "reasoning" -- otherwise discards
            # a response whose answer field is intact.
            parsed = json.loads(repair_json_escapes(clean_response), strict=False)
        raw = parsed.get("final_answer", "No final answer found")
        # Defensive: a model may return the answer as a one-element array rather than a
        # string, and str() on a list would leave literal brackets and quotes around the
        # program. Not the form seen in the MBPP arms -- there the value was a *string*
        # containing "[\"def f(): ...\"]", which unwrap_quoted_source handles below -- but
        # both spellings of the same habit reach the sandbox as an undefined name.
        if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], str):
            raw = raw[0]
        answer = str(raw)

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
            answer = unwrap_quoted_source(answer)

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
            return unwrap_quoted_source(answer)
        
        # Try finding single quotes: 'final_answer': '...'
        matches = re.findall(r"'final_answer'\s*:\s*'((?:[^'\\]|\\.)*)'", response, re.DOTALL)
        if matches:
            answer = matches[-1]
            answer = answer.replace("\\'", "'") # Restore escaped quotes
            answer = answer.replace("```python", "").replace("```", "").replace("\\n", "\n")
            answer = answer.replace('\xa0', ' ').replace('\t', '    ').strip()
            return unwrap_quoted_source(answer)

        # Unquoted value: `"final_answer": 229.27`. Both regexes above demand an opening
        # quote, so a numeric-answer task loses the answer entirely whenever the envelope
        # is malformed -- the model writes the number bare because it *is* a number, and
        # sometimes leaves a stray closing quote after it. Restricted to a numeric literal
        # so it can never swallow a trailing fragment of a broken object.
        matches = re.findall(r'"final_answer"\s*:\s*(-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)',
                             response)
        if matches:
            return matches[-1].strip()

        # If everything fails, print the raw output to see how the LLM broke it
        print(f"\n[Extraction Error] Failed to parse code. Raw LLM response:\n{response[:250]}...\n")
        return "raise ValueError('No valid code extracted from LLM')"
    
enc = tiktoken.get_encoding("cl100k_base")
def count_tokens(prompt: str) -> int:
    return len(enc.encode(prompt))


#: Columns of the per-sample prediction dump. Kept small and stable so rescore_from_logs.py
#: and compare_arms.py can read runs produced by either path.
PREDICTION_FIELDS = ("index", "final_answer", "target", "is_correct", "success")

def write_predictions_jsonl(path, sample_results):
    """Write per-sample test results, sorted by index, one JSON object per line."""
    rows = sorted(sample_results, key=lambda r: r.get("index", -1))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r.get(k) for k in PREDICTION_FIELDS}) + "\n")
    return path


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
        
        
        is_correct = data_processor.answer_is_correct(final_answer, target, get_original_item(task_dict))

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
                      use_json_mode=False, predictions_path=None) -> Tuple[Dict, Dict]:
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
        predictions_path: Optional path for a per-sample JSONL dump. Only aggregates were
            persisted before, which made a paired comparison between two runs impossible
            after the fact -- the per-problem correctness a McNemar test needs was computed
            here and then thrown away.

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
    
    if predictions_path:
        write_predictions_jsonl(predictions_path, all_sample_results)

    if all_sample_results:
        predictions = [r["final_answer"] for r in all_sample_results]
        ground_truths = [r["target"] for r in all_sample_results]
        
        original_items_for_accuracy = [get_original_item(test_samples[r["index"]]) for r in all_sample_results]

        overall_accuracy = data_processor.evaluate_accuracy(predictions, ground_truths, original_items_for_accuracy)
        
        total_correct = sum(1 for r in all_sample_results if r["is_correct"])
        total_total = len(all_sample_results)
        
        total_no_answer = sum(
            1 for r in all_sample_results 
            if r["final_answer"] in ["No final answer found", "raise ValueError('No valid code extracted from LLM')"]
        )
        
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


