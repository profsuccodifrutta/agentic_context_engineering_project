import os  # noqa: I001  -- grouping kept as upstream to keep the diff reviewable
import re
import json
from typing import List, Dict, Any, Optional


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load and process data from a JSONL file.
    
    Args:
        data_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} samples from {data_path}")
    return data

def parse_instruction_and_input(all_context):
    """Parse context to extract question and context parts for finlora_sentiment dataset
    
    Expected format:
    "Instruction: [INSTRUCTION].\nInput: [TEXT]\nAnswer: "
    """
    if "Input: " in all_context and "Instruction: " in all_context:
        # Split by "Input: " to separate instruction from input text
        instruction_part = all_context.split("Input: ")[0].strip()
        instruction_part = instruction_part.split("Instruction: ")[1].strip()
                
        remaining = all_context.split("Input: ")[1]
        input_text = remaining.split("Answer: ")[0].strip()
        return input_text, instruction_part
    
    return "", all_context


def parse_context_and_question_formula(all_context):
    """Parse context to extract question and context parts for formula dataset
    
    Expected format:
    "[some instruction] Question: \"[QUESTION TEXT]\". Answer:"
    """
    if "Question: " in all_context and ". Answer:" in all_context:
        # Split by "Question: " to separate instruction from question
        parts = all_context.split("Question: ", 1)
        instruction_part = parts[0].strip()
        
        # Extract question text (between "Question: " and ". Answer:")
        question_part = parts[1]
        question_text = question_part.split(". Answer:")[0].strip()
        # Remove quotes if present
        if question_text.startswith('"') and question_text.endswith('"'):
            question_text = question_text[1:-1]
        question_text += " Your answer should be a plain floating point number, round to the nearest hundredth if necessary. Do the necessary conversions, for example 5 million should be 5000000.0. "
        return "", question_text

    return "", all_context

#: Suffixes the dataset itself uses in its questions ("5 million should be 5000000.0").
_SCALE_WORDS = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "m": 1e6, "mn": 1e6,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "tn": 1e12,
}

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def normalize_numeric(value):
    """Reduce a model answer to a bare number string, or return None if it is not one.

    The dataset asks for "a numerical answer ... with no explanations or other text", and
    the ACE authors ran it against DeepSeek-V3.1, which complies. Smaller models do not:
    they return "$1,500.00", "15%", "5 million", a value still wrapped in the quotes the
    JSON envelope was supposed to own, or a sentence with the number inside it. Scoring
    those as wrong measures instruction-following, not the finance reasoning under test.

    Deliberately conservative: a string is only reduced to a number when it contains
    exactly one numeric token. "between 3 and 5" stays unparseable and stays wrong.

    A percent sign is stripped but the value is NOT divided by 100 -- the dataset's own
    targets are already in the units its formulas produce (Operating Margin = ... x 100
    has target 15.0), so rescaling here would silently change what is being measured.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return repr(float(value))

    text = str(value).strip()
    # Strip wrappers the JSON envelope should have owned, innermost last.
    for _ in range(3):
        stripped = text.strip().strip("`").strip()
        for quote in ('"""', "'''", '"', "'"):
            if len(stripped) > 2 * len(quote) and stripped.startswith(quote) and stripped.endswith(quote):
                stripped = stripped[len(quote):-len(quote)].strip()
                break
        if stripped == text:
            break
        text = stripped

    text = text.replace(",", "").replace("$", "").replace("%", "")
    text = text.replace("−", "-").replace("–", "-")  # unicode minus / en dash
    text = text.strip().rstrip(".").strip()

    scale = 1.0
    lowered = text.lower()
    for word, factor in _SCALE_WORDS.items():
        # Word-boundary match so "m" does not fire inside "sum" and "b" not inside "bps".
        if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", lowered):
            scale = factor
            lowered = re.sub(rf"(?<![a-z]){re.escape(word)}(?![a-z])", " ", lowered)
            break
    text = lowered if scale != 1.0 else text

    matches = _NUMBER_RE.findall(text)
    if len(matches) != 1:
        # Zero matches: not a number. More than one: ambiguous, and guessing which one the
        # model meant would be scoring the harness's opinion rather than the model's answer.
        return None
    try:
        return repr(float(matches[0]) * scale)
    except (TypeError, ValueError):
        return None


class DataProcessor:
    """
    Processor for handling data preprocessing, evaluation and accuracy computation.
    """
    
    def __init__(self, task_name: str):
        """
        Initialize the data processor.
        
        Args:
            task_name: The name of the task
        """
        self.task_name = task_name
    
    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Process task dataset format.

        Args:
            raw_data: Raw data from JSONL file
            parse_fn: data parsing function 

        Returns:
            Processed data in standard format
        """
        processed_data = []
        if self.task_name == "finer":
            parse_fn = parse_instruction_and_input
        elif self.task_name == "formula":
            parse_fn = parse_context_and_question_formula
        else:
            raise ValueError(f"Unknown task: {self.task_name}")

        for item in raw_data:
            context = item.get('context', '')
            target = item.get('target', '')

            # Parse context to extract the actual text to analyze and the instruction
            input_text, question = parse_fn(context)

            processed_item = {
                "context": input_text,  # The actual context text
                "question": question,   # The instruction/question
                "target": target,       # Ground truth sentiment
                "others": {
                    "original_context": context,
                    "task": self.task_name,
                    "data_source": "finlora"
                }
            }

            processed_data.append(processed_item)

        return processed_data
    
    def _finer_answer_is_correct(self, predicted: str, ground_truth: str, return_counts=False) -> bool:
        """XBRL dataset specific answer correctness check"""
        pred = predicted.split(",")
        pred = [val.lower().strip() for val in pred]
        label = ground_truth.split(",")
        label = [val.lower().strip() for val in label]
        count = 0

        if len(pred) != len(label):
            if len(pred) > len(label):
                pred = pred[:len(label)]
            else:
                padding_needed = len(label) - len(pred)
                pred += ([""] * padding_needed)

        for prediction, ground_truth in zip(pred, label):
            try:
                ground_truth = eval(ground_truth)
                prediction = eval(prediction.replace(",", "").replace("$", ""))
            except:
                pass
            if prediction == ground_truth:
                count += 1
        score = count / len(pred) if pred else 0
        if return_counts:
            return count, len(pred)
        return score == 1 
    
    def _formula_answer_is_correct(self, predicted: str, ground_truth: str,
                                   feedback: Optional[Dict] = None) -> bool:
        """formula dataset specific answer correctness check.

        Comparison itself is unchanged from upstream -- exact float equality. Only the
        normalisation in front of it is new; see normalize_numeric(). It is applied to the
        prediction and the target alike and is identical in every arm, so it cannot favour
        one memory over another.
        """
        pred_norm = normalize_numeric(predicted)
        truth_norm = normalize_numeric(ground_truth)

        if pred_norm is None or truth_norm is None:
            is_correct = str(predicted).strip() == str(ground_truth).strip()
            status = "unparseable_prediction" if pred_norm is None else "unparseable_target"
        else:
            is_correct = float(pred_norm) == float(truth_norm)
            status = "correct" if is_correct else "wrong_value"

        # The reflector's only view of the environment. Without expected-vs-got it is told
        # "does not match ground truth" and cannot see that it answered 0.15 where 15.0 was
        # wanted -- which is exactly the per-formula convention the playbook exists to learn.
        if feedback is not None:
            feedback["status"] = status
            feedback["expected"] = str(ground_truth)
            feedback["got"] = str(predicted)[-2000:]
            feedback["expected_normalized"] = truth_norm
            feedback["got_normalized"] = pred_norm

        return is_correct

    def answer_is_correct(self, predicted: str, ground_truth: str,
                          original_item: Optional[Dict] = None,
                          feedback: Optional[Dict] = None) -> bool:
        """
        Dataset-specific answer correctness check.

        Args:
            predicted: Model's answer
            ground_truth: Ground truth answer
            original_item: Raw dataset row. Unused here -- these tasks are scored against
                the target string alone -- but accepted because utils.check_answer passes
                it for every task.
            feedback: Optional dict, populated in place with what the environment saw.

        Returns:
            bool: True if answer is correct, False otherwise
        """
        if self.task_name == "finer":
            is_correct = self._finer_answer_is_correct(predicted, ground_truth)
            if feedback is not None:
                feedback["status"] = "correct" if is_correct else "wrong_value"
                feedback["expected"] = str(ground_truth)
                feedback["got"] = str(predicted)[-2000:]
            return is_correct
        elif self.task_name == "formula":
            return self._formula_answer_is_correct(predicted, ground_truth, feedback=feedback)
        else:
            raise ValueError(f"Unknown task: {self.task_name}")
    
    def _evaluate_finer_accuracy(self, out: List[str], target: List[str]) -> tuple:
        """FINER dataset specific accuracy evaluation"""
        if len(out) != len(target):
            raise ValueError("Input lists 'out' and 'target' must have the same length.")

        correct_count = 0
        total_count = 0

        for x, y in zip(out, target):
            correct, total = self._finer_answer_is_correct(x, y, return_counts=True)
            correct_count += correct
            total_count += total

        accuracy = 0.0
        if total_count > 0:
            accuracy = correct_count / total_count

        return accuracy
    
    def _evaluate_formula_accuracy(self, out: List[str], target: List[str]) -> tuple:
        """formula dataset specific accuracy evaluation"""
        if len(out) != len(target):
            raise ValueError("Input lists 'out' and 'target' must have the same length.")
        
        correct_count = 0

        for predicted, ground_truth in zip(out, target):
            if self._formula_answer_is_correct(predicted, ground_truth):
                is_correct = True
                correct_count += 1
            else:
                is_correct = False

        accuracy = 0.0
        if len(out) > 0:
            accuracy = correct_count / len(out)

        return accuracy

    
    def evaluate_accuracy(self, out: List[str], target: List[str],
                          original_items: Optional[List[Dict]] = None) -> float:
        """
        Dataset-specific accuracy evaluation.

        Args:
            out: List of model predictions
            target: List of ground truth targets
            original_items: Raw dataset rows, unused here. Accepted because utils passes
                them positionally for every task.

        Returns:
            float: accuracy
        """
        if self.task_name == "finer":
            return self._evaluate_finer_accuracy(out, target)
        elif self.task_name == "formula":
            return self._evaluate_formula_accuracy(out, target)
        else:
            raise ValueError(f"Unknown task: {self.task_name}")
    