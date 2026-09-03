import os  # noqa: I001  -- grouping kept as upstream to keep the diff reviewable
import re
import ast
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


#: The phrase every `finer` sub-question opens with. It does two jobs, which is why it is named
#: once here rather than spelled out in three patterns: it locates where the taxonomy preamble
#: ends, and it is itself a substring shared by every query, so it is stripped from every part.
_FINER_SUBQUESTION_STEM = "What is best tag for entity"

#: Where a `finer` question stops being boilerplate and starts being the four sub-questions.
#: Every row prints the same 140-tag taxonomy first, so two questions share their first ~6,479
#: characters and differ only after this marker.
_FINER_SUBQUESTION_MARKER = re.compile(r"\n\s*1\.\s*" + _FINER_SUBQUESTION_STEM, re.I)

#: Splits the tail into its numbered parts and strips the template from each. A part that does
#: not match is kept whole, so a format change degrades to the pre-2026-09-03 string rather than
#: dropping the row's text.
# `re.I` on all three, matching `_FINER_SUBQUESTION_MARKER`: the marker decides *whether* a row
# has sub-questions, and these two decide how they are split and stripped. If the marker matched a
# casing variant that these did not, the row would keep the template in every part and the drifted
# rules would win the ranking again -- silently, and the gate would not see it because the parts
# would still be distinct per row.
_FINER_SUBQUESTION_SPLIT = re.compile(
    r"(?=^\s*\d\.\s+" + _FINER_SUBQUESTION_STEM + r")", re.M | re.I
)
_FINER_SUBQUESTION = re.compile(
    r'^\s*\d\.\s+' + _FINER_SUBQUESTION_STEM + r'\s+"(.*?)"\s+in sentence:\s*"?(.*?)"?\s*$',
    re.S | re.I
)


def finer_retrieval_query(all_context):
    """What retrieval scores `finer` rules against: one string per sub-question, template removed.

    The generator prompt is untouched and still receives the whole question. Only what reaches
    the encoder changes.

    **Two shared substrings had to go, and they were found a fix apart.** First the 140-tag
    taxonomy every row prints before its sub-questions: it fills the encoder's 384-token window,
    so every row encoded to the same vector and one slice of the playbook was rendered on every
    call (2026-08-30). Then the sub-question template itself, `What is best tag for entity "X" in
    sentence:`, identical in every sub-question of every row: it is the vocabulary the drifted
    rules are written in, so those rules won every ranking while the terse `IF ... THEN <tag>`
    mappings the design is about lost (2026-09-03). Both are the same defect -- a substring shared
    by every query deciding the outcome -- and the second is why this returns a list of stripped
    parts rather than the tail as one string. The measurements are in
    docs/EXPERIMENT_FINER.md section 9 and are not duplicated here.

    Scoring each rule as its best match against any entry is what makes the split worth having:
    a rule that answers one sub-question exactly is no longer beaten by a rule vaguely related to
    all four. See `ace.memory.retrieval.score_against_query`.

    **Splitting also all but ends the window overflow.** As one string a third of rows exceeded
    ~1,500 characters, so their third and fourth sub-questions were never encoded. Per part the
    median is 246 to 249 characters, p90 546 to 573, and 3 of 1,764 test parts and 3 of 600
    validation parts still overflow -- reported by `check_finance_wiring.py` [1n], not gated,
    because the gated property is that no shared prefix survives, never the length.

    A row that carries neither the marker nor the template degrades to the whole text as a single
    entry, which is the pre-2026-08-30 behaviour.
    """
    match = _FINER_SUBQUESTION_MARKER.search(all_context)
    tail = all_context[match.start():].strip() if match else all_context

    # One list entry per sub-question, with the template removed. See the docstring.
    parts = [p.strip() for p in _FINER_SUBQUESTION_SPLIT.split(tail) if p.strip()]
    stripped = []
    for part in parts:
        sub = _FINER_SUBQUESTION.match(part)
        stripped.append(f"{sub.group(1)} {sub.group(2)}".strip() if sub else part)
    return stripped or [tail]




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


def normalize_finer_answer(text: str) -> str:
    """Reduce a model's four-tag answer to a bare comma-separated list.

    Two envelopes survive `extract_answer` intact and would each score a correct answer wrong,
    both found by check_finance_wiring [1j] before any finer run existed:

      * `"final_answer": "\\"A,B,C,D\\""` -- the double-stringification habit CLAUDE.md records
        for MBPP, where it made 28 of one arm's 257 answers unreadable and *none* in the other
        two. Left alone the outer quote characters attach to the first and last tag, so a
        perfect answer scores 2 of 4.
      * `"final_answer": ["A","B","C","D"]` -- the shape a model asked for four tags is most
        likely to invent. Left alone it stringifies to a list repr and scores 0 of 4.

    This lives here rather than in `extract_answer` on purpose. The knowledge "this answer is a
    list of tags" is specific to `finer`; putting it in the shared extractor would change how
    every task reads a multi-element array, and the formula and MBPP runs are finished.

    Applied to the prediction only. The targets come from the dataset and are already bare, and
    normalising the untrusted side alone keeps the comparison from being loosened on both.
    """
    text = (text or "").strip()
    for _ in range(3):  # bounded: a model nests these, but never deeply
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
            text = text[1:-1].strip()
            continue
        if text.startswith("[") and text.endswith("]"):
            try:
                items = ast.literal_eval(text)
            except (ValueError, SyntaxError, MemoryError, TypeError):
                break
            if not isinstance(items, (list, tuple)) or not all(
                    isinstance(x, str) for x in items):
                break
            text = ",".join(x.strip() for x in items)
            continue
        break
    return text


def finer_tag_counts(predicted: str, ground_truth: str):
    """Positional tag comparison for `finer`, returning (matched, total, per_position).

    One row of `finer` is four sub-questions answered in a single call, so its target is four
    comma-separated US GAAP tags and the comparison is positional: pred[i] against gold[i].
    Every consumer of that comparison goes through here -- the all-four boolean
    (`_finer_answer_is_correct`), the per-tag aggregate (`_evaluate_finer_accuracy`) and
    `compare_arms.py --metric finer_per_tag` -- so the three can never drift apart.

    `per_position` is [(i, predicted_tag, gold_tag, matched)], which the reflector needs: told
    only "does not match ground truth" it has to diff four long CamelCase strings itself and
    guess which one it got wrong. That is the same defect that made MBPP accumulate generic
    style advice, and it is repaired the same way, on the input side.

    Upstream used bare `eval()` on both sides. That evaluates model output as Python; the
    conversion never fires on tag names, so `ast.literal_eval` is behaviour-identical on this
    data and removes the execution path. `finer` has never been run here, so no result changes.
    """
    pred = [val.lower().strip() for val in normalize_finer_answer(predicted).split(",")]
    gold = [val.lower().strip() for val in (ground_truth or "").split(",")]

    if len(pred) > len(gold):
        pred = pred[:len(gold)]
    elif len(pred) < len(gold):
        pred += [""] * (len(gold) - len(pred))

    count, per_position = 0, []
    for index, (got, want) in enumerate(zip(pred, gold), start=1):
        left, right = got, want
        try:
            right = ast.literal_eval(want)
            left = ast.literal_eval(got.replace(",", "").replace("$", ""))
        except (ValueError, SyntaxError, MemoryError, TypeError):
            pass
        matched = left == right
        count += matched
        per_position.append((index, got, want, matched))
    return count, len(gold), per_position


def format_finer_positions(per_position) -> str:
    """One-line-per-tag rendering of finer_tag_counts()'s third return value."""
    return "; ".join(
        f"tag {i}: correct ({want})" if ok else f"tag {i}: WRONG, expected {want}, got {got or '(missing)'}"
        for i, got, want, ok in per_position
    )


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
                # The string retrieval scores rules against, when it must differ from the
                # question the generator reads. Absent for every task whose question already
                # fits the encoder, which leaves those tasks byte-identical. See
                # `finer_retrieval_query`.
                # Built from `question`, not from `context`: `question` is what
                # `ACE._retrieval_query` falls back to and what `analyze_walk_share.py` recovers
                # from the saved prompt, so transforming it keeps all three in step even if the
                # `Instruction:`/`Input:` envelope is ever added to `finer` and the parser starts
                # splitting the row.
                "retrieval_query": (finer_retrieval_query(question)
                                    if self.task_name == "finer" else None),
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
        count, total, _ = finer_tag_counts(predicted, ground_truth)
        if return_counts:
            return count, total
        return total > 0 and count == total

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
            count, total, per_position = finer_tag_counts(predicted, ground_truth)
            is_correct = total > 0 and count == total
            if feedback is not None:
                feedback["status"] = "correct" if is_correct else "wrong_value"
                feedback["expected"] = str(ground_truth)
                feedback["got"] = str(predicted)[-2000:]
                # Which of the four went wrong, not just that the row did. format_environment
                # _feedback renders unknown keys verbatim, so this needs no change there.
                feedback["tags_correct"] = f"{count} of {total}"
                feedback["per_tag"] = format_finer_positions(per_position)
                # The same quantity as a number, for the loop rather than for the prompt.
                # `is_correct` above is all-four-exact, which reports a 3-of-4 row as a total
                # failure and costs the reflector the `helpful` tag those three tags earned.
                # `format_environment_feedback` skips this key, so the reflector prompt is
                # unchanged byte for byte.
                feedback["partial_credit"] = (count / total) if total else 0.0
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
    