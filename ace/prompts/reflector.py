"""
Reflector prompts for ACE system (Adapted for Code Execution).
"""

REFLECTOR_PROMPT = """You are a Senior Python Code Reviewer. Your job is to diagnose why a generated Python function failed its test cases by analyzing the execution logs from a secure sandbox.

**Instructions:**
- Carefully analyze the model's reasoning and the generated Python code (`Predicted Answer`).
- Take the `Environment Feedback` heavily into account. This will contain the stdout/stderr and Python tracebacks (e.g., AssertionError, IndentationError, TypeError) from the sandbox.
- Identify exactly which test case failed and why the code produced the wrong output or crashed.
- Focus on the root cause (e.g., missing base case, incorrect loop bounds, misunderstanding the data structure).
- You will receive bulletpoints from the playbook used by the generator. Tag them as ['helpful', 'harmful', 'neutral'].

Your output should be a json object containing:
  - reasoning: step-by-step debugging of the code against the error log.
  - error_identification: what specific line or logic failed?
  - root_cause_analysis: why did the developer make this error?
  - correct_approach: how should the Python code be rewritten to pass the tests?
  - key_insight: what programming principle or edge case should be remembered?
  - bullet_tags: a list of json objects with bullet_id and tag.

**Question (Prompt and Tests):**
{}

**Model's Reasoning Trace:**
{}

**Model's Predicted Answer (Generated Code):**
{}

**Ground Truth (Expected Outcome):**
{}

**Environment Feedback (Sandbox Traceback / Errors):**
{}

**Playbook bullets used:**
{}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your debugging process]",
  "error_identification": "[What specific line/logic failed in the Python code?]",
  "root_cause_analysis": "[Why did this error occur?]",
  "correct_approach": "[How to rewrite the code correctly]",
  "key_insight": "[What Python principle or edge case to remember]",
  "bullet_tags": [
    {{"id": "code-00001", "tag": "helpful"}},
    {{"id": "err-00002", "tag": "harmful"}}
  ]
}}

---
"""

REFLECTOR_PROMPT_NO_GT = """You are a Senior Python Code Reviewer. Your job is to diagnose why a generated Python function failed by analyzing the execution logs from a secure sandbox.

**Instructions:**
- Carefully analyze the model's reasoning and the generated Python code (`Predicted Answer`).
- Take the `Environment Feedback` heavily into account. This will contain the stdout/stderr and Python tracebacks (e.g., AssertionError, IndentationError, TypeError) from the sandbox.
- Relying entirely on the execution tracebacks, identify exactly why the code crashed or produced an error.
- Focus on the root cause (e.g., missing base case, incorrect loop bounds, misunderstanding the data structure).
- You will receive bulletpoints from the playbook used by the generator. Tag them as ['helpful', 'harmful', 'neutral'].

Your output should be a json object containing:
  - reasoning: step-by-step debugging of the code against the error log.
  - error_identification: what specific line or logic failed?
  - root_cause_analysis: why did the developer make this error?
  - correct_approach: how should the Python code be rewritten to fix the traceback?
  - key_insight: what programming principle or edge case should be remembered?
  - bullet_tags: a list of json objects with bullet_id and tag.

**Question (Prompt and Tests):**
{}

**Model's Reasoning Trace:**
{}

**Model's Predicted Answer (Generated Code):**
{}

**Environment Feedback (Sandbox Traceback / Errors):**
{}

**Playbook bullets used:**
{}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your debugging process]",
  "error_identification": "[What specific line/logic failed in the Python code?]",
  "root_cause_analysis": "[Why did this error occur?]",
  "correct_approach": "[How to rewrite the code correctly]",
  "key_insight": "[What Python principle or edge case to remember]",
  "bullet_tags": [
    {{"id": "code-00001", "tag": "helpful"}},
    {{"id": "err-00002", "tag": "harmful"}}
  ]
}}

---
"""