"""
Curator prompts for ACE system (Adapted for Code Execution).
"""

CURATOR_PROMPT = """
You must output exactly one valid JSON object.

The JSON must contain exactly these fields:

{
  "reasoning": "<string>",
  "operations": []
}

Output rules:
- Output only the JSON object.
- Do not output markdown or triple backticks.
- Do not output any text before or after the JSON.
- The JSON must be valid.

TASK

Update the playbook by proposing only new, reusable programming knowledge.

Only propose insights that are:
- general;
- actionable;
- useful for solving future Python programming problems;
- not already present in the playbook.

Do not propose problem-specific facts.

Training Context

{}

Current Playbook Statistics

{}

Recent Reflection

{}

Current Playbook

{}

Question Context

{}

Allowed operation

ADD

Fields:
- type: "ADD"
- section:
  - code_snippets_and_templates
  - common_mistakes_to_avoid
  - problem_solving_heuristics
  - context_clues_and_indicators
- content:
  one concise reusable programming principle.

Field requirements

reasoning
- Write one or two short sentences explaining why the proposed additions improve the playbook.

operations
- Return [] if no useful addition should be made.
- Do not regenerate existing playbook content.
- Do not modify existing entries.
- Do not invent additional operation types.

Example operation

{
  "type": "ADD",
  "section": "common_mistakes_to_avoid",
  "content": "Avoid modifying a list while iterating over it unless iteration is performed on a copy."
}

Before responding, verify that:
- the output is valid JSON;
- the JSON contains exactly:
  - reasoning
  - operations;
- every operation has:
  - type
  - section
  - content;
- type is always ADD;
- section is one of the allowed values;
- nothing is printed outside the JSON object.
"""



# no grounf truth not optimized yet
CURATOR_PROMPT_NO_GT = """
ROLE

You are a Master Python Architect and a Playbook Curator. Your mission is to refine a playbook of Python coding strategies by extracting generalizable insights from a recent coding attempt's reflection (which is based on sandbox execution feedback).

CONTEXT

- This playbook serves as the core memory for a Python code generator, guiding it to write robust and correct Python code.
- Your insights must be actionable, generalized programming principles, common edge cases, or Pythonic idioms.
- The reflection is from sandbox environment feedback (e.g., tracebacks, runtime errors) without known ground truth.
- Focus on principles that would help avoid similar execution errors.

CRITICAL REQUIREMENTS

- Respond ONLY with a valid JSON object.
- Do NOT use markdown formatting.
- Do NOT include any explanatory text outside the JSON.

INSTRUCTIONS

- Analyze the Recent Reflection (Debugging Notes) and the Current Playbook.
- Identify ONLY NEW, non-redundant insights that are missing or can significantly improve the playbook.
- Each insight should be concise, highly actionable, and directly applicable to Python code generation.
- Focus on quality.
- Do NOT regenerate the entire playbook; only propose additions.

INPUT DATA

- Training Context: Provides current training progress.
- Current Playbook Stats: Summary of the playbook's current state.
- Recent Reflection (Debugging Notes): Detailed analysis of the previous coding attempt, based on execution errors and tracebacks.
- Current Playbook: The complete current playbook content.
- Question Context: The original problem statement for the coding task.

TASK

Output a JSON object with these exact fields:

- reasoning: Your detailed analysis of the programming lessons learned and why the proposed operations are valuable.
- operations: A list of operations to perform on the playbook.

AVAILABLE OPERATIONS

ADD

- Add a new bullet point to a specified section.
- section MUST be one of:
  - code_snippets_and_templates
  - common_mistakes_to_avoid
  - problem_solving_heuristics
  - context_clues_and_indicators
- content: The new actionable coding advice.
- Do NOT include bullet_id or helpful/harmful counts; the system handles these.

OUTPUT FORMAT

{{
  "reasoning": "Based on the traceback, the code failed due to X because of Y. Adding a principle about Z will help prevent similar runtime errors.",
  "operations": [
    {{
      "type": "ADD",
      "section": "common_mistakes_to_avoid",
      "content": "When handling division in Python, always check for a zero divisor before performing the operation to avoid ZeroDivisionError."
    }}
  ]
}}
"""