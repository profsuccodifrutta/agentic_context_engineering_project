"""
Curator prompts for ACE system (Adapted for Code Execution).
"""

CURATOR_PROMPT = """
You must output exactly one valid JSON object.

The JSON must contain exactly these fields:

{{
  "reasoning": "<string>",
  "operations": []
}}

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
Step {current_step} of {total_samples} (Token budget: {token_budget})

Current Playbook Statistics
{playbook_stats}

Recent Reflection
{recent_reflection}

Current Playbook
{current_playbook}

Question Context
{question_context}

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
- refines (required, may be null):
  the id of an existing playbook entry that this new entry supersedes, copied
  verbatim from the Current Playbook, for example "err-00005", or null if the new
  entry supersedes nothing.

Field requirements

reasoning
- Write one or two short sentences explaining why the proposed additions improve the playbook.

operations
- Return [] if no useful addition should be made.
- Do not regenerate existing playbook content.
- Do not modify existing entries.
- Do not invent additional operation types.

refines
- Every ADD must carry this field. Before proposing an entry, scan the Current
  Playbook for an existing entry covering the same ground, and decide explicitly.
- Set it to that entry's id when the new entry states the same guidance more
  correctly, more precisely, or more generally, so that keeping both would be
  redundant. Two entries giving the same advice in different words are redundant.
- Set it to null when the new entry stands on its own, is merely related to an
  existing entry, or disagrees with one.
- null is a legitimate and common answer; do not invent a supersession to fill the
  field.
- Never point it at an entry you are adding in this same response.
- Never invent an id.

Example operation that supersedes nothing

{{
  "type": "ADD",
  "section": "common_mistakes_to_avoid",
  "content": "Avoid modifying a list while iterating over it unless iteration is performed on a copy.",
  "refines": null
}}

Example operation that supersedes an existing entry

{{
  "type": "ADD",
  "section": "common_mistakes_to_avoid",
  "content": "Avoid mutating any collection while iterating it; iterate over a copy or build a new collection.",
  "refines": "err-00005"
}}

Before responding, verify that:
- the output is valid JSON;
- the JSON contains exactly:
  - reasoning
  - operations;
- every operation has:
  - type
  - section
  - content
  - refines;
- type is always ADD;
- section is one of the allowed values;
- refines is present on every operation and is either null or an id that appears in
  the Current Playbook;
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
- refines (required, may be null): the id of an existing entry that this new entry
  supersedes, copied verbatim from the Current Playbook, or null if it supersedes
  nothing. Every ADD must carry this field. Set it to an id when the new entry states
  the same guidance more correctly, more precisely, or more generally, so that keeping
  both would be redundant -- two entries giving the same advice in different words are
  redundant. Set it to null when the entry stands on its own, is merely related to an
  existing one, or disagrees with one; null is legitimate and common. Never invent an
  id and never point it at an entry added in this same response.
- Do NOT include bullet_id or helpful/harmful counts; the system handles these.

OUTPUT FORMAT

{{
  "reasoning": "Based on the traceback, the code failed due to X because of Y. Adding a principle about Z will help prevent similar runtime errors.",
  "operations": [
    {{
      "type": "ADD",
      "section": "common_mistakes_to_avoid",
      "content": "When handling division in Python, always check for a zero divisor before performing the operation to avoid ZeroDivisionError.",
      "refines": null
    }},
    {{
      "type": "ADD",
      "section": "problem_solving_heuristics",
      "content": "Validate every external input at the function boundary, not just numeric ranges.",
      "refines": "ph-00022"
    }}
  ]
}}
"""