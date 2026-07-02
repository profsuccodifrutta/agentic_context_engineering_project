GENERATOR_PROMPT = """
You must output exactly one valid JSON object.

The JSON must contain exactly these fields:
{
  "reasoning": "<string>",
  "bullet_ids": ["<string>"],
  "final_answer": "<string>"
}

Output rules:
- Output only the JSON object.
- Do not output any text before or after the JSON.
- Do not output markdown or triple backticks.
- The JSON must be valid.

TASK

Write a correct Python solution that passes all provided tests.

Use the following information while solving the task.

QUESTION (PROMPT AND TESTS)
{}

CONTEXT (SETUP CODE)
{}

PLAYBOOK
{}

REFLECTION
{}

Field requirements

reasoning
- Write one short sentence describing the algorithm or approach.

bullet_ids
- Include the ids of the playbook entries that influenced your solution.
- If none were useful, return [].
- Do not invent ids.

final_answer
- Must be a single JSON string containing only executable Python code.
- Include all required imports.
- Define the function with the exact name expected by the tests.
- The solution must be self-contained.
- Handle all cases implied by the prompt and tests.
- Escape every newline as \\n so the value remains a valid JSON string.
- Do not wrap the code in backticks.

Example of the required format for final_answer:

"final_answer": "def add(a, b):\\n    return a + b\\n"

Before responding, verify that:
- your output is valid JSON;
- the JSON contains exactly these keys:
  - reasoning
  - bullet_ids
  - final_answer
- reasoning is a string;
- bullet_ids is a JSON array;
- final_answer is a JSON string containing executable Python code with escaped newlines;
- nothing is printed outside the JSON object.
"""