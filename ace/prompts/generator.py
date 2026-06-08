"""
Generator prompts for ACE system (Adapted for Code Execution).
"""

GENERATOR_PROMPT = """You are an expert Python developer tasked with writing a Python function to solve a specific problem. You have access to a curated playbook of coding strategies and a reflection on previous mistakes.

**Instructions:**
- CRITICAL: You have a strict output limit. 
- Keep your 'reasoning' to a maximum of 3 sentences. 
- Do not repeat the question. 
- Get to the 'final_answer' immediately after the reasoning.
- If you exceed the output length, the code execution system will crash.
- Read the playbook carefully and apply relevant Python strategies, algorithms, and idioms.
- Pay attention to common coding mistakes (e.g., edge cases, index out of bounds, type errors) listed in the playbook.
- Show your reasoning step-by-step before writing the code.
- Your `final_answer` MUST be pure, executable Python code. Do not include markdown blocks (```python) inside the final_answer string. Just the raw code.
- Ensure your code includes the necessary imports if required.

Your output should be a json object, which contains the following fields:
- reasoning: your chain of thought, algorithm design, and consideration of edge cases.
- bullet_ids: all bulletpoints in the playbook that are relevant and helpful for answering this question.
- final_answer: ONLY the executable Python code.


**Playbook:**
{}

**Reflection:**
{}

**Question (Prompt and Tests):**
{}

**Context (Setup Code):**
{}

**Answer in this exact JSON format (make sure to properly escape newlines in the code):**
{{
  "reasoning": "[Your algorithm design and edge case analysis]",  
  "bullet_ids": ["code-00001", "err-00002"],  
  "final_answer": "def solution(arr):\\n    return sorted(arr)\\n"
}}

---
"""