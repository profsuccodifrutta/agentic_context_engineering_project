"""
Generator agent for ACE system.
Generates answers to questions using playbook and reflection.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from ..prompts import CODE_PROMPTS, PromptSet
from llm import timed_llm_call
from playbook_utils import extract_json_from_text

class Generator:
    """
    Generator agent that produces answers to questions using knowledge
    from a playbook and previous reflections.
    """
    
    def __init__(self, api_client, api_provider, model: str, max_tokens: int = 4096,
                 num_ctx: Optional[int] = None, seed: Optional[int] = None,
                 think: Optional[bool] = None, prompts: Optional[PromptSet] = None):
        """
        Initialize the Generator agent.

        Args:
            api_client: OpenAI client for LLM calls
            api_provider: API provider for LLM calls
            model: Model name to use for generation
            max_tokens: Maximum tokens for generation
            num_ctx: Pinned Ollama context window; None auto-sizes from prompt length
            seed: Ollama sampling seed
            think: Ollama reasoning switch; None leaves the model default alone
            prompts: Task's PromptSet; None keeps the code prompts
        """
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.seed = seed
        self.think = think
        self.prompts = prompts or CODE_PROMPTS
    
    def generate(
        self,
        question: str,
        playbook: str,
        context: str = "",
        reflection: str = "(empty)",
        use_json_mode: bool = False,
        call_id: str = "gen",
        log_dir: Optional[str] = None
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Generate an answer to a question using the playbook.
        
        Args:
            question: The question to answer
            playbook: The current playbook content
            context: Additional context for the question
            reflection: Previous reflection content
            use_json_mode: Whether to use JSON mode
            call_id: Unique identifier for this call
            log_dir: Directory for logging
            
        Returns:
            Tuple of (full_response, bullet_ids_used, call_info)
        """
        # Format the prompt. Via the PromptSet, because the code and finance templates bind
        # these four fields differently (positional vs named) and getting it wrong is silent.
        prompt = self.prompts.format_generator(question, context, playbook, reflection)
        
        response, call_info = timed_llm_call(
            self.api_client,
            self.api_provider,
            self.model,
            prompt,
            role="generator",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
            num_ctx=self.num_ctx,
            seed=self.seed,
            think=self.think
        )
        
        # Extract bullet IDs if using retrieval and reason mode
        bullet_ids = []
        bullet_ids = self._extract_bullet_ids(response, use_json_mode)
        
        return response, bullet_ids, call_info
    
    def _extract_bullet_ids(self, response: str, use_json_mode: bool) -> List[str]:
        """
        Extract bullet IDs from generator response.
        
        GENERATOR_PROMPT demands a JSON object regardless of whether the API was
        asked to enforce one, so the JSON path is tried in both modes. Gating it on
        use_json_mode meant that with the flag off (the default) extraction fell
        straight to the regex, which only accepted the bracketed "[abc-00001]" form
        found inside the playbook -- while the model writes them quoted, as
        "bullet_ids": ["abc-00001"]. The result was that essentially no bullet id
        was ever extracted and the helpful/harmful counters never moved.

        Args:
            response: The generator's response
            use_json_mode: Whether JSON mode was used

        Returns:
            List of bullet IDs, de-duplicated, in first-seen order
        """
        parsed = extract_json_from_text(response)
        if isinstance(parsed, dict) and isinstance(parsed.get("bullet_ids"), list):
            candidates = [str(b).strip() for b in parsed["bullet_ids"]]
            return self._dedupe([c for c in candidates if c])

        return self._extract_bullet_ids_regex(response)

    def _extract_bullet_ids_regex(self, text: str) -> List[str]:
        """
        Extract bullet IDs using regex pattern matching.

        Args:
            text: Text to extract bullet IDs from

        Returns:
            List of bullet IDs
        """
        # Matches an id inside brackets or quotes: [abc-00001], "abc-00001", 'ab-00042'.
        # The delimiter is required so ids merely mentioned in prose are not swept up.
        # {2,} rather than {3,} because two-letter slugs are reachable via the
        # initials fallback in get_section_slug.
        pattern = r"""[\[\'"]([a-z]{2,}-\d{5})[\]\'"]"""
        matches = re.findall(pattern, text)
        return self._dedupe(matches)

    @staticmethod
    def _dedupe(values: List[str]) -> List[str]:
        """Drop duplicates while preserving first-seen order."""
        seen = set()
        unique = []
        for value in values:
            if value not in seen:
                seen.add(value)
                unique.append(value)
        return unique