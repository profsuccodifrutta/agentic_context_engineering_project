"""
Reflector agent for ACE system.
Analyzes generator outputs and provides feedback on bullet usage.
"""

import json  # noqa: I001  -- grouping matches the sibling agents in this package
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from ..prompts import CODE_PROMPTS, PromptSet
from llm import timed_llm_call
from playbook_utils import extract_json_from_text
from utils import count_tokens

#: Token ceiling for the generator's reasoning trace inside the reflector prompt.
#:
#: The trace is the one field in that prompt with no natural bound: the generator can spend
#: its entire max_tokens budget rambling, and on the formula probe one produced 36,268
#: characters, which alone is 89.5% of a 40,522-character reflector prompt (14,675 tokens).
#: That call came back empty fifteen times in a row. `num_ctx` is pinned to 32768, but
#: Ollama clamps it to the model's own trained context and gemma4:e2b is smaller: every
#: reflector call at or below 12,433 tokens succeeded, the one at 14,822 was left 147 tokens
#: of output budget, spent them all on hidden reasoning, and returned nothing. The step's
#: learning signal was lost and fifteen retries were paid for it.
#:
#: 4000 tokens is still a very long trace -- roughly eleven thousand characters -- and keeps
#: the whole prompt clear of the wall with room for the reflector's own output. Truncation
#: is applied identically in every arm and never touches the playbook content, which is the
#: variable under test and renders longer in the graph arm.
MAX_TRACE_TOKENS = 4000

_TRUNCATION_MARKER = "\n\n[... middle of the reasoning trace omitted: it exceeded the " \
                     "reflector's context budget ...]\n\n"


def _truncate_trace(trace: str, max_tokens: int = MAX_TRACE_TOKENS) -> str:
    """Cut an over-long reasoning trace down, keeping its head and its tail.

    Head and tail rather than a prefix: the head carries the problem setup and the approach
    taken, the tail carries the final calculation and the answer. A diagnosis needs both,
    and what sits between them in a runaway trace is the model going in circles.
    """
    if not trace or count_tokens(trace) <= max_tokens:
        return trace

    # Work in characters, then verify: cheaper than a token-by-token search and the ratio
    # is stable enough that one corrective pass is always sufficient.
    ratio = len(trace) / max(count_tokens(trace), 1)
    budget = int(max_tokens * ratio)
    head_chars, tail_chars = int(budget * 0.6), int(budget * 0.4)
    out = trace[:head_chars] + _TRUNCATION_MARKER + trace[-tail_chars:]
    while count_tokens(out) > max_tokens and head_chars > 500:
        head_chars, tail_chars = int(head_chars * 0.8), int(tail_chars * 0.8)
        out = trace[:head_chars] + _TRUNCATION_MARKER + trace[-tail_chars:]
    return out


class Reflector:
    """
    Reflector agent that analyzes the generator's reasoning and tags
    bullets as helpful, harmful, or neutral.
    """
    
    def __init__(self, api_client, api_provider, model: str, max_tokens: int = 4096,
                 num_ctx: Optional[int] = None, seed: Optional[int] = None,
                 think: Optional[bool] = None, prompts: Optional[PromptSet] = None):
        """
        Initialize the Reflector agent.

        Args:
            api_client: OpenAI client for LLM calls
            api_provider: API provider for LLM calls
            model: Model name to use for reflection
            max_tokens: Maximum tokens for reflection
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
    
    def reflect(
        self,
        question: str,
        reasoning_trace: str,
        predicted_answer: str,
        ground_truth: Optional[str],
        environment_feedback: str,
        bullets_used: str,
        use_ground_truth: bool = True,
        use_json_mode: bool = False,
        call_id: str = "reflect",
        log_dir: Optional[str] = None,
        attempt_is_correct: Optional[bool] = None
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """
        Analyze the generator's output and tag bullets.

        Args:
            question: The original question
            reasoning_trace: The generator's reasoning
            predicted_answer: The generator's predicted answer
            ground_truth: The ground truth answer (if available)
            environment_feedback: Feedback about correctness
            bullets_used: String representation of bullets used
            use_ground_truth: Whether to use ground truth in reflection
            use_json_mode: Whether to use JSON mode
            call_id: Unique identifier for this call
            log_dir: Directory for logging
            attempt_is_correct: Outcome of the attempt being reflected on. When False,
                'helpful' tags are demoted to 'neutral' -- see _filter_bullet_tags.
                None disables the check.

        Returns:
            Tuple of (reflection_content, bullet_tags, call_info)
        """
        # A runaway trace overflows the served context and the call comes back empty, which
        # costs the retry budget and loses this step's learning signal entirely. See
        # MAX_TRACE_TOKENS.
        original_tokens = count_tokens(reasoning_trace) if reasoning_trace else 0
        reasoning_trace = _truncate_trace(reasoning_trace)
        if original_tokens > MAX_TRACE_TOKENS:
            print(f"[REFLECTOR] {call_id}: reasoning trace {original_tokens} tokens -> "
                  f"{count_tokens(reasoning_trace)}, head+tail kept")

        # Select the appropriate prompt
        if use_ground_truth and ground_truth:
            prompt = self.prompts.reflector.format(
                question,
                reasoning_trace,
                predicted_answer,
                ground_truth,
                environment_feedback,
                bullets_used
            )
        else:
            prompt = self.prompts.reflector_no_gt.format(
                question,
                reasoning_trace,
                predicted_answer,
                environment_feedback,
                bullets_used
            )
        
        response, call_info = timed_llm_call(
            self.api_client,
            self.api_provider,
            self.model,
            prompt,
            role="reflector",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
            num_ctx=self.num_ctx,
            seed=self.seed,
            think=self.think
        )
        
        # Extract bullet tags, keeping only ids the reflector was actually shown
        bullet_tags = self._extract_bullet_tags(
            response, use_json_mode, self._ids_shown(bullets_used), attempt_is_correct
        )

        return response, bullet_tags, call_info

    @staticmethod
    def _ids_shown(bullets_used: str) -> Set[str]:
        """
        Collect the bullet ids that were rendered into the prompt.

        extract_playbook_bullets() emits one "[id] helpful=X harmful=Y :: content"
        line per bullet, or a parenthesised sentinel when there are none.
        """
        if not bullets_used:
            return set()
        return set(re.findall(r'^\s*\[([^\]]+)\]\s*helpful=', bullets_used, re.MULTILINE))

    def _extract_bullet_tags(
        self,
        response: str,
        use_json_mode: bool,
        valid_ids: Optional[Set[str]] = None,
        attempt_is_correct: Optional[bool] = None
    ) -> List[Dict[str, str]]:
        """
        Extract bullet tags from reflector response.

        Tags whose 'id' was not among the bullets shown to the reflector are
        dropped. Models frequently copy the schema example's tag value into the
        'id' field (producing {"id": "helpful", ...}), and update_bullet_counts()
        ignores unknown ids without complaint, so unfiltered tags look like a
        working signal while silently updating nothing.

        Args:
            response: The reflector's response
            use_json_mode: Whether JSON mode was used
            valid_ids: Bullet ids the reflector was shown; None disables filtering
            attempt_is_correct: Outcome of the attempt; False demotes 'helpful' tags

        Returns:
            List of dicts with 'id' and 'tag' keys
        """
        parsed = extract_json_from_text(response)
        if isinstance(parsed, dict) and isinstance(parsed.get("bullet_tags"), list):
            bullet_tags = parsed["bullet_tags"]
        else:
            bullet_tags = self._extract_bullet_tags_scan(response)

        return self._filter_bullet_tags(bullet_tags, valid_ids, attempt_is_correct)

    @staticmethod
    def _extract_bullet_tags_scan(response: str) -> List[Any]:
        """Fallback: bracket-depth scan for the bullet_tags array in malformed output."""
        try:
            start_idx = response.find('"bullet_tags"')
            if start_idx == -1:
                return []

            bracket_idx = response.find('[', start_idx)
            if bracket_idx == -1:
                return []

            depth = 0
            end_idx = bracket_idx
            for i in range(bracket_idx, len(response)):
                if response[i] == '[':
                    depth += 1
                elif response[i] == ']':
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break

            return json.loads(response[bracket_idx:end_idx])
        except Exception as e:
            print(f"Warning: Failed to extract bullet tags: {e}")
            return []

    @staticmethod
    def _filter_bullet_tags(
        bullet_tags: List[Any],
        valid_ids: Optional[Set[str]],
        attempt_is_correct: Optional[bool] = None
    ) -> List[Dict[str, str]]:
        """
        Keep only well-formed tags that reference a bullet the reflector saw.

        When the attempt failed, 'helpful' is demoted to 'neutral'. That is enforcement
        of the tag definition, not a thumb on the scale: 'helpful' means the bullet moved
        this attempt toward the correct result, which is impossible in an attempt that was
        incorrect. It never manufactures 'harmful' -- blame stays the model's call. The
        code check exists because the prompt alone did not hold: across 120 reflections in
        run 20260802_220602 the reflector emitted 'harmful' zero times, tagging bullets
        'helpful' on 15 occasions inside attempts that had just failed their assertions.
        """
        allowed_tags = {'helpful', 'harmful', 'neutral'}
        kept, dropped, demoted = [], 0, 0

        for tag in bullet_tags:
            if not isinstance(tag, dict):
                dropped += 1
                continue

            bullet_id = str(tag.get('id') or tag.get('bullet') or '').strip()
            tag_value = str(tag.get('tag', '')).strip().lower()

            if not bullet_id or tag_value not in allowed_tags:
                dropped += 1
                continue
            if valid_ids is not None and bullet_id not in valid_ids:
                dropped += 1
                continue

            if attempt_is_correct is False and tag_value == 'helpful':
                tag_value = 'neutral'
                demoted += 1

            kept.append({'id': bullet_id, 'tag': tag_value})

        if dropped:
            print(f"Warning: dropped {dropped} bullet tag(s) with unknown ids or invalid tags")
        if demoted:
            print(f"Note: demoted {demoted} 'helpful' tag(s) to 'neutral' on a failed attempt")

        return kept