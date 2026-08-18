"""
==============================================================================
llm.py
==============================================================================

This file contains the LLM class for the project.

"""
import sys
import time
import random
from datetime import datetime
import openai
from logger import log_llm_call, log_problematic_request


def _make_console_encoding_safe():
    """
    Stop an emoji in a diagnostic from killing the run that emitted it.

    Progress and error messages here and across the loop are decorated with emoji (37 of
    them repo-wide). On Windows, stdout defaults to the ANSI codepage whenever it is not a
    terminal -- redirected to a file, piped, or captured by a task runner -- and printing
    one then raises UnicodeEncodeError. Because the worst offenders sit inside the
    empty-response handler below, the failure lands precisely when a call has already gone
    wrong: the real error is replaced by an encoding traceback and the run dies. This was
    observed for real on a check_model.py run whose second call came back empty.

    Only the error policy is relaxed, not the encoding, so output stays byte-identical
    wherever it already worked and undisplayable characters degrade to '?' instead of
    raising. Done at import because every entry point in this repo imports this module.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding.startswith("utf"):
            continue
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            # Not a reconfigurable text stream (pytest capture, a StringIO, a closed
            # stream). Nothing to do, and nothing worth failing an import over.
            pass


_make_console_encoding_safe()

# Set once per process the first time a call comes back without token counts, so the
# warning is visible without printing on every one of a run's ~1000 calls.
_USAGE_WARNING_EMITTED = False

# Same one-shot treatment for a server that accepts think=False and keeps thinking anyway.
_THINK_WARNING_EMITTED = False

def _consume_stream(stream_obj):
    """
    Accumulate an OpenAI-compatible chat-completion stream into the same pieces the
    non-streamed path reads off a completed response object.

    Streaming is what keeps long generations alive through the Cloudflare tunnel. A
    non-streamed request sends no bytes at all until the model has finished, so any
    generation slower than the proxy's read timeout is cut off with an HTTP 524 even
    though the origin is healthy and still working. Streaming emits a chunk per token,
    which keeps resetting that timer, so wall-clock generation length stops mattering.

    Returns:
        tuple: (content, usage, finish_reason, reasoning)
    """
    content_parts = []
    reasoning_parts = []
    usage = None
    finish_reason = None

    for chunk in stream_obj:
        # The usage-carrying chunk that `stream_options.include_usage` appends at the end
        # has an empty `choices` list, so usage must be read before indexing into choices.
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = chunk_usage

        if not getattr(chunk, "choices", None):
            continue

        choice = chunk.choices[0]
        if getattr(choice, "finish_reason", None):
            finish_reason = choice.finish_reason

        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        if getattr(delta, "content", None):
            content_parts.append(delta.content)
        # Mirrors the non-streamed diagnostic: some local models put chain-of-thought in a
        # side channel and leave `content` blank, which is otherwise indistinguishable
        # from a genuinely empty response.
        for extra_field in ("reasoning_content", "reasoning"):
            extra_value = getattr(delta, extra_field, None)
            if extra_value:
                reasoning_parts.append(extra_value)

    return "".join(content_parts), usage, finish_reason, "".join(reasoning_parts)


def timed_llm_call(client, api_provider, model, prompt, role, call_id, max_tokens=4096, log_dir=None,
                   sleep_seconds=15, retries_on_timeout=1000, attempt=1, use_json_mode=False,
                   retries_on_empty=15, empty_sleep_seconds=3,
                   empty_retry_escalate_after=5, empty_retry_max_temperature=0.7,
                   num_ctx=None, seed=None, stream=True, stream_include_usage=True, think=None,
                   think_mechanism="reasoning_effort"):
    """
    Make a timed LLM call with error handling and retry logic.

    EMPTY RESPONSE HANDLING STRATEGY:
    - An empty/None response is retried up to `retries_on_empty` times (short backoff via
      `empty_sleep_seconds`) before giving up, since it is usually a transient local-model glitch.
    - All calls start at temperature=0.0. If empty responses persist past
      `empty_retry_escalate_after` retries at temperature=0.0 (suggesting a deterministic
      stuck/empty output rather than a transient glitch), temperature is nudged up gradually
      on subsequent retries, capped at `empty_retry_max_temperature`, to try to break out of it.
      This only affects a call that is already failing — every other call stays fully
      deterministic at temperature=0.0.
    - If still empty after retries_on_empty attempts:
        - Training calls (call_id starts with 'train_'): Skip the entire training sample
        - Test calls (call_id starts with 'test_'): Mark as incorrect (return wrong answers)
    - All empty responses (including each retried attempt) are logged to problematic_requests/
      for SambaNova support analysis

    For test calls specifically: Returns "INCORRECT_DUE_TO_EMPTY_RESPONSE" repeated 4 times
    (comma-separated) to handle the 4-question format used in financial NER evaluation.

    Args:
        client: API client
        model: Model name to use
        prompt: Text prompt to send
        role: Role for logging (generator, reflector, curator)
        call_id: Unique identifier for this call (format: {train|test}_{role}_{details})
        max_tokens: Maximum tokens to generate
        log_dir: Directory for detailed logging
        sleep_seconds: Base sleep time between retries (timeouts/rate limits/server errors)
        retries_on_timeout: Maximum number of retries for timeouts/rate limits/server errors
        attempt: Current attempt number (for recursive calls)
        use_json_mode: Whether to use JSON mode for structured output
        retries_on_empty: Maximum number of retries for empty/None responses
        empty_sleep_seconds: Base sleep time between empty-response retries
        empty_retry_escalate_after: Number of empty retries to try at temperature=0.0
            before starting to escalate temperature
        empty_retry_max_temperature: Ceiling for the escalated temperature
        num_ctx: Ollama context window (prompt + completion tokens) for this call. If None,
            it is sized automatically from this call's prompt length and max_tokens, since
            Ollama otherwise falls back to its own low default regardless of max_tokens --
            which can silently cut generation short before the model reaches its final
            answer, producing an empty `content` even though tokens were spent.
            Pin it to a fixed value when comparing two prompt formats: auto-sizing makes
            the served context window a function of prompt length, so a longer rendering
            would be decoded under different conditions and the comparison would measure
            that as much as the change under test.
        seed: Ollama sampling seed. Without one the server RNG is seeded randomly, so runs
            are not reproducible even at temperature=0.0.
        stream: Consume the completion as a token stream rather than waiting for the whole
            response. This is a transport choice, not a sampling one -- the request is
            otherwise byte-identical, so with a fixed seed and temperature the generated
            text is the same either way. It exists because a non-streamed request sends
            nothing until generation finishes, so a slow completion looks indistinguishable
            from a hung origin to a reverse proxy and gets killed with an HTTP 524.
        stream_include_usage: Ask the server for a final usage chunk while streaming. Token
            counts are not otherwise recoverable from a stream, and every cost and fairness
            figure downstream is computed from them, so leave this on. Ignored when
            `stream` is False, where usage comes back on the response object.

    Returns:
        tuple: (response_text, call_info_dict)

    Special return values for empty responses (after retries_on_empty attempts are exhausted):
        - Training: ("INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, ...", call_info)
        - Testing: ("INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, ...", call_info)
    """
    global _USAGE_WARNING_EMITTED, _THINK_WARNING_EMITTED

    start_time = time.time()
    prompt_time = time.time()

    print(f"[{role.upper()}] Starting call {call_id}...")

    empty_attempt = 0
    current_temperature = 0.0

    # Normalized across the streamed and non-streamed paths so the empty-response
    # diagnostic below can read them without knowing which one ran.
    last_usage = None
    last_finish_reason = None
    last_reasoning = ""

    # Ollama's context window is independent of max_tokens and defaults low if unset;
    # size it from this specific call's prompt + requested output so short prompts don't
    # pay for a needlessly large KV cache. Rough estimate: ~3 chars/token, rounded up to
    # the nearest 1024, with a safety buffer and a hard ceiling to protect limited hardware.
    if num_ctx is None:
        estimated_prompt_tokens = len(prompt) // 3
        needed_ctx = estimated_prompt_tokens + max_tokens + 512
        num_ctx = min(32768, ((needed_ctx // 1024) + 1) * 1024)

    while True:
        try:
            # Get client
            active_client = client

            # Prepare API call parameters
            if api_provider == "openai":
                max_tokens_key = "max_completion_tokens"
            else:
                max_tokens_key = "max_tokens"

            options = {"num_ctx": num_ctx}
            if seed is not None:
                options["seed"] = seed

            extra_body = {"options": options}
            if think is not None:
                # `reasoning_effort` is the OpenAI-side spelling of Ollama's reasoning switch,
                # and it is the one this endpoint honours. Ollama's own top-level `think` field
                # belongs to /api/chat and is silently dropped on the OpenAI-compatible route:
                # measured against ollama.caduceus.lol, qwen3.5:2b answered a vowel-counting
                # prompt with 22,923 chars of reasoning and 5,716 tokens in 52.8s under
                # `think`, versus 169 tokens in 1.8s under `reasoning_effort`. The mechanism
                # stays selectable because a server that does not recognise a key ignores it
                # rather than erroring, so the only way to know is to measure -- see
                # check_model.py, and the reasoning_chars check below.
                if think_mechanism in ("think", "both"):
                    extra_body["think"] = think
                if think_mechanism in ("reasoning_effort", "both"):
                    extra_body["reasoning_effort"] = "none" if think is False else "medium"

            api_params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": current_temperature,
                max_tokens_key: max_tokens,
                "extra_body": extra_body
            }

            # Add JSON mode if requested
            if use_json_mode:
                api_params["response_format"] = {"type": "json_object"}

            if stream:
                api_params["stream"] = True
                if stream_include_usage:
                    api_params["stream_options"] = {"include_usage": True}

            # Cleared per attempt so a previous attempt's metadata cannot leak into the
            # diagnostic for this one.
            last_usage = None
            last_finish_reason = None
            last_reasoning = ""

            call_start = time.time()
            raw_response = active_client.chat.completions.create(**api_params)

            if stream:
                # Consumed inside the try block on purpose: with streaming, transport
                # errors surface while iterating rather than at create(), and they must
                # still reach the retry logic below.
                response_content, last_usage, last_finish_reason, last_reasoning = _consume_stream(raw_response)
            else:
                if not raw_response or not raw_response.choices or len(raw_response.choices) == 0:
                    raise Exception("Empty response from API")
                choice = raw_response.choices[0]
                response_content = choice.message.content
                last_usage = getattr(raw_response, "usage", None)
                last_finish_reason = getattr(choice, "finish_reason", None)
                last_reasoning = next(
                    (getattr(choice.message, f, None) or "" for f in ("reasoning_content", "reasoning")
                     if getattr(choice.message, f, None)),
                    "",
                )

            # Measured after the stream is drained, so this stays time-to-last-token in
            # both modes and latency remains comparable across runs.
            call_end = time.time()

            response_time = time.time()
            total_time = response_time - start_time

            if response_content is None or response_content.strip() == "":
                raise Exception("Empty response from API - content is None or blank")

            call_info = {
                "role": role,
                "call_id": call_id,
                "model": model,
                "prompt": prompt,
                "response": response_content,
                "prompt_time": prompt_time - start_time,
                "response_time": response_time - prompt_time,
                "total_time": total_time,
                "call_time": call_end - call_start,
                "prompt_length": len(prompt),
                "response_length": len(response_content),
                "prompt_num_tokens": getattr(last_usage, "prompt_tokens", None),
                "response_num_tokens": getattr(last_usage, "completion_tokens", None),
                "temperature_used": current_temperature,
                "num_ctx": num_ctx,
                "seed": seed,
                "stream": stream,
                "think": think,
                "think_mechanism": think_mechanism if think is not None else None,
                # Recorded even when thinking is off, so a server that ignored the flag
                # shows up in the logs as a run whose reasoning field was never empty.
                "reasoning_chars": len(last_reasoning),
                # Non-zero means this call stopped being greedy: the empty-response
                # handler escalated temperature to break a stuck generation.
                "empty_retries": empty_attempt,
            }
            
            if call_info["response_num_tokens"] is None and not _USAGE_WARNING_EMITTED:
                _USAGE_WARNING_EMITTED = True
                print(f"[{role.upper()}] WARNING: no token usage returned for {call_id}. "
                      f"Every cost and fairness number downstream is computed from these "
                      f"counts, so this run cannot be aggregated. If streaming is on, the "
                      f"server is likely ignoring stream_options.include_usage -- rerun with "
                      f"stream_include_usage disabled, or with streaming off, before "
                      f"spending a full run.")

            if think is False and last_reasoning and not _THINK_WARNING_EMITTED:
                _THINK_WARNING_EMITTED = True
                print(f"[{role.upper()}] WARNING: think=False was requested but {call_id} still "
                      f"returned {len(last_reasoning)} chars of reasoning. The server is ignoring "
                      f"the flag (mechanism={think_mechanism}), so this model is still thinking: "
                      f"it will burn output budget before answering and may return nothing at "
                      f"all on harder prompts. Check the Ollama version, or use a non-reasoning "
                      f"model.")

            print(f"[{role.upper()}] Call {call_id} completed in {total_time:.2f}s")

            if log_dir:
                log_llm_call(log_dir, call_info)

            return response_content, call_info
            
        except Exception as e:
            # Check for both timeout and rate limit errors
            is_timeout = any(k in str(e).lower() for k in ["timeout", "timed out", "connection"])
            is_rate_limit = any(k in str(e).lower() for k in ["rate limit", "429", "rate_limit_exceeded"])
            is_empty_response = "empty response" in str(e).lower() or "api returned none content" in str(e).lower()
            
            # Check for server errors (500, 502, 503, etc.) that should be retried
            is_server_error = False
            if hasattr(e, 'response'):
                try:
                    status_code = getattr(e.response, 'status_code', None)
                    if status_code and status_code >= 500:
                        is_server_error = True
                        print(f"[{role.upper()}] Server error detected: HTTP {status_code}")
                except:
                    pass
            
            # Also check for 500 errors in the error message itself
            if any(k in str(e).lower() for k in ["500 internal server error", "internal server error", "502 bad gateway", "503 service unavailable"]):
                is_server_error = True
                print(f"[{role.upper()}] Server error detected in message: {str(e)[:100]}...")
            
            # Also check for specific OpenAI exceptions
            if hasattr(openai, 'RateLimitError') and isinstance(e, openai.RateLimitError):
                is_rate_limit = True
            
            # Check for OpenAI InternalServerError
            if hasattr(openai, 'InternalServerError') and isinstance(e, openai.InternalServerError):
                is_server_error = True
                print(f"[{role.upper()}] OpenAI InternalServerError detected")
            
            # Debug empty response issues
            if is_empty_response:
                print(f"\n🚨 DEBUG: Empty response detected for {call_id}")
                print(f"📝 Exception type: {type(e).__name__}")
                print(f"📝 Exception message: {str(e)}")
                print(f"📝 Using JSON mode: {use_json_mode}")
                print(f"📝 Model: {model}")
                print(f"📝 Prompt length: {len(prompt)}")
                print(f"📝 Full exception details: {repr(e)}")
                # Diagnose whether this is a "blank content but tokens were spent" case,
                # which suggests the model burned its budget on hidden reasoning/thinking
                # rather than the final answer.
                # Reads the normalized values captured above rather than the raw response,
                # so this works the same whether the call was streamed or not.
                completion_tokens = getattr(last_usage, "completion_tokens", None)
                print(f"📝 Streamed: {stream}")
                print(f"📝 Finish reason: {last_finish_reason}, completion_tokens: {completion_tokens}")
                if last_reasoning:
                    print(f"📝 Non-empty reasoning field found ({len(last_reasoning)} chars) — "
                          f"model likely spent its output budget on hidden reasoning "
                          f"instead of the final answer.")
                if hasattr(e, 'response'):
                    print(f"📝 Raw response object: {e.response}")
                    if hasattr(e.response, 'text'):
                        print(f"📝 Raw response text: {e.response.text}")
                    if hasattr(e.response, 'content'):
                        print(f"📝 Raw response content: {e.response.content}")
                print("-" * 60)
                
                # Log problematic requests for SambaNova support
                log_problematic_request(call_id, prompt, model, api_params, e, log_dir)
            
            # For empty responses, retry a bounded number of times before giving up —
            # empty completions from small local models are often transient flukes.
            if is_empty_response:
                if empty_attempt < retries_on_empty:
                    empty_attempt += 1

                    # If temperature=0.0 keeps producing empty output, it may be a
                    # deterministic dead-end rather than a transient glitch — nudge
                    # temperature up on later retries to try to break out of it.
                    if empty_attempt > empty_retry_escalate_after:
                        current_temperature = min(empty_retry_max_temperature,
                                                   round(current_temperature + 0.15, 2))
                        print(f"[{role.upper()}] Call {call_id} still empty after "
                              f"{empty_attempt - 1} retries at temperature=0.0, escalating to "
                              f"temperature={current_temperature} for the next attempt")

                    jitter = random.uniform(0.5, 1.5)  # Add jitter to avoid thundering herd
                    sleep_time = empty_sleep_seconds * jitter
                    print(f"[{role.upper()}] Call {call_id} returned empty response, sleeping "
                          f"{sleep_time:.1f}s then retrying ({empty_attempt}/{retries_on_empty})...")
                    time.sleep(sleep_time)
                    continue

                # Retry budget exhausted — fall back to marking as incorrect
                # Log the problematic request for SambaNova support
                log_problematic_request(call_id, prompt, model, api_params, e, log_dir)

                # Check if this is a training or test call to decide behavior
                if call_id.startswith('train_'):
                    # In training: Mark as incorrect answer (same as testing)
                    print(f"[{role.upper()}] 🚨 Empty response in training - marking as INCORRECT for {call_id}")
                    error_time = time.time()
                    call_info = {
                        "role": role,
                        "call_id": call_id,
                        "model": model,
                        "prompt": prompt,
                        "error": "TRAINING_INCORRECT: " + str(e),
                        "total_time": error_time - start_time,
                        "prompt_length": len(prompt),
                        "response_length": 0,
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
                        "datetime": datetime.now().isoformat(),
                        "training_marked_incorrect_due_to_empty_response": True
                    }
                    if log_dir:
                        log_llm_call(log_dir, call_info)
                    
                    # Return a response that will be marked as incorrect
                    # For the 4-question format, we return 4 wrong answers
                    incorrect_response = "INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE"
                    return incorrect_response, call_info
                
                elif call_id.startswith('test_'):
                    # In testing: Treat as incorrect answer
                    print(f"[{role.upper()}] 🚨 Empty response in testing - marking as INCORRECT for {call_id}")
                    error_time = time.time()
                    call_info = {
                        "role": role,
                        "call_id": call_id,
                        "model": model,
                        "prompt": prompt,
                        "error": "TEST_INCORRECT: " + str(e),
                        "total_time": error_time - start_time,
                        "prompt_length": len(prompt),
                        "response_length": 0,
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
                        "datetime": datetime.now().isoformat(),
                        "test_marked_incorrect_due_to_empty_response": True
                    }
                    if log_dir:
                        log_llm_call(log_dir, call_info)
                    
                    # Return a response that will be marked as incorrect
                    # For the 4-question format, we return 4 wrong answers
                    incorrect_response = "INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE"
                    return incorrect_response, call_info
            
            # Retry logic for timeouts, rate limits, and server errors
            if (is_timeout or is_rate_limit or is_server_error) and attempt < retries_on_timeout:
                attempt += 1
                if is_rate_limit:
                    error_type = "rate limited"
                    base_sleep = sleep_seconds * 2
                elif is_server_error:
                    error_type = "server error (500+)"
                    base_sleep = sleep_seconds * 1.5  # Moderate delay for server errors
                elif is_empty_response:
                    error_type = "returned empty response"
                    base_sleep = sleep_seconds
                else:
                    error_type = "timed out"
                    base_sleep = sleep_seconds
                jitter = random.uniform(0.5, 1.5)  # Add jitter to avoid thundering herd
                sleep_time = base_sleep * jitter
                print(f"[{role.upper()}] Call {call_id} {error_type}, sleeping {sleep_time:.1f}s then retrying "
                      f"({attempt}/{retries_on_timeout})...")
                time.sleep(sleep_time)
                continue
            
            error_time = time.time()
            call_info = {
                "role": role,
                "call_id": call_id,
                "model": model,
                "prompt": prompt,
                "error": str(e),
                "total_time": error_time - start_time,
                "prompt_length": len(prompt),
                "attempt": attempt,
            }
            
            print(f"[{role.upper()}] Call {call_id} failed after {error_time - start_time:.2f}s: {e}")
            
            if log_dir:
                log_llm_call(log_dir, call_info)
            
            raise e
