# Adapter decisions

- The SDK is pinned at 1.99.9. The adapter uses synchronous, single-choice chat completions, sends the run's cap as `max_completion_tokens`, and sets SDK retries to zero. The eval runner's request-start clock is consulted immediately before invoking the SDK.
- An empty tool list omits the provider's tools and tool_choice fields. A supplied tool list preserves every schema, including when choice is none. Provider-specific options may add JSON fields. Core request controls cannot be overridden through the extra dictionary.
- Missing usage, inconsistent tool-call responses, unsupported finish reasons, and multiple choices fail explicitly. The error carries valid usage from a rejected completion and never includes the provider body, raw exception, credentials, issue text or source.
- Sampling defaults to temperature zero. A supplied seed passes through to the endpoint. Determinism depends on the provider backend. Timeouts are passed to the SDK per request; the eval runner separately enforces its per-instance wall-clock deadline.
- Distilled adapters must set STUDENT_MODEL to the model name actually served by vLLM. A custom teacher endpoint requires its own TEACHER_API_KEY. The Groq key fallback is restricted to the Groq endpoint.
- Unit tests run the real SDK through an in-memory HTTP transport and exercise request shapes, replay, pacing, timeouts, errors and no retries. Tests for graph orchestration can use the scripted FakeChatClient in tests/llm/fake.py.
