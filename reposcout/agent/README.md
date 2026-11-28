# Localization agent

The registered teacher localizers are `agent`, `agent_no_graph`, and `agent_no_verifier`. The student uses `agent_student`. The graph rewrites the issue, retrieves seed hits, explores through read-only tools, verifies citations, and ranks the available evidence within the configured budgets.

```sh
uv sync --extra dense
uv run --extra dense python -m reposcout.eval run --localizer agent --split dev --limit 3
```

Use the [provider setup](../llm/README.md) for Groq or vLLM configuration. See the [trace format](TRACE_FORMAT.md) for how to read a completed run.
