# Provider adapter

`OpenAIChatClient.from_env("teacher")` creates the Groq teacher. `OpenAIChatClient.from_env("student")` connects to a vLLM-compatible endpoint. Pass `client.model` to the frozen chat interface. Call `client.close()` when finished.

| Setting | Teacher | Student |
| --- | --- | --- |
| Endpoint | `TEACHER_BASE_URL`, defaults to `https://api.groq.com/openai/v1` | `STUDENT_BASE_URL`, defaults to `http://127.0.0.1:8000/v1` |
| Model | `TEACHER_MODEL`, defaults to `openai/gpt-oss-20b` | `STUDENT_MODEL`, defaults to `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Credential | `TEACHER_API_KEY` or `GROQ_API_KEY` for the default Groq endpoint | `STUDENT_API_KEY` for authenticated servers; a placeholder is used by unauthenticated local vLLM |

The adapter returns the assistant turn, original tool arguments, finish reason and usage. Keep the tool definitions in the request and set `tool_choice="none"` to force an answer after a tool call. Pass provider-specific settings through `extra`, for example `{"reasoning_effort": "low"}` for the Groq teacher. [Groq model reference](https://console.groq.com/docs/model/openai/gpt-oss-20b)

`LLMError.kind` identifies the failure and `status_code` carries the HTTP status when available. Record its safe diagnostic in the run stats. Include `error.usage` when supplied so a rejected or truncated completion counts toward the run budget.
