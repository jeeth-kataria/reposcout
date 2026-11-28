# Agent trace format, version 1

Each completed agent invocation writes one UTF-8 JSON Lines file under `traces/<run_name>/<instance_id>.jsonl`. Set `REPOSCOUT_TRACE_ROOT` to relocate the directory. Rows appear in execution order and carry `event`, `node`, and monotonic `elapsed_s` relative to the invocation. The writer replaces the final path atomically after closing the trace.

| Event | Payload |
| --- | --- |
| `instance` | `schema_version`, `instance_id`, `repo`, `base_commit`, and resolved `config` |
| `llm_request` | One-based `call`, `model`, complete OpenAI-format `messages`, function `tools`, `tool_choice`, `temperature`, `seed`, and `max_tokens` |
| `llm_response` | Matching `call`, OpenAI-format assistant `message`, token `usage`, and `finish_reason` |
| `llm_error` | Matching `call` and a safe `error` message |
| `tool_result` | Tool `name`, raw JSON `arguments`, nullable `tool_call_id`, `latency_s`, `text`, `citations`, `truncated`, nullable `error` |
| `node_end` | Node `latency_s` and current run `status` |
| `localization` | Final serialized `Localization` |
| Other events | Node diagnostics, including retrieval evidence, format repair, and explicit size limits |

Pair each successful request and response by `call` when producing supervised examples. Append the response message as the assistant target. Use the request's tool schemas and message history as context. Final status and format-repair counts are in `localization.stats`. Traces may contain issue text and source snippets and belong in the ignored trace directory.
