# Localization agent decisions

- Failed provider requests preserve seed candidates and produce a typed error result. JSON format repair can run once and consumes an LLM call.
- The tests use scripted completions to exercise tool recovery, invalid citations, loop-back and budget exhaustion without network calls.
- In the one-case live teacher smoke run, the first request failed before receiving a Groq response. DNS lookup for the Groq API also failed. The run reported zero provider tokens and supplied no live teacher benchmark measurements.
- The next exploration round retains earlier candidates and discards a saved verifier verdict when that candidate's evidence changes. The one format-only repair accepts a Markdown fence or trailing commas when the original typed candidate payload can be recovered and compared exactly.
- The reported format-failure rate is the proportion of predictions that attempted a format repair. It excludes malformed replies that never reached a repair call.
