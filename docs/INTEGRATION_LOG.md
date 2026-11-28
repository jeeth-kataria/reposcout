# Integration after Workstream D

| Check | Result |
| --- | --- |
| Python 3.12 unit suite and Ruff | 107 tests passed. Lint and formatting passed. |
| Real dev slice, index to leaderboard | BM25, hybrid RRF and hybrid stacktrace scored 3 of 3 each with no errors. |
| Live teacher on pinned repository | Provider connection failed after one attempted call and before any reported tokens. The agent emitted seed candidates and a safe error. |
| Trace format | The live failure trace parsed as version 1, including the final localization. Scripted tests validated a successful tool-call trace. |
| Minirepo localization | Scripted end-to-end agent localized the fixture boundary bug. The public CLI is part of Workstream F. |
| Full-split estimate and teacher ablations | Blocked by the provider connection. A zero-token failed request cannot support a cost projection. |

The integration patch in `reposcout/eval/worker.py` preserves the agent's concrete configuration across the subprocess boundary. Its regression test uses a nondefault seed and exploration limit. No frozen contracts changed.
