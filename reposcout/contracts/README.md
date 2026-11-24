# Contract guide

Read the interfaces in this order before building a consumer.

| File | What to check |
| --- | --- |
| [models.py](models.py) | How paths, inclusive spans, nested symbol IDs, ranked predictions, gold labels, and failed runs serialize |
| [index.py](index.py) and [parser.py](parser.py) | How callers request symbols, searches, file reads, and the index factory |
| [tools.py](tools.py) | Which seven tools exist and how output carries citations, truncation, and errors |
| [localizer.py](localizer.py) | How budgets and ablations enter a baseline or an agent |
| [llm.py](llm.py) | How messages, tool-call IDs, raw arguments, completion reasons, and usage travel between turns |
| [registry.py](registry.py) | How each workstream registers its own localizer factory |

[Contract decisions](NOTES.md) describe the conventions that later workstreams depend on. Run `uv run pytest tests/contracts` to exercise the validators.
