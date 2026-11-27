# Commit-bound Python index

`build_index` accepts a checkout, a detached worktree, or a bare Git repository. It resolves the requested commit once, reads regular Python blobs directly from Git, and returns a `RepoIndex`. The checkout may contain uncommitted changes. Every result cites the committed content.

```python
from pathlib import Path

from reposcout.index import build_index

index = build_index(Path("."), "HEAD", Path("cache/index"), dense=False)
try:
    for hit in index.bm25("shipping_label"):
        print(hit.symbol_id, hit.span, hit.score)
finally:
    index.close()
```

The default build includes dense retrieval. Install its dependencies with `uv sync --locked --extra dense`. The first dense build downloads the default CodeSearchNet model. Set `REPOSCOUT_EMBEDDING_MODEL` to an installed Sentence Transformers model directory or a Hub identifier to select another model. `IndexConfig` and `configured_index` in the index package's cache and repository modules expose the file-size limit, module chunk length, embedding token budget, batch size, and snippet length.

The per-commit manifest records parsed and reused file counts, skipped files by reason, vector cache hits, token caps, and build timings. Cache identity uses the Git remote URL when available. Local repositories without an origin use their shared Git directory. Worktrees from a single repository share parsed files and vectors.

Run the benchmark on an existing Git checkout or bare clone.

```sh
uv run python -m reposcout.index.benchmark cache/repos/django__django.git e7fd69d051eaa67cb17f172a39b57253e9cb831a --no-dense
```

Omit `--no-dense` to include the embedding stage. The report includes occupied SQLite pages by stage, shared store bytes, snapshot bytes, vector bytes and the warm reopen time.
