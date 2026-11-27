"""Exercise float16 cache and FAISS through build_index with a local deterministic encoder."""

from pathlib import Path
from typing import Any

import pytest
from conftest import GitMinirepo

from reposcout.index import dense
from reposcout.index.cache import IndexConfig
from reposcout.index.repository import configured_index


def test_dense_cache_and_search(
    minirepo_git: GitMinirepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("faiss")

    class Encoder:
        max_seq_length = 4

        def tokenizer(self, texts: list[str], **kwargs: Any) -> dict[str, list[int]]:
            return {"length": [len(text.split()) for text in texts]}

        def encode(self, texts: list[str], **kwargs: Any) -> Any:
            matrix = np.array(
                [[1.0, 4.0 if "rate_band" in s else 0.0] for s in texts], dtype="float32"
            )
            return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    monkeypatch.setattr(dense, "_model", lambda *args: Encoder())
    config = IndexConfig(model_id="fixture", token_budget=4)
    root, base, head = minirepo_git
    first = configured_index(root, base, tmp_path / "cache", dense=True, config=config)
    assert first.manifest["counts"]["capped_vectors"] > 0
    hits = first.dense("rate_band", 3)
    assert len(hits) == 3 and all(hit.source == "dense" for hit in hits)
    assert [hit.span.path for hit in hits] == sorted(hit.span.path for hit in hits)
    assert first.dense("", 3) == []
    second = configured_index(root, head, tmp_path / "cache", dense=True, config=config)
    assert second.manifest["counts"]["embedded"] == 1
    assert second.manifest["counts"]["reused_vectors"] > 0
    assert second.dense("rate_band", 3) == hits
