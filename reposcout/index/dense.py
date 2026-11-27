"""Content-addressed embeddings persisted as float16 and searched with FAISS."""

import hashlib
import json
import logging
import sqlite3
from collections.abc import Iterable
from functools import lru_cache
from itertools import batched
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def _model(model_id: str, token_budget: int) -> Any:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, trust_remote_code=False)
    model.max_seq_length = min(model.max_seq_length, token_budget)
    return model


def _capped(model: Any, texts: list[str]) -> list[bool]:
    counts = model.tokenizer(
        texts,
        truncation=False,
        padding=False,
        return_length=True,
        verbose=False,
    )["length"]
    return [n > model.max_seq_length for n in counts]


def build_vectors(
    texts: Iterable[str],
    store: sqlite3.Connection,
    destination: Path,
    model_id: str,
    token_budget: int,
    batch_size: int,
) -> dict[str, int]:
    import numpy as np

    store.execute(
        "create table if not exists vectors(key text primary key, vector blob, capped integer)"
    )
    namespace = json.dumps([model_id, token_budget]).encode() + b"\x00"
    stats = {"embedded": 0, "reused_vectors": 0, "capped_vectors": 0, "dimensions": 0}
    with destination.open("wb") as output:
        for batch in batched(texts, batch_size):
            keys = [hashlib.sha256(namespace + text.encode()).hexdigest() for text in batch]
            vectors: dict[str, tuple[bytes, int]] = {}
            missing = {}
            for key, text in zip(keys, batch, strict=True):
                existing = store.execute(
                    "select vector,capped from vectors where key=?", (key,)
                ).fetchone()
                if existing is None:
                    missing[key] = text
                else:
                    vectors[key] = existing
            if missing:
                model = _model(model_id, token_budget)
                inputs = list(missing.values())
                caps = _capped(model, inputs)
                matrix = model.encode(
                    inputs,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    batch_size=batch_size,
                    show_progress_bar=False,
                )
                for key, row, capped in zip(missing, matrix, caps, strict=True):
                    vectors[key] = (np.asarray(row, dtype="<f2").tobytes(), int(capped))
                store.executemany(
                    "insert or ignore into vectors values(?,?,?)",
                    [(key, *vectors[key]) for key in missing],
                )
            stats["embedded"] += len(missing)
            stats["reused_vectors"] += len(keys) - len(missing)
            for key in keys:
                vector, capped = vectors[key]
                output.write(vector)
                stats["dimensions"] = len(vector) // 2
                stats["capped_vectors"] += capped
    return stats


class DenseSearch:
    def __init__(self, path: Path, dimension: int, model_id: str, budget: int, batch: int):
        import faiss
        import numpy as np

        self.model_id = model_id
        self.budget = budget
        self.index = faiss.IndexFlatIP(dimension)
        matrix = np.memmap(path, dtype="<f2", mode="r").reshape(-1, dimension)
        for start in range(0, len(matrix), batch):
            vectors = matrix[start : start + batch].astype("float32")
            faiss.normalize_L2(vectors)
            self.index.add(vectors)

    def scores(self, query: str) -> list[tuple[int, float]]:
        model = _model(self.model_id, self.budget)
        if _capped(model, [query])[0]:
            logging.getLogger(__name__).warning(
                "Dense query capped at %d tokens", model.max_seq_length
            )
        vector = model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        scores, indices = self.index.search(vector, self.index.ntotal)
        return [(int(i) + 1, float(s)) for i, s in zip(indices[0], scores[0], strict=True)]
