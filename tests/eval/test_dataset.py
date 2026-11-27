import io
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from reposcout.eval import dataset


def page(number: int = 1) -> dict:
    return {
        "row_idx": number,
        "truncated_cells": [],
        "row": {
            "instance_id": f"bug-{number}",
            "base_commit": "a" * 40,
            "repo": "demo/p",
            "problem_statement": "find the issue",
            "patch": "source changes",
            "test_patch": "private label",
        },
    }


def test_paged_cache_withheld_labels_and_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def read(url: str, timeout: float) -> io.BytesIO:
        params = parse_qs(urlsplit(url).query)
        start = int(params["offset"][0])
        assert params["config"] == ["default"] and params["split"] == ["dev"]
        rows = [page(i) for i in range(start, min(start + 10, 23))]
        return io.BytesIO(json.dumps({"num_rows_total": 23, "rows": rows}).encode())

    monkeypatch.setattr(dataset, "urlopen", read)
    rows = dataset.load_split("dev", tmp_path)
    assert len(rows) == 23 and rows[0].patch == "source changes"
    assert rows[0].issue().gold_files == [] and "patch" not in rows[0].issue().model_dump()
    monkeypatch.setattr(dataset, "urlopen", lambda *a, **k: pytest.fail("cache missed"))
    assert dataset.load_split("dev", tmp_path) == rows


def test_partial_patch_is_rejected_without_caching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = page(0)
    row["truncated_cells"] = ["patch"]
    monkeypatch.setattr(
        dataset,
        "urlopen",
        lambda *a, **k: io.BytesIO(json.dumps({"num_rows_total": 23, "rows": [row]}).encode()),
    )
    with pytest.raises(ValueError, match="truncated"):
        dataset.load_split("dev", tmp_path)
    assert not (tmp_path / "dev.jsonl").exists()
