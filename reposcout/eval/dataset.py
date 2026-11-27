"""Read SWE-bench Lite rows without downloading benchmark environments."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal
from urllib.parse import urlencode
from urllib.request import urlopen

from pydantic import BaseModel

from reposcout.contracts.models import Instance

DATASET = "princeton-nlp/SWE-bench_Lite"
SPLIT_SIZES = {"dev": 23, "test": 300}
type Split = Literal["dev", "test"]


class DatasetRow(BaseModel):
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str

    def issue(self) -> Instance:
        # The Localizer receives this view with all evaluation labels withheld.
        return Instance.model_validate(self.model_dump(exclude={"patch"}))


def load_split(
    split: Split, data_dir: Path = Path("data"), *, page_size: int = 10, timeout_s: float = 60
) -> list[DatasetRow]:
    if split not in SPLIT_SIZES:
        raise ValueError(f"unknown SWE-bench Lite split {split!r}")
    if not 1 <= page_size <= 100:
        raise ValueError("Hugging Face page_size must be between 1 and 100")
    path = data_dir / f"{split}.jsonl"
    if path.exists():
        rows = [DatasetRow.model_validate_json(line) for line in path.read_text().splitlines()]
    else:
        rows = []
        while len(rows) < SPLIT_SIZES[split]:
            params = urlencode(
                {
                    "dataset": DATASET,
                    "config": "default",
                    "split": split,
                    "offset": len(rows),
                    "length": page_size,
                }
            )
            with urlopen(
                f"https://datasets-server.huggingface.co/rows?{params}", timeout=timeout_s
            ) as response:
                page = json.load(response)
            if page["num_rows_total"] != SPLIT_SIZES[split] or not page["rows"]:
                raise ValueError("SWE-bench Lite split size changed or the response was incomplete")
            for item in page["rows"]:
                if item["row_idx"] != len(rows) or set(item.get("truncated_cells", [])) & set(
                    DatasetRow.model_fields
                ):
                    raise ValueError(f"out-of-order or truncated row at offset {len(rows)}")
                rows.append(DatasetRow.model_validate(item["row"]))
    ids = {r.instance_id for r in rows}
    if len(rows) != SPLIT_SIZES[split] or len(ids) != len(rows):
        raise ValueError(f"expected {SPLIT_SIZES[split]} unique rows for {split}")
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", dir=data_dir, delete=False) as output:
            for row in rows:
                output.write(row.model_dump_json() + "\n")
        Path(output.name).replace(path)
    return rows
