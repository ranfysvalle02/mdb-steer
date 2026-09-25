import json
from pathlib import Path

import pytest

from mdb_steer.pipeline import load_dataset

DATA = Path(__file__).resolve().parent.parent / "data"


def test_bundled_datasets_are_valid_and_disjoint() -> None:
    cal = load_dataset(DATA / "calibration.jsonl")
    bench = load_dataset(DATA / "benchmark.jsonl")
    for items in (cal, bench):
        ids = [i["id"] for i in items]
        assert len(ids) == len(set(ids))
    assert not {i["query"] for i in cal} & {i["query"] for i in bench}, "benchmark must be held out"


def test_load_dataset_rejects_missing_fields(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "x", "query": "q"}) + "\n")
    with pytest.raises(ValueError, match="reference"):
        load_dataset(path)
