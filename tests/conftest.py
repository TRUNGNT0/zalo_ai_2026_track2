"""Fixture dùng chung. Dữ liệu giả và backend giả nằm ở evaluation/fakes.py (xem lý do ở đó)."""
import json
from pathlib import Path
import pytest

from datamate.config import Config
from evaluation.fakes import write_split

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def train_answers() -> list[dict]:
    # Check both data/train and dataset/train
    path = REPO / "data" / "train" / "answers.jsonl"
    if not path.exists():
        path = REPO / "dataset" / "train" / "answers.jsonl"
    if not path.exists():
        pytest.skip("data/train or dataset/train is not downloaded")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def split_dirs(tmp_path):
    write_split(tmp_path / "train")
    write_split(tmp_path / "test", with_answers=False)
    return str(tmp_path / "train"), str(tmp_path / "test"), str(tmp_path / "out")


@pytest.fixture
def cfg(split_dirs):
    train, test, out = split_dirs
    return Config(train_dir=train, test_dir=test, out_dir=out, backend="gemini", query_engine="pandas", max_biz=None,
                  n_samples=3, agree_min=2)

