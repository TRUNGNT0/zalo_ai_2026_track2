"""Test xuất dữ liệu SFT."""
import json

from evaluation.fakes import GOOD_CODE, FakeBackend
from datamate.agent.context import ContextBuilder
from evaluation.run_eval import run_train
from training.export_sft import export_sft


def test_export_sft_keeps_only_correct_answers(cfg, tmp_path):
    backend = FakeBackend([GOOD_CODE])
    _solver, split, ev, _subs, logs = run_train(cfg, backend)
    path = str(tmp_path / "sft.jsonl")
    n = export_sft(ev, logs, split, ContextBuilder(cfg), path)
    rows = [json.loads(x) for x in open(path, encoding="utf-8")]
    assert n == len(rows) == 4 and rows[0]["messages"][2]["content"].startswith("```python")


def test_build_exemplar_bank_and_reuse_as_few_shot(cfg, tmp_path):
    from datamate.agent.exemplars import ExemplarBank
    from training.build_exemplars import build_exemplar_bank
    _solver, split, ev, _subs, logs = run_train(cfg, FakeBackend([GOOD_CODE]))
    path = str(tmp_path / "bank.jsonl")
    assert build_exemplar_bank(ev, logs, split, path) == 4
    item = ExemplarBank.load(path).items[0]
    assert item["business_id"] == "b1" and "result_type" in item["code"]
