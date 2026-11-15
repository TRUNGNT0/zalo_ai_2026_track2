import json

import pytest

from evaluation.fakes import GOOD_CODE, OTHER_CODE, FakeBackend
from datamate.agent.pipeline import run_inference
from evaluation.run_eval import run_train
from datamate.agent.solver import GenerationBroken, Solver


def test_consensus_answers_and_scores_pa(cfg):
    backend = FakeBackend([GOOD_CODE, GOOD_CODE, GOOD_CODE])
    _solver, _split, ev, subs, _logs = run_train(cfg, backend)
    assert all(s == {"id": s["id"], "type": "number", "answer": 130000} for s in subs)
    assert list(ev.status.unique()) == ["correct"]


def test_abstains_without_consensus(cfg):
    backend = FakeBackend([GOOD_CODE, OTHER_CODE, "```python\nresult_type='number'\nresult=2.0\n```"], batch=True)
    solver = Solver(cfg, backend)
    from datamate.ingestion.datasets.loader import DatasetLoader
    subs, logs = solver.solve_split(DatasetLoader.load_split(cfg.train_dir), "train", limit_q=1)
    assert subs[0]["type"] == "abstain" and logs[0]["intent"] == "llm_no_consensus"


def test_self_repairs_then_reaches_consensus(cfg):
    # lần đầu code lỗi (cột không tồn tại), lần sửa thì đúng
    bad = '```python\nresult_type="number"\nresult=float(tables["sales"]["khong_co"].sum())\n```'
    backend = FakeBackend([bad, GOOD_CODE, GOOD_CODE, GOOD_CODE], batch=False)
    solver = Solver(cfg, backend)
    from datamate.ingestion.datasets.loader import DatasetLoader
    subs, _ = solver.solve_split(DatasetLoader.load_split(cfg.train_dir), "train", limit_q=1)
    assert subs[0]["answer"] == 130000


def test_stops_early_when_model_keeps_failing(cfg, monkeypatch):
    monkeypatch.setattr("app.agent.solver.MAX_FAIL_STREAK", 3)     # tập giả chỉ có 4 câu
    solver = Solver(cfg, FakeBackend([GOOD_CODE], fail=True))
    from datamate.ingestion.datasets.loader import DatasetLoader
    with pytest.raises(GenerationBroken):
        solver.solve_split(DatasetLoader.load_split(cfg.train_dir), "train")


def test_cache_is_reused_without_calling_model_again(cfg):
    backend = FakeBackend([GOOD_CODE])
    run_train(cfg, backend)
    calls = backend.calls
    run_train(cfg, backend)
    assert backend.calls == calls


def test_run_inference_writes_complete_valid_submission(cfg):
    cfg.limit_test = 2
    subs, _logs, problems = run_inference(cfg, FakeBackend([GOOD_CODE]))
    assert problems == []
    lines = [json.loads(x) for x in open(f"{cfg.out_dir}/submission.jsonl", encoding="utf-8")]
    assert [x["id"] for x in lines] == ["q0", "q1", "q2", "q3"]
    assert [x["type"] for x in lines] == ["number", "number", "abstain", "abstain"]


def test_stricter_agreement_for_a_type_forces_abstain(cfg):
    cfg.min_agree_by_type = {"number": 3}
    backend = FakeBackend([GOOD_CODE, GOOD_CODE, OTHER_CODE])       # chỉ 2/3 đồng thuận
    from datamate.ingestion.datasets.loader import DatasetLoader
    subs, logs = Solver(cfg, backend).solve_split(DatasetLoader.load_split(cfg.train_dir), "train", limit_q=1)
    assert subs[0]["type"] == "abstain" and logs[0]["agree"] == 2


def test_stricter_agreement_is_met_when_all_samples_agree(cfg):
    cfg.min_agree_by_type = {"number": 3}
    from datamate.ingestion.datasets.loader import DatasetLoader
    subs, _ = Solver(cfg, FakeBackend([GOOD_CODE])).solve_split(DatasetLoader.load_split(cfg.train_dir), "train", limit_q=1)
    assert subs[0]["answer"] == 130000
