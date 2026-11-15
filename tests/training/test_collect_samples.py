"""Thu dữ liệu SFT: mọi mẫu đúng, loại mẫu xấu, và ghi nhận lý do lỗi."""
import json

from evaluation.fakes import GOOD_CODE, OTHER_CODE, FakeBackend
from datamate.llm.base import LLMBackend
from datamate.agent.context import ContextBuilder
from datamate.agent.solver import Solver
from datamate.ingestion.datasets.loader import DatasetLoader
from evaluation.run_eval import run_train
from training.export_sft import correct_samples, export_sft, is_clean_code

class FixedBackend(LLMBackend):
    """Mỗi lần gọi trả đúng danh sách replies theo thứ tự (cùng kết quả cho mọi câu hỏi)."""
    supports_batch = True

    def __init__(self, replies):
        self.replies = replies

    def tag(self):
        return "fixed"

    def generate(self, system, prompt, n=1):
        return [self.replies[i % len(self.replies)] for i in range(n)]


NOISY_CODE = ('```python\n# Sửa lỗi: KeyError ở lần trước\ndf = tables["sales"].copy()\n'
              'df = df[df["trang_thai"] != "huy"]\nresult_type = "number"\nresult = float(df["thanh_tien"].sum())\n```')


def test_is_clean_code_rejects_error_commentary_and_long_comments():
    assert is_clean_code("df = 1\nresult = 2") and not is_clean_code("# KeyError lần trước\nresult = 1")
    assert not is_clean_code("\n".join(["# c"] * 7) + "\nresult = 1") and not is_clean_code("x" * 3001)


def test_collect_all_generates_every_sample_and_keeps_only_matching_ones(cfg):
    cfg.collect_all, cfg.n_samples, cfg.agree_min = True, 3, 2
    _s, split, ev, _subs, logs = run_train(cfg, FixedBackend([GOOD_CODE, OTHER_CODE, NOISY_CODE]), subset="all")
    assert all(len(lg["samples"]) == 3 for lg in logs)                     # không dừng sớm dù 2 mẫu đã đồng thuận
    kept = list(correct_samples(ev, logs, per_question=2))
    assert len(kept) == 4
    assert all('result = float(df["thanh_tien"].sum())' in c for _r, c in kept)   # chỉ mẫu GOOD (NOISY bị lọc, OTHER sai)


def test_export_sft_all_samples_writes_system_with_helpers(cfg, tmp_path):
    cfg.collect_all, cfg.n_samples, cfg.agree_min = True, 3, 2
    _s, split, ev, _subs, logs = run_train(cfg, FixedBackend([GOOD_CODE, GOOD_CODE, OTHER_CODE]))
    path = str(tmp_path / "sft.jsonl")
    n = export_sft(ev, logs, split, ContextBuilder(cfg), path, all_samples=True, per_question=1)
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    assert n == len(rows) == 4 and "period_range" in rows[0]["messages"][0]["content"]


def test_failure_reasons_are_recorded(cfg):
    bad = '```python\nresult_type = "number"\nresult = float(tables["sales"]["khong_co"].sum())\n```'
    cfg.max_retry = 0
    subs, logs = Solver(cfg, FakeBackend([bad])).solve_split(DatasetLoader.load_split(cfg.train_dir), "train", limit_q=1)
    assert logs[0]["intent"] == "llm_failed" and any("KeyError" in e for e in logs[0]["errors"])


def test_run_name_separates_cache_files(cfg):
    cfg.run_name = "modelA"
    run_train(cfg, FakeBackend([GOOD_CODE]))
    import os
    assert os.path.exists(f"{cfg.out_dir}/logs/llm_cache_train_modelA.jsonl")
