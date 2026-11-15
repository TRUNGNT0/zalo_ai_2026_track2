"""Hàm trợ giúp phải dùng được TRONG code do model sinh ra, và prompt phải nêu chúng."""
from evaluation.fakes import FakeBackend
from datamate.agent.context import ContextBuilder
from datamate.agent.prompts import build_system, prompt_hash
from datamate.agent.solver import Solver
from datamate.config import Config
from datamate.ingestion.datasets.loader import DatasetLoader

HELPER_CODE = ('```python\ndf = tables["sales"].copy()\nstart, end = period_range("this_month", AS_OF)\n'
               'mask = in_range(df["ngay"], start, end) & df["trang_thai"].map(counts_as_revenue)\n'
               'result_type = "number"\nresult = float(to_number(df.loc[mask, "thanh_tien"]).sum())\n```')


def test_helpers_available_inside_generated_code(cfg):
    split = DatasetLoader.load_split(cfg.train_dir)
    subs, _ = Solver(cfg, FakeBackend([HELPER_CODE])).solve_split(split, "train", limit_q=1)
    assert subs[0]["answer"] == 130000          # 90.000 + 20.000 + 20.000: cả 4 dòng đều trong tháng 8, bỏ đơn "huy" (60.000)


def test_system_prompt_mentions_helpers_only_when_enabled(cfg):
    assert "period_range" in build_system(cfg)
    cfg.use_helpers = False
    assert "period_range" not in build_system(cfg)


def test_prompt_hash_changes_with_helpers_and_collect_mode(cfg):
    base = prompt_hash(cfg, "t", "ctx")
    cfg.use_helpers = False
    assert prompt_hash(cfg, "t", "ctx") != base
    cfg.use_helpers = True
    cfg.collect_all = True
    assert prompt_hash(cfg, "t", "ctx") != base


def test_rulebook_select_shortens_context(cfg, split_dirs):
    train = split_dirs[0]
    open(f"{train}/rulebook.md", "w", encoding="utf-8").write(
        "mở đầu\n## 1. Đơn\nA\n## 2. Thời gian\nB\n## 3. Xếp hạng\n" + "X" * 500 + "\n## 9. Đáp án\nZ\n")
    split = DatasetLoader.load_split(train)
    full = ContextBuilder(cfg).build("Tong doanh thu?", "2026-08-27", split.business("b1"), split, "b1")[0]
    cfg.rulebook_select = True
    short = ContextBuilder(cfg).build("Tong doanh thu?", "2026-08-27", split.business("b1"), split, "b1")[0]
    assert "XXXX" in full and "XXXX" not in short and "## 9. Đáp án" in short
