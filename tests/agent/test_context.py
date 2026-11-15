"""Test dựng ngữ cảnh: gợi ý tất định, few-shot, và rulebook dài không làm hỏng."""
from datamate.agent.context import ContextBuilder
from datamate.agent.exemplars import ExemplarBank
from datamate.config import Config
from datamate.ingestion.datasets.loader import DatasetLoader


def build(cfg, split_dir, question, bank=None, business_id="b1"):
    split = DatasetLoader.load_split(split_dir)
    return ContextBuilder(cfg, bank=bank).build(question, "2026-08-27", split.business("b1"), split, business_id)[0]


def test_hints_precede_question_and_follow_profile(cfg, split_dirs):
    ctx = build(cfg, split_dirs[0], "Tuan truoc ca phe ban bao nhieu?")
    assert ctx.index("SCHEMA + PROFILE") < ctx.index("GỢI Ý TỪ MÃ") < ctx.index("CÂU HỎI:")
    assert '"tuần trước" = 2026-08-17 đến 2026-08-23' in ctx
    assert "khớp giá trị 'Ca phe'" in ctx


def test_hints_can_be_disabled(cfg, split_dirs):
    cfg.use_hints = False
    assert "GỢI Ý" not in build(cfg, split_dirs[0], "Tuan truoc ca phe ban bao nhieu?")


def test_long_rulebook_is_truncated_without_crashing(split_dirs):
    """Hồi quy: f-string cảnh báo cắt rulebook từng tham chiếu biến không tồn tại (NameError) khi rulebook quá dài."""
    train, _test, out = split_dirs
    open(f"{train}/rulebook.md", "w", encoding="utf-8").write("Quy tắc. " * 5000)       # 45k ký tự > giới hạn
    cfg = Config(train_dir=train, out_dir=out, backend="gemini")
    ctx = build(cfg, train, "Tong doanh thu?")
    assert "RULEBOOK:" in ctx and len(ctx) < 45000


def test_exemplars_included_and_same_business_excluded(cfg, split_dirs):
    bank = ExemplarBank([{"id": "x", "business_id": "other", "question": "Tong doanh thu thang truoc?",
                          "type": "number", "code": "result_type='number'\nresult=1"},
                         {"id": "y", "business_id": "b1", "question": "Tong doanh thu?", "type": "number",
                          "code": "SAME_BUSINESS"}])
    ctx = build(cfg, split_dirs[0], "Tong doanh thu?", bank=bank)
    assert "VÍ DỤ TƯƠNG TỰ" in ctx and "result=1" in ctx and "SAME_BUSINESS" not in ctx
