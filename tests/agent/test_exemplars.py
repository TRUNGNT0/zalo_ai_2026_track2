"""Test kho ví dụ few-shot."""
from datamate.agent.exemplars import ExemplarBank

ITEMS = [
    {"id": "a", "business_id": "b1", "question": "Tổng doanh thu tháng trước?", "type": "number", "code": "A"},
    {"id": "b", "business_id": "b2", "question": "Khách nào mua nhiều nhất tuần này?", "type": "string", "code": "B"},
    {"id": "c", "business_id": "b2", "question": "Doanh thu tháng này là bao nhiêu", "type": "number", "code": "C"},
]


def test_returns_most_similar_question_first():
    top = ExemplarBank(ITEMS).top_k("doanh thu thang truoc cua cua hang", k=2)
    assert top[0]["id"] == "a" and {t["id"] for t in top} <= {"a", "c"}


def test_excludes_same_business():
    top = ExemplarBank(ITEMS).top_k("Tổng doanh thu tháng trước?", k=3, exclude_business="b1")
    assert all(t["business_id"] != "b1" for t in top)


def test_no_overlap_returns_nothing_and_roundtrip(tmp_path):
    bank = ExemplarBank(ITEMS)
    assert bank.top_k("zzz qqq", k=2) == []
    path = str(tmp_path / "bank.jsonl")
    ExemplarBank.dump(ITEMS, path)
    assert len(ExemplarBank.load(path).items) == 3
