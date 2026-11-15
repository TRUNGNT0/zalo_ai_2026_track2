"""Dữ liệu giả và backend giả dùng cho test và chạy thử offline (không cần GPU hay API).

Cố ý nằm trong gói `evaluation` (không phải tests/ hay conftest.py): gói này luôn import được ở mọi chế độ của pytest,
còn `from conftest import ...` hoặc module đặt cạnh test phụ thuộc sys.path và có thể bị module lạ che mất.
"""
import json
import os

import pandas as pd

from datamate.llm.base import GenerationError, LLMBackend

SALES = pd.DataFrame({
    "ngay": ["2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27"],
    "ten_hang": ["Ca phe", "Tra", "Ca phe", "Tra"],
    "so_luong": [3, 1, 2, 1],
    "thanh_tien": [90000, 20000, 60000, 20000],
    "trang_thai": ["hoan thanh", "hoan thanh", "huy", "hoan thanh"],
})

GOOD_CODE = ('```python\ndf = tables["sales"].copy()\ndf = df[df["trang_thai"] != "huy"]\n'
             'result_type = "number"\nresult = float(df["thanh_tien"].sum())\n```')   # = 130000
OTHER_CODE = '```python\nresult_type = "number"\nresult = 1.0\n```'


class FakeBackend(LLMBackend):
    """Trả lần lượt các văn bản đã chuẩn bị (lặp lại phần tử cuối khi hết)."""

    def __init__(self, replies, batch=True, fail=False):
        self.replies, self.supports_batch, self.fail, self.calls = list(replies), batch, fail, 0

    def tag(self):
        return "fake"

    def generate(self, system, prompt, n=1):
        self.calls += 1
        if self.fail:
            raise GenerationError("hỏng giả lập")
        return [self.replies[min(self.calls - 1 + i, len(self.replies) - 1)] for i in range(n)]


def write_split(root, n_questions=4, with_answers=True):
    os.makedirs(f"{root}/data/b1", exist_ok=True)
    SALES.to_csv(f"{root}/data/b1/sales.csv", index=False)
    open(f"{root}/data/b1/README.md", "w", encoding="utf-8").write("Cửa hàng cà phê.")
    open(f"{root}/rulebook.md", "w", encoding="utf-8").write("Loại đơn hủy. Làm tròn 2 chữ số.")
    qs = [{"id": f"q{i}", "business_id": "b1", "question": "Tổng doanh thu?", "as_of_date": "2026-08-27"}
          for i in range(n_questions)]
    with open(f"{root}/questions.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in qs)
    if with_answers:
        with open(f"{root}/answers.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps({"id": q["id"], "type": "number", "answer": 130000}) + "\n" for q in qs)