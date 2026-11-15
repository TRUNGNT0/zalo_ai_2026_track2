"""Cache kết quả LLM theo câu, ghi nối tiếp ra jsonl để chạy lại thì tiếp tục được."""
from __future__ import annotations

import json
import os


class ResultCache:
    def __init__(self, path: str):
        self.path = path
        self._data: dict = {}

    def load(self) -> "ResultCache":
        if os.path.exists(self.path):
            for line in open(self.path, encoding="utf-8"):
                d = json.loads(line)
                if d.get("decision") is None and not any(d.get("samples") or []):
                    continue          # "API lỗi / không có mẫu nào" từ lần chạy cũ -> bỏ, để gọi lại
                self._data[d["id"]] = d      # dòng sau ghi đè dòng trước
        return self

    def __len__(self) -> int:
        return len(self._data)

    def get(self, qid):
        return self._data.get(qid)

    def put(self, entry: dict):
        self._data[entry["id"]] = entry
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
