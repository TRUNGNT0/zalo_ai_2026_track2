"""Kho ví dụ (câu hỏi, lời giải đã được kiểm đúng) từ tập train và truy xuất ví dụ tương tự (few-shot).

Truy xuất bằng TF-IDF trên từ đơn + cặp từ của câu hỏi đã bỏ dấu, viết thuần Python (không cần thư viện ngoài).
Khi đánh giá phải loại ví dụ cùng business với câu đang hỏi để mô phỏng đúng tình huống test (business chưa thấy).
"""
from __future__ import annotations

import json
import math
from collections import Counter

from datamate.text import norm


def _terms(text: str) -> list[str]:
    words = norm(text).split()
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


class ExemplarBank:
    def __init__(self, items: list[dict]):
        """items: {"id", "business_id", "question", "type", "code"}."""
        self.items = items
        docs = [Counter(_terms(it["question"])) for it in items]
        df: Counter = Counter()
        for d in docs:
            df.update(d.keys())
        n = len(items)
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
        self._vectors = [self._weigh(d) for d in docs]

    def _weigh(self, counts: Counter) -> dict[str, float]:
        weights = {t: c * self._idf.get(t, 1.0) for t, c in counts.items()}
        length = math.sqrt(sum(v * v for v in weights.values())) or 1.0
        return {t: v / length for t, v in weights.items()}

    @classmethod
    def load(cls, path: str) -> "ExemplarBank":
        with open(path, encoding="utf-8") as f:
            return cls([json.loads(line) for line in f if line.strip()])

    @staticmethod
    def dump(items: list[dict], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    def top_k(self, question: str, k: int = 2, exclude_business=None) -> list[dict]:
        q = self._weigh(Counter(_terms(question)))
        scored = []
        for it, weights in zip(self.items, self._vectors):
            if exclude_business is not None and it.get("business_id") == exclude_business:
                continue
            score = sum(w * weights.get(t, 0.0) for t, w in q.items())
            if score > 0:
                scored.append((score, it))
        scored.sort(key=lambda x: -x[0])
        return [it for _s, it in scored[:k]]
