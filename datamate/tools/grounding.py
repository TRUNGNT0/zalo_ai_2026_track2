"""Khớp từ trong câu hỏi với giá trị thật trong dữ liệu (chịu được thiếu dấu, hoa-thường, lỗi gõ nhẹ).

Tên sản phẩm/khách trong dữ liệu thường lệch so với cách người hỏi viết; việc tìm giá trị khớp là việc có đáp án
duy nhất nên giao cho mã, LLM chỉ nhận kết quả làm gợi ý.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

import pandas as pd

from datamate.text import norm

MAX_NGRAM = 6


@dataclass(frozen=True)
class Hit:
    phrase: str          # cụm trong câu hỏi (đã bỏ dấu)
    value: str           # giá trị gốc trong dữ liệu
    table: str
    column: str
    exact: bool


class ValueIndex:
    """Chỉ mục giá trị chuỗi của các cột có ít giá trị phân biệt (tên hàng, khách, trạng thái, nhóm...)."""

    def __init__(self, tables: dict, max_unique: int = 5000, min_len: int = 3):
        self.by_norm: dict[str, list[tuple[str, str, str]]] = {}
        for table, df in tables.items():
            for col in df.columns:
                s = df[col]
                if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
                    continue
                values = s.dropna().astype(str).unique()
                if len(values) > max_unique:
                    continue
                for v in values:
                    key = norm(v)
                    if len(key) < min_len or re.fullmatch(r"[\d\s.,:/-]+", key):     # bỏ số/ngày dạng chuỗi
                        continue
                    self.by_norm.setdefault(key, []).append((table, str(col), v))
        self._buckets: dict[tuple[str, int], list[str]] = {}
        for key in self.by_norm:
            self._buckets.setdefault((key[0], len(key.split())), []).append(key)

    def ground(self, question: str, max_hits: int = 6, fuzzy_cutoff: float = 0.88) -> list[Hit]:
        words = norm(question).split()
        hits: dict[str, Hit] = {}
        for n in range(min(MAX_NGRAM, len(words)), 0, -1):          # cụm dài trước
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n])
                if len(phrase) < 3 or phrase in hits:
                    continue
                if phrase in self.by_norm:
                    table, col, value = self.by_norm[phrase][0]
                    hits[phrase] = Hit(phrase, value, table, col, True)
                elif len(phrase) >= 5:
                    cands = self._buckets.get((phrase[0], n), [])
                    if 0 < len(cands) <= 3000:
                        close = difflib.get_close_matches(phrase, cands, n=1, cutoff=fuzzy_cutoff)
                        if close:
                            table, col, value = self.by_norm[close[0]][0]
                            hits[phrase] = Hit(phrase, value, table, col, False)
        # bỏ cụm nằm trọn trong một cụm dài hơn đã khớp
        keep = [h for h in hits.values()
                if not any(h.phrase != o.phrase and f" {h.phrase} " in f" {o.phrase} " for o in hits.values())]
        keep.sort(key=lambda h: (-len(h.phrase), h.phrase))
        return keep[:max_hits]


def format_hits(hits: list[Hit]) -> list[str]:
    return [f'- "{h.phrase}" khớp giá trị {h.value!r} (bảng {h.table}, cột {h.column})'
            + ("" if h.exact else ", khớp gần đúng") for h in hits]
