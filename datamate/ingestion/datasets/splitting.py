"""Chia business thành nhóm huấn luyện (fit) và nhóm giữ lại (holdout) để đo khả năng khái quát.

Test dùng business chưa từng thấy, nên mọi thứ rút ra từ train (few-shot, SFT, hiệu chỉnh ngưỡng) chỉ được dùng
nhóm fit; con số PA đáng tin là PA trên nhóm holdout. Chia theo băm ổn định nên chạy lại vẫn ra cùng kết quả.
"""
from __future__ import annotations

import hashlib

class BusinessSplitter:
    @staticmethod
    def holdout_businesses(business_ids, n_holdout: int) -> list:
        """n_holdout business giữ lại (xác định, không phụ thuộc thứ tự đầu vào)."""
        unique = sorted(set(business_ids), key=lambda b: hashlib.md5(str(b).encode()).hexdigest())
        return unique[:n_holdout]
