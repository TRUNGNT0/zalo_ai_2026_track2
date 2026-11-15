"""Chuẩn hóa văn bản tiếng Việt."""
from __future__ import annotations

import re
import unicodedata


def norm(s) -> str:
    """Bỏ dấu tiếng Việt, hạ chữ thường, gọn khoảng trắng. Câu hỏi thường viết không dấu/viết tắt."""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("đ", "d").replace("Đ", "D").lower()
    return re.sub(r"\s+", " ", s).strip()


def colnorm(c) -> str:
    """Tên cột về dạng snake_case không dấu, để so khớp bằng regex."""
    return re.sub(r"[^a-z0-9]+", "_", norm(c)).strip("_")
