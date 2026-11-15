"""Rút gọn rulebook theo câu hỏi để prompt ngắn hơn (cần cho fine-tune trên GPU nhỏ).

Luôn giữ: phần mở đầu, §1 (đơn/doanh thu), §2 (thời gian), §7-§10 (không trả lời, cách ghi, định dạng đáp án, từ ngữ).
Chỉ giữ §3 (xếp hạng), §4 (công nợ), §5 (kho/giá vốn), §6 (theo ngành) khi câu hỏi có từ khóa liên quan.
ĐÂY LÀ ĐÁNH ĐỔI: bỏ nhầm một luật cần thiết thì câu đó sai. Phải đo PA trên holdout có/không dùng hàm này.
"""
from __future__ import annotations

import re
import unicodedata

ALWAYS = {1, 2, 7, 8, 9, 10}
KEYWORDS = {
    3: r"ban chay|nhieu nhat|it nhat|cham nhat|top|xep hang|hang dau|khach|nhan vien|trung binh|loi nhat|bao nhieu (chai|lon|thung|cai|phan)",
    4: r"\bno\b|cong no|thu no|tra no|tuoi no|con no|khach no",
    5: r"nhap|ton\b|ton kho|kho|gia von|\blai\b|loi nhuan|het hang|\be\b|chi phi|gia ban|tang gia|ban duoc|nha cung cap|lo hang|han dung",
    6: r"size|ke don|thuoc|hoat chat|het han|dot giao|vat lieu|nha thuoc|ca phe|\([ml]\)",
}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text.replace("đ", "d").replace("Đ", "D").lower())


def split_sections(rulebook: str) -> list[tuple[int | None, str]]:
    """Tách theo dòng '## N. tiêu đề'; phần trước '##' đầu tiên có số None."""
    parts = re.split(r"(?m)^(?=## )", rulebook)
    out = []
    for part in parts:
        m = re.match(r"## (\d+)\.", part)
        out.append((int(m.group(1)) if m else None, part))
    return out


def select_rulebook(rulebook: str, question: str) -> str:
    qn = _norm(question)
    kept = []
    for number, part in split_sections(rulebook):
        if number is None or number in ALWAYS or re.search(KEYWORDS.get(number, "$^"), qn):
            kept.append(part)
    return "".join(kept)
