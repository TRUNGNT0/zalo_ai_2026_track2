"""Giao diện chung cho các backend LLM."""
from __future__ import annotations

from abc import ABC, abstractmethod


class GenerationError(Exception):
    """Backend không sinh được (mạng, OOM...). Kết quả của câu đó KHÔNG được cache."""


class QuotaExhausted(Exception):
    """Hết quota ngày của API: thử lại vô ích, phải dừng hẳn."""


class LLMBackend(ABC):
    #: True nếu `generate(n>1)` sinh n mẫu trong MỘT lượt (nhanh hơn gọi n lần); False thì solver gọi lần lượt
    supports_batch: bool = False

    @abstractmethod
    def generate(self, system: str, prompt: str, n: int = 1) -> list[str]:
        """Sinh n lời giải độc lập. Lỗi thì ném GenerationError (hoặc QuotaExhausted)."""

    @abstractmethod
    def tag(self) -> str:
        """Nhãn model/tham số, đưa vào khóa cache."""

    def count_tokens(self, text: str) -> int | None:
        """Số token của text (None nếu backend không đếm được)."""
        return None
