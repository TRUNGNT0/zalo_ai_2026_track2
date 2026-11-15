"""Backend Gemini (chỉ để phát triển/so sánh; vòng nộp cuối phải offline)."""
from __future__ import annotations

import os
import time
import json
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from datamate.config import Config
from .base import GenerationError, LLMBackend, QuotaExhausted
from .prompts import QUESTION_REWRITE_SYSTEM_PROMPT

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def _is_daily_quota(e: Exception) -> bool:
    s = str(e)
    return "RESOURCE_EXHAUSTED" in s and ("PerDay" in s or "per day" in s.lower())


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_APIKEY")
    if not key:
        load_dotenv(dotenv_path=ENV_FILE)
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_APIKEY")
    if key:
        return key
    try:                                            # Kaggle Secrets
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GEMINI_API_KEY")
    except Exception as e:
        raise RuntimeError("Thiếu khóa: đặt GEMINI_API_KEY (hoặc GEMINI_APIKEY) trong môi trường/.env, "
                           "hoặc Kaggle Secret GEMINI_API_KEY") from e


class GeminiBackend(LLMBackend):
    supports_batch = False                          # gọi lần lượt để còn dừng sớm khi đã đồng thuận (tiết kiệm quota)

    def __init__(self, cfg: Config, api_key: str | None = None):
        from google import genai
        from google.genai import types
        self._types = types
        self._model, self._thinking = cfg.gemini_model, cfg.thinking
        self._client = genai.Client(api_key=api_key or _api_key())

    def tag(self) -> str:
        return f"{self._model}|{self._thinking}"

    def _one(self, system: str, prompt: str, response_json_schema=None) -> str:
        last = None
        for i in range(4):                          # backoff khi bị rate limit / lỗi mạng
            try:
                r = self._client.models.generate_content(
                    model=self._model, contents=prompt,
                    config=self._types.GenerateContentConfig(
                        system_instruction=system,
                        response_mime_type="application/json" if response_json_schema else None,
                        response_json_schema=response_json_schema,
                        thinking_config=self._types.ThinkingConfig(thinking_level=self._thinking)))
                return r.text or ""
            except Exception as e:
                if _is_daily_quota(e):
                    raise QuotaExhausted("Hết quota request/ngày của Gemini. Đổi model, bật billing, hoặc chờ reset. "
                                         + str(e)[:200]) from e
                last = e
                time.sleep(2 * 2 ** i)
        raise GenerationError(f"Gemini lỗi sau 4 lần thử: {last!r}"[:300])

    def generate(self, system: str, prompt: str, n: int = 1) -> list[str]:
        return [self._one(system, prompt) for _ in range(n)]

    def generate_json(self, system: str, prompt: str, schema: dict) -> str:
        return self._one(system, prompt, response_json_schema=schema)

    def rewrite_question(self, question: str, *, protected_terms: Sequence[str] = ()) -> str:
        """Chuẩn hóa tiếng Việt qua API, không trả lời câu hỏi hoặc quy đổi thời gian.

        protected_terms: tên riêng/mã cần giữ nguyên chính tả nếu xuất hiện trong câu.
        Trả câu đã chuẩn hóa; phản hồi sai định dạng hoặc đổi số/thuật ngữ được bảo vệ
        gây GenerationError. Lỗi API/quota dùng cùng cơ chế với generate().
        """
        if not isinstance(question, str):
            raise TypeError("question phải là chuỗi")
        if not question.strip():
            raise ValueError("question không được rỗng")
        if isinstance(protected_terms, str) or any(
            not isinstance(term, str) or not term.strip() for term in protected_terms
        ):
            raise ValueError("protected_terms phải là danh sách các chuỗi không rỗng")
        terms = sorted({term for term in protected_terms if term in question})
        prompt = json.dumps({"question": question, "protected_terms": terms}, ensure_ascii=False)
        responses = self.generate(QUESTION_REWRITE_SYSTEM_PROMPT, prompt, n=1)
        try:
            result = json.loads(responses[0])
        except (IndexError, TypeError, json.JSONDecodeError) as e:
            raise GenerationError("Gemini rewrite không trả JSON hợp lệ") from e
        if (not isinstance(result, dict) or set(result) != {"question"}
                or not isinstance(result["question"], str) or not result["question"].strip()):
            raise GenerationError("Gemini rewrite phải trả một trường question không rỗng")
        rewritten = result["question"].strip()
        # Giữ nguyên chữ số, định dạng ngày/giờ, số thập phân và phần trăm.
        number_pattern = r"\d+(?:[.,:/-]\d+)*%?"
        if Counter(re.findall(number_pattern, question)) != Counter(re.findall(number_pattern, rewritten)):
            raise GenerationError("Gemini rewrite đã thay đổi số liệu hoặc mốc ngày/giờ")
        if any(question.count(term) != rewritten.count(term) for term in terms):
            raise GenerationError("Gemini rewrite đã thay đổi tên/mã cần giữ nguyên")
        return rewritten


# Compatibility client for manifest generation.
from dataclasses import dataclass, field
from google import genai
from google.genai import types

GEMINI_MODEL = "gemini-3.5-flash-lite"

@dataclass(frozen=True, slots=True)
class GeminiLLM:
    client: genai.Client
    model: str = GEMINI_MODEL
    min_request_interval_seconds: float = 0.0
    _last_request_at: float = field(default=0.0, init=False, repr=False, compare=False)

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        response_json_schema: dict[str, object] | None = None,
    ) -> str:
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")
        if self.min_request_interval_seconds < 0:
            raise ValueError("Request interval must not be negative.")

        now = time.monotonic()
        wait_seconds = self.min_request_interval_seconds - (
            now - self._last_request_at
        )
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        object.__setattr__(self, "_last_request_at", time.monotonic())

        config: types.GenerateContentConfig | None = None
        if system_instruction is not None or response_json_schema is not None:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type=(
                    "application/json" if response_json_schema is not None else None
                ),
                response_json_schema=response_json_schema,
                temperature=0,
            )

        if config is None:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
        else:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        if response.text is None:
            raise RuntimeError("Gemini returned no text content.")
        return response.text

    @classmethod
    def from_env(
        cls, *, min_request_interval_seconds: float = 0.0
    ) -> "GeminiLLM":
        load_dotenv(dotenv_path=ENV_FILE)
        api_key = os.getenv("GEMINI_APIKEY")
        if not api_key:
            raise RuntimeError(
                f"GEMINI_APIKEY is missing or empty. Set it in {ENV_FILE}."
            )

        return cls(
            client=genai.Client(api_key=api_key),
            min_request_interval_seconds=min_request_interval_seconds,
        )
