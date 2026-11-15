"""Dựng ngữ cảnh cho một câu hỏi: rulebook + README + profile cột, tự rút gọn cho vừa ngân sách token."""
from __future__ import annotations

from typing import Callable

from datamate.agent.exemplars import ExemplarBank
from datamate.agent.rulebook_select import select_rulebook
from datamate.config import Config
from datamate.ingestion.datasets.loader import Business, Split
from datamate.ingestion.datasets.profiling import DatasetProfiler
from datamate.text import norm
from datamate.tools.grounding import ValueIndex, format_hits
from datamate.tools.timeutil import describe_period

# mỗi bước: (tổng ký tự profile, ký tự mỗi bảng, có head(3) không, số giá trị phân loại liệt kê tối đa)
_FIT_STEPS_TAIL = [(10000, 2000, True, 12), (6000, 1200, False, 8), (3500, 700, False, 5)]


class ContextBuilder:
    """Dựng ngữ cảnh cho một câu hỏi. `count_tokens` (chỉ local) để rút gọn vừa ngân sách token."""

    def __init__(self, cfg: Config, count_tokens: Callable[[str], int] | None = None,
                 bank: ExemplarBank | None = None):
        self.cfg = cfg
        self.count_tokens = count_tokens
        self.bank = bank
        self.rulebook_limit, self.readme_limit, self.profile_limit = cfg.context_limits
        steps = [(self.profile_limit, 4000, True, 25)] + _FIT_STEPS_TAIL
        self.steps = steps if cfg.is_local else steps[:1]
        self._warned: set = set()

    def _warn_once(self, key, msg: str):
        if key not in self._warned:
            self._warned.add(key)
            print(msg)

    def _prepare(self, biz: Business, calendar):
        """Gộp sheet cùng cấu trúc một lần cho mỗi business."""
        if "merged" not in biz.cache:
            merged, notes = DatasetProfiler.merge_same_schema(biz.tables)
            if calendar is not None:
                merged["calendar"] = calendar
            biz.cache.update(merged=merged, merge_notes=notes, profiles={})

    def _extra(self, question: str, as_of, biz: Business, tables: dict, business_id) -> str:
        """Phần thay đổi theo từng câu hỏi (đặt SAU phần dùng chung của business để giữ được prefix cache):
        gợi ý tất định do mã tính + ví dụ tương tự từ kho."""
        parts = []
        if self.cfg.use_hints:
            lines = []
            period = describe_period(question, as_of, biz.cache["merged"].get("calendar"))
            if period:
                lines.append(period if "\n" in period else f"- Khoảng thời gian: {period}")
            if "value_index" not in biz.cache:
                biz.cache["value_index"] = ValueIndex(tables)
            lines += format_hits(biz.cache["value_index"].ground(question))
            if lines:
                parts.append("GỢI Ý TỪ MÃ (tự tính; nếu mâu thuẫn RULEBOOK hoặc dữ liệu thật thì bỏ qua):\n" + "\n".join(lines))
        if self.bank is not None:
            shots = self.bank.top_k(question, self.cfg.exemplar_k, exclude_business=business_id)
            if shots:
                body = "\n\n".join(f"Câu hỏi: {s['question']}\n```python\n{s['code'][:1500]}\n```" for s in shots)
                parts.append("VÍ DỤ TƯƠNG TỰ (lời giải đã được kiểm đúng ở doanh nghiệp khác; tên bảng/cột có thể khác, "
                             "chỉ học cách làm):\n" + body)
        return "\n\n".join(parts) + "\n\n" if parts else ""

    @staticmethod
    def _build(question, as_of, biz: Business, rulebook, profile, rulebook_chars, readme_chars, extra="") -> str:
        notes = biz.cache["merge_notes"]
        return (f"RULEBOOK:\n{rulebook[:rulebook_chars]}\n\nREADME:\n{biz.readme[:readme_chars]}\n\n"
                f"SCHEMA + PROFILE:\n{profile}\n\n"
                + ("LƯU Ý: " + " ".join(notes) + "\n\n" if notes else "") +
                f"AS_OF: {as_of}\n\n{extra}CÂU HỎI: {question}")

    def build(self, question: str, as_of, biz: Business, split: Split, business_id=None) -> tuple[str, dict]:
        """Trả (ngữ cảnh, bảng đã gộp để chạy code). `business_id` để loại ví dụ cùng business khỏi few-shot."""
        self._prepare(biz, split.calendar)
        tables = dict(biz.cache["merged"])
        extra = self._extra(question, as_of, biz, tables, business_id)
        profiles = biz.cache["profiles"]
        rulebook = select_rulebook(split.rulebook, question) if self.cfg.rulebook_select else split.rulebook
        for name, text, limit in (("rulebook", rulebook, self.rulebook_limit), ("readme", biz.readme, self.readme_limit)):
            if len(text) > limit:
                self._warn_once(name, f"{name} dài {len(text)} ký tự, bị cắt còn {limit}")
        budget = self.cfg.token_budget
        counter = self.count_tokens if budget is not None else None
        ctx = ""
        for si, (plim, ptab, head, mcat) in enumerate(self.steps):
            if si not in profiles:
                profiles[si] = DatasetProfiler.profile_text(tables, plim, ptab, head, mcat)
            ctx = self._build(question, as_of, biz, rulebook, profiles[si], self.rulebook_limit, self.readme_limit, extra)
            if counter is None or counter(ctx) <= budget:
                return ctx, tables
        # vẫn dài: cắt dần rulebook/README, cuối cùng chỉ giữ tên bảng + cột
        last = profiles[len(self.steps) - 1]
        fallbacks = ((self.rulebook_limit // 2, 1000, last), (self.rulebook_limit // 4, 0, last),
                     (1500, 0, profiles.setdefault("min", DatasetProfiler.profile_min(tables))))
        n = 0
        for rb, rd, prof in fallbacks:
            ctx = self._build(question, as_of, biz, rulebook, prof, rb, rd, extra)
            n = counter(ctx)
            if n <= budget:
                return ctx, tables
        self._warn_once(id(biz), f"prompt {n} token > ngân sách {budget} dù đã rút gọn hết mức; tăng max_model_len")
        return ctx, tables
