"""Compatibility helpers around the shared Vietnamese time parser."""
from __future__ import annotations

import re

import pandas as pd

from datamate.tools.viet_time import (
    TimeSpec,
    availability,
    effective,
    in_shift,
    normalize as normalize_time_query,
    parse_cell_date,
    parse_time_of_day,
    resolve,
)


def resolve_time(question: str, as_of, calendar=None) -> TimeSpec:
    """Resolve the question's full time semantics against its per-question as-of date."""
    if calendar is not None and "date" in calendar.columns:
        calendar = calendar.copy()
        calendar["date"] = pd.to_datetime(calendar["date"].map(parse_cell_date), errors="coerce")
        if "lunar_leap" in calendar.columns:
            calendar["lunar_leap"] = calendar["lunar_leap"].map(
                lambda value: value.strip().casefold() in {"1", "true", "yes"}
                if isinstance(value, str) else bool(value)
            )
    spec = resolve(question, pd.Timestamp(as_of).date(), cal=calendar)
    qn = normalize_time_query(question)
    if (re.search(r"\b(so voi|so sanh|cung ky)\b", qn)
            and (spec.baseline() is None or "cung_ky_unit_unknown" in spec.flags)):
        spec.flags.append("comparison_period_unresolved")
    return spec


def _simple_main_period(spec: TimeSpec):
    main = spec.main()
    if (len(main) != 1 or spec.baseline() is not None or spec.series or spec.stock
            or spec.shift or spec.baseline_shift or spec.needs_data or spec.flags
            or spec.time_cue):
        return None
    return main[0]


def resolve_period(question: str, as_of, calendar=None) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return one unambiguous inclusive interval for consumers that need a simple period."""
    period = _simple_main_period(resolve_time(question, as_of, calendar))
    if period is None:
        return None
    return pd.Timestamp(period.start), pd.Timestamp(period.end)


def describe_period(question: str, as_of, calendar=None) -> str | None:
    """Format the resolved periods, comparisons, and uncertainty as a prompt hint."""
    spec = resolve_time(question, as_of, calendar)
    if not (spec.periods or spec.flags or spec.needs_data or spec.shift
            or spec.baseline_shift or spec.time_cue):
        return None

    if (len(spec.periods) == 1 and spec.periods[0].role == "main"
            and not spec.flags and not spec.needs_data and not spec.shift
            and not spec.baseline_shift):
        period = spec.periods[0]
        label = period.label.split(" (", 1)[0]
        suffix = " (tuần tính Thứ Hai đến Chủ Nhật)" if "tuần" in label else ""
        return (f'"{label}" = {period.start:%Y-%m-%d} đến '
                f'{period.end:%Y-%m-%d}{suffix}')

    lines = []
    for period in spec.periods:
        role = {"main": "Kỳ chính", "baseline": "Kỳ so sánh", "anchor": "Mốc tham chiếu"}.get(
            period.role, period.role
        )
        lines.append(f"- {role} ({period.label}): {period.start:%Y-%m-%d} đến {period.end:%Y-%m-%d}")
    if spec.series:
        lines.append("- Kết quả cần giữ thứ tự chuỗi thời gian.")
    if spec.stock:
        lines.append(f"- Đây là câu hỏi trạng thái tồn tại đến hết ngày {spec.stock_date}.")
    if spec.shift:
        lines.append(f"- Lọc kỳ chính theo ca {spec.shift}.")
    if spec.baseline_shift:
        lines.append(f"- Lọc kỳ so sánh theo ca {spec.baseline_shift}.")
    if spec.needs_data:
        lines.append(f"- Cần thêm dữ liệu để xác định: {', '.join(spec.needs_data)}.")
    if spec.time_cue:
        lines.append("- Có cụm thời gian nhưng chưa phân giải được kỳ; cần đối chiếu RULEBOOK/dữ liệu, không tự suy đoán.")
    if spec.flags:
        lines.append(f"- Cần thận trọng với diễn giải: {'; '.join(spec.flags)}.")
    return "\n".join(lines)
