"""Chạy code do LLM sinh ra với giới hạn thời gian.

Lưu ý: đây KHÔNG phải sandbox bảo mật. Chỉ có giới hạn thời gian (SIGALRM, chỉ hoạt động ở luồng chính, Unix)
và bản sao bảng để code không làm hỏng dữ liệu gốc. Vòng nộp Docker nên chạy trong tiến trình con cách ly.
"""
from __future__ import annotations

import concurrent.futures
import signal

import numpy as np
import pandas as pd

from datamate.tools.helpers import HELPERS
from datamate.tools.viet_time import in_shift, parse_cell_date, parse_time_of_day


class CodeTimeout(Exception):
    pass


def _on_alarm(*_args):
    raise CodeTimeout("quá thời gian chạy")


def run_code(code: str, tables: dict, as_of, secs: int = 60):
    """Chạy `code` với tables/pd/np/AS_OF; trả (result_type, result) do code gán."""
    ns = {
        "tables": {k: v.copy() for k, v in tables.items()},
        "pd": pd,
        "np": np,
        "AS_OF": pd.Timestamp(as_of),
        "in_shift": in_shift,
        "parse_cell_date": parse_cell_date,
        "parse_time_of_day": parse_time_of_day,
        **HELPERS,
    }
    has_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "alarm")
    if has_alarm:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(secs)
        try:
            exec(code, ns)
        finally:
            signal.alarm(0)
    else:
        import threading
        exc = []
        def _target():
            try:
                exec(code, ns)
            except Exception as e:
                exc.append(e)
        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout=secs)
        if t.is_alive():
            raise CodeTimeout("quá thời gian chạy")
        if exc:
            raise exc[0]
    return ns["result_type"], ns["result"]
