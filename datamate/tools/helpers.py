"""Hàm trợ giúp tất định cho code do LLM sinh ra (được nạp sẵn vào vùng chạy code, xem sandbox.py).

Việc nào chỉ có MỘT đáp án đúng (đọc số tiền, đọc ngày, quy đổi "tuần trước", nhận diện đơn hủy/trả, khoảng Tết)
thì giao cho mã thay vì để model tự viết lại mỗi lần; model nhỏ hay sai đúng những chỗ này.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd


def norm_text(x) -> str:
    """Bỏ dấu, hạ chữ thường, gọn khoảng trắng. Dùng để so tên/trạng thái: 'Huỷ đơn' và 'hủy' đều thành 'huy ...'."""
    s = unicodedata.normalize("NFD", str(x))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s.replace("đ", "d").replace("Đ", "D").lower()).strip()


def _parse_number(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (bool, np.bool_)):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    raw = str(v).strip()
    negative = raw.startswith(("-", "−")) or (raw.startswith("(") and raw.endswith(")"))
    s = re.sub(r"[^\d.,]", "", raw)
    if not s or not re.search(r"\d", s):
        return np.nan
    if "," in s and "." in s:                       # dấu xuất hiện SAU CÙNG là dấu thập phân
        decimal = "," if s.rfind(",") > s.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        s = s.replace(thousands, "").replace(decimal, ".")
    elif "," in s or "." in s:
        sep = "," if "," in s else "."
        parts = s.split(sep)
        # '1,250' / '12.500' / '1.250.000' là hàng nghìn; '18.6' / '125,50' / '0,25' là thập phân
        grouped = len(parts) > 2 or (len(parts[1]) == 3 and parts[0] not in ("0", "") and len(parts[0]) <= 3)
        s = "".join(parts) if grouped else parts[0] + "." + parts[1]
    try:
        value = float(s)
    except ValueError:
        return np.nan
    return -value if negative else value


def to_number(x):
    """Đọc số từ chuỗi kiểu '1,250,000', '1.250.000đ', '12,5%', ' 99,000 '. Series -> Series float; không đọc được -> NaN."""
    if isinstance(x, pd.Series):
        return pd.to_numeric(x.map(_parse_number), errors="coerce")
    return _parse_number(x)


def to_date(x):
    """Đọc ngày. Chuỗi ISO (YYYY-MM-DD) giữ nguyên thứ tự; còn lại coi là dd/mm/yyyy (dayfirst). Lỗi -> NaT."""
    if isinstance(x, pd.Series):
        if pd.api.types.is_datetime64_any_dtype(x):
            return x
        s = x.astype(str).str.strip()
        iso = s.str.match(r"^\d{4}-\d{1,2}-\d{1,2}").mean() > 0.9
        return pd.to_datetime(s, errors="coerce", dayfirst=not iso, format="mixed")
    return to_date(pd.Series([x])).iloc[0]


def in_range(dates, start, end):
    """Mặt nạ: ngày nằm trong [start, end] tính theo NGÀY (bỏ giờ), gồm cả hai đầu."""
    d = to_date(dates).dt.normalize()
    return (d >= pd.Timestamp(start).normalize()) & (d <= pd.Timestamp(end).normalize())


def period_range(kind: str, as_of, n: int | None = None, month: int | None = None, year: int | None = None,
                 quarter: int | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Khoảng thời gian theo rulebook (đầu, cuối đều gồm). Tuần bắt đầu Thứ Hai.

    kind: today, yesterday, this_week, last_week, this_month, last_month, prev_month_same_days (tháng trước, CÙNG số
    ngày với tháng này tới hôm nay), this_year, last_year, this_quarter, last_quarter, last_n_days (cần n, gồm hôm nay),
    month (cần month, tùy chọn year), quarter (cần quarter, tùy chọn year), year (cần year).
    Tháng/quý không ghi năm: năm nay nếu không sau hôm nay, ngược lại năm trước.
    """
    t = pd.Timestamp(as_of).normalize()
    day = pd.Timedelta(days=1)

    def month_span(y: int, m: int):
        first = pd.Timestamp(y, m, 1)
        return first, first + pd.offsets.MonthEnd(0)

    def quarter_start(y: int, q: int):
        return pd.Timestamp(y, 3 * (q - 1) + 1, 1)

    if kind == "today":
        return t, t
    if kind == "yesterday":
        return t - day, t - day
    monday = t - pd.Timedelta(days=t.weekday())
    if kind == "this_week":
        return monday, t
    if kind == "last_week":
        return monday - 7 * day, monday - day
    if kind == "this_month":
        return t.replace(day=1), t
    if kind == "last_month":
        first_prev = t.replace(day=1) - pd.offsets.MonthBegin(1)
        return first_prev, t.replace(day=1) - day
    if kind == "prev_month_same_days":
        first_prev = t.replace(day=1) - pd.offsets.MonthBegin(1)
        last_prev = t.replace(day=1) - day
        return first_prev, min(first_prev + pd.Timedelta(days=t.day - 1), last_prev)
    if kind == "this_year":
        return pd.Timestamp(t.year, 1, 1), t
    if kind == "last_year":
        return pd.Timestamp(t.year - 1, 1, 1), pd.Timestamp(t.year - 1, 12, 31)
    q_now = (t.month - 1) // 3 + 1
    if kind == "this_quarter":
        return quarter_start(t.year, q_now), t
    if kind == "last_quarter":
        y, q = (t.year, q_now - 1) if q_now > 1 else (t.year - 1, 4)
        start = quarter_start(y, q)
        return start, start + pd.offsets.MonthEnd(3)
    if kind == "last_n_days":
        if not n or n < 1:
            raise ValueError("last_n_days cần n >= 1")
        return t - (n - 1) * day, t
    if kind == "month":
        if not month:
            raise ValueError("month cần tham số month")
        y = year if year else (t.year if pd.Timestamp(t.year, month, 1) <= t else t.year - 1)
        return month_span(y, month)
    if kind == "quarter":
        if not quarter:
            raise ValueError("quarter cần tham số quarter")
        y = year if year else (t.year if quarter_start(t.year, quarter) <= t else t.year - 1)
        start = quarter_start(y, quarter)
        return start, start + pd.offsets.MonthEnd(3)
    if kind == "year":
        if not year:
            raise ValueError("year cần tham số year")
        return pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
    raise ValueError(f"kind không hợp lệ: {kind!r}")


def tet_range(calendar: pd.DataFrame, year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Dịp Tết năm `year` theo rulebook: từ ngày dương của 23 tháng Chạp (năm âm year-1) đến HẾT ngày dương của
    mùng 10 tháng Giêng (năm âm year). `calendar` là tables["calendar"] (cột date, lunar_day, lunar_month, lunar_year)."""
    cal = calendar.copy()
    cal["_d"] = pd.to_datetime(cal["date"], errors="coerce")
    leap = cal["lunar_leap"].astype(bool) if "lunar_leap" in cal else pd.Series(False, index=cal.index)
    start = cal[(cal["lunar_year"] == year - 1) & (cal["lunar_month"] == 12) & (cal["lunar_day"] == 23) & ~leap]
    end = cal[(cal["lunar_year"] == year) & (cal["lunar_month"] == 1) & (cal["lunar_day"] == 10) & ~leap]
    if start.empty or end.empty:
        raise ValueError(f"lịch không phủ Tết năm {year}")
    return start["_d"].iloc[0], end["_d"].iloc[0]


_STATE_PATTERNS = [
    ("cancelled", r"\bhuy\b|\bhuy don\b|cancel|da huy"),
    ("returned", r"tra hang|hang tra|doi tra|hoan tra|tra ve|return|refund"),
    ("shipping", r"dang giao|van chuyen|dang van|shipping|delivering"),
    ("done", r"hoan thanh|thanh cong|\bxong\b|\bdone\b|complete"),
]


def order_state(status) -> str:
    """Trạng thái đơn theo rulebook: 'cancelled' | 'returned' | 'shipping' | 'done' | 'unknown'.
    Nhận mọi cách viết đã gặp: 'Huỷ đơn', 'hủy', 'Đã hủy đơn', 'hàng trả về', 'Đổi trả', 'Hoàn trả', 'xong'..."""
    s = norm_text(status)
    for state, pattern in _STATE_PATTERNS:
        if re.search(pattern, s):
            return state
    return "unknown"


def is_cancelled(status) -> bool:
    return order_state(status) == "cancelled"


def is_returned(status) -> bool:
    return order_state(status) == "returned"


def counts_as_revenue(status) -> bool:
    """Đơn tính vào doanh thu: không hủy và không trả hàng (rulebook §1). Đơn đang giao vẫn tính."""
    return order_state(status) not in ("cancelled", "returned")


def fill_group(df: pd.DataFrame, key: str, cols: list[str]) -> pd.DataFrame:
    """Ô trống ở các dòng sau của cùng một đơn lấy giá trị từ dòng đầu của đơn (rulebook §8). Trả bản sao."""
    out = df.copy()
    out[cols] = out.groupby(key)[cols].ffill()
    return out


HELPERS = {
    "norm_text": norm_text, "to_number": to_number, "to_date": to_date, "in_range": in_range,
    "period_range": period_range, "tet_range": tet_range, "order_state": order_state, "is_cancelled": is_cancelled,
    "is_returned": is_returned, "counts_as_revenue": counts_as_revenue, "fill_group": fill_group,
}

HELPERS_DOC = """
HÀM TRỢ GIÚP có sẵn (dùng thay vì tự viết lại; vẫn phải kiểm tra tên cột và kết quả):
- to_number(x): đọc số từ chuỗi '1,250,000' / '1.250.000đ' / '12,5%'; truyền cả Series; lỗi -> NaN.
- to_date(series): đọc ngày (YYYY-MM-DD giữ nguyên, còn lại coi là dd/mm/yyyy). in_range(series, start, end): mặt nạ theo NGÀY, gồm 2 đầu.
- period_range(kind, AS_OF, n=None, month=None, year=None, quarter=None) -> (start, end) theo rulebook; kind: today, yesterday,
  this_week, last_week, this_month, last_month, prev_month_same_days, this_year, last_year, this_quarter, last_quarter,
  last_n_days (cần n), month (cần month), quarter (cần quarter), year (cần year). KHÔNG gán cứng ngày: dùng hàm này với AS_OF.
- tet_range(tables["calendar"], year) -> (start, end) dịp Tết.
- order_state(status) -> 'cancelled'|'returned'|'shipping'|'done'|'unknown'; is_cancelled, is_returned, counts_as_revenue(status)
  (đơn tính doanh thu = không hủy, không trả hàng). Nhận mọi cách viết trạng thái ('Huỷ đơn', 'hủy', 'Đổi trả', ...).
- norm_text(x): bỏ dấu + hạ chữ thường để so tên. fill_group(df, key, cols): ô trống kế thừa dòng đầu của cùng đơn.
Ví dụ: s, e = period_range("last_week", AS_OF); m = in_range(df["Thời gian"], s, e) & df["Trạng thái"].map(counts_as_revenue)
"""
