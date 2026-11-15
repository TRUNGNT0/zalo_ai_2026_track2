"""Luật tất định theo ý định (mặc định tắt): chỉ trả lời khi CHẮC, mọi nghi ngờ thì abstain nhường cho LLM.

Hiện phủ: bán chạy/chậm nhất (trả TÊN theo số lượng) và tổng doanh thu trong một kỳ.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd

from datamate.text import colnorm, norm
from datamate.tools.timeutil import parse_cell_date, resolve_time

if TYPE_CHECKING:
    from datamate.ingestion.datasets.loader import Business

KEYS = dict(
    product=[r"ten_?(hang|sp|san_?pham|mon|mat_?hang)", r"^(san_?pham|mat_?hang|hang_?hoa|item|product|mon|ten)$",
             r"product_?name|item_?name"],
    qty=[r"so_?luong", r"^(sl|qty|quantity)$"],
    amount=[r"thanh_?tien", r"doanh_?thu", r"^(amount|line_?total|total)$", r"line_?amount"],
    date=[r"^(ngay|date)$", r"ngay_?(ban|dat|giao_?dich|don|lap)", r"order_?date|sale_?date|transaction", r"ngay",
          r"date", r"^(time|thoi_?gian)"],
    status=[r"trang_?thai", r"^status$", r"(^|_)huy(_|$)|cancel"],
)
# bảng nhập kho / tồn kho / chi phí không phải bảng bán hàng
NOT_SALES = re.compile(r"nhap|purchase|import|inventory|ton_?kho|stock|supplier|ncc|chi_?phi|expense")
# KHÔNG dùng "hoan" trần: nó khớp "hoàn thành" (đơn đã xong) và sẽ loại nhầm cả đơn hợp lệ
CANCEL_PAT = re.compile(r"\b(huy|cancel(l?ed)?|returned?|tra hang|hoan tra|hoan hang|hoan tien|refund(ed)?)\b")
TOP_PAT = re.compile(r"\bban (chay|(duoc )?nhieu nhat)\b")
LOW_PAT = re.compile(r"\bban (cham|(duoc )?it nhat)\b")
REV_PAT = re.compile(r"\bdoanh thu\b")
# có bộ lọc / phân rã / chỉ số khác -> luật không đủ thông tin
BLOCK_COMMON = re.compile(
    r"\b(nhom|loai|danh muc|kenh|khach|nhan vien|chi nhanh|nha cung cap|ca|phuong thuc|thanh toan|"
    r"voucher|khuyen mai|giam gia|top|moi|theo|tung|ton|loi nhuan|lai|trung binh|ty le)\b|so voi|%|\d")
ASKS_NUMBER = re.compile(r"\b(bao nhieu|so luong|may)\b")                 # top/low chỉ trả TÊN
REV_BAD = re.compile(r"\b(cao nhat|thap nhat|ngay nao|thang nao|tuan nao)\b")


def pick(cols, patterns):
    for pat in patterns:
        for c in cols:
            if re.search(pat, colnorm(c)):
                return c
    return None


def profile_table(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    return {k: pick(cols, v) for k, v in KEYS.items()}


def find_fact_table(tables: dict):
    """Chọn bảng nhiều dòng nhất có đủ product + qty + date. Trả (tên, profile) hoặc None."""
    best, best_n = None, -1
    for name, df in tables.items():
        prof = profile_table(df)
        if prof["product"] and prof["qty"] and prof["date"] and not NOT_SALES.search(colnorm(name)) and len(df) > best_n:
            best, best_n = (name, prof), len(df)
    return best


def is_cancel(x) -> bool:
    return bool(CANCEL_PAT.search(re.sub(r"[_\-]+", " ", norm(x))))


def mentions_product(qn: str, biz: "Business", table_name: str, product_col: str) -> bool:
    """Câu hỏi nhắc tới một sản phẩm cụ thể -> luật không biết lọc theo sản phẩm."""
    key = ("_prodnorm", table_name)
    if key not in biz.cache:
        values = biz.tables[table_name][product_col].dropna().astype(str).unique()
        biz.cache[key] = [p for p in {norm(v) for v in values} if len(p) >= 4]
    return any(re.search(rf"\b{re.escape(p)}\b", qn) for p in biz.cache[key])


def prep_fact(df: pd.DataFrame, prof: dict, period) -> pd.DataFrame:
    d = df.copy()
    table_year = re.search(r"\b(19|20)\d{2}\b", str(df.attrs.get("table_name", "")))
    fallback_year = int(table_year[0]) if table_year else None
    date_values = []
    for i, value in enumerate(d[prof["date"]]):
        sheet_year = fallback_year
        if "_sheet" in d.columns and pd.notna(d.iloc[i]["_sheet"]):
            sheet_match = re.search(r"\b(19|20)\d{2}\b", str(d.iloc[i]["_sheet"]))
            if sheet_match:
                sheet_year = int(sheet_match[0])
        parsed = parse_cell_date(value, sheet_year=sheet_year)
        date_values.append(pd.Timestamp(parsed) if parsed is not None else pd.NaT)
    d["_date"] = pd.to_datetime(date_values, errors="coerce")
    d = d[d["_date"].notna()]
    if prof["status"]:                          # loại đơn hủy/hoàn (xấp xỉ rulebook)
        d = d[~d[prof["status"]].astype(str).map(is_cancel)]
    if period:
        start, end = period
        d = d[(d["_date"] >= start) & (d["_date"] <= end)]
    d["_qty"] = pd.to_numeric(d[prof["qty"]], errors="coerce")
    if prof["amount"]:
        d["_amt"] = pd.to_numeric(d[prof["amount"]], errors="coerce")
    return d


def rule_solve(question: str, as_of, biz: "Business", rule_on: dict, calendar=None):
    """Trả (type, answer, intent, note). type="abstain" nếu không chắc."""
    qn = norm(question)
    top, low = bool(TOP_PAT.search(qn)), bool(LOW_PAT.search(qn))
    rev = bool(REV_PAT.search(qn))
    if top and low:
        return "abstain", None, "ambiguous", ""
    intent = "top" if top else "low" if low else "revenue" if rev else None
    if intent is None:
        return "abstain", None, "unsupported_intent", ""
    if not rule_on.get(intent, False):
        return "abstain", None, f"{intent}_off", ""
    if BLOCK_COMMON.search(qn) or (intent != "revenue" and ASKS_NUMBER.search(qn)) \
            or (intent == "revenue" and REV_BAD.search(qn)):
        return "abstain", None, f"{intent}_has_filter", ""
    fact = find_fact_table(biz.tables)
    if fact is None:
        return "abstain", None, "no_fact_table", ""
    name, prof = fact
    if mentions_product(qn, biz, name, prof["product"]):
        return "abstain", None, f"{intent}_mentions_product", name
    time_spec = resolve_time(question, as_of, calendar)
    main_periods = time_spec.main()
    if (len(main_periods) != 1 or time_spec.baseline() is not None or time_spec.series
            or time_spec.stock or time_spec.shift or time_spec.baseline_shift
            or time_spec.needs_data or time_spec.flags or time_spec.time_cue):
        return "abstain", None, "no_period", name
    period = (pd.Timestamp(main_periods[0].start), pd.Timestamp(main_periods[0].end))
    fact_df = biz.tables[name].copy()
    fact_df.attrs["table_name"] = name
    d = prep_fact(fact_df, prof, period)
    if d.empty:
        return "abstain", None, "empty_period", name
    if intent in ("top", "low"):
        grouped = d.groupby(prof["product"])["_qty"].sum().sort_values(ascending=(intent == "low"))
        if grouped.empty:
            return "abstain", None, "empty_group", name
        return "string", str(grouped.index[0]), intent, name
    if prof["amount"]:
        v = float(d["_amt"].sum())
        return "number", (int(v) if float(v).is_integer() else round(v, 2)), "revenue", name
    return "abstain", None, "no_amount_col", name
