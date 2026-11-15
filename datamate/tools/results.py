"""Chuẩn hóa kết quả code sinh ra: dạng JSON thuần, đúng kiểu theo `type`, dạng chuẩn để so các lần chạy."""
from __future__ import annotations

import json
import re
import unicodedata

import numpy as np
import pandas as pd

VALID_TYPES = {"string", "number", "percent", "date", "boolean", "list", "table", "unanswerable"}
CODE_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", re.S)


def is_null(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def extract_code(text: str | None) -> str | None:
    """Lấy khối code CUỐI có gán result (model đôi khi viết nháp rồi viết lại). Chịu được khối bị cắt cụt."""
    blocks = [b for b in CODE_FENCE.findall(text or "") if "result" in b]
    return blocks[-1] if blocks else None


def clean(v):
    """Đổi mọi kiểu numpy/pandas sang kiểu JSON thuần. NaN/inf -> None (json.dumps sẽ sinh `NaN` hỏng chuẩn JSON)."""
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return round(f, 4) if np.isfinite(f) else None
    if v is pd.NaT:
        return None
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        t = pd.Timestamp(v)
        return None if pd.isna(t) else t.strftime("%Y-%m-%d")
    if isinstance(v, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        return [clean(x) for x in list(v)]
    if isinstance(v, pd.DataFrame):
        return {"columns": list(map(str, v.columns)), "rows": clean(v.values.tolist())}
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    return v


def is_empty_result(typ: str, val) -> bool:
    """Kết quả rỗng/NaN -> không dùng (trừ unanswerable, nơi val chỉ là lý do)."""
    if typ == "unanswerable":
        return False
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, str) and not val.strip():
        return True
    if isinstance(val, list) and len(val) == 0:
        return True
    if isinstance(val, dict) and "rows" in val and len(val["rows"]) == 0:
        return True
    return False


def coerce_result(typ: str, val):
    """Ép/kiểm kết quả về đúng dạng JSON theo `type` (sai dạng = -1 điểm dù giá trị đúng).
    Không ép được -> ValueError; câu báo lỗi được đưa lại cho model sửa."""
    if typ == "unanswerable":
        return str(val)
    if isinstance(val, list) and len(val) == 1 and typ in ("number", "percent", "string", "date", "boolean"):
        val = val[0]                                  # Series 1 phần tử -> lấy phần tử
    if typ in ("number", "percent"):
        if isinstance(val, bool):
            raise ValueError(f"{typ} phải là số, không phải boolean")
        if isinstance(val, str):
            try:
                val = float(val.strip().rstrip("%").strip())
            except ValueError:
                raise ValueError(f"{typ} phải là số, nhận chuỗi {val!r}: hãy tự chuyển bằng float(...)")
        if not isinstance(val, (int, float)) or not np.isfinite(val):
            raise ValueError(f"{typ} phải là một số hữu hạn, nhận {type(val).__name__}")
        return val
    if typ == "string":
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return str(val)
        if not isinstance(val, str):
            raise ValueError(f"string phải là chuỗi đơn, nhận {type(val).__name__}")
        return val.strip()
    if typ == "date":
        m = re.match(r"^\s*(\d{4}-\d{2}-\d{2})", str(val))
        if not m:
            raise ValueError(f"date phải dạng 'YYYY-MM-DD', nhận {val!r}")
        return m.group(1)
    if typ == "boolean":
        if isinstance(val, bool):
            return val
        raise ValueError(f"boolean phải là True/False, nhận {val!r}")
    if typ == "list":
        if not isinstance(val, list):
            raise ValueError(f"list phải là list, nhận {type(val).__name__}")
        return val
    if typ == "table":
        if not (isinstance(val, dict) and isinstance(val.get("columns"), list) and isinstance(val.get("rows"), list)):
            raise ValueError('table phải là {"columns":[...],"rows":[[...],...]}')
        width = len(val["columns"])
        if any(not isinstance(r, list) or len(r) != width for r in val["rows"]):
            raise ValueError("mỗi hàng của table phải là list có độ dài bằng số cột")
        return val
    return val


def canon(typ: str, val) -> str:
    """Dạng chuẩn để so các lần chạy với nhau (làm tròn 2 chữ số, giữ dấu thanh, table so như tập hàng)."""
    if typ == "unanswerable":
        return "unanswerable"

    def f(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return round(float(x), 2)
        if isinstance(x, str):
            return unicodedata.normalize("NFC", x).strip().lower()
        if isinstance(x, list):
            return [f(i) for i in x]
        return x
    if typ == "table" and isinstance(val, dict):
        rows = sorted(json.dumps(f(r), ensure_ascii=False) for r in val.get("rows", []))
        return json.dumps([typ, [str(c).strip().lower() for c in val.get("columns", [])], rows], ensure_ascii=False)
    return json.dumps([typ, f(val)], ensure_ascii=False)


def final_fmt(typ: str, ans):
    """number/percent làm tròn 2 chữ số như luật chấm."""
    if typ in ("number", "percent") and not is_null(ans):
        try:
            v = round(float(ans), 2)
            return int(v) if v.is_integer() else v
        except Exception:
            return ans
    return ans
