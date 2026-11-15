"""Mô tả bảng cho prompt: profile cột, gộp các sheet cùng cấu trúc."""
from __future__ import annotations

import json

import pandas as pd

from datamate.text import norm

class DatasetProfiler:
    @staticmethod
    def describe_table(name: str, df: pd.DataFrame, max_cat: int = 25, per_table_limit: int = 6000,
                       with_head: bool = True) -> str:
        lines = [f"## {name}  shape={df.shape}"]
        for c in df.columns:
            s = df[c]
            nn = int(s.notna().sum())
            try:
                nu = int(s.nunique(dropna=True))
            except Exception:
                nu = -1
            info = f"- {c} [{s.dtype}] non_null={nn} unique={nu}"
            try:
                if pd.api.types.is_bool_dtype(s):
                    pass
                elif pd.api.types.is_numeric_dtype(s):
                    if nn:
                        info += f" min={s.min()} max={s.max()}"
                elif pd.api.types.is_datetime64_any_dtype(s):
                    info += f" min={s.min()} max={s.max()}"
                elif nn:
                    sample = s.dropna().astype(str).head(200)
                    # nhận dạng ngày theo mẫu chuỗi, không nhờ pandas parse (dd/mm vs mm/dd dễ đọc nhầm)
                    is_date = sample.str.match(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}").mean() > 0.9
                    if is_date:
                        ss = s.dropna().astype(str)
                        info += f" (chuỗi ngày, định dạng gốc) đầu={list(ss.head(2))} cuối={list(ss.tail(2))}"
                    else:
                        if 0 < nu <= 20000:
                            uniques = s.dropna().astype(str).unique()
                            n_norm = len({norm(v) for v in uniques})
                            if n_norm < len(uniques):
                                info += (f" [!] {len(uniques)} giá trị nhưng chỉ {n_norm} sau khi bỏ dấu/hoa-thường/"
                                         "khoảng trắng (có biến thể tên)")
                        if nu <= max_cat:
                            info += " values=" + json.dumps(sorted(map(str, s.dropna().unique())), ensure_ascii=False)
                        else:
                            top = s.value_counts().head(8)
                            info += " top=" + json.dumps([f"{k} ({v})" for k, v in top.items()], ensure_ascii=False)
            except Exception:
                pass
            lines.append(info)
        if with_head:
            lines.append("head(3):\n" + df.head(3).to_string())
        return "\n".join(lines)[:per_table_limit]

    @staticmethod
    def profile_text(tables: dict, limit: int, per_table: int = 6000, with_head: bool = True, max_cat: int = 25) -> str:
        parts, total = [], 0
        for name, df in tables.items():
            text = DatasetProfiler.describe_table(name, df, max_cat=max_cat, per_table_limit=per_table, with_head=with_head)
            if total + len(text) > limit:
                parts.append(f"## {name}  shape={df.shape}  cột={list(df.columns)}  (rút gọn do giới hạn độ dài)")
            else:
                parts.append(text)
                total += len(text)
        return "\n\n".join(parts)

    @staticmethod
    def profile_min(tables: dict) -> str:
        """Bản tối giản: chỉ tên bảng + 15 cột đầu."""
        return "\n".join(f"## {n} shape={df.shape} cột={list(map(str, df.columns))[:15]}" for n, df in tables.items())

    @staticmethod
    def merge_same_schema(tables: dict, min_sheets: int = 3) -> tuple[dict, list[str]]:
        """Gộp các sheet cùng tiền tố `ten::sheet` và cùng bộ cột (vd hóa đơn theo tháng) thành `ten::ALL`,
        thêm cột `_sheet`. Giảm độ dài prompt và để code sinh ra khỏi phải tự concat."""
        groups: dict = {}
        for name, df in tables.items():
            if "::" in name:
                stem, sheet = name.split("::", 1)
                groups.setdefault((stem, tuple(map(str, df.columns))), []).append((name, sheet, df))
        out, notes, used = dict(tables), [], {}
        for (stem, _cols), items in groups.items():
            if len(items) < min_sheets:
                continue
            k = used.get(stem, 0)
            used[stem] = k + 1
            new_name = f"{stem}::ALL" + (str(k) if k else "")
            parts = []
            for name, sheet, df in items:
                d = df.copy()
                d["_sheet"] = sheet
                parts.append(d)
                out.pop(name)
            out[new_name] = pd.concat(parts, ignore_index=True)
            notes.append(f"`{new_name}` = gộp {len(items)} sheet cùng cấu trúc ({items[0][1]} … {items[-1][1]}); "
                         f"cột `_sheet` là tên sheet gốc. Các sheet gốc không còn trong `tables`.")
        return out, notes
