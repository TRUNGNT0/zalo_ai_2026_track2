from datetime import date, datetime
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any, Iterable, Sequence
import datetime as dt
import math
import re
import unicodedata

import pandas as pd

from datamate.text import colnorm, norm
from datamate.ingestion.raw import RawTable, RawData

NUMBER_PATTERN = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?$")
GROUPED_NUMBER_PATTERN = re.compile(r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")


class IdentifierNormalizer:
    """Normalize source headers, physical identifiers and search aliases."""

    @staticmethod
    def aliases(source_name: str | None, description: str | None) -> list[str]:
        aliases: list[str] = []
        for candidate in (source_name, description):
            if candidate and candidate.strip() and candidate.strip() not in aliases:
                aliases.append(candidate.strip())
        return aliases

    @staticmethod
    def normalize(value: str, *, fallback: str = "column") -> str:
        value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
        value = re.sub(r"([A-Z])([A-Z][a-z])", r"\1_\2", value)
        normalized = unicodedata.normalize("NFKD", value.casefold().replace("đ", "d"))
        normalized = "".join(
            character
            for character in normalized
            if not unicodedata.combining(character)
        )
        normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
        if not normalized:
            normalized = fallback
        if normalized[0].isdigit():
            normalized = f"col_{normalized}"
        return normalized

    @staticmethod
    def unique(names: Sequence[str | None]) -> tuple[str, ...]:
        result: list[str] = []
        seen: dict[str, int] = {}
        for index, name in enumerate(names, start=1):
            base = IdentifierNormalizer.normalize(name or "", fallback=f"column_{index}")
            seen[base] = seen.get(base, 0) + 1
            result.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
        return tuple(result)

    @staticmethod
    def headers(values: Sequence[Any]) -> tuple[str | None, ...]:
        return tuple(
            None if value is None or not str(value).strip() else str(value)
            for value in values
        )


class ValueNormalizer:
    """Infer, convert, and normalize source values without losing identifiers."""

    @staticmethod
    def convert_table(raw: RawTable, readme: str, options: dict):
        return ValueNormalizer._convert_table(raw, readme, options)

    @staticmethod
    def parse_number(value, decimal=None, suffix_multipliers=None):
        return ValueNormalizer._parse_number(value, decimal, suffix_multipliers)

    @staticmethod
    def _parse_numeric_text(value: str) -> int | float | None:
        text = value.strip()
        if GROUPED_NUMBER_PATTERN.fullmatch(text):
            text = text.replace(",", "")
        elif not NUMBER_PATTERN.fullmatch(text):
            return None

        try:
            number = Decimal(text)
        except InvalidOperation:
            return None
        if not number.is_finite():
            return None
        if number == number.to_integral_value():
            return int(number)
        return float(number)

    @staticmethod
    def _normalized_value(value: Any) -> Any:
        if isinstance(value, str):
            parsed = ValueNormalizer._parse_numeric_text(value)
            return parsed if parsed is not None else value
        return value

    @staticmethod
    def infer_type(values: Iterable[Any], *, force_text: bool = False) -> str:
        if force_text:
            return "VARCHAR"

        observed = [ValueNormalizer._normalized_value(value) for value in values if value is not None]
        if not observed:
            return "VARCHAR"
        if all(isinstance(value, bool) for value in observed):
            return "BOOLEAN"
        if all(isinstance(value, int) and not isinstance(value, bool) for value in observed):
            if all(-(2**63) <= value < 2**63 for value in observed):
                return "BIGINT"
            return "VARCHAR"
        if all(
            isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)
            for value in observed
        ):
            return "DOUBLE"
        if all(isinstance(value, datetime) for value in observed):
            return "TIMESTAMP"
        if all(isinstance(value, date) and not isinstance(value, datetime) for value in observed):
            return "DATE"
        return "VARCHAR"

    @staticmethod
    def normalize(value: Any, duckdb_type: str) -> Any:
        if value is None:
            return None
        if duckdb_type == "VARCHAR":
            if isinstance(value, datetime):
                return value.isoformat(sep=" ")
            if isinstance(value, date):
                return value.isoformat()
            return str(value)
        if duckdb_type in {"BIGINT", "DOUBLE"}:
            normalized = ValueNormalizer._normalized_value(value)
            if isinstance(normalized, bool) or not isinstance(
                normalized, (int, float, Decimal)
            ):
                raise ValueError(f"Value {value!r} cannot be converted to {duckdb_type}.")
            return int(normalized) if duckdb_type == "BIGINT" else float(normalized)
        if duckdb_type == "TIMESTAMP" and isinstance(value, datetime):
            return value
        if duckdb_type == "DATE" and isinstance(value, date):
            return value
        if duckdb_type == "BOOLEAN" and isinstance(value, bool):
            return value
        raise ValueError(f"Value {value!r} cannot be converted to {duckdb_type}.")

    @staticmethod
    def is_identifier(semantic_name: str | None, source_name: str | None) -> bool:
        if source_name and re.match(r"^(?:mã|ma)[A-Z]", source_name, flags=re.IGNORECASE):
            return True
        tokens = {
            token
            for value in (semantic_name, source_name)
            if value
            for token in re.split(r"[^a-z0-9]+", value.casefold())
            if token
        }
        return bool(tokens.intersection({"id", "code", "ma", "mã", "key", "uuid"}))

    @staticmethod
    def missing(value):
        return value is None or (isinstance(value, str) and not value.strip())

    @staticmethod
    def identifier_column(name):
        name = colnorm(re.sub(r"([a-z])([A-Z])", r"\1_\2", name))
        return bool(re.search(r"(^|_)(ma|id|code|sku|barcode|phone|tel|cccd|cmnd|mst|sdt)(_|$)", name)
                    or name in ("hd", "so_hd", "so_kh")
                    or any(x in name for x in ("dien_thoai", "tax_code", "so_tai_khoan", "so_hoa_don")))

    @staticmethod
    def text_column(name, description):
        """Categorical meaning has priority over words such as money in examples."""
        name = colnorm(re.sub(r"([a-z])([A-Z])", r"\1_\2", name))
        description = colnorm(description)
        prefixes = ("ten", "nhom", "trang_thai", "phuong_thuc", "hinh_thuc", "cach", "kenh",
                    "ghi_chu", "dia_chi", "email", "don_vi", "dvt", "unit", "name", "status", "category", "method", "tien_do")
        return any(value == p or value.startswith(p + "_") for value in (name, description) for p in prefixes)

    @staticmethod
    def number_hint(name):
        name = colnorm(re.sub(r"([a-z])([A-Z])", r"\1_\2", name))
        return bool(re.search(r"(^|_)(gia|tien|luong|sluong|tong|no|ton|diem|qty|quantity|amount|price|total|stt|sl)(_|$)", name))

    @staticmethod
    def decimal_style(value: str):
        """Evidence only: repeated thousands groups or both separators."""
        s = value.strip().lstrip("+-")
        if "." in s and "," in s:
            return "," if s.rfind(",") > s.rfind(".") else "."
        for thousands, decimal in ((",", "."), (".", ",")):
            if re.fullmatch(r"\d{1,3}(?:" + re.escape(thousands) + r"\d{3}){2,}", s):
                return decimal
        if re.fullmatch(r"\d+[.,]\d{1,2}", s):
            return s[-3] if s[-3] in ".," else s[-2]
        return None

    @staticmethod
    def _parse_number(value, decimal=None, suffix_multipliers=None):
        multiplier = Decimal(1)
        if isinstance(value, str) and suffix_multipliers:
            stripped = value.strip()
            for suffix in sorted(suffix_multipliers, key=len, reverse=True):
                if suffix and stripped.casefold().endswith(suffix.casefold()):
                    multiplier = Decimal(str(suffix_multipliers[suffix]))
                    value = stripped[:-len(suffix)].strip()
                    break
        if isinstance(value, bool):
            raise ValueError("boolean_is_not_number")
        if isinstance(value, (int, float, Decimal)):
            result = Decimal(str(value))
        else:
            s = str(value).strip().replace("\u00a0", " ")
            if s.startswith("(") and s.endswith(")"):
                s = "-" + s[1:-1]
            if not re.fullmatch(r"[+-]?\d+(?:[., ]\d+)*", s):
                raise ValueError("invalid_number")
            if " " in s:
                if not re.fullmatch(r"[+-]?\d{1,3}(?: \d{3})+(?:[.,]\d+)?", s):
                    raise ValueError("invalid_grouping")
                s = s.replace(" ", "")
            style = decimal or ValueNormalizer.decimal_style(s)
            if style:
                thousands = "," if style == "." else "."
                integer = s.split(style)[0].lstrip("+-")
                if thousands in integer and not re.fullmatch(r"\d{1,3}(?:" + re.escape(thousands) + r"\d{3})+", integer):
                    raise ValueError("invalid_grouping")
                if thousands in s.split(style)[-1] and style in s:
                    raise ValueError("invalid_grouping")
                s = s.replace(thousands, "").replace(style, ".")
            elif "." in s or "," in s:
                raise ValueError("ambiguous_number")
            try:
                result = Decimal(s)
            except InvalidOperation:
                raise ValueError("invalid_number") from None
        if not result.is_finite():
            raise ValueError("non_finite_number")
        with localcontext() as context:
            context.prec = 80
            result *= multiplier
        if len(result.as_tuple().digits) > 38 or abs(result.adjusted()) > 37:
            raise ValueError("number_out_of_range")
        return result

    @staticmethod
    def date_policy(values, readme, override):
        if override:
            return override, "override"
        text = norm(readme)
        dmy = bool(re.search(r"dd[/.-]mm[/.-]yyyy|dd[/.-]mm[/.-]yy\b", text))
        mdy = bool(re.search(r"mm[/.-]dd[/.-]yyyy|mm[/.-]dd[/.-]yy\b", text))
        evidence = set()
        for value in values:
            match = re.match(r"^(\d{1,2})[/.-](\d{1,2})[/.-]\d{4}(?:\s|$)", str(value).strip())
            if match:
                first, second = map(int, match.groups())
                if first > 12:
                    evidence.add("dmy")
                if second > 12:
                    evidence.add("mdy")
        if dmy:
            evidence.add("dmy")
        if mdy:
            evidence.add("mdy")
        if len(evidence) == 1:
            return evidence.pop(), "README/column"
        return None, "conflict" if evidence else "ambiguous"

    @staticmethod
    def parse_date(value, policy, excel_epoch=None):
        if isinstance(value, (dt.datetime, dt.date, pd.Timestamp)):
            return pd.Timestamp(value)
        s = str(value).strip()
        if re.fullmatch(r"\d+(?:\.\d+)?", s):
            if not excel_epoch:
                raise ValueError("excel_serial_requires_explicit_epoch")
            from openpyxl.utils.datetime import from_excel, CALENDAR_MAC_1904, CALENDAR_WINDOWS_1900
            epoch = CALENDAR_MAC_1904 if excel_epoch == "1904" else CALENDAR_WINDOWS_1900
            return pd.Timestamp(from_excel(float(s), epoch=epoch))
        vietnamese = re.fullmatch(r"(\d{1,2})\s+(?:thg|thang)\s+(\d{1,2})\s+(\d{4})", norm(s))
        if vietnamese:
            day, month, year = map(int, vietnamese.groups())
            return pd.Timestamp(dt.datetime(year, month, day))
        if re.match(r"^\d{4}-\d{2}-\d{2}(?:[ T]|$)", s):
            return pd.Timestamp(dt.datetime.fromisoformat(s))
        if policy and policy not in ("dmy", "mdy"):
            return pd.Timestamp(dt.datetime.strptime(s, policy))
        if not policy:
            raise ValueError("ambiguous_date")
        match = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})(.*)", s)
        if not match:
            raise ValueError("invalid_date")
        first, second, year, tail = match.groups()
        day, month = (first, second) if policy == "dmy" else (second, first)
        return pd.Timestamp(dt.datetime.fromisoformat(f"{year}-{int(month):02}-{int(day):02}{tail}"))

    @staticmethod
    def row_flags(raw: RawTable):
        flags, seen = [], set()
        header = [norm(c) for c in raw.source.get("original_headers", raw.headers)]
        for row in raw.rows:
            current = []
            strings = [norm(v) for v in row]
            if all(ValueNormalizer.missing(v) for v in row):
                current.append("blank_row")
            if strings == header:
                current.append("repeated_header")
            if any(re.fullmatch(r"(?:tong(?: cong)?|tong ket|subtotal|grand total|total)(?:\s*:)?", s) for s in strings):
                current.append("possible_total")
            signature = RawData.json_text(row)
            if signature in seen and "blank_row" not in current:
                current.append("possible_duplicate")
            seen.add(signature)
            flags.append(current)
        return flags

    @staticmethod
    def _convert_table(raw: RawTable, readme: str, options: dict):
        unknown_columns = set(options.get("columns", {})) - set(raw.headers)
        if unknown_columns:
            raise ValueError(f"Unknown override columns: {sorted(unknown_columns)}")
        flags = ValueNormalizer.row_flags(raw)
        issues = list(raw.issues)
        data, columns = {}, []
        descriptions = {}
        file_name = raw.source.get("file", "").replace("\\", "/").rsplit("/", 1)[-1]
        section = re.search(r"^## Cột trong `" + re.escape(file_name) + r"`[^\n]*\n(.*?)(?=^## |\Z)", readme, re.M | re.S)
        if section:
            descriptions = dict(re.findall(r"^- \*\*(.*?)\*\*:\s*(.+)$", section.group(1), re.M))
        for j, name in enumerate(raw.headers):
            spec = options.get("columns", {}).get(name, {})
            values = [row[j] for row in raw.rows]
            valid = [value for i, value in enumerate(values) if not ValueNormalizer.missing(value) and "repeated_header" not in flags[i]]
            kind = spec.get("type")
            description = spec.get("description", descriptions.get(name, ""))
            primary_description = re.split(r"[;:(,]", description)[0]
            identifier = ValueNormalizer.identifier_column(name) or ValueNormalizer.identifier_column(primary_description)
            date_like = bool(valid) and sum(
                isinstance(value, (dt.date, dt.datetime)) or bool(re.match(
                    r"^\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}(?:\s|T|$)", str(value)))
                for value in valid
            ) / len(valid) >= .9
            if kind is None:
                if identifier or ValueNormalizer.text_column(name, primary_description):
                    kind = "string"
                elif date_like or any(keyword in colnorm(name) for keyword in ("ngay", "thoi_gian", "date")):
                    kind = "date"
                elif ValueNormalizer.number_hint(name) or ValueNormalizer.number_hint(primary_description) or (
                    valid and all(isinstance(value, (int, float, Decimal)) and not isinstance(value, bool) for value in valid)
                ):
                    numeric_evidence = not valid or any(
                        isinstance(value, (int, float, Decimal)) or re.match(r"^[+\-(]?\d", str(value).strip())
                        for value in valid
                    )
                    if numeric_evidence:
                        kind = "quantity" if any(QUANTITY.fullmatch(str(value).strip()) for value in valid) else "number"
                    else:
                        kind = "string"
                        issues.append(RawData.issue("numeric_hint_unconfirmed", column=name, review_required=True))
                elif valid and all(isinstance(value, bool) for value in valid):
                    kind = "boolean"
                else:
                    kind = "string"
            if kind not in ("string", "number", "integer", "date", "boolean", "quantity"):
                raise ValueError(f"Unsupported type {kind!r} for {name}")
            decimal = spec.get("decimal")
            if decimal not in (None, ".", ","):
                raise ValueError(f"Invalid decimal separator for {name}")
            excel_epoch = spec.get("excel_epoch") or raw.source.get("excel_epoch")
            if excel_epoch not in (None, "1900", "1904"):
                raise ValueError(f"Invalid excel_epoch for {name}; use string 1900 or 1904")
            suffix_multipliers = spec.get("suffix_multipliers", {})
            for suffix, multiplier in suffix_multipliers.items():
                numeric_multiplier = Decimal(str(multiplier))
                if not suffix or not numeric_multiplier.is_finite() or numeric_multiplier <= 0:
                    raise ValueError(f"Invalid suffix multiplier for {name}")
            styles = {
                ValueNormalizer.decimal_style(QUANTITY.fullmatch(value.strip()).group(1)
                              if kind == "quantity" and QUANTITY.fullmatch(value.strip()) else value)
                for value in valid if isinstance(value, str)
            } - {None}
            if decimal is None and len(styles) == 1:
                decimal = next(iter(styles))
            policy, policy_source = ValueNormalizer.date_policy(valid, readme, spec.get("date_format")) if kind == "date" else (None, None)
            if kind == "date" and not policy:
                requires_policy = any(
                    not isinstance(value, (dt.date, dt.datetime))
                    and not re.match(r"^\d{4}-", str(value))
                    and not re.fullmatch(r"\d{1,2}\s+(?:thg|thang)\s+\d{1,2}\s+\d{4}", norm(value))
                    and not (excel_epoch and re.fullmatch(r"\d+(?:\.\d+)?", str(value)))
                    for value in valid
                )
                if requires_policy:
                    issues.append(RawData.issue("date_policy_unresolved", column=name, reason=policy_source, review_required=True))
            converted = []
            quantity_values, quantity_units = [], []
            for i, value in enumerate(values):
                if ValueNormalizer.missing(value):
                    converted.append(None)
                    if kind == "quantity":
                        quantity_values.append(None)
                        quantity_units.append(None)
                    continue
                try:
                    if kind == "quantity":
                        text = str(value)
                        match = QUANTITY.fullmatch(text.strip())
                        numeric, unit = match.groups() if match else (value, None)
                        quantity_values.append(ValueNormalizer._parse_number(numeric, decimal, suffix_multipliers))
                        quantity_units.append(unit)
                        converted.append(text)
                        continue
                    if kind == "string":
                        result = str(value)
                        if identifier and isinstance(value, (int, float)) and not isinstance(value, bool):
                            if isinstance(value, float) and math.isfinite(value) and value.is_integer():
                                result = str(int(value))
                            fmt = raw.cell_metadata.get(f"{raw.positions[i].get('row')}:{j+1}", {}).get("number_format", "")
                            if re.fullmatch(r"0+", fmt) and result.isdigit():
                                result = result.zfill(len(fmt))
                            elif isinstance(value, float) and abs(value) >= 2**53:
                                issues.append(RawData.issue("identifier_precision_risk", column=name, position=raw.positions[i], review_required=True))
                    elif kind in ("number", "integer"):
                        result = ValueNormalizer._parse_number(value, decimal, suffix_multipliers)
                        if kind == "integer" and result != result.to_integral_value():
                            raise ValueError("not_integer")
                    elif kind == "date":
                        result = ValueNormalizer.parse_date(value, policy, excel_epoch)
                    else:
                        if value not in (True, False, "true", "false", "True", "False"):
                            raise ValueError("invalid_boolean")
                        result = value in (True, "true", "True")
                    converted.append(result)
                except (ValueError, TypeError, OverflowError) as exc:
                    if kind == "quantity":
                        converted.append(str(value))
                        quantity_values.append(None)
                        quantity_units.append(None)
                    else:
                        converted.append(None)
                    issues.append(RawData.issue("parse_failed", column=name, position=raw.positions[i], raw=str(value),
                                        target=kind, reason=str(exc), review_required=True))
            if kind == "integer":
                data[name] = pd.Series([int(value) if value is not None else None for value in converted], dtype=object)
            elif kind == "date":
                data[name] = pd.Series(converted, dtype="datetime64[ns]")
            else:
                data[name] = pd.Series(converted, dtype=object)
            columns.append({"name": name, "search_alias": colnorm(name), "type": "string" if kind == "quantity" else kind,
                            "identifier": identifier, "decimal": decimal, "date_format": policy,
                            "excel_epoch": excel_epoch, "date_evidence": policy_source, "unit": spec.get("unit"),
                            "description": description or None, "override": spec,
                            "null_count": sum(ValueNormalizer.missing(value) for value in converted)})
            if kind == "quantity":
                derived = []
                for suffix, series, dtype in (("__quantity", quantity_values, "number"), ("__unit", quantity_units, "string")):
                    derived_name = name + suffix
                    while derived_name.casefold() in {column.casefold() for column in list(raw.headers) + list(data)}:
                        derived_name += "_"
                    data[derived_name] = pd.Series(series, dtype=object)
                    columns.append({"name": derived_name, "search_alias": colnorm(derived_name), "type": dtype,
                                    "identifier": False, "decimal": decimal, "date_format": None,
                                    "date_evidence": None, "unit": None, "description": "Tách số/đơn vị, không quy đổi",
                                    "override": {}, "derived_from": name, "null_count": sum(value is None for value in series)})
                    derived.append(derived_name)
                issues.append(RawData.issue("quantity_split", column=name, derived_columns=derived, review_required=True))
        for i, row_flag in enumerate(flags):
            for flag in row_flag:
                issues.append(RawData.issue(flag, position=raw.positions[i], review_required=flag != "blank_row"))
        return pd.DataFrame(data), columns, flags, issues


QUANTITY = re.compile(r"^([+-]?\d+(?:[.,]\d+)*)\s+([^\d\s].*)$")
