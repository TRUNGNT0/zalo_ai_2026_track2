"""Raw source records shared by ingestion, typing, catalog, and persistence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from datamate.ingestion.sources.adapters.common import SourceTable


@dataclass
class RawTable:
    name: str
    source: dict
    headers: list[str]
    rows: list[list]
    positions: list[dict]
    issues: list[dict] = field(default_factory=list)
    cell_metadata: dict = field(default_factory=dict)


class RawData:
    @staticmethod
    def issue(code, **details):
        return {"code": code, **details}

    @staticmethod
    def json_text(value) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def quote_identifier(value: str) -> str:
        return '"' + value.replace('"', '""') + '"'

    @staticmethod
    def source_info(path: Path) -> dict:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "file": str(path.resolve()),
            "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size,
        }

    @staticmethod
    def header_info(raw: RawTable) -> tuple[int | None, str]:
        if raw.source["format"] in ("sqlite", "parquet"):
            return None, raw.source["format"] + "_schema"
        header = raw.source.get("header", {})
        method = raw.source.get("header_method", "heuristic")
        return header.get("row", header.get("record")), {
            "heuristic": "heuristic_header"
        }.get(method, method)

    @staticmethod
    def adapt_source_table(path: Path, root: Path, source_table: SourceTable) -> RawTable:
        rows = [list(row) for row in source_table.rows]
        relative = path.relative_to(root).as_posix()
        original_headers = list(source_table.source_headers)
        headers = [
            str(value) if value is not None and str(value).strip() else f"__unnamed_{index + 1}"
            for index, value in enumerate(original_headers)
        ]
        positions = source_table.positions or []
        if len(positions) != len(rows):
            positions = [{"record": index + 1} for index in range(len(rows))]
        format_name = path.suffix.casefold().lstrip(".")
        if format_name in ("xls", "xlsx", "xlsm", "sqlite", "sqlite3", "db"):
            name = f"{path.stem}::{source_table.source_object}"
        else:
            name = path.stem
        if path.parent != root:
            name = f"{path.parent.relative_to(root).as_posix()}/{name}"
        source = RawData.source_info(path) | {
            "format": "xlsx" if format_name == "xlsm" else format_name,
            "relative_file": relative,
            "source_object": source_table.source_object,
            "header": {"row": source_table.header_row_number},
            "header_method": source_table.header_status,
            "original_headers": original_headers,
            "source_types": list(source_table.source_types),
            "relationships": list(source_table.relationships),
        }
        if source_table.excel_epoch is not None:
            source["excel_epoch"] = source_table.excel_epoch
        return RawTable(
            name=name,
            source=source,
            headers=headers,
            rows=rows,
            positions=positions,
            issues=list(source_table.issues or []),
            cell_metadata=dict(source_table.cell_metadata or {}),
        )

    @staticmethod
    def read_tables(path: Path, root: Path, options=None) -> list[RawTable]:
        from datamate.ingestion.sources.dispatch import SourceReader

        relative = path.relative_to(root).as_posix()
        return [
            RawData.adapt_source_table(path, root, source_table)
            for source_table in SourceReader.read_tables(path, relative, options or {})
        ]
