"""Dispatch supported source files through their format-specific adapters."""
from __future__ import annotations

from pathlib import Path

from datamate.ingestion.sources.adapters.csv_adapter import CsvAdapter
from datamate.ingestion.sources.adapters.excel_adapter import ExcelAdapter
from datamate.ingestion.sources.adapters.parquet_adapter import ParquetAdapter
from datamate.ingestion.sources.adapters.sqlite_adapter import SqliteAdapter
from datamate.ingestion.sources.adapters.common import SourceTable

ADAPTERS = {
    ".csv": CsvAdapter,
    ".xls": ExcelAdapter,
    ".xlsx": ExcelAdapter,
    ".xlsm": ExcelAdapter,
    ".sqlite": SqliteAdapter,
    ".sqlite3": SqliteAdapter,
    ".db": SqliteAdapter,
    ".parquet": ParquetAdapter,
}


class SourceReader:
    @staticmethod
    def read_tables(path: Path, source_file: str, options=None) -> list[SourceTable]:
        suffix = path.suffix.casefold()
        adapter = ADAPTERS.get(suffix)
        if adapter is None:
            raise ValueError(f"Unsupported source extension: {suffix or '<none>'}")
        options = options or {}
        if adapter in (CsvAdapter, ExcelAdapter):
            source_tables = adapter.read(path, source_file, options.get("expected_headers", ()))
            if isinstance(source_tables, SourceTable):
                source_tables = [source_tables]
        else:
            source_tables = adapter.read(path, source_file)
        materialized = []
        for source_table in source_tables:
            rows = list(source_table.rows)
            source_table.rows = iter(rows)
            materialized.append(source_table)
        return materialized
