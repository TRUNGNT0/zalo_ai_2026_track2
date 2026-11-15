import sqlite3
from pathlib import Path

import duckdb
from openpyxl import Workbook

from datamate.ingestion.raw import RawData
from datamate.ingestion.sources.adapters.csv_adapter import CsvAdapter
from datamate.ingestion.sources.adapters.excel_adapter import ExcelAdapter
from datamate.ingestion.sources.adapters.parquet_adapter import ParquetAdapter
from datamate.ingestion.sources.adapters.sqlite_adapter import SqliteAdapter


def test_source_reading_dispatches_through_format_adapters(tmp_path, monkeypatch):
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text("ID,Amount\n001,10\n", encoding="utf-8")

    workbook = Workbook()
    workbook.active.append(["ID", "Amount"])
    workbook.active.append(["002", 20])
    excel_path = tmp_path / "sales.xlsx"
    workbook.save(excel_path)

    sqlite_path = tmp_path / "sales.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute("CREATE TABLE sales (id TEXT, amount INTEGER)")
        connection.execute("INSERT INTO sales VALUES ('003', 30)")

    parquet_path = tmp_path / "sales.parquet"
    with duckdb.connect() as connection:
        connection.execute(
            "COPY (SELECT '004' AS id, 40 AS amount) TO ? (FORMAT PARQUET)",
            [str(parquet_path)],
        )

    adapters = (CsvAdapter, ExcelAdapter, SqliteAdapter, ParquetAdapter)
    calls = []
    for adapter in adapters:
        original = adapter.read

        def spy(path, source_file, *args, _original=original, _adapter=adapter):
            calls.append(_adapter)
            return _original(path, source_file, *args)

        monkeypatch.setattr(adapter, "read", staticmethod(spy))

    sources = {}
    for path in (csv_path, excel_path, sqlite_path, parquet_path):
        raw_tables = RawData.read_tables(path, tmp_path)
        assert raw_tables
        sources[path.suffix] = raw_tables

    assert set(calls) == set(adapters)
    assert [sources[".csv"][0].source["format"],
            sources[".xlsx"][0].source["format"],
            sources[".sqlite"][0].source["format"],
            sources[".parquet"][0].source["format"]] == [
                "csv", "xlsx", "sqlite", "parquet"
            ]
    assert sources[".csv"][0].rows == [["001", "10"]]
    assert sources[".xlsx"][0].rows == [["002", 20]]
    assert sources[".sqlite"][0].rows == [["003", 30]]
    assert sources[".parquet"][0].rows == [["004", 40]]
