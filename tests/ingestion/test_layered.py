import json
import sqlite3
from decimal import Decimal

import duckdb
import pandas as pd
import pytest
from openpyxl import Workbook

from datamate.ingestion.layered import BusinessIngestion
from datamate.ingestion.processing.normalization import ValueNormalizer
from datamate.ingestion.raw import RawTable
from datamate.ingestion.store import IngestionStore


def test_csv_source_locale_and_no_dropped_rows(tmp_path):
    (tmp_path / "README.md").write_text("Ngày dạng dd/mm/yyyy.", encoding="utf-8")
    (tmp_path / "sales.csv").write_text(
        'Báo cáo bán hàng\nMã KH;Ngày;Thành tiền\n00123;03/04/2026;1.234,50\n'
        '00123;03/04/2026;1.234,50\nMã KH;Ngày;Thành tiền\nTổng cộng;;2.469,00\n', encoding="utf-8-sig")
    business = BusinessIngestion.ingest_business(tmp_path)
    table = business.tables["sales"]
    assert list(table.typed.columns) == ["Mã KH", "Ngày", "Thành tiền"]
    assert len(table.typed) == 4
    assert table.typed.iloc[0, 0] == "00123"
    assert table.typed.iloc[0, 1] == pd.Timestamp("2026-04-03")
    assert table.typed.iloc[0, 2] == Decimal("1234.50")
    assert table.raw.positions[0]["line_start"] == 3
    assert "possible_duplicate" in table.flags[1]
    assert "repeated_header" in table.flags[2]
    assert "possible_total" in table.flags[3]
    assert table.raw.rows[2][1] == "Ngày"
    assert len(table.raw.source["sha256"]) == 64


def test_ambiguous_values_are_null_not_guessed(tmp_path):
    (tmp_path / "sales.csv").write_text("Mã,Ngày,Thành tiền\n001,03/04/2026,\"1,234\"\n", encoding="utf-8")
    table = BusinessIngestion.ingest_business(tmp_path).tables["sales"]
    assert pd.isna(table.typed.iloc[0, 1])
    assert table.typed.iloc[0, 2] is None
    assert table.raw.rows[0] == ["001", "03/04/2026", "1,234"]
    assert any(i["code"] == "date_policy_unresolved" for i in table.issues)
    table = BusinessIngestion.ingest_business(tmp_path, overrides={"tables": {"sales": {"columns": {
        "Ngày": {"date_format": "mdy"}, "Thành tiền": {"decimal": "."}}}}}).tables["sales"]
    assert table.typed.iloc[0, 1] == pd.Timestamp("2026-03-04")
    assert table.typed.iloc[0, 2] == Decimal(1234)


def test_excel_formula_missing_cache_and_id_format(tmp_path):
    book = Workbook()
    ws = book.active
    ws.title = "Tháng 1"
    ws.append(["BÁO CÁO"])
    ws.append(["Mã khách", "Thành tiền"])
    ws.append([123, "=100+20"])
    ws["A3"].number_format = "00000"
    book.create_sheet("Trống")
    book.save(tmp_path / "sales.xlsx")
    business = BusinessIngestion.ingest_business(tmp_path)
    table = business.tables["sales::Tháng 1"]
    assert table.typed.iloc[0, 0] == "00123"
    assert table.typed.iloc[0, 1] is None
    assert table.columns[1]["type"] == "number"
    assert table.raw.cell_metadata["3:2"]["formula"] == "=100+20"
    assert any(i["code"] == "formula_cache_missing" for i in table.issues)
    assert "sales::Trống" in business.tables


def test_csv_multiline_and_literal_na(tmp_path):
    (tmp_path / "a.csv").write_text('Mã;Ghi chú\n001;"hai\ndòng"\n002;NA\n', encoding="utf-8")
    table = BusinessIngestion.ingest_business(tmp_path).tables["a"]
    assert table.raw.positions[0] == {"record": 2, "line_start": 2, "line_end": 3}
    assert table.typed.iloc[1, 1] == "NA"


def test_sqlite_quotes_without_rowid_and_nulls(tmp_path):
    with sqlite3.connect(tmp_path / "source.sqlite") as con:
        con.execute('CREATE TABLE "a""b" ("Mã" TEXT PRIMARY KEY, "Số lượng" INT) WITHOUT ROWID')
        con.execute('INSERT INTO "a""b" VALUES (?, ?)', ("001", None))
    table = BusinessIngestion.ingest_business(tmp_path).tables['source::a"b']
    assert table.raw.positions[0]["primary_key"] == {"Mã": "001"}
    assert table.typed.iloc[0, 0] == "001"
    assert table.typed.iloc[0, 1] is None


@pytest.mark.parametrize("value,expected", [("1.234,50", "1234.50"), ("1,234.50", "1234.50"),
                                            ("1,234,567", "1234567"), ("12,5", "12.5"),
                                            ("(123)", "-123"), ("1 234,50", "1234.50")])
def test_number_formats(value, expected):
    assert ValueNormalizer.parse_number(value) == Decimal(expected)


def test_invalid_number_grouping():
    for value in ("1,234", "12.34.567", "NaN", "1.23,45"):
        with pytest.raises(ValueError):
            ValueNormalizer.parse_number(value)


def test_currency_suffix_requires_explicit_policy():
    with pytest.raises(ValueError):
        ValueNormalizer.parse_number("29k")
    assert ValueNormalizer.parse_number("29k", ",", {"k": 1000}) == Decimal(29000)
    assert ValueNormalizer.parse_number("39.000đ", ",", {"đ": 1}) == Decimal(39000)
    large = "12345678901234567890123456789012345678"
    assert ValueNormalizer.parse_number(large) == Decimal(large)


def test_vietnamese_dates_and_explicit_excel_serial(tmp_path):
    (tmp_path / "a.csv").write_text('Ngày\n1 thg 6 2025\n45658\n', encoding="utf-8")
    plain = BusinessIngestion.ingest_business(tmp_path).tables["a"]
    assert plain.typed.iloc[0, 0] == pd.Timestamp("2025-06-01")
    assert pd.isna(plain.typed.iloc[1, 0])
    explicit = BusinessIngestion.ingest_business(tmp_path, overrides={"tables": {"a": {"columns": {
        "Ngày": {"excel_epoch": "1900"}}}}}).tables["a"]
    assert explicit.typed.iloc[1, 0] == pd.Timestamp("2025-01-01")


def test_excel_serial_date_uses_source_workbook_epoch(tmp_path):
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.append(["Mã hóa đơn", "Ngày bán"])
        worksheet.append(["O-1", 46143])
        workbook.save(tmp_path / "sales.xlsx")

        table = BusinessIngestion.ingest_business(tmp_path).tables["sales::Sheet"]

        assert table.raw.source["excel_epoch"] == "1900"
        assert table.typed.loc[0, "Ngày bán"] == pd.Timestamp("2026-05-01")
        assert not any(
            issue["code"] == "parse_failed" and issue["column"] == "Ngày bán"
            for issue in table.issues
        )


def test_readme_description_preserves_identifier_and_types_quantity(tmp_path):
    (tmp_path / "README.md").write_text('## Cột trong `a.csv`\n- **Số KH**: mã khách\n- **Quy đổi**: số lượng đã đổi\n', encoding="utf-8")
    (tmp_path / "a.csv").write_text('Số KH,Quy đổi\n001,24\n', encoding="utf-8")
    table = BusinessIngestion.ingest_business(tmp_path).tables["a"]
    assert table.typed.iloc[0, 0] == "001"
    assert table.typed.iloc[0, 1] == Decimal(24)


def test_readme_date_conflict_is_reported(tmp_path):
    (tmp_path / "README.md").write_text('Ngày dd/mm/yyyy', encoding="utf-8")
    (tmp_path / "a.csv").write_text('Ngày\n12/31/2026\n', encoding="utf-8")
    table = BusinessIngestion.ingest_business(tmp_path).tables["a"]
    assert pd.isna(table.typed.iloc[0, 0])
    assert any(i["code"] == "date_policy_unresolved" and i["reason"] == "conflict" for i in table.issues)


def test_overrides_with_unknown_column_are_rejected(tmp_path):
    (tmp_path / "a.csv").write_text('Mã\n001\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unknown override columns"):
        BusinessIngestion.ingest_business(tmp_path, overrides={"tables": {"a": {"columns": {"typo": {"type": "number"}}}}})


def test_quantity_split_preserves_source_units_and_values(tmp_path):
    (tmp_path / "a.csv").write_text('Mã;Số lượng\n001;1 bao 25kg\n002;3 chai\n003;2\n', encoding="utf-8")
    table = BusinessIngestion.ingest_business(tmp_path).tables["a"]
    assert table.typed["Số lượng"].tolist() == ["1 bao 25kg", "3 chai", "2"]
    assert table.typed["Số lượng__quantity"].tolist() == [Decimal(1), Decimal(3), Decimal(2)]
    assert table.typed["Số lượng__unit"].tolist() == ["bao 25kg", "chai", None]
    assert table.raw.headers == ["Mã", "Số lượng"]
    assert table.columns[2]["derived_from"] == "Số lượng"
    assert any(i["code"] == "quantity_split" for i in table.issues)


def test_shared_calendar_explicit_types(tmp_path):
    p = tmp_path / "calendar.csv"
    p.write_text('date,weekday,lunar_day,lunar_month,lunar_year,lunar_leap,holiday\n2026-01-01,4,13,11,2025,false,Tết\n', encoding="utf-8")
    shared = BusinessIngestion.ingest_calendar(p)
    frame = shared.frames["calendar"]
    assert frame.iloc[0]["weekday"] == 4
    assert frame.iloc[0]["lunar_leap"] is False
    assert frame.iloc[0]["date"] == pd.Timestamp("2026-01-01")


def test_categorical_descriptions_do_not_become_numbers(tmp_path):
    (tmp_path / "README.md").write_text('## Cột trong `a.csv`\n'
        '- **Tên hàng**: tên hàng; dòng Dùng điểm hoặc Phí ship\n'
        '- **HĐ**: số hoá đơn, kể cả dòng Dùng điểm\n'
        '- **Tiến độ**: trạng thái đơn\n'
        '- **Thanh toán bằng**: hình thức thanh toán: cash tiền mặt, debt ghi nợ\n', encoding="utf-8")
    (tmp_path / "a.csv").write_text('Tên hàng,HĐ,Tiến độ,Thanh toán bằng,Giá đỡ\nTã quần,ORD001,Xong,cash,CL 9D\n', encoding="utf-8")
    table = BusinessIngestion.ingest_business(tmp_path).tables["a"]
    assert table.typed.iloc[0].tolist() == ['Tã quần', 'ORD001', 'Xong', 'cash', 'CL 9D']
    assert all(c['type'] == 'string' for c in table.columns)


def test_retype_preserves_raw_order_formula_and_quantity(tmp_path):
    from datamate.ingestion.retype import RetypeService
    (tmp_path / "a.csv").write_text('Mã;Số lượng\n' + ''.join(f'{i:03};{i} chai\n' for i in range(1, 13)), encoding="utf-8")
    original = BusinessIngestion.ingest_business(tmp_path)
    database = IngestionStore.write_database(original, tmp_path / "out" / "data.duckdb")
    regenerated = RetypeService.retype_business(database)
    assert regenerated.tables["a"].raw.rows == original.tables["a"].raw.rows
    pd.testing.assert_frame_equal(regenerated.frames["a"], original.frames["a"])


def test_retype_keeps_partial_read_errors(tmp_path):
    from datamate.ingestion.retype import RetypeService
    (tmp_path / "a.csv").write_text('Mã\n001\n', encoding="utf-8")
    (tmp_path / "broken.xlsx").write_text('broken workbook')
    original = BusinessIngestion.ingest_business(tmp_path, strict=False)
    database = IngestionStore.write_database(original, tmp_path / "out" / "data.duckdb")
    assert RetypeService.retype_business(database).issues == original.issues
    with pytest.raises(ValueError, match="Unknown override tables"):
        RetypeService.retype_business(database, {"tables": {"typo": {}}})


def test_export_exact_decimals_and_lineage(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "sales.csv").write_text('Mã,Thành tiền\n001,0.10\n002,0.20\n', encoding="utf-8")
    business = BusinessIngestion.ingest_business(source)
    report = IngestionStore.write_report(business, tmp_path / "out")
    database = IngestionStore.write_database(business, tmp_path / "out" / "data.duckdb")
    table_id = report["tables"][0]["db_name"]
    with duckdb.connect(str(database), read_only=True) as con:
        assert con.execute(f'SELECT SUM("Thành tiền") FROM typed.{table_id}').fetchone()[0] == Decimal("0.30")
        result = con.execute(f'SELECT r.values_json FROM typed.{table_id} t JOIN raw.{table_id} r ON t.__ingest_row_id=r.row_id WHERE t."Mã"=\'001\'').fetchone()[0]
        assert json.loads(result) == ["001", "0.10"]
        assert con.execute(f"SELECT COUNT(*) FROM semantic.{table_id}").fetchone()[0] == 2
    assert (tmp_path / "out" / "REPORT.md").exists()
    from datamate.ingestion.review import build_review
    entry = {"business_id": source.name, "tables": 1, "rows": 2, "status": "complete", "issue_counts": report["issue_counts"]}
    # The split-level reviewer expects a database under each business folder.
    split = tmp_path / "split"
    (split / source.name).mkdir(parents=True)
    import shutil
    shutil.copyfile(database, split / source.name / "data.duckdb")
    (split / "summary.json").write_text(json.dumps({"businesses": [entry]}))
    assert build_review(split)["totals"]["rows"] == 2


def test_failures_are_not_silently_dropped(tmp_path):
    (tmp_path / "broken.xlsx").write_text("not an Excel workbook")
    with pytest.raises(RuntimeError, match="broken.xlsx"):
        BusinessIngestion.ingest_business(tmp_path)
    partial = BusinessIngestion.ingest_business(tmp_path, strict=False)
    assert partial.issues[0]["code"] == "file_read_failed"


def test_opt_in_split_uses_layered_ingestion(split_dirs):
    from datamate.ingestion.datasets.loader import DatasetLoader
    train, _, _ = split_dirs
    biz = DatasetLoader.load_split(train, layered=True).business("b1")
    assert biz.ingestion is not None
    assert biz.tables["sales"]["thanh_tien"].sum() == Decimal(190000)
