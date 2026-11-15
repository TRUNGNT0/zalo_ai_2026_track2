from pathlib import Path
from datetime import datetime
import json
import sqlite3

from jsonschema import Draft202012Validator
from openpyxl import Workbook
import duckdb

from datamate.ingestion.processing.normalization import IdentifierNormalizer, ValueNormalizer
from datamate.ingestion.processing.pipeline import IngestionPipeline

ROOT = Path(__file__).resolve().parents[1]
CATALOG_SCHEMA_PATH = ROOT / "schemas" / "catalog.schema.json"


class TestIngestion:
    @staticmethod
    def _write_source_files(business_dir: Path) -> None:
        business_dir.mkdir()
        (business_dir / "orders.csv").write_text(
            "\ufeffMã đơn,Thành tiền\n001,1,200\n002,2,500\n".replace(
                "001,1,200", '"001","1,200"'
            ).replace("002,2,500", '"002","2,500"'),
            encoding="utf-8",
        )

        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = "Tháng 1"
        worksheet.append(["BÁO CÁO BÁN HÀNG"])
        worksheet.append([None])
        worksheet.append(["Mã đơn", "Số lượng"])
        worksheet.append(["003", 2])
        worksheet.append(["004", 5])
        workbook.save(business_dir / "sales.xlsx")

        connection = sqlite3.connect(business_dir / "source.sqlite")
        connection.execute(
            'CREATE TABLE orders ("MaHoaDon" TEXT PRIMARY KEY, "SoLuong" INTEGER)'
        )
        connection.execute(
            'CREATE TABLE line_items ("order_id" TEXT REFERENCES orders("MaHoaDon"), '
            '"quantity" INTEGER)'
        )
        connection.executemany(
            'INSERT INTO orders VALUES (?, ?)',
            [("0005", 4), ("0006", 6)],
        )
        connection.execute('INSERT INTO line_items VALUES ("0005", 2)')
        connection.commit()
        connection.close()

        parquet_path = (business_dir / "events.parquet").as_posix().replace("'", "''")
        parquet_connection = duckdb.connect(":memory:")
        parquet_connection.execute(
            f"COPY (SELECT '0007' AS \"MaDon\", 7 AS \"SoLuong\") "
            f"TO '{parquet_path}' (FORMAT PARQUET)"
        )
        parquet_connection.close()

    @staticmethod
    def _manifest() -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "business_id": "biz_test",
            "tables": [
                {
                    "table_role": "sales_records",
                    "source_files": ["orders.csv", "sales.xlsx"],
                    "columns": {
                        "Mã đơn": "mã đơn",
                        "Thành tiền": "thành tiền sau giảm",
                        "Số lượng": "số lượng",
                    },
                    "source_refs": ["readme.sales"],
                },
                {
                    "table_role": "sales_records",
                    "source_files": ["source.sqlite"],
                    "columns": {
                        "orders.MaHoaDon": "mã hóa đơn",
                        "orders.SoLuong": "số lượng",
                    },
                    "source_refs": ["readme.sqlite"],
                },
                {
                    "table_role": "sales_records",
                    "source_files": ["events.parquet"],
                    "columns": {
                        "MaDon": "mã đơn",
                        "SoLuong": "số lượng",
                    },
                    "source_refs": ["readme.parquet"],
                },
            ],
            "facts": [],
            "unresolved": [],
        }

    def test_normalize_identifier_is_stable_and_disambiguates_collisions(self) -> None:
        assert IdentifierNormalizer.normalize("Mã đơn hàng") == "ma_don_hang"
        assert IdentifierNormalizer.unique(("Mã hàng", "Ma hang", None)) == (
            "ma_hang",
            "ma_hang_2",
            "column_3",
        )

    def test_profile_preserves_identifiers_and_parses_grouped_numeric_values(self) -> None:
        assert ValueNormalizer.is_identifier("ma_don", "Mã đơn")
        assert ValueNormalizer.is_identifier(None, "MaHoaDon")
        assert ValueNormalizer.infer_type(["001", "002"], force_text=True) == "VARCHAR"
        assert ValueNormalizer.infer_type(["1,200", "2,500"]) == "BIGINT"

    def test_ingestion_loads_supported_formats_and_builds_catalog(self, tmp_path: Path) -> None:
        business_dir = tmp_path / "biz_test"
        TestIngestion._write_source_files(business_dir)
        manifest_path = tmp_path / "biz_test.manifest.json"
        manifest_path.write_text(
            json.dumps(TestIngestion._manifest(), ensure_ascii=False),
            encoding="utf-8",
        )
        output_dir = tmp_path / "cache" / "biz_test"

        catalog = IngestionPipeline.ingest(business_dir, output_dir, manifest_path,
            overrides={"tables": {"orders": {"columns": {"Thành tiền": {"decimal": "."}}}}})

        Draft202012Validator.check_schema(
            json.loads(CATALOG_SCHEMA_PATH.read_text(encoding="utf-8"))
        )
        Draft202012Validator(
            json.loads(CATALOG_SCHEMA_PATH.read_text(encoding="utf-8"))
        ).validate(catalog)
        assert (output_dir / "data.duckdb").is_file()
        assert json.loads((output_dir / "catalog.json").read_text(encoding="utf-8")) == catalog
        assert len(catalog["tables"]) == 5
        assert {table["table_role"] for table in catalog["tables"]} == {
            "sales_records",
            None,
        }

        by_file = {table["source_files"][0]: table for table in catalog["tables"]}
        csv_columns = {
            column["source_name"]: column
            for column in by_file["orders.csv"]["columns"]
        }
        assert csv_columns["Mã đơn"]["physical_name"] == "ma_don"
        assert csv_columns["Mã đơn"]["semantic_name"] == "ma_don"
        assert csv_columns["Mã đơn"]["duckdb_type"] == "VARCHAR"
        assert csv_columns["Thành tiền"]["duckdb_type"] == "DECIMAL(38,0)"
        assert by_file["sales.xlsx"]["header"]["source_row_number"] == 3
        assert by_file["sales.xlsx"]["row_count"] == 2
        assert by_file["source.sqlite"]["columns"][0]["source_name"] == "MaHoaDon"
        assert by_file["source.sqlite"]["columns"][0]["semantic_name"] == "ma_hoa_don"
        assert by_file["source.sqlite"]["columns"][0]["source_type"] == "TEXT"
        assert catalog["relationships"] == [
            {
                "relationship_id": "sqlite_fk:source.sqlite:line_items:0",
                "from_table": "source_line_items",
                "from_columns": ["order_id"],
                "to_table": "source_orders",
                "to_columns": ["ma_hoa_don"],
                "status": "declared_not_data_verified",
                "source": "sqlite_foreign_key",
                "source_file": "source.sqlite",
            }
        ]

        connection = duckdb.connect(str(output_dir / "data.duckdb"), read_only=True)
        try:
            rows = connection.execute(
                'SELECT "ma_don", "thanh_tien" FROM "orders" ORDER BY "ma_don"'
            ).fetchall()
            assert rows == [("001", 1200), ("002", 2500)]
            parquet_table = by_file["events.parquet"]["physical_name"]
            rows = connection.execute(
                f'SELECT "ma_don", "so_luong" FROM "{parquet_table}"'
            ).fetchall()
            assert rows == [("0007", 7)]
        finally:
            connection.close()

    def test_ingestion_preserves_summary_in_raw_and_marks_typed_parse_failure(
        self, tmp_path: Path
    ) -> None:
        business_dir = tmp_path / "biz_mixed"
        business_dir.mkdir()
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.append(["Ngày"])
        for _ in range(1000):
            worksheet.append([datetime(2025, 1, 1, 8, 30)])
        worksheet.append(["Tổng tháng"])
        workbook.save(business_dir / "sales.xlsx")

        output_dir = tmp_path / "cache" / "biz_mixed"
        catalog = IngestionPipeline.ingest(business_dir, output_dir)

        table = catalog["tables"][0]
        assert table["columns"][0]["duckdb_type"] == "TIMESTAMP"
        assert any(
            issue["topic"] == "ingestion:sales::Sheet:parse_failed"
            for issue in catalog["unresolved"]
        )
        connection = duckdb.connect(
            str(output_dir / "data.duckdb"), read_only=True
        )
        try:
            row_count, failed = connection.execute(
                f'SELECT COUNT(*), COUNT(*) FILTER (WHERE ngay IS NULL) FROM "{table["physical_name"]}"'
            ).fetchone()
            assert row_count == 1001 and failed == 1
            assert table["columns"][0]["profile"]["parse_failed_count"] == 1
            raw = table["lineage"]["raw_table"]
            assert connection.execute(f"SELECT COUNT(*) FROM {raw} WHERE values_json LIKE '%Tổng tháng%'").fetchone()[0] == 1

        finally:
            connection.close()

    def test_ingestion_creates_conservative_manifest_before_catalog(self, tmp_path: Path) -> None:
        split_dir = tmp_path / "split"
        business_dir = split_dir / "data" / "biz_auto"
        business_dir.mkdir(parents=True)
        (business_dir / "README.md").write_text(
            "## Cột trong `sales.csv`\n- **Amount**: thành tiền dòng\n",
            encoding="utf-8",
        )
        (business_dir / "sales.csv").write_text("Amount,Status\n10,paid\n")
        manifest_path = split_dir / "manifests" / "biz_auto.json"

        catalog = IngestionPipeline.ingest(
            business_dir, tmp_path / "cache" / "biz_auto", rulebook="# Rules\n"
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        Draft202012Validator(
            json.loads((ROOT / "schemas" / "business_manifest.v2.schema.json").read_text())
        ).validate(manifest)
        assert manifest["schema_version"] == "2.0"
        assert all(table["table_role"] == "unknown" for table in manifest["tables"])
        assert any(issue["topic"].startswith("table_role:") for issue in manifest["unresolved"])
        assert any(issue["topic"].startswith("grain:") for issue in manifest["unresolved"])
        manifest_columns = {
            column["source_name"]: column
            for table in manifest["tables"] for column in table["columns"]
        }
        assert manifest_columns["Amount"]["semantic_name"] == "line_amount"
        assert manifest_columns["Amount"]["status"] == "documented"
        assert manifest_columns["Status"]["status"] == "unresolved"
        amount = next(
            column
            for table in catalog["tables"]
            for column in table["columns"]
            if column["source_name"] == "Amount"
        )
        assert amount["semantic_name"] == "line_amount"
        assert amount["semantic_status"] == "documented"
        assert any(issue["topic"].endswith(":Status") for issue in catalog["unresolved"])

    def test_quantity_parse_failure_is_attached_to_numeric_view_column(self, tmp_path: Path) -> None:
        business_dir = tmp_path / "biz_quantity"
        business_dir.mkdir()
        (business_dir / "items.csv").write_text(
            "Mã;Số lượng\n001;not-a-quantity\n002;3 chai\n",
            encoding="utf-8",
        )

        catalog = IngestionPipeline.ingest(business_dir, tmp_path / "cache" / "biz_quantity")
        columns = {column["source_name"]: column for column in catalog["tables"][0]["columns"]}

        assert columns["Số lượng"]["profile"]["parse_failed_count"] == 0
        assert columns["Số lượng__quantity"]["profile"]["parse_failed_count"] == 1
