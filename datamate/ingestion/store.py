"""Export one business to DuckDB and human/machine-readable audit reports."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pandas as pd

from .layered import IngestedBusiness
from .raw import RawData
class IngestionStore:
    @staticmethod
    def table_id(name):
        return "t_" + hashlib.sha256(name.encode()).hexdigest()[:16]

    @staticmethod
    def write_frame(con, target, frame, types):
        con.execute(f"CREATE TABLE {target} (" + ", ".join(f"{RawData.quote_identifier(c)} {t}" for c, t in types.items()) + ")")
        if len(frame):
            con.register("_ingestion_frame", frame)
            try:
                con.execute(f"INSERT INTO {target} SELECT " + ", ".join(f"CAST({RawData.quote_identifier(c)} AS {t})" for c, t in types.items()) + " FROM _ingestion_frame")
            finally:
                con.unregister("_ingestion_frame")

    @staticmethod
    def write_database(business: IngestedBusiness, destination, *, mode="dev"):
        from .artifacts import ArtifactPolicy
        ArtifactPolicy.validate_mode(mode)
        import duckdb
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".ingest-", suffix=".duckdb", dir=destination.parent)
        os.close(handle)
        os.unlink(temporary)  # DuckDB expects a new path, not an empty file.
        try:
            with duckdb.connect(temporary) as con:
                con.execute("BEGIN TRANSACTION")
                for schema in ("raw", "typed", "semantic", "meta"):
                    con.execute(f"CREATE SCHEMA {schema}")
                con.execute("CREATE TABLE meta.business (business_id VARCHAR, readme VARCHAR, rulebook VARCHAR, semantic_json VARCHAR)")
                con.execute("INSERT INTO meta.business VALUES (?, ?, ?, ?)", [business.business_id, business.readme, business.rulebook, RawData.json_text(business.semantic)])
                con.execute("CREATE TABLE meta.tables (logical_name VARCHAR, db_name VARCHAR, lineage_column VARCHAR, source_json VARCHAR, columns_json VARCHAR, cell_metadata_json VARCHAR)")
                con.execute("CREATE TABLE meta.issues (logical_name VARCHAR, code VARCHAR, details_json VARCHAR)")
                for problem in business.issues:
                    con.execute("INSERT INTO meta.issues VALUES (NULL, ?, ?)", [problem["code"], RawData.json_text(problem)])
                for name, table in business.tables.items():
                    db_name = IngestionStore.table_id(name)
                    lineage = "__ingest_row_id"
                    while lineage.casefold() in {c.casefold() for c in table.typed.columns}:
                        lineage = "_" + lineage
                    ids = [f"{db_name}:{i+1}" for i in range(len(table.typed))]
                    raw_frame = pd.DataFrame({"row_id": ids,
                                              "position_json": [RawData.json_text(p) for p in table.raw.positions],
                                              "values_json": [RawData.json_text(r) for r in table.raw.rows],
                                              "flags_json": [RawData.json_text(f) for f in table.flags]})
                    IngestionStore.write_frame(con, f"raw.{RawData.quote_identifier(db_name)}", raw_frame, dict.fromkeys(raw_frame.columns, "VARCHAR"))
                    frame, types = table.typed.copy(), {}
                    for column in table.columns:
                        c, kind = column["name"], column["type"]
                        if kind == "number":
                            values = [v for v in frame[c] if isinstance(v, Decimal)]
                            scale = max((max(0, -v.as_tuple().exponent) for v in values), default=0)
                            integer_digits = max((max(1, v.adjusted()+1) for v in values if v), default=1)
                            if scale + integer_digits > 38:
                                raise ValueError(f"{name}/{c}: precision exceeds DECIMAL(38); explicit policy required")
                            types[c] = f"DECIMAL(38,{scale})"
                            frame[c] = frame[c].map(lambda v: str(v) if v is not None else None)
                        else:
                            types[c] = {"integer": "HUGEINT", "date": "TIMESTAMP", "boolean": "BOOLEAN", "string": "VARCHAR"}[kind]
                    frame[lineage] = ids
                    types[lineage] = "VARCHAR"
                    # DuckDB cannot store zero data columns: empty sheets still get lineage.
                    IngestionStore.write_frame(con, f"typed.{RawData.quote_identifier(db_name)}", frame, types)
                    con.execute(f"CREATE VIEW semantic.{RawData.quote_identifier(db_name)} AS SELECT * FROM typed.{RawData.quote_identifier(db_name)}")
                    con.execute("INSERT INTO meta.tables VALUES (?, ?, ?, ?, ?, ?)",
                                [name, db_name, lineage, RawData.json_text(table.raw.source), RawData.json_text(table.columns), RawData.json_text(table.raw.cell_metadata)])
                    if table.issues:
                        issue_frame = pd.DataFrame({"logical_name": [name] * len(table.issues),
                                                    "code": [p["code"] for p in table.issues],
                                                    "details_json": [RawData.json_text(p) for p in table.issues]})
                        con.register("_issue_frame", issue_frame)
                        con.execute("INSERT INTO meta.issues SELECT * FROM _issue_frame")
                        con.unregister("_issue_frame")
                # Explicit, locally reviewed SELECT SQL only; no autogenerated business rules.
                for name, sql in business.semantic["views"].items():
                    if not sql.lstrip().lower().startswith("select ") or ";" in sql:
                        raise ValueError(f"View {name}: use a single SELECT without semicolons")
                    con.execute(f"CREATE VIEW semantic.{RawData.quote_identifier(name)} AS {sql}")
                from .organization.analyze import OrganizationAnalyzer
                organization = OrganizationAnalyzer.analyze_connection(con)
                OrganizationAnalyzer.persist_analysis(con, organization)
                from .organization.workflow import OrganizationWorkflow
                OrganizationWorkflow.organize_connection(con, organization)
                from .catalog import CatalogService
                catalog = CatalogService.install_catalog(con, business, destination.name, organization)
                con.execute("COMMIT")
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        from .documentation import DatabaseGuide
        DatabaseGuide.write_guide(destination)
        from datamate.storage.json_store import JsonStore
        JsonStore.write(catalog, destination.parent / "catalog.json")
        if business.manifest is not None:
            JsonStore.write(business.manifest, destination.parent / "manifest.json")
        else:
            (destination.parent / "manifest.json").unlink(missing_ok=True)
        if mode == "dev":
            from .organization.analyze import OrganizationAnalyzer
            from .organization.build import OrganizationBuilder
            from .organization.model import OrganizationModelBuilder
            OrganizationAnalyzer.write_analysis_report(organization, destination.parent)
            OrganizationBuilder.write_build_report(organization["stage2"], destination.parent)
            OrganizationModelBuilder.write_model_report(organization["stage3"], destination.parent)
        else:
            ArtifactPolicy.clean_reports(destination.parent)
        return destination

    @staticmethod
    def write_report(business, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tables, counts = [], Counter(p["code"] for p in business.issues)
        for name, table in business.tables.items():
            codes = Counter(p["code"] for p in table.issues)
            counts.update(codes)
            tables.append({"name": name, "db_name": IngestionStore.table_id(name), "rows": len(table.typed),
                           "source": table.raw.source, "columns": table.columns,
                           "issue_counts": dict(codes), "issue_samples": table.issues[:50]})
        report = {"business_id": business.business_id, "tables": tables, "issues": business.issues,
                  "issue_counts": dict(counts), "semantic": business.semantic}
        (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        lines = [f"# Ingestion {business.business_id}", "", "## Thống kê", "",
                 f"- Bảng: {len(tables)}; bản ghi: {sum(t['rows'] for t in tables)}.",
                 "- Raw giữ nguồn; Typed giữ toàn bộ bản ghi; Semantic mặc định là view toàn bộ Typed.",
                 "- Không tự xóa dòng tổng/trùng/tiêu đề lặp; phải rà cờ trước khi tổng hợp.",
                 "- Lỗi chuyển kiểu -> NULL ở Typed, giá trị gốc còn ở Raw và meta.issues.", "",
                 "## Sự kiện và cảnh báo", "", "| Mã | Số lượng |", "|---|---:|"]
        lines.extend(f"| {code} | {count} |" for code, count in sorted(counts.items()))
        lines.extend(["", "## Các bảng", ""])
        for table in tables:
            lines.extend([f"### {table['name']}", "", f"- SQL: `typed.{table['db_name']}` / `semantic.{table['db_name']}`.",
                          f"- Số dòng: {table['rows']}; nguồn: `{table['source']['file']}`.", "",
                          "| Cột gốc | Kiểu | Quy tắc |", "|---|---|---|"])
            for c in table["columns"]:
                detail = f"date={c['date_format']}; decimal={c['decimal']}; unit={c['unit']}"
                lines.append(f"| {c['name'].replace('|', '/')} | {c['type']} | {detail} |")
            lines.append("")
        lines.extend(["## Cần rà soát", "", "Chi tiết vị trí và ví dụ: report.json; toàn bộ cảnh báo: meta.issues trong DuckDB.",
                      "Đợt 1 phân tích khóa/quan hệ: ORGANIZATION.md, organization.json và meta.key_candidates / meta.relationships. "
                      "Accepted là kết luận theo policy trên snapshot. Đợt 2 gộp/nối: ORGANIZATION_BUILD.md; đợt 3 tách/tổng hợp nguồn: ORGANIZATION_MODEL.md. Chưa áp ràng buộc vật lý, doanh thu hoặc quy đổi đơn vị."])
        (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report
