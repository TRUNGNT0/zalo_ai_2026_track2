"""Build SQL context from adapter-produced DuckDB and catalog, never answer keys."""
import hashlib
import json
import re
from pathlib import Path

from datamate.ingestion.processing.pipeline import IngestionPipeline
from datamate.ingestion.catalog import PIPELINE_VERSION, CatalogService
from datamate.ingestion.sources.adapters.common import TabularLayout
from datamate.text import norm
from datamate.tools.sql import SQLExecutor
from datamate.tools.timeutil import describe_period


class SQLContextBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.prepared = {}

    def prepare(self, split, business_id):
        root = Path(split.root).resolve()
        key = (str(root), business_id)
        if key in self.prepared:
            return self.prepared[key]
        business_dir = root / "data" / business_id
        if business_dir.resolve().parent != root / "data":
            raise ValueError("business_id không hợp lệ")
        output = Path(self.cfg.database_dir) / business_id / "current"
        database = output / "data.duckdb"
        if self.cfg.ingestion_mode == "direct":
            from datamate.ingest import load_business, profile

            direct_db, meta = load_business(
                root, business_id, canon=self.cfg.canon_names, database_path=database
            )
            try:
                self._build_direct_layers(direct_db, business_id, meta)
                known_tables = {table["name"] for table in meta["tables"]}
                for (name,) in direct_db.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'main' AND starts_with(table_name, 'v_') ORDER BY table_name"
                ).fetchall():
                    if name not in known_tables:
                        meta["tables"].append(profile(direct_db, name, "cached semantic view"))
                catalog = self._direct_catalog(direct_db, business_id, meta)
            finally:
                direct_db.close()
        else:
            manifest = (Path(self.cfg.manifest_dir) if self.cfg.manifest_dir else root / "manifests") / f"{business_id}.json"
            if self.cfg.require_prepared_database and not manifest.is_file():
                raise ValueError(f"Missing prepared manifest: {manifest}")
            manifest = manifest if manifest.exists() else None
            overrides = self.cfg.ingestion_overrides.get(business_id, split.ingestion_overrides.get(business_id))
            signature = IngestionPipeline.input_fingerprint(business_dir, rulebook=split.rulebook, manifest_path=manifest, overrides=overrides)
            catalog = None
            if database.exists() and not self.cfg.rebuild_database:
                import duckdb
                try:
                    catalog = CatalogService.read_catalog(database)
                except (duckdb.Error, ValueError):
                    pass  # Old/incomplete cache: rebuild from sources.
            if (catalog is None or catalog.get("pipeline_version") != PIPELINE_VERSION
                    or catalog.get("input_fingerprint") != signature):
                if self.cfg.require_prepared_database:
                    raise ValueError(f"Database missing or stale: {database}; run prepare_datasets() first")
                catalog = IngestionPipeline.ingest(business_dir, output, manifest,
                    rulebook=split.rulebook, overrides=overrides)
        if catalog["business_id"] != business_id:
            raise ValueError("Catalog không thuộc business đang hỏi")
        tables = []
        with SQLExecutor.connect(database, split.calendar) as connection:
            # Point 2: Check and apply cached canonical views & debt layer from DataMate
            views_dir = Path(getattr(self.cfg, "views_from", "configs/public_v9") or "configs/public_v9")
            if getattr(self.cfg, "views", "cached") != "none" and views_dir.is_dir():
                v_file = views_dir / f"_views_{business_id}.json"
                d_file = views_dir / f"_debt_{business_id}.json"
                try:
                    has_v_sales = bool(connection.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'v_sales'").fetchone())
                except Exception:
                    has_v_sales = False
                if not has_v_sales and v_file.is_file():
                    try:
                        import json
                        from datamate import views
                        v_sql = json.loads(v_file.read_text(encoding="utf-8"))
                        views.build(connection, v_sql)
                    except Exception as err:
                        print(f"[{business_id}] Bỏ qua tải v_sales từ {v_file.name}: {err}", flush=True)
                if getattr(self.cfg, "debt_views", True) and d_file.is_file():
                    try:
                        has_debt = bool(connection.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'v_debt_orders'").fetchone())
                    except Exception:
                        has_debt = False
                    if not has_debt:
                        try:
                            import json
                            from datamate import debts
                            d_sqls = json.loads(d_file.read_text(encoding="utf-8"))
                            if d_sqls:
                                debts.build(connection, d_sqls)
                        except Exception as err:
                            print(f"[{business_id}] Bỏ qua tải debt layer từ {d_file.name}: {err}", flush=True)

            for table in catalog["tables"]:
                name = TabularLayout.quote_identifier(table["physical_name"])
                _, rows = SQLExecutor.query(connection, f"SELECT * FROM {name} LIMIT 3")
                columns = []
                for column in table["columns"]:
                    profile = column.get("profile", {})
                    row_count = profile.get("row_count", table["row_count"])
                    null_count = profile.get("null_count", 0)
                    parse_audited = "parse_failed_count" not in profile or profile["parse_failed_count"] is not None
                    parse_failed_count = profile.get("parse_failed_count", 0) if parse_audited else None
                    source_null_count = max(0, null_count - parse_failed_count) if parse_audited else None
                    present_count = max(0, row_count - source_null_count) if parse_audited else None
                    item = {k: column.get(k) for k in (
                        "physical_name", "source_name", "duckdb_type", "description",
                        "semantic_name", "semantic_status", "unit_basis", "aggregation", "time_role", "profile", "conversion")}
                    item["completeness"] = {
                        "row_count": row_count,
                        "source_null_count": source_null_count,
                        "parse_failed_count": parse_failed_count,
                        "typed_value_count": max(0, row_count - null_count),
                        "parse_success_rate": (
                            round((present_count - parse_failed_count) / present_count, 4)
                            if parse_audited and present_count else None
                        ),
                        "status": (
                            "not_audited" if not parse_audited else
                            "parse_failures" if parse_failed_count else "no_parse_failures"
                        ),
                    }
                    col = TabularLayout.quote_identifier(column["physical_name"])
                    if (column.get("time_role") or re.search(r"ngay|date|thoi_gian", column["physical_name"])):
                        year = re.search(r"\b(?:19|20)\d{2}\b", " ".join(table["source_objects"]))
                        expression = (f"CAST({col} AS DATE)" if column["duckdb_type"] in ("DATE", "TIMESTAMP") else
                                      f"vn_date_year(CAST({col} AS VARCHAR), {year[0]})" if year
                                      else f"vn_date(CAST({col} AS VARCHAR))")
                        try:
                            _, span = SQLExecutor.query(connection, f"SELECT min({expression}), max({expression}) FROM {name}")
                            item["date_range"] = span[0]
                        except Exception as error:
                            print(f"[{business_id}] Bỏ qua date_range {table['physical_name']}.{column['physical_name']}: {error}"[:300], flush=True)
                    if re.search(r"trang_thai|status|ket_qua|tinh_trang", column["physical_name"]):
                        try:
                            _, counts = SQLExecutor.query(connection,
                                f"SELECT {col}, count(*) AS n FROM {name} GROUP BY 1 ORDER BY 2 DESC LIMIT 100")
                            item["values"] = counts
                            item["revenue_status_values"] = [
                                {"value": value, "count": count,
                                 "classification": SQLContextBuilder.classify_revenue_status(value)}
                                for value, count in counts
                            ]
                        except Exception as error:
                            print(f"[{business_id}] Bỏ qua giá trị trạng thái {table['physical_name']}.{column['physical_name']}: {error}"[:300], flush=True)
                    columns.append(item)
                tables.append({"name": table["physical_name"], "sources": table["source_files"],
                               "source_objects": table["source_objects"], "role": table.get("table_role"),
                               "grain": table.get("grain"), "row_count": table["row_count"],
                               "columns": columns, "sample_rows": rows,
                               "lineage": table.get("lineage"), "issue_counts": table.get("issue_counts", {})})
        readme_path = business_dir / "README.md"
        context = {"tables": tables, "relationships": catalog.get("relationships", []),
                   "facts": catalog.get("facts", []), "unresolved": catalog.get("unresolved", []),
                   "quality": catalog.get("quality", {}),
                   "organization": self.organization_context(catalog),
                   "semantic_views": catalog.get("semantic_views", []),
                   "ingestion_mode": self.cfg.ingestion_mode,
                   "ingestion_issues": self._data_issues(catalog.get("ingestion_issues", [])),
                   "rulebook": split.rulebook,
                   "readme": readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""}
        if split.calendar is not None:
            context["calendar"] = {"name": "calendar", "columns": list(split.calendar.columns),
                                   "source": "calendar.csv",
                                   "sha256": hashlib.sha256(split.calendar.to_json(
                                       orient="split", date_format="iso").encode()).hexdigest()}
        with database.open("rb") as stream:
            fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
        self.prepared[key] = database, context, fingerprint
        return self.prepared[key]

    @staticmethod
    def _direct_catalog(connection, business_id, meta):
        tables = []
        for table in meta["tables"]:
            columns = []
            for name, kind, distinct, values in table["columns"]:
                quoted_table = TabularLayout.quote_identifier(table["name"])
                quoted_column = TabularLayout.quote_identifier(name)
                null_count = connection.execute(
                    f"SELECT count(*) - count({quoted_column}) FROM {quoted_table}"
                ).fetchone()[0]
                columns.append({
                    "physical_name": name,
                    "source_name": name,
                    "duckdb_type": kind,
                    "description": None,
                    "semantic_name": None,
                    "semantic_status": None,
                    "unit_basis": None,
                    "aggregation": None,
                    "time_role": None,
                    "profile": {"row_count": table["rows"], "null_count": null_count, "parse_failed_count": None},
                    "values": values if distinct is not None else None,
                })
            tables.append({
                "physical_name": table["name"],
                "source_files": [table["source"]],
                "source_objects": [table["source"].split("#", 1)[-1]],
                "table_role": None,
                "grain": None,
                "row_count": table["rows"],
                "columns": columns,
                "lineage": None,
                "issue_counts": {},
            })
        return {
            "business_id": business_id,
            "tables": tables,
            "relationships": [],
            "facts": [],
            "unresolved": [],
            "quality": {"status": "not_audited"},
            "organization": {},
            "semantic_views": [],
            "ingestion_issues": meta.get("ingestion_issues", []),
        }

    def _build_direct_layers(self, connection, business_id, meta):
        if self.cfg.views == "none":
            return
        views_dir = Path(getattr(self.cfg, "views_from", "configs/public_v9") or "configs/public_v9")
        for issue in self.apply_saved_layers(connection, views_dir, business_id, bool(self.cfg.debt_views)):
            meta.setdefault("ingestion_issues", []).append(issue)
            print(f"[direct-ingest] {business_id}/{issue['file']}: {issue['error']}"[:400], flush=True)

    @staticmethod
    def _saved_layer_path(views_dir, filename):
        """File layer đã lưu nằm ở <views_dir>/logs/<filename> (cách pipeline.py ghi/đọc); chấp nhận cả <views_dir>/<filename>."""
        for candidate in (Path(views_dir) / "logs" / filename, Path(views_dir) / filename):
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def apply_saved_layers(connection, views_dir, business_id, debt_views=True):
        """Dựng v_sales/v_orders (+ debt layer) từ JSON đã lưu. Trả về danh sách lỗi (rỗng nếu ổn)."""
        import json
        from datamate import views, debts
        issues = []
        vp = SQLContextBuilder._saved_layer_path(views_dir, f"_views_{business_id}.json")
        if vp is None:
            issues.append({"file": f"_views_{business_id}.json", "error": f"Không tìm thấy trong {views_dir}(/logs)"})
        else:
            try:
                saved = json.loads(vp.read_text(encoding="utf-8"))
                sql = saved.get("sql") if isinstance(saved, dict) else saved
                sql = views.rewrap(sql)
                if sql:
                    views.build(connection, sql)
                else:
                    issues.append({"file": vp.name, "error": "JSON không có trường 'sql'"})
            except Exception as error:
                issues.append({"file": vp.name, "error": f"{type(error).__name__}: {error}"})
        if debt_views:
            dp = SQLContextBuilder._saved_layer_path(views_dir, f"_debt_{business_id}.json")
            if dp is not None:
                try:
                    saved = json.loads(dp.read_text(encoding="utf-8"))
                    sqls = saved.get("sqls") if isinstance(saved, dict) else saved
                    if sqls and (not isinstance(saved, dict) or saved.get("used", True)):
                        debts.build(connection, sqls)
                except Exception as error:
                    issues.append({"file": dp.name, "error": f"{type(error).__name__}: {error}"})
        return issues

    @staticmethod
    def _data_issues(issues):
        """README/tài liệu .md không phải nguồn dữ liệu: đừng để model coi đó là file dữ liệu bị lỗi."""
        kept = []
        for issue in issues or []:
            text = json.dumps(issue, ensure_ascii=False, default=str).lower()
            if re.search(r"readme|\.md\b|\.markdown\b", text):
                continue
            kept.append(issue)
        return kept

    @staticmethod
    def organization_context(catalog):
        """Send applied view mappings/grain, not installed SQL or rejected candidates."""
        organization = catalog.get("organization", {})
        result = {k: organization.get(k) for k in ("policy", "policy_version", "relationships")}
        for key in ("transformations", "models"):
            result[key] = [{k: v for k, v in op.items()
                            if k not in ("installed_sql", "definition_fingerprint", "queries")}
                           for op in organization.get(key, []) if op["status"] == "applied"]
        return result

    @staticmethod
    def classify_revenue_status(value):
        status = norm(str(value))
        if any(term in status for term in ("khong giao", "giao that bai", "giao khong thanh cong")):
            return "exclude_failed_delivery"
        if any(term in status for term in ("tra hang", "doi tra", "returned")):
            return "exclude_returned"
        if any(term in status.split() for term in ("huy", "cancel", "cancelled", "canceled", "void")):
            return "exclude_cancelled"
        if any(term in status for term in ("dang van chuyen", "dang giao", "in transit")):
            return "include_in_transit"
        if any(term in status for term in ("da giao", "hoan thanh", "delivered", "completed")):
            return "include_completed"
        return "unknown"

    @staticmethod
    def question_context(context, original, rewritten, as_of, calendar, db=None, bank=None, business_id=None):
        from datamate.hints import question_hints
        meta = {"tables": context.get("tables", [])}
        rulebook_txt = context.get("rulebook", "")
        dynamic_hints = question_hints(original, str(as_of), db=db, meta=meta, rulebook=rulebook_txt)
        time_hint = describe_period(rewritten, as_of, calendar)
        
        res = {
            **context,
            "original_question": original,
            "question": rewritten,
            "as_of_date": str(as_of),
            "time_hint": time_hint,
            "hints": dynamic_hints,
        }
        if bank is not None and business_id:
            try:
                exemplars = bank.retrieve(original, business_id, k=4)
                if exemplars:
                    res["exemplars"] = exemplars
            except Exception:
                pass
        return res