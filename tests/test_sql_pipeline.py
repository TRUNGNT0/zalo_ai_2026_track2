import json
from pathlib import Path

import duckdb
import pytest

from datamate.config import Config
from datamate.tools.sql import SQLExecutor
from evaluation.run_eval import run_train


class SQLBackend:
    def __init__(self, plans):
        self.plans = iter(plans)
        self.prompts = []
        self.systems = []

    def tag(self):
        return "fake-sql"

    def generate_json(self, system, prompt, schema):
        self.systems.append(system)
        self.prompts.append(prompt)
        return json.dumps(next(self.plans))


class TestSQLExecutor:
    @pytest.fixture
    def database(self, tmp_path):
        path = tmp_path / "business.duckdb"
        with duckdb.connect(str(path)) as connection:
            connection.execute("CREATE TABLE sales AS SELECT 1 AS amount UNION ALL SELECT 2")
        return path

    def test_select_and_date_helpers(self, database):
        with SQLExecutor.connect(database) as c:
            cols, rows = SQLExecutor.query(c, "SELECT SUM(amount) FROM sales")
            assert SQLExecutor.answer("number", cols, rows) == 3
            assert SQLExecutor.query(c, "SELECT vn_date('04/05/2026'), vn_date_year('4/5', 2025)")[1] == [
                ["2026-05-04", "2025-05-04"]]
            assert SQLExecutor.query(c, "WITH x AS (SELECT * FROM sales) SELECT count(*) FROM x")[1] == [[2]]

    @pytest.mark.parametrize("sql", [
        "DELETE FROM sales", "DROP TABLE sales", "SELECT 1; SELECT 2",
        "COPY sales TO '/tmp/sql-pipeline-must-not-write.csv'", "PRAGMA version",
        "SET enable_external_access=true", "ATTACH '/tmp/no.db' AS other",
        "-- model comment\nPrAgMa version", "/* SELECT */ PRAGMA version",
        "/* outer /* nested */ comment */ PRAGMA version",
        "WITH x AS (SELECT 1) DELETE FROM sales",
    ])
    def test_rejects_mutations_and_multiple_statements(self, database, sql):
        with SQLExecutor.connect(database) as c:
            with pytest.raises(ValueError, match="SELECT"):
                SQLExecutor.query(c, sql)
            assert SQLExecutor.query(c, "SELECT count(*) FROM sales")[1] == [[2]]

    @pytest.mark.parametrize("sql", [
        "-- heading\nSELECT SUM(amount) FROM sales",
        "/* outer /* nested */ comment */ select SUM(amount) FROM sales",
        "/* heading */ WITH x AS (SELECT * FROM sales) SELECT SUM(amount) FROM x",
    ])
    def test_select_with_leading_comments(self, database, sql):
        with SQLExecutor.connect(database) as c:
            assert SQLExecutor.query(c, sql)[1] == [[3]]

    def test_pragma_rejected_before_parser_rewriting(self):
        class RewritingConnection:
            def extract_statements(self, sql):
                pytest.fail("PRAGMA must be rejected before DuckDB can rewrite it to SELECT")

        with pytest.raises(ValueError, match="SELECT"):
            SQLExecutor.query(RewritingConnection(), "PRAGMA version")

    def test_keywords_in_string_literals_are_allowed(self, database):
        with SQLExecutor.connect(database) as c:
            assert SQLExecutor.query(c, "SELECT 'PRAGMA version; -- /* text */'")[1] == [["PRAGMA version; -- /* text */"]]

    def test_external_reads_are_disabled(self, database, tmp_path):
        outside = tmp_path / "secret.csv"
        outside.write_text("secret\nnot_allowed\n")
        with SQLExecutor.connect(database) as c:
            with pytest.raises(duckdb.Error):
                SQLExecutor.query(c, f"SELECT * FROM read_csv_auto('{outside}')")

    def test_query_row_limit_and_timeout(self, database):
        with SQLExecutor.connect(database) as c:
            with pytest.raises(ValueError, match="vượt quá"):
                SQLExecutor.query(c, "SELECT * FROM range(11)", max_rows=10)
            with pytest.raises(duckdb.InterruptException):
                SQLExecutor.query(c, "SELECT sum(i*j) FROM range(1000000000) a(i), range(1000000000) b(j)", timeout=0.05)
            assert SQLExecutor.query(c, "SELECT 1")[1] == [[1]]

    def test_rejects_scalar_with_multiple_rows_and_null(self):
        with pytest.raises(ValueError, match="một dòng"):
            SQLExecutor.answer("number", ["v"], [[1], [2]])
        with pytest.raises(ValueError, match="NULL"):
            SQLExecutor.answer("number", ["v"], [[None]])
        assert SQLExecutor.answer("list", ["v"], [[1], [2]]) == [1, 2]


class TestSQLPipeline:
    @staticmethod
    def config(split_dirs, **kwargs):
        train, test, out = split_dirs
        return Config(train_dir=train, test_dir=test, out_dir=out, backend="gemini",
                      query_engine="sql", database_dir=f"{out}/databases", limit_train=1,
                      n_samples=1, agree_min=1, rewrite_questions=False, **kwargs)

    @staticmethod
    def plan(sql="SELECT SUM(thanh_tien) FROM sales WHERE vn_norm(trang_thai) != 'huy'"):
        return {"type": "number", "sql": sql, "reason": "Loại đơn hủy theo rulebook"}

    def test_adapter_to_sql_to_gold_and_cache(self, split_dirs):
        cfg = self.config(split_dirs)
        backend = SQLBackend([self.plan()])
        _, _, ev, subs, logs = run_train(cfg, backend)
        assert subs == [{"id": "q0", "type": "number", "answer": 130000}]
        assert ev.status.tolist() == ["correct"]
        assert logs[0]["sql"].startswith("SELECT")
        assert Path(logs[0]["database"]).is_file()
        assert Path(cfg.out_dir, "evaluation.jsonl").is_file()
        prompt = json.loads(backend.prompts[0])
        assert prompt["question"] == "Tổng doanh thu?"
        assert prompt["rulebook"] == "Loại đơn hủy. Làm tròn 2 chữ số."
        sales = next(table for table in prompt["tables"] if table["name"] == "sales")
        assert sales["row_count"] == 4
        assert {column["physical_name"] for column in sales["columns"]} >= {"thanh_tien", "trang_thai"}
        statuses = next(column for column in sales["columns"] if column["physical_name"] == "trang_thai")
        assert statuses["revenue_status_values"] == [
            {"value": "hoan thanh", "count": 3, "classification": "include_completed"},
            {"value": "huy", "count": 1, "classification": "exclude_cancelled"},
        ]
        assert "exclude_failed_delivery" in backend.systems[0]
        assert "SELECT DISTINCT *" in backend.systems[0]
        assert "parse_failed_count > 0" in backend.systems[0]
        _, _, again, _, cached_logs = run_train(cfg, backend)
        assert again.status.tolist() == ["correct"]
        assert len(backend.prompts) == 1
        assert cached_logs[0]["cache_hit"]

    def test_direct_ingestion_runs_without_manifest_and_reports_bad_files(self, split_dirs):
        from datamate.agent.sql_context import SQLContextBuilder
        from datamate.ingestion.datasets.loader import DatasetLoader

        train, _, _ = split_dirs
        broken = Path(train, "data", "b1", "broken.xlsx")
        broken.write_text("not an Excel workbook", encoding="utf-8")
        cfg = self.config(split_dirs, ingestion_mode="direct")
        split = DatasetLoader.load_split(train)
        database, context, _ = SQLContextBuilder(cfg).prepare(split, "b1")

        assert database.is_file()
        assert context["ingestion_mode"] == "direct"
        assert context["ingestion_issues"][0]["file"] == "broken.xlsx"
        assert not Path(train, "manifests").exists()
        sales = next(table for table in context["tables"] if table["name"] == "sales")
        amount = next(column for column in sales["columns"] if column["physical_name"] == "thanh_tien")
        assert amount["completeness"]["status"] == "not_audited"
        assert amount["completeness"]["parse_failed_count"] is None
        with SQLExecutor.connect(database, split.calendar) as connection:
            assert SQLExecutor.query(connection, "SELECT SUM(thanh_tien) FROM sales")[1] == [[190000]]

    def test_revenue_status_classification_covers_failed_and_in_transit(self):
        from datamate.agent.sql_context import SQLContextBuilder

        assert SQLContextBuilder.classify_revenue_status("Không giao được") == "exclude_failed_delivery"
        assert SQLContextBuilder.classify_revenue_status("Đang vận chuyển") == "include_in_transit"
        assert SQLContextBuilder.classify_revenue_status("Đổi trả") == "exclude_returned"
        assert SQLContextBuilder.classify_revenue_status("Trạng thái mới") == "unknown"

    def test_gold_is_not_used_to_generate_or_repair_sql(self, split_dirs):
        cfg = self.config(split_dirs)
        answer_path = Path(cfg.train_dir, "answers.jsonl")
        rows = [json.loads(line) for line in answer_path.read_text().splitlines()]
        rows[0]["answer"] = 987654321
        answer_path.write_text("\n".join(json.dumps(row) for row in rows))
        backend = SQLBackend([self.plan("SELECT SUM(missing_column) FROM sales"), self.plan()])
        _, _, ev, subs, logs = run_train(cfg, backend)
        assert ev.status.tolist() == ["wrong"]
        assert subs[0]["answer"] == 130000
        assert len(backend.prompts) == 2 and logs[0]["errors"]
        assert all("987654321" not in p for p in backend.prompts)

    def test_prompt_reports_parse_completeness(self, split_dirs):
        cfg = self.config(split_dirs, ingestion_overrides={"b1": {"tables": {"sales": {
            "columns": {"thanh_tien": {"type": "number"}}
        }}}})
        source = Path(cfg.train_dir, "data", "b1", "sales.csv")
        source.write_text(
            "ngay,ten_hang,so_luong,thanh_tien,trang_thai\n"
            "2026-08-24,Ca phe,3,90000,hoan thanh\n"
            "2026-08-25,Tra,1,invalid,hoan thanh\n"
            "2026-08-26,Ca phe,2,60000,huy\n"
            "2026-08-27,Tra,1,20000,hoan thanh\n",
            encoding="utf-8",
        )
        backend = SQLBackend([{"type": "unanswerable", "sql": "", "reason": "Một giá trị thành tiền lỗi parse"}])
        _, _, _, _, logs = run_train(cfg, backend)
        prompt = json.loads(backend.prompts[0])
        sales = next(table for table in prompt["tables"] if table["name"] == "sales")
        amount = next(column for column in sales["columns"] if column["physical_name"] == "thanh_tien")
        assert amount["completeness"] == {
            "row_count": 4,
            "source_null_count": 0,
            "parse_failed_count": 1,
            "typed_value_count": 3,
            "parse_success_rate": 0.75,
            "status": "parse_failures",
        }
        assert logs[0]["type"] == "unanswerable"

    def test_no_consensus_abstains(self, split_dirs):
        cfg = self.config(split_dirs)
        cfg.n_samples, cfg.agree_min = 2, 2
        backend = SQLBackend([self.plan("SELECT 1"), self.plan("SELECT 2")])
        _, _, ev, subs, _ = run_train(cfg, backend)
        assert subs == [{"id": "q0", "type": "abstain"}]
        assert ev.status.tolist() == ["abstain"]

    def test_calendar_changes_invalidate_cache(self, split_dirs):
        cfg = self.config(split_dirs)
        path = Path(cfg.train_dir, "calendar.csv")
        header = "date,lunar_day,lunar_month,lunar_year,lunar_leap,holiday\n"
        path.write_text(header + "2026-08-24,12,7,2026,False,\n")
        backend = SQLBackend([self.plan(), self.plan()])
        run_train(cfg, backend)
        path.write_text(header + "2026-08-24,13,7,2026,False,\n")
        _, _, _, _, logs = run_train(cfg, backend)
        assert len(backend.prompts) == 2
        assert not logs[0]["cache_hit"]

    def test_unanswerable_does_not_execute_sql(self, split_dirs):
        cfg = self.config(split_dirs)
        backend = SQLBackend([{"type": "unanswerable", "sql": "", "reason": "Kỳ ngoài dữ liệu"}])
        _, _, _, subs, _ = run_train(cfg, backend)
        assert subs[0]["type"] == "unanswerable"
