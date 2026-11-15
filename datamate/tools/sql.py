"""Bounded, read-only DuckDB execution for model-generated SELECT statements."""
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
import re
from threading import Timer

import duckdb

from datamate.text import norm
from datamate.tools.results import coerce_result, final_fmt, is_empty_result
from datamate.tools.viet_time import parse_cell_date


class SQLExecutor:
    @staticmethod
    def query_keyword(sql: str) -> str:
        """Read the original leading keyword, skipping SQL comments (including nested blocks).

        DuckDB 1.x can rewrite PRAGMA into a SELECT during extract_statements;
        the parsed statement type alone therefore cannot enforce our input contract.
        This lexical check supplements, rather than replaces, DuckDB's parser.
        """
        if not isinstance(sql, str):
            return ""
        position = 0
        while position < len(sql):
            if sql[position].isspace():
                position += 1
            elif sql.startswith("--", position):
                end = re.search(r"[\r\n]", sql[position + 2:])
                if end is None:
                    return ""
                position += 2 + end.end()
            elif sql.startswith("/*", position):
                depth = 1
                position += 2
                while position < len(sql) and depth:
                    if sql.startswith("/*", position):
                        depth += 1
                        position += 2
                    elif sql.startswith("*/", position):
                        depth -= 1
                        position += 2
                    else:
                        position += 1
                if depth:
                    return ""
            else:
                keyword = re.match(r"[A-Za-z_][A-Za-z_0-9]*", sql[position:])
                return keyword[0].upper() if keyword else ""
        return ""

    @staticmethod
    def date_value(value: str):
        return parse_cell_date(value)

    @staticmethod
    def date_with_year(value: str, year: int):
        return parse_cell_date(value, sheet_year=year)

    @staticmethod
    def connect(path: Path, calendar=None):
        connection = duckdb.connect(str(path), read_only=True, config={
            "enable_external_access": "false", "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false", "memory_limit": "512MB", "threads": "2",
            "max_temp_directory_size": "0B",
        })
        try:
            connection.create_function("vn_date", SQLExecutor.date_value, ["VARCHAR"], "DATE", null_handling="special")
            connection.create_function("vn_date_year", SQLExecutor.date_with_year,
                                       ["VARCHAR", "INTEGER"], "DATE", null_handling="special")
            connection.create_function("vn_norm", norm, ["VARCHAR"], "VARCHAR")
            if calendar is not None:
                exists = connection.execute(
                    "SELECT 1 FROM information_schema.tables WHERE table_name = 'calendar' LIMIT 1"
                ).fetchone()
                if not exists:
                    connection.register("calendar", calendar)
            connection.execute("SET lock_configuration = true")
            return connection
        except Exception:
            connection.close()
            raise

    @staticmethod
    def query(connection, sql: str, *, timeout=30, max_rows=1000):
        if SQLExecutor.query_keyword(sql) not in {"SELECT", "WITH"}:
            raise ValueError("Chỉ chấp nhận đúng một câu SELECT/WITH, không cho phép thay đổi dữ liệu")
        statements = connection.extract_statements(sql)
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            raise ValueError("Chỉ chấp nhận đúng một câu SELECT/WITH, không cho phép thay đổi dữ liệu")
        timer = Timer(timeout, connection.interrupt)
        timer.daemon = True
        timer.start()
        try:
            cursor = connection.execute(sql)
            columns = [item[0] for item in cursor.description]
            rows = cursor.fetchmany(max_rows + 1)
            if len(rows) > max_rows:
                raise ValueError(f"Kết quả vượt quá {max_rows} dòng; cần tổng hợp hoặc giới hạn theo câu hỏi")
            return columns, [[SQLExecutor.json_value(v) for v in row] for row in rows]
        finally:
            timer.cancel()
            timer.join()

    @staticmethod
    def json_value(value):
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, (list, tuple)):
            return [SQLExecutor.json_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): SQLExecutor.json_value(v) for k, v in value.items()}
        return value

    @staticmethod
    def answer(result_type, columns, rows):
        if result_type == "table":
            result = {"columns": columns, "rows": rows}
        elif result_type == "list":
            if len(columns) != 1:
                raise ValueError("Đáp án list cần SQL trả một cột theo đúng thứ tự")
            result = [row[0] for row in rows]
        else:
            if len(columns) != 1 or len(rows) != 1:
                raise ValueError("Đáp án scalar cần đúng một dòng, một cột")
            result = rows[0][0]
        if is_empty_result(result_type, result):
            raise ValueError("Kết quả rỗng/NULL; kiểm tra phạm vi dữ liệu trước khi kết luận")
        return final_fmt(result_type, coerce_result(result_type, result))
