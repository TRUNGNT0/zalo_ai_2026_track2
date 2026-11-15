from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote
import sqlite3

from datamate.ingestion.sources.adapters.common import SourceTable, TabularLayout

FETCH_SIZE = 1000


class SqliteAdapter:
    """Read SQLite tables and declared foreign keys without modifying sources."""

    @staticmethod
    def read(path: Path, source_file: str) -> Iterator[SourceTable]:
        connection = sqlite3.connect(
            f"file:{quote(str(path.resolve()), safe='/')}?mode=ro",
            uri=True,
        )
        try:
            objects = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            for (table_name,) in objects:
                table_info = connection.execute(
                    f"PRAGMA table_info({TabularLayout.quote_identifier(table_name)})"
                ).fetchall()
                columns = tuple(str(column[1]) for column in table_info)
                primary_key = [
                    (int(column[5]), str(column[1]), index)
                    for index, column in enumerate(table_info)
                    if int(column[5]) > 0
                ]
                primary_key.sort()
                source_types = tuple(column[2] or None for column in table_info)
                foreign_keys = connection.execute(
                    f"PRAGMA foreign_key_list({TabularLayout.quote_identifier(table_name)})"
                ).fetchall()
                grouped_foreign_keys: dict[int, list[tuple[Any, ...]]] = {}
                for foreign_key in foreign_keys:
                    grouped_foreign_keys.setdefault(int(foreign_key[0]), []).append(
                        tuple(foreign_key)
                    )
                relationships = []
                for foreign_key_id, foreign_key_rows in sorted(
                    grouped_foreign_keys.items()
                ):
                    ordered_columns = sorted(
                        foreign_key_rows, key=lambda row: int(row[1])
                    )
                    target_table = str(ordered_columns[0][2])
                    target_columns = [row[4] for row in ordered_columns]
                    if any(column is None for column in target_columns):
                        target_info = connection.execute(
                            f"PRAGMA table_info({TabularLayout.quote_identifier(target_table)})"
                        ).fetchall()
                        primary_key = sorted(
                            (key_order, column_name)
                            for row in target_info
                            if (key_order := int(row[5])) > 0
                            for column_name in (str(row[1]),)
                        )
                        if len(primary_key) != len(ordered_columns):
                            continue
                        target_columns = [column for _, column in primary_key]
                    relationships.append(
                        {
                            "source_file": source_file,
                            "from_table": table_name,
                            "from_columns": [str(row[3]) for row in ordered_columns],
                            "to_table": target_table,
                            "to_columns": [str(column) for column in target_columns],
                            "constraint_id": foreign_key_id,
                        }
                    )
                cursor = connection.execute(
                    f"SELECT * FROM {TabularLayout.quote_identifier(table_name)}"
                )
                positions = []
                issues = [] if primary_key else [{"code": "sqlite_no_primary_key", "review_required": True}]

                yield SourceTable(
                    source_file=source_file,
                    source_object=table_name,
                    source_headers=columns,
                    rows=SqliteAdapter._rows_for_table(
                        cursor, positions, [(name, index) for _, name, index in primary_key]
                    ),
                    header_row_number=None,
                    header_status="sqlite_schema",
                    source_types=source_types,
                    relationships=tuple(relationships),
                    positions=positions,
                    issues=issues,
                )
        finally:
            connection.close()

    @staticmethod
    def _rows_for_table(
        row_cursor: sqlite3.Cursor, positions, primary_key: list[tuple[str, int]]
    ) -> Iterator[tuple[object, ...]]:
        record = 0
        while batch := row_cursor.fetchmany(FETCH_SIZE):
            for row in batch:
                record += 1
                position = {"record": record}
                if primary_key:
                    position["primary_key"] = {
                        column: row[index] for column, index in primary_key
                    }
                positions.append(position)
                yield row
