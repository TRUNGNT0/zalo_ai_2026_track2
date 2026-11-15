from pathlib import Path
from typing import Iterator

import duckdb

from datamate.ingestion.sources.adapters.common import SourceTable

FETCH_SIZE = 1000


class ParquetAdapter:
    """Stream Parquet tables through DuckDB."""

    @staticmethod
    def read(path: Path, source_file: str) -> Iterator[SourceTable]:
        connection = duckdb.connect(":memory:")
        try:
            cursor = connection.execute(
                "SELECT * FROM read_parquet(?)",
                [str(path)],
            )
            headers = tuple(description[0] for description in cursor.description)
            positions = []

            yield SourceTable(
                source_file=source_file,
                source_object=path.stem,
                source_headers=headers,
                rows=ParquetAdapter._rows(cursor, positions),
                header_row_number=None,
                header_status="parquet_schema",
                positions=positions,
            )
        finally:
            connection.close()

    @staticmethod
    def _rows(cursor, positions) -> Iterator[tuple[object, ...]]:
        record = 0
        while batch := cursor.fetchmany(FETCH_SIZE):
            for row in batch:
                record += 1
                positions.append({"record": record})
                yield row
