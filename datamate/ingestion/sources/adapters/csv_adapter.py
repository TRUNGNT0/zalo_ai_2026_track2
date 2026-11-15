from itertools import islice
from pathlib import Path
from typing import Iterable
import csv

from datamate.ingestion.sources.adapters.common import SourceTable, TabularLayout
from datamate.ingestion.processing.normalization import IdentifierNormalizer

HEADER_SCAN_ROWS = 50


class CsvAdapter:
    """Read CSV sources while preserving header and row provenance."""

    @staticmethod
    def read(
        path: Path,
        source_file: str,
        expected_headers: Iterable[str] = (),
    ) -> SourceTable:
        file_handle = path.open("r", encoding="utf-8-sig", newline="")
        try:
            sample = file_handle.read(8192)
            file_handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error:
                dialect = csv.excel
            lines = sample.splitlines()
            scores = {}
            for delimiter in ",;\t|":
                widths = [
                    len(next(csv.reader([line], delimiter=delimiter)))
                    for line in lines if line.strip()
                ]
                multi_column = [width for width in widths if width > 1]
                if multi_column:
                    common_width = max(set(multi_column), key=multi_column.count)
                    scores[delimiter] = (
                        sum(width == common_width for width in multi_column),
                        common_width,
                    )
            if scores:
                best = max(scores, key=scores.get)
                if scores[best][0] > scores.get(dialect.delimiter, (0, 0))[0]:
                    dialect.delimiter = best
            reader = csv.reader(file_handle, dialect)

            row_positions = {}
            rows = CsvAdapter._numbered_rows(reader, file_handle, row_positions)
            buffered = list(islice(rows, HEADER_SCAN_ROWS))
            header_number, header_values, header_status = TabularLayout.choose_header(
                buffered, expected_headers
            )
            headers = IdentifierNormalizer.headers(header_values)
            remaining = ((row_number, values) for row_number, values in rows)
            positions = []
            return SourceTable(
                source_file=source_file,
                source_object=path.stem,
                source_headers=headers,
                rows=TabularLayout.prepare_rows(
                    buffered, remaining, header_number, len(headers),
                    positions=positions, position_details=row_positions),
                header_row_number=header_number,
                header_status=header_status,
                positions=positions,
            )
        except Exception:
            file_handle.close()
            raise

    @staticmethod
    def _numbered_rows(reader, file_handle, positions):
        try:
            previous_line = 0
            row_number = 0
            for values in reader:
                line_end = reader.line_num
                if TabularLayout.nonempty_row(values):
                    row_number += 1
                    positions[row_number] = {
                        "record": row_number,
                        "line_start": previous_line + 1,
                        "line_end": line_end,
                    }
                    yield row_number, tuple(None if value == "" else value for value in values)
                previous_line = line_end
        finally:
            file_handle.close()
