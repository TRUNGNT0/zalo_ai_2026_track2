from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator

from openpyxl import load_workbook
try:
    import xlrd
except ImportError:
    xlrd = None

from datamate.ingestion.sources.adapters.common import SourceTable, TabularLayout
from datamate.ingestion.processing.normalization import IdentifierNormalizer

HEADER_SCAN_ROWS = 50


class ExcelAdapter:
    """Stream workbook sheets from XLS and XLSX sources."""

    @staticmethod
    def read(
        path: Path,
        source_file: str,
        expected_headers: Iterable[str] = (),
    ) -> Iterator[SourceTable]:
        if path.suffix.casefold() == ".xls":
            yield from ExcelAdapter._read_xls(path, source_file, expected_headers)
            return

        workbook = load_workbook(path, read_only=True, data_only=False)
        value_workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            for worksheet in workbook.worksheets:
                value_worksheet = value_workbook[worksheet.title]
                cell_metadata = {}
                issues = []
                numbered_rows = (
                    (row_number, tuple(values))
                    for row_number, values in ExcelAdapter._xlsx_rows(
                        worksheet, value_worksheet, cell_metadata, issues
                    )
                    if TabularLayout.nonempty_row(values)
                )
                buffered = list(islice(numbered_rows, HEADER_SCAN_ROWS))
                if not buffered:
                    yield SourceTable(
                        source_file=source_file,
                        source_object=worksheet.title,
                        source_headers=(),
                        rows=iter(()),
                        header_row_number=None,
                        header_status="empty_sheet",
                        positions=[],
                        issues=[{"code": "empty_sheet"}],
                    )
                    continue
                header_number, header_values, header_status = TabularLayout.choose_header(
                    buffered, expected_headers
                )
                headers = IdentifierNormalizer.headers(header_values)
                positions = []
                remaining = (
                    (row_number, values) for row_number, values in numbered_rows
                )
                yield SourceTable(
                    source_file=source_file,
                    source_object=worksheet.title,
                    source_headers=headers,
                    rows=TabularLayout.prepare_rows(
                        buffered, remaining, header_number, len(headers),
                        positions=positions, position_key="row",
                    ),
                    header_row_number=header_number,
                    header_status=header_status,
                    excel_epoch="1904" if workbook.epoch.year == 1904 else "1900",
                    positions=positions,
                    issues=issues,
                    cell_metadata=cell_metadata,
                )
        finally:
            value_workbook.close()
            workbook.close()

    @staticmethod
    def _read_xls(
        path: Path,
        source_file: str,
        expected_headers: Iterable[str],
    ) -> Iterator[SourceTable]:
        if xlrd is None:
            raise ImportError("Gói 'xlrd' cần thiết để đọc file Excel định dạng cũ .xls. Vui lòng cài đặt: pip install xlrd")
        workbook = xlrd.open_workbook(path.as_posix(), on_demand=True)
        try:
            for worksheet in workbook.sheets():

                numbered_rows = ExcelAdapter._rows_for_sheet(worksheet, workbook.datemode)
                buffered = list(islice(numbered_rows, HEADER_SCAN_ROWS))
                if not buffered:
                    continue
                header_number, header_values, header_status = TabularLayout.choose_header(
                    buffered, expected_headers
                )
                headers = IdentifierNormalizer.headers(header_values)
                positions = []
                remaining = (
                    (row_number, values) for row_number, values in numbered_rows
                )
                yield SourceTable(
                    source_file=source_file,
                    source_object=worksheet.name,
                    source_headers=headers,
                    rows=TabularLayout.prepare_rows(
                        buffered, remaining, header_number, len(headers),
                        positions=positions, position_key="row",
                    ),
                    header_row_number=header_number,
                    header_status=header_status,
                    excel_epoch="1904" if workbook.datemode == 1 else "1900",
                    positions=positions,
                )
        finally:
            workbook.release_resources()

    @staticmethod
    def _xls_row_values(worksheet: Any, row_index: int, datemode: int) -> tuple[object, ...]:
        if datemode not in (0, 1):
            raise ValueError(f"Unsupported Excel date mode: {datemode}.")
        values = []
        for cell in worksheet.row(row_index):
            if cell.ctype == xlrd.XL_CELL_DATE:
                if not isinstance(cell.value, (int, float)):
                    raise ValueError(
                        f"Excel date cell at row {row_index + 1} is not numeric."
                    )
                values.append(
                    xlrd.xldate.xldate_as_datetime(float(cell.value), datemode)
                )
            elif cell.ctype in {xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK}:
                values.append(None)
            else:
                values.append(cell.value)
        return tuple(values)

    @staticmethod
    def _rows_for_sheet(worksheet, datemode) -> Iterator[tuple[int, tuple[object, ...]]]:
        for row_number in range(1, worksheet.nrows + 1):
            values = ExcelAdapter._xls_row_values(
                worksheet, row_number - 1, datemode
            )
            if TabularLayout.nonempty_row(values):
                yield row_number, values

    @staticmethod
    def _xlsx_rows(worksheet, value_worksheet, cell_metadata, issues):
        for row_number, (raw_row, cached_row) in enumerate(
            zip(worksheet.iter_rows(), value_worksheet.iter_rows()), start=1
        ):
            values = []
            for column_number, (cell, cached) in enumerate(zip(raw_row, cached_row), start=1):
                value = cell.value
                if cell.data_type == "f":
                    cell_metadata[f"{row_number}:{column_number}"] = {
                        "formula": value, "cached": cached.value, "number_format": cell.number_format
                    }
                    value = cached.value
                    if value is None:
                        issues.append({
                            "code": "formula_cache_missing", "row": row_number,
                            "column": column_number, "formula": cell.value
                        })
                elif cell.data_type == "e":
                    cell_metadata[f"{row_number}:{column_number}"] = {"error": value}
                    issues.append({
                        "code": "excel_error", "row": row_number,
                        "column": column_number, "value": value
                    })
                    value = None
                elif value is not None and cell.number_format not in ("General", "@"):
                    cell_metadata[f"{row_number}:{column_number}"] = {
                        "number_format": cell.number_format
                    }
                values.append(value)
            yield row_number, tuple(values)
