from dataclasses import dataclass
from itertools import chain
from typing import Any, Iterable, Iterator, Sequence


# Cấu trúc dữ liệu để biểu diễn một bảng nguồn, bao gồm thông tin về file nguồn, tên bảng, các cột, các hàng dữ liệu và các quan hệ giữa các cột cho các loại file nguồn khác nhau (CSV, Excel, Parquet, v.v.). Bảng nguồn này được sử dụng để tạo BusinessManifest cho từng doanh nghiệp.
@dataclass(slots=True)
class SourceTable:
    source_file: str # đường dẫn đến file nguồn, ví dụ: "data/orders.csv"
    source_object: str # tên bảng hoặc đối tượng dữ liệu trong file nguồn, ví dụ: "orders"
    source_headers: tuple[str | None, ...] # danh sách tên cột từ file nguồn, ví dụ: ("order_id", "customer_id", "order_date")
    rows: Iterator[tuple[Any, ...]] # các hàng dữ liệu từ file nguồn, mỗi hàng là một tuple chứa giá trị của các cột
    header_row_number: int | None # số dòng của hàng header trong file nguồn, nếu có; None nếu không xác định được
    header_status: str # trạng thái của hàng header, ví dụ: "manifest_header_match", "heuristic_header", "first_nonempty_row"
    source_types: tuple[str | None, ...] = () # danh sách kiểu dữ liệu của các cột từ file nguồn, ví dụ: ("string", "string", "date")
    relationships: tuple[dict[str, Any], ...] = () # danh sách các quan hệ giữa các cột trong bảng nguồn, mỗi quan hệ là một dict chứa thông tin về các cột liên quan và loại quan hệ
    positions: list[dict[str, Any]] | None = None
    issues: list[dict[str, Any]] | None = None
    cell_metadata: dict[str, dict[str, Any]] | None = None
    excel_epoch: str | None = None


class TabularLayout:
    """Select headers and stream rows using shared tabular layout rules."""

    @staticmethod
    def quote_identifier(value: str) -> str:
        return '"' + value.replace('"', '""') + '"'


    # Hàm kiểm tra xem một hàng dữ liệu có ít nhất một giá trị không rỗng hay không. Một giá trị được coi là không rỗng nếu nó không phải là None và sau khi chuyển sang chuỗi và loại bỏ khoảng trắng, nó vẫn còn ký tự.
    @staticmethod
    def nonempty_row(values: Sequence[Any]) -> bool:
        return any(value is not None and str(value).strip() for value in values)    
    
    # Hàm chuẩn bị các hàng dữ liệu từ một danh sách các hàng đã được đệm và một iterable các hàng còn lại, bỏ qua các hàng trước hàng header và các hàng rỗng, đồng thời đảm bảo rằng số lượng cột của mỗi hàng không vượt quá số cột của header. Nếu một hàng có nhiều cột hơn header nhưng có giá trị không rỗng ở các cột vượt quá, nó sẽ ném ra lỗi ValueError.
    @staticmethod
    def prepare_rows(
        buffered_rows: Sequence[tuple[int, tuple[Any, ...]]],
        remaining_rows: Iterable[tuple[int, tuple[Any, ...]]],
        header_index: int,
        width: int,
        positions: list[dict[str, Any]] | None = None,
        position_key: str = "record",
        position_details: dict[int, dict[str, Any]] | None = None,
    ) -> Iterator[tuple[Any, ...]]:
        for row_number, values in chain(buffered_rows, remaining_rows):
            if row_number <= header_index or not TabularLayout.nonempty_row(values):
                continue
            if len(values) > width and any(value is not None for value in values[width:]):
                raise ValueError(
                    f"Row {row_number} has {len(values)} cells but the header has "
                    f"only {width} columns."
                )
            if positions is not None:
                positions.append((position_details or {}).get(row_number, {position_key: row_number}))
            yield tuple(values[index] if index < len(values) else None for index in range(width))

    @staticmethod
    def choose_header(
        candidates: Sequence[tuple[int, tuple[Any, ...]]],
        expected_headers: Iterable[str],
    ) -> tuple[int, tuple[Any, ...], str]:
        if not candidates:
            raise ValueError("Source table contains no non-empty rows.")

        expected = {value.strip().casefold() for value in expected_headers if value.strip()}
        if expected:
            scored = [
                (
                    sum(
                        1
                        for value in values
                        if value is not None and str(value).strip().casefold() in expected
                    ),
                    sum(
                        1
                        for value in values
                        if value is not None and str(value).strip()
                    ),
                    row_number,
                    values,
                )
                for row_number, values in candidates
            ]
            best = max(scored, key=lambda item: (item[0], item[1], -item[2]))
            if best[0] > 0:
                return best[2], best[3], "manifest_header_match"


        best_candidate = max(candidates, key=TabularLayout._header_score)
        if len([value for value in best_candidate[1] if isinstance(value, str) and value.strip()]) >= 2:
            status = "heuristic_header"
        else:
            status = "first_nonempty_row"
        return best_candidate[0], best_candidate[1], status

    @staticmethod
    def _header_score(candidate: tuple[int, tuple[Any, ...]]) -> tuple[int, float, int]:
        row_number, values = candidate
        strings = [
            str(value).strip()
            for value in values
            if isinstance(value, str) and value.strip()
        ]
        normalized = [value.casefold() for value in strings]
        unique_ratio = len(set(normalized)) / len(normalized) if normalized else 0.0
        return len(strings), unique_ratio, -row_number
