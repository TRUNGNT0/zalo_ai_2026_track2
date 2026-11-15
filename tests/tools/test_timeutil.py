"""Test quy đổi thời gian tương đối."""
import pandas as pd

from datamate.text import norm
from datamate.tools.rules import prep_fact
from datamate.tools.timeutil import describe_period, resolve_period, resolve_time
from datamate.tools.viet_time import parse_cell_date, parse_time_of_day, in_shift


def test_resolve_period_week_and_month():
    qn = norm("tuần trước")
    assert resolve_period(qn, "2026-08-27") == (pd.Timestamp("2026-08-17"), pd.Timestamp("2026-08-23"))
    assert resolve_period(norm("tháng trước"), "2026-08-27") == (pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-31"))
    assert resolve_period(norm("tháng này"), "2026-08-27") == (pd.Timestamp("2026-08-01"), pd.Timestamp("2026-08-27"))
    assert resolve_period(norm("so với cùng kỳ"), "2026-08-27") is None
    assert resolve_period("quý 5 và tháng này", "2026-08-27") is None


def test_describe_period_gives_hint_or_none():
    assert describe_period(norm("tuần trước bán gì"), "2026-08-27") == \
        '"tuần trước" = 2026-08-17 đến 2026-08-23 (tuần tính Thứ Hai đến Chủ Nhật)'
    assert describe_period(norm("doanh thu tổng"), "2026-08-27") is None
    assert "comparison_period_unresolved" in describe_period(
        norm("tháng này so với cùng kỳ"), "2026-08-27"
    )


def test_comparison_periods_are_available_to_context():
    spec = resolve_time("Doanh thu tháng này so với cùng kỳ năm ngoái", "2026-08-27")
    assert [(p.start, p.end, p.role) for p in spec.periods] == [
        (pd.Timestamp("2026-08-01").date(), pd.Timestamp("2026-08-27").date(), "main"),
        (pd.Timestamp("2025-08-01").date(), pd.Timestamp("2025-08-27").date(), "baseline"),
    ]
    hint = describe_period("Doanh thu tháng này so với cùng kỳ năm ngoái", "2026-08-27")
    assert "Kỳ chính" in hint and "Kỳ so sánh" in hint
    assert resolve_period("Doanh thu tháng này so với cùng kỳ năm ngoái", "2026-08-27") is None


def test_shift_comparison_and_price_anchor_are_described():
    shift_hint = describe_period(
        "Ca chiều tuần trước so với ca sáng thì doanh thu chênh bao nhiêu %",
        "2026-03-04",
    )
    assert "Lọc kỳ chính theo ca chieu" in shift_hint
    assert "Lọc kỳ so sánh theo ca sang" in shift_hint

    price_hint = describe_period(
        "Doanh thu 7 ngày kể từ ngày 02/06/2026 so với 7 ngày trước đó",
        "2026-08-05",
    )
    assert "Mốc tham chiếu" in price_hint
    assert "2026-06-02" in price_hint


def test_unresolved_time_cue_is_exposed_to_context():
    spec = resolve_time("Doanh thu quý 5", "2026-08-27")
    assert spec.time_cue
    assert spec.status == "unresolved"
    assert "chưa phân giải được kỳ" in describe_period("Doanh thu quý 5", "2026-08-27")


def test_tet_resolution_parses_calendar_date_strings():
    calendar = pd.DataFrame({
        "date": ["2026-02-10", "2026-03-01"],
        "lunar_day": [23, 10],
        "lunar_month": [12, 1],
        "lunar_year": [2025, 2026],
        "lunar_leap": ["False", "False"],
    })
    spec = resolve_time("Tet nam 2026", "2026-08-27", calendar)
    assert [(p.start, p.end) for p in spec.main()] == [
        (pd.Timestamp("2026-02-10").date(), pd.Timestamp("2026-03-01").date())
    ]


def test_parse_vietnamese_dates_and_sheet_year():
    assert parse_cell_date("04/05/2026").isoformat() == "2026-05-04"
    assert parse_cell_date("2026-05-04").isoformat() == "2026-05-04"
    assert parse_cell_date("4/5", sheet_year=2025).isoformat() == "2025-05-04"
    assert parse_cell_date("31/02/2026") is None


def test_parse_time_of_day_and_shift():
    assert parse_time_of_day("08:30:15") == 8 * 3600 + 30 * 60 + 15
    assert in_shift(parse_time_of_day("13:59"), "sang")
    assert not in_shift(parse_time_of_day("14:00"), "sang")


def test_rule_date_filter_uses_vietnamese_date_parser_and_sheet_year():
    df = pd.DataFrame({
        "ngay": ["04/05/2026", "2026-05-05", "4/5"],
        "so_luong": [1, 2, 4],
        "_sheet": ["May 2026", "May 2026", "2026"],
    })
    filtered = prep_fact(df, {"date": "ngay", "qty": "so_luong", "amount": None, "status": None},
                         (pd.Timestamp("2026-05-04"), pd.Timestamp("2026-05-05")))
    assert filtered["so_luong"].tolist() == [1, 2, 4]
