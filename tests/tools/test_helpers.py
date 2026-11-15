"""Test hàm trợ giúp tất định."""
import numpy as np
import pandas as pd
import pytest

from datamate.tools.helpers import (HELPERS, counts_as_revenue, fill_group, in_range, norm_text, order_state, period_range,
                               tet_range, to_date, to_number)


@pytest.mark.parametrize("raw,expected", [
    ("1,250,000", 1250000), ("1.250.000đ", 1250000), (" 99,000 ", 99000), ("12,5%", 12.5), ("18.6", 18.6),
    ("125,50", 125.5), ("12.500", 12500), ("1,250", 1250), ("0,25", 0.25), ("1.234,56", 1234.56), ("1,234.56", 1234.56),
    ("(1,000)", -1000), ("-45.000", -45000), (7, 7.0), ("4820000", 4820000)])
def test_to_number_reads_vietnamese_and_english_formats(raw, expected):
    assert to_number(raw) == pytest.approx(expected)


def test_to_number_bad_values_become_nan_and_series_works():
    assert np.isnan(to_number("abc")) and np.isnan(to_number(None)) and np.isnan(to_number(True))
    out = to_number(pd.Series(["1,000", "x", None, 5]))
    assert out.iloc[0] == 1000 and np.isnan(out.iloc[1]) and np.isnan(out.iloc[2]) and out.iloc[3] == 5


def test_to_date_iso_and_dayfirst():
    d = to_date(pd.Series(["05/09/2026", "31/12/2025"]))
    assert d.iloc[0] == pd.Timestamp("2026-09-05") and d.iloc[1] == pd.Timestamp("2025-12-31")
    assert to_date(pd.Series(["2026-09-05", "2026-10-01 10:30:00"])).iloc[0] == pd.Timestamp("2026-09-05")
    assert to_date("07/09/2026") == pd.Timestamp("2026-09-07")


def test_in_range_includes_both_ends_ignoring_time():
    s = pd.Series(["2026-09-01 00:00:01", "2026-09-07 23:59:59", "2026-09-08 00:00:00"])
    assert list(in_range(s, "2026-09-01", "2026-09-07")) == [True, True, False]


@pytest.mark.parametrize("kind,kwargs,expected", [
    ("this_week", {}, ("2026-09-07", "2026-09-07")),                                   # 07/09/2026 là Thứ Hai
    ("last_week", {}, ("2026-08-31", "2026-09-06")),
    ("this_month", {}, ("2026-09-01", "2026-09-07")),
    ("last_month", {}, ("2026-08-01", "2026-08-31")),
    ("prev_month_same_days", {}, ("2026-08-01", "2026-08-07")),
    ("this_year", {}, ("2026-01-01", "2026-09-07")),
    ("last_year", {}, ("2025-01-01", "2025-12-31")),
    ("this_quarter", {}, ("2026-07-01", "2026-09-07")),
    ("last_quarter", {}, ("2026-04-01", "2026-06-30")),
    ("last_n_days", {"n": 7}, ("2026-09-01", "2026-09-07")),
    ("month", {"month": 6}, ("2026-06-01", "2026-06-30")),                              # không ghi năm, không sau hôm nay
    ("month", {"month": 11}, ("2025-11-01", "2025-11-30")),                             # sau hôm nay -> năm trước
    ("month", {"month": 2, "year": 2025}, ("2025-02-01", "2025-02-28")),
    ("quarter", {"quarter": 4}, ("2025-10-01", "2025-12-31")),
    ("year", {"year": 2025}, ("2025-01-01", "2025-12-31"))])
def test_period_range_follows_rulebook(kind, kwargs, expected):
    assert period_range(kind, "2026-09-07", **kwargs) == (pd.Timestamp(expected[0]), pd.Timestamp(expected[1]))


def test_prev_month_same_days_is_capped_by_month_length():
    assert period_range("prev_month_same_days", "2026-03-31") == (pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-28"))
    assert period_range("last_quarter", "2026-02-10") == (pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31"))


def test_period_range_rejects_bad_input():
    with pytest.raises(ValueError):
        period_range("last_n_days", "2026-09-07")
    with pytest.raises(ValueError):
        period_range("nope", "2026-09-07")


@pytest.mark.parametrize("raw,state", [
    ("Huỷ đơn", "cancelled"), ("hủy", "cancelled"), ("Đã hủy đơn", "cancelled"), ("cancel", "cancelled"),
    ("hàng trả về", "returned"), ("Đổi trả", "returned"), ("Hoàn trả", "returned"), ("trả hàng", "returned"),
    ("Đang giao hàng", "shipping"), ("Đang vận chuyển", "shipping"),
    ("Thành công", "done"), ("xong", "done"), ("Đã hoàn thành", "done"), ("Hoàn thành", "done"), ("???", "unknown")])
def test_order_state_handles_all_spellings(raw, state):
    assert order_state(raw) == state


def test_counts_as_revenue_matches_rulebook():
    assert [counts_as_revenue(s) for s in ("xong", "Đang giao hàng", "hủy", "Đổi trả")] == [True, True, False, False]


def test_tet_range_uses_lunar_calendar():
    cal = pd.DataFrame({"date": ["2026-01-12", "2026-02-27", "2025-12-30"], "lunar_day": [23, 10, 23],
                        "lunar_month": [12, 1, 12], "lunar_year": [2025, 2026, 2025], "lunar_leap": [False] * 3})
    assert tet_range(cal, 2026) == (pd.Timestamp("2026-01-12"), pd.Timestamp("2026-02-27"))
    with pytest.raises(ValueError):
        tet_range(cal, 2030)


def test_fill_group_and_exports():
    df = pd.DataFrame({"order": ["a", "a", "b"], "status": ["xong", None, "huy"]})
    assert list(fill_group(df, "order", ["status"])["status"]) == ["xong", "xong", "huy"]
    assert norm_text("  Huỷ  ĐƠN ") == "huy don" and "period_range" in HELPERS
