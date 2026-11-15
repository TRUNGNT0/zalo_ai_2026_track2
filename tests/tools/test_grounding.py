"""Test khớp từ trong câu hỏi với giá trị thật trong dữ liệu."""
import pandas as pd

from datamate.tools.grounding import ValueIndex, format_hits


def make_index():
    df = pd.DataFrame({"ten_hang": ["Cà phê sữa", "Trà sữa trân châu", "Bánh mì"], "so_luong": [1, 2, 3],
                       "ngay": ["2026-08-01"] * 3})
    return ValueIndex({"sales": df})


def test_matches_ignoring_accents_and_case():
    hits = make_index().ground("Tuan nay CA PHE SUA ban duoc bao nhieu")
    assert [(h.phrase, h.value, h.column) for h in hits] == [("ca phe sua", "Cà phê sữa", "ten_hang")]


def test_prefers_longest_phrase_over_its_parts():
    hits = make_index().ground("doanh thu tra sua tran chau")
    assert [h.phrase for h in hits] == ["tra sua tran chau"]          # không báo thêm "tra sua"


def test_fuzzy_match_tolerates_small_typo():
    hits = make_index().ground("tra sua tran chao hom qua")
    assert hits and hits[0].value == "Trà sữa trân châu" and not hits[0].exact


def test_ignores_numbers_dates_and_numeric_columns():
    index = make_index()
    assert index.ground("2026-08-01 va 2") == []
    assert all("so_luong" not in line and "ngay" not in line for line in format_hits(index.ground("banh mi")))


def test_format_hits_mentions_table_and_column():
    line = format_hits(make_index().ground("banh mi"))[0]
    assert "Bánh mì" in line and "bảng sales" in line and "cột ten_hang" in line
