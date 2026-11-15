import numpy as np
import pandas as pd
import pytest

from datamate.tools.results import canon, clean, coerce_result, extract_code, final_fmt, is_empty_result


def test_clean_drops_nan_and_converts_numpy_types():
    assert clean(float("nan")) is None and clean(np.float64("inf")) is None
    assert clean([1, np.nan]) == [1, None]
    assert clean(pd.NaT) is None
    assert clean(pd.Timestamp("2026-08-27 10:00")) == "2026-08-27"
    assert clean(pd.DataFrame({"a": [1]})) == {"columns": ["a"], "rows": [[1]]}
    assert clean(np.int64(3)) == 3 and clean(np.bool_(True)) is True


def test_coerce_number_accepts_and_rejects():
    assert coerce_result("number", [5.0]) == 5.0
    assert coerce_result("percent", "12.5%") == 12.5
    for bad in (True, None, [1, 2], "abc", float("inf")):
        with pytest.raises(ValueError):
            coerce_result("number", bad)


def test_coerce_date_boolean_table():
    assert coerce_result("date", "2026-08-27 00:00:00") == "2026-08-27"
    with pytest.raises(ValueError):
        coerce_result("date", "27/08/2026")
    with pytest.raises(ValueError):
        coerce_result("boolean", 1)
    with pytest.raises(ValueError):
        coerce_result("table", {"columns": ["a"], "rows": [[1, 2]]})
    assert coerce_result("string", 12) == "12"


def test_extract_code_takes_last_block_and_truncated_block():
    text = "nháp\n```python\nresult=1\n```\nsửa:\n```python\nresult_type='number'\nresult=2\n```"
    assert "result=2" in extract_code(text)
    assert extract_code("```python\nresult_type='number'\nresult=3") is not None
    assert extract_code("không có code") is None


def test_canon_groups_equivalent_values_and_empty_check():
    assert canon("string", " Cà Phê ") == canon("string", "cà phê")
    assert canon("number", 1.001) == canon("number", 1.0)
    assert is_empty_result("list", []) and not is_empty_result("unanswerable", "")
    assert final_fmt("number", 3.0) == 3 and final_fmt("number", 2.456) == 2.46
