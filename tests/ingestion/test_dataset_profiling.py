import pandas as pd

from datamate.ingestion.datasets.loader import DatasetLoader
from datamate.ingestion.datasets.profiling import DatasetProfiler


def test_fix_header_moves_title_row_up():
    df = pd.DataFrame([[None, None, None], ["ten", "sl", "gia"], ["a", 1, 2]], columns=["Unnamed: 0", "Unnamed: 1", "Unnamed: 2"])
    assert list(DatasetLoader.fix_header(df).columns) == ["ten", "sl", "gia"]


def test_load_split_and_profile(split_dirs):
    train, _test, _out = split_dirs
    split = DatasetLoader.load_split(train)
    biz = split.business("b1")
    assert "sales" in biz.tables and biz.readme.startswith("Cửa hàng")
    text = DatasetProfiler.profile_text(biz.tables, 5000)
    assert "so_luong" in text and "head(3)" in text


def test_merge_same_schema_combines_sheets():
    df = pd.DataFrame({"a": [1]})
    tables = {f"hd::T{i}": df for i in range(3)} | {"khac": df}
    merged, notes = DatasetProfiler.merge_same_schema(tables)
    assert "hd::ALL" in merged and len(merged["hd::ALL"]) == 3 and "khac" in merged and notes
