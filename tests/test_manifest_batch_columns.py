from datamate.manifests.local import LocalManifestGenerator


def make(names):
    return {"tables": [{"columns": [{"source_name": n} for n in names]}]}


def test_extra_and_duplicate_columns_are_dropped_in_order():
    part, dropped = LocalManifestGenerator.restrict_columns(make(["A", "B", "X", "A", "Y"]), ["A", "B", "C"])
    assert [c["source_name"] for c in part["tables"][0]["columns"]] == ["A", "B"]
    assert dropped == ["X", "A", "Y"]


def test_exact_match_is_untouched():
    part, dropped = LocalManifestGenerator.restrict_columns(make(["A", "B"]), ["A", "B"])
    assert dropped == [] and len(part["tables"][0]["columns"]) == 2


def test_vietnamese_names_and_missing_detection():
    expected = ["Đơn vị tính", "Giá bán", "Giá vốn"]
    part, dropped = LocalManifestGenerator.restrict_columns(make(["Đơn vị tính", "Mã hàng hóa", "Giá bán"]), expected)
    names = {c["source_name"] for c in part["tables"][0]["columns"]}
    assert [n for n in expected if n not in names] == ["Giá vốn"] and dropped == ["Mã hàng hóa"]
