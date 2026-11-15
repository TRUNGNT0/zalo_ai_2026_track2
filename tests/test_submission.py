import json

from datamate.submission import dump_line, make_sub, validate_submission, write_submission


def test_writes_every_id_in_order_without_nan(tmp_path):
    path = str(tmp_path / "s.jsonl")
    subs = [make_sub("q2", "number", 5), make_sub("q0", "number", float("nan"))]
    rows = write_submission(subs, ["q0", "q1", "q2"], path)
    assert [r["id"] for r in rows] == ["q0", "q1", "q2"]
    lines = [json.loads(line) for line in open(path, encoding="utf-8")]
    assert lines[0] == {"id": "q0", "type": "abstain"}            # NaN -> abstain thay vì file hỏng
    assert lines[1] == {"id": "q1", "type": "abstain"}            # câu chưa chạy
    assert validate_submission(path, ["q0", "q1", "q2"]) == []


def test_validate_catches_problems(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"a","type":"number"}\n{"id":"a","type":"zzz"}\n{"id":"b","type":"number","answer":NaN}\n',
                    encoding="utf-8")
    problems = validate_submission(str(path), ["a", "b", "c"])
    text = " | ".join(problems)
    assert "thiếu answer" in text and "id trùng" in text and "type lạ" in text and "NaN" in text and "thiếu id c" in text


def test_dump_line_unanswerable_has_reason():
    assert json.loads(dump_line(make_sub("q", "unanswerable", "thiếu dữ liệu")))["reason"] == "thiếu dữ liệu"
