import pandas as pd

from evaluation.scoring import evaluate, match, norm_official, pa


def test_match_string_keeps_tone_marks_and_alias():
    assert match("Cà  phê!", "cà phê", None, "string", "string")
    assert not match("ca phe", "cà phê", None, "string", "string")
    assert match("cf", "cà phê", ["CF"], "string", "string")
    assert not match("cf", "cà phê", ["CF"], "string", "string", use_alias=False)


def test_match_number_type_and_unanswerable():
    assert match(130000.004, 130000, None, "number", "number")
    assert not match(1, 1, None, "number", "percent")          # sai type = sai
    assert match("lý do", None, None, "unanswerable", "unanswerable")
    assert not match(1, None, None, "number", "unanswerable")


def test_match_table_ignores_column_and_row_order():
    pred = {"columns": ["B", "A"], "rows": [[2, 1], [4, 3]]}
    gold = {"columns": ["a", "b"], "rows": [[3, 4], [1, 2]]}
    assert match(pred, gold, None, "table", "table")


def test_evaluate_and_pa():
    subs = [{"id": "q0", "type": "number", "answer": 5}, {"id": "q1", "type": "number", "answer": 9},
            {"id": "q2", "type": "abstain"}]
    logs = [{"id": i, "source": "llm", "intent": "llm", "agree": 2} for i in ("q0", "q1", "q2")]
    ans = pd.DataFrame({"id": ["q0", "q1", "q2"], "type": "number", "answer": [5, 5, 5]})
    ev = evaluate(subs, logs, ans)
    assert list(ev.status) == ["correct", "wrong", "abstain"]
    assert pa(ev) == 0.0
    assert norm_official("A_b-C") == "a b c"
