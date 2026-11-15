"""Chấm theo luật cuộc thi: Penalized Accuracy (đúng +1, sai -1, abstain 0)."""
from __future__ import annotations

import json
import re
import unicodedata

import pandas as pd

from datamate.tools.results import is_null


def norm_official(s) -> str:
    """NFC + lowercase + ký tự không phải chữ/số/khoảng trắng -> khoảng trắng + gộp khoảng trắng. GIỮ dấu thanh."""
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = re.sub(r"[^\w\s]|_", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _cell(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return round(float(x), 2)
    if isinstance(x, str):
        try:
            return round(float(x), 2)
        except ValueError:
            return norm_official(x)
    return x


def match(pred, gold, alias, typ: str, gold_type: str, use_alias: bool = True) -> bool:
    """Đáp án `pred` (type `typ`) có khớp đáp án chuẩn `gold` (type `gold_type`) không."""
    if gold_type == "unanswerable":
        return typ == "unanswerable"
    if typ != gold_type or is_null(gold):          # type phải bằng type chuẩn
        return False
    try:
        if typ == "string":
            ok = {norm_official(gold)} | ({norm_official(a) for a in (alias or [])} if use_alias else set())
            return norm_official(pred) in ok
        if typ in ("number", "percent"):
            return round(float(pred), 2) == round(float(gold), 2)
        if typ == "date":
            return str(pred).strip() == str(gold).strip()
        if typ == "boolean":
            return bool(pred) == bool(gold)
        if typ == "list":
            return [_cell(x) for x in pred] == [_cell(x) for x in gold]
        if typ == "table":
            pc, gc = [norm_official(c) for c in pred["columns"]], [norm_official(c) for c in gold["columns"]]
            prow = [[_cell(x) for x in row] for row in pred["rows"]]
            if set(pc) == set(gc) and len(pc) == len(gc):
                idx = [pc.index(c) for c in gc]
                prow = [[row[i] for i in idx] for row in prow]
            elif len(pc) != len(gc):
                return False
            grow = [[_cell(x) for x in row] for row in gold["rows"]]

            def as_set(rows):
                return {json.dumps(r, ensure_ascii=False, default=str) for r in rows}
            return as_set(prow) == as_set(grow)
    except Exception:
        return False
    return pred == gold


def evaluate(subs: list[dict], logs: list[dict], answers: pd.DataFrame, use_alias: bool = True) -> pd.DataFrame:
    """Ghép bài nộp + log + đáp án chuẩn, thêm cột `status` ∈ {correct, wrong, abstain}."""
    sub_df = pd.DataFrame(subs)
    if "answer" not in sub_df:
        sub_df["answer"] = None
    log_df = pd.DataFrame(logs)[["id", "source", "intent", "agree"]]
    gold = answers.rename(columns={"type": "gold_type", "answer": "gold"})
    if "alias" not in gold:
        gold["alias"] = None
    ev = sub_df.merge(log_df, on="id", how="left").merge(gold, on="id", how="left")
    ev["answered"] = ev["type"] != "abstain"

    def status(r):
        if not r.answered:
            return "abstain"
        alias = r.alias if isinstance(r.alias, list) else None
        return "correct" if match(r.answer, r.gold, alias, r.type, r.gold_type, use_alias) else "wrong"
    ev["status"] = [status(r) for r in ev.itertuples()]
    return ev


def pa(df: pd.DataFrame) -> float:
    return ((df.status == "correct").sum() - (df.status == "wrong").sum()) / max(len(df), 1)


def breakdown(ev: pd.DataFrame, by: str) -> pd.DataFrame:
    """Bảng đúng/sai/abstain/precision/PA theo một cột (source, intent, gold_type...)."""
    t = ev.groupby(by).agg(n=("id", "size"), correct=("status", lambda s: (s == "correct").sum()),
                           wrong=("status", lambda s: (s == "wrong").sum()),
                           abstain=("status", lambda s: (s == "abstain").sum()))
    t["precision"] = (t.correct / (t.correct + t.wrong)).round(3)
    t["PA"] = ((t.correct - t.wrong) / t.n).round(3)
    return t


def summary_text(ev: pd.DataFrame) -> str:
    n = len(ev)
    c, w, a = [(ev.status == s).sum() for s in ("correct", "wrong", "abstain")]
    lines = [f"Tổng câu: {n} | đúng {c} (+1) | sai {w} (−1) | abstain {a} (0)",
             f"==> PA = {pa(ev):+.4f}   (trả lời hết, không abstain sẽ là accuracy*2-1)"]
    if c + w:
        lines.append(f"Độ phủ: {(c + w) / n:.1%} | precision trên câu đã trả lời: {c / (c + w):.1%} "
                     "(cần > 50% mới có lợi khi trả lời)")
    return "\n".join(lines)
