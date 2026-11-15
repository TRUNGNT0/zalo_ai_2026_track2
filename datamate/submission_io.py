"""Ghi bÃ i ná»™p vÃ  nháº­t kÃ½ tá»«ng cÃ¢u theo Ä‘á»‹nh dáº¡ng cá»§a ban tá»• chá»©c (Docker guideline má»¥c 3.2 vÃ  3.3).

Chá»‰ dÃ¹ng thÆ° viá»‡n chuáº©n Ä‘á»ƒ kiá»ƒm thá»­ riÃªng Ä‘Æ°á»£c. NguyÃªn táº¯c: cÃ³ Ä‘Ã¡p Ã¡n nÃ o lÃ  ghi ngay; lá»¡ bá»‹ dá»«ng giá»¯a chá»«ng
thÃ¬ nhá»¯ng cÃ¢u Ä‘Ã£ ghi váº«n Ä‘Æ°á»£c cháº¥m, cÃ¢u cÃ²n láº¡i lÃ  abstain.
"""
from __future__ import annotations

import json
import os


class IncrementalSubmission:
    """Giá»¯ Ä‘Ã¡p Ã¡n trong bá»™ nhá»› vÃ  ghi láº¡i `submission.jsonl` nguyÃªn tá»­ sau má»—i `flush_every` cÃ¢u.

    Ngay khi táº¡o Ä‘á»‘i tÆ°á»£ng, file ná»™p Ä‘Ã£ Ä‘áº§y Ä‘á»§ má»i id vá»›i abstain (há»£p lá»‡ náº¿u cháº¡y cháº¿t sá»›m).
    Má»—i cÃ¢u má»™t dÃ²ng á»Ÿ `logs/questions.jsonl` vá»›i id, answer, elapsed_ms, thinking, code, output.
    """

    def __init__(self, result_dir: str, ids, flush_every: int = 5):
        self.ids = list(dict.fromkeys(ids))               # giá»¯ thá»© tá»± file cÃ¢u há»i, bá» id trÃ¹ng
        self.rows: dict = {}
        self.logged: set = set()
        self.flush_every = max(1, flush_every)
        self.path = os.path.join(result_dir, "submission.jsonl")
        self.log_dir = os.path.join(result_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, "questions.jsonl")
        self._log = open(self.log_path, "a", encoding="utf-8")
        self._write_submission()

    @staticmethod
    def dump_line(obj: dict) -> str:
        """Má»™t dÃ²ng JSON chuáº©n (cáº¥m NaN/Infinity). KhÃ´ng ghi Ä‘Æ°á»£c thÃ¬ cÃ¢u Ä‘Ã³ thÃ nh abstain thay vÃ¬ há»ng cáº£ file."""
        try:
            return json.dumps(obj, ensure_ascii=False, default=str, allow_nan=False)
        except ValueError:
            return json.dumps({"id": obj.get("id"), "type": "abstain"}, ensure_ascii=False)

    def _write_submission(self) -> None:
        rows = [self.rows.get(i) or {"id": i, "type": "abstain"} for i in self.ids]
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(self.dump_line(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, self.path)                        # Ä‘á»•i tÃªn nguyÃªn tá»­: ngÆ°á»i Ä‘á»c khÃ´ng bao giá» tháº¥y file dá»Ÿ

    @staticmethod
    def record(sub: dict, log: dict) -> dict:
        """Báº£n ghi nháº­t kÃ½ cá»§a má»™t cÃ¢u. `thinking` lÃ  null vÃ¬ táº¯t cháº¿ Ä‘á»™ suy nghÄ©; `code` lÃ  SQL Ä‘Ã£ cháº¡y (náº¿u cÃ³)."""
        answer = {key: value for key, value in sub.items() if key != "id"}
        try:
            elapsed_ms = max(0, int(round(float(log.get("elapsed_s") or 0.0) * 1000)))
        except (TypeError, ValueError):
            elapsed_ms = 0
        return {"id": sub["id"], "answer": answer, "elapsed_ms": elapsed_ms, "thinking": log.get("thinking"),
                "code": log.get("sql") or None, "output": log.get("answer"), "source": log.get("source"),
                "agree": log.get("agree"), "intent": log.get("intent"), "note": str(log.get("note") or "")[:300],
                "rewritten_question": log.get("rewritten_question")}

    def _append_log(self, record: dict) -> None:
        self._log.write(self.dump_line(record) + "\n")
        self._log.flush()
        self.logged.add(record["id"])

    def add(self, sub: dict, log: dict) -> None:
        """Gá»i sau má»—i cÃ¢u: nhá»› Ä‘Ã¡p Ã¡n, ghi nháº­t kÃ½ ngay, vÃ  cá»© `flush_every` cÃ¢u thÃ¬ ghi láº¡i file ná»™p."""
        if sub["id"] not in self.ids or sub["id"] in self.logged:
            return
        self.rows[sub["id"]] = sub
        self._append_log(self.record(sub, log))
        if len(self.logged) % self.flush_every == 0:
            self._write_submission()

    def finalize(self, note: str = "not processed") -> int:
        """Ghi ná»‘t: cÃ¢u chÆ°a cÃ³ báº£n ghi nháº­t kÃ½ Ä‘Æ°á»£c ghi abstain vá»›i elapsed_ms 0. Tráº£ sá»‘ cÃ¢u chÆ°a Ä‘Æ°á»£c xá»­ lÃ½."""
        missing = [i for i in self.ids if i not in self.logged]
        for qid in missing:
            self._append_log({"id": qid, "answer": {"type": "abstain"}, "elapsed_ms": 0, "thinking": None,
                              "code": None, "output": None, "source": "abstain", "note": note})
        self._write_submission()
        self._log.close()
        return len(missing)

