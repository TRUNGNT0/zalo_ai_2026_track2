"""Tạo, ghi và kiểm định file nộp submission.jsonl."""
from __future__ import annotations

import json
import os

from datamate.tools.results import VALID_TYPES

MAX_BYTES = 5 * 1024 * 1024


def make_sub(qid, typ, ans) -> dict:
    if typ in (None, "abstain"):
        return {"id": qid, "type": "abstain"}
    if typ == "unanswerable":
        return {"id": qid, "type": typ, "reason": str(ans)[:300]}
    return {"id": qid, "type": typ, "answer": ans}


def dump_line(sub: dict) -> str:
    """Một dòng JSON chuẩn (cấm NaN/Infinity). Không ghi được thì chuyển câu đó thành abstain thay vì hỏng cả file."""
    try:
        return json.dumps(sub, ensure_ascii=False, default=str, allow_nan=False)
    except ValueError:
        return json.dumps({"id": sub["id"], "type": "abstain"}, ensure_ascii=False)


def fill_missing(subs: list[dict], ids) -> list[dict]:
    """Mỗi id trong questions.jsonl có đúng một dòng, theo thứ tự file câu hỏi; câu chưa chạy = abstain."""
    by_id = {s["id"]: s for s in subs}
    return [by_id.get(i) or {"id": i, "type": "abstain"} for i in ids]


def write_submission(subs: list[dict], ids, path: str) -> list[dict]:
    rows = fill_missing(subs, ids)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in rows:
            f.write(dump_line(s) + "\n")
    return rows


def _bad_const(c):
    raise ValueError(f"hằng không hợp lệ trong JSON: {c}")


def validate_submission(path: str, ids) -> list[str]:
    """Kiểm nhanh trước khi nộp: JSON chuẩn, đủ và duy nhất id, type hợp lệ, có answer khi cần, dưới 5MB."""
    problems, seen = [], set()
    if os.path.getsize(path) > MAX_BYTES:
        problems.append("file > 5MB")
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        try:
            d = json.loads(line, parse_constant=_bad_const)
        except ValueError as e:
            problems.append(f"dòng {n}: {e}")
            continue
        if d.get("id") in seen:
            problems.append(f"dòng {n}: id trùng {d.get('id')}")
        seen.add(d.get("id"))
        t = d.get("type")
        if t not in VALID_TYPES | {"abstain"}:
            problems.append(f"dòng {n}: type lạ {t!r}")
        elif t not in ("abstain", "unanswerable") and d.get("answer") is None:
            problems.append(f"dòng {n}: thiếu answer")
    problems += [f"thiếu id {i}" for i in list(set(ids) - seen)[:5]]
    problems += [f"id lạ {i}" for i in list(seen - set(ids))[:5]]
    return problems


def check(sub_path, questions_path) -> list[str]:
    from pathlib import Path
    q_ids = [json.loads(l)["id"] for l in Path(questions_path).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    return validate_submission(str(sub_path), q_ids)


def merge_shards(out_dir, questions_path, shards=None, quiet=False, copy_logs=True) -> None:
    from pathlib import Path
    out_dir = Path(out_dir)
    shard_dirs = sorted(shards.glob("shard*")) if shards else sorted(out_dir.glob("shard*"))
    all_subs = {}
    for sd in shard_dirs:
        sub_file = sd / "submission.jsonl"
        if sub_file.exists():
            for line in sub_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("type") != "abstain" or r["id"] not in all_subs:
                        all_subs[r["id"]] = r
    q_ids = [json.loads(l)["id"] for l in Path(questions_path).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    write_submission(list(all_subs.values()), q_ids, str(out_dir / "submission.jsonl"))
    if not quiet:
        print(f"Merged {len(shard_dirs)} shards into {out_dir / 'submission.jsonl'}")


def fill_logs(log_dir, questions_path):
    from pathlib import Path
    log_dir = Path(log_dir)
    questions = Path(questions_path)
    n = 0
    for line in questions.open(encoding="utf-8-sig"):
        if not line.strip():
            continue
        q = json.loads(line)
        f = log_dir / f"{q['id']}.json"
        if not f.exists():
            as_of = q.get("as_of_date") or q.get("as_of_day")
            rec = {"id": q["id"], "business_id": q.get("business_id"), "question": q.get("question"),
                   "as_of_date": as_of, "as_of_day": as_of,
                   "answer": {"type": "abstain"}, "elapsed_ms": 0, "thinking": None, "code": None, "output": None,
                   "note": "not processed: the business failed to load or the run stopped before reaching it"}
            f.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
            n += 1
    print(f"{n} questions without a log record got an abstain record in {log_dir}")
    return n


def main():
    import argparse
    import sys
    from pathlib import Path
    ap = argparse.ArgumentParser(description="Datamate submission utilities")
    ap.add_argument("command", choices=["check", "merge", "fill-logs"])
    ap.add_argument("path", type=Path, help="check: submission file; merge: run dir; fill-logs: log dir")
    ap.add_argument("--questions", type=Path, default=Path("dataset/public_test/questions.jsonl"))
    ap.add_argument("--shards", type=Path, help="merge: directory holding shard*/")
    ap.add_argument("--no-copy-logs", action="store_true", help="merge: leave shards logs where they are")
    args = ap.parse_args()
    if args.command == "merge":
        merge_shards(args.path, args.questions, shards=args.shards, copy_logs=not args.no_copy_logs)
        return
    if args.command == "fill-logs":
        fill_logs(args.path, args.questions)
        return
    errors = check(args.path, args.questions)
    for e in errors[:20]:
        print("ERROR", e)
    if errors:
        sys.exit(1)
    print("format OK")


if __name__ == "__main__":
    main()


