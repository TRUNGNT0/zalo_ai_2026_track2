"""Tổng hợp một lần chạy thành JSON để so sánh model/cấu hình, kèm sai số của PA."""
from __future__ import annotations

import json
import math
import os

import pandas as pd

from evaluation import scoring


def summarize(ev: pd.DataFrame, logs: list[dict], meta: dict) -> dict:
    n = len(ev)
    counts = {s: int((ev.status == s).sum()) for s in ("correct", "wrong", "abstain")}
    score = scoring.pa(ev)
    # độ lệch chuẩn của PA = sqrt(phương sai của (+1/-1/0) / n)
    values = ev.status.map({"correct": 1, "wrong": -1, "abstain": 0})
    stderr = float(values.std(ddof=0) / math.sqrt(n)) if n else 0.0
    answered = counts["correct"] + counts["wrong"]
    intents = pd.Series([lg["intent"] for lg in logs]).value_counts().to_dict() if logs else {}
    seconds = [lg.get("elapsed_s", 0) for lg in logs]
    return {**meta, "n": n, **counts, "pa": round(score, 4), "pa_stderr": round(stderr, 4),
            "coverage": round(answered / n, 4) if n else 0.0,
            "precision": round(counts["correct"] / answered, 4) if answered else None,
            "intents": intents, "avg_seconds_per_question": round(sum(seconds) / len(seconds), 2) if seconds else None,
            "by_gold_type": scoring.breakdown(ev, "gold_type").reset_index().to_dict("records")}


def save_summary(summary: dict, out_dir: str, name: str) -> str:
    os.makedirs(f"{out_dir}/runs", exist_ok=True)
    path = f"{out_dir}/runs/{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    return path
