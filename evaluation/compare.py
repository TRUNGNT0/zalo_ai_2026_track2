"""So sánh các lần chạy đã lưu: python -m evaluation.compare --out-dir <thư mục>"""
from __future__ import annotations

import argparse
import glob
import json

import pandas as pd


def load_runs(out_dir: str) -> pd.DataFrame:
    rows = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(f"{out_dir}/runs/*.json"))]
    cols = ["run_name", "model", "subset", "n", "correct", "wrong", "abstain", "pa", "pa_stderr", "coverage",
            "precision", "avg_seconds_per_question"]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.reindex(columns=cols).sort_values("pa", ascending=False).reset_index(drop=True)


def verdict(df: pd.DataFrame) -> str:
    """Nói thẳng khi chênh lệch giữa hai lần chạy đầu bảng còn nằm trong sai số."""
    if len(df) < 2:
        return ""
    a, b = df.iloc[0], df.iloc[1]
    gap = a.pa - b.pa
    noise = (a.pa_stderr ** 2 + b.pa_stderr ** 2) ** 0.5
    if gap <= noise:
        return f"CHÊNH {gap:+.3f} <= sai số ~{noise:.3f}: chưa đủ cơ sở nói {a.run_name} hơn {b.run_name}. Cần thêm câu holdout."
    return f"{a.run_name} hơn {b.run_name} {gap:+.3f} (> sai số ~{noise:.3f})."


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    df = load_runs(a.out_dir)
    print(df.to_string() if not df.empty else "chưa có lần chạy nào trong runs/")
    print(verdict(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
