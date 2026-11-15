"""Tóm tắt lần chạy và so sánh có tính sai số."""
import pandas as pd

from datamate.llm.hardware import MODEL_9B, apply_recommendation, recommend
from datamate.config import Config
from evaluation.compare import load_runs, verdict
from evaluation.report import save_summary, summarize


def make_ev(statuses):
    return pd.DataFrame({"id": [f"q{i}" for i in range(len(statuses))], "status": statuses,
                         "gold_type": ["number"] * len(statuses)})


def test_summarize_reports_pa_and_stderr():
    s = summarize(make_ev(["correct"] * 6 + ["wrong"] * 3 + ["abstain"] * 21), [], {"run_name": "r"})
    assert s["n"] == 30 and s["pa"] == 0.1 and s["coverage"] == 0.3 and s["precision"] == 0.6667
    assert 0.09 < s["pa_stderr"] < 0.11          # khớp ước tính ±0,10 đã nêu cho 30 câu


def test_compare_flags_differences_within_noise(tmp_path):
    a = summarize(make_ev(["correct"] * 6 + ["wrong"] * 3 + ["abstain"] * 21), [], {"run_name": "a", "model": "m1", "subset": "holdout"})
    b = summarize(make_ev(["correct"] * 5 + ["wrong"] * 3 + ["abstain"] * 22), [], {"run_name": "b", "model": "m2", "subset": "holdout"})
    save_summary(a, str(tmp_path), "a"), save_summary(b, str(tmp_path), "b")
    df = load_runs(str(tmp_path))
    assert list(df.run_name) == ["a", "b"] and "chưa đủ cơ sở" in verdict(df)


def test_compare_accepts_clear_winner(tmp_path):
    a = summarize(make_ev(["correct"] * 60 + ["abstain"] * 40), [], {"run_name": "a", "model": "m", "subset": "h"})
    b = summarize(make_ev(["wrong"] * 30 + ["abstain"] * 70), [], {"run_name": "b", "model": "m", "subset": "h"})
    save_summary(a, str(tmp_path), "a"), save_summary(b, str(tmp_path), "b")
    assert "a hơn b" in verdict(load_runs(str(tmp_path)))


def test_hardware_recommendation_by_vram():
    assert recommend(40)["local_model"] == MODEL_9B and recommend(80)["local_engine"] == "hf"
    cfg = apply_recommendation(Config(backend="local"), {"name": "A100", "vram_gb": 40.0, "bf16": True, "count": 1})
    assert cfg.local_model == MODEL_9B
