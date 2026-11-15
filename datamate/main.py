"""Điểm vào lúc nộp bài: python -m app.main  (đọc câu hỏi ở --test-dir, ghi submission.jsonl ở --out-dir)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datamate.agent.pipeline import run_inference
from datamate.config import Config
from datamate.llm import create_backend
from datamate.llm.local_common import is_qwen3_family


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--backend", choices=["local", "gemini"], default="local")
    p.add_argument("--query-engine", choices=["sql", "pandas"], default="sql")
    p.add_argument("--database-dir")
    p.add_argument("--ingestion-overrides", type=Path, help="JSON keyed by business id, shared with ingestion CLI")
    p.add_argument("--rebuild-database", action="store_true")
    p.add_argument("--no-rewrite", action="store_true")
    p.add_argument("--limit-train", type=int)
    p.add_argument("--train-dir")
    p.add_argument("--test-dir", default=os.environ.get("TEST_DIR"))
    p.add_argument("--out-dir", default=os.environ.get("OUT_DIR"))
    p.add_argument("--max-biz", type=int, help="số business train để chạy thử")
    p.add_argument("--per-biz", type=int, help="tối đa câu mỗi business")
    p.add_argument("--limit-test", type=int)
    p.add_argument("--time-budget", type=float, help="giây; hết giờ thì câu còn lại ghi abstain")
    p.add_argument("--n-samples", type=int)
    p.add_argument("--agree-min", type=int)
    p.add_argument("--gemini-model")
    p.add_argument("--local-model", default=os.environ.get("LOCAL_MODEL"))
    p.add_argument("--local-engine", choices=["auto", "vllm", "hf"])
    p.add_argument("--local-lora", help="thư mục LoRA adapter (sau fine-tune); nên gộp vào model rồi dùng --local-model")
    p.add_argument("--temperature", type=float)
    p.add_argument("--max-model-len", type=int, help="tổng token tối đa (prompt + trả lời); giảm khi hết VRAM, vd 12288")
    p.add_argument("--max-retry", type=int)
    p.add_argument("--hf-quant", choices=["auto", "4bit", "none"],
                   help="engine hf: none = nạp bf16 đầy đủ (nhanh), 4bit = nén bitsandbytes (ít VRAM, chậm); auto = theo VRAM")
    p.add_argument("--no-llm", action="store_true", help="không gọi model (chỉ luật, nếu bật)")
    p.add_argument("--exemplars", help="kho few-shot (.jsonl) dựng từ train bằng evaluation.run_eval --build-exemplars")
    p.add_argument("--no-hints", action="store_true", help="tắt gợi ý tất định (khoảng thời gian, giá trị khớp)")
    p.add_argument("--no-helpers", action="store_true", help="không nêu hàm trợ giúp (to_number, period_range...) trong prompt")
    p.add_argument("--auto-model", action="store_true", help="chọn model/engine theo VRAM GPU (ghi đè --local-model/--local-engine)")
    p.add_argument("--rulebook-select", action="store_true", help="chỉ đưa mục rulebook liên quan câu hỏi vào prompt")
    p.add_argument("--no-thinking", action="store_true", help="tắt chế độ suy nghĩ của model (Qwen3): enable_thinking=False")
    p.add_argument("--collect-all", action="store_true", help="sinh đủ n-samples, không dừng sớm (thu dữ liệu SFT/few-shot)")
    p.add_argument("--run-name", default="", help="nhãn lần chạy: tách cache và file kết quả giữa các model/cấu hình")
    return p


def config_from_args(a: argparse.Namespace) -> Config:
    mapping = {"train_dir": a.train_dir, "test_dir": a.test_dir, "out_dir": a.out_dir, "max_biz": a.max_biz,
               "per_biz": a.per_biz, "limit_test": a.limit_test, "time_budget_s": a.time_budget,
               "n_samples": a.n_samples, "agree_min": a.agree_min, "gemini_model": a.gemini_model,
               "local_model": a.local_model, "local_engine": a.local_engine, "local_lora": a.local_lora,
               "temperature": a.temperature, "max_retry": a.max_retry, "max_model_len": a.max_model_len,
               "hf_quant": a.hf_quant}
    mapping["exemplar_path"] = a.exemplars
    mapping.update(query_engine=a.query_engine, database_dir=a.database_dir,
                   ingestion_overrides=json.loads(a.ingestion_overrides.read_text()) if a.ingestion_overrides else {},
                   rebuild_database=a.rebuild_database, rewrite_questions=not a.no_rewrite,
                   limit_train=a.limit_train)
    cfg = Config(backend=a.backend, use_hints=not a.no_hints, use_helpers=not a.no_helpers, collect_all=a.collect_all, rulebook_select=a.rulebook_select,
                 run_name=a.run_name, chat_kwargs={"enable_thinking": False} if a.no_thinking else {},
                 **{k: v for k, v in mapping.items() if v is not None})
    if a.auto_model and cfg.is_local:
        from datamate.llm.hardware import apply_recommendation, detect_gpu
        gpu = detect_gpu()
        cfg = apply_recommendation(cfg, gpu)
        print(f"GPU {gpu['name']} {gpu['vram_gb']}GB -> model {cfg.local_model} engine {cfg.local_engine}")
    if cfg.is_local and not cfg.chat_kwargs and is_qwen3_family(cfg.local_model):
        cfg.chat_kwargs = {"enable_thinking": False}      # Qwen3/3.5 mặc định suy nghĩ dài: tắt để 1024 token đủ cho code
        print("Họ Qwen3: tự đặt enable_thinking=False")
    return cfg


def main(argv=None) -> int:
    parser = add_common_args(argparse.ArgumentParser(prog="datamate.main", description=__doc__))
    a = parser.parse_args(argv)
    cfg = config_from_args(a)
    backend = None if a.no_llm else create_backend(cfg)
    _subs, _logs, problems = run_inference(cfg, backend, use_llm=not a.no_llm)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
