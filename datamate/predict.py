"""CLI entry point for Datamate ZAIC Hybrid Pipeline.

Supports running SQL prediction over shops with DuckDB, LLM-based rewrite, multi-sampling,
automatic error repair, consensus voting, abstention, and canonical views/debt layers.

Usage:
    python -m datamate.predict --data dataset/public_test --output runs/public --model Qwen/Qwen3.5-9B
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from datamate.config import Config
from datamate.llm import create_backend
from datamate.agent.pipeline import run_inference


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Datamate ZAIC 2026 Inference")
    p.add_argument("--data", type=Path, required=True, help="Path to data split folder (e.g. dataset/public_test)")
    p.add_argument("--output", type=Path, required=True, help="Output directory for submission.jsonl and logs")
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B", help="Model name or local path")
    p.add_argument("--lora", type=str, nargs="*", default=None, help="Optional LoRA adapter directory or list of adapters")
    p.add_argument("--backend", choices=("local", "gemini"), default="local", help="LLM backend")
    p.add_argument("--local-engine", choices=("auto", "vllm", "hf"), default="auto", help="Local inference engine")
    p.add_argument("--hf-quant", choices=("auto", "4bit", "none"), default="auto", help="HF quantization mode")
    p.add_argument("-k", "--k", "--n-samples", dest="k", type=int, default=3, help="Number of SQL samples per question")
    p.add_argument("--agree-min", type=int, default=2, help="Minimum matching consensus needed to answer")
    p.add_argument("--max-retry", "--repairs", dest="max_retry", type=int, default=2, help="Max self-repair turns for failed SQL queries")
    p.add_argument("--views", choices=("cached", "llm", "none"), default="cached", help="Semantic views mode")
    p.add_argument("--views-from", type=Path, default=Path("configs/public_v9"), help="Directory containing saved views JSONs")
    p.add_argument("--ingestion-mode", choices=("direct", "layered"), default="direct",
                   help="direct reads source files into DuckDB; layered builds manifests/catalog during inference")
    p.add_argument("--fewshot", type=Path, default=Path("assets/fewshot_bank.jsonl"), help="Few-shot exemplars bank file")
    p.add_argument("--fewshot-k", type=int, default=4, help="Number of fewshot examples")
    p.add_argument("--limit", type=int, default=None, help="Max questions to run")
    p.add_argument("--ids", type=Path, default=None, help="Optional list of question IDs to run")
    p.add_argument("--max-biz", type=int, default=None, help="Max businesses to run")
    p.add_argument("--time-budget", "--deadline", dest="time_budget", type=float, default=None, help="Max execution time in seconds")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p.add_argument("--threshold", type=float, default=0.5, help="Min vote share to answer")
    p.add_argument("--shard", default="0/1", help="i/n shard for distributed GPU processes")
    p.add_argument("--quant", type=str, default=None, help="Quantization mode (e.g. fp8, 4bit)")
    p.add_argument("--context", type=int, default=32768, help="Context length")
    p.add_argument("--max-tokens", type=int, default=2048, help="Max tokens generated")
    p.add_argument("--think-k", type=int, default=0, help="Thinking samples count")
    p.add_argument("--think-budget", type=int, default=8192, help="Thinking token budget")
    p.add_argument("--think-below", type=float, default=0.75, help="Thinking threshold")
    p.add_argument("--gpu-mem", type=float, default=0.9, help="GPU memory fraction")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    p.add_argument("--engine", type=str, default=None, help="Extra engine config JSON string")
    p.add_argument("--log-dir", type=Path, default=None, help="Custom log dir")
    p.add_argument("--merge-into", type=Path, default=None, help="Merge into directory")
    p.add_argument("--view-think", type=int, default=0, help="View thinking budget")
    # Boolean flags from flags.sh / datamate-26
    p.add_argument("--notes", action="store_true", help="Add notes before SQL")
    p.add_argument("--lora-thinking", action="store_true", help="Use LoRA for thinking samples")
    p.add_argument("--debt-views", action="store_true", default=True, help="Load debt views")
    p.add_argument("--debt-route", action="store_true", help="Route debt questions to debt macros")
    p.add_argument("--canon-names", action="store_true", default=True, help="Canonicalize names")
    p.add_argument("--answer-unanswerable", action="store_true", default=True, help="Allow unanswerable answers")
    p.add_argument("--catalog-cols", action="store_true", help="Include catalog columns")
    p.add_argument("--hide-raw", action="store_true", help="Hide raw tables")
    p.add_argument("--rule-lines", action="store_true", help="Include rule lines")
    p.add_argument("--save-prompts", action="store_true", help="Save prompts in trace logs")
    args = p.parse_args(argv)
    if not hasattr(args, "repairs"):
        args.repairs = getattr(args, "max_retry", 2)
    return args


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(parents=True, exist_ok=True)
    lora_path = args.lora[0] if isinstance(args.lora, list) and args.lora else args.lora

    n_samples = getattr(args, "k", 3)
    args.n_samples = n_samples

    cfg = Config(
        test_dir=str(args.data),
        train_dir=str(args.data.parent / "train") if (args.data.parent / "train").exists() else str(args.data),
        out_dir=str(args.output),
        database_dir=str(args.output / "databases"),
        manifest_dir=str(args.output / "manifests") if (args.output / "manifests").exists() else None,
        ingestion_mode=args.ingestion_mode,
        backend=args.backend,
        query_engine="sql",
        local_model=args.model,
        local_lora=lora_path,
        local_engine=args.local_engine,
        hf_quant=args.hf_quant,
        n_samples=n_samples,
        agree_min=args.agree_min,
        max_retry=args.max_retry,
        temperature=args.temperature,
        limit_test=args.limit,
        max_biz=args.max_biz,
        time_budget_s=args.time_budget,
        views=args.views,
        views_from=str(args.views_from) if args.views_from and args.views_from.is_dir() else None,
        fewshot_bank=str(args.fewshot) if args.fewshot and args.fewshot.is_file() else None,
    )

    print(f"=== Datamate ZAIC 2026 Pipeline ===")
    print(f"Data: {args.data} | Output: {args.output}")
    print(f"Model: {args.model} (engine={args.local_engine}, quant={args.hf_quant})")
    print(f"Sampling: {args.n_samples} samples, agree_min={args.agree_min}, views={args.views}")

    backend = create_backend(cfg) if cfg.backend else None
    subs, logs, problems = run_inference(cfg, backend)

    elapsed = round(time.time() - t0, 1)
    print(f"Hoàn tất sau {elapsed}s | Số câu: {len(subs)} | Vấn đề kiểm định: {len(problems)}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
