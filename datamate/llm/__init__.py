"""Tạo backend LLM theo cấu hình."""
from __future__ import annotations

from datamate.config import Config
from .base import GenerationError, LLMBackend, QuotaExhausted

__all__ = ["GenerationError", "LLMBackend", "QuotaExhausted", "create_backend"]


def _create_local(cfg: Config) -> LLMBackend:
    import gc

    import torch
    from .local_common import has_module, resolve_model_path

    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        msg = "Không thấy GPU: vào Settings -> Accelerator chọn GPU T4 x2 (hoặc P100), rồi chạy lại từ đầu."
        if not cfg.allow_cpu:
            raise RuntimeError(msg + " (Đặt allow_cpu=True nếu CỐ Ý chạy CPU để thử luồng với model nhỏ.)")
        print(msg, "Chạy CPU sẽ rất chậm.")
    path = resolve_model_path(cfg.local_model)
    engine = cfg.local_engine if cfg.local_engine != "auto" else ("vllm" if has_module("vllm") else "hf")
    if engine == "vllm":
        try:
            from .vllm_local import VLLMBackend
            return VLLMBackend(cfg, path, n_gpu)
        except Exception as e:
            if cfg.local_engine == "vllm":
                raise
            print(f"vLLM không nạp được ({type(e).__name__}: {str(e)[:300]}) -> tự chuyển sang engine hf")
            gc.collect()
            torch.cuda.empty_cache()
    from .hf_local import HFBackend
    return HFBackend(cfg, path, n_gpu)


def create_backend(cfg: Config) -> LLMBackend:
    if cfg.backend == "gemini":
        from .gemini import GeminiBackend
        return GeminiBackend(cfg)
    return _create_local(cfg)
