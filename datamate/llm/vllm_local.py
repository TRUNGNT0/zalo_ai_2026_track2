"""Backend local bằng vLLM (nhanh, có prefix cache; cài dễ xung đột torch trên Kaggle)."""
from __future__ import annotations

import os

from datamate.config import Config
from .base import GenerationError, LLMBackend


class VLLMBackend(LLMBackend):
    supports_batch = True
    engine = "vllm"

    def __init__(self, cfg: Config, path: str, n_gpu: int):
        from transformers import AutoTokenizer
        self.cfg = cfg
        os.environ.update({k: str(v) for k, v in cfg.vllm_env.items()})     # phải đặt TRƯỚC khi import vllm
        tp = max(1, min(cfg.tensor_parallel, n_gpu))
        if tp > 1 and os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0":
            os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING")                # TP>1 bắt buộc nhiều tiến trình
        try:
            from vllm import LLM
        except Exception as e:
            raise RuntimeError(
                f"Không import được vllm ({type(e).__name__}: {str(e)[:200]}).\n"
                "Thường do `pip install vllm` đổi bản torch lệch với torchaudio/torchvision có sẵn của Kaggle.\n"
                "Cách chắc nhất: Stop session để reset môi trường, KHÔNG cài vllm, dùng local_engine='auto' hoặc 'hf'."
            ) from e
        kw = dict(model=path, dtype="half", max_model_len=cfg.max_model_len, gpu_memory_utilization=cfg.gpu_memory_utilization,
                  tensor_parallel_size=tp, enable_prefix_caching=True)
        if cfg.local_lora:
            kw.update(enable_lora=True, max_lora_rank=cfg.lora_rank)
        kw.update(cfg.vllm_extra)
        self.llm = LLM(**kw)
        self.tok = AutoTokenizer.from_pretrained(path)

    def tag(self) -> str:
        c = self.cfg
        return f"local|{c.local_model}|{c.local_lora}|vllm|T{c.temperature}|L{c.max_model_len}"

    def count_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False).input_ids)

    def generate(self, system: str, prompt: str, n: int = 1) -> list[str]:
        from vllm import SamplingParams
        cfg = self.cfg
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        sp = SamplingParams(temperature=cfg.temperature, top_p=0.95, max_tokens=cfg.max_new_tokens, n=n)
        kw = {}
        if cfg.local_lora:
            from vllm.lora.request import LoRARequest
            kw["lora_request"] = LoRARequest("sft", 1, cfg.local_lora)
        if cfg.chat_kwargs:
            kw["chat_template_kwargs"] = cfg.chat_kwargs
        try:
            out = self.llm.chat(msgs, sp, use_tqdm=False, **kw)
        except Exception as e:
            raise GenerationError(repr(e)[:300]) from e
        return [o.text for o in out[0].outputs]
