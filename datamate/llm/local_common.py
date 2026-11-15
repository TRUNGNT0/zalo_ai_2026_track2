"""Tiện ích chung cho backend local (tải model, đọc cấu hình lượng tử hóa)."""
from __future__ import annotations

import importlib.util
import json
import os


def has_module(*mods: str) -> bool:
    for m in mods:
        try:
            if importlib.util.find_spec(m):
                return True
        except (ImportError, ValueError):
            pass
    return False


def resolve_model_path(name: str) -> str:
    """Đường dẫn thư mục thì dùng luôn; tên repo HF thì tải về /tmp/models (cần Internet)."""
    if os.path.isdir(name):
        return name
    from huggingface_hub import snapshot_download
    dst = "./models/" + name.replace("/", "__")
    print(f"Tải {name} về {dst} (cần bật Internet trong Settings của notebook) ...")
    return snapshot_download(name, local_dir=dst)


def read_quant_config(path: str) -> dict:
    try:
        return json.load(open(os.path.join(path, "config.json"))).get("quantization_config", {}) or {}
    except Exception:
        return {}


def is_prequantized(path: str) -> bool:
    return bool(read_quant_config(path))


def quant_method(path: str) -> str:
    return str(read_quant_config(path).get("quant_method", "")).lower()


def is_qwen3_family(name: str) -> bool:
    """Qwen3 / Qwen3.5 mặc định chạy chế độ suy nghĩ (<think>), phải tắt thì 1024 token mới ra được code."""
    return "qwen3" in name.lower()


def read_text_config(path: str) -> dict:
    """config.json của model; model đa phương thức (Qwen3.5) để cấu hình phần chữ trong `text_config`."""
    try:
        cfg = json.load(open(os.path.join(path, "config.json"), encoding="utf-8"))
    except Exception:
        return {}
    return cfg.get("text_config") or cfg


def is_hybrid_model(path: str) -> bool:
    """Model lai linear-attention + full-attention (Qwen3.5, Qwen3-Next): cache của chúng không phải DynamicCache thường,
    nên không nạp prompt theo từng đoạn bằng prefill_chunked được."""
    cfg = read_text_config(path)
    return "linear_attention" in (cfg.get("layer_types") or []) or "linear_num_value_heads" in cfg


def weights_gb(path: str) -> float | None:
    """Dung lượng trọng số (GB) đọc từ chỉ mục safetensors hoặc cộng kích thước các tệp; None nếu không biết."""
    try:
        total = json.load(open(os.path.join(path, "model.safetensors.index.json"))).get("metadata", {}).get("total_size")
        if total:
            return total / 1e9
    except Exception:
        pass
    try:
        sizes = [os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".safetensors")]
    except OSError:
        return None
    return sum(sizes) / 1e9 if sizes else None


def use_4bit(mode: str, vram_gb: float, weights: float | None = None) -> bool:
    """Có nén 4-bit (bitsandbytes) khi nạp không. "auto": chỉ nén khi trọng số bf16 + ~30% chỗ cho KV/activation không vừa VRAM.
    Nén 4-bit tiết kiệm VRAM nhưng sinh token chậm hơn nhiều, nên GPU đủ chỗ (A100) thì nạp đầy đủ."""
    if mode in ("4bit", "none"):
        return mode == "4bit"
    return vram_gb < (weights * 1.3 if weights else 38)


def sampling_kwargs(model_name: str, temperature: float) -> dict:
    """Tham số lấy mẫu: Qwen3/3.5 ở chế độ không suy nghĩ khuyến nghị top_p 0,8 và top_k 20 (model card); còn lại giữ top_p 0,95."""
    if is_qwen3_family(model_name):
        return {"temperature": temperature, "top_p": 0.8, "top_k": 20}
    return {"temperature": temperature, "top_p": 0.95}
