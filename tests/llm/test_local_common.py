"""Nhận diện model và quy tắc nạp cho Qwen3.5 (không cần GPU, không tải model)."""
import json

import pytest

from datamate.config import Config
from datamate.llm.hardware import MODEL_9B, apply_recommendation, recommend
from datamate.llm.local_common import is_hybrid_model, is_qwen3_family, sampling_kwargs, use_4bit, weights_gb


@pytest.mark.parametrize("name, expected", [("Qwen/Qwen3.5-9B", True), ("Qwen/Qwen3-14B", True),
                                            ("Qwen/Qwen2.5-Coder-14B-Instruct", False), ("/content/merged", False)])
def test_qwen3_family_detection(name, expected):
    assert is_qwen3_family(name) is expected


@pytest.mark.parametrize("mode, vram, weights, expected", [
    ("auto", 15.0, 19.3, True),       # T4: 9B bf16 không vừa
    ("auto", 24.1, 19.3, True),       # L4: 19,3GB + ~30% KV/activation vượt 24GB
    ("auto", 42.5, 19.3, False),      # A100 40GB: nạp bf16 đầy đủ
    ("auto", 42.5, 29.6, False),      # 14B trên A100 40GB vẫn vừa
    ("auto", 24.1, 29.6, True),
    ("auto", 40.0, None, False),      # không biết kích thước: ngưỡng 38GB cũ
    ("auto", 24.0, None, True),
    ("4bit", 80.0, 19.3, True),
    ("none", 8.0, 19.3, False),
])
def test_use_4bit_by_mode_vram_and_weight_size(mode, vram, weights, expected):
    assert use_4bit(mode, vram, weights) is expected


def test_weights_gb_reads_safetensors_index(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 19_306_208_736}}))
    assert weights_gb(str(tmp_path)) == pytest.approx(19.306, abs=0.01)


def test_weights_gb_falls_back_to_file_sizes(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"x" * 1000)
    (tmp_path / "b.safetensors").write_bytes(b"x" * 500)
    assert weights_gb(str(tmp_path)) == pytest.approx(1500 / 1e9)


def test_weights_gb_unknown_when_no_files(tmp_path):
    assert weights_gb(str(tmp_path)) is None
    assert weights_gb(str(tmp_path / "missing")) is None


def test_hybrid_model_detected_from_text_config(tmp_path):
    cfg = {"model_type": "qwen3_5", "text_config": {"layer_types": ["linear_attention"] * 3 + ["full_attention"]}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    assert is_hybrid_model(str(tmp_path)) is True


def test_plain_transformer_is_not_hybrid(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2", "num_hidden_layers": 28}))
    assert is_hybrid_model(str(tmp_path)) is False
    assert is_hybrid_model(str(tmp_path / "missing")) is False


def test_sampling_for_qwen3_adds_top_k_and_lower_top_p():
    assert sampling_kwargs("Qwen/Qwen3.5-9B", 0.7) == {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
    assert sampling_kwargs("Qwen/Qwen2.5-Coder-7B-Instruct", 0.7) == {"temperature": 0.7, "top_p": 0.95}


@pytest.mark.parametrize("vram", [15.0, 24.0, 42.5])
def test_recommendation_is_qwen35_9b_on_hf_engine(vram):
    recommendation = recommend(vram)
    assert (recommendation["local_model"], recommendation["local_engine"]) == (MODEL_9B, "hf")
    cfg = apply_recommendation(Config(backend="local"), {"vram_gb": vram})
    assert cfg.local_model == "Qwen/Qwen3.5-9B" and cfg.local_engine == "hf"


def test_invalid_hf_quant_is_rejected():
    with pytest.raises(ValueError):
        Config(backend="gemini", hf_quant="8bit")
