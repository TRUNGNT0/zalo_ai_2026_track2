import json
from copy import deepcopy
from pathlib import Path
import pytest
from jsonschema import Draft202012Validator, ValidationError

from datamate.config import Config
from datamate.llm.base import GenerationError
from datamate.llm.local_service import LocalLLMService
from datamate.llm.hf_local import ProgressCriteria
from datamate.local_pipeline import LocalPipelineConfig, LocalSQLPipeline
from datamate.manifests.local import BATCH_SYSTEM_PROMPT, LocalManifestGenerator
from datamate.manifests.service import SCHEMA_FILE


class FakeBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def tag(self):
        return "fake-backend"

    def count_tokens(self, text):
        return len(text) // 4

    def prompt_tokens(self, system, prompt):
        return len(system + prompt) // 4

    def generate_with_options(self, system, prompt, **options):
        self.calls.append((system, prompt, options))
        if not self.responses:
            raise RuntimeError("No more responses")
        resp = self.responses.pop(0)
        return [resp if isinstance(resp, str) else json.dumps(resp, ensure_ascii=False)]


def test_error_formatting_two_tables_reports_violates_max_items():
    """(a) với schema batch và output 2 bảng, chuỗi lỗi từ LocalLLMService.json chứa violates maxItems=1."""
    schema = {
        "type": "object",
        "properties": {
            "business_id": {"type": "string"},
            "tables": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "source_file": {"const": "Danh sách hàng hóa.xlsx"},
                        "header_row_number": {"const": 1}
                    },
                    "required": ["source_file", "header_row_number"]
                }
            }
        },
        "required": ["tables"]
    }
    two_tables_response = {
        "tables": [
            {"source_file": "Danh sách hàng hóa.xlsx", "header_row_number": 1},
            {"source_file": "So_chi_tiet_ban_hang.xlsx", "header_row_number": 4}
        ]
    }
    backend = FakeBackend([two_tables_response, two_tables_response, two_tables_response])
    cfg = Config(max_retry=2, max_model_len=4096)
    service = LocalLLMService(backend, cfg, manifest_tokens=1024)

    with pytest.raises(GenerationError) as exc_info:
        service.json("sys", '{"prompt": 1}', schema, task="manifest")

    err_msg = str(exc_info.value)
    assert "tables: violates maxItems=1" in err_msg
    assert "tables/1/header_row_number: 1 was expected" in err_msg
    assert "tables/1/source_file: 'Danh sách hàng hóa.xlsx' was expected" in err_msg
    assert len(exc_info.value.raw_outputs) == 3


def test_batch_schema_has_unresolved_limits():
    """(b) schema batch có unresolved.maxItems=4 và maxLength=160."""
    raw_schema = json.loads(SCHEMA_FILE.with_name("business_manifest.v2.schema.json").read_text())
    schema = deepcopy(raw_schema)
    table = schema["properties"]["tables"]
    table.update(minItems=1, maxItems=1)
    unresolved = schema["properties"]["unresolved"]
    unresolved["maxItems"] = 4
    for field in ("topic", "reason", "needed_evidence"):
        unresolved["items"]["properties"][field]["maxLength"] = 160

    # Test valid Draft202012 schema
    Draft202012Validator.check_schema(schema)
    v = Draft202012Validator(schema)

    # Valid manifest batch with 4 unresolved items
    manifest = {
        "schema_version": "2.0",
        "business_id": "b1",
        "tables": [{
            "table_role": "sales_records",
            "grain": "line_item",
            "source_file": "data.csv",
            "source_object": "data",
            "header_row_number": 1,
            "columns": [],
            "source_refs": []
        }],
        "relationships": [],
        "facts": [],
        "rule_refs": [],
        "unresolved": [
            {"topic": f"topic_{i}", "reason": f"reason_{i}", "needed_evidence": f"evidence_{i}"}
            for i in range(4)
        ]
    }
    v.validate(manifest)

    # Reject 5 unresolved items
    manifest_5 = deepcopy(manifest)
    manifest_5["unresolved"].append({"topic": "t5", "reason": "r5", "needed_evidence": "e5"})
    errors = list(v.iter_errors(manifest_5))
    assert any(e.validator == "maxItems" and e.path[0] == "unresolved" for e in errors)

    # Reject item with length > 160
    manifest_long = deepcopy(manifest)
    manifest_long["unresolved"][0]["reason"] = "x" * 200
    errors = list(v.iter_errors(manifest_long))
    assert any(e.validator == "maxLength" for e in errors)


def test_batch_system_prompt_overrides_single_entry_rule():
    """(c) BATCH_SYSTEM_PROMPT chứa đoạn ghi đè và được dùng ở cả fit_evidence, chữ ký cache và lời gọi json."""
    assert "CHẾ ĐỘ BATCH" in BATCH_SYSTEM_PROMPT
    assert "ĐÚNG MỘT entry" in BATCH_SYSTEM_PROMPT
    assert "source_index" in BATCH_SYSTEM_PROMPT

    backend = FakeBackend([])
    cfg = Config(max_retry=1, max_model_len=4096)
    service = LocalLLMService(backend, cfg, manifest_tokens=1024)
    generator = LocalManifestGenerator(service)

    # fit_evidence uses BATCH_SYSTEM_PROMPT
    prompt = {
        "target_source": {"source_file": "a.csv", "source_object": "a", "columns": []},
        "source_index": [],
        "readme_sections": [],
        "rulebook_sections": [],
        "evidence_ids": [],
        "instruction": ""
    }
    schema = deepcopy(generator.schema)
    fitted = generator.fit_evidence(prompt, schema)
    assert fitted is not None


def test_local_pipeline_config_and_engine_selection():
    """Kiểm tra LocalPipelineConfig manifest_tokens=3072 mặc định và hỗ trợ local_engine."""
    cfg = LocalPipelineConfig()
    assert cfg.manifest_tokens == 3072
    assert cfg.local_engine == "hf"
    assert cfg.generation_timeout == 600.0

    pipeline = LocalSQLPipeline(cfg)
    assert pipeline.base.local_engine == "hf"
    assert pipeline.base.generation_timeout == 600.0

    cfg_vllm = LocalPipelineConfig(local_engine="vllm", manifest_tokens=3072)
    pipeline_vllm = LocalSQLPipeline(cfg_vllm)
    assert pipeline_vllm.base.local_engine == "vllm"


def test_progress_criteria_logging_and_timeout(capsys):
    """Kiểm tra ProgressCriteria log prefill, decode interval và timeout."""
    class DummyTensor:
        def __init__(self, shape):
            self.shape = shape

    criteria = ProgressCriteria(prompt_len=100, max_new_tokens=256, max_time=1.0, log_interval=128)

    # Step 1: prefill
    criteria(DummyTensor((1, 101)))
    out = capsys.readouterr().out
    assert "[hf] prefill 100 tokens" in out

    # Step 128: decode log
    criteria(DummyTensor((1, 229)))
    out = capsys.readouterr().out
    assert "[hf] decode 129/256 tokens" in out

    # Timeout
    criteria_short = ProgressCriteria(prompt_len=10, max_new_tokens=50, max_time=0.001)
    import time
    time.sleep(0.01)
    with pytest.raises(TimeoutError, match="exceeded max_time"):
        criteria_short(DummyTensor((1, 11)))
