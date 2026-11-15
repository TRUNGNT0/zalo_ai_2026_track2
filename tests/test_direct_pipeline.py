import json
from pathlib import Path
import pytest

from datamate.local_pipeline import LocalPipelineConfig, LocalSQLPipeline
from evaluation.fakes import write_split


class DirectFakeTransport:
    def tag(self):
        return "fake-direct"

    def count_tokens(self, text):
        return len(text) // 4

    def prompt_tokens(self, system, prompt):
        return self.count_tokens(system + prompt) + 10

    def generate_with_options(self, system, prompt, **options):
        result = {"type": "number", "sql": "SELECT sum(thanh_tien) FROM sales WHERE vn_norm(trang_thai) != 'huy'",
                  "reason": "Loại đơn hủy"}
        return [json.dumps(result, ensure_ascii=False)]


def test_direct_pipeline_end_to_end_without_manifests(tmp_path):
    train_dir = tmp_path / "dataset" / "train"
    write_split(train_dir, n_questions=2, with_answers=True)
    sales_csv = train_dir / "data" / "b1" / "sales.csv"
    sales_csv.write_text("ngay,thanh_tien,trang_thai\n2026-08-27,50000,hoan thanh\n2026-08-27,80000,hoan thanh\n", encoding="utf-8")

    cfg = LocalPipelineConfig(
        dataset_dir=str(tmp_path / "dataset"),
        out_dir=str(tmp_path / "out"),
        n_samples=1,
        agree_min=1,
        ingestion_mode="direct",
        require_prepared_database=False,
    )
    from datamate.llm.local_service import LocalLLMService
    from datamate.config import Config
    transport = DirectFakeTransport()
    service = LocalLLMService(transport, Config(max_new_tokens=100, max_model_len=4096), manifest_tokens=100)
    pipeline = LocalSQLPipeline(cfg, llm=service)

    # 1. Prepare datasets in direct mode (uses load_business, does not call LLM for manifests)
    reports = pipeline.prepare_datasets(splits=("train",))
    assert reports["train"][0]["status"] == "ready"
    assert (Path(tmp_path / "out" / "databases" / "train" / "b1" / "current" / "data.duckdb")).is_file()

    # 2. Validate databases
    val = pipeline.validate_databases(splits=("train",))
    assert val[0]["status"] == "ready"

    # 3. Smoke test evaluation
    result = pipeline.smoke_test()
    subs = result[3]
    assert len(subs) == 2
    assert subs[0]["type"] == "number"
    assert subs[0]["answer"] == 130000
