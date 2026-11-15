"""Full integration with a scripted local transport; real ingestion and DuckDB execution."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from datamate.config import Config
from datamate.local_pipeline import LocalPipelineConfig, LocalSQLPipeline
from datamate.llm.base import GenerationError
from datamate.llm.local_service import LocalLLMService
from datamate.manifests.service import BusinessManifestService
from evaluation.fakes import write_split


class LocalTransport:
    def __init__(self, manifest):
        self.manifest = manifest
        self.calls = []

    def tag(self):
        return "scripted-local"

    def count_tokens(self, text):
        return len(text) // 4

    def prompt_tokens(self, system, prompt):
        return self.count_tokens(system + prompt) + 10

    def generate_with_options(self, system, prompt, **options):
        self.calls.append((system, prompt, options))
        request, _ = json.JSONDecoder().raw_decode(prompt)  # retries append validation feedback
        if "target_source" in request:
            source = request["target_source"]
            result = deepcopy(self.manifest)
            result["tables"] = [t for t in result["tables"] if
                                (t["source_file"], t["source_object"]) == (source["source_file"], source["source_object"])]
            result["tables"][0]["columns"] = [c for c in result["tables"][0]["columns"]
                                               if c["source_name"] in {v["source_name"] for v in source["columns"]}]
            result["unresolved"] = result["unresolved"][:4]
        elif "protected_terms" in request:
            result = {"question": request["question"]}
        else:
            result = {"type": "number", "sql": "SELECT sum(thanh_tien) FROM sales WHERE trang_thai != 'huy'",
                      "reason": "Loại đơn hủy"}
        return [json.dumps(result, ensure_ascii=False)]


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    # Even an accidental import of the remote client is a failure in this path.
    monkeypatch.setitem(sys.modules, "app.llm.gemini", None)
    for split in ("train", "public_test"):
        root = tmp_path / "dataset" / split
        write_split(root, n_questions=2, with_answers=split == "train")
        (root / "calendar.csv").write_text("date,lunar_day,lunar_month,lunar_year,lunar_leap,holiday\n2026-08-27,15,7,2026,False,\n")
        (root / "data/b1/products.csv").write_text("id,name\n1,Coffee\n")
    train = tmp_path / "dataset/train"
    manifest = BusinessManifestService.build_draft(train / "data/b1", (train / "rulebook.md").read_text())
    transport = LocalTransport(manifest)
    cfg = LocalPipelineConfig(dataset_dir=str(tmp_path / "dataset"), out_dir=str(tmp_path / "out"),
                              n_samples=1, agree_min=1)
    model_cfg = Config(max_new_tokens=cfg.sql_tokens, max_model_len=cfg.max_model_len)
    service = LocalLLMService(transport, model_cfg, manifest_tokens=cfg.manifest_tokens)
    return LocalSQLPipeline(cfg, llm=service), transport


def test_raw_data_to_manifests_duckdb_evaluation_and_submission(pipeline):
    runner, transport = pipeline
    submission = runner.run_all()
    rows = [json.loads(line) for line in submission.read_text().splitlines()]
    assert len(rows) == 2 and all(row["answer"] == 130000 for row in rows)
    report = json.loads((Path(runner.config.out_dir) / "train/summary.json").read_text())
    assert report["correct"] == 2 and report["query_engine"] == "sql"
    for split in ("train", "public_test"):
        assert not (Path(runner.config.dataset_dir) / split / "manifests").exists()
        database = Path(runner.config.out_dir) / "databases" / split / "b1/current/data.duckdb"
        assert database.is_file()
    manifest_calls = [c for c in transport.calls if "target_source" in json.loads(c[1])]
    assert len(manifest_calls) == 4  # two independent objects in each split
    assert any('"protected_terms"' in c[1] for c in transport.calls)
    before = len(transport.calls)
    reports = runner.prepare_datasets()
    assert all(row["reused"] for rows in reports.values() for row in rows)
    runner.evaluate()
    runner.predict_public_test()
    assert len(transport.calls) == before
    assert all('"answers"' not in prompt for _, prompt, _ in transport.calls)


@pytest.mark.parametrize("split", ["train", "public_test"])
def test_selected_split_runs_without_other_dataset(pipeline, split):
    import shutil
    runner, _ = pipeline
    other = "public_test" if split == "train" else "train"
    shutil.rmtree(Path(runner.config.dataset_dir) / other)
    assert set(runner.prepare_datasets(splits=(split,))) == {split}
    assert all(row["split"] == split for row in runner.validate_databases((split,)))
    assert not (Path(runner.config.out_dir) / "databases" / other).exists()
    if split == "train":
        runner.smoke_test()
        runner.evaluate()
    else:
        assert runner.predict_public_test().is_file()


def test_agent_requires_preparation_and_detects_source_changes(pipeline):
    runner, _ = pipeline
    with pytest.raises(RuntimeError, match="Missing prepared"):
        runner.evaluate()
    runner.prepare_datasets()
    path = Path(runner.config.dataset_dir) / "train/data/b1/sales.csv"
    path.write_text(path.read_text().replace("90000", "90001"))
    with pytest.raises(RuntimeError, match="Stale database"):
        runner.evaluate()


def test_local_json_retry_and_rewrite_guard():
    cfg = Config(max_retry=1)
    transport = LocalTransport({})
    replies = iter(['not json', '{"question":"Mua 13 sản phẩm"}'])
    transport.generate_with_options = lambda *a, **kw: [next(replies)]
    service = LocalLLMService(transport, cfg)
    with pytest.raises(GenerationError, match="changed numbers"):
        service.rewrite_question("Mua 12 sản phẩm")


def test_context_overflow_stops_before_generation():
    transport = LocalTransport({})
    service = LocalLLMService(transport, Config(max_model_len=50))
    with pytest.raises(GenerationError, match="context budget"):
        service.generate_json("system", "long prompt" * 30, {"type": "object"})
    assert not transport.calls


def test_manifest_batches_cover_wide_source_without_duplicate_tables(tmp_path):
    from datamate.manifests.local import LocalManifestGenerator
    business = tmp_path / "data" / "wide"
    business.mkdir(parents=True)
    (business / "README.md").write_text("Wide source")
    (business / "wide.csv").write_text(",".join(f"c{i}" for i in range(25)) + "\n" + ",".join("1" for _ in range(25)) + "\n")
    transport = LocalTransport(BusinessManifestService.build_draft(business))
    service = LocalLLMService(transport, Config())
    path = LocalManifestGenerator(service).generate(business, "", tmp_path / "manifests")
    result = json.loads(path.read_text())
    assert len(result["tables"]) == 1
    assert len(result["tables"][0]["columns"]) == 25
    assert len(transport.calls) == 3


@pytest.fixture
def conflicting_manifest(tmp_path):
    from datamate.manifests.migration import ManifestMigrator
    business = tmp_path / "data" / "biz_101"
    business.mkdir(parents=True)
    (business / "ban_hang.csv").write_text("Ngày tháng,Thanh tien,Quy ra ĐV lẻ\n2026-08-24,12000,6\n")
    (business / "README.md").write_text(
        "# Dữ liệu biz_101\n## Quy ước\nNgày tháng là ngày bán, Thanh tien là thành tiền sau giảm, "
        "Quy ra ĐV lẻ là số lượng đã đổi ra đơn vị cơ bản.\n")
    draft = BusinessManifestService.build_draft(business)
    table = draft["tables"][0]
    table["columns"] = [ManifestMigrator.mapped_column(name, description, ["readme.conventions"]) for name, description in (
        ("Ngày tháng", "ngày bán"), ("Thanh tien", "thành tiền sau giảm"),
        ("Quy ra ĐV lẻ", "số lượng đã đổi ra đơn vị cơ bản"))]
    table["columns"][0]["time_role"] = "transaction_date"
    table["columns"][1]["aggregation"] = "none"
    table["columns"][2]["aggregation"] = "none"
    return business, draft


def test_biz101_metadata_conflicts_are_canonicalized_and_audited(conflicting_manifest, tmp_path):
    from datamate.manifests.local import LocalManifestGenerator
    from datamate.manifests.validation import ManifestValidator
    business, draft = conflicting_manifest
    assert len([e for c in draft["tables"][0]["columns"] for e in ManifestValidator.validate_column_semantics(c)]) == 3
    transport = LocalTransport(draft)
    generator = LocalManifestGenerator(LocalLLMService(transport, Config()))
    output = tmp_path / "manifests"
    path = generator.generate(business, "", output)
    result = json.loads(path.read_text())
    columns = result["tables"][0]["columns"]
    assert columns[0]["time_role"] == "order_date"
    assert columns[1]["aggregation"] == columns[2]["aggregation"] == "sum"
    assert all(ManifestValidator.validate_column_semantics(c) == [] for c in columns)
    changes = json.loads(next((output / "reports/normalization/biz_101").glob("*.json")).read_text())["corrections"]
    assert len(changes) == 3
    assert {c["source_name"] for c in changes} == {"Ngày tháng", "Thanh tien", "Quy ra ĐV lẻ"}
    assert draft["tables"][0]["columns"][0]["time_role"] == "transaction_date"  # no mutation of model reply
    generator.generate(business, "", output)
    assert len(transport.calls) == 1


def test_metadata_normalization_preserves_unknowns_and_unit_errors(conflicting_manifest):
    from datamate.manifests.local import LocalManifestGenerator
    from datamate.manifests.validation import ManifestValidator
    _, draft = conflicting_manifest
    columns = draft["tables"][0]["columns"]
    columns[0]["time_role"] = None
    columns[1]["aggregation"] = "unknown"
    columns[1]["unit_basis"] = "base_unit"
    columns[2]["status"] = "unresolved"
    result, changes = LocalManifestGenerator.canonicalize_metadata(draft)
    assert result == draft and not changes
    assert any("unit_basis=currency" in e for e in ManifestValidator.validate_column_semantics(result["tables"][0]["columns"][1]))


def test_metadata_normalization_does_not_accept_invalid_evidence(conflicting_manifest, tmp_path):
    from datamate.manifests.local import LocalManifestGenerator
    business, draft = conflicting_manifest
    draft["tables"][0]["columns"][0]["source_refs"] = ["invented.reference"]
    transport = LocalTransport(draft)
    generator = LocalManifestGenerator(LocalLLMService(transport, Config()))
    output = tmp_path / "manifests"
    with pytest.raises(ValueError, match="Unknown column evidence"):
        generator.generate(business, "", output)
    assert not (output / "biz_101.json").exists()
    assert len(transport.calls) == 3
    assert "previous_columns" in transport.calls[-1][1]
    assert list((output / "reports/failures/biz_101").glob("*.json"))
