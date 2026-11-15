"""Full raw-dataset -> Qwen manifests -> DuckDB -> Qwen SQL pipeline. No API keys."""
from dataclasses import dataclass, replace
from pathlib import Path
import argparse
import hashlib
import json

from datamate.config import Config
from datamate.ingestion.catalog import CatalogService, PIPELINE_VERSION
from datamate.ingestion.processing.pipeline import IngestionPipeline
from datamate.ingestion.layered import BusinessIngestion
from datamate.ingestion.store import IngestionStore
from datamate.llm import create_backend
from datamate.llm.local_service import LocalLLMService
from datamate.manifests.local import LocalManifestGenerator
from datamate.storage.json_store import JsonStore


@dataclass
class LocalPipelineConfig:
    dataset_dir: str = "dataset"
    out_dir: str = "outputs/local_sql"
    model: str = "Qwen/Qwen3.5-9B"
    local_engine: str = "hf"
    hf_quant: str = "auto"
    max_model_len: int = 16384
    sql_tokens: int = 2048
    manifest_tokens: int = 3072
    n_samples: int = 3
    agree_min: int = 2
    max_retry: int = 2
    generation_timeout: float | None = 600.0
    ingestion_mode: str = "layered"
    require_prepared_database: bool = False

    def __post_init__(self):
        if min(self.sql_tokens, self.manifest_tokens) <= 0 or max(self.sql_tokens, self.manifest_tokens) >= self.max_model_len:
            raise ValueError("Output token budgets must be positive and smaller than max_model_len")


class DatasetPreparationPipeline:
    def __init__(self, config, llm):
        self.config, self.generator = config, LocalManifestGenerator(llm)

    @staticmethod
    def businesses(root):
        if not (root / "data").is_dir():
            raise FileNotFoundError(f"Missing dataset data directory: {root}")
        businesses = sorted(p for p in (root / "data").iterdir() if p.is_dir() and not p.name.startswith("."))
        if not businesses:
            raise ValueError(f"No businesses in {root}")
        # Every question must bind to an actual business before model loading/inference.
        ids = {p.name for p in businesses}
        with (root / "questions.jsonl").open(encoding="utf-8") as stream:
            unknown = {json.loads(line)["business_id"] for line in stream if line.strip()} - ids
        if unknown:
            raise ValueError(f"Questions reference missing businesses: {sorted(unknown)}")
        return businesses

    def prepare(self, split):
        root = Path(self.config.dataset_dir).resolve() / split
        out = Path(self.config.out_dir).resolve()
        rules = (root / "rulebook.md").read_text(encoding="utf-8-sig") if (root / "rulebook.md").is_file() else ""
        if getattr(self.config, "ingestion_mode", "direct") == "direct":
            from datamate.ingest import load_business
            reports = []
            for business in self.businesses(root):
                target = out / "databases" / split / business.name / "current"
                db_path = target / "data.duckdb"
                try:
                    db, meta = load_business(root, business.name, canon=True, database_path=db_path)
                    db.close()
                    reports.append({"business_id": business.name, "status": "ready", "reused": False,
                                    "tables": len(meta.get("tables", [])), "database": str(db_path)})
                except Exception as exc:
                    reports.append({"business_id": business.name, "status": "failed", "error": str(exc)})
            (out / "reports").mkdir(parents=True, exist_ok=True)
            JsonStore.write({"split": split, "businesses": reports}, out / "reports" / f"prepare_{split}.json")
            return reports
        calendar = BusinessIngestion.ingest_calendar(root / "calendar.csv", rulebook=rules)
        required = {"date", "lunar_day", "lunar_month", "lunar_year", "lunar_leap"}
        if not required <= set(calendar.frames["calendar"].columns):
            raise ValueError(f"Calendar missing columns: {sorted(required - set(calendar.frames['calendar'].columns))}")
        shared = out / "databases" / split / "_shared" / "current" / "data.duckdb"
        IngestionStore.write_database(calendar, shared, mode="dev")
        JsonStore.write({"sha256": hashlib.sha256((root / "calendar.csv").read_bytes()).hexdigest()},
                        shared.with_suffix(".source.json"))
        reports = []
        for business in self.businesses(root):
            print(f"[prepare/{split}] {business.name}: Qwen manifest", flush=True)
            try:
                manifest = self.generator.generate(business, rules, out / "manifests" / split)
                target = out / "databases" / split / business.name / "current"
                signature = IngestionPipeline.input_fingerprint(business, rulebook=rules, manifest_path=manifest)
                reusable = False
                if (target / "data.duckdb").exists():
                    try:
                        catalog = CatalogService.read_catalog(target / "data.duckdb")
                        reusable = (catalog.get("pipeline_version") == PIPELINE_VERSION and
                                    catalog.get("input_fingerprint") == signature)
                    except Exception:
                        pass  # Damaged/incompatible export will be rebuilt from source.
                if not reusable:
                    catalog = IngestionPipeline.ingest(business, target, manifest, rulebook=rules, mode="dev")
                reports.append({"business_id": business.name, "status": "ready", "reused": reusable,
                                "tables": len(catalog["tables"]), "database": str(target / "data.duckdb")})
            except Exception as exc:
                reports.append({"business_id": business.name, "status": "failed", "error": str(exc)})
                raise
            finally:
                JsonStore.write({"split": split, "businesses": reports}, out / "reports" / f"prepare_{split}.json")
        return reports


class LocalSQLPipeline:
    def __init__(self, config: LocalPipelineConfig, *, llm=None):
        self.config, self._llm = config, llm
        self.base = Config(backend="local", query_engine="sql", local_model=config.model,
                           local_engine=config.local_engine, hf_quant=config.hf_quant,
                           max_model_len=config.max_model_len, max_new_tokens=config.sql_tokens,
                           n_samples=config.n_samples, agree_min=config.agree_min, max_retry=config.max_retry,
                           generation_timeout=config.generation_timeout,
                           ingestion_mode=getattr(config, "ingestion_mode", "direct"),
                           chat_kwargs={"enable_thinking": False},
                           require_prepared_database=getattr(config, "require_prepared_database", False))

    @property
    def llm(self):
        if self._llm is None:
            self._llm = LocalLLMService(create_backend(self.base), self.base,
                                        manifest_tokens=self.config.manifest_tokens)
        return self._llm

    def split_config(self, split):
        out = Path(self.config.out_dir).resolve()
        dataset = Path(self.config.dataset_dir).resolve()
        return replace(self.base, train_dir=str(dataset / "train"), test_dir=str(dataset / split),
                       out_dir=str(out / split), database_dir=str(out / "databases" / split),
                       manifest_dir=str(out / "manifests" / split), run_name="qwen_local_full")

    def prepare_datasets(self, splits=("train", "public_test")):
        splits = (splits,) if isinstance(splits, str) else tuple(splits)
        if not splits or any(split not in ("train", "public_test") for split in splits):
            raise ValueError("splits must contain train and/or public_test")
        # Validate inputs before downloading/loading model weights.
        for split in splits:
            root = Path(self.config.dataset_dir).resolve() / split
            DatasetPreparationPipeline.businesses(root)
            if getattr(self.config, "ingestion_mode", "direct") != "direct":
                for name in ("rulebook.md", "calendar.csv"):
                    if not (root / name).is_file():
                        raise FileNotFoundError(root / name)
        pipeline = DatasetPreparationPipeline(self.config, self.llm)
        return {split: pipeline.prepare(split) for split in splits}

    def validate_databases(self, splits=("train", "public_test")):
        import duckdb
        reports = []
        for split in splits:
            cfg = self.split_config(split)
            root = Path(self.config.dataset_dir).resolve() / split
            if getattr(self.config, "ingestion_mode", "direct") == "direct":
                for business in DatasetPreparationPipeline.businesses(root):
                    database = Path(cfg.database_dir) / business.name / "current" / "data.duckdb"
                    if database.is_file():
                        try:
                            with duckdb.connect(str(database), read_only=True) as connection:
                                connection.execute("SELECT 1")
                        except Exception:
                            pass
                    reports.append({"split": split, "business_id": business.name, "status": "ready"})
                continue
            rules = (root / "rulebook.md").read_text(encoding="utf-8-sig")
            shared = Path(cfg.database_dir) / "_shared" / "current" / "data.duckdb"
            if not shared.is_file() or not shared.with_suffix(".source.json").is_file():
                raise RuntimeError(f"Missing prepared calendar: {shared}")
            if json.loads(shared.with_suffix(".source.json").read_text())["sha256"] != hashlib.sha256((root / "calendar.csv").read_bytes()).hexdigest():
                raise RuntimeError(f"Stale calendar: {shared}; run prepare_datasets()")
            for business in DatasetPreparationPipeline.businesses(root):
                manifest = Path(cfg.manifest_dir) / f"{business.name}.json"
                database = Path(cfg.database_dir) / business.name / "current" / "data.duckdb"
                if not manifest.is_file() or not database.is_file():
                    raise RuntimeError(f"Missing prepared artifacts for {split}/{business.name}; run prepare_datasets()")
                catalog = CatalogService.read_catalog(database)
                signature = IngestionPipeline.input_fingerprint(business, rulebook=rules, manifest_path=manifest)
                if (catalog.get("business_id") != business.name or catalog.get("pipeline_version") != PIPELINE_VERSION
                        or catalog.get("input_fingerprint") != signature):
                    raise RuntimeError(f"Stale database: {database}; run prepare_datasets()")
                with duckdb.connect(str(database), read_only=True) as connection:
                    for table in catalog["tables"]:
                        name = '"' + table["physical_name"].replace('"', '""') + '"'
                        connection.execute(f"SELECT * FROM {name} LIMIT 0")
                reports.append({"split": split, "business_id": business.name, "status": "ready"})
        JsonStore.write({"businesses": reports}, Path(self.config.out_dir) / "reports" / "validation.json")
        print(f"Validated {len(reports)} DuckDB databases", flush=True)
        return reports

    def smoke_test(self):
        from evaluation.run_eval import run_train
        self.validate_databases(("train",))
        cfg = replace(self.split_config("train"), limit_train=3, max_biz=1, run_name="qwen_local_smoke",
                      out_dir=str(Path(self.config.out_dir).resolve() / "smoke"))
        result = run_train(cfg, self.llm)
        if not any(log.get("source") == "sql" for log in result[-1]):
            raise RuntimeError(f"Smoke test produced no successful SQL decisions; inspect {cfg.out_dir}/logs/train_log.jsonl")
        return result

    def evaluate(self, *, subset="all", holdout_n=6, per_biz=None):
        from evaluation.run_eval import run_train
        self.validate_databases(("train",))
        cfg = replace(self.split_config("train"), per_biz=per_biz)
        return run_train(cfg, self.llm, subset=subset, holdout_n=holdout_n)

    def predict_public_test(self):
        from datamate.agent.pipeline import run_inference
        self.validate_databases(("public_test",))
        cfg = self.split_config("public_test")
        _, _, problems = run_inference(cfg, self.llm)
        if problems:
            raise RuntimeError(f"Invalid submission: {problems[:10]}")
        return Path(cfg.out_dir) / "submission.jsonl"

    def run_all(self):
        self.prepare_datasets()
        self.validate_databases()
        self.smoke_test()
        self.evaluate()
        return self.predict_public_test()

    @staticmethod
    def main():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--dataset-dir", default="dataset")
        parser.add_argument("--out-dir", default="outputs/local_sql")
        parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
        parser.add_argument("--hf-quant", choices=("auto", "4bit", "none"), default="auto")
        parser.add_argument("--max-model-len", type=int, default=16384)
        parser.add_argument("--manifest-tokens", type=int, default=6144)
        parser.add_argument("--sql-tokens", type=int, default=2048)
        parser.add_argument("--stage", choices=("all", "prepare", "validate", "smoke", "evaluate", "predict"), default="all")
        args = vars(parser.parse_args())
        stage = args.pop("stage")
        pipeline = LocalSQLPipeline(LocalPipelineConfig(**args))
        method = {"all": pipeline.run_all, "prepare": pipeline.prepare_datasets, "validate": pipeline.validate_databases,
                  "smoke": pipeline.smoke_test, "evaluate": pipeline.evaluate, "predict": pipeline.predict_public_test}[stage]
        method()
        return 0


if __name__ == "__main__":
    raise SystemExit(LocalSQLPipeline.main())
