"""Các lệnh tạo và kiểm tra manifest nghiệp vụ."""

from pathlib import Path
from typing import Sequence
import argparse
import json

from jsonschema import Draft202012Validator

from datamate.manifests.service import BusinessManifestService, SCHEMA_FILE
from datamate.manifests.validation import ManifestValidator
from datamate.storage.json_store import JsonStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEMINI_REQUEST_INTERVAL_SECONDS = 4.25
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset" / "train" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "train" / "manifests"
DEFAULT_SCHEMA_FILE = SCHEMA_FILE.with_name("business_manifest.v2.schema.json")

class BuildManifestsCommand:
    """Tạo manifest, gói bằng chứng và báo cáo cho dữ liệu doanh nghiệp."""

    @staticmethod
    def main(argv: Sequence[str] | None = None) -> int:
        """Đọc tham số, tạo manifest cho từng doanh nghiệp và lưu kết quả.

        Bỏ qua manifest đã tồn tại trừ khi bật ``--overwrite``; khởi tạo LLM
        khi cần xử lý doanh nghiệp đầu tiên. Trả về mã 0 khi hoàn tất.
        """
        parser = argparse.ArgumentParser(
            description="Generate one schema-validated BusinessManifest per train business."
        )
        parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
        parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_FILE)
        parser.add_argument("--business-id")
        parser.add_argument("--overwrite", action="store_true")
        parser.add_argument("--backend", choices=("local", "gemini"), default="local")
        parser.add_argument("--local-model", default="Qwen/Qwen3.5-9B")
        parser.add_argument("--hf-quant", choices=("auto", "4bit", "none"), default="auto")
        parser.add_argument("--max-model-len", type=int, default=16384)
        args = parser.parse_args(argv)

        schema = json.loads(args.schema.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        rulebook_text = (args.dataset_dir.parent / "rulebook.md").read_text(encoding="utf-8")
        business_dirs = sorted(path for path in args.dataset_dir.iterdir() if path.is_dir())
        if args.business_id:
            business_dirs = [path for path in business_dirs if path.name == args.business_id]
            if not business_dirs:
                raise ValueError(f"Business not found: {args.business_id}")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        evidence_dir = args.output_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        llm = None
        if args.backend == "local":
            from datamate.config import Config
            from datamate.llm import create_backend
            from datamate.llm.local_service import LocalLLMService
            from datamate.manifests.local import LocalManifestGenerator
            if args.schema.resolve() != DEFAULT_SCHEMA_FILE.resolve():
                raise ValueError("Local generation uses BusinessManifest v2 schema")
            cfg = Config(local_model=args.local_model, local_engine="hf", hf_quant=args.hf_quant,
                         max_model_len=args.max_model_len, chat_kwargs={"enable_thinking": False})
            generator = LocalManifestGenerator(LocalLLMService(create_backend(cfg), cfg))
            for business_dir in business_dirs:
                if args.overwrite:
                    (args.output_dir / "provenance" / f"{business_dir.name}.json").unlink(missing_ok=True)
                    import shutil
                    shutil.rmtree(args.output_dir / "batches" / business_dir.name, ignore_errors=True)
                generator.generate(business_dir, rulebook_text, args.output_dir)
            return 0
        for business_dir in business_dirs:
            output_path = args.output_dir / f"{business_dir.name}.json"
            if output_path.exists() and not args.overwrite:
                print(f"Skipping existing manifest: {output_path}")
                continue
            
            # Xây dựng đầu vào cho manifest trong đó business_dir là thư mục của doanh nghiệp, rulebook_text là nội dung của rulebook và schema là schema JSON. Đầu ra là một dict chứa thông tin về các bảng nguồn, các đoạn README, các quan sát dữ liệu và các đoạn rulebook liên quan.  
            business_input = BusinessManifestService.build_input(business_dir, rulebook_text, schema)
            if llm is None:
                from datamate.llm.gemini import GeminiLLM
                llm = GeminiLLM.from_env(
                    min_request_interval_seconds=GEMINI_REQUEST_INTERVAL_SECONDS
                )
            
            # Tạo manifest bằng cách gọi phương thức generate của BusinessManifestService, truyền vào llm, business_input và schema. Sau đó lưu gói bằng chứng vào thư mục evidence, kiểm tra manifest với dữ liệu nguồn và các evidence_id, lưu báo cáo kiểm tra vào thư mục reports và lưu manifest cuối cùng vào output_path. In ra thông báo đã tạo manifest.    
            manifest = BusinessManifestService.generate(llm, business_input, schema)
            JsonStore.write(BusinessManifestService.build_evidence(business_input), evidence_dir / f"{business_dir.name}.json")
            report = ManifestValidator.validate(manifest, business_input["source_inventory"], business_input["_evidence_ids"])
            JsonStore.write(report, args.output_dir / "reports" / f"{business_dir.name}.json")
            JsonStore.write(manifest, output_path)
            print(f"Created {output_path}")
        return 0


class AuditManifestsCommand:
    """Đối chiếu manifest đã lưu với dữ liệu nguồn hiện tại, không gọi LLM."""

    @staticmethod
    def main():
        """Kiểm tra schema và nguồn của manifest, rồi lưu báo cáo kiểm tra.

        Trả về 1 nếu có manifest phát sinh lỗi kiểm tra; ngược lại trả về 0.
        """
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--dataset-dir', type=Path, default=Path('dataset/train/data'))
        parser.add_argument('--manifest-dir', type=Path, default=Path('dataset/train/manifests'))
        parser.add_argument('--output-dir', type=Path, required=True)
        parser.add_argument('--business-id')
        args = parser.parse_args()
        schema = json.loads(SCHEMA_FILE.read_text())
        rulebook = (args.dataset_dir.parent / 'rulebook.md').read_text()
        failed = False
        for root in sorted(args.dataset_dir.iterdir()):
            if not root.is_dir() or (args.business_id and root.name != args.business_id):
                continue
            manifest = json.loads((args.manifest_dir / f'{root.name}.json').read_text())
            Draft202012Validator(schema).validate(manifest)
            data = BusinessManifestService.build_input(root, rulebook, schema)
            report = ManifestValidator.validate(manifest, data['source_inventory'], data['_evidence_ids'])
            JsonStore.write(report, args.output_dir / f'{root.name}.json')
            failed |= bool(report['errors'])
            print(root.name, report['coverage'], report['errors'], flush=True)
        return int(failed)
