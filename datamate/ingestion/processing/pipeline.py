"""Unified ingestion facade: readers -> Raw/Typed/Semantic -> SQL catalog."""
from pathlib import Path
import argparse
import hashlib
import json

from datamate.ingestion.catalog import PIPELINE_VERSION, CatalogService
from datamate.ingestion.layered import BusinessIngestion
from datamate.ingestion.store import IngestionStore
from datamate.storage.json_store import JsonStore


class IngestionPipeline:
    @staticmethod
    def ensure_manifest(business_dir, output_dir, manifest_path=None, *, rulebook=""):
        business_dir, output_dir = Path(business_dir), Path(output_dir)
        if manifest_path is None:
            manifest_path = (
                business_dir.parent.parent / "manifests" / f"{business_dir.name}.json"
                if business_dir.parent.name == "data"
                else output_dir / "manifest.json"
            )
            if not manifest_path.exists():
                from datamate.manifests.service import BusinessManifestService

                manifest = BusinessManifestService.build_draft(business_dir, rulebook)
                JsonStore.write(manifest, manifest_path)
                print(f"Created local BusinessManifest draft: {manifest_path}", flush=True)
                return manifest_path, manifest
        return manifest_path, CatalogService.load_manifest(manifest_path, business_dir.name)

    @staticmethod
    def ingest(business_dir, output_dir, manifest_path=None, *, rulebook="", overrides=None, mode="production", strict=True):
        business_dir, output_dir = Path(business_dir), Path(output_dir)
        manifest_path, manifest = IngestionPipeline.ensure_manifest(
            business_dir, output_dir, manifest_path, rulebook=rulebook
        )
        fingerprint = IngestionPipeline.input_fingerprint(business_dir, rulebook=rulebook, manifest_path=manifest_path, overrides=overrides)
        business = BusinessIngestion.ingest_business(business_dir, rulebook=rulebook, overrides=overrides, strict=strict, manifest=manifest)
        business.input_fingerprint = fingerprint
        database = IngestionStore.write_database(business, output_dir / "data.duckdb", mode=mode)
        if mode == "dev":
            IngestionStore.write_report(business, output_dir)
        return CatalogService.read_catalog(database)

    @staticmethod
    def main(argv=None):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--business-dir", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--manifest", type=Path)
        parser.add_argument("--rulebook", type=Path)
        parser.add_argument("--overrides", type=Path, help="JSON overrides for this business")
        parser.add_argument("--mode", choices=("dev", "production"), default="production")
        args = parser.parse_args(argv)
        catalog = IngestionPipeline.ingest(args.business_dir, args.output_dir, args.manifest,
            rulebook=args.rulebook.read_text(encoding="utf-8-sig") if args.rulebook else "",
            overrides=json.loads(args.overrides.read_text()) if args.overrides else None, mode=args.mode)
        print(f"Ingested {catalog['business_id']}: {len(catalog['tables'])} tables into {args.output_dir / catalog['database_file']}")
        return 0

    @staticmethod
    def input_fingerprint(business_dir, *, rulebook="", manifest_path=None, overrides=None):
        root = Path(business_dir).resolve()
        digest = hashlib.sha256(json.dumps({"version": PIPELINE_VERSION, "root": str(root),
            "rulebook": rulebook, "overrides": overrides or {}}, sort_keys=True, ensure_ascii=False).encode())
        for path in sorted(root.rglob("*")):
            if not path.is_file() or any(p.startswith(".") for p in path.relative_to(root).parts):
                continue
            # Include unknown files too: adding an unsupported source must invalidate cache.
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            with path.open("rb") as stream:
                digest.update(hashlib.file_digest(stream, "sha256").digest())
        if manifest_path is not None:
            digest.update(Path(manifest_path).read_bytes())
        return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(IngestionPipeline.main())
