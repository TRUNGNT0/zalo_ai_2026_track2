"""python -m app.ingestion.run --split-dir train --out-dir outputs/ingestion/train"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .datasets.loader import DatasetLoader
from .layered import BusinessIngestion
from .store import IngestionStore
from .processing.pipeline import IngestionPipeline


class IngestionCommand:
    @staticmethod
    def main(argv=None):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--split-dir", required=True)
        parser.add_argument("--out-dir", required=True)
        parser.add_argument("--business", action="append", help="Repeat to select businesses; default all")
        parser.add_argument("--overrides", type=Path, help="JSON keyed by business id")
        parser.add_argument("--allow-partial", action="store_true", help="Continue after file errors, record them; exit remains nonzero")
        parser.add_argument("--mode", choices=("dev", "production"), default="dev")
        a = parser.parse_args(argv)
        root, output = Path(a.split_dir), Path(a.out_dir)
        if not (root / "data").is_dir():
            root = Path(DatasetLoader.find_root_dir(a.split_dir))
        output.mkdir(parents=True, exist_ok=True)
        rules = (root / "rulebook.md").read_text(encoding="utf-8-sig") if (root / "rulebook.md").exists() else ""
        overrides = json.loads(a.overrides.read_text(encoding="utf-8")) if a.overrides else {}
        if (root / "calendar.csv").exists():
            shared = BusinessIngestion.ingest_calendar(root / "calendar.csv", rulebook=rules)
            IngestionStore.write_database(shared, output / "_shared" / "data.duckdb", mode=a.mode)
            if a.mode == "dev":
                IngestionStore.write_report(shared, output / "_shared")
        businesses = sorted(p for p in (root / "data").iterdir() if p.is_dir())
        if a.business:
            unknown = set(a.business) - {p.name for p in businesses}
            if unknown:
                parser.error(f"Unknown businesses: {sorted(unknown)}")
            businesses = [p for p in businesses if p.name in a.business]
        summary, counts, failed = [], Counter(), False
        for directory in businesses:
            print(f"[ingestion] {directory.name}", flush=True)
            try:
                manifest_path = root / "manifests" / f"{directory.name}.json"
                manifest_path, manifest = IngestionPipeline.ensure_manifest(
                    directory, output / directory.name,
                    manifest_path if manifest_path.exists() else None,
                    rulebook=rules,
                )
                business = BusinessIngestion.ingest_business(directory, rulebook=rules, overrides=overrides.get(directory.name), strict=not a.allow_partial,
                                           manifest=manifest)
                business.input_fingerprint = IngestionPipeline.input_fingerprint(directory, rulebook=rules, manifest_path=manifest_path,
                                                              overrides=overrides.get(directory.name))
                target = output / directory.name
                IngestionStore.write_database(business, target / "data.duckdb", mode=a.mode)
                report = IngestionStore.write_report(business, target) if a.mode == "dev" else {"issue_counts": dict(Counter(p["code"] for p in business.issues + [i for t in business.tables.values() for i in t.issues]))}
                counts.update(report["issue_counts"])
                failed |= bool(business.issues)
                summary.append({"business_id": directory.name, "tables": len(business.tables),
                                "rows": sum(len(t.typed) for t in business.tables.values()),
                                "issue_counts": report["issue_counts"], "status": "partial" if business.issues else "complete"})
                print(f"  {summary[-1]['tables']} tables, {summary[-1]['rows']} rows", flush=True)
            except Exception as exc:
                failed = True
                summary.append({"business_id": directory.name, "status": "failed", "error": str(exc)})
                print(f"  FAILED: {exc}", flush=True)
            finally:
                if a.mode == "dev":
                    (output / "summary.json").write_text(json.dumps({"businesses": summary, "issue_counts": dict(counts)}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Output ({a.mode}): {output.resolve()}")
        from .review import build_review
        if a.mode == "dev":
            build_review(output)
        else:
            from .artifacts import ArtifactPolicy
            ArtifactPolicy.clean_reports(output, split=True)
        return int(failed)

if __name__ == "__main__":
    raise SystemExit(IngestionCommand.main())
