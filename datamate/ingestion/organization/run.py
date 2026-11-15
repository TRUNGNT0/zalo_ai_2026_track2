"""Analyze existing DuckDB exports, without rereading CSV/Excel or calling an API.

python -m app.ingestion.organization.run --out-dir outputs/ingestion/train
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .analyze import OrganizationAnalyzer, POLICY_VERSION
from .build import BUILD_POLICY
from .model import MODEL_POLICY


class OrganizationCommand:
    @staticmethod
    def main(argv=None):
        parser = argparse.ArgumentParser(description=__doc__)
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--database", type=Path, help="One existing business data.duckdb")
        source.add_argument("--out-dir", type=Path, help="Analyze business subfolders; skips _shared")
        parser.add_argument("--business", action="append", help="Filter businesses under --out-dir; repeatable")
        parser.add_argument("--mode", choices=("dev", "production"), default="dev")
        args = parser.parse_args(argv)
        if args.business and not args.out_dir:
            parser.error("--business requires --out-dir")
        if args.database:
            targets = [args.database]
            summary_dir = args.database.parent
        else:
            if not args.out_dir.is_dir():
                parser.error(f"Not a directory: {args.out_dir}")
            targets = sorted(p for p in args.out_dir.glob("*/data.duckdb") if p.parent.name != "_shared")
            if args.business:
                unknown = set(args.business) - {p.parent.name for p in targets}
                if unknown:
                    parser.error(f"Unknown businesses: {sorted(unknown)}")
                targets = [p for p in targets if p.parent.name in args.business]
            summary_dir = args.out_dir
        if not targets:
            parser.error("No business databases found")
        rows = []
        for database in targets:
            print(f"[organization] {database.parent.name}", flush=True)
            try:
                result = OrganizationAnalyzer.analyze_database(database, mode=args.mode)
                entry = {"business_id": database.parent.name, "status": "complete",
                         "columns": len(result["profiles"]),
                         "keys": dict(Counter(k["status"] for k in result["keys"])),
                         "relationships": dict(Counter(r["status"] for r in result["relationships"])),
                         "transformations": dict(Counter(t["status"] for t in result["stage2"]["transformations"])),
                         "applied_by_kind": dict(Counter(t["kind"] for t in result["stage2"]["transformations"] if t["status"] == "applied")),
                         "model_transformations": dict(Counter(t["status"] for t in result["stage3"]["transformations"])),
                         "model_applied_by_kind": dict(Counter(t["kind"] for t in result["stage3"]["transformations"] if t["status"] == "applied")),
                         "model_views": sum(len(t["outputs"]) for t in result["stage3"]["transformations"] if t["status"] == "applied")}
            except Exception as exc:
                entry = {"business_id": database.parent.name, "status": "failed", "error": str(exc)}
            rows.append(entry)
            print("  " + json.dumps(entry, ensure_ascii=False), flush=True)
            if args.mode == "dev" and summary_dir.is_dir():
                (summary_dir / "organization-summary.json").write_text(json.dumps(
                    {"policy_version": POLICY_VERSION, "build_policy_version": BUILD_POLICY, "model_policy_version": MODEL_POLICY, "businesses": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.mode == "production":
            from ..artifacts import ArtifactPolicy
            ArtifactPolicy.clean_reports(summary_dir, split=True)
        return int(any(r["status"] == "failed" for r in rows))

if __name__ == "__main__":
    raise SystemExit(OrganizationCommand.main())
