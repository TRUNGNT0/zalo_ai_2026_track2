"""Rebuild typed data from stored raw rows after changing policies, without rereading Excel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .layered import IngestedBusiness, IngestedTable, BusinessIngestion
from .raw import RawTable, RawData
from .processing.normalization import ValueNormalizer
from .store import IngestionStore

READER_ISSUES = {"header_detected", "header_renamed", "ragged_row", "encoding_fallback",
                 "delimiter_fallback", "formula_cache_missing", "excel_error", "empty_sheet", "sqlite_no_primary_key"}


class RetypeService:
    @staticmethod
    def retype_business(database, overrides=None):
        import duckdb
        overrides = overrides or {}
        if overrides.get("files"):
            raise ValueError("File/header/encoding overrides require a full ingestion run")
        checked = set()
        with duckdb.connect(str(database), read_only=True) as con:
            business_id, readme, rulebook, semantic_json = con.execute("SELECT * FROM meta.business").fetchone()
            previous = json.loads(semantic_json)
            result = IngestedBusiness(business_id, readme, rulebook)
            if con.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema='meta' AND table_name='catalog'").fetchone()[0]:
                result.manifest = json.loads(con.execute("SELECT manifest_json FROM meta.catalog").fetchone()[0])
            # A policy change invalidates any source-cache fingerprint.
            result.input_fingerprint = None
            result.issues = [json.loads(text) for (text,) in con.execute(
                "SELECT details_json FROM meta.issues WHERE logical_name IS NULL").fetchall()]
            inherited = {"links": previous.get("confirmed_links", []), "views": previous.get("views", {})}
            effective = inherited | overrides
            table_rows = con.execute("SELECT * FROM meta.tables").fetchall()
            unknown_tables = set(overrides.get("tables", {})) - {row[0] for row in table_rows}
            if unknown_tables:
                raise ValueError(f"Unknown override tables: {sorted(unknown_tables)}")
            for name, db_name, _, source_json, columns_json, cells_json in table_rows:
                source = json.loads(source_json)
                if source["file"] not in checked:
                    if RawData.source_info(Path(source["file"]))["sha256"] != source["sha256"]:
                        raise ValueError(f"Source changed: {source['file']}; run ingestion again")
                    checked.add(source["file"])
                rows, positions = [], []
                # row_id has an unpadded numeric suffix; preserve source order numerically.
                query = f"SELECT values_json, position_json FROM raw.{RawData.quote_identifier(db_name)} ORDER BY CAST(regexp_extract(row_id, ':(\\d+)$', 1) AS BIGINT)"
                for values, position in con.execute(query).fetchall():
                    rows.append(json.loads(values))
                    positions.append(json.loads(position))
                placeholders = ",".join("?" for _ in READER_ISSUES)
                raw_issues = [json.loads(text) for (text,) in con.execute(
                    f"SELECT details_json FROM meta.issues WHERE logical_name=? AND code IN ({placeholders})",
                    [name, *sorted(READER_ISSUES)]).fetchall()]
                old_columns = [c for c in json.loads(columns_json) if "derived_from" not in c]
                raw = RawTable(name, source, [c["name"] for c in old_columns], rows, positions, raw_issues, json.loads(cells_json))
                column_options = {c["name"]: c["override"] for c in old_columns if c.get("override")}
                table_options = overrides.get("tables", {}).get(name, {})
                for column, spec in table_options.get("columns", {}).items():
                    column_options[column] = column_options.get(column, {}) | spec
                frame, columns, flags, issues = ValueNormalizer.convert_table(
                    raw, readme, table_options | {"columns": column_options}
                )
                result.tables[name] = IngestedTable(raw, frame, columns, flags, issues)
        return BusinessIngestion.finish_semantic(result, effective)

    @staticmethod
    def retype_one(target, overrides, mode="dev"):
        target = Path(target)
        print(f"[retype] {target.name}", flush=True)
        business = RetypeService.retype_business(target / "data.duckdb", overrides)
        IngestionStore.write_database(business, target / "data.duckdb", mode=mode)
        if mode == "dev":
            return IngestionStore.write_report(business, target)["issue_counts"]
        from collections import Counter
        return dict(Counter(i["code"] for i in business.issues + [p for t in business.tables.values() for p in t.issues]))

    @staticmethod
    def main(argv=None):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--out-dir", required=True, type=Path)
        parser.add_argument("--overrides", type=Path)
        parser.add_argument("--workers", type=int, default=1, help="Independent business processes; default 1")
        parser.add_argument("--mode", choices=("dev", "production"), default="dev")
        a = parser.parse_args(argv)
        if a.workers < 1:
            parser.error("workers must be >= 1")
        overrides = json.loads(a.overrides.read_text()) if a.overrides else {}
        summary_path = a.out_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {
            "businesses": [{"business_id": p.parent.name, "status": "complete"}
                           for p in sorted(a.out_dir.glob("*/data.duckdb")) if p.parent.name != "_shared"]}
        if not summary["businesses"]:
            parser.error("No business databases found")
        entries = [entry for entry in summary["businesses"] if entry["status"] != "failed"]
        targets = [a.out_dir / entry["business_id"] for entry in entries]
        policies = [overrides.get(entry["business_id"]) for entry in entries]
        if a.workers == 1:
            for entry, target, policy in zip(entries, targets, policies):
                entry["issue_counts"] = RetypeService.retype_one(target, policy, a.mode)
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=a.workers) as executor:
                for entry, counts in zip(entries, executor.map(RetypeService.retype_one, targets, policies, [a.mode] * len(targets))):
                    entry["issue_counts"] = counts
        import duckdb
        for entry, target in zip(entries, targets):
            with duckdb.connect(str(target / "data.duckdb"), read_only=True) as con:
                names = con.execute("SELECT db_name FROM meta.tables").fetchall()
                entry["tables"] = len(names)
                entry["rows"] = sum(con.execute(f"SELECT count(*) FROM raw.{RawData.quote_identifier(name)}").fetchone()[0] for (name,) in names)
                entry["status"] = "partial" if con.execute("SELECT count(*) FROM meta.issues WHERE logical_name IS NULL").fetchone()[0] else "complete"
        from collections import Counter
        counts = Counter()
        for entry in summary["businesses"]:
            counts.update(entry.get("issue_counts", {}))
        summary["issue_counts"] = dict(counts)
        if a.mode == "dev":
            (a.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
            from .review import build_review
            build_review(a.out_dir)
        else:
            from .artifacts import ArtifactPolicy
            ArtifactPolicy.clean_reports(a.out_dir, split=True)

if __name__ == "__main__":
    RetypeService.main()
